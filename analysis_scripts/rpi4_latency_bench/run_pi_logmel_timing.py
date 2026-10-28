# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Measure the training repo's numpy ``log_mel`` on the Raspberry Pi 4.

Exports ``librosa.filters.mel(...).T`` (exactly what the training repo's
``mel_filter`` returns), copies it, the course repo's ``training/features.py``
and ``pi_logmel_timing.py`` to ``~/<remote_dir>/logmel/`` on the Pi, checks
that every core uses the ``performance`` governor (aborts otherwise), and runs
the timing with 1 and 4 BLAS threads. It records ``vcgencmd get_throttled``
and the temperature before and after each run, and checks the Pi's log-mel of
a fixed chirp against the host's (numpy versions differ). Writes
``results/pi_logmel_timing_<timestamp>.json`` and prints a summary. Nothing
is installed on the Pi.
"""

from __future__ import annotations

import argparse
import base64
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (REPO_ROOT / "src", SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from crest.microcontrollers.raspberry_pi4 import RaspberryPi4Device, run_command  # noqa: E402
from run_rpi4_latency_bench import DEFAULT_RESULTS_DIR, check_governors  # noqa: E402

DEFAULT_TRAINING_REPO = REPO_ROOT.parent / "embedded-systme-project"


def parse_kv(stdout: str) -> dict:
    """Collect key=value tokens from the remote output."""
    values = {}
    for line in stdout.splitlines():
        for token in line.split():
            key, sep, value = token.partition("=")
            if sep:
                values[key] = value
    return values


def throttle_state(device: RaspberryPi4Device) -> dict:
    """``vcgencmd get_throttled`` and the SoC temperature."""
    probe = device._ssh("vcgencmd get_throttled; vcgencmd measure_temp", device.options.ssh_connect_timeout_s + 10)
    return parse_kv(probe.stdout)


def main() -> int:
    """Copy, run, compare and summarize."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default=None)
    parser.add_argument("--python", default="crest-litert/bin/python")
    parser.add_argument("--remote-dir", default="crest_rpi4")
    parser.add_argument("--training-repo", type=Path, default=DEFAULT_TRAINING_REPO)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()

    import librosa

    sys.path.insert(0, str(args.training_repo))
    from training.features import log_mel, mel_filter

    build = SCRIPT_DIR / "results" / "logmel_build"
    build.mkdir(parents=True, exist_ok=True)
    mel = mel_filter(16000, 512, 64)
    np.save(build / "mel_257x64.npy", mel.astype(np.float32))

    device = RaspberryPi4Device(device_options={"host": args.host, "user": args.user, "python": args.python,
                                                "remote_dir": args.remote_dir})
    session = check_governors(device, "performance")
    print(f"session check: {session}", flush=True)
    if not session["ok"]:
        reason = session.get("error") or f"governors {session.get('governors')}"
        print(f"ABORT: Pi not reachable or not every core on the 'performance' governor ({reason}); nothing was run.")
        return 3
    remote = f"{args.remote_dir}/logmel"
    mkdir = device._ssh(f"mkdir -p {shlex.quote(remote)}", 30)
    if mkdir.returncode != 0:
        print(mkdir.stderr)
        return 1
    copy = run_command(["scp", "-o", "BatchMode=yes", "-q", str(build / "mel_257x64.npy"),
                        str(args.training_repo / "training" / "features.py"), str(SCRIPT_DIR / "pi_logmel_timing.py"),
                        f"{device.options.destination}:{remote}/"], 120)
    if copy.returncode != 0:
        print(copy.stderr)
        return 1

    # Host reference for the check: same chirp as pi_logmel_timing.chirp(1.0).
    audio = {"sample_rate": 16000, "duration_sec": 1, "n_mels": 64, "frame_length_ms": 25, "hop_length_ms": 10,
             "fft_length": 512, "log_offset": 1.0e-6, "spectrum": "power"}
    n = 16000
    t = np.arange(n) / 16000
    x = 10 ** (-12 / 20) * np.sin(2 * np.pi * (50 * t + (7950 - 50) / 2 * t ** 2))
    chirp = np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16).astype(np.float32) / np.float32(32768.0)
    host_ref = log_mel(chirp, audio)[..., 0]

    report = {"session": session, "librosa": librosa.__version__, "host_numpy": np.__version__, "runs": {}}
    for threads in args.threads:
        before = throttle_state(device)
        cmd = (f"{shlex.quote(args.python)} {remote}/pi_logmel_timing.py --features {remote}/features.py "
               f"--mel {remote}/mel_257x64.npy --threads {threads}")
        result = device._ssh(cmd, 600)
        after = throttle_state(device)
        values = parse_kv(result.stdout)
        if result.returncode != 0 or "check_b64" not in values:
            print(result.stdout[-2000:], result.stderr[-2000:])
            return 1
        pi = np.frombuffer(base64.b64decode(values.pop("check_b64")), "<f4").reshape(host_ref.shape)
        err = np.abs(pi.astype(np.float64) - host_ref.astype(np.float64))
        run = {key: (float(value) if key.endswith(("median", "min", "p95")) else value) for key, value in values.items()}
        run.update(throttled_before=before.get("throttled"), throttled_after=after.get("throttled"),
                   temp_before=before.get("temp"), temp_after=after.get("temp"),
                   check_max_abs_err_vs_host=float(err.max()), check_mean_abs_err_vs_host=float(err.mean()))
        report["runs"][str(threads)] = run
        print(f"threads={threads}: frame median {run['frame_us_median']:.1f} us, 1 s window {run['window_1s_us_median']:.1f} us, "
              f"2 s window {run['window_2s_us_median']:.1f} us; pi numpy {run.get('numpy')}; "
              f"max |pi - host| = {err.max():.2e}; throttled {run['throttled_before']} -> {run['throttled_after']}, "
              f"temp {run['temp_before']} -> {run['temp_after']}", flush=True)
    out = DEFAULT_RESULTS_DIR / f"pi_logmel_timing_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
