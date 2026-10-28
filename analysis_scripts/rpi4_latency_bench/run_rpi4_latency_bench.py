# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Time staged candidates on a Raspberry Pi 4 through ``RaspberryPi4Device``.

Each candidate directory needs ``g_model.tflite`` (for example the ones
staged for the RP2040 bench in ``models/rp2040_bench/``). For every candidate
the model and runner are copied to the Pi, ``--warmup`` + ``--runs`` invokes
are run, and ``results/<model>_rpi4_t<threads>_<timestamp>.txt`` is written in
the RP2040 bench's results format (header, ``# summary:``, raw per-run lines,
remote output). ``parse_results_file`` and the summary tooling can read it.

Before any run, every core's cpufreq governor must equal
``--require-governor`` (default ``performance``), otherwise the script stops.
Each results file records the thread count, governor, clock, temperature and
``vcgencmd get_throttled`` before/after the timed runs, and ``flags`` marks
under-voltage, frequency capping, throttling or soft temperature limits
(active after the run or newly set during it) and SoC temperatures >= 80 C.

Nothing is installed on the Pi: ``--python`` must already point at an
interpreter with a LiteRT runtime.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.microcontrollers.arduino_nano_rp2040 import parse_run_lines  # noqa: E402
from crest.microcontrollers.raspberry_pi4 import (  # noqa: E402
    BOARD_ID,
    RaspberryPi4Device,
    parse_peak_rss_delta,
)

DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidates", nargs="+", type=Path, help="Candidate directories with g_model.tflite.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default=None)
    parser.add_argument("--python", default="crest-litert/bin/python", help="Interpreter on the Pi (relative to home).")
    parser.add_argument("--remote-dir", default="crest_rpi4")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-xnnpack", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=300.0, help="Limit for one remote run.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--require-governor",
        default="performance",
        help="Abort before running anything unless every core uses this cpufreq governor.",
    )
    return parser.parse_args()


def percentile_nearest_rank(values: List[int], pct: float) -> int:
    """Return the nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[max(1, math.ceil(pct / 100.0 * len(ordered))) - 1]


# vcgencmd get_throttled bits: 0-3 are "now", 16-19 "has occurred since boot".
_THROTTLE_BITS = {0: "under_voltage", 1: "freq_capped", 2: "throttled", 3: "soft_temp_limit"}
HOT_TEMP_C = 80.0


def runner_env(lines: List[str]) -> Dict[str, str]:
    """Collect ``key=value`` pairs from the runner's environment lines."""
    env: Dict[str, str] = {}
    for line in lines:
        if line.startswith(("board=", "runs:", "timer output", "stderr:")):
            continue
        for token in line.split():
            key, sep, value = token.partition("=")
            if sep:
                env[key] = value
    return env


def throttle_flags(before: str, after: str, temp_after_c: str) -> List[str]:
    """Return the flags for one run.

    Parameters
    ----------
    before, after : str
        ``vcgencmd get_throttled`` values (hex) before and after the timed
        runs, or ``na``.
    temp_after_c : str
        SoC temperature after the runs.

    Returns
    -------
    list[str]
        ``<cond>_now`` when a condition is active after the run,
        ``<cond>_during_run`` when its sticky bit was newly set during the run,
        ``throttle_unknown`` when the flags could not be read, and ``hot``
        when the SoC reached ``HOT_TEMP_C``. Empty when the run is clean.
    """
    flags: List[str] = []
    try:
        before_bits, after_bits = int(before, 16), int(after, 16)
    except ValueError:
        flags.append("throttle_unknown")
    else:
        for bit, name in _THROTTLE_BITS.items():
            if after_bits & (1 << bit):
                flags.append(f"{name}_now")
            sticky = 1 << (bit + 16)
            if after_bits & sticky and not before_bits & sticky:
                flags.append(f"{name}_during_run")
    try:
        if float(temp_after_c) >= HOT_TEMP_C:
            flags.append("hot")
    except ValueError:
        pass
    return flags


def check_governors(device: RaspberryPi4Device, required: str) -> Dict[str, Any]:
    """Read every core's cpufreq governor and current frequency over ssh.

    Parameters
    ----------
    device : RaspberryPi4Device
        Configured device.
    required : str
        Governor every core must use.

    Returns
    -------
    dict[str, Any]
        ``ok``, ``governors`` (one per core), ``cur_khz``, ``throttled`` and
        ``temp`` at session start, plus ``error`` when ssh failed.
    """
    probe = device._ssh(
        "cat /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; echo ---; "
        "cat /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq; echo ---; "
        "vcgencmd get_throttled; vcgencmd measure_temp",
        device.options.ssh_connect_timeout_s + 20,
    )
    if probe.returncode != 0:
        return {"ok": False, "error": probe.stderr.strip()}
    governors, freqs, extra = (part.split() for part in probe.stdout.split("---"))
    return {
        "ok": bool(governors) and all(g == required for g in governors),
        "governors": governors,
        "cur_khz": freqs,
        "throttled": next((e.split("=")[-1] for e in extra if e.startswith("throttled=")), "na"),
        "temp": next((e.split("=")[-1] for e in extra if e.startswith("temp=")), "na"),
    }


def bench_one(device: RaspberryPi4Device, candidate: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Upload and measure one candidate and write its results file."""
    candidate = candidate.resolve()
    compile_result = device.compile(sketch_path=candidate, arena_kb=-1, window_size=0, num_channels=0)
    status, reason, lines, records = "infeasible", None, [], []
    if not compile_result.success:
        reason = "flash_overflow" if compile_result.overflow_kind == "flash" else "missing_model"
    else:
        upload = device.upload(sketch_path=candidate, build_dir=None, serial_port=None)
        if not upload.success:
            reason, lines = "upload_failed", upload.log.splitlines()
        else:
            measure = device.measure(serial_timeout_s=args.timeout_s, measured_inference_runs=args.runs,
                                     sketch_path=candidate)
            lines = measure.serial_log
            records = parse_run_lines(lines)
            ok = [r for r in records if r.get("status") == "ok"]
            bad = [r for r in records if r.get("status") == "infeasible"]
            if measure.latency_s is not None and ok:
                status = "ok"
            else:
                reason = bad[0].get("reason") if bad else "no_telemetry"
    peak_rss = parse_peak_rss_delta(lines)
    env_map = runner_env(lines)
    flags = throttle_flags(env_map.get("throttled_before", "na"), env_map.get("throttled", "na"),
                           env_map.get("temp_after_c", "na")) if status == "ok" else []
    env = [line for line in lines if line.split("=", 1)[0] in {
        "runtime", "num_threads", "python", "cpu_governor", "temp_before_c", "rss_before_load_bytes"}]
    stamp = datetime.now()
    out = args.results_dir / f"{candidate.name}_{BOARD_ID}_t{args.threads}_{stamp:%Y%m%d_%H%M%S}.txt"
    text = [
        "# CREST Raspberry Pi 4 latency/memory result",
        f"# generated_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"# board: {BOARD_ID}  model: {candidate.name}  port: {device.options.destination}",
        f"# timed_runs: {args.runs}  warmup_runs: {args.warmup}  preprocess_on_device: no (preprocess_us=-1)",
        "# ram_bytes = runner peak RSS increase from model load to the last run; flash_bytes = .tflite size",
        "#",
        "# summary:",
    ]
    if status == "ok":
        invoke = [int(r["invoke_us"]) for r in records if r.get("status") == "ok"]
        text += [
            f"#   status=ok arena_kb=-1 arena_used_bytes=-1",
            f"#   flash_bytes={compile_result.flash_bytes} ram_bytes={peak_rss}",
            f"#   n={len(invoke)} median_invoke_us={statistics.median(invoke):.1f} "
            f"p95_invoke_us={percentile_nearest_rank(invoke, 95)} min_invoke_us={min(invoke)} "
            f"max_invoke_us={max(invoke)} mean_invoke_us={statistics.fmean(invoke):.1f}",
            "#   p95 uses the nearest-rank method.",
            f"#   threads={args.threads} xnnpack={'off' if args.no_xnnpack else 'default'} "
            f"runtime={env_map.get('runtime', 'na')} governor={env_map.get('cpu_governor', 'na')} "
            f"cpu_cur_khz={env_map.get('cpu_cur_khz', 'na')}",
            f"#   throttled_before={env_map.get('throttled_before', 'na')} "
            f"throttled_after={env_map.get('throttled', 'na')} temp_before_c={env_map.get('temp_before_c', 'na')} "
            f"temp_after_c={env_map.get('temp_after_c', 'na')} flags={'+'.join(flags) or 'none'}",
        ]
    else:
        text.append(f"#   status=infeasible reason={reason}")
    # Environment goes after the raw lines so parse_results_file's summary
    # section (which ends at "# raw") only contains summary fields.
    text += ["#", "# raw per-run lines:"] + [r["raw"] for r in records]
    text += ["", "# environment:"] + [f"#   {line}" for line in env]
    text += ["", "# full remote output:"] + [f"#   {line}" for line in lines]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(text) + "\n", encoding="utf-8")
    summary = {"model": candidate.name, "status": status, "reason": reason, "file": str(out)}
    if status == "ok":
        summary.update(median_us=statistics.median(invoke), p95_us=percentile_nearest_rank(invoke, 95),
                       ram_bytes=peak_rss, flash_bytes=compile_result.flash_bytes,
                       temp_after_c=env_map.get("temp_after_c"), throttled=env_map.get("throttled"),
                       flags="+".join(flags) or "none")
    return summary


def main() -> int:
    """Benchmark every candidate; return non-zero if any failed."""
    args = parse_args()
    device = RaspberryPi4Device(device_options={
        "host": args.host, "user": args.user, "python": args.python, "remote_dir": args.remote_dir,
        "num_threads": args.threads, "xnnpack": not args.no_xnnpack, "warmup_runs": args.warmup,
    })
    session = check_governors(device, args.require_governor)
    print(f"session check: {session}", flush=True)
    if not session["ok"]:
        reason = session.get("error") or f"got {session.get('governors')}"
        print(f"ABORT: Pi not reachable or not every core on the '{args.require_governor}' governor ({reason}); "
              "nothing was run.", flush=True)
        return 3
    failures = 0
    for candidate in args.candidates:
        summary = bench_one(device, candidate, args)
        print(summary, flush=True)
        failures += summary["status"] != "ok"
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
