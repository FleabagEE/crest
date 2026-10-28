# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Time the training repo's numpy ``log_mel`` on the Pi (runs ON the Pi).

Imports the course repo's ``training/features.py`` from ``--features`` and
calls its ``log_mel`` unchanged. The only substitution is ``mel_filter``,
which returns the matrix in ``--mel`` (``librosa.filters.mel(...).T``
exported on the host by ``run_pi_logmel_timing.py``) so librosa does not have
to be installed on the Pi.

Prints ``key=value`` lines:

- ``frame_us_*``: one frame (400 samples, i.e. the work for one 10 ms hop);
- ``window_1s_us_*``: a 1 s window (16,000 samples, 98 frames);
- ``window_2s_us_*``: a 2 s window (32,000 samples, 198 frames);
- ``check_b64``: float32 log-mel of a fixed 1 s chirp, for comparison with
  the host.

BLAS/OpenMP threads are pinned to ``--threads`` before numpy is imported, so
the mel matrix product uses 1 or 4 cores to match the inference thread
setting. Standard library + numpy only.
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--features", required=True, help="Path to training/features.py from the course repo.")
parser.add_argument("--mel", required=True, help="Mel matrix .npy, shape (257, 64), float32.")
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--frame-runs", type=int, default=500)
parser.add_argument("--window-runs", type=int, default=50)
parser.add_argument("--warmup", type=int, default=5)
args = parser.parse_args()
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = str(args.threads)

import base64  # noqa: E402
import importlib.util  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

AUDIO = {
    "sample_rate": 16000, "duration_sec": 1, "n_mels": 64, "frame_length_ms": 25, "hop_length_ms": 10,
    "fft_length": 512, "log_offset": 1.0e-6, "spectrum": "power",
}


def load_features(path, mel_path):
    """Import features.py and substitute the exported mel matrix."""
    spec = importlib.util.spec_from_file_location("crest_training_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mel = np.load(mel_path).astype(np.float32)
    module.mel_filter = lambda sample_rate, fft_length, n_mels: mel
    return module


def time_call(fn, runs, warmup):
    """Median/min/p95 microseconds of ``fn()`` over ``runs`` calls after warm-up."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1000.0)
    samples.sort()
    return {
        "median": statistics.median(samples),
        "min": samples[0],
        "p95": samples[max(0, int(round(0.95 * len(samples))) - 1)],
        "n": len(samples),
    }


def chirp(seconds):
    """Deterministic test waveform (no RNG), int16-quantized like the golden clips."""
    n = int(16000 * seconds)
    t = np.arange(n) / 16000
    x = 10 ** (-12 / 20) * np.sin(2 * np.pi * (50 * t + (7950 - 50) / (2 * seconds) * t ** 2))
    return (np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16).astype(np.float32) / np.float32(32768.0))


def main():
    """Run the timings and print key=value lines."""
    features = load_features(args.features, args.mel)
    log_mel = features.log_mel
    w1, w2 = chirp(1.0), chirp(2.0)
    frame = w1[:400].copy()
    results = {
        "frame_us": time_call(lambda: log_mel(frame, AUDIO), args.frame_runs, args.warmup),
        "window_1s_us": time_call(lambda: log_mel(w1, AUDIO), args.window_runs, args.warmup),
        "window_2s_us": time_call(lambda: log_mel(w2, AUDIO), args.window_runs, args.warmup),
    }
    print(f"threads={args.threads} numpy={np.__version__} python={sys.version.split()[0]}", flush=True)
    for name, stats in results.items():
        print(" ".join(f"{name}_{key}={value:.1f}" if key != "n" else f"{name}_n={value}" for key, value in stats.items()),
              flush=True)
    check = log_mel(w1, AUDIO)[..., 0].astype("<f4")
    print(f"check_shape={check.shape[0]}x{check.shape[1]}", flush=True)
    print("check_b64=" + base64.b64encode(check.tobytes()).decode("ascii"), flush=True)


if __name__ == "__main__":
    main()
