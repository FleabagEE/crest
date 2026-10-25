# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark several RP2040 candidates: probe, size the run, then measure.

For each candidate directory, in order:

1. Pick the arena from ``analyze_tflite_memory`` (estimate + 10%, clipped to
   the RAM cap) unless ``--arena-kb`` is given.
2. Probe: ``--probe-runs`` timed runs after ``--probe-warmup`` warm-up runs.
   The serial timeout assumes a pessimistic 1 M MAC/s.
3. Full run: N = clamp((budget - warm-up time) / latency, min-runs, max-runs)
   timed runs after ``--warmup`` warm-up runs. The serial timeout is
   1.25 x the expected time + 30 s.

Whenever a run reports ``arena_too_small``, the arena grows by
``--arena-step-kb`` (up to the RAM cap) and that phase is repeated. Each run
writes ``results/<model>_<probe|full>_a<KiB>_<timestamp>.txt`` through
``run_rp2040_latency_bench.py``. ``--skip-done`` skips candidates that
already have an ``ok`` full-run file, so an interrupted plan resumes where it
stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
for _path in (SRC_DIR, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from analyze_tflite_memory import (  # noqa: E402
    DEFAULT_STATIC_RAM_BASE,
    analyze_tflite,
    recommended_arena_kb,
)
from run_rp2040_latency_bench import DEFAULT_RESULTS_DIR, parse_results_file  # noqa: E402

BENCH_SCRIPT = SCRIPT_DIR / "run_rp2040_latency_bench.py"
PESSIMISTIC_MAC_PER_S = 1.0e6


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidates", nargs="+", type=Path, help="Candidate directories, run in this order.")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--arena-kb", type=int, default=None, help="Starting arena for every candidate.")
    parser.add_argument("--arena-step-kb", type=int, default=5)
    parser.add_argument("--budget-s", type=float, default=120.0, help="Target duration of one full run.")
    parser.add_argument("--min-runs", type=int, default=30)
    parser.add_argument("--max-runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--probe-runs", type=int, default=10)
    parser.add_argument("--probe-warmup", type=int, default=2)
    parser.add_argument("--skip-done", action="store_true", help="Skip candidates with an ok full-run file.")
    return parser.parse_args()


def full_run_count(latency_s: float, args: argparse.Namespace) -> int:
    """Timed runs that fit the budget after warm-up, clamped to min/max.

    Parameters
    ----------
    latency_s : float
        Probe median latency.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Number of timed runs.
    """
    fit = math.floor((args.budget_s - args.warmup * latency_s) / latency_s)
    return int(min(args.max_runs, max(args.min_runs, fit)))


def run_phase(
    candidate: Path,
    info: Dict[str, Any],
    *,
    phase: str,
    arena_kb: int,
    runs: int,
    warmup: int,
    timeout_s: int,
    args: argparse.Namespace,
    largest_arena_kb: int,
) -> Dict[str, Any]:
    """Run one phase, growing the arena while it is too small.

    Parameters
    ----------
    candidate : Path
        Candidate directory.
    info : dict[str, Any]
        ``analyze_tflite`` output for the candidate.
    phase : str
        ``probe`` or ``full``.
    arena_kb : int
        Starting arena.
    runs, warmup, timeout_s : int
        Bench settings.
    args : argparse.Namespace
        Parsed arguments.
    largest_arena_kb : int
        Largest arena allowed by the RAM cap.

    Returns
    -------
    dict[str, Any]
        Parsed results file of the last attempt.
    """
    while True:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = args.results_dir / f"{candidate.name}_{phase}_a{arena_kb}_{stamp}.txt"
        print(
            f"[{candidate.name}] {phase}: arena={arena_kb} KiB runs={runs} warmup={warmup} timeout={timeout_s} s",
            flush=True,
        )
        cmd = [
            sys.executable, str(BENCH_SCRIPT),
            "--candidate-dir", str(candidate),
            "--port", args.port,
            "--window-size", str(info["window"]),
            "--num-channels", str(info["channels"]),
            "--arena-kb", str(arena_kb),
            "--runs", str(runs),
            "--warmup", str(warmup),
            "--serial-timeout-s", str(timeout_s),
            "--output", str(output),
        ]
        subprocess.run(cmd, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        result = parse_results_file(output) if output.exists() else {"status": "missing", "path": str(output)}
        print(
            f"[{candidate.name}] {phase}: status={result.get('status')} reason={result.get('reason')} "
            f"median_invoke_us={result.get('median_invoke_us')} arena_used_bytes={result.get('arena_used_bytes')}",
            flush=True,
        )
        if result.get("reason") == "arena_too_small" and arena_kb + args.arena_step_kb <= largest_arena_kb:
            arena_kb += args.arena_step_kb
            continue
        result["arena_kb_used"] = arena_kb
        return result


def done_full_run(candidate: Path, results_dir: Path) -> Optional[Path]:
    """Return an existing ``ok`` full-run file for the candidate, if any.

    Parameters
    ----------
    candidate : Path
        Candidate directory.
    results_dir : Path
        Results directory.

    Returns
    -------
    Path | None
        Newest matching file.
    """
    for path in sorted(results_dir.glob(f"{candidate.name}_full_a*.txt"), reverse=True):
        if parse_results_file(path).get("status") == "ok":
            return path
    return None


def main() -> int:
    """Run the plan.

    Returns
    -------
    int
        ``0`` when every candidate finished with an ``ok`` full run.
    """
    args = parse_args()
    from crest.microcontrollers.arduino_nano_rp2040 import BOARD_DEFAULT_SPEC

    ram_cap = int(BOARD_DEFAULT_SPEC.max_ram_bytes)
    largest_arena_kb = (ram_cap - DEFAULT_STATIC_RAM_BASE) // 1024
    args.results_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for candidate in (path.resolve() for path in args.candidates):
        if args.skip_done and (existing := done_full_run(candidate, args.results_dir)):
            print(f"[{candidate.name}] skipped: already done ({existing.name})", flush=True)
            continue
        info = analyze_tflite(candidate)
        arena_kb = args.arena_kb or recommended_arena_kb(
            info["arena_estimate_bytes"], static_ram_base=DEFAULT_STATIC_RAM_BASE, ram_cap=ram_cap
        )
        probe_timeout = max(
            60, math.ceil((args.probe_warmup + args.probe_runs) * info["total_macs"] / PESSIMISTIC_MAC_PER_S * 1.5)
        )
        probe = run_phase(
            candidate, info, phase="probe", arena_kb=arena_kb, runs=args.probe_runs,
            warmup=args.probe_warmup, timeout_s=probe_timeout, args=args, largest_arena_kb=largest_arena_kb,
        )
        if probe.get("status") != "ok":
            print(f"[{candidate.name}] infeasible in probe: {probe.get('reason')}", flush=True)
            failures += 1
            continue
        latency_s = float(probe["median_invoke_us"]) / 1e6
        runs = full_run_count(latency_s, args)
        timeout = math.ceil((args.warmup + runs) * latency_s * 1.25 + 30)
        full = run_phase(
            candidate, info, phase="full", arena_kb=int(probe["arena_kb_used"]), runs=runs,
            warmup=args.warmup, timeout_s=timeout, args=args, largest_arena_kb=largest_arena_kb,
        )
        if full.get("status") != "ok":
            failures += 1
        print(json.dumps({key: full.get(key) for key in ("path", "status", "median_invoke_us", "p95_invoke_us", "n")}))
    print("PLAN DONE", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
