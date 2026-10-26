# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Write ``rpi4_summary.csv``: one row per (model, board, threads).

The columns are those of ``rp2040_summary.csv`` (see
``analysis_scripts/rp2040_latency_bench/summarize_rp2040_results.py``), with
Pi meanings where the board differs:

- ``static_ram_bytes``: the runner's peak RSS increase (model load to last run).
- ``flash_bytes``: the ``.tflite`` size.
- ``arena_used_bytes``, ``arena_kb``: empty (no tensor arena on Linux).
- ``core_version``: the LiteRT runtime (e.g. ``ai_edge_litert@2.3.0``).
- ``clock_hz``: ``scaling_cur_freq`` of cpu0 at the end of the run.

These columns are appended: ``threads``, ``xnnpack``, ``governor``,
``temp_after_c``, ``throttled_before``, ``throttled_after`` and ``flags``
(``none`` for a clean run; see ``run_rpi4_latency_bench.throttle_flags``).
For each (model, threads), the newest ``ok`` results file is used.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RP2040_BENCH = REPO_ROOT / "analysis_scripts" / "rp2040_latency_bench"
for _path in (REPO_ROOT / "src", RP2040_BENCH):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_rp2040_latency_bench import parse_results_file  # noqa: E402
from stage_tflite_candidate import DEFAULT_CANDIDATE_ROOT  # noqa: E402
from summarize_rp2040_results import COLUMNS as RP2040_COLUMNS, build_row  # noqa: E402

DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
EXTRA_COLUMNS = ["threads", "xnnpack", "governor", "temp_after_c", "throttled_before", "throttled_after", "flags"]
COLUMNS = RP2040_COLUMNS + EXTRA_COLUMNS
_THREADS_RE = re.compile(r"_rpi4_t(\d+)_")


def pick_result(model: str, threads: int, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the newest ``ok`` result (else newest any) for (model, threads)."""
    mine = [
        r for r in results
        if r.get("model") == model and (m := _THREADS_RE.search(Path(r["path"]).name)) and int(m.group(1)) == threads
    ]
    mine.sort(key=lambda r: Path(r["path"]).stat().st_mtime)
    ok = [r for r in mine if r.get("status") == "ok"]
    return (ok or mine or [None])[-1]


def pi_row(candidate: Path, threads: int, result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build one CSV row on top of the RP2040 row builder."""
    result = dict(result) if result else None
    if result:
        # Pi results carry -1 placeholders for the arena; leave those cells empty.
        for key in ("arena_kb", "arena_used_bytes"):
            if result.get(key) == -1:
                result[key] = ""
    row = build_row(candidate, result, clock_hz="", core="")
    row["board"] = "rpi4"
    row["threads"] = threads
    if result:
        khz = result.get("cpu_cur_khz")
        row.update(
            core_version=result.get("runtime", ""),
            clock_hz=int(khz) * 1000 if isinstance(khz, int) else "",
            xnnpack=result.get("xnnpack", ""),
            governor=result.get("governor", ""),
            temp_after_c=result.get("temp_after_c", ""),
            throttled_before=result.get("throttled_before", ""),
            throttled_after=result.get("throttled_after", ""),
            flags=result.get("flags", ""),
        )
    return {column: row.get(column, "") for column in COLUMNS}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "rpi4_summary.csv")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--models", nargs="+", required=True, help="Candidate names, in row order.")
    return parser.parse_args()


def main() -> int:
    """Write the CSV (threads vary fastest within each model)."""
    args = parse_args()
    results = [parse_results_file(path) for path in sorted(args.results_dir.glob("*_rpi4_t*_*.txt"))]
    rows = [
        pi_row(args.candidate_root / model, threads, pick_result(model, threads, results))
        for model in args.models
        for threads in args.threads
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['model']:26s} t={row['threads']} {row['status']:8s} median_us={row['median_us']} "
              f"fits={row['fits_stride']} flags={row['flags']}")
    print(f"csv: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
