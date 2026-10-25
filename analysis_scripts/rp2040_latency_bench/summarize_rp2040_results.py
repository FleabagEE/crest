# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Write one CSV row per (model, board) from RP2040 bench results.

Rows come from candidate directories (``candidate.json`` + ``g_model.tflite``)
joined with the newest ``ok`` results file for that model in the results
directory (full-run files preferred). A candidate with no ``ok`` result gets
a row with ``status`` set to the newest file's status/reason.

``mac_per_s`` = MACs / median latency. ``fits_stride`` means median latency <
stride, with the stride taken from the export duration (``1s``/``2s``) or,
failing that, from the input window (<= 101 frames -> 1 s, else 2 s).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
for _path in (SRC_DIR, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from analyze_tflite_memory import analyze_tflite  # noqa: E402
from run_rp2040_latency_bench import DEFAULT_RESULTS_DIR, parse_results_file  # noqa: E402
from stage_tflite_candidate import DEFAULT_CANDIDATE_ROOT  # noqa: E402

# Width tokens of the course-project supernet (its training/utils.py WIDTHS).
WIDTH_MULTIPLIERS = {"1": 1.0, "5": 0.5, "25": 0.25}
# clocks_init in mbed_nano 4.6.0's libmbed.a: pll_sys VCO 1500 MHz / 6 / 2,
# clk_sys configured to 125,000,000 Hz (read from the binary, not measured).
DEFAULT_CLOCK_HZ = 125_000_000
CORE_DIR = REPO_ROOT / "tools" / "arduino-data" / "packages" / "arduino" / "hardware" / "mbed_nano"

COLUMNS = [
    "model", "board", "width", "window", "macs", "params", "flash_bytes", "static_ram_bytes",
    "arena_used_bytes", "median_us", "p95_us", "n_runs", "mac_per_s", "stride_s", "fits_stride",
    "val_acc", "val_macro_f1", "core_version", "clock_hz", "arena_kb", "status", "results_file",
]


def core_version() -> str:
    """Return the installed ``arduino:mbed_nano`` version.

    Returns
    -------
    str
        For example ``arduino:mbed_nano@4.6.0``, or ``unknown``.
    """
    versions = sorted(path.name for path in CORE_DIR.glob("*") if path.is_dir())
    return f"arduino:mbed_nano@{versions[-1]}" if versions else "unknown"


def pick_result(model: str, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Choose the result for one model.

    Parameters
    ----------
    model : str
        Model id (candidate directory name).
    results : list[dict[str, Any]]
        All parsed results files.

    Returns
    -------
    dict[str, Any] | None
        Newest ``ok`` full run, else newest ``ok`` run, else newest run.
    """
    mine = sorted((r for r in results if r.get("model") == model), key=lambda r: Path(r["path"]).stat().st_mtime)
    ok = [r for r in mine if r.get("status") == "ok"]
    full = [r for r in ok if "_full_" in Path(r["path"]).name]
    for group in (full, ok, mine):
        if group:
            return group[-1]
    return None


def build_row(candidate: Path, result: Optional[Dict[str, Any]], clock_hz: int, core: str) -> Dict[str, Any]:
    """Build one CSV row.

    Parameters
    ----------
    candidate : Path
        Candidate directory.
    result : dict[str, Any] | None
        Chosen results file.
    clock_hz : int
        Core clock recorded in the row.
    core : str
        Core version recorded in the row.

    Returns
    -------
    dict[str, Any]
        Row keyed by ``COLUMNS``.
    """
    meta = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
    analysis = analyze_tflite(candidate)
    window = int(analysis["window"])
    duration = str(meta.get("duration", ""))
    stride_s = float(duration[:-1]) if duration.endswith("s") else (1.0 if window <= 101 else 2.0)
    width = WIDTH_MULTIPLIERS.get(str(meta.get("width_token")), "") if meta.get("width_token") else ""
    row: Dict[str, Any] = {
        "model": candidate.name,
        "board": (result or {}).get("board", "nano_rp2040_connect"),
        "width": width,
        "window": window,
        "macs": analysis["total_macs"],
        "params": meta.get("params", ""),
        "stride_s": stride_s,
        "val_acc": meta.get("val_accuracy", ""),
        "val_macro_f1": meta.get("val_macro_f1", ""),
        "core_version": core,
        "clock_hz": clock_hz,
        "status": "no_result",
    }
    if result is not None:
        row.update(
            flash_bytes=result.get("flash_bytes", ""),
            static_ram_bytes=result.get("ram_bytes", ""),
            arena_kb=result.get("arena_kb", ""),
            results_file=str(Path(result["path"]).relative_to(REPO_ROOT)) if Path(result["path"]).is_relative_to(REPO_ROOT) else result["path"],
            status=result.get("status") if result.get("status") == "ok" else f"{result.get('status')}:{result.get('reason')}",
        )
        if result.get("status") == "ok":
            median_us = float(result["median_invoke_us"])
            row.update(
                arena_used_bytes=result.get("arena_used_bytes"),
                median_us=median_us,
                p95_us=result.get("p95_invoke_us"),
                n_runs=result.get("n"),
                mac_per_s=round(analysis["total_macs"] / (median_us / 1e6)),
                fits_stride=median_us / 1e6 < stride_s,
            )
    return {column: row.get(column, "") for column in COLUMNS}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "rp2040_summary.csv")
    parser.add_argument("--clock-hz", type=int, default=DEFAULT_CLOCK_HZ)
    parser.add_argument("--models", nargs="*", default=None, help="Candidate names, in row order (default: all).")
    return parser.parse_args()


def main() -> int:
    """Write the summary CSV.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args()
    results = [parse_results_file(path) for path in sorted(args.results_dir.glob("*.txt"))]
    if args.models:
        candidates = [args.candidate_root / name for name in args.models]
    else:
        candidates = sorted(p for p in args.candidate_root.iterdir() if (p / "candidate.json").exists())
    core = core_version()
    rows = [build_row(c, pick_result(c.name, results), args.clock_hz, core) for c in candidates]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['model']:26s} {row['status']:8s} median_us={row['median_us']} fits={row['fits_stride']}")
    print(f"csv: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
