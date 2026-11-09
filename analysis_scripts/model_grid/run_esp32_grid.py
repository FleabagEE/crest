# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Run the model-grid measurement plan on the XIAO ESP32-S3.

For every model whose ``plan_esp32s3`` in ``model_grid_triage.csv`` is a run,
this calls ``run_rp2040_bench_plan.py --board xiao_esp32s3 --skip-done`` with:

- the planned timed and warm-up runs (``--min-runs = --max-runs = N``);
- a starting arena from the ESP-NN rule, max(1.7 x RP2040 estimate,
  estimate + largest conv filter) + 4 KiB, capped at ``INTERNAL_MAX_KB``;
- ``--arena-step-kb`` steps (default 32 KiB) on ``arena_too_small``;
- internal SRAM, unless the plan says PSRAM. When internal SRAM fails, the
  model is retried with a PSRAM arena starting at the internal cap, since
  every smaller arena has already been tried or does not link.

``arena_used_bytes`` is reported by the interpreter, so the step size only
affects how many uploads a model needs, not the measured arena.

Models must be staged as ``models/rp2040_bench/<model>_int8`` (see
``stage_tflite_candidate.py``). Set ``ARDUINO_CLI_BIN`` to a wrapper adding
``--jobs 4`` on memory-limited WSL hosts.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PLAN_SCRIPT = REPO_ROOT / "analysis_scripts" / "rp2040_latency_bench" / "run_rp2040_bench_plan.py"
TRIAGE_CSV = SCRIPT_DIR / "model_grid_triage.csv"

# Largest arena that links in internal SRAM with the 11-op latency sketch:
# 232 KiB links, 233 KiB overflows dram0_0_seg by 96 B (measured 2026-11-08).
INTERNAL_MAX_KB = 232
ESP_NN_START_RATIO = 1.7


def start_arena_kb(estimate: int, weight_copy: int, psram: bool) -> int:
    """Starting arena in KiB for one model."""
    kb = math.ceil(max(ESP_NN_START_RATIO * estimate, weight_copy) / 1024) + 4
    return kb if psram else min(kb, INTERNAL_MAX_KB)


def run_plan(model: str, arena_kb: int, runs: int, warmup: int, memory: str, args: argparse.Namespace) -> int:
    """Run one model through the plan runner; return its exit code."""
    cmd = [
        sys.executable, str(PLAN_SCRIPT), str(args.candidate_root / f"{model}_int8"),
        "--board", "xiao_esp32s3", "--port", args.port,
        "--arena-kb", str(arena_kb), "--arena-step-kb", str(args.arena_step_kb),
        "--min-runs", str(runs), "--max-runs", str(runs), "--warmup", str(warmup),
        "--arena-memory", memory, "--skip-done",
    ]
    print(f"=== {model} memory={memory} start_arena={arena_kb} KiB N={runs} warm={warmup}", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", help="Only these models (default: every planned run).")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--arena-step-kb", type=int, default=32)
    parser.add_argument("--candidate-root", type=Path, default=REPO_ROOT / "models" / "rp2040_bench")
    parser.add_argument("--triage-csv", type=Path, default=TRIAGE_CSV)
    return parser.parse_args()


def main() -> int:
    """Run every planned ESP32-S3 model; return non-zero if any failed."""
    args = parse_args()
    rows = [r for r in csv.DictReader(args.triage_csv.open(encoding="utf-8"))
            if not r["plan_esp32s3"].startswith("skip") and (not args.models or r["model"] in args.models)]
    failures = 0
    for row in rows:
        match = re.match(r"run N=(\d+) warm=(\d+)", row["plan_esp32s3"])
        runs, warmup = int(match.group(1)), int(match.group(2))
        psram = "PSRAM" in row["plan_esp32s3"]
        estimate, weight_copy = int(row["arena_estimate_b"]), int(row["esp32_arena_weightcopy_b"])
        memory = "psram" if psram else "internal_sram"
        rc = run_plan(row["model"], start_arena_kb(estimate, weight_copy, psram), runs, warmup, memory, args)
        if rc != 0 and not psram:
            print(f"=== {row['model']}: internal SRAM failed, PSRAM fallback from {INTERNAL_MAX_KB} KiB", flush=True)
            rc = run_plan(row["model"], INTERNAL_MAX_KB, runs, warmup, "psram", args)
        failures += rc != 0
    print("GRID DONE", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
