# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the RP2040 latency bench helper scripts."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT_DIR / "analysis_scripts" / "rp2040_latency_bench"
for _path in (ROOT_DIR / "src", BENCH_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import analyze_tflite_memory  # noqa: E402
import run_rp2040_bench_plan  # noqa: E402
import run_rp2040_latency_bench  # noqa: E402
import stage_tflite_candidate  # noqa: E402

RESULTS_TEXT = """# CREST Nano RP2040 Connect latency/memory result
# board: nano_rp2040_connect  model: m1  port: /dev/ttyACM0
# timed_runs: 3  warmup_runs: 5  preprocess_on_device: no (preprocess_us=-1)
#
# arena sweep:
#   arena_kb=10 status=infeasible reason=arena_too_small flash_bytes=1 ram_bytes=2 upload_s=1.0
#   arena_kb=40 status=ok reason=None flash_bytes=359887 ram_bytes=91472 upload_s=19.0
#
# summary:
#   status=ok arena_kb=40 arena_used_bytes=27588
#   flash_bytes=359887 ram_bytes=91472 (arena included in ram_bytes)
#   n=3 median_invoke_us=101.0 p95_invoke_us=102 min_invoke_us=100 max_invoke_us=102 mean_invoke_us=101.0
#
# raw per-run lines:
board=nano_rp2040_connect model=m1 run=1 invoke_us=100 preprocess_us=-1 arena_used_bytes=27588 status=ok
board=nano_rp2040_connect model=m1 run=2 invoke_us=101 preprocess_us=-1 arena_used_bytes=27588 status=ok
board=nano_rp2040_connect model=m1 run=3 invoke_us=102 preprocess_us=-1 arena_used_bytes=27588 status=ok

# full serial log of the final attempt:
#   board=nano_rp2040_connect model=m1 run=1 invoke_us=100 preprocess_us=-1 arena_used_bytes=27588 status=ok
"""


class ResultsParsingTests(unittest.TestCase):
    """Tests covering results-file parsing."""

    def test_parse_results_file_reads_summary_not_sweep(self):
        # Summary fields must come from the final summary, not the arena sweep lines.
        """Validate results-file parsing."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m1_full_a40_20261025_000000.txt"
            path.write_text(RESULTS_TEXT, encoding="utf-8")
            parsed = run_rp2040_latency_bench.parse_results_file(path)
        self.assertEqual(parsed["model"], "m1")
        self.assertEqual(parsed["status"], "ok")
        self.assertEqual(parsed["arena_kb"], 40)
        self.assertEqual(parsed["arena_used_bytes"], 27588)
        self.assertEqual(parsed["median_invoke_us"], 101.0)
        self.assertEqual(parsed["p95_invoke_us"], 102)
        self.assertEqual(parsed["timed_runs"], 3)
        # Commented serial-log copies of run lines are not double counted.
        self.assertEqual(parsed["invoke_us"], [100, 101, 102])


class PlanSizingTests(unittest.TestCase):
    """Tests covering arena and run-count sizing."""

    def test_recommended_arena_adds_margin_and_respects_cap(self):
        # 10% margin normally; clipped to the RAM cap but never below the raw estimate.
        """Validate arena recommendation."""
        self.assertEqual(
            analyze_tflite_memory.recommended_arena_kb(21_984, static_ram_base=50_512, ram_cap=225_000), 24
        )
        self.assertEqual(
            analyze_tflite_memory.recommended_arena_kb(158_592, static_ram_base=50_512, ram_cap=225_000), 170
        )
        self.assertEqual(
            analyze_tflite_memory.recommended_arena_kb(200_000, static_ram_base=50_512, ram_cap=225_000), 196
        )

    def test_full_run_count_budget_and_floor(self):
        # Fast models cap at max_runs; slow models never drop below min_runs.
        """Validate full-run sizing."""
        args = argparse.Namespace(budget_s=120.0, warmup=5, min_runs=30, max_runs=100)
        self.assertEqual(run_rp2040_bench_plan.full_run_count(0.179, args), 100)
        self.assertEqual(run_rp2040_bench_plan.full_run_count(1.2822, args), 88)
        self.assertEqual(run_rp2040_bench_plan.full_run_count(2.8323, args), 37)
        self.assertEqual(run_rp2040_bench_plan.full_run_count(5.4819, args), 30)


class StagingMetadataTests(unittest.TestCase):
    """Tests covering course-repo export metadata."""

    def test_run_json_metadata_and_candidate_name(self):
        # The exports layout yields subnet metadata and a stable candidate name.
        """Validate run.json metadata extraction."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run1"
            subnet = run_dir / "subnets" / "DSCNN_64_SC_5"
            export = subnet / "exports" / "2s" / "int8.tflite"
            export.parent.mkdir(parents=True)
            export.write_bytes(b"x")
            (run_dir / "run.json").write_text(json.dumps({"state": {"held_out_test_evaluated": False}}))
            (subnet / "run.json").write_text(
                json.dumps(
                    {
                        "architecture": {"channels": [16, 32, 64, 64], "num_parameters": 60874},
                        "metadata": {"canonical_name": "DSCNN_64_SC_5", "effective_width": "5"},
                        "evaluation": {"2s": {"val": {"int8_tflite": {"accuracy": 0.69, "macro_f1": 0.70}}}},
                        "state": {},
                    }
                )
            )
            meta = stage_tflite_candidate.run_json_metadata(export)
            name = stage_tflite_candidate.default_candidate_name(export)
        self.assertEqual(name, "DSCNN_64_SC_5_2s_int8")
        self.assertEqual(meta["width_token"], "5")
        self.assertEqual(meta["block_channels"], [16, 32, 64, 64])
        self.assertEqual(meta["params"], 60874)
        self.assertEqual((meta["val_accuracy"], meta["val_macro_f1"]), (0.69, 0.70))
        self.assertIs(meta["held_out_test_evaluated"], False)
        self.assertIsNone(stage_tflite_candidate.run_json_metadata(Path(tmp) / "plain.tflite"))


if __name__ == "__main__":
    unittest.main()
