# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the Raspberry Pi 4 LiteRT-over-SSH backend."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.errors import (  # noqa: E402
    HIL_ERROR_FLASH_OVERFLOW,
    HIL_ERROR_LATENCY,
    HIL_ERROR_OK,
    HIL_ERROR_UPLOAD,
)
from crest.microcontrollers import get_device, list_device_specs, raspberry_pi4, resolve_device_options  # noqa: E402

RUNNER_OUTPUT = [
    "board=rpi4 model=cand run=1 invoke_us=150 preprocess_us=-1 arena_used_bytes=-1 status=ok",
    "board=rpi4 model=cand run=2 invoke_us=152 preprocess_us=-1 arena_used_bytes=-1 status=ok",
    "runtime=ai_edge_litert@1.4.0",
    "rss_before_load_bytes=1000 peak_rss_bytes=5000 peak_rss_delta_bytes=4000 peak_window=vmhwm model_file_bytes=10",
    "runs: 2",
    "timer output: 0.000151000",
]


def _done(returncode=0, stdout="", stderr=""):
    """Build a fake CompletedProcess."""
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def _candidate(tmp: str, size: int = 10) -> Path:
    """Create a candidate directory with a fake model of ``size`` bytes."""
    candidate = Path(tmp) / "cand"
    candidate.mkdir()
    (candidate / "g_model.tflite").write_bytes(b"\0" * size)
    return candidate


class RaspberryPi4ContractTests(unittest.TestCase):
    """Tests covering options, spec and registry wiring."""

    def test_spec_and_flags(self):
        # The Pi uses the single-shot arena sentinel and never measures energy.
        """Validate spec and capability flags."""
        device = raspberry_pi4.RaspberryPi4Device(device_options={"host": "pi.local"})
        self.assertEqual(device.spec.name, "RASPBERRY_PI_4")
        self.assertEqual(device.spec.arena_sizes_kb, [-1])
        self.assertIsNone(device.spec.fqbn)
        self.assertFalse(device.requires_arena_validation())
        self.assertFalse(device.supports_energy_measurement())
        self.assertTrue(device.requires_candidate_model())

    def test_options_and_validation(self):
        # Shared runtime keys are ignored; malformed SSH/runtime options are rejected.
        """Validate option resolution."""
        options = raspberry_pi4.resolve_rpi4_options(
            {"host": "pi.local", "user": "rpi4", "num_threads": 4, "xnnpack": False, "runtime_mode": "back_to_back"}
        )
        self.assertEqual(options.destination, "rpi4@pi.local")
        self.assertEqual((options.num_threads, options.xnnpack, options.warmup_runs), (4, False, 5))
        with self.assertRaises(ValueError):
            raspberry_pi4.RaspberryPi4Options().destination
        for bad in ({"num_threads": 0}, {"xnnpack": "yes"}, {"host": "pi local"}, {"ssh_connect_timeout_s": 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                raspberry_pi4.resolve_rpi4_options(bad)

    def test_registry(self):
        # get_device and resolve_device_options route RASPBERRY_PI_4 to this backend.
        """Validate registry wiring."""
        self.assertEqual(list_device_specs()["RASPBERRY_PI_4"]["arena_sizes"], [-1])
        options = resolve_device_options("raspberry_pi_4", {"rpi4": {"host": "pi.local", "num_threads": 2}})
        self.assertEqual(options, {"host": "pi.local", "num_threads": 2})
        device = get_device("RASPBERRY_PI_4", device_options=options)
        self.assertIsInstance(device, raspberry_pi4.RaspberryPi4Device)


class RaspberryPi4FlowTests(unittest.TestCase):
    """Tests covering compile/upload/measure/evaluate with mocked SSH."""

    def setUp(self):
        """Create a device with a fixed host."""
        self.device = raspberry_pi4.RaspberryPi4Device(
            device_options={"host": "pi.local", "user": "rpi4", "num_threads": 4}
        )

    def test_upload_and_measure_commands(self):
        # Upload creates the remote dir then scp's model + runner; measure runs the runner with the options.
        """Validate the ssh/scp command lines."""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _candidate(tmp)
            with patch.object(raspberry_pi4, "run_command", return_value=_done()) as run_mock:
                self.assertTrue(self.device.upload(sketch_path=candidate, build_dir=None, serial_port=None).success)
            mkdir_cmd, scp_cmd = (call.args[0] for call in run_mock.call_args_list)
            self.assertEqual(mkdir_cmd[0], "ssh")
            self.assertIn("BatchMode=yes", mkdir_cmd)
            self.assertEqual(mkdir_cmd[-1], "mkdir -p crest_rpi4/cand")
            self.assertEqual(scp_cmd[0], "scp")
            self.assertEqual(scp_cmd[-1], "rpi4@pi.local:crest_rpi4/cand/")
            self.assertTrue(scp_cmd[-2].endswith("crest_litert_runner.py"))
            with patch.object(raspberry_pi4, "run_command", return_value=_done(stdout="\n".join(RUNNER_OUTPUT))) as run_mock:
                result = self.device.measure(serial_timeout_s=30, measured_inference_runs=2, sketch_path=candidate)
            remote = run_mock.call_args.args[0][-1]
        self.assertIn("crest-litert/bin/python crest_rpi4/cand/crest_litert_runner.py", remote)
        self.assertIn("--runs 2 --warmup 5 --threads 4", remote)
        self.assertNotIn("--no-xnnpack", remote)
        self.assertAlmostEqual(result.latency_s, 0.000151)
        self.assertEqual(raspberry_pi4.parse_peak_rss_delta(result.serial_log), 4000)

    def test_evaluate_success_and_error_mapping(self):
        # Success reports file size + RSS delta; scp failure, missing latency and oversize models map to HIL errors.
        """Validate evaluate()."""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _candidate(tmp)
            outputs = [_done(), _done(), _done(stdout="\n".join(RUNNER_OUTPUT))]
            with patch.object(raspberry_pi4, "run_command", side_effect=outputs):
                ok = self.device.evaluate(dirpath=candidate, arena_kb=-1, window_size=98, num_channels=64,
                                          measured_inference_runs=2, harness_fqbn="ignored")
            self.assertEqual((ok.error_code, ok.flash_bytes, ok.ram_bytes, ok.arena_bytes), (HIL_ERROR_OK, 10, 4000, -1))
            self.assertAlmostEqual(ok.latency_s, 0.000151)
            with patch.object(raspberry_pi4, "run_command", side_effect=[_done(), _done(returncode=1, stderr="denied")]):
                self.assertEqual(self.device.evaluate(dirpath=candidate, arena_kb=-1, window_size=98,
                                                      num_channels=64).error_code, HIL_ERROR_UPLOAD)
            failed_run = _done(returncode=2, stdout="board=rpi4 model=cand run=0 invoke_us=-1 preprocess_us=-1 "
                                                    "arena_used_bytes=-1 status=infeasible reason=no_tflite_runtime")
            with patch.object(raspberry_pi4, "run_command", side_effect=[_done(), _done(), failed_run]):
                self.assertEqual(self.device.evaluate(dirpath=candidate, arena_kb=-1, window_size=98,
                                                      num_channels=64).error_code, HIL_ERROR_LATENCY)
            with patch.object(raspberry_pi4, "run_command") as run_mock:
                compile_only = self.device.evaluate(dirpath=candidate, arena_kb=-1, window_size=98,
                                                    num_channels=64, run_hil=False)
            run_mock.assert_not_called()
            self.assertEqual((compile_only.error_code, compile_only.flash_bytes), (HIL_ERROR_OK, 10))
            with patch.object(raspberry_pi4, "DEFAULT_MAX_FLASH_BYTES", 5):
                small = raspberry_pi4.RaspberryPi4Device(device_options={"host": "pi.local"})
            self.assertEqual(small.evaluate(dirpath=candidate, arena_kb=-1, window_size=98,
                                            num_channels=64).error_code, HIL_ERROR_FLASH_OVERFLOW)


class ThrottleFlagTests(unittest.TestCase):
    """Tests covering the Pi bench's throttle/temperature flags."""

    def test_throttle_flags(self):
        # Active conditions, newly set sticky bits, unreadable flags and heat are all flagged; old sticky bits are not.
        """Validate throttle flag decoding."""
        bench_dir = ROOT_DIR / "analysis_scripts" / "rpi4_latency_bench"
        if str(bench_dir) not in sys.path:
            sys.path.insert(0, str(bench_dir))
        import run_rpi4_latency_bench as bench

        self.assertEqual(bench.throttle_flags("0x0", "0x0", "55.0"), [])
        self.assertEqual(bench.throttle_flags("0x0", "0x50005", "60.0"),
                         ["under_voltage_now", "under_voltage_during_run", "throttled_now", "throttled_during_run"])
        self.assertEqual(bench.throttle_flags("0x10000", "0x10000", "60.0"), [])
        self.assertEqual(bench.throttle_flags("na", "na", "81.2"), ["throttle_unknown", "hot"])
        env = bench.runner_env(RUNNER_OUTPUT + ["temp_before_c=40.0 temp_after_c=45.0 throttled_before=0x0 throttled=0x0"])
        self.assertEqual((env["runtime"], env["throttled"], env["temp_after_c"]), ("ai_edge_litert@1.4.0", "0x0", "45.0"))


class RunnerScriptTests(unittest.TestCase):
    """Tests covering the on-device runner, executed locally."""

    def test_runner_reports_infeasible_for_missing_model(self):
        # A missing model must yield status=infeasible and no timer output.
        """Validate runner failure output."""
        proc = subprocess.run(
            [sys.executable, "-I", str(raspberry_pi4.RUNNER_PATH), "--model", "missing.tflite", "--runs", "1"],
            capture_output=True, text=True, timeout=300, check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("status=infeasible", proc.stdout)
        self.assertNotIn("timer output:", proc.stdout)


if __name__ == "__main__":
    unittest.main()
