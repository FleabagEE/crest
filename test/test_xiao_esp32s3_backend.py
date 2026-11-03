# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the Seeed XIAO ESP32-S3 Sense latency-only backend."""

import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT_DIR / "analysis_scripts" / "rp2040_latency_bench"
for _path in (ROOT_DIR / "src", BENCH_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from crest.devices import UploadResult, _sync_arduino_sketch_variant_for_config  # noqa: E402
from crest.microcontrollers import get_device, list_device_specs, resolve_device_options, xiao_esp32s3  # noqa: E402

import bench_boards  # noqa: E402
import summarize_rp2040_results  # noqa: E402

SKETCH = ROOT_DIR / "sketches" / "crest_inference_latency.ino"


class ContractTests(unittest.TestCase):
    """Tests covering spec, options and registry wiring."""

    def test_spec_and_board_options(self):
        # Defaults pin hwcdc USB, 240 MHz and PSRAM off; PSRAM placement turns the board's PSRAM on.
        """Validate the spec and pinned board options."""
        spec = xiao_esp32s3.BOARD_DEFAULT_SPEC
        self.assertEqual((spec.name, spec.fqbn), ("XIAO_ESP32S3_SENSE", "esp32:esp32:XIAO_ESP32S3"))
        self.assertEqual(spec.max_ram_bytes, 327_680)
        options = xiao_esp32s3.resolve_esp32s3_options({"runtime_mode": "back_to_back"})
        board = options.board_options()
        self.assertEqual((board["USBMode"], board["CPUFreq"], board["PSRAM"]), ("hwcdc", "240", "disabled"))
        psram = xiao_esp32s3.resolve_esp32s3_options({"arena_memory": "psram", "arena_sizes_kb": [100, 400]})
        self.assertEqual(psram.board_options()["PSRAM"], "opi")
        for bad in ({"arena_memory": "flash"}, {"cpu_mhz": 200}, {"arena_sizes_kb": [400]}, {"arena_sizes_kb": [40, 20]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                xiao_esp32s3.resolve_esp32s3_options(bad)

    def test_registry_defines_and_staging(self):
        # The registry builds the backend; it compiles with its board options and stages the latency sketch.
        """Validate registry, build defines and sketch staging."""
        self.assertEqual(list_device_specs()["XIAO_ESP32S3_SENSE"]["fqbn"], xiao_esp32s3.BOARD_FQBN)
        options = resolve_device_options("xiao_esp32s3_sense", {"esp32s3": {"arena_memory": "psram", "warmup_runs": 3}})
        self.assertEqual(options, {"arena_memory": "psram", "warmup_runs": 3})
        device = get_device("XIAO_ESP32S3_SENSE", device_options=options)
        self.assertIsInstance(device, xiao_esp32s3.XiaoESP32S3Device)
        self.assertFalse(device.supports_energy_measurement())
        self.assertEqual(device.runtime_mode_build_defines(),
                         {"CREST_WARMUP_RUNS": 3, "CREST_ARENA_IN_PSRAM": 1, "CREST_ESP32_CPU_MHZ": 240})
        self.assertEqual(device._board_options["PSRAM"], "opi")
        with tempfile.TemporaryDirectory() as tmp:
            staged = _sync_arduino_sketch_variant_for_config(
                {"device": {"name": "XIAO_ESP32S3_SENSE"}, "training": {"energy_aware": True, "input_mode": "uniform"}},
                Path(tmp) / "cand")
            self.assertEqual(staged.read_text(encoding="utf-8"), SKETCH.read_text(encoding="utf-8"))

    def test_upload_passes_board_options(self):
        # Uploads must use the same --board-options as the compile, and retry on failure.
        """Validate upload()."""
        device = xiao_esp32s3.XiaoESP32S3Device(serial_port="/dev/ttyACM9", device_options={"upload_retries": 1})
        outcomes = [UploadResult(success=False, log="x"), UploadResult(success=True, log="ok")]
        with patch.object(xiao_esp32s3, "run_upload_once", side_effect=outcomes) as upload_mock, patch.object(
            xiao_esp32s3, "wait_for_serial_port", return_value=True
        ), patch.object(xiao_esp32s3.time, "sleep"):
            result = device.upload(sketch_path=Path("cand"), build_dir=Path("build"), serial_port=None)
        self.assertTrue(result.success)
        self.assertEqual(upload_mock.call_count, 2)
        self.assertEqual(upload_mock.call_args.kwargs["board_options"]["USBMode"], "hwcdc")
        self.assertEqual(upload_mock.call_args.kwargs["fqbn"], "esp32:esp32:XIAO_ESP32S3")


class SketchTests(unittest.TestCase):
    """Tests covering the ESP32 branch of the latency sketch."""

    def test_esp32_branch_uses_core_tflm_with_all_needed_ops(self):
        # On ESP32, Chirale must not be included (it would replace ESP-NN) and the resolver must cover every op.
        """Validate the ESP32 sketch branch."""
        text = SKETCH.read_text(encoding="utf-8")
        esp_block = text.split("#if defined(ARDUINO_ARCH_ESP32)", 1)[1].split("#else", 1)[0]
        self.assertNotIn("Chirale_TensorFlowLite.h", esp_block)
        self.assertIn("#include <Chirale_TensorFlowLite.h>", text.split("#if defined(ARDUINO_ARCH_ESP32)", 1)[1].split("#endif", 1)[0])
        ops = set(re.findall(r"resolver\.Add(\w+)\(\)", text))
        self.assertEqual(ops, {"Conv2D", "DepthwiseConv2D", "FullyConnected", "Mean", "Pack", "Reshape", "Shape",
                               "StridedSlice"})
        self.assertIn("MicroMutableOpResolver<8>", text)
        for needle in ("WiFi.mode(WIFI_OFF);", "btStop();", "setCpuFrequencyMhz(CREST_ESP32_CPU_MHZ);",
                       "heap_caps_aligned_alloc", "esp_ptr_external_ram(tensor_arena)"):
            self.assertIn(needle, text)

    def test_parse_env_line(self):
        # The env line carries clock, radio state and arena placement.
        """Validate parse_env_line()."""
        env = xiao_esp32s3.parse_env_line([
            "board=xiao_esp32s3 model=m run=1 invoke_us=1 preprocess_us=-1 arena_used_bytes=1 status=ok",
            "env cpu_mhz=240 wifi_mode=0 bt_started=0 arena_memory=internal_sram arena_addr=0x3FC9B9A0",
        ])
        self.assertEqual((env["cpu_mhz"], env["wifi_mode"], env["arena_memory"]), ("240", "0", "internal_sram"))
        self.assertEqual(xiao_esp32s3.parse_env_line(["runs: 1"]), {})


class BenchToolTests(unittest.TestCase):
    """Tests covering the shared bench helpers."""

    def test_bench_boards_and_summary_board_filter(self):
        # Both boards are benchable, and summaries never mix results from the other board.
        """Validate bench_boards and pick_result()'s board filter."""
        self.assertEqual(set(bench_boards.BOARDS), {"rp2040", "xiao_esp32s3"})
        self.assertIsInstance(bench_boards.make_device("xiao_esp32s3"), xiao_esp32s3.XiaoESP32S3Device)
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, board in enumerate(("nano_rp2040_connect", "xiao_esp32s3")):
                path = Path(tmp) / f"r{i}.txt"
                path.write_text("")
                paths.append({"path": str(path), "model": "m", "board": board, "status": "ok"})
            picked = summarize_rp2040_results.pick_result("m", paths, "nano_rp2040_connect")
            self.assertEqual(picked["board"], "nano_rp2040_connect")
            self.assertEqual(summarize_rp2040_results.pick_result("m", paths, "xiao_esp32s3")["board"], "xiao_esp32s3")


if __name__ == "__main__":
    unittest.main()
