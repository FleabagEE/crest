# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the Arduino Nano RP2040 Connect latency-only backend."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure `src` is importable when tests run via `python -m unittest`.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.devices import (  # noqa: E402
    MeasureResult,
    UploadResult,
    _sync_arduino_sketch_variant_for_config,
)
from crest.microcontrollers import (  # noqa: E402
    arduino_base,
    arduino_nano_rp2040,
    get_device,
    list_device_specs,
    resolve_device_options,
)

SKETCHES_DIR = ROOT_DIR / "sketches"
LATENCY_SKETCH = SKETCHES_DIR / "crest_inference_latency.ino"


def _config(name: str, *, energy_aware: bool = False, input_mode: str = "uniform") -> dict:
    """Build a minimal config mapping for sketch staging.

    Parameters
    ----------
    name : str
        Device name.
    energy_aware : bool, optional
        Value for ``training.energy_aware``.
    input_mode : str, optional
        Value for ``training.input_mode``.

    Returns
    -------
    dict
        Config mapping accepted by ``_sync_arduino_sketch_variant_for_config``.
    """
    return {
        "device": {"name": name},
        "training": {"energy_aware": energy_aware, "input_mode": input_mode},
    }


class RP2040ContractTests(unittest.TestCase):
    """Tests covering the RP2040 board contract and option resolution."""

    def test_contract_symbols_and_spec(self):
        # The RP2040 module should expose the same contract symbols as BLE33.
        """Validate RP2040 contract symbols and default spec."""
        for symbol in (
            "BOARD_NAME",
            "BOARD_FQBN",
            "BOARD_DEFAULT_SPEC",
            "resolve_rp2040_options",
            "build_rp2040_spec",
            "ArduinoNanoRP2040Device",
        ):
            self.assertTrue(hasattr(arduino_nano_rp2040, symbol), f"missing symbol: {symbol}")
        spec = arduino_nano_rp2040.BOARD_DEFAULT_SPEC
        self.assertEqual(spec.name, "ARDUINO_NANO_RP2040_CONNECT")
        self.assertEqual(spec.fqbn, "arduino:mbed_nano:nanorp2040connect")
        self.assertEqual(spec.toolchain, "arduino-cli")
        self.assertEqual(spec.arena_sizes_kb, list(arduino_nano_rp2040.DEFAULT_ARENA_SIZES_KB))
        self.assertLess(spec.arena_sizes_kb[-1] * 1024, spec.max_ram_bytes)

    def test_options_defaults_and_overrides(self):
        # Shared runtime keys are ignored; RP2040 keys override defaults.
        """Validate option defaults and overrides."""
        defaults = arduino_nano_rp2040.resolve_rp2040_options({"runtime_mode": "back_to_back"})
        self.assertEqual(defaults, arduino_nano_rp2040.ArduinoNanoRP2040Options())
        resolved = arduino_nano_rp2040.resolve_rp2040_options(
            {"arena_sizes_kb": [16, 32, 64], "warmup_runs": 0, "upload_retries": 4}
        )
        self.assertEqual(resolved.arena_sizes_kb, (16, 32, 64))
        self.assertEqual(resolved.warmup_runs, 0)
        self.assertEqual(resolved.upload_retries, 4)
        self.assertEqual(arduino_nano_rp2040.build_rp2040_spec(resolved).arena_sizes_kb, [16, 32, 64])

    def test_options_reject_invalid_values(self):
        # Malformed arena sweeps and negative counts must fail before any build.
        """Validate invalid option rejection."""
        for bad in (
            {"arena_sizes_kb": [40, 20]},
            {"arena_sizes_kb": [10, 10]},
            {"arena_sizes_kb": []},
            {"arena_sizes_kb": "40"},
            {"arena_sizes_kb": [300]},
            {"warmup_runs": -1},
            {"upload_timeout_s": 0},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                arduino_nano_rp2040.resolve_rp2040_options(bad)

    def test_registry_and_device_options(self):
        # The registry should construct the RP2040 class and expose its spec.
        """Validate registry wiring and device.rp2040 option plumbing."""
        specs = list_device_specs()
        self.assertEqual(
            specs["ARDUINO_NANO_RP2040_CONNECT"]["fqbn"], arduino_nano_rp2040.BOARD_FQBN
        )
        options = resolve_device_options(
            "arduino_nano_rp2040_connect",
            {"runtime_mode": "back_to_back", "rp2040": {"arena_sizes_kb": [20, 40], "warmup_runs": 3}},
        )
        self.assertEqual(
            options,
            {"arena_sizes_kb": [20, 40], "warmup_runs": 3, "runtime_mode": "back_to_back"},
        )
        device = get_device("ARDUINO_NANO_RP2040_CONNECT", device_options=options, serial_port="/dev/ttyACM9")
        self.assertIsInstance(device, arduino_nano_rp2040.ArduinoNanoRP2040Device)
        self.assertEqual(device.spec.arena_sizes_kb, [20, 40])
        self.assertFalse(device.supports_energy_measurement())
        self.assertEqual(device.runtime_mode_build_defines(), {"CREST_WARMUP_RUNS": 3})
        self.assertIsNone(resolve_device_options("ARDUINO_NANO_RP2040_CONNECT", {}))
        with self.assertRaises(ValueError):
            resolve_device_options("ARDUINO_NANO_RP2040_CONNECT", {"rp2040": {"warmup_runs": -2}})


class RP2040SketchStagingTests(unittest.TestCase):
    """Tests covering RP2040 sketch selection and the latency sketch contract."""

    def test_stages_latency_sketch_even_when_energy_aware(self):
        # RP2040 never stages an energy sketch.
        """Validate RP2040 always stages the latency sketch."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cand"
            staged = _sync_arduino_sketch_variant_for_config(
                _config("ARDUINO_NANO_RP2040_CONNECT", energy_aware=True), out
            )
            self.assertEqual(staged.read_text(encoding="utf-8"), LATENCY_SKETCH.read_text(encoding="utf-8"))
            self.assertTrue((out / "common" / "crest_hil_config.h").exists())

    def test_rejects_cadenced_and_analysis_modes(self):
        # Cadenced and non-uniform input modes are not implemented for RP2040.
        """Validate cadenced and non-uniform staging rejection."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _sync_arduino_sketch_variant_for_config(
                    _config("ARDUINO_NANO_RP2040_CONNECT"), Path(tmp) / "a", runtime_phase="cadenced"
                )
            with self.assertRaises(ValueError):
                _sync_arduino_sketch_variant_for_config(
                    _config("ARDUINO_NANO_RP2040_CONNECT", energy_aware=True, input_mode="oxiod_real"),
                    Path(tmp) / "b",
                )

    def test_ble33_staging_unchanged(self):
        # The RP2040 gate must not alter BLE33 sketch selection.
        """Validate BLE33 staging still selects its usual sketches."""
        with tempfile.TemporaryDirectory() as tmp:
            for energy_aware, expected in ((False, "crest_inference_no_energy.ino"), (True, "crest_inference_energy.ino")):
                staged = _sync_arduino_sketch_variant_for_config(
                    _config("ARDUINO_NANO_33_BLE_SENSE", energy_aware=energy_aware), Path(tmp) / expected[:-4]
                )
                self.assertEqual(
                    staged.read_text(encoding="utf-8"),
                    (SKETCHES_DIR / expected).read_text(encoding="utf-8"),
                )

    def test_latency_sketch_is_patchable_and_harness_free(self):
        # The host patcher must find its defines, and no harness GPIO may be driven.
        """Validate the latency sketch contract."""
        text = LATENCY_SKETCH.read_text(encoding="utf-8")
        self.assertNotIn("CREST_HARNESS_TRIGGER_PIN", text)
        self.assertNotIn("CREST_HARNESS_ARM_PIN", text)
        self.assertNotIn("pinMode", text)
        self.assertIn("arena_used_bytes()", text)
        with tempfile.TemporaryDirectory() as tmp:
            sketch_dir = Path(tmp) / "my-model 1"
            sketch_dir.mkdir()
            ino = sketch_dir / f"{sketch_dir.name}.ino"
            ino.write_text(text, encoding="utf-8")
            arduino_base._patch_sketch_constants(sketch_dir, 64, 201, 64)
            arduino_nano_rp2040._stamp_model_id(sketch_dir)
            patched = ino.read_text(encoding="utf-8")
        self.assertIn("#define CREST_TENSOR_ARENA_BYTES (64 * 1024)", patched)
        self.assertIn('#define CREST_MODEL_ID "my-model_1"', patched)


class RP2040TelemetryTests(unittest.TestCase):
    """Tests covering per-run telemetry parsing and measurement mapping."""

    LOG = [
        "DUT READY",
        "board=nano_rp2040_connect model=m run=1 invoke_us=1200 preprocess_us=-1 arena_used_bytes=30000 status=ok",
        "board=nano_rp2040_connect model=m run=2 invoke_us=1180 preprocess_us=-1 arena_used_bytes=30000 status=ok",
        "runs: 2",
        "timer output: 0.001190",
    ]

    def test_parse_run_lines(self):
        # Only key=value run lines are parsed and integer fields are typed.
        """Validate per-run line parsing."""
        records = arduino_nano_rp2040.parse_run_lines(self.LOG)
        self.assertEqual([record["run"] for record in records], [1, 2])
        self.assertEqual(records[0]["invoke_us"], 1200)
        self.assertEqual(records[0]["preprocess_us"], -1)
        self.assertEqual(records[0]["arena_used_bytes"], 30000)
        self.assertEqual(records[0]["status"], "ok")
        self.assertEqual(arduino_base._parse_latency_from_log(self.LOG), (0.00119, None))

    def test_measure_maps_infeasible_line_to_arena_error(self):
        # Without a TFLM allocation message, the sketch's own infeasible line marks an undersized arena.
        """Validate infeasible-line mapping in measure()."""
        line = (
            "board=nano_rp2040_connect model=m run=0 invoke_us=-1 preprocess_us=-1 "
            "arena_used_bytes=-1 status=infeasible reason=arena_too_small arena_bytes=10240"
        )
        fake = arduino_base.MeasureResult(latency_s=None, arena_error_line=None, serial_log=["DUT READY", line], power_metrics=None)
        device = arduino_nano_rp2040.ArduinoNanoRP2040Device(serial_port="/dev/null")
        with patch.object(arduino_base, "measure_serial", return_value=fake) as measure_mock:
            result = device.measure(
                serial_port=None,
                baud_rate=115200,
                serial_timeout_s=1.0,
                harness_serial_port="/dev/ttyACM1",
                harness_fqbn="ignored",
            )
        self.assertIsInstance(result, MeasureResult)
        self.assertEqual(result.arena_error_line, line)
        self.assertIsNone(measure_mock.call_args.kwargs["harness_serial_port"])


class RP2040UploadTests(unittest.TestCase):
    """Tests covering upload retries and 1200-bps touch behavior."""

    def test_upload_retries_with_touch_then_succeeds(self):
        # A failed first attempt should trigger a touch and a second attempt.
        """Validate upload retry flow."""
        device = arduino_nano_rp2040.ArduinoNanoRP2040Device(
            serial_port="/dev/ttyACM9", device_options={"upload_retries": 2}
        )
        outcomes = [UploadResult(success=False, log="timeout"), UploadResult(success=True, log="ok")]
        with patch.object(arduino_nano_rp2040, "run_upload_once", side_effect=outcomes) as upload_mock, patch.object(
            arduino_nano_rp2040, "touch_1200bps", return_value=True
        ) as touch_mock, patch.object(arduino_nano_rp2040, "wait_for_serial_port", return_value=True), patch.object(
            arduino_nano_rp2040.time, "sleep"
        ):
            result = device.upload(sketch_path=Path("/tmp/x"), build_dir=Path("/tmp/b"), serial_port=None)
        self.assertTrue(result.success)
        self.assertEqual(upload_mock.call_count, 2)
        touch_mock.assert_called_once_with("/dev/ttyACM9")
        self.assertIn("upload attempt 2/3", result.log)

    def test_upload_gives_up_after_retries(self):
        # Every attempt failing should return a failed result with all logs.
        """Validate upload failure after retries."""
        device = arduino_nano_rp2040.ArduinoNanoRP2040Device(
            serial_port="/dev/ttyACM9", device_options={"upload_retries": 1}
        )
        with patch.object(
            arduino_nano_rp2040, "run_upload_once", return_value=UploadResult(success=False, log="hang")
        ) as upload_mock, patch.object(arduino_nano_rp2040, "touch_1200bps", return_value=False), patch.object(
            arduino_nano_rp2040, "wait_for_serial_port", return_value=False
        ), patch.object(arduino_nano_rp2040.time, "sleep"):
            result = device.upload(sketch_path=Path("/tmp/x"), build_dir=Path("/tmp/b"), serial_port=None)
        self.assertFalse(result.success)
        self.assertEqual(upload_mock.call_count, 2)

    def test_compile_stamps_model_id_and_uses_rp2040_fqbn(self):
        # compile() should stamp the model id, then reuse the shared Arduino compile path.
        """Validate compile wiring."""
        with tempfile.TemporaryDirectory() as tmp:
            sketch_dir = Path(tmp) / "cand"
            _sync_arduino_sketch_variant_for_config(_config("ARDUINO_NANO_RP2040_CONNECT"), sketch_dir)
            fake = arduino_base.CompileResult(success=True, log="", flash_bytes=1, ram_bytes=2, overflow_kind=None, build_dir=Path(tmp))
            device = arduino_nano_rp2040.ArduinoNanoRP2040Device()
            with patch.object(arduino_base, "compile_sketch", return_value=fake) as compile_mock:
                device.compile(sketch_path=sketch_dir, arena_kb=40, window_size=201, num_channels=64)
            text = (sketch_dir / "cand.ino").read_text(encoding="utf-8")
        self.assertEqual(compile_mock.call_args.kwargs["fqbn"], "arduino:mbed_nano:nanorp2040connect")
        self.assertIn('#define CREST_MODEL_ID "cand"', text)
        self.assertIn("#define CREST_TENSOR_ARENA_BYTES (40 * 1024)", text)


if __name__ == "__main__":
    unittest.main()
