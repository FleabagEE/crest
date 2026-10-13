# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for hardware conversion and HIL helper utilities."""

import json
import inspect
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import serial
# os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import tensorflow as tf  # type: ignore[attr-defined]
from tcn import TCN
from unittest.mock import Mock, patch

tf.get_logger().setLevel('ERROR')  # Suppresses INFO and WARNING from TF's Python logger

# Ensure `src` is importable when tests run via `python -m unittest`.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.hardware import (  # noqa: E402
    XXD_BIN,
    DEVICE_SPECS,
    HIL_ERROR_COMPILE,
    HIL_ERROR_FLASH_OVERFLOW,
    HIL_ERROR_LATENCY,
    HIL_ERROR_OK,
    HIL_ERROR_RAM_OVERFLOW,
    HIL_ERROR_UNDER_SIZED,
    HIL_ERROR_UPLOAD,
    HIL_controller,
    HIL_spec,
    HIL_MASTER_ARENA_EXHAUSTED,
    HIL_MASTER_DEVICE_NOT_FOUND,
    HIL_MASTER_FLASH_OVERFLOW,
    HIL_MASTER_RAM_OVERFLOW,
    HIL_MASTER_FATAL,
    HIL_MASTER_SUCCESS,
    _pop_retry_hint_bytes,
    _store_retry_hint_bytes,
    _convert_to_cpp_model_python,
    _convert_to_cpp_model_xxd,
    _dequantize_tflite_tensor,
    _ordered_tflite_output_details,
    _quantize_tflite_tensor,
    _representative_calibration_samples,
    _tflite_subprocess_env,
    arena_size_candidates,
    convert_to_cpp_model,
    convert_to_tflite_model,
    get_model_memory_usage,
    predict_tflite_model,
    predict_tflite_model_subprocess,
    return_hardware_specs,
    TFLiteSubprocessError,
)
from crest import hil_protocol  # noqa: E402
from crest.devices import ArduinoDevice, CandidatePrepareRequest  # noqa: E402
import crest.microcontrollers.arduino_base as arduino_base_module  # noqa: E402
from crest.microcontrollers.arduino_base import (  # noqa: E402
    ARDUINO_CLI_BIN,
    ARDUINO_CLI_CONFIG,
    CompileResult as ArduinoCompileResult,
    UploadResult as ArduinoUploadResult,
    MeasureResult as ArduinoMeasureResult,
    _augment_upload_error,
    _classify_compile_failure,
    _collect_latency_seconds,
    _compute_retry_hint_bytes,
    _merge_power_metrics,
    _resolve_build_dir,
    _resolve_platform_txt_path,
    compile_sketch,
    measure_harness_only_open_session,
    measure_serial,
    _parse_memory_from_compile,
    _parse_memory_from_size_recipe,
    _parse_power_metrics,
    _patch_sketch_constants,
    _replace_define,
    _sum_size_regex_matches,
    upload_sketch,
    normalize_power_metrics,
)
from crest.microcontrollers import (  # noqa: E402
    arduino_ble33,
    arduino_portenta_h7,
    get_device,
    list_device_specs,
)
from test.test_support import SKETCH_SOURCE_DIR, TinyModelMixin, _cli_exists  # noqa: E402


class _FakeCompletedProcess:
    """Lightweight stand-in for subprocess.CompletedProcess used in tests."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        """Initialize the instance.

        Parameters
        ----------
        returncode : int
            Subprocess return code exposed by the fake result.
        stdout : str
            Captured stdout exposed by the fake result.
        stderr : str
            Captured stderr exposed by the fake result.
        """
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


COMPILE_SAMPLE_OUTPUT = (
    "Sketch uses 376104 bytes (38%) of program storage space. Maximum is 983040 bytes.\n"
    "Global variables use 98112 bytes (37%) of dynamic memory, leaving 164032 bytes for local variables. Maximum is 262144 bytes."
)
FLASH_OVERFLOW_STDERR = (
    "toolchains/arduino-data/packages/arduino/tools/arm-none-eabi-gcc/"
    "7-2017q4/bin/../lib/gcc/arm-none-eabi/7.2.1/../../../../arm-none-eabi/bin/ld: "
    "build/arduino/odom_tcn.ino.elf section `.text' will not fit in region `FLASH'\n"
    "toolchains/arduino-data/packages/arduino/tools/arm-none-eabi-gcc/"
    "7-2017q4/bin/../lib/gcc/arm-none-eabi/7.2.1/../../../../arm-none-eabi/bin/ld: "
    "region `FLASH' overflowed by 3814108 bytes\n"
    "collect2: error: ld returned 1 exit status\n"
    "Error during build: exit status 1\n"
)
RAM_OVERFLOW_STDERR = (
    "build/arduino/linker_script.ld:138 cannot move location counter backwards "
    "(from 0000000020091d48 to 000000002003fc00)\n"
    "collect2: error: ld returned 1 exit status\n"
    "Error during build: exit status 1\n"
)

class ConversionHelperTests(TinyModelMixin, unittest.TestCase):
    """Validate host-side TFLite conversion and C-array export helpers.

    Notes
    -----
    The tests in this class use :class:`TinyModelMixin` to keep conversion
    coverage fast while still exercising float export, int8 post-training
    quantization, representative calibration validation, and legacy firmware
    source generation paths.
    """

    def test_convert_to_tflite_model_creates_file(self):
        # TFLite conversion should create the expected output file so later deployment steps can consume it directly.
        """Validate convert to tflite model creates file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_float.tflite"
            convert_to_tflite_model(self.model, self.train_x, output_name=tflite_path)
            self.assertTrue(tflite_path.exists())
            self.assertGreater(tflite_path.stat().st_size, 0)

    def test_convert_to_tflite_model_quantized_flow(self):
        # Quantized TFLite conversion should still follow the expected end-to-end artifact path.
        """Validate convert to tflite model quantized flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_int8.tflite"
            convert_to_tflite_model(
                self.model,
                self.train_x,
                quantization_mode="int8_ptq",
                output_name=tflite_path,
            )
            self.assertTrue(tflite_path.exists())
            self.assertGreater(tflite_path.stat().st_size, 0)

    def test_convert_to_tflite_model_float_does_not_require_calibration(self):
        """Float TFLite export should not need representative data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_float.tflite"
            convert_to_tflite_model(
                self.model,
                training_data=None,
                quantization_mode="float",
                output_name=tflite_path,
            )

            self.assertTrue(tflite_path.exists())
            self.assertGreater(tflite_path.stat().st_size, 0)

    def test_convert_to_tflite_model_int8_requires_calibration(self):
        """Int8 PTQ export should reject missing representative data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_int8.tflite"

            with self.assertRaisesRegex(ValueError, "calibration"):
                convert_to_tflite_model(
                    self.model,
                    training_data=None,
                    quantization_mode="int8_ptq",
                    output_name=tflite_path,
                )

    def test_convert_to_tflite_model_int8_rejects_empty_calibration(self) -> None:
        """Validate that empty int8 calibration arrays fail early.

        Returns
        -------
        None
            The test passes when conversion raises a user-facing
            ``ValueError`` before invoking the TFLite converter.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_int8.tflite"

            with self.assertRaisesRegex(ValueError, "at least one"):
                convert_to_tflite_model(
                    self.model,
                    training_data=np.empty((0, 4), dtype=np.float32),
                    quantization_mode="int8_ptq",
                    output_name=tflite_path,
                )

    def test_convert_to_tflite_model_int8_rejects_nonfinite_calibration(self) -> None:
        """Validate that non-finite int8 calibration arrays fail early.

        Returns
        -------
        None
            The test passes when conversion raises a user-facing
            ``ValueError`` before invoking the TFLite converter.
        """
        calibration = np.asarray(self.train_x, dtype=np.float32).copy()
        calibration[0, 0] = np.nan

        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_int8.tflite"

            with self.assertRaisesRegex(ValueError, "finite"):
                convert_to_tflite_model(
                    self.model,
                    training_data=calibration,
                    quantization_mode="int8_ptq",
                    output_name=tflite_path,
                )

    def test_representative_calibration_samples_are_spread_across_split(self) -> None:
        """Validate representative samples are spread across ordered splits.

        Returns
        -------
        None
            The test passes when capped sampling includes the first and last
            windows and does not use a contiguous prefix.
        """
        data = np.arange(1200, dtype=np.float32).reshape(1200, 1)

        selected = _representative_calibration_samples(data)

        self.assertEqual(selected.shape, (1000, 1))
        self.assertEqual(float(selected[0, 0]), 0.0)
        self.assertEqual(float(selected[-1, 0]), 1199.0)
        self.assertGreater(float(selected[500, 0]), 500.0)

    def test_tflite_quant_helpers_use_interpreter_params(self):
        """Tensor helpers should use TFLite scale and zero-point details."""
        detail = {"dtype": np.int8, "quantization": (0.5, -3)}
        values = np.asarray([[0.0, 1.0, -1.0]], dtype=np.float32)

        quantized = _quantize_tflite_tensor(values, detail)
        dequantized = _dequantize_tflite_tensor(quantized, detail)

        np.testing.assert_array_equal(quantized, np.asarray([[-3, -1, -5]], dtype=np.int8))
        np.testing.assert_allclose(dequantized, values, atol=0.5)

    def test_ordered_tflite_output_details_uses_signature_declaration_order(self) -> None:
        """Validate TFLite output details follow signature declaration order.

        Returns
        -------
        None
            The test passes when non-lexical signature output names are returned
            in the flatbuffer-declared order instead of sorted order.
        """
        runner = Mock()
        runner.get_output_details.return_value = {
            "z_vel": {"index": 10},
            "a_vel": {"index": 11},
        }
        interpreter = Mock()
        interpreter.get_output_details.return_value = [{"index": 11}, {"index": 10}]
        interpreter.get_signature_list.return_value = {
            "serving_default": {"outputs": ["z_vel", "a_vel"]}
        }
        interpreter.get_signature_runner.return_value = runner

        ordered = _ordered_tflite_output_details(interpreter)

        self.assertEqual([detail["index"] for detail in ordered], [10, 11])

    def test_convert_to_cpp_model_old_emits_sources(self):
        # Legacy C-array export should emit the expected source files for older embedded workflows.
        """Validate convert to cpp model old emits sources."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_float.tflite"
            convert_to_tflite_model(self.model, self.train_x, output_name=tflite_path)
            out_dir = Path(tmpdir) / "cpp_old"
            source_path, header_path = _convert_to_cpp_model_python(tflite_path, out_dir)
            expected_len = len(tflite_path.read_bytes())
            source_text = source_path.read_text()
            self.assertIn(f"const int g_model_len = {expected_len};", source_text)
            header_text = header_path.read_text()
            self.assertIn("TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_", header_text)

    @unittest.skipUnless(shutil.which("xxd"), "xxd command required for this test.")
    def test_convert_to_cpp_model_via_xxd(self):
        # The xxd-based C-array export should produce the same kind of firmware-ready sources as the Python path.
        """Validate convert to cpp model via xxd."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_float.tflite"
            convert_to_tflite_model(self.model, self.train_x, output_name=tflite_path)
            out_dir = Path(tmpdir) / "cpp_xxd"
            source_path, header_path = convert_to_cpp_model(tflite_path, out_dir)
            self.assertTrue(source_path.exists())
            self.assertTrue(header_path.exists())
            source_text = source_path.read_text()
            self.assertIn('#include "model.h"', source_text)
            self.assertIn("alignas(8) const unsigned char g_model[]", source_text)

    def test_convert_to_cpp_model_missing_source_raises(self):
        # Missing model files should fail before the C-array export path tries to stage any output.
        """Validate convert to cpp model missing source raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bogus_model = Path(tmpdir) / "missing_model.tflite"
            with self.assertRaises(FileNotFoundError):
                convert_to_cpp_model(bogus_model, Path(tmpdir) / "out")

    @unittest.skipUnless(shutil.which("xxd"), "xxd command required for this test.")
    def test_convert_to_cpp_model_handles_corrupt_bytes(self):
        # Even arbitrary bytes should still flow through the raw C-array export path without needing a valid TFLite parse.
        """Validate convert to cpp model handles corrupt bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_corrupt.tflite"
            tflite_path.write_bytes(os.urandom(128))
            out_dir = Path(tmpdir) / "cpp_corrupt"
            source_path, header_path = convert_to_cpp_model(tflite_path, out_dir)
            self.assertTrue(source_path.exists())
            self.assertTrue(header_path.exists())
            self.assertGreater(source_path.stat().st_size, 0)

    @unittest.skipUnless(XXD_BIN, "xxd command required for parity validation.")
    def test_python_and_xxd_emit_matching_sources(self):
        # Python and xxd export paths should stay byte-for-byte aligned so embedded builds do not depend on the conversion route.
        """Validate python and xxd emit matching sources."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_stub.tflite"
            tflite_path.write_bytes(bytes(range(64)))
            python_dir = Path(tmpdir) / "py_cpp"
            xxd_dir = Path(tmpdir) / "xxd_cpp"
            py_source, py_header = _convert_to_cpp_model_python(tflite_path, python_dir)
            xxd_source, xxd_header = _convert_to_cpp_model_xxd(tflite_path, xxd_dir)

            self.assertEqual(py_header.read_text(), xxd_header.read_text())

            py_source_text = py_source.read_text()
            xxd_source_text = xxd_source.read_text()
            hex_pattern = re.compile(r"0x[0-9a-f]{2}")
            self.assertListEqual(hex_pattern.findall(py_source_text), hex_pattern.findall(xxd_source_text))

            len_pattern = re.compile(r"const int g_model_len = (\d+);")
            py_len = len_pattern.search(py_source_text)
            xxd_len = len_pattern.search(xxd_source_text)
            self.assertIsNotNone(py_len)
            self.assertIsNotNone(xxd_len)
            self.assertEqual(py_len.group(1), xxd_len.group(1))

    def test_convert_to_cpp_model_output_dir_conflict(self):
        # C-array export should fail when the output path conflicts with an existing file or directory.
        """Validate convert to cpp model output dir conflict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "not_a_directory"
            output_path.write_text("stub")
            model_path = Path(tmpdir) / "model_float.tflite"
            convert_to_tflite_model(self.model, self.train_x, output_name=model_path)
            with self.assertRaises(FileExistsError):
                convert_to_cpp_model(model_path, output_path)


class TFLiteSubprocessPredictionTests(TinyModelMixin, unittest.TestCase):
    """Validate isolated host-side TFLite prediction behavior.

    Notes
    -----
    The tests in this class cover both direct interpreter prediction and the
    subprocess wrapper used by NAS evaluation. Multi-output coverage asserts
    that TFLite predictions preserve the Keras output contract.
    """

    def test_subprocess_env_disables_visible_gpus(self) -> None:
        """Subprocess environment should hide training GPUs from TensorFlow."""
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1", "PYTHONPATH": "existing"}, clear=False):
            env = _tflite_subprocess_env()

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertTrue(env["PYTHONPATH"].startswith(str(SRC_DIR)))
        self.assertIn("existing", env["PYTHONPATH"])

    def test_subprocess_prediction_matches_direct_single_output(self) -> None:
        """Subprocess prediction should match direct single-output inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_float.tflite"
            convert_to_tflite_model(self.model, self.train_x, output_name=tflite_path)

            direct = predict_tflite_model(tflite_path, self.train_x[:3])
            isolated = predict_tflite_model_subprocess(tflite_path, self.train_x[:3], timeout_sec=60.0)

        self.assertIsInstance(isolated, np.ndarray)
        self.assertEqual(isolated.shape, direct.shape)
        self.assertEqual(isolated.dtype, direct.dtype)
        np.testing.assert_allclose(isolated, direct, rtol=1e-5, atol=1e-5)

    def test_subprocess_prediction_preserves_multi_output_order(self) -> None:
        """Subprocess prediction should preserve ordered multi-output results."""
        inputs = tf.keras.Input(shape=(4,), name="input")
        shared = tf.keras.layers.Dense(3, activation="relu")(inputs)
        output_0 = tf.keras.layers.Dense(1, activation="linear", name="first")(shared)
        output_1 = tf.keras.layers.Dense(2, activation="linear", name="second")(shared)
        model = tf.keras.Model(inputs=inputs, outputs=[output_0, output_1])

        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "multi_output.tflite"
            convert_to_tflite_model(model, self.train_x, output_name=tflite_path)

            direct = predict_tflite_model(tflite_path, self.train_x[:4])
            isolated = predict_tflite_model_subprocess(tflite_path, self.train_x[:4], timeout_sec=60.0)

        self.assertIsInstance(direct, list)
        self.assertIsInstance(isolated, list)
        self.assertEqual(len(isolated), 2)
        for isolated_output, direct_output in zip(isolated, direct):
            self.assertEqual(isolated_output.shape, direct_output.shape)
            np.testing.assert_allclose(isolated_output, direct_output, rtol=1e-5, atol=1e-5)

    def test_prediction_preserves_keras_multi_output_order(self) -> None:
        """Validate direct TFLite prediction preserves Keras output order.

        Returns
        -------
        None
            The test passes when direct host-side TFLite predictions match
            Keras output shapes and values in the original output order.
        """
        inputs = tf.keras.Input(shape=(4,), name="input")
        shared = tf.keras.layers.Dense(3, activation="relu")(inputs)
        output_0 = tf.keras.layers.Dense(1, activation="linear", name="first")(shared)
        output_1 = tf.keras.layers.Dense(2, activation="linear", name="second")(shared)
        model = tf.keras.Model(inputs=inputs, outputs=[output_0, output_1])

        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "multi_output.tflite"
            convert_to_tflite_model(model, self.train_x, output_name=tflite_path)

            keras_outputs = model.predict(self.train_x[:4])
            tflite_outputs = predict_tflite_model(tflite_path, self.train_x[:4])

        self.assertIsInstance(tflite_outputs, list)
        self.assertEqual(len(tflite_outputs), 2)
        for keras_output, tflite_output in zip(keras_outputs, tflite_outputs):
            self.assertEqual(tflite_output.shape, keras_output.shape)
            np.testing.assert_allclose(tflite_output, keras_output, rtol=1e-5, atol=1e-5)

    def test_subprocess_nonzero_return_raises(self) -> None:
        """Nonzero worker exits should raise a structured subprocess error."""
        failed_process = _FakeCompletedProcess(returncode=2, stderr="worker failed")

        with patch("crest.hardware.subprocess.run", return_value=failed_process):
            with self.assertRaises(TFLiteSubprocessError) as raised:
                predict_tflite_model_subprocess("model.tflite", self.train_x[:1], timeout_sec=0.1)

        self.assertFalse(raised.exception.timeout)
        self.assertEqual(raised.exception.return_code, 2)
        self.assertIn("worker failed", raised.exception.stderr_tail)

    def test_subprocess_signal_return_keeps_stderr_tail(self) -> None:
        """Signal-style worker exits should keep useful stderr diagnostics."""
        stderr = "prefix\n" + ("x" * 4100) + "abort details"
        failed_process = _FakeCompletedProcess(returncode=-6, stderr=stderr)

        with patch("crest.hardware.subprocess.run", return_value=failed_process):
            with self.assertRaises(TFLiteSubprocessError) as raised:
                predict_tflite_model_subprocess("model.tflite", self.train_x[:1], timeout_sec=0.1)

        self.assertEqual(raised.exception.return_code, -6)
        self.assertIn("abort details", raised.exception.stderr_tail)
        self.assertLessEqual(len(raised.exception.stderr_tail), 4000)

    def test_subprocess_timeout_raises_timeout_error(self) -> None:
        """Timed-out workers should be reported as timeout subprocess errors."""
        timeout = subprocess.TimeoutExpired(
            cmd=["python", "-m", "crest.tflite_predict_worker"],
            timeout=0.1,
            stderr="hung inside interpreter",
        )

        with patch("crest.hardware.subprocess.run", side_effect=timeout):
            with self.assertRaises(TFLiteSubprocessError) as raised:
                predict_tflite_model_subprocess("model.tflite", self.train_x[:1], timeout_sec=0.1)

        self.assertTrue(raised.exception.timeout)
        self.assertIsNone(raised.exception.return_code)
        self.assertIn("hung inside interpreter", raised.exception.stderr_tail)

    def test_subprocess_missing_output_contract_raises(self) -> None:
        """Zero-exit workers without outputs should raise a subprocess error."""
        completed_process = _FakeCompletedProcess(returncode=0, stderr="no output file")

        with patch("crest.hardware.subprocess.run", return_value=completed_process):
            with self.assertRaises(TFLiteSubprocessError) as raised:
                predict_tflite_model_subprocess("model.tflite", self.train_x[:1], timeout_sec=0.1)

        self.assertFalse(raised.exception.timeout)
        self.assertEqual(raised.exception.return_code, 0)
        self.assertIn("no output file", raised.exception.stderr_tail)


class SpecHelperTests(unittest.TestCase):
    """Tests covering spec helper behavior."""

    def test_return_hardware_specs_known_device(self):
        # Known-device hardware-spec lookups should return the expected memory and timing limits.
        """Validate return hardware specs known device."""
        ram, flash = return_hardware_specs("ARDUINO_NANO_33_BLE_SENSE")
        self.assertGreater(ram, 0)
        self.assertGreater(flash, 0)

    def test_return_hardware_specs_unknown_device(self):
        # Unknown devices should raise immediately instead of pretending to know their hardware limits.
        """Validate return hardware specs unknown device."""
        with self.assertRaises(ValueError):
            return_hardware_specs("NOT_A_BOARD")

    def test_return_hardware_specs_portenta_requires_device_options(self):
        # Portenta hardware-spec resolution should require device options so limits are tied to an explicit core split.
        """Validate return hardware specs Portenta requires device options."""
        with self.assertRaises(ValueError):
            return_hardware_specs("PORTENTA_H7")

    def test_return_hardware_specs_portenta_case_insensitive(self):
        # Portenta hardware-spec lookup should normalize device names so case-only input differences do not change the result.
        """Validate return hardware specs Portenta case insensitive."""
        options = {"target_core": "cm7", "split": "75_25", "security": "none"}
        upper_ram, upper_flash = return_hardware_specs("PORTENTA_H7", device_options=options)
        lower_ram, lower_flash = return_hardware_specs("portenta_h7", device_options=options)
        self.assertEqual((lower_ram, lower_flash), (upper_ram, upper_flash))

    def test_arena_size_candidates_happy_path(self):
        # Arena-size enumeration should return the expected candidate list for supported devices.
        """Validate arena size candidates happy path."""
        arena = arena_size_candidates("ARDUINO_NANO_33_BLE_SENSE")
        self.assertIsInstance(arena, np.ndarray)
        self.assertGreater(len(arena), 0)

    def test_arena_size_candidates_invalid_device(self):
        # Unsupported devices should fail before arena-size exploration can start.
        """Validate arena size candidates invalid device."""
        with self.assertRaises(ValueError):
            arena_size_candidates("UNKNOWN_DEVICE")

    def test_arena_size_candidates_portenta_requires_device_options(self):
        # Portenta arena-size candidates should require device options because the split affects available memory.
        """Validate arena size candidates Portenta requires device options."""
        with self.assertRaises(ValueError):
            arena_size_candidates("PORTENTA_H7")

    def test_arena_size_candidates_portenta_case_insensitive(self):
        # Portenta arena-size lookup should normalize name casing before it inspects device options.
        """Validate arena size candidates Portenta case insensitive."""
        options = {"target_core": "cm7", "split": "75_25", "security": "none"}
        upper = arena_size_candidates("PORTENTA_H7", device_options=options)
        lower = arena_size_candidates("portenta_h7", device_options=options)
        self.assertTrue(np.array_equal(lower, upper))

    def test_get_model_memory_usage_quantized_smaller(self):
        # Quantized models should report a smaller memory footprint than their float counterpart.
        """Validate get model memory usage quantized smaller."""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(4,)),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        float_usage = get_model_memory_usage(1, model, quantized=False)
        quant_usage = get_model_memory_usage(1, model, quantized=True)
        self.assertGreater(float_usage, 0)
        self.assertLessEqual(quant_usage, float_usage)


class DeviceCatalogTests(unittest.TestCase):
    """Tests covering device catalog behavior."""

    def test_catalog_devices_are_accessible(self):
        # The built-in device catalog should expose the expected device entries.
        """Validate catalog devices are accessible."""
        for name, spec in DEVICE_SPECS.items():
            options = (
                {"target_core": "cm7", "split": "75_25", "security": "none"}
                if name == "PORTENTA_H7"
                else None
            )
            ram, flash = return_hardware_specs(name, device_options=options)
            self.assertEqual(ram, spec["max_ram"])
            self.assertEqual(flash, spec["max_flash"])
            arena = arena_size_candidates(name, device_options=options)
            self.assertTrue(np.array_equal(arena, spec["arena_sizes"]))
            self.assertGreater(len(arena), 0)

    def test_catalog_allows_new_device_entries(self):
        # The device catalog should accept new entries without breaking existing lookups.
        """Validate catalog allows new device entries."""
        new_name = "TEST_DEVICE"
        new_spec = {
            "arena_sizes": np.array([5, 15, 25]),
            "max_ram": 123_456,
            "max_flash": 654_321,
            "fqbn": "example:fqbn",
        }
        with patch.dict(DEVICE_SPECS, {new_name: new_spec}, clear=False):
            ram, flash = return_hardware_specs(new_name)
            arena = arena_size_candidates(new_name)
            self.assertEqual(ram, new_spec["max_ram"])
            self.assertEqual(flash, new_spec["max_flash"])
            self.assertTrue(np.array_equal(arena, new_spec["arena_sizes"]))


class MemoryUsageBoundaryTests(unittest.TestCase):
    """Tests covering memory usage boundary behavior."""

    def _build_dense_model(self) -> tf.keras.Model:
        """Build dense model.

        Returns
        -------
        tf.keras.Model
            Constructed dense model.
        """
        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(4,), name="input"),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

    def _build_example_model(self) -> tf.keras.Model:
        """Build example model.

        Returns
        -------
        tf.keras.Model
            Constructed example model.
        """
        timesteps = 200
        input_dim = 6
        inputs = tf.keras.Input(shape=(timesteps, input_dim), name="imu_window")
        features = TCN(
            nb_filters=32,
            kernel_size=5,
            dilations=[1, 2, 4, 8, 16],
            dropout_rate=0.1,
            use_skip_connections=True,
            use_batch_norm=True,
        )(inputs)
        features = tf.keras.layers.Reshape((32, 1))(features)
        features = tf.keras.layers.MaxPooling1D(pool_size=2)(features)
        features = tf.keras.layers.Flatten()(features)
        features = tf.keras.layers.Dense(64, activation="relu", name="pre_dense")(features)
        vel_x = tf.keras.layers.Dense(1, activation="linear", name="velx")(features)
        vel_y = tf.keras.layers.Dense(1, activation="linear", name="vely")(features)
        model = tf.keras.Model(inputs=[inputs], outputs=[vel_x, vel_y])
        model.compile(optimizer="adam", loss={"velx": "mse", "vely": "mse"})
        model.build((None, timesteps, input_dim))
        return model

    def test_memory_usage_matches_manual_estimate(self):
        # Memory estimation should still match the manual layer-by-layer calculation.
        """Validate memory usage matches manual estimate."""
        model = self._build_dense_model()
        batch_size = 1
        usage = get_model_memory_usage(batch_size, model, quantized=False)

        shapes_mem_count = 0
        for layer in model.layers:
            out_shape = getattr(layer, "output_shape", None)
            if out_shape is None:
                continue
            if isinstance(out_shape, list):
                out_shape = out_shape[0]
            elems = 1
            for dim in out_shape:
                if dim is None:
                    continue
                elems *= dim
            shapes_mem_count += elems

        trainable = np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
        non_trainable = np.sum([tf.keras.backend.count_params(p) for p in model.non_trainable_weights])
        expected = 4.0 * (batch_size * shapes_mem_count + trainable + non_trainable)
        self.assertAlmostEqual(usage, expected, places=5)

    def test_memory_usage_respects_float_precision(self):
        # Memory estimation should scale with the model's float precision settings.
        """Validate memory usage respects float precision."""
        original_floatx = tf.keras.backend.floatx()
        try:
            tf.keras.backend.set_floatx("float16")
            model_fp16 = self._build_dense_model()
            usage_fp16 = get_model_memory_usage(1, model_fp16, quantized=False)

            tf.keras.backend.set_floatx("float64")
            model_fp64 = self._build_dense_model()
            usage_fp64 = get_model_memory_usage(1, model_fp64, quantized=False)
        finally:
            tf.keras.backend.set_floatx(original_floatx)

        self.assertLess(usage_fp16, usage_fp64)
        self.assertGreater(usage_fp64, 0)

    def test_memory_usage_outpaces_model_serialization(self):
        # Estimated memory usage should stay larger than the serialized artifact when runtime tensors dominate.
        """Validate memory usage outpaces model serialization."""
        model = self._build_example_model()
        usage = get_model_memory_usage(1, model, quantized=False)
        param_bytes = 4.0 * model.count_params()
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "dense_model.tflite"
            calibration_data = np.random.rand(8, 200, 6).astype(np.float32)
            convert_to_tflite_model(model, calibration_data, output_name=tflite_path)
            flatbuffer_bytes = tflite_path.stat().st_size

        self.assertGreaterEqual(usage, param_bytes)
        self.assertGreater(usage, flatbuffer_bytes * 0.25)


class SketchHelperTests(unittest.TestCase):
    """Tests covering sketch helper behavior."""

    def test_replace_define_updates_value(self):
        # Define replacement should update the requested constant without disturbing the rest of the file.
        """Validate replace define updates value."""
        text = "#define CREST_WINDOW_SIZE 100\nvoid loop() {}\n"
        updated = _replace_define(text, "CREST_WINDOW_SIZE", "256")
        self.assertIn("CREST_WINDOW_SIZE 256", updated)

    def test_patch_sketch_constants_edits_ino(self):
        # Sketch constant patching should rewrite the `.ino` file in place with the requested values.
        """Validate patch sketch constants edits ino."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir)
            ino_path = sketch_dir / "crest.ino"
            ino_path.write_text(
                "\n".join(
                    [
                        "#define CREST_WINDOW_SIZE 100",
                        "#define CREST_NUM_CHANNELS 1",
                        "#define CREST_TENSOR_ARENA_BYTES (10 * 1024)",
                    ]
                )
            )
            _patch_sketch_constants(sketch_dir, arena_kb=42, window_size=256, num_channels=3)
            text = ino_path.read_text()
            self.assertIn("CREST_WINDOW_SIZE 256", text)
            self.assertIn("CREST_NUM_CHANNELS 3", text)
            self.assertIn("CREST_TENSOR_ARENA_BYTES (42 * 1024)", text)

    def test_patch_sketch_constants_accepts_audio_feature_shape(self):
        """Audio feature tensors should map to Arduino window/channel macros."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir)
            ino_path = sketch_dir / "audio_dscnn.ino"
            ino_path.write_text(
                "\n".join(
                    [
                        "#define CREST_WINDOW_SIZE 100",
                        "#define CREST_NUM_CHANNELS 1",
                        "#define CREST_TENSOR_ARENA_BYTES (10 * 1024)",
                    ]
                )
            )
            _patch_sketch_constants(sketch_dir, arena_kb=96, window_size=201, num_channels=64)
            text = ino_path.read_text()
            self.assertIn("CREST_WINDOW_SIZE 201", text)
            self.assertIn("CREST_NUM_CHANNELS 64", text)

    def test_patch_sketch_constants_updates_latency_budget_when_present(self):
        # Sketch patching should update an existing latency-budget constant so generated test firmware matches the requested timing window.
        """Validate patch sketch constants updates latency budget when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir)
            ino_path = sketch_dir / "crest.ino"
            ino_path.write_text(
                "\n".join(
                    [
                        "#define CREST_WINDOW_SIZE 100",
                        "#define CREST_NUM_CHANNELS 1",
                        "#define CREST_TENSOR_ARENA_BYTES (10 * 1024)",
                        "#define CREST_LATENCY_BUDGET_MS 200",
                    ]
                )
            )
            _patch_sketch_constants(
                sketch_dir,
                arena_kb=42,
                window_size=256,
                num_channels=3,
                latency_budget_ms=75.8,
            )
            text = ino_path.read_text()
            self.assertIn("CREST_LATENCY_BUDGET_MS 76", text)

    def test_parse_memory_from_compile_extracts_numbers(self):
        # Compile-output parsing should recover flash and RAM counts from the standard Arduino summary.
        """Validate parse memory from compile extracts numbers."""
        sample_output = (
            "Sketch uses 376104 bytes (38%) of program storage space. Maximum is 983040 bytes. \n \
            Global variables use 98112 bytes (37%) of dynamic memory, leaving 164032 bytes for local variables. Maximum is 262144 bytes."
        )
        flash_bytes, ram_bytes = _parse_memory_from_compile(sample_output)
        self.assertEqual(flash_bytes, 376104)
        self.assertEqual(ram_bytes, 98112)

    def test_parse_memory_from_compile_handles_missing_data(self):
        # Compile-output parsing should return unknown memory figures when the build log omits them.
        """Validate parse memory from compile handles missing data."""
        flash_bytes, ram_bytes = _parse_memory_from_compile("no relevant information")
        self.assertIsNone(flash_bytes)
        self.assertIsNone(ram_bytes)


class CompileFailureClassificationTests(unittest.TestCase):
    """Tests covering compile failure classification behavior."""

    def test_classify_returns_none_for_normal_output(self):
        # Normal compiler output should not classify as an overflow so successful builds stay on the happy path.
        """Validate classify returns none for normal output."""
        result = _classify_compile_failure(COMPILE_SAMPLE_OUTPUT)
        self.assertIsNone(result)

    def test_classify_detects_flash_overflow(self):
        # Linker flash-overflow text must classify as flash so NAS pruning can stop obviously too-large candidates.
        """Validate classify detects flash overflow."""
        result = _classify_compile_failure(FLASH_OVERFLOW_STDERR)
        self.assertEqual(result, "flash")

    def test_classify_detects_ram_overflow(self):
        # RAM-overflow diagnostics must classify as RAM so pruning does not misreport a working memory failure as flash.
        """Validate classify detects ram overflow."""
        result = _classify_compile_failure(RAM_OVERFLOW_STDERR)
        self.assertEqual(result, "ram")


class PortentaOptionValidationTests(unittest.TestCase):
    """Tests covering Portenta option validation behavior."""

    def test_missing_target_core_raises(self):
        # Portenta split resolution should fail when the target core is missing instead of guessing a board half.
        """Validate missing target core raises."""
        with self.assertRaises(ValueError):
            arduino_portenta_h7.resolve_portenta_h7_options({})

    def test_invalid_split_raises(self):
        # Unsupported Portenta split values should fail before compilation starts.
        """Validate invalid split raises."""
        with self.assertRaises(ValueError):
            arduino_portenta_h7.resolve_portenta_h7_options({"target_core": "cm7", "split": "bad_split"})

    def test_cm4_100_0_rejected(self):
        # CM4 cannot own the entire split, so the option parser should reject the unsupported 100/0 layout early.
        """Validate CM4 100 0 rejected."""
        with self.assertRaises(ValueError):
            arduino_portenta_h7.resolve_portenta_h7_options({"target_core": "cm4", "split": "100_0"})


class ArduinoBoardContractShapeTests(unittest.TestCase):
    """Tests covering Arduino board contract shape behavior."""

    def _assert_spec_complete(self, spec, *, expected_name: str, expected_fqbn: str):
        """Exercise assert spec complete behavior.

        Parameters
        ----------
        spec : object
            Device spec checked by the assertion helper.
        expected_name : str
            Expected device name asserted by the helper.
        expected_fqbn : str
            Expected Arduino FQBN asserted by the helper.
        """
        self.assertEqual(spec.name, expected_name)
        self.assertEqual(spec.fqbn, expected_fqbn)
        self.assertEqual(spec.toolchain, "arduino-cli")
        self.assertIsInstance(spec.arena_sizes_kb, list)
        self.assertGreater(len(spec.arena_sizes_kb), 0)
        self.assertGreater(spec.max_ram_bytes, 0)
        self.assertGreater(spec.max_flash_bytes, 0)

    def test_ble33_contract_symbols_and_spec(self):
        # The BLE33 contract should expose the expected symbol set and hardware spec so registry lookups remain stable.
        """Validate BLE33 contract symbols and spec."""
        required_symbols = (
            "BOARD_NAME",
            "BOARD_FQBN",
            "BOARD_DEFAULT_SPEC",
            "resolve_ble33_options",
            "build_ble33_spec",
            "ArduinoBLE33Device",
        )
        for symbol in required_symbols:
            self.assertTrue(hasattr(arduino_ble33, symbol), f"missing symbol: {symbol}")
        resolved_options = arduino_ble33.resolve_ble33_options({"ignored": "value"})
        self.assertIsNone(resolved_options)
        built_spec = arduino_ble33.build_ble33_spec(resolved_options)
        self._assert_spec_complete(
            built_spec,
            expected_name=arduino_ble33.BOARD_NAME,
            expected_fqbn=arduino_ble33.BOARD_FQBN,
        )
        self.assertEqual(built_spec, arduino_ble33.BOARD_DEFAULT_SPEC)

    def test_portenta_contract_symbols_and_spec(self):
        # The Portenta contract should expose the expected symbol set and board-specific limits.
        """Validate Portenta contract symbols and spec."""
        required_symbols = (
            "BOARD_NAME",
            "BOARD_FQBN",
            "BOARD_DEFAULT_SPEC",
            "PortentaH7BoardOptions",
            "resolve_portenta_h7_options",
            "build_portenta_h7_spec",
            "ArduinoPortentaH7Device",
        )
        for symbol in required_symbols:
            self.assertTrue(hasattr(arduino_portenta_h7, symbol), f"missing symbol: {symbol}")
        resolved_options = arduino_portenta_h7.resolve_portenta_h7_options({"target_core": "cm7"})
        built_spec = arduino_portenta_h7.build_portenta_h7_spec(options=resolved_options)
        self._assert_spec_complete(
            built_spec,
            expected_name=arduino_portenta_h7.BOARD_NAME,
            expected_fqbn=arduino_portenta_h7.BOARD_FQBN,
        )
        self._assert_spec_complete(
            arduino_portenta_h7.BOARD_DEFAULT_SPEC,
            expected_name=arduino_portenta_h7.BOARD_NAME,
            expected_fqbn=arduino_portenta_h7.BOARD_FQBN,
        )


class ArduinoRegistryContractTests(unittest.TestCase):
    """Tests covering Arduino registry contract behavior."""

    def test_list_device_specs_includes_ble_and_portenta(self):
        # Listing device specs should include both BLE33 and Portenta entries so the public catalog stays complete.
        """Validate list device specs includes ble and Portenta."""
        specs = list_device_specs()
        self.assertIn("ARDUINO_NANO_33_BLE_SENSE", specs)
        self.assertIn("PORTENTA_H7", specs)
        self.assertEqual(
            specs["ARDUINO_NANO_33_BLE_SENSE"]["fqbn"],
            arduino_ble33.BOARD_FQBN,
        )
        self.assertEqual(
            specs["PORTENTA_H7"]["fqbn"],
            arduino_portenta_h7.BOARD_FQBN,
        )

    def test_get_device_constructs_portenta_with_options(self):
        # Device construction should pass Portenta board options through to the concrete device instance.
        """Validate get device constructs Portenta with options."""
        device = get_device(
            "PORTENTA_H7",
            device_options={"target_core": "cm7", "split": "75_25", "security": "none"},
        )
        self.assertIsInstance(device, arduino_portenta_h7.ArduinoPortentaH7Device)
        self.assertEqual(device.resolved_options.target_core, "cm7")
        self.assertEqual(device.resolved_options.split, "75_25")
        self.assertEqual(device.resolved_options.security, "none")

    def test_get_device_normalizes_registry_name_case_and_whitespace(self):
        # Device lookup should normalize registry-name casing and whitespace before resolving the catalog entry.
        """Validate get device normalizes registry name case and whitespace."""
        device = get_device(
            "  portenta_h7  ",
            device_options={"target_core": "cm7", "split": "75_25", "security": "none"},
        )
        self.assertIsInstance(device, arduino_portenta_h7.ArduinoPortentaH7Device)

    def test_get_device_non_arduino_legacy_entry_raises_actionable_error(self):
        # Legacy non-Arduino device aliases should raise an actionable error instead of silently misrouting the request.
        """Validate get device non Arduino legacy entry raises actionable error."""
        with self.assertRaises(ValueError) as context:
            get_device("ARCH_MAX")
        message = str(context.exception)
        self.assertIn("no registered backend", message)
        self.assertIn("DeviceInterface", message)

    def test_get_device_normalizes_legacy_name_in_actionable_error(self):
        # Legacy-name errors should show the normalized name so callers can see exactly what lookup failed.
        """Validate get device normalizes legacy name in actionable error."""
        with self.assertRaises(ValueError) as context:
            get_device("  arch_max  ")
        message = str(context.exception)
        self.assertIn("Device 'ARCH_MAX'", message)
        self.assertIn("no registered backend", message)

    def test_portenta_runtime_mode_resolution(self):
        # Portenta runtime-mode resolution should preserve the CM4/CM7 split semantics the backend depends on.
        """Validate Portenta runtime mode resolution."""
        cm7 = get_device(
            "PORTENTA_H7",
            device_options={"target_core": "cm7", "split": "75_25", "security": "none"},
        )
        cm4 = get_device(
            "PORTENTA_H7",
            device_options={"target_core": "cm4", "split": "50_50", "security": "none"},
        )

        self.assertEqual(cm7.runtime_measure_mode(), "direct_serial")
        self.assertEqual(cm7.runtime_mode_build_defines(), {})
        self.assertEqual(cm4.runtime_measure_mode(), "harness_only")
        self.assertEqual(cm4.runtime_mode_build_defines()["CREST_AUTOSTART"], 1)
        self.assertEqual(cm4.runtime_mode_build_defines()["CREST_SKIP_SERIAL_WAIT"], 1)

    def test_portenta_cm4_prepare_for_runtime_bootstraps_cm7_helper(self):
        # CM4 runtime preparation should bootstrap the CM7 helper image before the CM4 run can start.
        """Validate Portenta CM4 prepare for runtime bootstraps CM7 helper."""
        cm4 = get_device(
            "PORTENTA_H7",
            device_options={"target_core": "cm4", "split": "50_50", "security": "none"},
        )
        with patch.object(cm4, "_ensure_cm7_boot_helper") as helper_mock:
            cm4.prepare_for_runtime(runtime_mode="harness_only", serial_port="/dev/ttyACM0")
        helper_mock.assert_called_once_with(serial_port="/dev/ttyACM0")

    def test_portenta_cm7_prepare_for_runtime_skips_bootstrap(self):
        # CM7 runtime preparation should skip the helper bootstrap path that only CM4 depends on.
        """Validate Portenta CM7 prepare for runtime skips bootstrap."""
        cm7 = get_device(
            "PORTENTA_H7",
            device_options={"target_core": "cm7", "split": "75_25", "security": "none"},
        )
        with patch.object(cm7, "_ensure_cm7_boot_helper") as helper_mock:
            cm7.prepare_for_runtime(runtime_mode="direct_serial", serial_port="/dev/ttyACM0")
        helper_mock.assert_not_called()


class ArduinoCommandOptionTests(unittest.TestCase):
    """Tests covering Arduino command option behavior."""

    def test_resolve_build_dir_hash_includes_board_options(self):
        # Build-directory hashing should include board options so distinct Portenta layouts do not reuse one cache entry.
        """Validate resolve build dir hash includes board options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir) / "odom_tcn"
            sketch_dir.mkdir()
            cm7_build_dir = _resolve_build_dir(
                sketch_dir,
                "arduino:mbed_portenta:envie_m7",
                {"CREST_AUTOSTART": 1},
                board_options={"target_core": "cm7", "split": "75_25", "security": "none"},
            )
            cm4_build_dir = _resolve_build_dir(
                sketch_dir,
                "arduino:mbed_portenta:envie_m7",
                {"CREST_AUTOSTART": 1},
                board_options={"target_core": "cm4", "split": "50_50", "security": "none"},
            )
        self.assertNotEqual(cm7_build_dir, cm4_build_dir)

    def test_compile_sketch_includes_board_options(self):
        """Sketch compilation should log CLI path and forward board options.

        Returns
        -------
        None
            The test passes when the compile command includes board options and
            the Arduino CLI path diagnostic is emitted through logging.
        """
        # Sketch compilation should forward board options so generated binaries match the selected board layout.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir) / "odom_tcn"
            sketch_dir.mkdir()
            compile_result = _FakeCompletedProcess(
                returncode=0,
                stdout=COMPILE_SAMPLE_OUTPUT,
                stderr="",
            )
            with patch(
                "crest.microcontrollers.arduino_base.subprocess.run",
                return_value=compile_result,
            ) as mock_run:
                previous_logged = arduino_base_module._ARDUINO_CLI_PATH_LOGGED
                arduino_base_module._ARDUINO_CLI_PATH_LOGGED = False
                try:
                    with self.assertLogs("crest.microcontrollers.arduino_base", level="DEBUG") as captured:
                        result = compile_sketch(
                            sketch_path=sketch_dir,
                            fqbn="arduino:mbed_portenta:envie_m7",
                            board_options={
                                "target_core": "cm7",
                                "split": "75_25",
                                "security": "none",
                            },
                        )
                finally:
                    arduino_base_module._ARDUINO_CLI_PATH_LOGGED = previous_logged

        self.assertTrue(result.success)
        command = mock_run.call_args.args[0]
        self.assertIn("Using Arduino CLI at:", "\n".join(captured.output))
        self.assertIn("--board-options", command)
        board_index = command.index("--board-options")
        self.assertEqual(
            command[board_index + 1],
            "security=none,split=75_25,target_core=cm7",
        )

    def test_upload_sketch_includes_board_options(self):
        # Sketch upload should forward board options so the right core and split receive the firmware.
        """Validate upload sketch includes board options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir) / "odom_tcn"
            sketch_dir.mkdir()
            build_dir = sketch_dir / ".arduino-build" / "arduino_mbed_portenta_envie_m7"
            build_dir.mkdir(parents=True, exist_ok=True)
            upload_result = _FakeCompletedProcess(returncode=0, stdout="ok", stderr="")
            with patch(
                "crest.microcontrollers.arduino_base.subprocess.run",
                return_value=upload_result,
            ) as mock_run:
                result = upload_sketch(
                    sketch_path=sketch_dir,
                    fqbn="arduino:mbed_portenta:envie_m7",
                    build_dir=build_dir,
                    serial_port="/dev/ttyACM0",
                    board_options={"target_core": "cm4", "split": "50_50", "security": "none"},
                )

        self.assertTrue(result.success)
        command = mock_run.call_args.args[0]
        self.assertIn("--board-options", command)
        board_index = command.index("--board-options")
        self.assertEqual(
            command[board_index + 1],
            "security=none,split=50_50,target_core=cm4",
        )

    def test_compile_sketch_leaves_memory_unknown_without_summary_or_recipe(self):
        # When Arduino build output lacks a size summary, memory accounting should stay unknown instead of inventing values.
        """Validate compile sketch leaves memory unknown without summary or recipe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir) / "odom_tcn"
            sketch_dir.mkdir()
            compile_result = _FakeCompletedProcess(returncode=0, stdout="compile ok", stderr="")
            with patch(
                "crest.microcontrollers.arduino_base.subprocess.run",
                return_value=compile_result,
            ), patch(
                "crest.microcontrollers.arduino_base._parse_memory_from_compile",
                return_value=(None, None),
            ), patch(
                "crest.microcontrollers.arduino_base._parse_memory_from_size_recipe",
                return_value=(None, None),
            ):
                result = compile_sketch(
                    sketch_path=sketch_dir,
                    fqbn="arduino:mbed_portenta:envie_m7",
                )

        self.assertIsNone(result.flash_bytes)
        self.assertIsNone(result.ram_bytes)

    def test_parse_memory_from_size_recipe_uses_platform_regexes(self):
        # Size-recipe parsing should use each platform's regex contract so memory extraction stays portable.
        """Validate parse memory from size recipe uses platform regexes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir) / "build"
            build_dir.mkdir(parents=True)
            platform_dir = Path(tmpdir) / "platform"
            platform_dir.mkdir(parents=True)
            (platform_dir / "platform.txt").write_text(
                "\n".join(
                    [
                        'recipe.size.regex=^(?:\\.data|\\.text|\\.rodata)\\S*?\\s+([0-9]+).*',
                        "recipe.size.regex.data=^(?:\\.data|\\.bss)\\s+([0-9]+).*",
                    ]
                )
            )
            (build_dir / "build.options.json").write_text(
                (
                    '{'
                    f'"hardwareFolders":"{platform_dir},{platform_dir}",'
                    '"fqbn":"arduino:mbed_portenta:envie_m7"'
                    "}"
                )
            )
            elf_path = build_dir / "sketch.ino.elf"
            elf_path.write_bytes(b"ELF")
            size_binary = Path(tmpdir) / "toolchain" / "bin" / "arm-none-eabi-size"
            size_stdout = (
                "section              size        addr\n"
                ".text              149680   134479872\n"
                ".data                6184   603979776\n"
                ".bss                57120   603985984\n"
                ".heap              459936   604043104\n"
                ".lwip_sec          278528   805306368\n"
            )
            with patch(
                "crest.microcontrollers.arduino_base._resolve_arm_size_binary",
                return_value=str(size_binary),
            ), patch(
                "crest.microcontrollers.arduino_base._find_compiled_elf",
                return_value=elf_path,
            ), patch(
                "crest.microcontrollers.arduino_base.subprocess.run",
                return_value=_FakeCompletedProcess(returncode=0, stdout=size_stdout, stderr=""),
            ):
                flash_bytes, ram_bytes = _parse_memory_from_size_recipe(build_dir)

        self.assertEqual(flash_bytes, 155_864)
        self.assertEqual(ram_bytes, 63_304)

    def test_resolve_platform_txt_path_supports_hardware_folders_list(self):
        # Platform.txt resolution should search hardware-folders lists in the same order Arduino tooling expects.
        """Validate resolve platform txt path supports hardware folders list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir) / "build"
            build_dir.mkdir(parents=True)
            platform_dir = Path(tmpdir) / "platform"
            platform_dir.mkdir(parents=True)
            expected = platform_dir / "platform.txt"
            expected.write_text("recipe.size.regex=^.*$")
            (build_dir / "build.options.json").write_text(
                json.dumps(
                    {
                        "hardwareFolders": [str(platform_dir)],
                        "fqbn": "arduino:mbed_portenta:envie_m7",
                    }
                )
            )

            resolved = _resolve_platform_txt_path(build_dir)

        self.assertEqual(resolved, expected)

    def test_resolve_platform_txt_path_skips_empty_entries(self):
        # Platform.txt resolution should ignore empty folder entries instead of treating them like valid paths.
        """Validate resolve platform txt path skips empty entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            build_dir = root / "build"
            build_dir.mkdir(parents=True)
            # If empty entries are not filtered, Path("") probes CWD/platform.txt.
            (root / "platform.txt").write_text("recipe.size.regex=^bad$")
            platform_dir = root / "platform"
            platform_dir.mkdir(parents=True)
            expected = platform_dir / "platform.txt"
            expected.write_text("recipe.size.regex=^good$")
            (build_dir / "build.options.json").write_text(
                json.dumps(
                    {
                        "hardwareFolders": ["", "   ", str(platform_dir)],
                        "fqbn": "arduino:mbed_portenta:envie_m7",
                    }
                )
            )
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                resolved = _resolve_platform_txt_path(build_dir)
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(resolved, expected)

    def test_compile_sketch_uses_size_recipe_when_summary_missing(self):
        # Size-recipe parsing should backfill memory accounting when the standard summary block is absent.
        """Validate compile sketch uses size recipe when summary missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir) / "odom_tcn"
            sketch_dir.mkdir()
            compile_result = _FakeCompletedProcess(returncode=0, stdout="compile ok", stderr="")
            with patch(
                "crest.microcontrollers.arduino_base.subprocess.run",
                return_value=compile_result,
            ), patch(
                "crest.microcontrollers.arduino_base._parse_memory_from_compile",
                return_value=(None, None),
            ), patch(
                "crest.microcontrollers.arduino_base._parse_memory_from_size_recipe",
                return_value=(155_864, 63_304),
            ):
                result = compile_sketch(
                    sketch_path=sketch_dir,
                    fqbn="arduino:mbed_portenta:envie_m7",
                )

        self.assertEqual(result.flash_bytes, 155_864)
        self.assertEqual(result.ram_bytes, 63_304)

    def test_sum_size_regex_matches_without_capture_group_returns_none(self):
        # Malformed size-regex matches should return None instead of claiming a bogus memory total.
        """Validate sum size regex matches without capture group returns none."""
        output = ".text 149680 134479872\n.data 6184 603979776\n"
        total = _sum_size_regex_matches(output, r"^(?:\\.text|\\.data)\\s+\\d+.*")
        self.assertIsNone(total)

    def test_upload_permission_error_appends_linux_guidance(self):
        # Upload permission errors should append the Linux guidance so the failure is actionable from the first message.
        """Validate upload permission error appends linux guidance."""
        augmented = _augment_upload_error("dfu-util: LIBUSB_ERROR_ACCESS")
        self.assertIn("udev", augmented)
        self.assertIn("Linux", augmented)


class SerialHelperTests(unittest.TestCase):
    """Tests covering serial helper behavior."""

    class _DummySerial:
        """Dummy serial stream used by serial helper tests."""

        def __init__(self, responses):
            """Initialize the instance.

            Parameters
            ----------
            responses : object
                Byte responses yielded by the dummy serial stream.
            """
            self._responses = iter(responses)

        def readline(self):
            """Read line.

            Returns
            -------
            object
                Next byte response from the dummy serial stream.
            """
            try:
                return next(self._responses)
            except StopIteration:
                return b""

        def __enter__(self):
            """Enter the context manager.

            Returns
            -------
            object
                Active context manager instance.
            """
            return self

        def __exit__(self, exc_type, exc, tb):
            """Exit the context manager.

            Parameters
            ----------
            exc_type : object
                Exception type passed to the context manager.
            exc : object
                Exception instance passed to the context manager.
            tb : object
                Traceback object passed to the context manager.

            Returns
            -------
            object
                Whether the exception was handled by the context manager.
            """
            return False

    def test_collect_latency_returns_value(self):
        # Latency collection should parse and return a numeric latency sample from the serial stream.
        """Validate collect latency returns value."""
        responses = [b"ignored\n", b"timer output: 0.42\n"]

        def factory(*_args, **_kwargs):
            """Create a dummy serial connection.

            Parameters
            ----------
            *_args : tuple[object, ...]
                Positional arguments captured by the test double.
            **_kwargs : dict[str, object]
                Keyword arguments captured by the test double.

            Returns
            -------
            object
                Dummy serial instance returned by the patched factory.
            """
            return self._DummySerial(responses)

        with patch("crest.microcontrollers.arduino_base.serial.Serial", side_effect=factory):
            latency, arena_line, serial_log = _collect_latency_seconds(
                "COM1", 115200, timeout_s=0.05
            )
        self.assertIsNotNone(latency)
        assert latency is not None
        self.assertAlmostEqual(latency, 0.42)
        self.assertIsNone(arena_line)
        self.assertEqual(serial_log, ["ignored", "timer output: 0.42"])

    def test_collect_latency_handles_timeout(self):
        # Latency collection should return the timeout sentinel when the serial stream never reports a sample.
        """Validate collect latency handles timeout."""
        responses = [b"", b""]

        def factory(*_args, **_kwargs):
            """Create a dummy serial connection.

            Parameters
            ----------
            *_args : tuple[object, ...]
                Positional arguments captured by the test double.
            **_kwargs : dict[str, object]
                Keyword arguments captured by the test double.

            Returns
            -------
            object
                Dummy serial instance returned by the patched factory.
            """
            return self._DummySerial(responses)

        with patch("crest.microcontrollers.arduino_base.serial.Serial", side_effect=factory):
            latency, arena_line, serial_log = _collect_latency_seconds(
                "COM2", 115200, timeout_s=0.01
            )
        self.assertIsNone(latency)
        self.assertIsNone(arena_line)
        self.assertEqual(serial_log, [])

    def test_collect_latency_invalid_port_raises(self):
        # Serial-port failures should surface immediately instead of being misreported as a valid latency timeout.
        """Validate collect latency invalid port raises."""
        with patch(
            "crest.microcontrollers.arduino_base.serial.Serial",
            side_effect=serial.SerialException("boom"),
        ):
            with self.assertRaises(RuntimeError):
                _collect_latency_seconds("COM3", 115200, timeout_s=0.01)

    def test_collect_latency_handles_non_numeric_payload(self):
        # Latency collection should reject non-numeric serial payloads instead of treating them like valid timings.
        """Validate collect latency handles non-numeric payload."""
        responses = [b"timer output: not-a-float\n"]

        def factory(*_args, **_kwargs):
            """Create a dummy serial connection.

            Parameters
            ----------
            *_args : tuple[object, ...]
                Positional arguments captured by the test double.
            **_kwargs : dict[str, object]
                Keyword arguments captured by the test double.

            Returns
            -------
            object
                Dummy serial instance returned by the patched factory.
            """
            return self._DummySerial(responses)

        with patch("crest.microcontrollers.arduino_base.serial.Serial", side_effect=factory):
            latency, arena_line, serial_log = _collect_latency_seconds(
                "COM4", 115200, timeout_s=0.01
            )
        self.assertIsNone(latency)
        self.assertIsNone(arena_line)
        self.assertEqual(serial_log, ["timer output: not-a-float"])

    def test_collect_latency_detects_arena_error(self):
        # Detect collect latency detects arena error so error classification and pruning stay stable.
        """Validate collect latency detects arena error."""
        responses = [b"size is too small for all buffers\n"]

        def factory(*_args, **_kwargs):
            """Create a dummy serial connection.

            Parameters
            ----------
            *_args : tuple[object, ...]
                Positional arguments captured by the test double.
            **_kwargs : dict[str, object]
                Keyword arguments captured by the test double.

            Returns
            -------
            object
                Dummy serial instance returned by the patched factory.
            """
            return self._DummySerial(responses)

        with patch("crest.microcontrollers.arduino_base.serial.Serial", side_effect=factory):
            latency, arena_line, serial_log = _collect_latency_seconds(
                "COM4", 115200, timeout_s=0.01
            )
        self.assertIsNone(latency)
        self.assertIsNotNone(arena_line)
        self.assertEqual(serial_log, ["size is too small for all buffers"])


class RetryHintHelperTests(unittest.TestCase):
    """Tests covering retry hint helper behavior."""

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        _store_retry_hint_bytes(None)

    def test_compute_retry_hint_uses_missing_field(self):
        # Retry-hint logic should fall back to the missing-field value when the measurement did not report a new size.
        """Validate compute retry hint uses missing field."""
        current_bytes = 100_000
        line = "Failed... missing: 4096"
        hint = _compute_retry_hint_bytes(current_bytes, line)
        self.assertEqual(hint, current_bytes + 4096 + 2048)

    def test_compute_retry_hint_uses_requested_field(self):
        # Retry-hint logic should use the requested size field when the backend reported one.
        """Validate compute retry hint uses requested field."""
        current_bytes = 50_000
        line = "Requested: 60000, available: 123"
        hint = _compute_retry_hint_bytes(current_bytes, line)
        self.assertEqual(hint, 60_000 + 2048)

    def test_compute_retry_hint_returns_none_without_growth(self):
        # Retry-hint logic should return None when it cannot justify a larger retry window.
        """Validate compute retry hint returns none without growth."""
        current_bytes = 40_000
        line = "Requested: 1000, missing: 0"
        hint = _compute_retry_hint_bytes(current_bytes, line)
        self.assertIsNone(hint)
        self.assertIsNone(_compute_retry_hint_bytes(current_bytes, None))

    def test_store_and_pop_retry_hint_bytes(self):
        # Retry-hint bytes should round-trip through storage so later retries can reuse the computed jump.
        """Validate store and pop retry hint bytes."""
        _store_retry_hint_bytes(12_345)
        self.assertEqual(_pop_retry_hint_bytes(), 12_345)
        self.assertIsNone(_pop_retry_hint_bytes())


class ProtocolHandshakeTests(unittest.TestCase):
    """Tests covering protocol handshake behavior."""

    class _DummySerial:
        """Dummy serial stream used by serial helper tests."""

        def __init__(self, responses):
            """Initialize the instance.

            Parameters
            ----------
            responses : object
                Byte responses yielded by the dummy serial stream.
            """
            self._responses = iter(responses)

        def readline(self):
            """Read line.

            Returns
            -------
            object
                Next byte response from the dummy serial stream.
            """
            try:
                return next(self._responses)
            except StopIteration:
                return b""

        def reset_input_buffer(self):
            """Reset input buffer."""
            return None

        def __enter__(self):
            """Enter the context manager.

            Returns
            -------
            object
                Active context manager instance.
            """
            return self

        def __exit__(self, exc_type, exc, tb):
            """Exit the context manager.

            Parameters
            ----------
            exc_type : object
                Exception type passed to the context manager.
            exc : object
                Exception instance passed to the context manager.
            tb : object
                Traceback object passed to the context manager.

            Returns
            -------
            object
                Whether the exception was handled by the context manager.
            """
            return False

    def test_run_handshake_sends_ping_and_start_without_arm(self):
        # The harness handshake should prime the session with ping/start traffic before any arm pulse is needed.
        """Validate run handshake sends ping and start without arm."""
        harness_lines = [
            b"HARNESS READY\n",
            b"ARMED\n",
            b"runs: 10\n",
            b"harness timer output: 0.250000\n",
            b"DONE\n",
        ]
        dut_lines = [
            b"DUT READY\n",
            b"runs: 10\n",
            b"timer output: 0.250000\n",
        ]

        def serial_factory(port, *args, **kwargs):
            """Create a dummy serial connection.

            Parameters
            ----------
            port : object
                Serial port name used by the fake connection.
            *args : tuple[object, ...]
                Arguments supplied when opening the serial connection.
            **kwargs : dict[str, object]
                Keyword arguments forwarded to the helper.

            Returns
            -------
            object
                Dummy serial connection returned by the patched factory.

            Raises
            ------
            AssertionError
                If existing validation or execution checks fail.
            """
            if port == "/dev/harness":
                return self._DummySerial(harness_lines)
            if port == "/dev/dut":
                return self._DummySerial(dut_lines)
            raise AssertionError(f"Unexpected port: {port}")

        sent_commands: list[str] = []
        with patch("crest.hil_protocol.serial.Serial", side_effect=serial_factory):
            with patch(
                "crest.hil_protocol._send_line",
                side_effect=lambda _ser, text: sent_commands.append(text),
            ):
                result = hil_protocol.run_handshake(
                    dut_port="/dev/dut",
                    harness_port="/dev/harness",
                    baud_rate=115200,
                    dut_ready_timeout_s=1.0,
                    dut_timer_timeout_s=1.0,
                    harness_ready_timeout_s=1.0,
                    harness_active_timeout_s=1.0,
                    harness_done_timeout_s=1.0,
                )

        self.assertIsNone(result.error)
        self.assertTrue(result.dut_timer_found)
        self.assertTrue(result.harness_done)
        self.assertEqual(result.runs_dut, 10)
        self.assertEqual(result.runs_harness, 10)
        self.assertIn("PING", sent_commands)
        self.assertIn("START", sent_commands)
        self.assertNotIn("ARM", sent_commands)


class HarnessMetricSelectionTests(unittest.TestCase):
    """Tests covering harness metric selection behavior."""

    def test_parse_power_metrics_reads_clock_and_dwt_tags(self):
        # Power-metric parsing should capture clock and DWT tags so later scoring can reason about runtime fidelity.
        """Validate parse power metrics reads clock and dwt tags."""
        parsed = _parse_power_metrics(
            [
                "clock hz output: 480000000",
                "dwt cycles per inference output: 122345.5",
            ]
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertAlmostEqual(parsed["clock_hz"], 480000000.0)
        self.assertAlmostEqual(parsed["dwt_cycles_per_inference"], 122345.5)

    def test_normalize_power_metrics_defaults_clock_and_dwt(self):
        # Power-metric normalization should fill in stable clock and DWT defaults when the harness did not report them.
        """Validate normalize power metrics defaults clock and dwt."""
        normalized = normalize_power_metrics({"harness_latency_s": 0.2})
        self.assertAlmostEqual(normalized["harness_latency_s"], 0.2)
        self.assertEqual(normalized["clock_hz"], -1.0)
        self.assertEqual(normalized["dwt_cycles_per_inference"], -1.0)

    def test_merge_power_metrics_uses_secondary_when_primary_has_sentinel(self):
        # Power-metric merging should promote secondary readings when the primary path only has sentinels.
        """Validate merge power metrics uses secondary when primary has sentinel."""
        merged = _merge_power_metrics(
            primary={
                "harness_latency_s": 0.25,
                "clock_hz": -1.0,
                "dwt_cycles_per_inference": -1.0,
            },
            secondary={
                "clock_hz": 480000000.0,
                "dwt_cycles_per_inference": 123456.0,
            },
        )
        self.assertIsNotNone(merged)
        assert merged is not None
        self.assertAlmostEqual(merged["harness_latency_s"], 0.25)
        self.assertAlmostEqual(merged["clock_hz"], 480000000.0)
        self.assertAlmostEqual(merged["dwt_cycles_per_inference"], 123456.0)

    def test_merge_power_metrics_keeps_valid_primary_values(self):
        # Power-metric merging should preserve valid primary readings instead of overwriting them with follow-up data.
        """Validate merge power metrics keeps valid primary values."""
        merged = _merge_power_metrics(
            primary={
                "clock_hz": 400000000.0,
                "dwt_cycles_per_inference": 111111.0,
            },
            secondary={
                "clock_hz": 480000000.0,
                "dwt_cycles_per_inference": 123456.0,
            },
        )
        self.assertIsNotNone(merged)
        assert merged is not None
        self.assertAlmostEqual(merged["clock_hz"], 400000000.0)
        self.assertAlmostEqual(merged["dwt_cycles_per_inference"], 111111.0)

    def test_measure_serial_discards_energy_without_dut_timer(self):
        # Energy should be dropped when the DUT never reported its timer because the harness window cannot be aligned safely.
        """Validate measure serial discards energy without dut timer."""
        handshake_result = hil_protocol.HandshakeResult(
            dut_log=["runs: 10"],
            harness_log=[
                "runs: 10",
                "energy output (mJ): 10.0",
                "harness timer output: 0.250000",
                "DONE",
            ],
            dut_timer_found=False,
            harness_done=True,
            runs_dut=10,
            runs_harness=10,
            error=None,
        )
        compile_result = ArduinoCompileResult(
            success=True,
            log="ok",
            flash_bytes=None,
            ram_bytes=None,
            overflow_kind=None,
            build_dir=Path("build"),
        )
        upload_result = ArduinoUploadResult(success=True, log="ok")

        with patch(
            "crest.microcontrollers.arduino_base.compile_harness_sketch",
            return_value=compile_result,
        ), patch(
            "crest.microcontrollers.arduino_base.upload_harness_sketch",
            return_value=upload_result,
        ), patch(
            "crest.hil_protocol.run_handshake",
            return_value=handshake_result,
        ):
            result = measure_serial(
                serial_port="/dev/dut",
                baud_rate=115200,
                serial_timeout_s=1.0,
                harness_serial_port="/dev/harness",
                harness_auto_flash="always",
            )

        self.assertIsNone(result.latency_s)
        self.assertIsNone(result.power_metrics)

    def test_measure_serial_discards_energy_on_run_mismatch(self):
        # Energy should be dropped when DUT and harness disagree on run count so cross-window samples do not contaminate scoring.
        """Validate measure serial discards energy on run mismatch."""
        handshake_result = hil_protocol.HandshakeResult(
            dut_log=["runs: 10", "timer output: 0.250000"],
            harness_log=[
                "runs: 8",
                "energy output (mJ): 10.0",
                "harness timer output: 0.250000",
                "DONE",
            ],
            dut_timer_found=True,
            harness_done=True,
            runs_dut=10,
            runs_harness=8,
            error=None,
        )
        compile_result = ArduinoCompileResult(
            success=True,
            log="ok",
            flash_bytes=None,
            ram_bytes=None,
            overflow_kind=None,
            build_dir=Path("build"),
        )
        upload_result = ArduinoUploadResult(success=True, log="ok")

        with patch(
            "crest.microcontrollers.arduino_base.compile_harness_sketch",
            return_value=compile_result,
        ), patch(
            "crest.microcontrollers.arduino_base.upload_harness_sketch",
            return_value=upload_result,
        ), patch(
            "crest.hil_protocol.run_handshake",
            return_value=handshake_result,
        ):
            result = measure_serial(
                serial_port="/dev/dut",
                baud_rate=115200,
                serial_timeout_s=1.0,
                harness_serial_port="/dev/harness",
                harness_auto_flash="always",
            )

        self.assertAlmostEqual(result.latency_s, 0.25)
        self.assertIsNone(result.power_metrics)

    def test_measure_serial_keeps_latency_when_harness_done_missing(self):
        # If the harness misses DONE after latency was captured, the latency result should survive while the error is still reported.
        """Validate measure serial keeps latency when harness done missing."""
        handshake_result = hil_protocol.HandshakeResult(
            dut_log=["runs: 10", "timer output: 0.125000"],
            harness_log=["runs: 10", "harness error: active_timeout"],
            dut_timer_found=True,
            harness_done=False,
            runs_dut=10,
            runs_harness=10,
            error=None,
        )
        compile_result = ArduinoCompileResult(
            success=True,
            log="ok",
            flash_bytes=None,
            ram_bytes=None,
            overflow_kind=None,
            build_dir=Path("build"),
        )
        upload_result = ArduinoUploadResult(success=True, log="ok")

        with patch(
            "crest.microcontrollers.arduino_base.compile_harness_sketch",
            return_value=compile_result,
        ), patch(
            "crest.microcontrollers.arduino_base.upload_harness_sketch",
            return_value=upload_result,
        ), patch(
            "crest.hil_protocol.run_handshake",
            return_value=handshake_result,
        ):
            result = measure_serial(
                serial_port="/dev/dut",
                baud_rate=115200,
                serial_timeout_s=1.0,
                harness_serial_port="/dev/harness",
                harness_auto_flash="always",
            )

        self.assertAlmostEqual(result.latency_s, 0.125)
        self.assertIsNone(result.power_metrics)

    def test_measure_serial_merges_dut_clock_with_harness_energy(self):
        # Serial measurement should combine DUT clock telemetry with harness energy so the final payload contains both perspectives.
        """Validate measure serial merges dut clock with harness energy."""
        handshake_result = hil_protocol.HandshakeResult(
            dut_log=[
                "runs: 10",
                "clock hz output: 480000000",
                "dwt cycles per inference output: 123456",
                "timer output: 0.250000",
            ],
            harness_log=[
                "runs: 10",
                "energy output (mJ): 10.0",
                "avg power output (mW): 40.0",
                "harness timer output: 0.250000",
                "DONE",
            ],
            dut_timer_found=True,
            harness_done=True,
            runs_dut=10,
            runs_harness=10,
            error=None,
        )
        compile_result = ArduinoCompileResult(
            success=True,
            log="ok",
            flash_bytes=None,
            ram_bytes=None,
            overflow_kind=None,
            build_dir=Path("build"),
        )
        upload_result = ArduinoUploadResult(success=True, log="ok")

        with patch(
            "crest.microcontrollers.arduino_base.compile_harness_sketch",
            return_value=compile_result,
        ), patch(
            "crest.microcontrollers.arduino_base.upload_harness_sketch",
            return_value=upload_result,
        ), patch(
            "crest.hil_protocol.run_handshake",
            return_value=handshake_result,
        ):
            result = measure_serial(
                serial_port="/dev/dut",
                baud_rate=115200,
                serial_timeout_s=1.0,
                harness_serial_port="/dev/harness",
                harness_auto_flash="always",
            )

        self.assertIsNotNone(result.power_metrics)
        assert result.power_metrics is not None
        self.assertAlmostEqual(result.power_metrics["clock_hz"], 480000000.0)
        self.assertAlmostEqual(result.power_metrics["dwt_cycles_per_inference"], 123456.0)
        self.assertAlmostEqual(result.power_metrics["energy_mj_per_inference"], 10.0)

    def test_measure_serial_compiles_harness_with_measured_inference_runs(self):
        # Harness compilation should inherit the measured-run count so the helper firmware matches the DUT timing window.
        """Validate measure serial compiles harness with measured inference runs."""
        handshake_result = hil_protocol.HandshakeResult(
            dut_log=["runs: 7", "timer output: 0.125000"],
            harness_log=["runs: 7", "harness timer output: 0.125000", "DONE"],
            dut_timer_found=True,
            harness_done=True,
            runs_dut=7,
            runs_harness=7,
            error=None,
        )
        compile_result = ArduinoCompileResult(
            success=True,
            log="ok",
            flash_bytes=None,
            ram_bytes=None,
            overflow_kind=None,
            build_dir=Path("build"),
        )
        upload_result = ArduinoUploadResult(success=True, log="ok")

        with patch(
            "crest.microcontrollers.arduino_base.compile_harness_sketch",
            return_value=compile_result,
        ) as compile_mock, patch(
            "crest.microcontrollers.arduino_base.upload_harness_sketch",
            return_value=upload_result,
        ), patch(
            "crest.hil_protocol.run_handshake",
            return_value=handshake_result,
        ):
            measure_serial(
                serial_port="/dev/dut",
                baud_rate=115200,
                serial_timeout_s=1.0,
                measured_inference_runs=7,
                harness_serial_port="/dev/harness",
                harness_auto_flash="always",
            )

        self.assertEqual(compile_mock.call_args.kwargs["build_defines"]["CREST_INFERENCE_RUNS"], 7)

    def test_measure_harness_only_open_session_uses_harness_latency(self):
        # Harness-only open-session measurements should source latency from the harness because the DUT is not reporting directly.
        """Validate measure harness only open session uses harness latency."""
        session = hil_protocol.HarnessSessionResult(
            harness_log=[
                "runs: 1",
                "harness timer output: 0.250000",
                "energy output (mJ): 10.0",
                "avg power output (mW): 40.0",
                "DONE",
            ],
            harness_ready=True,
            harness_done=True,
            runs_harness=1,
            error=None,
        )
        with patch(
            "crest.microcontrollers.arduino_base.hil_protocol.wait_for_harness_done",
            return_value=session,
        ):
            result = measure_harness_only_open_session(harness=object())

        self.assertAlmostEqual(result.latency_s, 0.25)
        self.assertIsNotNone(result.power_metrics)
        assert result.power_metrics is not None
        self.assertAlmostEqual(result.power_metrics["harness_latency_s"], 0.25)
        self.assertAlmostEqual(result.power_metrics["energy_mj_per_inference"], 10.0)

    def test_measure_harness_only_open_session_timeout_sets_error(self):
        # Harness-only session timeouts should surface as latency errors once bring-up has already succeeded.
        """Validate measure harness only open session timeout sets error."""
        session = hil_protocol.HarnessSessionResult(
            harness_log=[
                "runs: 1",
                "harness timer output: 0.250000",
                "energy output (mJ): 10.0",
            ],
            harness_ready=True,
            harness_done=False,
            runs_harness=1,
            error="harness_done_timeout",
        )
        with patch(
            "crest.microcontrollers.arduino_base.hil_protocol.wait_for_harness_done",
            return_value=session,
        ):
            result = measure_harness_only_open_session(harness=object())

        self.assertIsNone(result.latency_s)
        self.assertTrue(result.serial_log[0].startswith("HARNESS_ERROR: "))
        self.assertIsNotNone(result.power_metrics)


class HarnessOnlyOrderingTests(unittest.TestCase):
    """Tests covering harness only ordering behavior."""

    def test_harness_only_opens_harness_before_upload(self):
        # Harness-only runs should open the harness session before upload so READY failures stay attached to the correct stage.
        """Validate harness only opens harness before upload."""
        device = get_device(
            "PORTENTA_H7",
            serial_port="/dev/ttyACM0",
            device_options={"target_core": "cm4", "split": "50_50", "security": "none"},
        )
        compile_result = ArduinoCompileResult(
            success=True,
            log="ok",
            flash_bytes=123,
            ram_bytes=456,
            overflow_kind=None,
            build_dir=Path("fake_build"),
        )
        upload_result = ArduinoUploadResult(success=True, log="ok")
        measure_result = ArduinoMeasureResult(
            latency_s=0.123,
            arena_error_line=None,
            serial_log=["HARNESS: DONE"],
            power_metrics={"harness_latency_s": 0.123},
        )
        prime_result = hil_protocol.HarnessSessionResult(
            harness_log=["HARNESS READY"],
            harness_ready=True,
            harness_done=False,
            runs_harness=None,
            error=None,
        )
        events: list[str] = []

        class _DummyHarness:
            """Minimal harness used to exercise controller ordering."""

            def __init__(self, *_args, **_kwargs) -> None:
                """Initialize the instance.

                Parameters
                ----------
                *_args : tuple[object, ...]
                    Positional arguments captured by the test double.
                **_kwargs : dict[str, object]
                    Keyword arguments captured by the test double.
                """
                events.append("serial_open")

            def __enter__(self):
                """Enter the context manager.

                Returns
                -------
                object
                    Active context manager instance.
                """
                events.append("serial_enter")
                return self

            def __exit__(self, exc_type, exc, tb):
                """Exit the context manager.

                Parameters
                ----------
                exc_type : object
                    Exception type passed to the context manager.
                exc : object
                    Exception instance passed to the context manager.
                tb : object
                    Traceback object passed to the context manager.

                Returns
                -------
                object
                    Whether the exception was handled by the context manager.
                """
                events.append("serial_exit")
                return False

        def _upload_side_effect(*args, **kwargs):
            """Exercise upload side effect behavior.

            Parameters
            ----------
            *args : tuple[object, ...]
                Upload callback arguments captured by the test.
            **kwargs : dict[str, object]
                Keyword arguments forwarded to the helper.

            Returns
            -------
            object
                Upload result returned by the fake callback.
            """
            del args, kwargs
            events.append("upload")
            return upload_result

        with patch.object(device, "compile", return_value=compile_result), patch(
            "crest.devices.arduino_base.ensure_harness_firmware",
            side_effect=lambda **_kwargs: events.append("ensure_harness"),
        ), patch.object(
            device,
            "prepare_for_runtime",
            side_effect=lambda **_kwargs: events.append("prepare_runtime"),
        ), patch("crest.devices.serial.Serial", side_effect=_DummyHarness), patch(
            "crest.devices.hil_protocol.prime_harness_session",
            side_effect=lambda **_kwargs: (events.append("prime"), prime_result)[1],
        ), patch.object(device, "upload", side_effect=_upload_side_effect), patch(
            "crest.devices.arduino_base.measure_harness_only_open_session",
            side_effect=lambda **_kwargs: (events.append("measure"), measure_result)[1],
        ):
            result = device.evaluate(
                dirpath=Path("candidate"),
                arena_kb=32,
                window_size=128,
                num_channels=6,
                serial_port="/dev/ttyACM0",
                run_hil=True,
                harness_serial_port="/dev/ttyACM1",
            )

        self.assertEqual(result.error_code, HIL_ERROR_OK)
        self.assertLess(events.index("prepare_runtime"), events.index("serial_open"))
        self.assertLess(events.index("serial_open"), events.index("upload"))
        self.assertLess(events.index("prime"), events.index("upload"))

    def test_harness_only_missing_diagnostics_maps_to_latency_error(self):
        # If the harness omits timing diagnostics after the run starts, classify the result as a latency-side failure.
        # diagnostics, so timeout-like misses map to HIL_ERROR_LATENCY.
        """Validate harness only missing diagnostics maps to latency error."""
        device = get_device(
            "PORTENTA_H7",
            serial_port="/dev/ttyACM0",
            device_options={"target_core": "cm4", "split": "50_50", "security": "none"},
        )
        compile_result = ArduinoCompileResult(
            success=True,
            log="ok",
            flash_bytes=123,
            ram_bytes=456,
            overflow_kind=None,
            build_dir=Path("fake_build"),
        )
        upload_result = ArduinoUploadResult(success=True, log="ok")
        measure_result = ArduinoMeasureResult(
            latency_s=None,
            arena_error_line=None,
            serial_log=["HARNESS_ERROR: harness_done_timeout"],
            power_metrics=None,
        )
        prime_result = hil_protocol.HarnessSessionResult(
            harness_log=["HARNESS READY"],
            harness_ready=True,
            harness_done=False,
            runs_harness=None,
            error=None,
        )

        class _DummyHarness:
            """Minimal harness used to exercise controller ordering."""

            def __init__(self, *_args, **_kwargs) -> None:
                """Initialize the instance.

                Parameters
                ----------
                *_args : tuple[object, ...]
                    Positional arguments captured by the test double.
                **_kwargs : dict[str, object]
                    Keyword arguments captured by the test double.
                """
                return None

            def __enter__(self):
                """Enter the context manager.

                Returns
                -------
                object
                    Active context manager instance.
                """
                return self

            def __exit__(self, exc_type, exc, tb):
                """Exit the context manager.

                Parameters
                ----------
                exc_type : object
                    Exception type passed to the context manager.
                exc : object
                    Exception instance passed to the context manager.
                tb : object
                    Traceback object passed to the context manager.

                Returns
                -------
                object
                    Whether the exception was handled by the context manager.
                """
                return False

        with patch.object(device, "compile", return_value=compile_result), patch(
            "crest.devices.arduino_base.ensure_harness_firmware",
            return_value=True,
        ), patch.object(
            device,
            "prepare_for_runtime",
            return_value=None,
        ), patch("crest.devices.serial.Serial", side_effect=_DummyHarness), patch(
            "crest.devices.hil_protocol.prime_harness_session",
            return_value=prime_result,
        ), patch.object(device, "upload", return_value=upload_result), patch(
            "crest.devices.arduino_base.measure_harness_only_open_session",
            return_value=measure_result,
        ):
            result = device.evaluate(
                dirpath=Path("candidate"),
                arena_kb=32,
                window_size=128,
                num_channels=6,
                serial_port="/dev/ttyACM0",
                run_hil=True,
                harness_serial_port="/dev/ttyACM1",
            )

        self.assertEqual(result.error_code, HIL_ERROR_LATENCY)
        self.assertEqual(result.latency_s, -1.0)
        self.assertIsNone(result.retry_hint_bytes)

    def test_harness_only_not_ready_maps_to_upload_error(self):
        # If the harness never announces READY, treat it as bring-up/upload failure because inference never actually started.
        """Validate harness only not ready maps to upload error."""
        device = get_device(
            "PORTENTA_H7",
            serial_port="/dev/ttyACM0",
            device_options={"target_core": "cm4", "split": "50_50", "security": "none"},
        )
        compile_result = ArduinoCompileResult(
            success=True,
            log="ok",
            flash_bytes=123,
            ram_bytes=456,
            overflow_kind=None,
            build_dir=Path("fake_build"),
        )
        prime_result = hil_protocol.HarnessSessionResult(
            harness_log=["noise"],
            harness_ready=False,
            harness_done=False,
            runs_harness=None,
            error="harness_ready_timeout",
        )

        class _DummyHarness:
            """Minimal harness used to exercise controller ordering."""

            def __init__(self, *_args, **_kwargs) -> None:
                """Initialize the instance.

                Parameters
                ----------
                *_args : tuple[object, ...]
                    Positional arguments captured by the test double.
                **_kwargs : dict[str, object]
                    Keyword arguments captured by the test double.
                """
                return None

            def __enter__(self):
                """Enter the context manager.

                Returns
                -------
                object
                    Active context manager instance.
                """
                return self

            def __exit__(self, exc_type, exc, tb):
                """Exit the context manager.

                Parameters
                ----------
                exc_type : object
                    Exception type passed to the context manager.
                exc : object
                    Exception instance passed to the context manager.
                tb : object
                    Traceback object passed to the context manager.

                Returns
                -------
                object
                    Whether the exception was handled by the context manager.
                """
                return False

        with patch.object(device, "compile", return_value=compile_result), patch(
            "crest.devices.arduino_base.ensure_harness_firmware",
            return_value=True,
        ), patch.object(
            device,
            "prepare_for_runtime",
            return_value=None,
        ), patch("crest.devices.serial.Serial", side_effect=_DummyHarness), patch(
            "crest.devices.hil_protocol.prime_harness_session",
            return_value=prime_result,
        ), patch.object(device, "upload") as upload_mock:
            result = device.evaluate(
                dirpath=Path("candidate"),
                arena_kb=32,
                window_size=128,
                num_channels=6,
                serial_port="/dev/ttyACM0",
                run_hil=True,
                harness_serial_port="/dev/ttyACM1",
            )

        self.assertEqual(result.error_code, HIL_ERROR_UPLOAD)
        self.assertEqual(result.latency_s, -1.0)
        upload_mock.assert_not_called()

    def test_harness_only_prepare_failure_maps_to_upload_error(self):
        # Harness-only runtime preparation failures should stay in the upload bucket so callers can separate staging failures from measurement timeouts.
        """Validate harness only prepare failure maps to upload error."""
        device = get_device(
            "PORTENTA_H7",
            serial_port="/dev/ttyACM0",
            device_options={"target_core": "cm4", "split": "50_50", "security": "none"},
        )
        compile_result = ArduinoCompileResult(
            success=True,
            log="ok",
            flash_bytes=123,
            ram_bytes=456,
            overflow_kind=None,
            build_dir=Path("fake_build"),
        )

        with patch.object(device, "compile", return_value=compile_result), patch(
            "crest.devices.arduino_base.ensure_harness_firmware",
            side_effect=RuntimeError("Harness compile failed."),
        ), patch.object(
            device,
            "prepare_for_runtime",
            return_value=None,
        ), patch("crest.devices.serial.Serial") as serial_mock, patch.object(
            device, "upload"
        ) as upload_mock:
            result = device.evaluate(
                dirpath=Path("candidate"),
                arena_kb=32,
                window_size=128,
                num_channels=6,
                serial_port="/dev/ttyACM0",
                run_hil=True,
                harness_serial_port="/dev/ttyACM1",
            )

        self.assertEqual(result.error_code, HIL_ERROR_UPLOAD)
        self.assertEqual(result.latency_s, -1.0)
        serial_mock.assert_not_called()
        upload_mock.assert_not_called()

    def test_harness_only_prepare_for_runtime_runtimeerror_maps_to_upload_error(self):
        # prepare_for_runtime failures in harness-only mode should stay in the upload bucket because runtime execution never actually began.
        """Validate harness only prepare for runtime runtimeerror maps to upload error."""
        device = get_device(
            "PORTENTA_H7",
            serial_port="/dev/ttyACM0",
            device_options={"target_core": "cm4", "split": "50_50", "security": "none"},
        )
        compile_result = ArduinoCompileResult(
            success=True,
            log="ok",
            flash_bytes=123,
            ram_bytes=456,
            overflow_kind=None,
            build_dir=Path("fake_build"),
        )

        with patch.object(device, "compile", return_value=compile_result), patch.object(
            device,
            "prepare_for_runtime",
            side_effect=RuntimeError("CM7 boot helper failed."),
        ), patch(
            "crest.devices.arduino_base.ensure_harness_firmware"
        ) as ensure_mock, patch("crest.devices.serial.Serial") as serial_mock, patch.object(
            device, "upload"
        ) as upload_mock:
            result = device.evaluate(
                dirpath=Path("candidate"),
                arena_kb=32,
                window_size=128,
                num_channels=6,
                serial_port="/dev/ttyACM0",
                run_hil=True,
                harness_serial_port="/dev/ttyACM1",
            )

        self.assertEqual(result.error_code, HIL_ERROR_UPLOAD)
        self.assertEqual(result.latency_s, -1.0)
        ensure_mock.assert_not_called()
        serial_mock.assert_not_called()
        upload_mock.assert_not_called()


class DeviceTimeoutPassThroughTests(unittest.TestCase):
    """Tests covering device timeout pass through behavior."""

    def test_arduino_prepare_candidate_float_does_not_require_calibration(self):
        """Arduino float exports should stage without representative data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = Path(tmpdir) / "candidate"
            request = CandidatePrepareRequest(
                config=Mock(),
                model=Mock(),
                model_variant="approx_trained",
                artifact_root=artifact_root,
                tflite_model_path=Path(tmpdir) / "model.tflite",
                calibration_split=None,
                quantization_mode="float",
                input_shape=(32, 6),
            )
            device = ArduinoDevice("ARDUINO_NANO_33_BLE_SENSE")

            with patch("crest.hardware.convert_to_tflite_model") as tflite_mock, patch(
                "crest.hardware.convert_to_cpp_model"
            ) as cpp_mock, patch("crest.devices._sync_arduino_sketch_variant_for_config") as sync_mock:
                prepared_dir = device.prepare_candidate(request=request)

        self.assertEqual(prepared_dir, artifact_root)
        self.assertIsNone(tflite_mock.call_args.kwargs["training_data"])
        self.assertEqual(tflite_mock.call_args.kwargs["quantization_mode"], "float")
        cpp_mock.assert_called_once()
        sync_mock.assert_called_once_with(request.config, artifact_root)

    def test_arduino_device_measure_preserves_zero_timeouts(self):
        # Zero timeout overrides should survive measurement setup instead of being replaced by defaults.
        """Validate Arduino device measure preserves zero timeouts."""
        device = ArduinoDevice("ARDUINO_NANO_33_BLE_SENSE")
        fake_result = ArduinoMeasureResult(
            latency_s=0.1,
            arena_error_line=None,
            serial_log=["timer output: 0.1"],
            power_metrics=None,
        )
        with patch(
            "crest.microcontrollers.arduino_base.measure_serial",
            return_value=fake_result,
        ) as mock_measure:
            device.measure(
                serial_port="/dev/dut",
                baud_rate=115200,
                serial_timeout_s=1.0,
                dut_ready_timeout_s=0.0,
                harness_serial_port="/dev/harness",
                harness_ready_timeout_s=0.0,
                harness_arm_timeout_s=0.0,
                harness_active_timeout_s=0.0,
                harness_done_timeout_s=0.0,
            )

        call_kwargs = mock_measure.call_args.kwargs
        self.assertEqual(call_kwargs["dut_ready_timeout_s"], 0.0)
        self.assertEqual(call_kwargs["harness_ready_timeout_s"], 0.0)
        self.assertEqual(call_kwargs["harness_arm_timeout_s"], 0.0)
        self.assertEqual(call_kwargs["harness_active_timeout_s"], 0.0)
        self.assertEqual(call_kwargs["harness_done_timeout_s"], 0.0)

    def test_arduino_device_measure_forwards_measured_inference_runs(self):
        # Measured-run overrides should flow through Arduino measurement so latency and energy use the requested averaging window.
        """Validate Arduino device measure forwards measured inference runs."""
        device = ArduinoDevice("ARDUINO_NANO_33_BLE_SENSE")
        fake_result = ArduinoMeasureResult(
            latency_s=0.1,
            arena_error_line=None,
            serial_log=["timer output: 0.1"],
            power_metrics=None,
        )
        with patch(
            "crest.microcontrollers.arduino_base.measure_serial",
            return_value=fake_result,
        ) as mock_measure:
            device.measure(
                serial_port="/dev/dut",
                baud_rate=115200,
                serial_timeout_s=1.0,
                measured_inference_runs=7,
            )

        self.assertEqual(mock_measure.call_args.kwargs["measured_inference_runs"], 7)

    def test_arduino_device_evaluate_compiles_dut_with_measured_inference_runs(self):
        # Arduino DUT compilation should bake in the measured-run count so the flashed sketch matches the requested loop length.
        """Validate Arduino device evaluate compiles dut with measured inference runs."""
        device = ArduinoDevice("ARDUINO_NANO_33_BLE_SENSE", serial_port="/dev/ttyACM0")
        compile_result = ArduinoCompileResult(
            success=True,
            log="ok",
            flash_bytes=123,
            ram_bytes=456,
            overflow_kind=None,
            build_dir=Path("fake_build"),
        )
        upload_result = ArduinoUploadResult(success=True, log="ok")
        measure_result = ArduinoMeasureResult(
            latency_s=0.1,
            arena_error_line=None,
            serial_log=["timer output: 0.1"],
            power_metrics=None,
        )
        with patch.object(device, "compile", return_value=compile_result) as compile_mock, patch.object(
            device, "upload", return_value=upload_result
        ), patch.object(
            device, "measure", return_value=measure_result
        ):
            device.evaluate(
                dirpath=Path("candidate"),
                arena_kb=32,
                window_size=128,
                num_channels=6,
                serial_port="/dev/ttyACM0",
                run_hil=True,
                measured_inference_runs=7,
            )

        self.assertEqual(compile_mock.call_args.kwargs["build_defines"]["CREST_INFERENCE_RUNS"], 7)

    def test_portenta_cm4_harness_only_compiles_harness_with_measured_inference_runs(self):
        # Portenta CM4 harness-only runs should compile the helper harness with the requested measured-run count.
        """Validate Portenta CM4 harness only compiles harness with measured inference runs."""
        device = get_device(
            "PORTENTA_H7",
            serial_port="/dev/ttyACM0",
            device_options={"target_core": "cm4", "split": "50_50", "security": "none"},
        )
        compile_result = ArduinoCompileResult(
            success=True,
            log="ok",
            flash_bytes=123,
            ram_bytes=456,
            overflow_kind=None,
            build_dir=Path("fake_build"),
        )
        upload_result = ArduinoUploadResult(success=True, log="ok")
        measure_result = ArduinoMeasureResult(
            latency_s=0.1,
            arena_error_line=None,
            serial_log=["HARNESS: DONE"],
            power_metrics=None,
        )
        prime_result = hil_protocol.HarnessSessionResult(
            harness_log=["HARNESS READY"],
            harness_ready=True,
            harness_done=False,
            runs_harness=None,
            error=None,
        )

        class _DummyHarness:
            """Minimal harness used to exercise controller ordering."""

            def __enter__(self):
                """Enter the context manager.

                Returns
                -------
                object
                    Active context manager instance.
                """
                return self

            def __exit__(self, exc_type, exc, tb):
                """Exit the context manager.

                Parameters
                ----------
                exc_type : object
                    Exception type passed to the context manager.
                exc : object
                    Exception instance passed to the context manager.
                tb : object
                    Traceback object passed to the context manager.

                Returns
                -------
                object
                    Whether the exception was handled by the context manager.
                """
                return False

        with patch.object(device, "compile", return_value=compile_result), patch.object(
            device, "upload", return_value=upload_result
        ), patch.object(
            device, "prepare_for_runtime", return_value=None
        ), patch(
            "crest.devices.serial.Serial", return_value=_DummyHarness()
        ), patch(
            "crest.devices.hil_protocol.prime_harness_session", return_value=prime_result
        ), patch(
            "crest.devices.arduino_base.measure_harness_only_open_session",
            return_value=measure_result,
        ), patch(
            "crest.devices.arduino_base.ensure_harness_firmware"
        ) as ensure_mock:
            device.evaluate(
                dirpath=Path("candidate"),
                arena_kb=32,
                window_size=128,
                num_channels=6,
                serial_port="/dev/ttyACM0",
                run_hil=True,
                harness_serial_port="/dev/ttyACM1",
                measured_inference_runs=7,
            )

        self.assertEqual(ensure_mock.call_args.kwargs["build_defines"]["CREST_INFERENCE_RUNS"], 7)


class HILSpecErrorTests(unittest.TestCase):
    """Tests covering HIL spec error behavior."""

    def test_hil_spec_default_dirpath_uses_odom_candidate_dir(self):
        """HIL spec should default to the renamed odometry candidate directory."""
        signature = inspect.signature(HIL_spec)

        self.assertEqual(signature.parameters["dirpath"].default, "odom_tcn/")

    def test_hil_controller_default_dirpath_uses_odom_candidate_dir(self):
        """HIL controller should default to the renamed odometry candidate directory."""
        signature = inspect.signature(HIL_controller)

        self.assertEqual(signature.parameters["dirpath"].default, "odom_tcn/")

    def _write_sketch(self, sketch_dir: Path) -> None:
        """Write sketch.

        Parameters
        ----------
        sketch_dir : Path
            Sketch directory forwarded to the hardware helper.
        """
        sketch_dir.mkdir(parents=True, exist_ok=True)
        (sketch_dir / "crest.ino").write_text(
            "\n".join(
                [
                    "#define CREST_WINDOW_SIZE 100",
                    "#define CREST_NUM_CHANNELS 1",
                    "#define CREST_TENSOR_ARENA_BYTES (10 * 1024)",
                ]
            )
        )

    def test_hil_spec_upload_failure_sets_error_flag(self):
        # HIL spec generation should flag upload failures so later phases do not treat them like runtime timings.
        """Validate hil spec upload failure sets error flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir)
            self._write_sketch(sketch_dir)
            compile_result = _FakeCompletedProcess(stdout=COMPILE_SAMPLE_OUTPUT)
            upload_result = _FakeCompletedProcess(returncode=1)
            with patch(
                "crest.microcontrollers.arduino_base.subprocess.run",
                side_effect=[compile_result, upload_result],
            ) as mock_run:
                ram, flash, latency, arena_bytes, err, _power = HIL_spec(
                    dirpath=sketch_dir,
                    chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                    serial_port="/dev/ttyMock",
                    compile_only=False,
                )
            self.assertEqual(err, HIL_ERROR_UPLOAD)
            self.assertEqual(latency, -1.0)
            self.assertEqual(ram, 98112)
            self.assertEqual(flash, 376104)
            self.assertGreater(arena_bytes, 0)
            self.assertEqual(mock_run.call_count, 2)

    def test_hil_spec_latency_timeout_sets_error_flag(self):
        # HIL spec generation should flag latency timeouts so runtime failures stay distinguishable from build failures.
        """Validate hil spec latency timeout sets error flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir)
            self._write_sketch(sketch_dir)
            compile_result = _FakeCompletedProcess(stdout=COMPILE_SAMPLE_OUTPUT)
            upload_result = _FakeCompletedProcess(stdout="upload ok")
            with patch(
                "crest.microcontrollers.arduino_base.subprocess.run",
                side_effect=[compile_result, upload_result],
            ):
                with patch(
                    "crest.hil_protocol.run_dut_only",
                    return_value=["boot ok"],
                ):
                    ram, flash, latency, arena_bytes, err, _power = HIL_spec(
                        dirpath=sketch_dir,
                        chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                        serial_port="/dev/ttyMock",
                        compile_only=False,
                    )
        self.assertEqual(err, HIL_ERROR_LATENCY)
        self.assertEqual(latency, -1.0)
        self.assertEqual(ram, 98112)
        self.assertEqual(flash, 376104)
        self.assertGreater(arena_bytes, 0)

    def test_hil_spec_rejects_out_of_range_arena_index(self):
        # HIL spec generation should reject arena indices outside the candidate list.
        """Validate hil spec rejects out of range arena index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir)
            self._write_sketch(sketch_dir)
            with self.assertRaises(IndexError):
                HIL_spec(
                    dirpath=sketch_dir,
                    chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                    idx=999,
                    compile_only=True,
                )

    def test_hil_spec_detects_flash_overflow(self):
        # Detect HIL spec detects flash overflow so error classification and pruning stay stable.
        """Validate hil spec detects flash overflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir)
            self._write_sketch(sketch_dir)
            compile_result = _FakeCompletedProcess(
                returncode=1,
                stdout="",
                stderr=FLASH_OVERFLOW_STDERR,
            )
            with patch(
                "crest.microcontrollers.arduino_base.subprocess.run",
                return_value=compile_result,
            ) as mock_run:
                ram, flash, latency, arena_bytes, err, _power = HIL_spec(
                    dirpath=sketch_dir,
                    chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                    compile_only=True,
                )
        self.assertEqual(err, HIL_ERROR_FLASH_OVERFLOW)
        self.assertEqual((ram, flash, latency), (-1, -1, -1.0))
        self.assertGreater(arena_bytes, 0)
        self.assertEqual(mock_run.call_count, 1)

    def test_hil_spec_detects_ram_overflow_message(self):
        # Detect HIL spec detects RAM overflow message so error classification and pruning stay stable.
        """Validate hil spec detects ram overflow message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir)
            self._write_sketch(sketch_dir)
            compile_result = _FakeCompletedProcess(
                returncode=1,
                stdout="",
                stderr=RAM_OVERFLOW_STDERR,
            )
            with patch(
                "crest.microcontrollers.arduino_base.subprocess.run",
                return_value=compile_result,
            ) as mock_run:
                ram, flash, latency, arena_bytes, err, _power = HIL_spec(
                    dirpath=sketch_dir,
                    chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                    compile_only=True,
                )
        self.assertEqual(err, HIL_ERROR_RAM_OVERFLOW)
        self.assertEqual((ram, flash, latency), (-1, -1, -1.0))
        self.assertGreater(arena_bytes, 0)
        self.assertEqual(mock_run.call_count, 1)

    def test_hil_spec_maps_arena_errors_to_under_sized_flag(self):
        # Arena-sizing failures should set the undersized flag so the search loop can widen the next attempt.
        """Validate hil spec maps arena errors to under sized flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir)
            self._write_sketch(sketch_dir)
            compile_result = _FakeCompletedProcess(stdout=COMPILE_SAMPLE_OUTPUT)
            upload_result = _FakeCompletedProcess(stdout="upload ok")
            with patch(
                    "crest.microcontrollers.arduino_base.subprocess.run",
                    side_effect=[compile_result, upload_result],
                ):
                with patch(
                    "crest.hil_protocol.run_dut_only",
                    return_value=["size is too small for all buffers"],
                ):
                    ram, flash, latency, arena_bytes, err, _power = HIL_spec(
                        dirpath=sketch_dir,
                        chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                        serial_port="/dev/ttyMock",
                        compile_only=False,
                    )
        self.assertEqual(err, HIL_ERROR_UNDER_SIZED)
        self.assertEqual(latency, -1.0)
        self.assertEqual(ram, 98112)
        self.assertEqual(flash, 376104)
        self.assertGreater(arena_bytes, 0)


class HILControllerTests(unittest.TestCase):
    """Tests covering HIL controller behavior."""

    @staticmethod
    def _controller_device(arena_sizes_kb: np.ndarray, *, name: str = "ARDUINO_NANO_33_BLE_SENSE"):
        """Create a controller device test double.

        Parameters
        ----------
        arena_sizes_kb : np.ndarray
            Arena sizes in kilobytes used by the helper.
        name : str
            Parameter name requested by the fake sampler.

        Returns
        -------
        object
            Device specification exposed by the fake controller.
        """
        class _Spec:
            """Tests covering spec behavior."""

            def __init__(self, spec_name: str, arenas: np.ndarray) -> None:
                """Initialize the instance.

                Parameters
                ----------
                spec_name : str
                    Device specification name exposed by the fake object.
                arenas : np.ndarray
                    Arena sizes exposed by the fake device specification.
                """
                self.name = spec_name
                self.arena_sizes_kb = [int(v) for v in arenas]

        class _Device:
            """Tests covering device behavior."""

            def __init__(self, spec_name: str, arenas: np.ndarray) -> None:
                """Initialize the instance.

                Parameters
                ----------
                spec_name : str
                    Device specification name exposed by the fake object.
                arenas : np.ndarray
                    Arena sizes exposed by the fake device specification.
                """
                self.spec = _Spec(spec_name, arenas)

        return _Device(name, arena_sizes_kb)

    def test_hil_controller_success_on_first_candidate(self):
        # The HIL controller should stop at the first successful arena candidate.
        """Validate hil controller success on first candidate."""
        arena_candidates = np.array([10, 20])
        device = self._controller_device(arena_candidates)
        hil_return = (64000, 128000, 0.25, 10 * 1024, HIL_ERROR_OK, None)
        with patch("crest.hardware.HIL_spec",
            return_value=hil_return,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
                device=device,
            )
        self.assertEqual(mock_spec.call_count, 1)
        self.assertEqual(ram, 64000)
        self.assertEqual(flash, 128000)
        self.assertAlmostEqual(latency, 0.25)
        self.assertEqual(arena_bytes, 10 * 1024)
        self.assertEqual(master_error, HIL_MASTER_SUCCESS)
        self.assertIsNone(_power_metrics)

    def test_hil_controller_exhausts_candidates(self):
        # The HIL controller should return the final failure once every arena candidate has been tried.
        """Validate hil controller exhausts candidates."""
        arena_candidates = np.array([10, 20])
        device = self._controller_device(arena_candidates)

        def hil_side_effect(**_kwargs):
            """Return fake HIL metrics for the requested arena.

            Parameters
            ----------
            **_kwargs : dict[str, object]
                Keyword arguments captured by the test double.

            Returns
            -------
            object
                Metrics returned by the fake HIL callback.
            """
            idx = _kwargs["idx"]
            arena = arena_candidates[idx] * 1024
            return (50000, 100000, -1.0, arena, HIL_ERROR_LATENCY, None)

        with patch("crest.hardware.HIL_spec",
            side_effect=hil_side_effect,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
                device=device,
            )
        self.assertEqual(mock_spec.call_count, len(arena_candidates))
        self.assertEqual(
            (ram, flash, latency, arena_bytes, master_error, _power_metrics),
            (-1, -1, -1.0, -1, HIL_MASTER_ARENA_EXHAUSTED, None),
        )

    def test_hil_controller_integrity_failure_is_infrastructure_without_arena_retry(self):
        """An integrity failure must stop arena search and retain infrastructure evidence."""
        from crest.errors import HIL_ERROR_INTEGRITY, HIL_MASTER_INFRASTRUCTURE
        device = self._controller_device(np.array([-1]), name="STM32_NUCLEO_N657X0_Q")
        evidence = {"measurement_record_paths": ["first.json", "second.json"]}
        with patch("crest.hardware.HIL_spec", return_value=(50000, 100000, -1.0, -1024, HIL_ERROR_INTEGRITY, evidence)) as spec:
            result = HIL_controller(dirpath="unused", chosen_device="STM32_NUCLEO_N657X0_Q", run_hil=True, device=device)
        self.assertEqual(spec.call_count, 1)
        self.assertEqual(result[4], HIL_MASTER_INFRASTRUCTURE)
        self.assertEqual(result[5], evidence)

    def test_hil_controller_single_shot_stm_timeout_is_fatal(self):
        # Single-shot STM32 timeouts should be treated as fatal because there is no arena-search retry path to recover them.
        # runtime failure instead of being rewritten as arena exhaustion.
        """Validate hil controller single shot stm timeout is fatal."""
        arena_candidates = np.array([-1])
        device = self._controller_device(arena_candidates, name="STM32_NUCLEO_N657X0_Q")

        with patch(
            "crest.hardware.HIL_spec",
            return_value=(50000, 100000, -1.0, -1024, HIL_ERROR_LATENCY, None),
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="STM32_NUCLEO_N657X0_Q",
                run_hil=True,
                device=device,
            )

        self.assertEqual(mock_spec.call_count, 1)
        self.assertEqual(
            (ram, flash, latency, arena_bytes, master_error, _power_metrics),
            (50000, 100000, -1.0, -1024, HIL_MASTER_FATAL, None),
        )

    def test_hil_controller_non_arena_failure(self):
        # Non-arena failures should surface immediately instead of triggering another arena candidate.
        """Validate hil controller non arena failure."""
        arena_candidates = np.array([10, 20])
        device = self._controller_device(arena_candidates)
        hil_return = (72000, 160000, -1.0, 10 * 1024, HIL_ERROR_COMPILE, None)
        with patch("crest.hardware.HIL_spec",
            return_value=hil_return,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
                device=device,
            )
        self.assertEqual(mock_spec.call_count, 1)
        self.assertEqual(
            (ram, flash, latency, arena_bytes, master_error),
            (72000, 160000, -1.0, 10 * 1024, HIL_MASTER_FATAL),
        )

    def test_hil_controller_reports_flash_overflow(self):
        # Flash overflows should bubble out of the HIL controller with the stable overflow code.
        """Validate hil controller reports flash overflow."""
        arena_candidates = np.array([10])
        device = self._controller_device(arena_candidates)
        hil_return = (-1, -1, -1.0, 10 * 1024, HIL_ERROR_FLASH_OVERFLOW, None)
        with patch("crest.hardware.HIL_spec",
            return_value=hil_return,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
                device=device,
            )
        self.assertEqual(mock_spec.call_count, 1)
        self.assertEqual(
            (ram, flash, latency, arena_bytes, master_error),
            (-1, -1, -1.0, 10 * 1024, HIL_MASTER_FLASH_OVERFLOW),
        )

    def test_hil_controller_reports_device_not_found(self):
        # Device lookup failures should surface immediately so the caller sees a catalog problem, not a measurement failure.
        """Validate hil controller reports device not found."""
        arena_candidates = np.array([10])
        device = self._controller_device(arena_candidates)
        hil_return = (64000, 128000, -1.0, 10 * 1024, HIL_ERROR_UPLOAD, None)
        with patch("crest.hardware.HIL_spec",
            return_value=hil_return,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
                device=device,
            )
        self.assertEqual(mock_spec.call_count, 1)
        self.assertEqual(
            (ram, flash, latency, arena_bytes, master_error),
            (64000, 128000, -1.0, 10 * 1024, HIL_MASTER_DEVICE_NOT_FOUND),
        )

    def test_hil_controller_prefers_smallest_successful_arena(self):
        # When several arena sizes work, the HIL controller should keep the smallest successful one.
        """Validate hil controller prefers smallest successful arena."""
        arena_candidates = np.array([10, 20, 40, 80])
        device = self._controller_device(arena_candidates)
        call_log: list[int] = []

        def hil_side_effect(**kwargs):
            """Return fake HIL metrics for the requested arena.

            Parameters
            ----------
            **kwargs : dict[str, object]
                Keyword arguments forwarded to the helper.

            Returns
            -------
            object
                Metrics returned by the fake HIL callback.
            """
            idx = kwargs["idx"]
            call_log.append(idx)
            arena = arena_candidates[idx] * 1024
            if len(call_log) == 1:
                self.assertEqual(idx, 1)
                return (64000, 128000, 0.25, arena, HIL_ERROR_OK, None)
            self.assertEqual(idx, 0)
            return (-1, -1, -1.0, arena, HIL_ERROR_UNDER_SIZED, None)

        with patch("crest.hardware.HIL_spec",
            side_effect=hil_side_effect,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
                device=device,
            )
        self.assertEqual(mock_spec.call_count, 2)
        self.assertListEqual(call_log, [1, 0])
        self.assertEqual(master_error, HIL_MASTER_SUCCESS)
        self.assertEqual(arena_bytes, 20 * 1024)
        self.assertEqual((ram, flash), (64000, 128000))
        self.assertAlmostEqual(latency, 0.25)

    def test_hil_controller_reports_master_ram_overflow_at_smallest(self):
        # If even the smallest arena candidate overflows, the controller should report master RAM overflow.
        """Validate hil controller reports master ram overflow at smallest."""
        arena_candidates = np.array([10])
        device = self._controller_device(arena_candidates)
        hil_return = (-1, -1, -1.0, 10 * 1024, HIL_ERROR_RAM_OVERFLOW, None)
        with patch("crest.hardware.HIL_spec",
            return_value=hil_return,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
                device=device,
            )
        self.assertEqual(mock_spec.call_count, 1)
        self.assertEqual(master_error, HIL_MASTER_RAM_OVERFLOW)
        self.assertEqual((ram, flash, latency, arena_bytes), hil_return[:4])

    def test_hil_controller_retains_success_after_smaller_failure(self):
        # A later smaller-arena failure should not erase the best successful candidate already found.
        """Validate hil controller retains success after smaller failure."""
        arena_candidates = np.array([10, 20, 40])
        device = self._controller_device(arena_candidates)
        call_order: list[int] = []

        def hil_side_effect(**kwargs):
            """Return fake HIL metrics for the requested arena.

            Parameters
            ----------
            **kwargs : dict[str, object]
                Keyword arguments forwarded to the helper.

            Returns
            -------
            object
                Metrics returned by the fake HIL callback.
            """
            idx = kwargs["idx"]
            call_order.append(idx)
            arena = arena_candidates[idx] * 1024
            if len(call_order) == 1:
                self.assertEqual(idx, 1)
                return (70000, 150000, 0.3, arena, HIL_ERROR_OK, None)
            self.assertEqual(idx, 0)
            return (-1, -1, -1.0, arena, HIL_ERROR_RAM_OVERFLOW, None)

        with patch("crest.hardware.HIL_spec",
            side_effect=hil_side_effect,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
                device=device,
            )
        self.assertEqual(mock_spec.call_count, 2)
        self.assertListEqual(call_order, [1, 0])
        self.assertEqual(master_error, HIL_MASTER_SUCCESS)
        self.assertEqual(arena_bytes, 20 * 1024)
        self.assertEqual((ram, flash), (70000, 150000))
        self.assertAlmostEqual(latency, 0.3)

    def test_hil_controller_uses_retry_hint_to_jump(self):
        # Retry hints should let the HIL controller skip directly to a more plausible arena candidate.
        """Validate hil controller uses retry hint to jump."""
        _store_retry_hint_bytes(None)
        arena_candidates = np.array([10, 20, 40, 80])
        device = self._controller_device(arena_candidates)
        call_sequence: list[int] = []

        def hil_side_effect(**kwargs):
            """Return fake HIL metrics for the requested arena.

            Parameters
            ----------
            **kwargs : dict[str, object]
                Keyword arguments forwarded to the helper.

            Returns
            -------
            object
                Metrics returned by the fake HIL callback.
            """
            idx = kwargs["idx"]
            call_sequence.append(idx)
            arena = arena_candidates[idx] * 1024
            # _store_retry_hint_bytes(None)  # Removed: should only be called before the first call
            if len(call_sequence) == 1:
                _store_retry_hint_bytes(70 * 1024)
                return (-1, -1, -1.0, arena, HIL_ERROR_UNDER_SIZED, None)
            if idx == 3:
                return (61000, 120000, 0.2, arena, HIL_ERROR_OK, None)
            # The controller should probe the next smaller arena after a success.
            self.assertEqual(idx, 2)
            return (-1, -1, -1.0, arena, HIL_ERROR_UNDER_SIZED, None)

        with patch("crest.hardware.HIL_spec",
            side_effect=hil_side_effect,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
                device=device,
            )

        self.assertEqual(call_sequence[:2], [1, 3])
        self.assertEqual(call_sequence[-1], 2)
        self.assertEqual(master_error, HIL_MASTER_SUCCESS)
        self.assertEqual(arena_bytes, 80 * 1024)
        self.assertEqual((ram, flash), (61000, 120000))
        self.assertAlmostEqual(latency, 0.2)

    def test_hil_controller_uses_injected_device_spec_not_catalog(self):
        # Injected device specs should take precedence so tests can pin the HIL controller to a precise hardware contract.
        """Validate hil controller uses injected device spec not catalog."""
        arena_candidates = np.array([11, 33, 77])
        device = self._controller_device(arena_candidates, name="PORTENTA_H7")
        observed_indices: list[int] = []

        def hil_side_effect(**kwargs):
            """Return fake HIL metrics for the requested arena.

            Parameters
            ----------
            **kwargs : dict[str, object]
                Keyword arguments forwarded to the helper.

            Returns
            -------
            object
                Metrics returned by the fake HIL callback.
            """
            idx = kwargs["idx"]
            observed_indices.append(idx)
            arena = arena_candidates[idx] * 1024
            if idx == 1:
                return (61000, 120000, 0.2, arena, HIL_ERROR_OK, None)
            return (-1, -1, -1.0, arena, HIL_ERROR_UNDER_SIZED, None)

        with patch(
            "crest.hardware.arena_size_candidates",
            side_effect=AssertionError("arena_size_candidates should not be used for injected devices"),
        ), patch(
            "crest.hardware.HIL_spec",
            side_effect=hil_side_effect,
        ):
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="PORTENTA_H7",
                device_options=None,
                run_hil=False,
                device=device,
            )

        self.assertEqual(master_error, HIL_MASTER_SUCCESS)
        self.assertEqual(observed_indices[0], 1)
        self.assertEqual(arena_bytes, 33 * 1024)
        self.assertEqual((ram, flash), (61000, 120000))
        self.assertAlmostEqual(latency, 0.2)


class IntegrationTests(TinyModelMixin, unittest.TestCase):
    """Tests covering integration behavior."""

    @unittest.skipUnless(shutil.which("xxd"), "xxd command required for this test.")
    def test_compile_only_pipeline(self):
        # Compile-only HIL runs should stop after staging and size accounting without entering runtime measurement.
        """Validate compile only pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tflite_path = tmp_path / "model_full.tflite"
            convert_to_tflite_model(self.model, self.train_x, output_name=tflite_path)
            cpp_dir = tmp_path / "cpp_out"
            convert_to_cpp_model(tflite_path, cpp_dir)
            self.assertTrue((cpp_dir / "model.cc").exists())
            self.assertTrue((cpp_dir / "model.h").exists())

            sketch_dir = tmp_path / "odom_tcn"
            sketch_dir.mkdir()
            (sketch_dir / "crest.ino").write_text(
                "\n".join(
                    [
                        "#define CREST_WINDOW_SIZE 100",
                        "#define CREST_NUM_CHANNELS 1",
                        "#define CREST_TENSOR_ARENA_BYTES (10 * 1024)",
                    ]
                )
            )

            compile_result = _FakeCompletedProcess(stdout=COMPILE_SAMPLE_OUTPUT)
            with patch("crest.microcontrollers.arduino_base.subprocess.run", return_value=compile_result) as mock_run:
                    ram, flash, latency, arena_bytes, err, _power = HIL_spec(
                    dirpath=sketch_dir,
                    chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                    compile_only=True,
                )

        self.assertEqual(err, HIL_ERROR_OK)
        self.assertEqual(latency, -1.0)
        self.assertEqual(ram, 98112)
        self.assertEqual(flash, 376104)
        self.assertGreater(arena_bytes, 0)
        compile_args = mock_run.call_args[0][0]
        self.assertIn("compile", compile_args)

if __name__ == "__main__":
    defaultTest=None
    # defaultTest='ConversionHelperTests.test_convert_to_tflite_model_creates_file'

    unittest.main(defaultTest=defaultTest)
