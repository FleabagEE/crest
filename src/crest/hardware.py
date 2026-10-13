# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Legacy HIL orchestration helpers and export utilities for CREST.

This module still owns the compatibility path that exports Keras models to
TFLite/C arrays, resolves device resource limits, performs single-attempt HIL
runs, and searches tensor-arena sizes while the backend migration to dedicated
device classes is in progress.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union, Optional, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import logging, absl.logging
tf.get_logger().setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
logger = logging.getLogger(__name__)

import subprocess
import time
import tempfile

from .devices import DeviceInterface, DEVICE_SPECS, get_device_spec  # legacy catalog moved to devices.py (Phase 1 bridge)
from .errors import (
    HIL_ERROR_OK,
    HIL_ERROR_COMPILE,
    HIL_ERROR_LATENCY,
    HIL_ERROR_UNDER_SIZED,
    HIL_ERROR_FLASH_OVERFLOW,
    HIL_ERROR_RAM_OVERFLOW,
    HIL_ERROR_UPLOAD,
    HIL_ERROR_INTEGRITY,
    HIL_MASTER_PENDING,
    HIL_MASTER_SUCCESS,
    HIL_MASTER_ARENA_EXHAUSTED,
    HIL_MASTER_FATAL,
    HIL_MASTER_FLASH_OVERFLOW,
    HIL_MASTER_RAM_OVERFLOW,
    HIL_MASTER_DEVICE_NOT_FOUND,
    HIL_MASTER_INFRASTRUCTURE,
)
from .microcontrollers.arduino_base import normalize_power_metrics
from .microcontrollers import get_device as get_microcontroller_device

VALID_TFLITE_QUANTIZATION_MODES = {"float", "int8_ptq"}
MAX_REPRESENTATIVE_EXAMPLES = 1000


class TFLiteSubprocessError(RuntimeError):
    """Describe a failed isolated TFLite prediction subprocess.

    Parameters
    ----------
    model_path : str | pathlib.Path
        TFLite flatbuffer path supplied to the worker.
    return_code : int | None
        Worker process return code, or ``None`` when the process timed out
        before returning.
    timeout : bool
        Whether the worker exceeded the parent-side timeout.
    stderr_tail : str
        Tail of worker stderr captured for diagnostics.
    command : Sequence[str]
        Command used to launch the worker process.
    """

    def __init__(
        self,
        *,
        model_path: Union[str, Path],
        return_code: int | None,
        timeout: bool,
        stderr_tail: str,
        command: Sequence[str],
    ) -> None:
        """Initialize one subprocess failure description.

        Parameters
        ----------
        model_path : str | pathlib.Path
            TFLite flatbuffer path supplied to the worker.
        return_code : int | None
            Worker process return code, or ``None`` for timeout failures.
        timeout : bool
            Whether the worker exceeded the parent-side timeout.
        stderr_tail : str
            Captured worker stderr tail.
        command : Sequence[str]
            Command used to launch the worker process.

        Returns
        -------
        None
        """
        self.model_path = Path(model_path)
        self.return_code = return_code
        self.timeout = bool(timeout)
        self.stderr_tail = stderr_tail
        self.command = tuple(command)
        reason = "timed out" if self.timeout else f"exited with code {self.return_code}"
        super().__init__(
            f"TFLite subprocess {reason} for {self.model_path}. "
            f"stderr tail: {self.stderr_tail or '<empty>'}"
        )


def _tflite_subprocess_env() -> dict[str, str]:
    """Build an environment that can import ``crest`` in subprocesses.

    Returns
    -------
    dict[str, str]
        Environment mapping with the repository ``src`` directory prepended to
        ``PYTHONPATH`` and GPU visibility disabled for TensorFlow imports.
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    src_root = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([src_root, existing_pythonpath])
    else:
        env["PYTHONPATH"] = src_root
    return env


def _stderr_tail(stderr: object, *, max_chars: int = 4000) -> str:
    """Normalize captured stderr to a bounded diagnostic string.

    Parameters
    ----------
    stderr : object
        Captured stderr value from ``subprocess``.
    max_chars : int, optional
        Maximum number of trailing characters to keep.

    Returns
    -------
    str
        Normalized stderr tail.
    """
    if stderr is None:
        return ""
    if isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    else:
        text = str(stderr)
    return text[-max_chars:]

def _probe_xxd() -> Optional[str]:
    """Return the resolved ``xxd`` path when available.

    Returns
    -------
    str | None
        Resolved ``xxd`` executable path when the binary exists and responds
        to a simple version probe, otherwise ``None``.
    """
    candidate = shutil.which("xxd")
    if not candidate:
        return None
    try:
        # Some xxd builds (e.g., macOS 14) exit with status 1 for -h, so use
        # -v which cleanly reports the version and returns 0 when executable.
        subprocess.run(
            [candidate, "-v"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return candidate


def _xxd_supports_custom_names(xxd_path: Optional[str]) -> bool:
    """Detect whether the host ``xxd`` binary accepts the ``-n`` flag.

    Parameters
    ----------
    xxd_path : str | None
        Candidate ``xxd`` executable path.

    Returns
    -------
    bool
        ``True`` when ``xxd -i -n ...`` succeeds on the current host.
    """
    if not xxd_path:
        return False
    temp_file: Optional[tempfile.NamedTemporaryFile] = None
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(b"\x00")
        temp_file.flush()
        temp_file.close()
        subprocess.run(
            [xxd_path, "-i", "-n", "probe_symbol", temp_file.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        if temp_file is not None:
            Path(temp_file.name).unlink(missing_ok=True)
    return True


XXD_BIN = _probe_xxd()
if XXD_BIN and not _xxd_supports_custom_names(XXD_BIN):
    XXD_BIN = None
if not XXD_BIN:
    logger.warning("xxd not found on PATH or doesn't support names; convert_to_cpp_model will use Python fallback.")

# -----------------------------------------------------------------------------
# Conversion helpers
# -----------------------------------------------------------------------------
def convert_to_tflite_model(
    model: tf.keras.Model,
    training_data=None,
    quantization_mode: str = "float",
    output_name: Union[str, Path] = "g_model.tflite",
) -> None:
    """Export a Keras model to a TensorFlow Lite flatbuffer.

    Parameters
    ----------
    model : tf.keras.Model
        Source Keras model to serialize.
    training_data : array-like, optional
        Calibration samples used when ``quantization_mode`` is ``"int8_ptq"``.
        The array must include a sample axis because representative batches are
        emitted one sample at a time.
    quantization_mode : str, optional
        Deployment export mode. ``"float"`` emits a float32 model and
        ``"int8_ptq"`` applies full-integer post-training quantization.
    output_name : Union[str, Path], optional
        Destination filename for the flatbuffer.

    Returns
    -------
    None

    Notes
    -----
    The representative dataset is capped at an evenly sampled subset of the
    provided calibration data. Input and output dtypes stay ``float32`` unless
    ``int8_ptq`` is selected.

    Raises
    ------
    ValueError
        If existing validation or execution checks fail.
    """
    output_path = Path(output_name)
    normalized_mode = str(quantization_mode).strip().lower()
    if normalized_mode not in VALID_TFLITE_QUANTIZATION_MODES:
        raise ValueError("quantization_mode must be one of: float, int8_ptq.")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    if normalized_mode == "int8_ptq":
        if training_data is None:
            raise ValueError("int8_ptq export requires representative calibration data.")
        data = np.asarray(training_data, dtype=np.float32)
        if data.ndim < 2:
            raise ValueError("`training_data` must include a sample dimension.")
        if len(data) == 0:
            raise ValueError("`training_data` must include at least one calibration sample.")
        if not np.all(np.isfinite(data)):
            raise ValueError("`training_data` calibration samples must be finite.")

        representative_samples = _representative_calibration_samples(data)

        def representative_dataset() -> Iterable[Sequence[tf.Tensor]]:
            """Yield calibration samples for post-training quantization.

            Yields
            ------
            list[tf.Tensor]
                Single-sample batches formatted for the TFLite converter.
            """
            for sample in representative_samples:
                # Yield calibrated batches so the converter can determine proper scale/zero-point.
                yield [tf.convert_to_tensor(sample[np.newaxis, ...], tf.float32)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    flatbuffer = converter.convert()
    # Persist the flatbuffer so the downstream conversion step can embed it.
    output_path.write_bytes(flatbuffer)


def _representative_calibration_samples(data: np.ndarray) -> np.ndarray:
    """Select a bounded, spread-out representative calibration subset.

    Parameters
    ----------
    data : numpy.ndarray
        Model-ready calibration inputs with a leading sample axis.

    Returns
    -------
    numpy.ndarray
        At most ``MAX_REPRESENTATIVE_EXAMPLES`` samples. When the input split is
        larger than the cap, examples are selected evenly across the split
        instead of taking one contiguous prefix.
    """
    if len(data) <= MAX_REPRESENTATIVE_EXAMPLES:
        return data
    indices = np.linspace(
        0,
        len(data) - 1,
        num=MAX_REPRESENTATIVE_EXAMPLES,
        dtype=np.int64,
    )
    return data[indices]


def _quantize_tflite_tensor(value: np.ndarray, tensor_detail: dict[str, object]) -> np.ndarray:
    """Quantize one input tensor using interpreter-provided parameters.

    Parameters
    ----------
    value : numpy.ndarray
        Float32 tensor batch to feed into the interpreter.
    tensor_detail : dict[str, object]
        Entry from ``interpreter.get_input_details()``.

    Returns
    -------
    numpy.ndarray
        Tensor cast or quantized to the interpreter input dtype.
    """
    dtype = tensor_detail["dtype"]
    if dtype in (np.float32, np.float64):
        return value.astype(dtype)
    scale, zero_point = tensor_detail.get("quantization", (0.0, 0))
    if not scale:
        return value.astype(dtype)
    quantized = np.round(value / float(scale) + int(zero_point))
    info = np.iinfo(dtype)
    return np.clip(quantized, info.min, info.max).astype(dtype)


def _dequantize_tflite_tensor(value: np.ndarray, tensor_detail: dict[str, object]) -> np.ndarray:
    """Dequantize one output tensor using interpreter-provided parameters.

    Parameters
    ----------
    value : numpy.ndarray
        Raw tensor returned by the interpreter.
    tensor_detail : dict[str, object]
        Entry from ``interpreter.get_output_details()``.

    Returns
    -------
    numpy.ndarray
        Float32 output tensor.
    """
    dtype = tensor_detail["dtype"]
    if dtype in (np.float32, np.float64):
        return np.asarray(value, dtype=np.float32)
    scale, zero_point = tensor_detail.get("quantization", (0.0, 0))
    if not scale:
        return np.asarray(value, dtype=np.float32)
    return (np.asarray(value, dtype=np.float32) - int(zero_point)) * float(scale)


def _ordered_tflite_output_details(interpreter: tf.lite.Interpreter) -> list[dict[str, object]]:
    """Return output tensor details in the model signature order when available.

    Parameters
    ----------
    interpreter : tf.lite.Interpreter
        Allocated interpreter for the model under evaluation.

    Returns
    -------
    list[dict[str, object]]
        Output tensor details ordered to match Keras ``model.outputs`` for
        models exported with a single TFLite signature. Falls back to the
        interpreter's default order when signature metadata is unavailable.
    """
    output_details = interpreter.get_output_details()
    signature_list = interpreter.get_signature_list()
    if len(signature_list) != 1:
        return output_details

    signature_key = next(iter(signature_list))
    signature_output_names = list(signature_list[signature_key].get("outputs", ()))
    signature_runner = interpreter.get_signature_runner(signature_key)
    signature_outputs = signature_runner.get_output_details()
    if len(signature_outputs) != len(output_details):
        return output_details
    if set(signature_output_names) != set(signature_outputs):
        return output_details

    return [
        dict(signature_outputs[output_name])
        for output_name in signature_output_names
    ]


def predict_tflite_model(
    tflite_path: Union[str, Path],
    inputs,
) -> np.ndarray | list[np.ndarray]:
    """Run host-side TFLite inference over a split input batch.

    Parameters
    ----------
    tflite_path : str | pathlib.Path
        TFLite flatbuffer path.
    inputs : array-like
        Split inputs with a leading sample dimension.

    Returns
    -------
    numpy.ndarray | list[numpy.ndarray]
        Float32 predictions normalized to the task-facing Keras shape. Single
        output models return one array; multi-output models return an ordered
        list of arrays.

    Raises
    ------
    ValueError
        If existing validation or execution checks fail.
    """
    data = np.asarray(inputs, dtype=np.float32)
    if data.ndim < 1:
        raise ValueError("TFLite evaluation inputs must include a sample dimension.")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1:
        raise ValueError("TFLite evaluation supports single-input models only.")
    input_detail = input_details[0]
    sample_shape = tuple(int(dim) for dim in input_detail["shape"][1:])
    if sample_shape and tuple(data.shape[1:]) != sample_shape:
        interpreter.resize_tensor_input(input_detail["index"], [1, *data.shape[1:]])
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = _ordered_tflite_output_details(interpreter)

    outputs: list[list[np.ndarray]] = [[] for _ in output_details]
    for sample in data:
        batch = sample[np.newaxis, ...].astype(np.float32)
        interpreter.set_tensor(
            input_detail["index"],
            _quantize_tflite_tensor(batch, input_detail),
        )
        interpreter.invoke()
        for output_index, output_detail in enumerate(output_details):
            raw_output = interpreter.get_tensor(output_detail["index"])
            outputs[output_index].append(_dequantize_tflite_tensor(raw_output, output_detail))

    merged_outputs = [np.concatenate(parts, axis=0) for parts in outputs]
    if len(merged_outputs) == 1:
        return merged_outputs[0]
    return merged_outputs


def predict_tflite_model_subprocess(
    tflite_path: Union[str, Path],
    inputs,
    timeout_sec: float = 900.0,
) -> np.ndarray | list[np.ndarray]:
    """Run host-side TFLite inference in an isolated Python subprocess.

    Parameters
    ----------
    tflite_path : str | pathlib.Path
        TFLite flatbuffer path.
    inputs : array-like
        Split inputs with a leading sample dimension.
    timeout_sec : float, optional
        Maximum number of seconds to wait for the worker process.

    Returns
    -------
    numpy.ndarray | list[numpy.ndarray]
        Worker predictions. Single-output models return one array; multi-output
        models return an ordered list of arrays.

    Raises
    ------
    TFLiteSubprocessError
        If the worker times out, exits nonzero, or does not write the expected
        output contract.
    """
    model_path = Path(tflite_path)
    data = np.asarray(inputs, dtype=np.float32)
    command: list[str] = []
    with tempfile.TemporaryDirectory(prefix="crest_tflite_") as tmpdir:
        temp_dir = Path(tmpdir)
        inputs_path = temp_dir / "inputs.npz"
        outputs_path = temp_dir / "outputs.npz"
        np.savez(inputs_path, inputs=data)
        command = [
            sys.executable,
            "-m",
            "crest.tflite_predict_worker",
            "--model",
            str(model_path),
            "--inputs",
            str(inputs_path),
            "--outputs",
            str(outputs_path),
        ]
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=_tflite_subprocess_env(),
                timeout=float(timeout_sec),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TFLiteSubprocessError(
                model_path=model_path,
                return_code=None,
                timeout=True,
                stderr_tail=_stderr_tail(exc.stderr),
                command=command,
            ) from exc

        if completed.returncode != 0:
            raise TFLiteSubprocessError(
                model_path=model_path,
                return_code=completed.returncode,
                timeout=False,
                stderr_tail=_stderr_tail(completed.stderr),
                command=command,
            )

        try:
            with np.load(outputs_path, allow_pickle=False) as output_archive:
                num_outputs = int(np.asarray(output_archive["num_outputs"]).item())
                if num_outputs < 1:
                    raise ValueError("TFLite worker wrote no outputs.")
                predictions = [
                    np.asarray(output_archive[f"output_{output_index}"])
                    for output_index in range(num_outputs)
                ]
        except (OSError, KeyError, ValueError) as exc:
            raise TFLiteSubprocessError(
                model_path=model_path,
                return_code=completed.returncode,
                timeout=False,
                stderr_tail=_stderr_tail(completed.stderr),
                command=command,
            ) from exc

    if len(predictions) == 1:
        return predictions[0]
    return predictions


def convert_to_cpp_model(
        tflite_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    array_name: str = "g_model",
    source_name: str = "model.cc",
    header_name: str = "model.h",
) -> Tuple[Path, Path]:
    """Materialize a `.tflite` flatbuffer as C sources for TensorFlow Lite Micro.

    Uses XXD when available, otherwise falls back to a Python implementation.

    Parameters
    ----------
    tflite_path : Union[str, Path]
        Path to the serialized TensorFlow Lite flatbuffer.
    output_dir : Union[str, Path]
        Destination directory for the generated source and header files.
    array_name : str, optional
        Symbol to use for the embedded byte array.
    source_name : str, optional
        Filename for the generated translation unit.
    header_name : str, optional
        Filename for the generated header.

    Returns
    -------
    Tuple[pathlib.Path, pathlib.Path]
        Absolute paths to the generated `model.cc` and `model.h` files.
    """
    # Choose the export path once from the module-level capability probe so
    # repeated conversions do not keep rediscovering `xxd`.
    if XXD_BIN:
        return _convert_to_cpp_model_xxd(
            tflite_path,
            output_dir,
            array_name=array_name,
            source_name=source_name,
            header_name=header_name,
        )
    else:
        return _convert_to_cpp_model_python(
            tflite_path,
            output_dir,
            array_name=array_name,
            source_name=source_name,
            header_name=header_name,
        )


def _convert_to_cpp_model_python(
    tflite_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    array_name: str = "g_model",
    source_name: str = "model.cc",
    header_name: str = "model.h",
) -> Tuple[Path, Path]:
    """Materialize a `.tflite` flatbuffer as C sources for TensorFlow Lite Micro.

    Acts as a fallback when `xxd -i` is missing or the host version does not
    implement the `-n` flag (macOS 14). This guarantees the build tooling works
    on any developer machine but comes with two tradeoffs: conversion speed is
    slower than piping through `xxd`, and the emitted layout may differ slightly
    from the canonical TF Micro examples (e.g., indentation/line width).

    Parameters
    ----------
    tflite_path : Union[str, Path]
        Path to the serialized TensorFlow Lite flatbuffer.
    output_dir : Union[str, Path]
        Destination directory for the generated source and header files.
    array_name : str, optional
        Symbol to use for the embedded byte array.
    source_name : str, optional
        Filename for the generated translation unit.
    header_name : str, optional
        Filename for the generated header.

    Returns
    -------
    Tuple[pathlib.Path, pathlib.Path]
        Absolute paths to the generated `model.cc` and `model.h` files.

    Raises
    ------
    FileNotFoundError
        If existing validation or execution checks fail.
    """
    tflite_path = Path(tflite_path)
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_bytes = tflite_path.read_bytes()
    model_len = len(model_bytes)

    bytes_per_line = 12
    hex_lines = []
    # Emit the flatbuffer as a hex array with predictable row width for readability.
    for index in range(0, model_len, bytes_per_line):
        chunk = model_bytes[index: index + bytes_per_line]
        hex_line = ", ".join(f"0x{value:02x}" for value in chunk)
        hex_lines.append(f"  {hex_line},")
    if hex_lines:
        hex_lines[-1] = hex_lines[-1].rstrip(",")  # remove trailing comma on final line

    body = "\n".join(hex_lines)
    source = (
        f'#include "{header_name}"\n\n'
        f"alignas(8) const unsigned char {array_name}[] = {{\n"
        f"{body}\n"
        "};\n"
        f"const int {array_name}_len = {model_len};\n"
    )

    header = [
        "#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_\n",
        "#define TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_\n",
        f"extern const unsigned char {array_name}[];\n",
        f"extern const int {array_name}_len;\n",
        "#endif\n",
    ]

    source_path = output_dir / source_name
    header_path = output_dir / header_name
    source_path.write_text(source)
    header_path.write_text("".join(header))
    return source_path.resolve(), header_path.resolve()

def _convert_to_cpp_model_xxd(
    tflite_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    array_name: str = "g_model",
    source_name: str = "model.cc",
    header_name: str = "model.h",
) -> Tuple[Path, Path]:
    """Generate C sources by delegating to the `xxd -i` command-line tool.

    Parameters
    ----------
    tflite_path : Union[str, Path]
        Path to the `.tflite` model.
    output_dir : Union[str, Path]
        Destination directory for `model.cc`/`model.h`.
    array_name : str, optional
        Symbol to use for the generated array.
    source_name : str, optional
        Output translation unit filename.
    header_name : str, optional
        Output header filename.

    Returns
    -------
    Tuple[pathlib.Path, pathlib.Path]
        Absolute paths to the generated source and header files.

    Raises
    ------
    FileNotFoundError
        If existing validation or execution checks fail.
    """
    # Use a temporary file so the raw `xxd` output can be normalized to the
    # TensorFlow Lite Micro style expected by downstream sources.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(tflite_path)
    if not model_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".cc", delete=False
    ) as temp_source:
        subprocess.run(
            ["xxd", "-i", "-n", array_name, str(model_path)],
            stdout=temp_source,
            check=True,
        )
        temp_source_path = Path(temp_source.name)

    lines = temp_source_path.read_text().splitlines(True)
    lines.insert(0, f'#include "{header_name}"\n')
    lines = [w.replace("unsigned int", "const int") for w in lines]
    lines = [w.replace("unsigned char", "alignas(8) const unsigned char") for w in lines]

    source_path = output_dir / source_name
    header_path = output_dir / header_name
    source_path.write_text("".join(lines))

    header = [
        "#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_\n",
        "#define TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_\n",
        f"extern const unsigned char {array_name}[];\n",
        f"extern const int {array_name}_len;\n",
        "#endif\n",
    ]
    header_path.write_text("".join(header))

    temp_source_path.unlink(missing_ok=True)
    return source_path.resolve(), header_path.resolve()


# -----------------------------------------------------------------------------
# Hardware metadata accessors
# -----------------------------------------------------------------------------
def return_hardware_specs(
    device_name: str,
    device_options: Optional[Dict[str, str]] = None,
) -> Tuple[int, int]:
    """Retrieve RAM and flash limits for a supported device.

    Parameters
    ----------
    device_name : str
        Identifier present in DEVICE_SPECS.
    device_options : dict[str, str] | None, optional
        Optional board-specific options used to resolve dynamic limits (for
        example Portenta core split).

    Returns
    -------
    Tuple[int, int]
        Maximum RAM bytes and flash bytes allowed on the device.

    Notes
    -----
    ``device_name`` is normalized to uppercase before lookup. Dynamic boards
    such as Portenta H7 resolve limits through their backend wrapper when
    ``device_options`` are provided.

    Raises
    ------
    ValueError
        If existing validation or execution checks fail.
    """
    normalized_name = str(device_name).strip().upper()
    if normalized_name == "PORTENTA_H7" and not device_options:
        raise ValueError(
            "PORTENTA_H7 requires device_options with at least target_core "
            "(for example {'target_core': 'cm7'})."
        )
    if device_options:
        spec = get_microcontroller_device(
            normalized_name,
            device_options=device_options,
        ).spec
        return int(spec.max_ram_bytes), int(spec.max_flash_bytes)
    if normalized_name not in DEVICE_SPECS:
        # Direct backend imports can defer registry population during a cycle.
        # Reuse the catalog's lazy refresh, preserving custom/legacy entries.
        get_device_spec(normalized_name)
    spec = DEVICE_SPECS[normalized_name]
    return int(spec["max_ram"]), int(spec["max_flash"])


def get_model_memory_usage(
    batch_size,
    model,
    dtype_bytes: Optional[float] = None,
    quantized: bool = False,
):
    """Estimate the memory usage of a Keras model in bytes.

    Parameters
    ----------
    batch_size : int
        Batch size for inference.
    model : tf.keras.Model
        The Keras model to analyze.
    dtype_bytes : float, optional
        Override the bytes consumed per scalar value.
    quantized : bool, optional
        Treat tensors as int8 when ``dtype_bytes`` is not supplied.

    Returns
    -------
    float
        Total estimated memory usage in bytes.

    Notes
    -----
    This is an estimation helper based on layer output shapes and parameter
    counts. It is useful for relative comparisons, but it is not an allocator-
    accurate replacement for real device-side memory measurements.
    """
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == 'Model':
            # Recursively calculate memory for nested models.
            internal_model_mem_count += get_model_memory_usage(
                batch_size,
                layer,
                dtype_bytes=dtype_bytes,
                quantized=quantized,
            )
        single_layer_mem = 1
        out_shape = getattr(layer, 'output_shape', None)
        if out_shape is None:
            # Some layers (for example newer InputLayer variants) do not expose
            # output_shape, so skip them rather than guessing activation size.
            continue
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            # Multiply dimensions to get total elements per layer output.
            single_layer_mem *= s
        # Accumulate total elements across all layers.
        shapes_mem_count += single_layer_mem

    # Count trainable parameters using TF 2.x backend.
    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    # Count non-trainable parameters using TF 2.x backend.
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    # Determine byte size based on Keras float precision.
    if dtype_bytes is not None:
        number_size = float(dtype_bytes)
    else:
        if quantized:
            number_size = 1.0
        else:
            number_size = 4.0
            if K.floatx() == 'float16':
                number_size = 2.0
            if K.floatx() == 'float64':
                number_size = 8.0

    # Calculate total memory: activations + parameters.
    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    bytes_size = (total_memory + internal_model_mem_count)
    return bytes_size


def arena_size_candidates(
    device_name: str,
    device_options: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Return the tensor-arena sweep (in kilobytes) for a device.

    Parameters
    ----------
    device_name : str
        Identifier present in DEVICE_SPECS.
    device_options : dict[str, str] | None, optional
        Optional board-specific options used to resolve dynamic arena sweeps.

    Returns
    -------
    numpy.ndarray
        Candidate arena sizes expressed in KiB.

    Raises
    ------
    ValueError
        If existing validation or execution checks fail.
    """
    normalized_name = str(device_name).strip().upper()
    if normalized_name == "PORTENTA_H7" and not device_options:
        raise ValueError(
            "PORTENTA_H7 requires device_options with at least target_core "
            "(for example {'target_core': 'cm7'})."
        )
    if device_options:
        spec = get_microcontroller_device(
            normalized_name,
            device_options=device_options,
        ).spec
        return np.array([int(value) for value in spec.arena_sizes_kb])
    if normalized_name not in DEVICE_SPECS:
        # Direct backend imports can defer registry population during a cycle.
        # Reuse the catalog's lazy refresh, preserving custom/legacy entries.
        get_device_spec(normalized_name)
    spec = DEVICE_SPECS[normalized_name]
    return np.array(spec["arena_sizes"])


# -----------------------------------------------------------------------------
# Hardware-in-the-loop preparation
# -----------------------------------------------------------------------------
_HIL_ERROR_LABELS = {
    HIL_ERROR_OK: "HIL_ERROR_OK",
    HIL_ERROR_COMPILE: "HIL_ERROR_COMPILE",
    HIL_ERROR_LATENCY: "HIL_ERROR_LATENCY",
    HIL_ERROR_UNDER_SIZED: "HIL_ERROR_UNDER_SIZED",
    HIL_ERROR_FLASH_OVERFLOW: "HIL_ERROR_FLASH_OVERFLOW",
    HIL_ERROR_RAM_OVERFLOW: "HIL_ERROR_RAM_OVERFLOW",
    HIL_ERROR_UPLOAD: "HIL_ERROR_UPLOAD",
    HIL_ERROR_INTEGRITY: "HIL_ERROR_INTEGRITY",
}

_HIL_MASTER_ERROR_LABELS = {
    HIL_MASTER_PENDING: "HIL_MASTER_PENDING",
    HIL_MASTER_SUCCESS: "HIL_MASTER_SUCCESS",
    HIL_MASTER_ARENA_EXHAUSTED: "HIL_MASTER_ARENA_EXHAUSTED",
    HIL_MASTER_FATAL: "HIL_MASTER_FATAL",
    HIL_MASTER_FLASH_OVERFLOW: "HIL_MASTER_FLASH_OVERFLOW",
    HIL_MASTER_RAM_OVERFLOW: "HIL_MASTER_RAM_OVERFLOW",
    HIL_MASTER_DEVICE_NOT_FOUND: "HIL_MASTER_DEVICE_NOT_FOUND",
    HIL_MASTER_INFRASTRUCTURE: "HIL_MASTER_INFRASTRUCTURE",
}


def describe_error_code(code: int, *, prefer_master: bool = True) -> str:
    """Return the symbolic name for a HIL or master error code.

    Parameters
    ----------
    code : int
        Shared HIL or controller/master error code.
    prefer_master : bool, optional
        Whether master/controller labels should take precedence when both
        namespaces define the same numeric code.

    Returns
    -------
    str
        Stable symbolic label for ``code``.
    """
    lookup_order = (
        (_HIL_MASTER_ERROR_LABELS, _HIL_ERROR_LABELS)
        if prefer_master
        else (_HIL_ERROR_LABELS, _HIL_MASTER_ERROR_LABELS)
    )
    for table in lookup_order:
        if code in table:
            return table[code]
    return f"UNKNOWN_ERROR_{code}"


RETRY_BACKOFF_SECONDS = 1.0
_retry_hint_bytes: Optional[int] = None


def _store_retry_hint_bytes(value: Optional[int]) -> None:
    """Cache arena retry guidance for the most recent ``HIL_spec`` call.

    Parameters
    ----------
    value : int | None
        Suggested next arena size in bytes, when the backend can infer one.
    """
    global _retry_hint_bytes
    _retry_hint_bytes = value


def _pop_retry_hint_bytes() -> Optional[int]:
    """Fetch and clear the most recent arena retry hint.

    Returns
    -------
    int | None
        Suggested next arena size in bytes, or ``None`` when no hint was
        recorded.
    """
    global _retry_hint_bytes
    value = _retry_hint_bytes
    _retry_hint_bytes = None
    return value

def HIL_spec(
    dirpath: Union[str, Path] = 'odom_tcn/',
    chosen_device: str = 'ARDUINO_NANO_33_BLE_SENSE',
    device_options: Optional[Dict[str, str]] = None,
    arenaSizes: Optional[Sequence[int]] = None,
    idx: int = 0,
    window_size: int = 400,
    number_of_channels: int = 6,
    serial_port: Optional[str] = None,
    baud_rate: int = 115200, # potentially highest baud rate for BLE 33
    serial_timeout_s: float = 12.0,
    measured_inference_runs: int = 10,
    dut_ready_timeout_s: Optional[float] = None,
    harness_serial_port: Optional[str] = None,
    harness_fqbn: Optional[str] = None,
    harness_auto_flash: Optional[str] = None,
    harness_arm_pin: Optional[int] = None,
    harness_trigger_pin: Optional[int] = None,
    dut_arm_hold_ms: Optional[int] = None,
    harness_stable_low_ms: Optional[int] = None,
    harness_ready_timeout_s: Optional[float] = None,
    harness_arm_timeout_s: Optional[float] = None,
    harness_active_timeout_s: Optional[float] = None,
    harness_done_timeout_s: Optional[float] = None,
    compile_only: bool = False,
    device: Optional[DeviceInterface] = None,
) -> Tuple[int, int, float, int, int, Optional[Dict[str, Optional[float]]]]:
    """Compile, deploy, and optionally profile CREST on the target hardware.

    When ``compile_only`` is True the function stops after compilation so the
    caller can reuse the RAM/flash measurements without requiring a physical
    board. This keeps the objective function agnostic to whether HIL is
    connected while still sourcing metrics from the toolchain.

    Parameters
    ----------
    dirpath : Union[str, Path], optional
        Firmware project directory containing the CREST sources.
    chosen_device : str, optional
        Hardware identifier that maps into DEVICE_SPECS.
    device_options : dict[str, str] | None, optional
        Optional board-specific options forwarded to the device factory.
    arenaSizes : Sequence[int], optional
        Custom arena sweep in KiB; defaults to the catalog entry.
    idx : int, optional
        Position inside the arena sweep to test.
    window_size : int, optional
        Sliding window length supplied to the firmware.
    number_of_channels : int, optional
        Number of sensor channels per window.
    serial_port : str, optional
        Serial port used for upload and latency capture.
    baud_rate : int, optional
        Serial baud rate for latency capture.
    serial_timeout_s : float, optional
        Seconds to wait for the `timer output:` line.
    dut_ready_timeout_s : float, optional
        Seconds to wait for DUT READY before sending START.
    compile_only : bool, optional
        Skip upload/latency capture and return compile metrics only.
    device : DeviceInterface | None, optional
        Optional device implementation to use for compile/upload/measure.

    Returns
    -------
    Tuple[int, int, float, int, int, Optional[Dict[str, Optional[float]]]]
        Tuple of (RAM bytes, flash bytes, latency seconds, arena bytes, error flag,
        optional power telemetry parsed from the serial log).

    Raises
    ------
    FileNotFoundError
        If existing validation or execution checks fail.
    ValueError
        If existing validation or execution checks fail.
    IndexError
        If existing validation or execution checks fail.
    """
    # Retry hints are a side channel from one attempt back to
    # ``HIL_controller`` so stale advice must be cleared before every run.
    _store_retry_hint_bytes(None)
    requested_device = chosen_device
    if device is not None:
        # An injected backend object already resolved the authoritative device
        # identity, so prefer its spec over the caller's string.
        chosen_device = device.spec.name
    logger.info(
        "HIL_spec: start attempt (requested_device=%s, resolved_device=%s, idx=%d, run_hil=%s, device_options=%s)",
        requested_device,
        chosen_device,
        idx,
        not compile_only,
        device_options,
    )
    if device is None:
        device = get_microcontroller_device(
            chosen_device,
            serial_port=serial_port,
            device_options=device_options,
        )
    spec = device.spec

    # Resolve the sketch path up-front so all subsequent operations use absolute paths.
    sketch_path = Path(dirpath).resolve()
    if not sketch_path.exists():
        raise FileNotFoundError(f"Sketch directory not found: {sketch_path}")

    # Mirror the original HIL sweep: choose a single arena candidate for this attempt.
    arena_sweep_list = (
        list(arenaSizes)
        if arenaSizes is not None
        else [int(value) for value in spec.arena_sizes_kb]
    )
    if not arena_sweep_list:
        raise ValueError(f"No arena sizes registered for {chosen_device}.")
    if idx < 0 or idx >= len(arena_sweep_list):
        raise IndexError(f"arenaSizes index {idx} out of range for device {chosen_device}.")
    arena_kb = int(arena_sweep_list[idx])

    metrics = device.evaluate(
        dirpath=sketch_path,
        arena_kb=arena_kb,
        window_size=window_size,
        num_channels=number_of_channels,
        serial_port=serial_port,
        run_hil=not compile_only,
        baud_rate=baud_rate,
        serial_timeout_s=serial_timeout_s,
        measured_inference_runs=measured_inference_runs,
        dut_ready_timeout_s=dut_ready_timeout_s,
        harness_serial_port=harness_serial_port,
        harness_fqbn=harness_fqbn,
        harness_auto_flash=harness_auto_flash,
        harness_arm_pin=harness_arm_pin,
        harness_trigger_pin=harness_trigger_pin,
        dut_arm_hold_ms=dut_arm_hold_ms,
        harness_stable_low_ms=harness_stable_low_ms,
        harness_ready_timeout_s=harness_ready_timeout_s,
        harness_arm_timeout_s=harness_arm_timeout_s,
        harness_active_timeout_s=harness_active_timeout_s,
        harness_done_timeout_s=harness_done_timeout_s,
    )
    logger.info(
        "HIL_spec: evaluate complete (ram=%d, flash=%d, latency=%s, err=%d)",
        metrics.ram_bytes,
        metrics.flash_bytes,
        metrics.latency_s,
        metrics.error_code,
    )
    _store_retry_hint_bytes(metrics.retry_hint_bytes)
    logger.info(
        "Latency capture result: latency_s=%s",
        metrics.latency_s if metrics.latency_s >= 0 else "None",
    )
    return (
        metrics.ram_bytes,
        metrics.flash_bytes,
        metrics.latency_s,
        metrics.arena_bytes,
        metrics.error_code,
        metrics.power_metrics,
    )


def HIL_controller(
    dirpath: Union[str, Path] = 'odom_tcn/',
    chosen_device: str = 'ARDUINO_NANO_33_BLE_SENSE',
    device_options: Optional[Dict[str, str]] = None,
    window_size: int = 400,
    number_of_channels: int = 6,
    serial_port: Optional[str] = None,
    baud_rate: int = 115200,
    serial_timeout_s: float = 12.0,
    measured_inference_runs: int = 10,
    run_hil: bool = True,
    dut_ready_timeout_s: Optional[float] = None,
    harness_serial_port: Optional[str] = None,
    harness_fqbn: Optional[str] = None,
    harness_auto_flash: Optional[str] = None,
    harness_arm_pin: Optional[int] = None,
    harness_trigger_pin: Optional[int] = None,
    dut_arm_hold_ms: Optional[int] = None,
    harness_stable_low_ms: Optional[int] = None,
    harness_ready_timeout_s: Optional[float] = None,
    harness_arm_timeout_s: Optional[float] = None,
    harness_active_timeout_s: Optional[float] = None,
    harness_done_timeout_s: Optional[float] = None,
    device: Optional[DeviceInterface] = None,
) -> Tuple[
    int,
    int,
    float,
    int,
    int,
    Optional[Dict[str, Optional[float]]],
]:
    """Search for the smallest arena size that compiles and runs successfully.

    ``run_hil`` toggles whether uploads/latency capture occur; when False the
    controller enters compile-only mode so offline Optuna trials can still rely
    on compiler-derived RAM/flash numbers for scoring.

    Parameters
    ----------
    dirpath : Union[str, Path], optional
        Firmware project directory containing CREST sources.
    chosen_device : str, optional
        Hardware identifier that maps into DEVICE_SPECS.
    device_options : dict[str, str] | None, optional
        Optional board-specific options forwarded to the device factory.
    window_size : int, optional
        Sliding window length supplied to the firmware.
    number_of_channels : int, optional
        Number of sensor channels per window.
    serial_port : str, optional
        Serial port used for upload and latency capture.
    baud_rate : int, optional
        Serial baud rate for latency capture.
    serial_timeout_s : float, optional
        Timeout when waiting for the `timer output:` line.
    dut_ready_timeout_s : float, optional
        Seconds to wait for DUT READY before sending START.
    device : DeviceInterface | None, optional
        Optional device implementation to use for compile/upload/measure.

    Returns
    -------
    Tuple[int, int, float, int, int, Optional[Dict[str, Optional[float]]]]
        Final RAM bytes, flash bytes, latency seconds, arena bytes, error code, and
        optional power telemetry captured from the winning firmware run.

    Notes
    -----
    The search maintains an open interval where ``low_idx`` is the largest
    known failing arena and ``high_idx`` is the smallest known successful
    arena. Successful runs search downward for a smaller arena, undersized or
    latency failures search upward, and RAM overflow shrinks the upper bound.
    When a backend already produced one successful run, that last-success
    metric set is retained even if a later probe exits with a non-success
    master code.
    """
    if device is None:
        device = get_microcontroller_device(
            chosen_device,
            serial_port=serial_port,
            device_options=device_options,
        )
    chosen_device = device.spec.name
    # When a device object exists, its resolved spec is the source of truth.
    arena_sweep_list = [int(value) for value in device.spec.arena_sizes_kb]
    finRAM = -1
    finFlash = -1
    finLatency = -1.0
    idealArenaBytes = -1
    masterError = HIL_MASTER_PENDING
    last_success_metrics: Optional[Tuple[int, int, float, int, Optional[Dict[str, Optional[float]]]]] = None
    had_retry_failures = False

    # Start with the full arena sweep interval open: no known failing lower
    # bound and no known successful upper bound yet.
    low_idx = -1
    high_idx = len(arena_sweep_list)

    def _next_candidate(low: int, high: int, preferred: int) -> Optional[int]:
        """Clamp a preferred search index to the current open interval.

        Parameters
        ----------
        low : int
            Inclusive lower bound index known to fail.
        high : int
            Exclusive upper bound index known to succeed.
        preferred : int
            Candidate index to try next.

        Returns
        -------
        int | None
            Next valid candidate index, or ``None`` when the interval is empty.
        """
        lower_bound = low + 1
        upper_bound = high - 1
        if lower_bound > upper_bound:
            return None
        return max(lower_bound, min(upper_bound, preferred))

    next_idx = _next_candidate(low_idx, high_idx, (low_idx + high_idx) // 2)
    iteration_count = 0
    max_iterations = max(1, len(arena_sweep_list) * 3)
    tested_bounds: dict[int, Tuple[int, int]] = {}
    finPower_metrics: Optional[Dict[str, Optional[float]]] = None

    compile_only = not run_hil  # compile-only allows proxy runs to reuse compiler metrics
    while (
        masterError == HIL_MASTER_PENDING
        and low_idx + 1 < high_idx
        and next_idx is not None
    ):
        iteration_count += 1
        if iteration_count > max_iterations:
            logger.error(
                "HIL_controller exceeded max iterations (%d); aborting to avoid infinite loop.",
                max_iterations,
            )
            masterError = HIL_MASTER_FATAL
            break

        current_idx = next_idx
        bounds_signature = (low_idx, high_idx)
        previous_bounds = tested_bounds.get(current_idx)
        # Guard against a backend repeatedly returning the same hint or branch
        # outcome without shrinking the search bracket.
        if previous_bounds == bounds_signature:
            logger.error(
                "HIL_controller detected repeated idx=%d without bracket shrink (low=%d high=%d); aborting sweep.",
                current_idx,
                low_idx,
                high_idx,
            )
            masterError = HIL_MASTER_FATAL
            break
        tested_bounds[current_idx] = bounds_signature

        (
            ram_bytes,
            flash_bytes,
            latency_s,
            arena_bytes,
            err_flag,
            power_metrics,
        ) = HIL_spec(
            dirpath=dirpath,
            chosen_device=chosen_device,
            device_options=device_options,
            arenaSizes=arena_sweep_list,
            idx=current_idx,
            window_size=window_size,
            number_of_channels=number_of_channels,
            serial_port=serial_port,
            baud_rate=baud_rate,
            serial_timeout_s=serial_timeout_s,
            measured_inference_runs=measured_inference_runs,
            dut_ready_timeout_s=dut_ready_timeout_s,
            harness_serial_port=harness_serial_port,
            harness_fqbn=harness_fqbn,
            harness_auto_flash=harness_auto_flash,
            harness_arm_pin=harness_arm_pin,
            harness_trigger_pin=harness_trigger_pin,
            dut_arm_hold_ms=dut_arm_hold_ms,
            harness_stable_low_ms=harness_stable_low_ms,
            harness_ready_timeout_s=harness_ready_timeout_s,
            harness_arm_timeout_s=harness_arm_timeout_s,
            harness_active_timeout_s=harness_active_timeout_s,
            harness_done_timeout_s=harness_done_timeout_s,
            compile_only=compile_only,
            device=device,
        )
        retry_hint_bytes = _pop_retry_hint_bytes()

        logger.info(
            "HIL_controller attempt idx=%d arena=%d KiB err_flag=%d",
            current_idx,
            arena_sweep_list[current_idx],
            err_flag,
        )

        if err_flag != HIL_ERROR_OK:
            # During arena search, latency/undersized failures are expected stepping stones.
            # Keep these as INFO so users don't misread successful sweeps as hard failures.
            log_fn = logger.info if err_flag in (HIL_ERROR_LATENCY, HIL_ERROR_UNDER_SIZED) else logger.warning
            log_fn(
                "HIL_controller failure reason: %s (err_flag=%d, arena=%d KiB, ram=%d bytes, flash=%d bytes, latency=%.3f s)",
                describe_error_code(err_flag, prefer_master=False),
                err_flag,
                arena_sweep_list[current_idx],
                ram_bytes,
                flash_bytes,
                latency_s,
            )

        if err_flag == HIL_ERROR_OK:
            # Successful run: capture metrics and stop searching.
            finRAM = ram_bytes
            finFlash = flash_bytes
            finLatency = latency_s
            idealArenaBytes = arena_bytes
            finPower_metrics = power_metrics
            last_success_metrics = (finRAM, finFlash, finLatency, idealArenaBytes, finPower_metrics)
            high_idx = current_idx
            candidate = current_idx - 1
            next_idx = _next_candidate(low_idx, high_idx, candidate)
            if next_idx is None:
                masterError = HIL_MASTER_SUCCESS
            continue
        elif err_flag in (HIL_ERROR_LATENCY, HIL_ERROR_UNDER_SIZED):
            # Arena too small; advance to the next candidate.
            had_retry_failures = True
            low_idx = max(low_idx, current_idx)
            candidate = min(current_idx + 1, (low_idx + high_idx) // 2)
            if retry_hint_bytes is not None:
                # Backends may report the first arena size likely to fit; jump
                # forward to that point when it narrows the sweep faster.
                target_idx = next(
                    (i for i, kb in enumerate(arena_sweep_list) if kb * 1024 >= retry_hint_bytes),
                    None,
                )
                if target_idx is not None and target_idx > current_idx:
                    logger.info(
                        "Arena retry hint suggests jumping to idx=%d size=%d KiB (target_bytes=%d)",
                        target_idx,
                        arena_sweep_list[target_idx],
                        retry_hint_bytes,
                    )
                    candidate = max(candidate, target_idx)
            next_idx = _next_candidate(low_idx, high_idx, candidate)
            if next_idx is None:
                if last_success_metrics:
                    masterError = HIL_MASTER_SUCCESS
                elif len(arena_sweep_list) == 1 and err_flag == HIL_ERROR_LATENCY:
                    # STM32 uses a documented single-shot arena sentinel. A
                    # runtime timeout on that path is a real runtime failure,
                    # not arena exhaustion.
                    finRAM = ram_bytes
                    finFlash = flash_bytes
                    finLatency = latency_s
                    idealArenaBytes = arena_bytes
                    finPower_metrics = power_metrics
                    masterError = HIL_MASTER_FATAL
                else:
                    masterError = HIL_MASTER_ARENA_EXHAUSTED
            else:
                time.sleep(RETRY_BACKOFF_SECONDS)
            continue
        elif err_flag == HIL_ERROR_RAM_OVERFLOW:
            if current_idx == 0 and last_success_metrics is None:
                # Already at the smallest arena; surface RAM overflow upstream.     
                finRAM = ram_bytes
                finFlash = flash_bytes
                finLatency = latency_s
                idealArenaBytes = arena_bytes
                finPower_metrics = power_metrics
                masterError = HIL_MASTER_RAM_OVERFLOW
                break
            high_idx = min(high_idx, current_idx)
            next_idx = _next_candidate(low_idx, high_idx, (low_idx + high_idx) // 2)
            if next_idx is None:
                masterError = HIL_MASTER_SUCCESS if last_success_metrics else HIL_MASTER_ARENA_EXHAUSTED
            continue
        elif err_flag == HIL_ERROR_FLASH_OVERFLOW:
            # Exceeds flash/RAM limits; signal Optuna to prune.
            finRAM = ram_bytes
            finFlash = flash_bytes
            finLatency = latency_s
            idealArenaBytes = arena_bytes
            finPower_metrics = power_metrics
            masterError = HIL_MASTER_FLASH_OVERFLOW
        elif err_flag == HIL_ERROR_INTEGRITY:
            finRAM, finFlash, finLatency = ram_bytes, flash_bytes, latency_s
            idealArenaBytes, finPower_metrics = arena_bytes, power_metrics
            masterError = HIL_MASTER_INFRASTRUCTURE
            break
        elif err_flag == HIL_ERROR_UPLOAD:
            # Upload failures usually mean the board disappeared; stop the sweep immediately.
            finRAM = ram_bytes
            finFlash = flash_bytes
            finLatency = latency_s
            idealArenaBytes = arena_bytes
            finPower_metrics = power_metrics
            masterError = HIL_MASTER_DEVICE_NOT_FOUND
            break
        else:
            # Non-arena failure (e.g., flash overflow). Surface immediately.
            finRAM = ram_bytes
            finFlash = flash_bytes
            finLatency = latency_s
            idealArenaBytes = arena_bytes
            finPower_metrics = power_metrics
            masterError = HIL_MASTER_FATAL

    if masterError != HIL_MASTER_SUCCESS and last_success_metrics is not None:
        # Preserve the smallest known good arena metrics even if a later probe
        # fails while exploring nearby bounds.
        finRAM, finFlash, finLatency, idealArenaBytes, finPower_metrics = last_success_metrics

    if masterError == HIL_MASTER_PENDING:
        masterError = HIL_MASTER_SUCCESS if last_success_metrics else HIL_MASTER_ARENA_EXHAUSTED

    if masterError == HIL_MASTER_SUCCESS and had_retry_failures:
        logger.info(
            "HIL_controller recovered after arena retries: arena_bytes=%d ram_bytes=%d flash_bytes=%d latency=%.3f s",
            idealArenaBytes,
            finRAM,
            finFlash,
            finLatency,
        )

    if masterError != HIL_MASTER_SUCCESS:
        logger.warning(
            (
                "HIL_controller exiting with master error %s (code=%d, arena_bytes=%d, "
                "ram_bytes=%d, flash_bytes=%d, latency=%.3f s, power_metrics=%s)"
            ),
            describe_error_code(masterError),
            masterError,
            idealArenaBytes,
            finRAM,
            finFlash,
            finLatency,
            finPower_metrics,
        )
    return finRAM, finFlash, finLatency, idealArenaBytes, masterError, finPower_metrics
