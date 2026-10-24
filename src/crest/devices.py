# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Device contracts and Arduino bridge helpers for CREST HIL execution.

This module defines the shared device result types used across HIL backends,
maintains the compatibility device catalog, and provides the temporary
Arduino-backed bridge that still powers several board workflows during the
microcontroller refactor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence

import numpy as np
import logging
import serial
import shutil

from .errors import (
    HIL_ERROR_OK,
    HIL_ERROR_COMPILE,
    HIL_ERROR_LATENCY,
    HIL_ERROR_UNDER_SIZED,
    HIL_ERROR_FLASH_OVERFLOW,
    HIL_ERROR_RAM_OVERFLOW,
    HIL_ERROR_UPLOAD,
)
from . import hil_protocol
from .microcontrollers import arduino_base
from .pipeline_types import DataSplit

logger = logging.getLogger(__name__)
RuntimeMeasureMode = Literal["direct_serial", "harness_only"]


@dataclass(frozen=True)
class DeviceSpec:
    """Static hardware capabilities and constraints.

    Parameters
    ----------
    name : str
        Human-readable device identifier.
    arena_sizes_kb : Sequence[int]
        Candidate tensor arena sizes in KiB.
    max_ram_bytes : int
        Maximum usable RAM on the device in bytes.
    max_flash_bytes : int
        Maximum usable flash on the device in bytes.
    max_external_flash_bytes : int | None, optional
        Maximum usable external flash on the device in bytes, when the backend
        supports separate weight storage outside the primary program image.
        Not used by all boards.
    fqbn : str | None, optional
        Fully qualified board name (Arduino CLI only).
    toolchain : str | None, optional
        Toolchain identifier (e.g., "arduino-cli", "stm32-cube").

    Attributes
    ----------
    name : str
        Human-readable device identifier.
    arena_sizes_kb : Sequence[int]
        Candidate tensor arena sizes in KiB.
    max_ram_bytes : int
        Maximum usable RAM on the device in bytes.
    max_flash_bytes : int
        Maximum usable flash on the device in bytes.
    max_external_flash_bytes : Optional[int]
        Maximum usable external flash on the device in bytes, when the backend
        supports separate weight storage outside the primary program image.
        Not used by all boards.
    fqbn : Optional[str]
        Fully qualified board name (Arduino CLI only).
    toolchain : Optional[str]
        Toolchain identifier (e.g., "arduino-cli", "stm32-cube").
    """

    name: str
    arena_sizes_kb: Sequence[int]
    max_ram_bytes: int
    max_flash_bytes: int
    max_external_flash_bytes: Optional[int] = None
    fqbn: Optional[str] = None
    toolchain: Optional[str] = None


@dataclass(frozen=True)
class DeviceMetrics:
    """Standardized telemetry captured after a HIL or proxy run.

    Parameters
    ----------
    ram_bytes : int
        RAM usage in bytes (or -1 if unavailable).
    flash_bytes : int
        Flash usage in bytes (or -1 if unavailable).
    latency_s : float
        Inference latency in seconds (or -1.0 if unavailable).
    arena_bytes : int
        Tensor arena size used in bytes.
    error_code : int
        Error code emitted by the device evaluation.
    power_metrics : dict[str, Any] | None, optional
        Optional raw power metrics parsed from the serial log.
    external_flash_bytes : int | None, optional
        External flash bytes consumed by staged weight blobs when the backend
        uses separate weight storage. Not used by Arduino boards.
    retry_hint_bytes : int | None, optional
        Suggested arena size in bytes for the next attempt, when available.

    Attributes
    ----------
    ram_bytes : int
        RAM usage in bytes (or -1 if unavailable).
    flash_bytes : int
        Flash usage in bytes (or -1 if unavailable).
    latency_s : float
        Inference latency in seconds (or -1.0 if unavailable).
    arena_bytes : int
        Tensor arena size used in bytes.
    error_code : int
        Error code emitted by the device evaluation.
    power_metrics : Optional[Dict[str, Any]]
        Optional raw power metrics parsed from the serial log.
    external_flash_bytes : Optional[int]
        External flash bytes consumed by staged weight blobs when the backend
        uses separate weight storage. Not used by Arduino boards.
    retry_hint_bytes : Optional[int]
        Suggested arena size in bytes for the next attempt, when available.
    """

    ram_bytes: int
    flash_bytes: int
    latency_s: float
    arena_bytes: int
    error_code: int
    power_metrics: Optional[Dict[str, Any]] = None
    external_flash_bytes: Optional[int] = None
    retry_hint_bytes: Optional[int] = None


@dataclass(frozen=True)
class CompileResult:
    """Result of a device-specific compile step.

    Parameters
    ----------
    success : bool
        True when compilation succeeded.
    log : str
        Combined stdout/stderr from the compile step.
    flash_bytes : int | None
        Parsed flash usage in bytes, if available.
    ram_bytes : int | None
        Parsed RAM usage in bytes, if available.
    overflow_kind : str | None
        "flash" or "ram" when an overflow is detected.
    build_dir : pathlib.Path | None
        Build cache directory if applicable.
    arena_bytes : int | None
        Parsed activation arena size in bytes, if available.
    heap_bytes : int | None
        Linker-reserved heap bytes, if available.
    stack_bytes : int | None
        Linker-reserved stack bytes, if available.
    external_flash_bytes : int | None, optional
        External flash bytes consumed by staged weight blobs when the backend
        uses separate weight storage.
        Not used by all boards.
    boot_build_dir : pathlib.Path | None, optional
        STM32-only Boot-project build directory when the backend produces
        separate boot and application artifacts.
    boot_elf_path : pathlib.Path | None, optional
        STM32-only Boot-project ELF path used for debug loading.
    app_build_dir : pathlib.Path | None, optional
        STM32-only App-project build directory when the backend produces
        separate boot and application artifacts.
    app_elf_path : pathlib.Path | None, optional
        STM32-only App-project ELF path.
    signed_app_bin_path : pathlib.Path | None, optional
        STM32-only trusted application image path emitted after signing.
    fsbl_copy_window_bytes : int | None, optional
        STM32 LRUN copy-window size compiled into the bootloader.

    Attributes
    ----------
    success : bool
        True when compilation succeeded.
    log : str
        Combined stdout/stderr from the compile step.
    flash_bytes : Optional[int]
        Parsed flash usage in bytes, if available.
    ram_bytes : Optional[int]
        Parsed RAM usage in bytes, if available.
    overflow_kind : Optional[str]
        "flash" or "ram" when an overflow is detected.
    build_dir : Optional[Path]
        Build cache directory if applicable.
    arena_bytes : Optional[int]
        Parsed activation arena size in bytes, if available.
    heap_bytes : Optional[int]
        Linker-reserved heap bytes, if available.
    stack_bytes : Optional[int]
        Linker-reserved stack bytes, if available.
    external_flash_bytes : Optional[int]
        External flash bytes consumed by staged weight blobs when the backend
        uses separate weight storage.
        Not used by all boards.
    boot_build_dir : Optional[Path]
        STM32-only Boot-project build directory when the backend produces
        separate boot and application artifacts.
    boot_elf_path : Optional[Path]
        STM32-only Boot-project ELF path used for debug loading.
    app_build_dir : Optional[Path]
        STM32-only App-project build directory when the backend produces
        separate boot and application artifacts.
    app_elf_path : Optional[Path]
        STM32-only App-project ELF path.
    signed_app_bin_path : Optional[Path]
        STM32-only trusted application image path emitted after signing.
    fsbl_copy_window_bytes : Optional[int]
        STM32 LRUN copy-window size compiled into the bootloader.
    """

    success: bool
    log: str
    flash_bytes: Optional[int]
    ram_bytes: Optional[int]
    overflow_kind: Optional[str]
    build_dir: Optional[Path] = None
    arena_bytes: Optional[int] = None
    heap_bytes: Optional[int] = None
    stack_bytes: Optional[int] = None
    external_flash_bytes: Optional[int] = None
    boot_build_dir: Optional[Path] = None
    boot_elf_path: Optional[Path] = None
    app_build_dir: Optional[Path] = None
    app_elf_path: Optional[Path] = None
    signed_app_bin_path: Optional[Path] = None
    fsbl_copy_window_bytes: Optional[int] = None


@dataclass(frozen=True)
class UploadResult:
    """Result of a device-specific upload step.

    Parameters
    ----------
    success : bool
        True when upload succeeded.
    log : str
        Combined stdout/stderr from the upload step.

    Attributes
    ----------
    success : bool
        True when upload succeeded.
    log : str
        Combined stdout/stderr from the upload step.
    """

    success: bool
    log: str


@dataclass(frozen=True)
class MeasureResult:
    """Parsed telemetry from a device measurement pass.

    Parameters
    ----------
    latency_s : float | None
        Inference latency in seconds, if available.
    arena_error_line : str | None
        First arena allocation error line, if found.
    serial_log : list[str]
        Decoded serial log captured during measurement.
    power_metrics : dict[str, float | None] | None
        Parsed power metrics fields, if present.

    Attributes
    ----------
    latency_s : Optional[float]
        Inference latency in seconds, if available.
    arena_error_line : Optional[str]
        First arena allocation error line, if found.
    serial_log : list[str]
        Decoded serial log captured during measurement.
    power_metrics : Optional[Dict[str, Optional[float]]]
        Parsed power metrics fields, if present.
    """

    latency_s: Optional[float]
    arena_error_line: Optional[str]
    serial_log: list[str]
    power_metrics: Optional[Dict[str, Optional[float]]]


@dataclass(frozen=True)
class CandidatePrepareRequest:
    """Typed request passed into backend candidate preparation.

    Parameters
    ----------
    config : Any
        Loaded CREST configuration object.
    model : Any
        Model instance to export and stage for one backend attempt.
    model_variant : str
        Human-readable export variant label.
    artifact_root : pathlib.Path
        Root directory chosen by orchestration for backend-owned artifacts.
    tflite_model_path : pathlib.Path
        Explicit TFLite output path selected by orchestration.
    calibration_split : DataSplit | None
        Calibration split used for representative export data when required.
    quantization_mode : str
        Deployment quantization mode selected for this candidate.
    input_shape : tuple[int, ...] | None
        Logical model input shape excluding the batch dimension.
    checkpoint_path : pathlib.Path | str | None, optional
        Checkpoint path associated with trained export variants when one
        exists.

    Attributes
    ----------
    config : Any
        Loaded CREST configuration object.
    model : Any
        Model instance to export and stage for one backend attempt.
    model_variant : str
        Human-readable export variant label.
    artifact_root : Path
        Root directory chosen by orchestration for backend-owned artifacts.
    tflite_model_path : Path
        Explicit TFLite output path selected by orchestration.
    calibration_split : DataSplit | None
        Calibration split used for representative export data when required.
    quantization_mode : str
        Deployment quantization mode selected for this candidate.
    input_shape : tuple[int, ...] | None
        Logical model input shape excluding the batch dimension.
    checkpoint_path : Path | str | None
        Checkpoint path associated with trained export variants when one
        exists.
    """

    config: Any
    model: Any
    model_variant: str
    artifact_root: Path
    tflite_model_path: Path
    calibration_split: DataSplit | None
    quantization_mode: str
    input_shape: tuple[int, ...] | None
    checkpoint_path: Path | str | None = None


class DeviceInterface(ABC):
    """Contract for device-specific compile/upload/measure workflows."""

    def requires_candidate_model(self) -> bool:
        """Return whether this backend consumes generated model artifacts.

        Returns
        -------
        bool
            ``True`` when the backend needs a generated candidate model before compilation.
        """
        return True

    def requires_training_data(self) -> bool:
        """Return whether candidate preparation needs calibration/training data.

        Returns
        -------
        bool
            ``True`` when calibration or training data must be available during candidate preparation.
        """
        return True

    def requires_arena_validation(self) -> bool:
        """Return whether ``arena_bytes`` should be treated as a required metric.

        Returns
        -------
        bool
            ``True`` when missing arena accounting should invalidate the device result.
        """
        return True

    def supports_energy_measurement(self) -> bool:
        """Return whether this backend can produce real energy metrics.

        Returns
        -------
        bool
            ``True`` when the backend can report measured energy rather than proxy values.
        """
        return True

    def supports_runtime_measurement(self) -> bool:
        """Return whether this backend can run upload/measurement passes.

        Returns
        -------
        bool
            ``True`` when the backend can upload and execute firmware on a target device.
        """
        return True

    def runtime_measure_mode(self) -> RuntimeMeasureMode:
        """Return runtime measurement mode for this device profile.

        Returns
        -------
        RuntimeMeasureMode
            Measurement path used by orchestration for this backend.
        """
        return "direct_serial"

    def runtime_mode_build_defines(self) -> Dict[str, int]:
        """Return additional compile-time defines for the runtime mode.

        Returns
        -------
        Dict[str, int]
            Compile-time defines required by the selected runtime mode.
        """
        return {}

    def prepare_for_runtime(self, *, runtime_mode: RuntimeMeasureMode, serial_port: str) -> None:
        """Run optional board-specific setup before upload/measurement.

        Parameters
        ----------
        runtime_mode : RuntimeMeasureMode
            Runtime profile requested for the backend measurement pass.
        serial_port : str
            Serial device path used to communicate with the DUT or harness.
        """
        del runtime_mode, serial_port

    def prepare_candidate(
        self,
        *,
        request: CandidatePrepareRequest,
    ) -> Path:
        """Prepare backend-owned build inputs and return the compile directory.

        Parameters
        ----------
        request : CandidatePrepareRequest
            Candidate preparation payload assembled by orchestration.

        Returns
        -------
        Path
            Directory that should be passed to the backend compile step.

        Raises
        ------
        NotImplementedError
            If existing validation or execution checks fail.
        """
        del request
        raise NotImplementedError("prepare_candidate() must be implemented by backend classes.")

    def cleanup_prepared_candidate(self, prepared_dir: Path | None) -> None:
        """Release backend-owned candidate artifacts after one evaluation.

        Parameters
        ----------
        prepared_dir : Path | None
            Path previously returned by ``prepare_candidate()``.
        """
        del prepared_dir

    def set_input_mode(
        self,
        input_mode: str,
        *,
        outputs_dir: Path,
        config: Any,
        sketches_dir: Path | None = None,
        runtime_phase: str = "back_to_back",
    ) -> Path | None:
        """Apply an input-mode change when the backend uses that concept.

        Parameters
        ----------
        input_mode : str
            Input-header generation mode selected for the staged sketch.
        outputs_dir : Path
            Directory where generated sketch or backend artifacts are written.
        config : Any
            Loaded CREST configuration object or mapping.
        sketches_dir : Path | None
            Repository sketch root used to locate checked-in Arduino templates.
        runtime_phase : str
            Runtime phase label used to select generated telemetry settings.

        Returns
        -------
        Path | None
            Staged input header path when a backend rewrites one, otherwise
            ``None``.
        """
        del input_mode, outputs_dir, config, sketches_dir, runtime_phase
        return None

    @property
    @abstractmethod
    def spec(self) -> DeviceSpec:
        """Return the device specification metadata.

        Returns
        -------
        DeviceSpec
            Static limits and toolchain metadata for this device.
        """
    @abstractmethod
    def compile(
        self,
        *,
        sketch_path: Path,
        arena_kb: int,
        window_size: int,
        num_channels: int,
        build_defines: Optional[Dict[str, int]] = None,
    ) -> CompileResult:
        """Compile the firmware for the target device.

        Parameters
        ----------
        sketch_path : pathlib.Path
            Path to the firmware project directory.
        arena_kb : int
            Tensor arena size in KiB for this attempt.
        window_size : int
            Sliding window length used by the model.
        num_channels : int
            Number of sensor channels per window.

        Returns
        -------
        CompileResult
            Parsed compile output and resource usage.
        """
    @abstractmethod
    def upload(
        self,
        *,
        sketch_path: Path,
        build_dir: Optional[Path],
        serial_port: Optional[str],
    ) -> UploadResult:
        """Upload compiled firmware to the device.

        Parameters
        ----------
        sketch_path : pathlib.Path
            Path to the firmware project directory.
        build_dir : pathlib.Path | None
            Build directory containing compiled artifacts.
        serial_port : str | None
            Serial port to upload to.

        Returns
        -------
        UploadResult
            Upload success flag and log output.
        """
    @abstractmethod
    def measure(
        self,
        *,
        serial_port: Optional[str],
        baud_rate: int,
        serial_timeout_s: float,
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
    ) -> MeasureResult:
        """Capture latency/power metrics from the device.

        Parameters
        ----------
        serial_port : str | None
            Serial port used for log capture.
        baud_rate : int
            Serial baud rate.
        serial_timeout_s : float
            Timeout for log capture.
        dut_ready_timeout_s : float | None
            Time to wait for DUT READY before sending START.
        harness_serial_port : str | None
            Optional serial port for the INA228 harness.
        harness_fqbn : str | None
            FQBN for the harness board (Arduino CLI only).
        harness_auto_flash : str | None
            Harness flashing policy (once, always, never).
        harness_arm_pin : int | None
            Harness/DUT arm pin number (active-low).
        harness_trigger_pin : int | None
            Harness/DUT trigger pin number.
        dut_arm_hold_ms : int | None
            Arm-low hold duration before DUT sets trigger HIGH.
        harness_stable_low_ms : int | None
            Stable-low window required before harness arms.
        harness_ready_timeout_s : float | None
            Timeout waiting for HARNESS READY.
        harness_arm_timeout_s : float | None
            Harness-side arm timeout while waiting for a valid D2 trigger after
            D3 active-low arming. This value is compiled into harness firmware
            as `CREST_HARNESS_ARM_TIMEOUT_MS`; set to 0 to disable it.
        harness_active_timeout_s : float | None
            Maximum time to wait for a harness measurement window.
        harness_done_timeout_s : float | None
            Timeout waiting for DONE response.

        Returns
        -------
        MeasureResult
            Parsed latency/power metrics.
        """
    @abstractmethod
    def evaluate(
        self,
        *,
        dirpath: str | Path,
        arena_kb: int,
        window_size: int,
        num_channels: int,
        serial_port: Optional[str] = None,
        run_hil: bool = True,
        baud_rate: int = 115200,
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
    ) -> DeviceMetrics:
        """Compile, upload, and measure a model on the target device.

        Parameters
        ----------
        dirpath : str | pathlib.Path
            Path to the sketch or project directory containing firmware sources.
        arena_kb : int
            Tensor arena size in KiB for this attempt.
        window_size : int
            Sliding window length used by the model.
        num_channels : int
            Number of sensor channels in each window.
        serial_port : str | None, optional
            Serial port for upload and log capture.
        run_hil : bool, optional
            When False, compile only (no upload/measurement).
        baud_rate : int, optional
            Serial baud rate for log capture.
        serial_timeout_s : float, optional
            Timeout for latency/power log capture.
        dut_ready_timeout_s : float | None, optional
            Time to wait for DUT READY before sending START.

        Returns
        -------
        DeviceMetrics
            Normalized metrics for the run.
        """
_LEGACY_DEVICE_SPECS = {
    "NUCLEO_F746ZG": {
        "arena_sizes": np.array([10, 30, 50, 75, 100, 150, 175, 200, 250, 280, 280]),
        "max_ram": 300_000,
        "max_flash": 800_000,
        "fqbn": None,
    },
    "NUCLEO_L476RG": {
        "arena_sizes": np.array([10, 25, 40, 70, 85, 100, 100]),
        "max_ram": 100_000,
        "max_flash": 800_000,
        "fqbn": None,
    },
    "NUCLEO_F446RE": {
        "arena_sizes": np.array([10, 25, 40, 70, 85, 100, 100]),
        "max_ram": 100_000,
        "max_flash": 400_000,
        "fqbn": None,
    },
    "ARCH_MAX": {
        "arena_sizes": np.array([10, 25, 40, 70, 95, 120, 140, 160, 170, 170]),
        "max_ram": 180_000,
        "max_flash": 400_000,
        "fqbn": None,
    },
    "ARDUINO_NANO_RP2040_CONNECT": {
        "arena_sizes": np.array([10, 25, 40, 70, 95, 120, 140, 150, 160, 180, 200, 210, 220]),
        "max_ram": 225_000,
        "max_flash": 15_000_000,
        # Superseded by the registered arduino_nano_rp2040 backend spec.
        "fqbn": "arduino:mbed_nano:nanorp2040connect",
    },
}


def _registry_device_specs() -> Dict[str, Dict[str, Any]]:
    """Load default device specs from microcontroller registry classes.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Compatibility-style spec map keyed by device name.
    """
    try:
        from .microcontrollers import list_device_specs
    except Exception:
        return {}

    try:
        loaded_specs = list_device_specs()
    except Exception:
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for device_name, spec in loaded_specs.items():
        normalized[device_name] = {
            "arena_sizes": np.array([int(value) for value in spec.get("arena_sizes", [])]),
            "max_ram": int(spec.get("max_ram", 0)),
            "max_flash": int(spec.get("max_flash", 0)),
            "fqbn": spec.get("fqbn"),
        }
    return normalized


def _rebuild_device_specs() -> Dict[str, Dict[str, Any]]:
    """Build compatibility DEVICE_SPECS from legacy + registry-backed metadata.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Merged compatibility map used by legacy call sites.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for device_name, raw in _LEGACY_DEVICE_SPECS.items():
        merged[device_name] = {
            "arena_sizes": np.array(raw["arena_sizes"]),
            "max_ram": int(raw["max_ram"]),
            "max_flash": int(raw["max_flash"]),
            "fqbn": raw.get("fqbn"),
        }
    merged.update(_registry_device_specs())
    return merged


# Initialize with legacy entries first; registry-backed entries are merged after
# ArduinoDevice is defined to avoid import cycles.
DEVICE_SPECS: Dict[str, Dict[str, Any]] = {
    device_name: {
        "arena_sizes": np.array(raw["arena_sizes"]),
        "max_ram": int(raw["max_ram"]),
        "max_flash": int(raw["max_flash"]),
        "fqbn": raw.get("fqbn"),
    }
    for device_name, raw in _LEGACY_DEVICE_SPECS.items()
}


def get_device_spec(name: str) -> DeviceSpec:
    """Return a DeviceSpec built from the compatibility DEVICE_SPECS registry.

    Parameters
    ----------
    name : str
        Device identifier that must exist in DEVICE_SPECS.

    Returns
    -------
    DeviceSpec
        Normalized device specification.

    Raises
    ------
    ValueError
        If existing validation or execution checks fail.
    """
    if name not in DEVICE_SPECS:
        # Retry loading registry defaults in case callers imported this module
        # before optional board modules became available.
        DEVICE_SPECS.update(_registry_device_specs())
    if name not in DEVICE_SPECS:
        raise ValueError(f"Unknown device '{name}'. Supported devices: {list(DEVICE_SPECS)}")
    raw = DEVICE_SPECS[name]
    arena_sizes = list(raw.get("arena_sizes", []))
    fqbn = raw.get("fqbn")
    toolchain = "arduino-cli" if fqbn else None
    return DeviceSpec(
        name=name,
        arena_sizes_kb=arena_sizes,
        max_ram_bytes=int(raw.get("max_ram", 0)),
        max_flash_bytes=int(raw.get("max_flash", 0)),
        fqbn=fqbn,
        toolchain=toolchain,
    )


def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
    """Read a value from a dict-like or namespace-like object.

    Parameters
    ----------
    container : Any
        Mapping-like or attribute-style object to inspect.
    key : str
        Field name to resolve.
    default : Any, optional
        Fallback value when ``key`` is missing.

    Returns
    -------
    Any
        Resolved field value or ``default``.
    """
    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(container, key, default)


def _normalize_portenta_target_core(raw_value: object) -> str:
    """Normalize and validate a Portenta target-core token.

    Parameters
    ----------
    raw_value : object
        Raw config value that should describe the Portenta target core.

    Returns
    -------
    str
        Normalized ``cm7`` or ``cm4`` token.

    Raises
    ------
    ValueError
        If existing validation or execution checks fail.
    """
    normalized = str(raw_value).strip().lower() if raw_value is not None else ""
    if normalized not in {"cm7", "cm4"}:
        raise ValueError(
            "Set device.portenta.target_core to 'cm7' or 'cm4' when device.name is PORTENTA_H7."
        )
    return normalized


def arduino_staged_sketch_path(outputs_dir: Path) -> Path:
    """Return the Arduino CLI-compatible staged sketch path.

    Parameters
    ----------
    outputs_dir : Path
        Candidate directory that will contain the staged sketch.

    Returns
    -------
    Path
        Path whose filename matches the candidate directory basename.

    Raises
    ------
    ValueError
        If existing validation or execution checks fail.
    """
    candidate_dir = Path(outputs_dir)
    if not candidate_dir.name:
        raise ValueError(f"Candidate directory has no basename: {candidate_dir}")
    return candidate_dir / f"{candidate_dir.name}.ino"


def _sync_arduino_sketch_variant_for_config(
    config: Any,
    outputs_dir: Path,
    *,
    sketches_dir: Path | None = None,
    runtime_phase: str = "back_to_back",
) -> Path:
    """Copy the Arduino sketch variant selected by config into ``outputs_dir``.

    Parameters
    ----------
    config : Any
        Loaded CREST configuration object.
    outputs_dir : Path
        Destination directory that will own the staged sketch.
    sketches_dir : Path | None, optional
        Optional override for the repository sketch root.
    runtime_phase : str, optional
        Runtime phase label used to choose the cadenced sketch variant.

    Returns
    -------
    Path
        Path to the staged ``{outputs_dir.name}.ino`` file.

    Raises
    ------
    ValueError
        If the runtime phase or configured Arduino input mode is unsupported.
    FileNotFoundError
        If the selected sketch or required analysis header is missing.
    """
    repo_root = Path(__file__).resolve().parents[2]
    sketch_variants_dir = (repo_root / "sketches") if sketches_dir is None else Path(sketches_dir)
    normalized_device_name = str(_cfg_get(_cfg_get(config, "device", None), "name", "")).strip().upper()
    normalized_runtime_phase = str(runtime_phase).strip().lower() or "back_to_back"
    if normalized_runtime_phase not in {"back_to_back", "cadenced"}:
        raise ValueError(
            f"Unsupported runtime_phase '{runtime_phase}'. Expected 'back_to_back' or 'cadenced'."
        )

    def _resolve_uniform_variant_dir() -> Path:
        """Resolve the directory containing uniform-input Arduino sketches.

        Returns
        -------
        Path
            Sketch variant directory compatible with the active device config.
        """
        if normalized_device_name == "PORTENTA_H7":
            portenta_cfg = _cfg_get(_cfg_get(config, "device", None), "portenta", None)
            _normalize_portenta_target_core(
                _cfg_get(portenta_cfg, "target_core", None) if portenta_cfg is not None else None
            )
        return sketch_variants_dir

    training_cfg = _cfg_get(config, "training", None)
    energy_aware = bool(_cfg_get(training_cfg, "energy_aware", False))
    input_mode = str(_cfg_get(training_cfg, "input_mode", "uniform")).strip().lower()
    header_name: str | None = None
    if normalized_device_name == "ARDUINO_NANO_RP2040_CONNECT":
        # Latency/memory-only target: always stage the per-run latency sketch,
        # whatever training.energy_aware says (config validation rejects it).
        if normalized_runtime_phase == "cadenced":
            raise ValueError("ARDUINO_NANO_RP2040_CONNECT does not support the cadenced runtime phase.")
        if input_mode != "uniform":
            raise ValueError(
                "ARDUINO_NANO_RP2040_CONNECT only supports training.input_mode='uniform'."
            )
        variant_dir = sketch_variants_dir
        variant_name = "crest_inference_latency.ino"
    elif normalized_runtime_phase == "cadenced":
        if input_mode != "uniform":
            raise ValueError(
                "Arduino cadenced runtime only supports training.input_mode='uniform'."
            )
        variant_dir = _resolve_uniform_variant_dir()
        variant_name = "crest_inference_energy_cadenced.ino"
    elif not energy_aware:
        variant_dir = _resolve_uniform_variant_dir()
        variant_name = "crest_inference_no_energy.ino"
    else:
        variants = {
            "uniform": ("crest_inference_energy.ino", _resolve_uniform_variant_dir(), None),
            "oxiod_representative": (
                "crest_inference_representative.ino",
                sketch_variants_dir / "analysis_sketches",
                "oxiod_input_data.h",
            ),
            "oxiod_real": (
                "crest_inference_real_data.ino",
                sketch_variants_dir / "analysis_sketches",
                "oxiod_input_data.h",
            ),
            "urbansound8k_representative": (
                "crest_inference_representative.ino",
                sketch_variants_dir / "analysis_sketches",
                "urbansound8k_input_data.h",
            ),
            "urbansound8k_real": (
                "crest_inference_real_data.ino",
                sketch_variants_dir / "analysis_sketches",
                "urbansound8k_input_data.h",
            ),
        }
        if input_mode not in variants:
            allowed = ", ".join(sorted(variants))
            raise ValueError(
                f"Unsupported input_mode '{input_mode}'. Expected one of: {allowed}."
            )
        variant_name, variant_dir, header_name = variants[input_mode]

    variant_source = variant_dir / variant_name
    if not variant_source.exists():
        raise FileNotFoundError(f"Sketch variant not found: {variant_source}")

    outputs_dir.mkdir(parents=True, exist_ok=True)
    # Arduino CLI requires a sketch folder to contain an .ino with the same
    # basename. Preserve that rule for every model family candidate directory.
    sketch_target = arduino_staged_sketch_path(outputs_dir)
    shutil.copyfile(variant_source, sketch_target)

    common_source = sketch_variants_dir / "common"
    if common_source.exists():
        shutil.copytree(common_source, outputs_dir / "common", dirs_exist_ok=True)

    variant_common_source = variant_dir / "common"
    if variant_common_source.exists() and variant_common_source != common_source:
        shutil.copytree(variant_common_source, outputs_dir / "common", dirs_exist_ok=True)

    if header_name is not None:
        header_source = sketch_variants_dir / "analysis_sketches" / header_name
        if not header_source.exists():
            raise FileNotFoundError(f"Input header not found: {header_source}")
        shutil.copyfile(header_source, outputs_dir / "crest_input_data.h")
    return sketch_target


class ArduinoDevice(DeviceInterface):
    """Temporary bridge that uses the Arduino CLI helpers.

    Notes
    -----
    FIXME: Remove this class once per-device microcontroller classes are wired
    into the HIL controller. It exists to keep current behavior intact while
    Phase 2 refactoring is introduced.
    """

    def __init__(
        self,
        device_name: str,
        *,
        serial_port: Optional[str] = None,
        device_options: Optional[Mapping[str, object]] = None,
        spec_override: Optional[DeviceSpec] = None,
    ) -> None:
        """Initialize an Arduino-backed device wrapper.

        Parameters
        ----------
        device_name : str
            Logical device identifier.
        serial_port : str | None, optional
            Serial port used for upload and measurement.
        device_options : Mapping[str, object] | None, optional
            Optional board-option mapping forwarded to subclasses.
        spec_override : DeviceSpec | None, optional
            Explicit spec override used by board-specific wrappers.

        Raises
        ------
        ValueError
            If existing validation or execution checks fail.
        """
        self._device_name = device_name
        self._device_options = dict(device_options or {})
        self._spec = spec_override or get_device_spec(device_name)
        if self._spec.fqbn is None:
            raise ValueError(f"Device '{device_name}' has no Arduino FQBN.")
        self._serial_port = serial_port
        self._board_options = self._resolve_board_options(self._device_options)

    @property
    def spec(self) -> DeviceSpec:
        """Return the device specification metadata.

        Returns
        -------
        DeviceSpec
            Static limits and Arduino FQBN metadata for this wrapper.
        """
        return self._spec

    def runtime_measure_mode(self) -> RuntimeMeasureMode:
        """Return runtime measurement mode for this board wrapper.

        Returns
        -------
        RuntimeMeasureMode
            Measurement path used by orchestration for this backend.
        """
        return "direct_serial"

    def requires_candidate_model(self) -> bool:
        """Return whether Arduino backends consume generated model artifacts.

        Returns
        -------
        bool
            ``True`` when the backend needs a generated candidate model before compilation.
        """
        return True

    def requires_training_data(self) -> bool:
        """Return whether Arduino candidate preparation needs calibration data.

        Returns
        -------
        bool
            ``True`` when calibration or training data must be available during candidate preparation.
        """
        return True

    def requires_arena_validation(self) -> bool:
        """Return whether Arduino backends require arena validation.

        Returns
        -------
        bool
            ``True`` when missing arena accounting should invalidate the device result.
        """
        return True

    def supports_energy_measurement(self) -> bool:
        """Return whether Arduino backends support real energy measurement.

        Returns
        -------
        bool
            ``True`` when the backend can report measured energy rather than proxy values.
        """
        return True

    def runtime_mode_build_defines(self) -> Dict[str, int]:
        """Return additional compile-time defines for runtime behavior.

        Returns
        -------
        Dict[str, int]
            Compile-time defines required by the selected runtime mode.
        """
        return {}

    def prepare_for_runtime(self, *, runtime_mode: RuntimeMeasureMode, serial_port: str) -> None:
        """Run optional board-specific setup before upload/measurement.

        Parameters
        ----------
        runtime_mode : RuntimeMeasureMode
            Runtime profile requested for the backend measurement pass.
        serial_port : str
            Serial device path used to communicate with the DUT or harness.
        """
        del runtime_mode, serial_port

    def prepare_candidate(
        self,
        *,
        request: CandidatePrepareRequest,
    ) -> Path:
        """Prepare Arduino build artifacts and return the active sketch directory.

        Parameters
        ----------
        request : CandidatePrepareRequest
            Candidate-export request that already includes the built Keras
            model, calibration split, artifact root, and TFLite destination.

        Returns
        -------
        Path
            Staged Arduino sketch directory ready for compile/upload.

        Notes
        -----
        The current Arduino flow quantizes against
        ``request.calibration_split.inputs`` and then stages the selected
        sketch variant beside the generated C model sources.

        Raises
        ------
        ValueError
            If existing validation or execution checks fail.
        """
        if request.model is None:
            raise ValueError("Arduino candidate preparation requires a built Keras model.")
        if request.quantization_mode == "int8_ptq" and request.calibration_split is None:
            raise ValueError("Arduino candidate preparation requires calibration/training data for int8_ptq.")

        from .hardware import convert_to_cpp_model, convert_to_tflite_model

        convert_to_tflite_model(
            model=request.model,
            training_data=None if request.calibration_split is None else request.calibration_split.inputs,
            quantization_mode=request.quantization_mode,
            output_name=str(request.tflite_model_path),
        )
        convert_to_cpp_model(
            tflite_path=request.tflite_model_path,
            output_dir=request.artifact_root,
        )
        _sync_arduino_sketch_variant_for_config(request.config, request.artifact_root)
        return request.artifact_root

    def set_input_mode(
        self,
        input_mode: str,
        *,
        outputs_dir: Path,
        config: Any,
        sketches_dir: Path | None = None,
        runtime_phase: str = "back_to_back",
    ) -> Path | None:
        """Update Arduino input mode and resynchronize the active sketch.

        Parameters
        ----------
        input_mode : str
            Input-header generation mode selected for the staged sketch.
        outputs_dir : Path
            Directory where generated sketch or backend artifacts are written.
        config : Any
            Loaded CREST configuration object or mapping.
        sketches_dir : Path | None
            Repository sketch root used to locate checked-in Arduino templates.
        runtime_phase : str
            Runtime phase label used to select generated telemetry settings.

        Returns
        -------
        Path | None
            Path to the synchronized sketch variant, when one was staged.
        """
        config.training.input_mode = str(input_mode).lower()
        return _sync_arduino_sketch_variant_for_config(
            config,
            outputs_dir,
            sketches_dir=sketches_dir,
            runtime_phase=runtime_phase,
        )

    def _resolve_board_options(
        self, device_options: Optional[Mapping[str, object]]
    ) -> Optional[Dict[str, str]]:
        """Resolve board options consumed by Arduino CLI compile/upload calls.

        Parameters
        ----------
        device_options : Mapping[str, object] | None
            Raw board option mapping supplied by callers.

        Returns
        -------
        Dict[str, str] | None
            Normalized board options for Arduino CLI, or ``None`` when unused.
        """
        del device_options
        return None

    def compile(
        self,
        *,
        sketch_path: Path,
        arena_kb: int,
        window_size: int,
        num_channels: int,
        build_defines: Optional[Dict[str, int]] = None,
    ) -> CompileResult:
        """Compile firmware using the Arduino CLI toolchain.

        Parameters
        ----------
        sketch_path : Path
            Directory containing the Arduino sketch to compile or upload.
        arena_kb : int
            Tensor arena size in KiB requested for the staged candidate.
        window_size : int
            Number of timesteps compiled into the model input buffer.
        num_channels : int
            Number of sensor or feature channels compiled into the model input buffer.
        build_defines : Optional[Dict[str, int]]
            Compile-time macro definitions forwarded to the backend toolchain.

        Returns
        -------
        CompileResult
            Parsed compile result with size accounting and failure classification.
        """
        logger.info(
            "ArduinoDevice.compile: patching constants and compiling sketch at %s",
            sketch_path,
        )
        arduino_base._patch_sketch_constants(
            sketch_path,
            arena_kb,
            window_size,
            num_channels,
            latency_budget_ms=self._device_options.get("latency_budget_ms"),
        )
        result = arduino_base.compile_sketch(
            sketch_path=sketch_path,
            fqbn=self._spec.fqbn,
            build_defines=build_defines,
            board_options=self._board_options,
        )
        return CompileResult(
            success=result.success,
            log=result.log,
            flash_bytes=result.flash_bytes,
            ram_bytes=result.ram_bytes,
            overflow_kind=result.overflow_kind,
            build_dir=result.build_dir,
        )

    def upload(
        self,
        *,
        sketch_path: Path,
        build_dir: Optional[Path],
        serial_port: Optional[str],
    ) -> UploadResult:
        """Upload firmware using the Arduino CLI toolchain.

        Parameters
        ----------
        sketch_path : Path
            Directory containing the Arduino sketch to compile or upload.
        build_dir : Optional[Path]
            Backend build-cache directory containing compiled artifacts.
        serial_port : Optional[str]
            Serial device path used to communicate with the DUT or harness.

        Returns
        -------
        UploadResult
            Parsed upload result with combined tool output.

        Raises
        ------
        ValueError
            If existing validation or execution checks fail.
        """
        use_serial_port = self._serial_port if serial_port is None else serial_port
        if use_serial_port is None:
            raise ValueError("serial_port must be provided for Arduino upload.")
        if build_dir is None:
            raise ValueError("build_dir must be provided for Arduino upload.")
        logger.info("ArduinoDevice.upload: uploading sketch to %s", use_serial_port)
        result = arduino_base.upload_sketch(
            sketch_path=sketch_path,
            fqbn=self._spec.fqbn,
            build_dir=build_dir,
            serial_port=use_serial_port,
            board_options=self._board_options,
        )
        return UploadResult(success=result.success, log=result.log)

    def measure(
        self,
        *,
        serial_port: Optional[str],
        baud_rate: int,
        serial_timeout_s: float,
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
    ) -> MeasureResult:
        """Measure latency/power using the Arduino serial log.

        Parameters
        ----------
        serial_port : str | None
            DUT serial port override. Falls back to the instance default.
        baud_rate : int
            DUT serial baud rate.
        serial_timeout_s : float
            Timeout for DUT-side serial telemetry.
        measured_inference_runs : int, optional
            Number of inference runs encoded into the deployed firmware.
        dut_ready_timeout_s : float | None, optional
            Optional DUT-ready timeout override.
        harness_serial_port : str | None, optional
            Optional harness serial port when harness-assisted measurement is
            enabled.

        Returns
        -------
        MeasureResult
            Parsed latency, optional arena diagnostics, serial log, and power
            metrics captured for the run.

        Raises
        ------
        ValueError
            If existing validation or execution checks fail.
        """
        use_serial_port = self._serial_port if serial_port is None else serial_port
        if use_serial_port is None:
            raise ValueError("serial_port must be provided for Arduino measurement.")
        logger.info(
            "ArduinoDevice.measure: starting serial measurement (dut=%s, harness=%s)",
            use_serial_port,
            harness_serial_port,
        )
        # Preserve explicit zero-valued overrides; only `None` should fall back
        # to the shared harness defaults.
        _default = lambda value, fallback: fallback if value is None else value
        result = arduino_base.measure_serial(
            serial_port=use_serial_port,
            baud_rate=baud_rate,
            serial_timeout_s=serial_timeout_s,
            measured_inference_runs=max(1, int(measured_inference_runs)),
            dut_ready_timeout_s=_default(dut_ready_timeout_s, 5.0),
            harness_serial_port=harness_serial_port,
            harness_fqbn=_default(harness_fqbn, "arduino:mbed_nano:nano33ble"),
            harness_auto_flash=_default(harness_auto_flash, "once"),
            harness_arm_pin=_default(harness_arm_pin, 3),
            harness_trigger_pin=_default(harness_trigger_pin, 2),
            dut_arm_hold_ms=_default(dut_arm_hold_ms, 600),
            harness_stable_low_ms=_default(harness_stable_low_ms, 500),
            harness_ready_timeout_s=_default(harness_ready_timeout_s, 5.0),
            harness_arm_timeout_s=_default(harness_arm_timeout_s, 5.0),
            harness_active_timeout_s=_default(harness_active_timeout_s, 30.0),
            harness_done_timeout_s=_default(harness_done_timeout_s, 5.0),
        )
        return MeasureResult(
            latency_s=result.latency_s,
            arena_error_line=result.arena_error_line,
            serial_log=result.serial_log,
            power_metrics=result.power_metrics,
        )

    def evaluate(
        self,
        *,
        dirpath: str | Path,
        arena_kb: int,
        window_size: int,
        num_channels: int,
        serial_port: Optional[str] = None,
        run_hil: bool = True,
        baud_rate: int = 115200,
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
    ) -> DeviceMetrics:
        """Run a compile/upload/measure loop using the Arduino toolchain.

        Parameters
        ----------
        dirpath : str | Path
            Staged sketch directory for the candidate.
        arena_kb : int
            Tensor arena size to compile into the sketch.
        window_size : int
            Sliding window length compiled into the sketch.
        num_channels : int
            Input channel count compiled into the sketch.
        serial_port : str | None, optional
            DUT serial port override for upload/runtime measurement.
        run_hil : bool, optional
            When ``False``, stop after compile and return compile-time metrics
            only.

        Returns
        -------
        DeviceMetrics
            Normalized compile/upload/runtime metrics for one attempt.

        Notes
        -----
        Compile overflow classification takes precedence over generic compile
        failure. For harness-only runtime modes, upload and timing are routed
        through the harness flow instead of the normal DUT serial path.

        Raises
        ------
        ValueError
            If existing validation or execution checks fail.
        """
        sketch_path = Path(dirpath).resolve()
        arena_bytes = arena_kb * 1024
        runtime_mode = self.runtime_measure_mode()
        build_defines = {
            "CREST_INFERENCE_RUNS": max(1, int(measured_inference_runs)),
            "CREST_HARNESS_ARM_PIN": 3 if harness_arm_pin is None else int(harness_arm_pin),
            "CREST_HARNESS_TRIGGER_PIN": 2 if harness_trigger_pin is None else int(harness_trigger_pin),
            "CREST_DUT_ARM_HOLD_MS": 600 if dut_arm_hold_ms is None else int(dut_arm_hold_ms),
            "CREST_HARNESS_STABLE_LOW_MS": 500
            if harness_stable_low_ms is None
            else int(harness_stable_low_ms),
        }
        build_defines.update(self.runtime_mode_build_defines())
        # Build defines carry both measurement cadence and harness wiring so
        # every compile artifact is specific to the active runtime setup.
        compile_result = self.compile(
            sketch_path=sketch_path,
            arena_kb=arena_kb,
            window_size=window_size,
            num_channels=num_channels,
            build_defines=build_defines,
        )

        overflow_kind = compile_result.overflow_kind
        if compile_result.ram_bytes is not None and compile_result.ram_bytes > self._spec.max_ram_bytes:
            overflow_kind = "ram"

        if overflow_kind is not None:
            error_code = (
                HIL_ERROR_FLASH_OVERFLOW
                if overflow_kind == "flash"
                else HIL_ERROR_RAM_OVERFLOW
            )
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=arena_bytes,
                error_code=error_code,
            )

        if not compile_result.success:
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=arena_bytes,
                error_code=HIL_ERROR_COMPILE,
            )

        if not run_hil:
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=arena_bytes,
                error_code=HIL_ERROR_OK,
            )

        use_serial_port = self._serial_port if serial_port is None else serial_port
        if use_serial_port is None:
            raise ValueError("serial_port must be provided when running HIL uploads.")

        # Preserve explicit zero-valued timeout overrides; callers sometimes
        # intentionally disable wait slack for test coverage.
        _default = lambda value, fallback: fallback if value is None else value
        measure_result: MeasureResult

        def _upload_error_metrics() -> DeviceMetrics:
            """Build a metric payload representing an upload failure.

            Returns
            -------
            DeviceMetrics
                Metrics object populated with compile-time usage and an upload
                failure code.
            """
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=arena_bytes,
                error_code=HIL_ERROR_UPLOAD,
            )

        if runtime_mode == "harness_only":
            # Harness-only boards keep the harness session open across DUT
            # upload because host-visible DUT serial is not the reliable timing
            # channel for that runtime mode.
            use_harness_port = harness_serial_port
            if not use_harness_port:
                raise ValueError(
                    "harness_serial_port must be provided for harness_only runtime measurement."
                )
            harness_fqbn_value = _default(harness_fqbn, "arduino:mbed_nano:nano33ble")
            harness_auto_flash_value = _default(harness_auto_flash, "once")
            harness_ready_timeout = float(_default(harness_ready_timeout_s, 5.0))
            harness_arm_timeout = float(_default(harness_arm_timeout_s, 5.0))
            harness_active_timeout = float(_default(harness_active_timeout_s, 30.0))
            harness_done_timeout = float(_default(harness_done_timeout_s, 5.0))
            harness_build_defines = {
                "CREST_INFERENCE_RUNS": max(1, int(measured_inference_runs)),
                "CREST_HARNESS_ARM_PIN": int(_default(harness_arm_pin, 3)),
                "CREST_HARNESS_TRIGGER_PIN": int(_default(harness_trigger_pin, 2)),
                "CREST_DUT_ARM_HOLD_MS": int(_default(dut_arm_hold_ms, 600)),
                "CREST_HARNESS_STABLE_LOW_MS": int(_default(harness_stable_low_ms, 500)),
                "CREST_HARNESS_ARM_TIMEOUT_MS": max(
                    0, int(round(harness_arm_timeout * 1000.0))
                ),
                "CREST_HARNESS_ACTIVE_TIMEOUT_MS": max(
                    0, int(round(harness_active_timeout * 1000.0))
                ),
            }
            try:
                self.prepare_for_runtime(runtime_mode=runtime_mode, serial_port=use_serial_port)
                arduino_base.ensure_harness_firmware(
                    harness_serial_port=use_harness_port,
                    harness_fqbn=harness_fqbn_value,
                    harness_auto_flash=harness_auto_flash_value,
                    build_defines=harness_build_defines,
                )
                logger.info(
                    "ArduinoDevice.evaluate[harness_only]: opening harness session before DUT upload "
                    "(harness=%s, dut=%s)",
                    use_harness_port,
                    use_serial_port,
                )
                with serial.Serial(use_harness_port, baudrate=baud_rate, timeout=0.1) as harness:
                    prime_result = hil_protocol.prime_harness_session(
                        harness=harness,
                        harness_ready_timeout_s=harness_ready_timeout,
                        harness_log=[],
                        flush_input=True,
                    )
                    if not prime_result.harness_ready:
                        logger.warning(
                            "Harness session was not ready before DUT upload (error=%s).",
                            prime_result.error,
                        )
                        return _upload_error_metrics()
                    else:
                        logger.info(
                            "ArduinoDevice.evaluate[harness_only]: uploading DUT while harness session is open."
                        )
                        upload_result = self.upload(
                            sketch_path=sketch_path,
                            build_dir=compile_result.build_dir,
                            serial_port=use_serial_port,
                        )
                        if not upload_result.success:
                            return _upload_error_metrics()
                        logger.info(
                            "ArduinoDevice.evaluate[harness_only]: DUT upload complete, waiting for harness DONE."
                        )
                        # In Portenta CM4 harness-only runs, host DUT Serial is
                        # not a reliable diagnostics channel, so runtime error
                        # typing depends on harness-visible signals only.
                        measure_result = arduino_base.measure_harness_only_open_session(
                            harness=harness,
                            harness_active_timeout_s=harness_active_timeout,
                            harness_done_timeout_s=harness_done_timeout,
                            harness_log=prime_result.harness_log,
                        )
            except serial.SerialException as exc:
                logger.warning("Harness serial session failed: %s", exc)
                return _upload_error_metrics()
            except RuntimeError as exc:
                logger.warning("Runtime preparation failed: %s", exc)
                return _upload_error_metrics()
        else:
            try:
                self.prepare_for_runtime(runtime_mode=runtime_mode, serial_port=use_serial_port)
            except RuntimeError as exc:
                logger.warning("Runtime preparation failed: %s", exc)
                return _upload_error_metrics()
            upload_result = self.upload(
                sketch_path=sketch_path,
                build_dir=compile_result.build_dir,
                serial_port=use_serial_port,
            )
            if not upload_result.success:
                return _upload_error_metrics()

            measure_result = self.measure(
                serial_port=use_serial_port,
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
        if measure_result.latency_s is None:
            if runtime_mode == "harness_only" and measure_result.arena_error_line is None:
                logger.warning(
                    "Harness-only runtime timeout without DUT arena diagnostics; "
                    "classifying as HIL_ERROR_LATENCY."
                )
            retry_hint = arduino_base._compute_retry_hint_bytes(
                arena_bytes, measure_result.arena_error_line
            )
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=arena_bytes,
                error_code=HIL_ERROR_UNDER_SIZED
                if measure_result.arena_error_line
                else HIL_ERROR_LATENCY,
                power_metrics=measure_result.power_metrics,
                retry_hint_bytes=retry_hint,
            )

        return DeviceMetrics(
            ram_bytes=compile_result.ram_bytes or -1,
            flash_bytes=compile_result.flash_bytes or -1,
            latency_s=measure_result.latency_s,
            arena_bytes=arena_bytes,
            error_code=HIL_ERROR_OK,
            power_metrics=measure_result.power_metrics,
        )


# Backward-compatible alias for the bridge class name.
# FIXME: Remove once the migration completes.
ArduinoLegacyDevice = ArduinoDevice

# Merge registry-backed class specs now that ArduinoDevice is fully defined.
DEVICE_SPECS.update(_registry_device_specs())


__all__ = [
    "RuntimeMeasureMode",
    "DeviceSpec",
    "DeviceMetrics",
    "CompileResult",
    "UploadResult",
    "MeasureResult",
    "DeviceInterface",
    "DEVICE_SPECS",
    "get_device_spec",
    "ArduinoDevice",
    "ArduinoLegacyDevice",
]
