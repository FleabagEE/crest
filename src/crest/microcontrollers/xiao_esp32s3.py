# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Board wrapper for the Seeed XIAO ESP32-S3 Sense backend.

Latency- and memory-only, mirroring the Nano RP2040 Connect backend: same
``ArduinoDevice`` compile/upload/measure flow and the same staged sketch
(``sketches/crest_inference_latency.ino``), with these ESP32-S3 specifics:

- FQBN ``esp32:esp32:XIAO_ESP32S3`` from the esp32 Arduino core 3.3.12. TFLM
  is the core's prebuilt ``esp-tflite-micro``, linked with ESP-NN's
  ESP32-S3-optimized kernels; the sketch does not include Chirale on ESP32.
- Board options pin ``USBMode=hwcdc`` (USB-Serial/JTAG for both the app and
  the ROM bootloader, so the USB VID:PID does not change across uploads),
  the CPU clock (default 240 MHz), the partition scheme and PSRAM.
- ``device.esp32s3.arena_memory``: ``internal_sram`` (static array, counted
  in the compile's RAM figure) or ``psram`` (``heap_caps`` allocation in the
  8 MB OPI PSRAM; enables the board's PSRAM option).
- Wi-Fi and Bluetooth are switched off in ``setup()``; the sketch prints an
  ``env`` line with the clock, radio state and arena address.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..devices import ArduinoDevice, CompileResult, DeviceSpec, MeasureResult, UploadResult
from .arduino_nano_rp2040 import (
    _as_int,
    _as_positive_float,
    _stamp_model_id,
    parse_run_lines,
    run_upload_once,
    wait_for_serial_port,
)

logger = logging.getLogger(__name__)

BOARD_NAME = "XIAO_ESP32S3_SENSE"
BOARD_FQBN = "esp32:esp32:XIAO_ESP32S3"
BOARD_ID = "xiao_esp32s3"
CORE_VERSION = "esp32:esp32@3.3.12"
ARENA_MEMORY_CHOICES = ("internal_sram", "psram")
CPU_MHZ_CHOICES = (80, 160, 240)

DEFAULT_ARENA_SIZES_KB = (10, 25, 40, 70, 95, 120, 140, 160, 180, 200)
# upload.maximum_data_size for XIAO_ESP32S3 (static DRAM available to the app).
DEFAULT_MAX_RAM_BYTES = 327_680
# default_8MB partition scheme: 3 MB app partition.
DEFAULT_MAX_FLASH_BYTES = 3_145_728

ESP32S3_OPTION_KEYS = (
    "arena_sizes_kb",
    "warmup_runs",
    "upload_timeout_s",
    "upload_retries",
    "port_wait_s",
    "arena_memory",
    "cpu_mhz",
)


@dataclass(frozen=True)
class XiaoESP32S3Options:
    """Resolved XIAO ESP32-S3 backend options.

    Parameters
    ----------
    arena_sizes_kb : tuple[int, ...]
        Ascending tensor-arena sweep in KiB.
    warmup_runs : int
        Untimed ``Invoke()`` calls before the timed window.
    upload_timeout_s : float
        Wall-clock limit for one ``arduino-cli upload`` attempt.
    upload_retries : int
        Extra upload attempts after a failure.
    port_wait_s : float
        Time to wait for the serial port to come back after an upload.
    arena_memory : str
        ``internal_sram`` or ``psram``.
    cpu_mhz : int
        CPU clock (80, 160 or 240).
    """

    arena_sizes_kb: tuple[int, ...] = DEFAULT_ARENA_SIZES_KB
    warmup_runs: int = 5
    upload_timeout_s: float = 120.0
    upload_retries: int = 2
    port_wait_s: float = 10.0
    arena_memory: str = "internal_sram"
    cpu_mhz: int = 240

    def board_options(self) -> Dict[str, str]:
        """Arduino CLI ``--board-options`` for compile and upload."""
        return {
            "USBMode": "hwcdc",
            "CDCOnBoot": "default",
            "CPUFreq": str(self.cpu_mhz),
            "PSRAM": "opi" if self.arena_memory == "psram" else "disabled",
            "PartitionScheme": "default_8MB",
            "DebugLevel": "none",
            "EraseFlash": "none",
            "UploadSpeed": "921600",
        }


def resolve_esp32s3_options(device_options: Optional[Mapping[str, object]]) -> XiaoESP32S3Options:
    """Resolve XIAO ESP32-S3 options (unknown keys are ignored).

    Parameters
    ----------
    device_options : Mapping[str, object] | None
        Raw options from ``device.esp32s3`` plus shared runtime keys.

    Returns
    -------
    XiaoESP32S3Options
        Validated options.

    Raises
    ------
    ValueError
        If an option is malformed.
    """
    raw = {k: v for k, v in dict(device_options or {}).items() if v not in (None, "")}
    defaults = XiaoESP32S3Options()
    arena_raw = raw.get("arena_sizes_kb")
    if arena_raw is None:
        arena_sizes = defaults.arena_sizes_kb
    else:
        if isinstance(arena_raw, (str, bytes)) or not isinstance(arena_raw, Sequence) or not arena_raw:
            raise ValueError("device.esp32s3.arena_sizes_kb must be a non-empty list of integers.")
        arena_sizes = tuple(_as_int("arena_sizes_kb", v, minimum=1) for v in arena_raw)
        if list(arena_sizes) != sorted(set(arena_sizes)):
            raise ValueError("device.esp32s3.arena_sizes_kb must be strictly ascending.")
    arena_memory = str(raw.get("arena_memory", defaults.arena_memory))
    if arena_memory not in ARENA_MEMORY_CHOICES:
        raise ValueError(f"device.esp32s3.arena_memory must be one of {ARENA_MEMORY_CHOICES}.")
    if arena_memory == "internal_sram" and arena_sizes[-1] * 1024 >= DEFAULT_MAX_RAM_BYTES:
        raise ValueError(
            f"device.esp32s3.arena_sizes_kb max {arena_sizes[-1]} KiB does not fit the "
            f"{DEFAULT_MAX_RAM_BYTES}-byte internal SRAM cap; use arena_memory: psram."
        )
    cpu_mhz = _as_int("cpu_mhz", raw.get("cpu_mhz", defaults.cpu_mhz), minimum=1)
    if cpu_mhz not in CPU_MHZ_CHOICES:
        raise ValueError(f"device.esp32s3.cpu_mhz must be one of {CPU_MHZ_CHOICES}.")
    return XiaoESP32S3Options(
        arena_sizes_kb=arena_sizes,
        warmup_runs=_as_int("warmup_runs", raw.get("warmup_runs", defaults.warmup_runs), minimum=0),
        upload_timeout_s=_as_positive_float("upload_timeout_s", raw.get("upload_timeout_s", defaults.upload_timeout_s)),
        upload_retries=_as_int("upload_retries", raw.get("upload_retries", defaults.upload_retries), minimum=0),
        port_wait_s=_as_positive_float("port_wait_s", raw.get("port_wait_s", defaults.port_wait_s)),
        arena_memory=arena_memory,
        cpu_mhz=cpu_mhz,
    )


def build_esp32s3_spec(options: Optional[XiaoESP32S3Options] = None) -> DeviceSpec:
    """Build the XIAO ESP32-S3 ``DeviceSpec``.

    With ``arena_memory: psram`` the arena is not part of static RAM, so the
    RAM cap still applies to the rest of the sketch.
    """
    resolved = options or XiaoESP32S3Options()
    return DeviceSpec(
        name=BOARD_NAME,
        arena_sizes_kb=list(resolved.arena_sizes_kb),
        max_ram_bytes=DEFAULT_MAX_RAM_BYTES,
        max_flash_bytes=DEFAULT_MAX_FLASH_BYTES,
        fqbn=BOARD_FQBN,
        toolchain="arduino-cli",
    )


BOARD_DEFAULT_SPEC = build_esp32s3_spec()


def parse_env_line(lines: Sequence[str]) -> Dict[str, str]:
    """Return the ``env key=value ...`` fields printed by the sketch on ESP32."""
    for line in lines:
        if line.startswith("env "):
            return dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)
    return {}


class XiaoESP32S3Device(ArduinoDevice):
    """Seeed XIAO ESP32-S3 Sense device wrapper (latency and memory only).

    Parameters
    ----------
    serial_port : str | None, optional
        USB-Serial/JTAG port (``/dev/ttyACM*``).
    device_options : dict[str, Any] | None, optional
        Options from ``device.esp32s3`` plus shared runtime options.
    """

    def __init__(self, *, serial_port: Optional[str] = None, device_options: Optional[dict[str, Any]] = None) -> None:
        """Resolve options and build the spec."""
        self._esp32_options = resolve_esp32s3_options(device_options)
        super().__init__(
            BOARD_NAME,
            serial_port=serial_port,
            device_options=dict(device_options or {}),
            spec_override=build_esp32s3_spec(self._esp32_options),
        )

    @property
    def esp32_options(self) -> XiaoESP32S3Options:
        """Return the resolved options."""
        return self._esp32_options

    def _resolve_board_options(self, device_options: Optional[Mapping[str, object]]) -> Optional[Dict[str, str]]:
        """Return the pinned Arduino CLI board options (see ``XiaoESP32S3Options``)."""
        del device_options
        return resolve_esp32s3_options(self._device_options).board_options()

    def supports_energy_measurement(self) -> bool:
        """Return ``False``: latency and memory only."""
        return False

    def runtime_mode_build_defines(self) -> Dict[str, int]:
        """Warm-up count, arena placement and CPU clock for the latency sketch."""
        return {
            "CREST_WARMUP_RUNS": int(self._esp32_options.warmup_runs),
            "CREST_ARENA_IN_PSRAM": int(self._esp32_options.arena_memory == "psram"),
            "CREST_ESP32_CPU_MHZ": int(self._esp32_options.cpu_mhz),
        }

    def compile(
        self,
        *,
        sketch_path: Path,
        arena_kb: int,
        window_size: int,
        num_channels: int,
        build_defines: Optional[Dict[str, int]] = None,
    ) -> CompileResult:
        """Stamp the model id, then compile through the shared Arduino path."""
        _stamp_model_id(Path(sketch_path))
        return super().compile(sketch_path=sketch_path, arena_kb=arena_kb, window_size=window_size,
                               num_channels=num_channels, build_defines=build_defines)

    def upload(self, *, sketch_path: Path, build_dir: Optional[Path], serial_port: Optional[str]) -> UploadResult:
        """Upload with esptool through arduino-cli, with a timeout and retries.

        Raises
        ------
        ValueError
            If the serial port or build directory is missing.
        """
        use_serial_port = self._serial_port if serial_port is None else serial_port
        if use_serial_port is None:
            raise ValueError("serial_port must be provided for Arduino upload.")
        if build_dir is None:
            raise ValueError("build_dir must be provided for Arduino upload.")
        options = self._esp32_options
        logs: List[str] = []
        attempts = options.upload_retries + 1
        for attempt in range(1, attempts + 1):
            wait_for_serial_port(use_serial_port, options.port_wait_s)
            result = run_upload_once(sketch_path=Path(sketch_path), build_dir=Path(build_dir),
                                     serial_port=use_serial_port, timeout_s=options.upload_timeout_s,
                                     fqbn=BOARD_FQBN, board_options=options.board_options())
            logs.append(f"--- upload attempt {attempt}/{attempts} ---\n{result.log}")
            if result.success:
                wait_for_serial_port(use_serial_port, options.port_wait_s)
                return UploadResult(success=True, log="\n".join(logs))
            if attempt < attempts:
                time.sleep(2.0)
        return UploadResult(success=False, log="\n".join(logs))

    def measure(
        self,
        *,
        serial_port: Optional[str],
        baud_rate: int,
        serial_timeout_s: float,
        measured_inference_runs: int = 10,
        dut_ready_timeout_s: Optional[float] = None,
        harness_serial_port: Optional[str] = None,
        **harness_kwargs: Any,
    ) -> MeasureResult:
        """DUT-only serial measurement; maps the sketch's arena infeasible line."""
        del harness_kwargs, harness_serial_port
        use_serial_port = self._serial_port if serial_port is None else serial_port
        if use_serial_port is not None:
            wait_for_serial_port(use_serial_port, self._esp32_options.port_wait_s)
        result = super().measure(serial_port=serial_port, baud_rate=baud_rate, serial_timeout_s=serial_timeout_s,
                                 measured_inference_runs=measured_inference_runs,
                                 dut_ready_timeout_s=dut_ready_timeout_s, harness_serial_port=None)
        if result.latency_s is None and result.arena_error_line is None:
            for record in parse_run_lines(result.serial_log):
                if record.get("status") == "infeasible" and record.get("reason") == "arena_too_small":
                    return dataclasses.replace(result, arena_error_line=record["raw"])
        return result
