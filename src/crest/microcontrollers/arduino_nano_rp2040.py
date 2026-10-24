# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Board wrapper for the Arduino Nano RP2040 Connect backend.

This backend mirrors the BLE33 wrapper (same ``arduino:mbed_nano`` core and
``ArduinoDevice`` compile/upload/measure flow) but is latency- and memory-only:
it never builds the INA228 or harness paths and reports no energy support.

Board-specific differences from BLE33:

- the staged sketch is ``sketches/crest_inference_latency.ino``, which emits
  one ``key=value`` line per timed run after a warm-up phase;
- ``device.rp2040.arena_sizes_kb`` overrides the arena sweep per target;
- uploads run with a timeout, retries, and an explicit 1200-bps touch,
  because the RP2040 has to re-enter its USB bootloader for every upload.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import serial

from ..devices import ArduinoDevice, CompileResult, DeviceSpec, MeasureResult, UploadResult
from . import arduino_base

logger = logging.getLogger(__name__)

BOARD_NAME = "ARDUINO_NANO_RP2040_CONNECT"
BOARD_FQBN = "arduino:mbed_nano:nanorp2040connect"
BOARD_ID = "nano_rp2040_connect"
LATENCY_SKETCH_NAME = "crest_inference_latency.ino"

# The RP2040 has 264 KB of SRAM, but mbed OS keeps part of it for its heap,
# thread stacks and the USB stack. The RAM cap and largest arena are
# conservative until a real build reports the free headroom.
DEFAULT_ARENA_SIZES_KB = (10, 25, 40, 70, 95, 120, 140, 160, 180, 200)
DEFAULT_MAX_RAM_BYTES = 225_000
DEFAULT_MAX_FLASH_BYTES = 15_000_000

RP2040_OPTION_KEYS = (
    "arena_sizes_kb",
    "warmup_runs",
    "upload_timeout_s",
    "upload_retries",
    "port_wait_s",
)

RUN_LINE_RE = re.compile(r"^board=(?P<board>\S+)\s+model=(?P<model>\S+)\s+run=(?P<run>-?\d+)\s+(?P<rest>.*)$")
_KEY_VALUE_RE = re.compile(r"(\w+)=(\S+)")
_INT_FIELDS = frozenset({"run", "invoke_us", "preprocess_us", "arena_used_bytes", "arena_bytes"})


@dataclass(frozen=True)
class ArduinoNanoRP2040Options:
    """Resolved Nano RP2040 Connect backend options.

    Parameters
    ----------
    arena_sizes_kb : tuple[int, ...]
        Ascending tensor-arena sweep in KiB.
    warmup_runs : int
        Untimed ``Invoke()`` calls before the timed window.
    upload_timeout_s : float
        Wall-clock limit for one ``arduino-cli upload`` attempt.
    upload_retries : int
        Extra upload attempts after the first failure.
    port_wait_s : float
        Time to wait for the serial port to re-enumerate after upload or reset.
    """

    arena_sizes_kb: tuple[int, ...] = DEFAULT_ARENA_SIZES_KB
    warmup_runs: int = 5
    upload_timeout_s: float = 90.0
    upload_retries: int = 2
    port_wait_s: float = 10.0


def _as_int(name: str, value: object, *, minimum: int) -> int:
    """Coerce one integer option and enforce a lower bound.

    Parameters
    ----------
    name : str
        Option name used in error messages.
    value : object
        Raw option value.
    minimum : int
        Smallest accepted value.

    Returns
    -------
    int
        Validated integer.

    Raises
    ------
    ValueError
        If the value is not an integer or is below ``minimum``.
    """
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"device.rp2040.{name} must be an integer >= {minimum}.")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"device.rp2040.{name} must be an integer >= {minimum}.") from exc
    if parsed < minimum:
        raise ValueError(f"device.rp2040.{name} must be an integer >= {minimum}.")
    return parsed


def _as_positive_float(name: str, value: object) -> float:
    """Coerce one positive float option.

    Parameters
    ----------
    name : str
        Option name used in error messages.
    value : object
        Raw option value.

    Returns
    -------
    float
        Validated positive float.

    Raises
    ------
    ValueError
        If the value is not a positive number.
    """
    if isinstance(value, bool):
        raise ValueError(f"device.rp2040.{name} must be a positive number.")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"device.rp2040.{name} must be a positive number.") from exc
    if parsed <= 0.0:
        raise ValueError(f"device.rp2040.{name} must be a positive number.")
    return parsed


def resolve_rp2040_options(
    device_options: Optional[Mapping[str, object]],
) -> ArduinoNanoRP2040Options:
    """Resolve Nano RP2040 Connect board options.

    Parameters
    ----------
    device_options : Mapping[str, object] | None
        Raw options. Keys outside ``RP2040_OPTION_KEYS`` (for example the
        shared ``runtime_mode`` and ``latency_budget_ms``) are ignored.

    Returns
    -------
    ArduinoNanoRP2040Options
        Validated options with defaults filled in.

    Raises
    ------
    ValueError
        If any RP2040 option is malformed.
    """
    raw = dict(device_options or {})
    defaults = ArduinoNanoRP2040Options()

    arena_raw = raw.get("arena_sizes_kb")
    if arena_raw in (None, ""):
        arena_sizes = defaults.arena_sizes_kb
    else:
        if isinstance(arena_raw, (str, bytes)) or not isinstance(arena_raw, Sequence):
            raise ValueError("device.rp2040.arena_sizes_kb must be a non-empty list of integers.")
        arena_sizes = tuple(_as_int("arena_sizes_kb", value, minimum=1) for value in arena_raw)
        if not arena_sizes:
            raise ValueError("device.rp2040.arena_sizes_kb must be a non-empty list of integers.")
        if list(arena_sizes) != sorted(set(arena_sizes)):
            raise ValueError("device.rp2040.arena_sizes_kb must be strictly ascending.")
    if arena_sizes[-1] * 1024 >= DEFAULT_MAX_RAM_BYTES:
        raise ValueError(
            f"device.rp2040.arena_sizes_kb max {arena_sizes[-1]} KiB does not fit the "
            f"{DEFAULT_MAX_RAM_BYTES}-byte RAM cap."
        )

    def _pick(key: str, default: Any) -> Any:
        value = raw.get(key)
        return default if value in (None, "") else value

    return ArduinoNanoRP2040Options(
        arena_sizes_kb=arena_sizes,
        warmup_runs=_as_int("warmup_runs", _pick("warmup_runs", defaults.warmup_runs), minimum=0),
        upload_timeout_s=_as_positive_float(
            "upload_timeout_s", _pick("upload_timeout_s", defaults.upload_timeout_s)
        ),
        upload_retries=_as_int(
            "upload_retries", _pick("upload_retries", defaults.upload_retries), minimum=0
        ),
        port_wait_s=_as_positive_float("port_wait_s", _pick("port_wait_s", defaults.port_wait_s)),
    )


def build_rp2040_spec(options: Optional[ArduinoNanoRP2040Options] = None) -> DeviceSpec:
    """Build the Nano RP2040 Connect ``DeviceSpec``.

    Parameters
    ----------
    options : ArduinoNanoRP2040Options | None, optional
        Resolved options. ``None`` uses the defaults.

    Returns
    -------
    DeviceSpec
        Device specification with the per-target arena sweep applied.
    """
    resolved = options or ArduinoNanoRP2040Options()
    return DeviceSpec(
        name=BOARD_NAME,
        arena_sizes_kb=list(resolved.arena_sizes_kb),
        max_ram_bytes=DEFAULT_MAX_RAM_BYTES,
        max_flash_bytes=DEFAULT_MAX_FLASH_BYTES,
        fqbn=BOARD_FQBN,
        toolchain="arduino-cli",
    )


BOARD_DEFAULT_SPEC = build_rp2040_spec()


def parse_run_lines(lines: Sequence[str]) -> List[Dict[str, Any]]:
    """Parse per-run ``board=... run=... status=...`` telemetry lines.

    Parameters
    ----------
    lines : Sequence[str]
        Decoded serial log lines.

    Returns
    -------
    list[dict[str, Any]]
        One record per telemetry line, in log order. Integer fields are
        converted to ``int``; ``raw`` keeps the original line.
    """
    records: List[Dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        match = RUN_LINE_RE.match(stripped)
        if match is None:
            continue
        record: Dict[str, Any] = {
            "board": match.group("board"),
            "model": match.group("model"),
            "run": int(match.group("run")),
            "raw": stripped,
        }
        for key, value in _KEY_VALUE_RE.findall(match.group("rest")):
            if key in _INT_FIELDS:
                try:
                    record[key] = int(value)
                    continue
                except ValueError:
                    pass
            record[key] = value
        records.append(record)
    return records


def wait_for_serial_port(serial_port: str, timeout_s: float, *, settle_s: float = 0.5) -> bool:
    """Wait until a serial device node exists again.

    Parameters
    ----------
    serial_port : str
        Device path such as ``/dev/ttyACM0``.
    timeout_s : float
        Maximum wait time.
    settle_s : float, optional
        Extra delay after the node appears, so udev and USB CDC finish setup.

    Returns
    -------
    bool
        ``True`` when the node exists before the timeout.
    """
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        if os.path.exists(serial_port):
            time.sleep(settle_s)
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def touch_1200bps(serial_port: str) -> bool:
    """Open and close the port at 1200 baud to request the USB bootloader.

    Parameters
    ----------
    serial_port : str
        Application-mode CDC port.

    Returns
    -------
    bool
        ``True`` when the touch was sent, ``False`` when the port was missing
        or could not be opened (for example because the board is already in
        the bootloader).
    """
    if not os.path.exists(serial_port):
        return False
    try:
        with serial.Serial(serial_port, baudrate=1200) as port:
            port.dtr = False
    except (serial.SerialException, OSError) as exc:
        logger.warning("1200-bps touch on %s failed: %s", serial_port, exc)
        return False
    return True


def run_upload_once(
    *,
    sketch_path: Path,
    build_dir: Path,
    serial_port: str,
    timeout_s: float,
    fqbn: str = BOARD_FQBN,
) -> UploadResult:
    """Run one ``arduino-cli upload`` with a hard timeout.

    Parameters
    ----------
    sketch_path : Path
        Staged sketch directory.
    build_dir : Path
        Build directory produced by compile.
    serial_port : str
        Application-mode CDC port.
    timeout_s : float
        Wall-clock limit. On expiry the whole process group (arduino-cli and
        the ``picotool`` child it spawns) is killed.
    fqbn : str, optional
        Board FQBN.

    Returns
    -------
    UploadResult
        Upload status and combined output.
    """
    upload_cmd = [
        arduino_base.ARDUINO_CLI_BIN,
        "--config-file",
        arduino_base.ARDUINO_CLI_CONFIG,
        "upload",
        "-p",
        serial_port,
        "--fqbn",
        fqbn,
        "--build-path",
        str(build_dir),
        str(sketch_path),
    ]
    proc = subprocess.Popen(
        upload_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        log = (
            f"{stdout}\n{stderr}\nUpload timed out after {timeout_s:.0f} s; the board probably "
            "did not re-enter the RP2040 USB bootloader (or, under WSL, the bootloader "
            "device was not attached with usbipd)."
        )
        return UploadResult(success=False, log=log)
    return UploadResult(success=proc.returncode == 0, log=f"{stdout}\n{stderr}")


class ArduinoNanoRP2040Device(ArduinoDevice):
    """Arduino Nano RP2040 Connect device wrapper (latency and memory only).

    Parameters
    ----------
    serial_port : str | None, optional
        Target serial port for upload and measurement.
    device_options : dict[str, Any] | None, optional
        Options from ``device.rp2040`` plus shared runtime options.
    """

    def __init__(
        self,
        *,
        serial_port: Optional[str] = None,
        device_options: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the RP2040 wrapper and resolve its arena sweep.

        Parameters
        ----------
        serial_port : str | None, optional
            Target serial port for upload and measurement.
        device_options : dict[str, Any] | None, optional
            Options from ``device.rp2040`` plus shared runtime options.
        """
        self._rp2040_options = resolve_rp2040_options(device_options)
        super().__init__(
            BOARD_NAME,
            serial_port=serial_port,
            device_options=dict(device_options or {}),
            spec_override=build_rp2040_spec(self._rp2040_options),
        )

    @property
    def rp2040_options(self) -> ArduinoNanoRP2040Options:
        """Return the resolved RP2040 options.

        Returns
        -------
        ArduinoNanoRP2040Options
            Options used for builds, uploads and measurement.
        """
        return self._rp2040_options

    def supports_energy_measurement(self) -> bool:
        """Return ``False``: this target never measures energy.

        Returns
        -------
        bool
            Always ``False``.
        """
        return False

    def runtime_mode_build_defines(self) -> Dict[str, int]:
        """Return the warm-up count compiled into the latency sketch.

        Returns
        -------
        Dict[str, int]
            ``CREST_WARMUP_RUNS`` for the staged sketch.
        """
        return {"CREST_WARMUP_RUNS": int(self._rp2040_options.warmup_runs)}

    def compile(
        self,
        *,
        sketch_path: Path,
        arena_kb: int,
        window_size: int,
        num_channels: int,
        build_defines: Optional[Dict[str, int]] = None,
    ) -> CompileResult:
        """Stamp the model id into the sketch, then compile as BLE33 does.

        Parameters
        ----------
        sketch_path : Path
            Staged sketch directory.
        arena_kb : int
            Tensor arena size in KiB.
        window_size : int
            Input frames compiled into the sketch.
        num_channels : int
            Input channels (mel bins) compiled into the sketch.
        build_defines : Optional[Dict[str, int]]
            Extra compile-time defines.

        Returns
        -------
        CompileResult
            Compile status and memory usage.
        """
        _stamp_model_id(Path(sketch_path))
        return super().compile(
            sketch_path=sketch_path,
            arena_kb=arena_kb,
            window_size=window_size,
            num_channels=num_channels,
            build_defines=build_defines,
        )

    def upload(
        self,
        *,
        sketch_path: Path,
        build_dir: Optional[Path],
        serial_port: Optional[str],
    ) -> UploadResult:
        """Upload with a timeout, retries and a 1200-bps touch between attempts.

        Parameters
        ----------
        sketch_path : Path
            Staged sketch directory.
        build_dir : Optional[Path]
            Build directory produced by compile.
        serial_port : Optional[str]
            Application-mode CDC port.

        Returns
        -------
        UploadResult
            Result of the last attempt, with every attempt's log concatenated.

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
        options = self._rp2040_options
        logs: List[str] = []
        attempts = options.upload_retries + 1
        for attempt in range(1, attempts + 1):
            logger.info(
                "ArduinoNanoRP2040Device.upload: attempt %d/%d on %s",
                attempt,
                attempts,
                use_serial_port,
            )
            result = run_upload_once(
                sketch_path=Path(sketch_path),
                build_dir=Path(build_dir),
                serial_port=use_serial_port,
                timeout_s=options.upload_timeout_s,
                fqbn=self._spec.fqbn or BOARD_FQBN,
            )
            logs.append(f"--- upload attempt {attempt}/{attempts} ---\n{result.log}")
            if result.success:
                if not wait_for_serial_port(use_serial_port, options.port_wait_s):
                    logger.warning(
                        "Upload succeeded but %s did not re-enumerate within %.1f s.",
                        use_serial_port,
                        options.port_wait_s,
                    )
                return UploadResult(success=True, log="\n".join(logs))
            if attempt < attempts:
                # arduino-cli already performs a 1200-bps touch; repeating it
                # here covers runs where the CDC port came back late or the
                # previous attempt was killed mid-transfer.
                wait_for_serial_port(use_serial_port, options.port_wait_s)
                touched = touch_1200bps(use_serial_port)
                logs.append(f"retry: explicit 1200-bps touch sent={touched}")
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
        """Measure through the DUT-only serial path and keep per-run lines.

        Parameters
        ----------
        serial_port : str | None
            DUT serial port override.
        baud_rate : int
            DUT serial baud rate.
        serial_timeout_s : float
            Timeout waiting for ``timer output:``.
        measured_inference_runs : int, optional
            Timed runs compiled into the sketch.
        dut_ready_timeout_s : float | None, optional
            DUT-ready timeout override.
        harness_serial_port : str | None, optional
            Ignored. This target has no harness; the DUT-only path is forced.
        **harness_kwargs : Any
            Ignored harness settings accepted for interface compatibility.

        Returns
        -------
        MeasureResult
            Parsed latency and serial log. A ``status=infeasible
            reason=arena_too_small`` line counts as an arena error when TFLM
            did not print its own allocation message.
        """
        del harness_kwargs
        if harness_serial_port:
            logger.info("ArduinoNanoRP2040Device.measure: ignoring harness_serial_port (latency-only target).")
        use_serial_port = self._serial_port if serial_port is None else serial_port
        if use_serial_port is not None:
            wait_for_serial_port(use_serial_port, self._rp2040_options.port_wait_s)
        result = super().measure(
            serial_port=serial_port,
            baud_rate=baud_rate,
            serial_timeout_s=serial_timeout_s,
            measured_inference_runs=measured_inference_runs,
            dut_ready_timeout_s=dut_ready_timeout_s,
            harness_serial_port=None,
        )
        if result.latency_s is None and result.arena_error_line is None:
            for record in parse_run_lines(result.serial_log):
                if record.get("status") == "infeasible" and record.get("reason") == "arena_too_small":
                    return dataclasses.replace(result, arena_error_line=record["raw"])
        return result


def _stamp_model_id(sketch_path: Path) -> None:
    """Rewrite ``CREST_MODEL_ID`` in the staged sketch with the directory name.

    Parameters
    ----------
    sketch_path : Path
        Staged sketch directory.
    """
    ino_files = sorted(sketch_path.glob("*.ino"))
    if not ino_files:
        return
    text = ino_files[0].read_text(encoding="utf-8")
    if "CREST_MODEL_ID" not in text:
        return
    model_id = re.sub(r"[^A-Za-z0-9_.-]", "_", sketch_path.name) or "unnamed"
    text = arduino_base._replace_define(text, "CREST_MODEL_ID", f'"{model_id}"')
    ino_files[0].write_text(text, encoding="utf-8")
