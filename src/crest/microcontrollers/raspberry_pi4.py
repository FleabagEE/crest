# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Raspberry Pi 4 backend: LiteRT over SSH, latency and memory only.

This is a new-vendor backend that implements ``DeviceInterface`` directly
(there is no Arduino toolchain and no tensor arena). The CREST host keeps
doing export and orchestration; the Pi only runs inference:

- ``prepare_candidate`` exports the ``.tflite`` (float or full-int8) with
  ``convert_to_tflite_model`` and skips the C-array step.
- ``compile`` builds nothing. It reports ``flash_bytes`` as the ``.tflite``
  file size.
- ``upload`` copies the model and ``sketches/rpi4/crest_litert_runner.py``
  to ``<remote_dir>/<candidate>/`` with ``scp``.
- ``measure`` runs the runner over ``ssh``. It reuses CREST's ``timer
  output:`` parser for latency and reads ``peak_rss_delta_bytes`` as
  ``ram_bytes``.

The arena sweep is the single-shot ``[-1]`` sentinel used by the STM32
backend, and ``requires_arena_validation()`` is ``False``. Energy is never
measured. SSH uses ``BatchMode`` (key-based login only) and
host-key checking against the user's ``known_hosts``.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..devices import (
    CandidatePrepareRequest,
    CompileResult,
    DeviceInterface,
    DeviceMetrics,
    DeviceSpec,
    MeasureResult,
    UploadResult,
)
from ..errors import (
    HIL_ERROR_COMPILE,
    HIL_ERROR_FLASH_OVERFLOW,
    HIL_ERROR_LATENCY,
    HIL_ERROR_OK,
    HIL_ERROR_UPLOAD,
)
from . import arduino_base

logger = logging.getLogger(__name__)

BOARD_NAME = "RASPBERRY_PI_4"
BOARD_ID = "rpi4"
TOOLCHAIN = "litert-ssh"
RUNNER_PATH = Path(__file__).resolve().parents[3] / "sketches" / "rpi4" / "crest_litert_runner.py"
RUNNER_NAME = RUNNER_PATH.name
MODEL_NAME = "g_model.tflite"

# Limits used by the shared feasibility checks. RAM is the runner's peak RSS
# increase; flash is the .tflite size. Both are generous on a 2 GB Pi 4 but
# keep candidates from swapping or filling the SD card.
DEFAULT_MAX_RAM_BYTES = 1_000_000_000
DEFAULT_MAX_FLASH_BYTES = 500_000_000

RPI4_OPTION_KEYS = (
    "host",
    "user",
    "ssh_port",
    "remote_dir",
    "python",
    "num_threads",
    "xnnpack",
    "warmup_runs",
    "ssh_connect_timeout_s",
    "upload_timeout_s",
)

_PEAK_RSS_RE = re.compile(r"\bpeak_rss_delta_bytes=(-?\d+)")


@dataclass(frozen=True)
class RaspberryPi4Options:
    """Resolved Raspberry Pi 4 backend options.

    Parameters
    ----------
    host : str
        Pi hostname or IP address. Required for upload and measurement.
    user : str | None
        SSH user; ``None`` uses the SSH config default.
    ssh_port : int
        SSH port.
    remote_dir : str
        Directory on the Pi, relative to the login home unless absolute.
    python : str
        Interpreter on the Pi that has a LiteRT runtime installed.
    num_threads : int
        Interpreter threads (the Pi 4 has 4 Cortex-A72 cores).
    xnnpack : bool
        Keep the runtime's default XNNPACK delegate.
    warmup_runs : int
        Untimed ``invoke()`` calls before the timed window.
    ssh_connect_timeout_s : float
        ``ConnectTimeout`` for ssh/scp.
    upload_timeout_s : float
        Wall-clock limit for one copy step.
    """

    host: Optional[str] = None
    user: Optional[str] = None
    ssh_port: int = 22
    remote_dir: str = "crest_rpi4"
    python: str = "crest-litert/bin/python"
    num_threads: int = 1
    xnnpack: bool = True
    warmup_runs: int = 5
    ssh_connect_timeout_s: float = 10.0
    upload_timeout_s: float = 120.0

    @property
    def destination(self) -> str:
        """Return ``user@host`` (or ``host``) for ssh/scp."""
        if not self.host:
            raise ValueError("device.rpi4.host must be set for Raspberry Pi 4 upload/measurement.")
        return f"{self.user}@{self.host}" if self.user else str(self.host)


def _positive_int(name: str, value: Any, *, minimum: int) -> int:
    """Validate one integer option."""
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"device.rpi4.{name} must be an integer >= {minimum}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"device.rpi4.{name} must be an integer >= {minimum}.") from exc
    if parsed < minimum:
        raise ValueError(f"device.rpi4.{name} must be an integer >= {minimum}.")
    return parsed


def _positive_float(name: str, value: Any) -> float:
    """Validate one positive float option."""
    if isinstance(value, bool):
        raise ValueError(f"device.rpi4.{name} must be a positive number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"device.rpi4.{name} must be a positive number.") from exc
    if parsed <= 0.0:
        raise ValueError(f"device.rpi4.{name} must be a positive number.")
    return parsed


def resolve_rpi4_options(device_options: Optional[Mapping[str, object]]) -> RaspberryPi4Options:
    """Resolve Raspberry Pi 4 options.

    Parameters
    ----------
    device_options : Mapping[str, object] | None
        Raw options; keys outside ``RPI4_OPTION_KEYS`` (shared runtime keys)
        are ignored.

    Returns
    -------
    RaspberryPi4Options
        Validated options. ``host`` may still be ``None``; it is only
        required when something is uploaded or measured.

    Raises
    ------
    ValueError
        If an option is malformed.
    """
    raw = {key: value for key, value in dict(device_options or {}).items() if value not in (None, "")}
    defaults = RaspberryPi4Options()
    xnnpack = raw.get("xnnpack", defaults.xnnpack)
    if not isinstance(xnnpack, bool):
        raise ValueError("device.rpi4.xnnpack must be true or false.")
    for key in ("host", "user", "remote_dir", "python"):
        if key in raw and (not isinstance(raw[key], str) or not raw[key].strip()):
            raise ValueError(f"device.rpi4.{key} must be a non-empty string.")
        if key in raw and any(ch.isspace() for ch in raw[key].strip()):
            raise ValueError(f"device.rpi4.{key} must not contain whitespace.")
    return RaspberryPi4Options(
        host=raw.get("host"),
        user=raw.get("user"),
        ssh_port=_positive_int("ssh_port", raw.get("ssh_port", defaults.ssh_port), minimum=1),
        remote_dir=str(raw.get("remote_dir", defaults.remote_dir)),
        python=str(raw.get("python", defaults.python)),
        num_threads=_positive_int("num_threads", raw.get("num_threads", defaults.num_threads), minimum=1),
        xnnpack=xnnpack,
        warmup_runs=_positive_int("warmup_runs", raw.get("warmup_runs", defaults.warmup_runs), minimum=0),
        ssh_connect_timeout_s=_positive_float(
            "ssh_connect_timeout_s", raw.get("ssh_connect_timeout_s", defaults.ssh_connect_timeout_s)
        ),
        upload_timeout_s=_positive_float("upload_timeout_s", raw.get("upload_timeout_s", defaults.upload_timeout_s)),
    )


def build_rpi4_spec(options: Optional[RaspberryPi4Options] = None) -> DeviceSpec:
    """Build the Raspberry Pi 4 ``DeviceSpec``.

    Parameters
    ----------
    options : RaspberryPi4Options | None, optional
        Unused; kept for symmetry with other board modules.

    Returns
    -------
    DeviceSpec
        Spec with the single-shot ``[-1]`` arena sentinel.
    """
    del options
    return DeviceSpec(
        name=BOARD_NAME,
        arena_sizes_kb=[-1],
        max_ram_bytes=DEFAULT_MAX_RAM_BYTES,
        max_flash_bytes=DEFAULT_MAX_FLASH_BYTES,
        fqbn=None,
        toolchain=TOOLCHAIN,
    )


BOARD_DEFAULT_SPEC = build_rpi4_spec()


def run_command(cmd: List[str], timeout_s: float) -> subprocess.CompletedProcess:
    """Run a command with a hard timeout that kills its whole process group.

    Parameters
    ----------
    cmd : list[str]
        Command and arguments.
    timeout_s : float
        Wall-clock limit.

    Returns
    -------
    subprocess.CompletedProcess
        Result; on timeout ``returncode`` is ``-9`` and stderr says so.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        return subprocess.CompletedProcess(cmd, -9, stdout, f"{stderr}\ntimed out after {timeout_s:.0f} s")
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def parse_peak_rss_delta(lines: List[str]) -> Optional[int]:
    """Return ``peak_rss_delta_bytes`` from runner output, if present."""
    for line in lines:
        match = _PEAK_RSS_RE.search(line)
        if match:
            return int(match.group(1))
    return None


class RaspberryPi4Device(DeviceInterface):
    """Raspberry Pi 4 device wrapper (LiteRT over SSH).

    Parameters
    ----------
    serial_port : str | None, optional
        Ignored; accepted for interface compatibility.
    device_options : dict[str, Any] | None, optional
        Options from ``device.rpi4`` plus shared runtime options.
    """

    def __init__(
        self,
        *,
        serial_port: Optional[str] = None,
        device_options: Optional[dict[str, Any]] = None,
    ) -> None:
        """Resolve options and build the spec."""
        del serial_port
        self._options = resolve_rpi4_options(device_options)
        self._spec = build_rpi4_spec(self._options)

    @property
    def spec(self) -> DeviceSpec:
        """Return the device spec."""
        return self._spec

    @property
    def options(self) -> RaspberryPi4Options:
        """Return the resolved options."""
        return self._options

    def requires_arena_validation(self) -> bool:
        """Return ``False``: Linux has no tensor arena (``[-1]`` sentinel)."""
        return False

    def supports_energy_measurement(self) -> bool:
        """Return ``False``: latency and memory only."""
        return False

    # -- SSH helpers -------------------------------------------------------

    def _ssh_base(self) -> List[str]:
        """Return the shared ssh option list."""
        return [
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={int(round(self._options.ssh_connect_timeout_s))}",
        ]

    def _ssh(self, remote_command: str, timeout_s: float) -> subprocess.CompletedProcess:
        """Run one command on the Pi."""
        cmd = ["ssh", *self._ssh_base(), "-p", str(self._options.ssh_port), self._options.destination, remote_command]
        return run_command(cmd, timeout_s)

    def _remote_candidate_dir(self, sketch_path: Path) -> str:
        """Return the per-candidate directory on the Pi."""
        return f"{self._options.remote_dir.rstrip('/')}/{sketch_path.name}"

    # -- DeviceInterface ---------------------------------------------------

    def prepare_candidate(self, *, request: CandidatePrepareRequest) -> Path:
        """Export the candidate ``.tflite`` (no C array, no sketch).

        Parameters
        ----------
        request : CandidatePrepareRequest
            Export request from orchestration.

        Returns
        -------
        Path
            ``request.artifact_root`` containing ``g_model.tflite``.

        Raises
        ------
        ValueError
            If the model or required calibration data is missing.
        """
        if request.model is None:
            raise ValueError("Raspberry Pi 4 candidate preparation requires a built Keras model.")
        if request.quantization_mode == "int8_ptq" and request.calibration_split is None:
            raise ValueError("Raspberry Pi 4 candidate preparation requires calibration data for int8_ptq.")
        from ..hardware import convert_to_tflite_model

        artifact_root = Path(request.artifact_root)
        artifact_root.mkdir(parents=True, exist_ok=True)
        convert_to_tflite_model(
            model=request.model,
            training_data=None if request.calibration_split is None else request.calibration_split.inputs,
            quantization_mode=request.quantization_mode,
            output_name=str(request.tflite_model_path),
        )
        staged = artifact_root / MODEL_NAME
        if Path(request.tflite_model_path).resolve() != staged.resolve():
            staged.write_bytes(Path(request.tflite_model_path).read_bytes())
        return artifact_root

    def compile(
        self,
        *,
        sketch_path: Path,
        arena_kb: int,
        window_size: int,
        num_channels: int,
        build_defines: Optional[Dict[str, int]] = None,
    ) -> CompileResult:
        """Check the staged model; report its size as ``flash_bytes``.

        Parameters
        ----------
        sketch_path : Path
            Candidate directory containing ``g_model.tflite``.
        arena_kb, window_size, num_channels : int
            Unused (the model carries its own shapes; no arena on Linux).
        build_defines : dict[str, int] | None, optional
            Unused.

        Returns
        -------
        CompileResult
            ``success`` when the model file exists; ``ram_bytes`` is unknown
            until measurement.
        """
        del arena_kb, window_size, num_channels, build_defines
        model = Path(sketch_path) / MODEL_NAME
        if not model.is_file():
            return CompileResult(success=False, log=f"missing {model}", flash_bytes=None, ram_bytes=None,
                                 overflow_kind=None, build_dir=None)
        size = model.stat().st_size
        overflow = "flash" if size > self._spec.max_flash_bytes else None
        return CompileResult(success=overflow is None, log=f"{MODEL_NAME}: {size} bytes", flash_bytes=size,
                             ram_bytes=None, overflow_kind=overflow, build_dir=Path(sketch_path))

    def upload(self, *, sketch_path: Path, build_dir: Optional[Path], serial_port: Optional[str]) -> UploadResult:
        """Copy the model and runner to the Pi with scp.

        Parameters
        ----------
        sketch_path : Path
            Candidate directory.
        build_dir : Path | None
            Unused.
        serial_port : str | None
            Unused.

        Returns
        -------
        UploadResult
            Combined ssh/scp output.
        """
        del build_dir, serial_port
        sketch_path = Path(sketch_path)
        remote = self._remote_candidate_dir(sketch_path)
        mkdir = self._ssh(f"mkdir -p {shlex.quote(remote)}", self._options.upload_timeout_s)
        if mkdir.returncode != 0:
            return UploadResult(success=False, log=f"ssh mkdir failed ({mkdir.returncode}): {mkdir.stderr.strip()}")
        cmd = [
            "scp", *self._ssh_base(), "-P", str(self._options.ssh_port), "-q",
            str(sketch_path / MODEL_NAME), str(RUNNER_PATH),
            f"{self._options.destination}:{remote}/",
        ]
        copy = run_command(cmd, self._options.upload_timeout_s)
        log = f"scp -> {remote}: rc={copy.returncode} {copy.stdout.strip()} {copy.stderr.strip()}".strip()
        return UploadResult(success=copy.returncode == 0, log=log)

    def measure(
        self,
        *,
        serial_port: Optional[str] = None,
        baud_rate: int = 115200,
        serial_timeout_s: float = 120.0,
        measured_inference_runs: int = 10,
        dut_ready_timeout_s: Optional[float] = None,
        sketch_path: Optional[Path] = None,
        **harness_kwargs: Any,
    ) -> MeasureResult:
        """Run the uploaded runner over ssh and parse its output.

        Parameters
        ----------
        serial_port, baud_rate, dut_ready_timeout_s, **harness_kwargs
            Unused; accepted for interface compatibility.
        serial_timeout_s : float
            Wall-clock limit for the remote run.
        measured_inference_runs : int
            Timed runs.
        sketch_path : Path
            Candidate directory (selects the remote directory).

        Returns
        -------
        MeasureResult
            ``latency_s`` from ``timer output:``; ``serial_log`` holds the
            runner's stdout lines (plus stderr when it failed).

        Raises
        ------
        ValueError
            If ``sketch_path`` is missing.
        """
        del serial_port, baud_rate, dut_ready_timeout_s, harness_kwargs
        if sketch_path is None:
            raise ValueError("RaspberryPi4Device.measure requires sketch_path.")
        sketch_path = Path(sketch_path)
        remote = self._remote_candidate_dir(sketch_path)
        runner_args = [
            self._options.python,
            f"{remote}/{RUNNER_NAME}",
            "--model", f"{remote}/{MODEL_NAME}",
            "--model-id", sketch_path.name,
            "--runs", str(max(1, int(measured_inference_runs))),
            "--warmup", str(self._options.warmup_runs),
            "--threads", str(self._options.num_threads),
        ]
        if not self._options.xnnpack:
            runner_args.append("--no-xnnpack")
        result = self._ssh(" ".join(shlex.quote(arg) for arg in runner_args), serial_timeout_s)
        lines = result.stdout.splitlines()
        if result.returncode != 0:
            lines += [f"remote_exit={result.returncode}"] + [f"stderr: {line}" for line in result.stderr.splitlines()[-20:]]
        latency_s, _ = arduino_base._parse_latency_from_log(lines)
        return MeasureResult(latency_s=latency_s, arena_error_line=None, serial_log=lines, power_metrics=None)

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
        serial_timeout_s: float = 120.0,
        measured_inference_runs: int = 10,
        **harness_kwargs: Any,
    ) -> DeviceMetrics:
        """Check, upload and measure one candidate.

        Parameters
        ----------
        dirpath : str | Path
            Candidate directory with ``g_model.tflite``.
        arena_kb, window_size, num_channels : int
            Unused (single-shot sentinel; shapes come from the model).
        serial_port : str | None, optional
            Unused.
        run_hil : bool, optional
            ``False`` returns the file-size metrics without touching the Pi.
        baud_rate : int, optional
            Unused.
        serial_timeout_s : float, optional
            Limit for the remote run.
        measured_inference_runs : int, optional
            Timed runs.
        **harness_kwargs : Any
            Ignored harness settings.

        Returns
        -------
        DeviceMetrics
            ``flash_bytes`` = model size, ``ram_bytes`` = peak RSS delta,
            ``arena_bytes`` = ``-1``. ``HIL_ERROR_UPLOAD`` when ssh/scp fails,
            ``HIL_ERROR_LATENCY`` when the run produces no latency.
        """
        del arena_kb, serial_port, baud_rate, harness_kwargs
        sketch_path = Path(dirpath).resolve()
        compile_result = self.compile(sketch_path=sketch_path, arena_kb=-1, window_size=window_size,
                                      num_channels=num_channels)
        flash = compile_result.flash_bytes if compile_result.flash_bytes is not None else -1
        if not compile_result.success:
            error_code = HIL_ERROR_FLASH_OVERFLOW if compile_result.overflow_kind == "flash" else HIL_ERROR_COMPILE
            return DeviceMetrics(ram_bytes=-1, flash_bytes=flash, latency_s=-1.0, arena_bytes=-1,
                                 error_code=error_code)
        if not run_hil:
            return DeviceMetrics(ram_bytes=-1, flash_bytes=flash, latency_s=-1.0, arena_bytes=-1,
                                 error_code=HIL_ERROR_OK)
        upload = self.upload(sketch_path=sketch_path, build_dir=None, serial_port=None)
        if not upload.success:
            logger.warning("RaspberryPi4Device upload failed: %s", upload.log)
            return DeviceMetrics(ram_bytes=-1, flash_bytes=flash, latency_s=-1.0, arena_bytes=-1,
                                 error_code=HIL_ERROR_UPLOAD)
        measure = self.measure(serial_timeout_s=serial_timeout_s, measured_inference_runs=measured_inference_runs,
                               sketch_path=sketch_path)
        peak_rss = parse_peak_rss_delta(measure.serial_log)
        if measure.latency_s is None:
            logger.warning("RaspberryPi4Device run produced no latency: %s", " | ".join(measure.serial_log[-5:]))
            return DeviceMetrics(ram_bytes=-1, flash_bytes=flash, latency_s=-1.0, arena_bytes=-1,
                                 error_code=HIL_ERROR_LATENCY)
        return DeviceMetrics(ram_bytes=-1 if peak_rss is None else peak_rss, flash_bytes=flash,
                             latency_s=measure.latency_s, arena_bytes=-1, error_code=HIL_ERROR_OK)
