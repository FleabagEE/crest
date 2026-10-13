# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Direct-serial runtime protocol helpers for the STM32 backend.

This module parses the UART protocol emitted by the STM32 CREST runtime and
provides a replayable serial monitor abstraction so callers can coordinate
``INIT -> READY -> START -> RUN`` session phases without losing earlier log
context.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import serial


logger = logging.getLogger(__name__)

DEFAULT_BOOT_TIMEOUT_S = 5.0
FLOAT_CAPTURE = r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
RUNS_RE = re.compile(r"^runs:\s*(?P<value>\d+)$", re.IGNORECASE)
DUT_FLOAT_PATTERNS = {
    "clock_hz": re.compile(rf"^clock hz output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "inference_seq": re.compile(rf"^inference seq output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "dwt_cycles_per_inference": re.compile(
        rf"^dwt cycles per inference output.*?:\s*{FLOAT_CAPTURE}$",
        re.IGNORECASE,
    ),
    "timer_output_s": re.compile(rf"^timer output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "timer_per_inference_s": re.compile(
        rf"^timer per inference output.*?:\s*{FLOAT_CAPTURE}$",
        re.IGNORECASE,
    ),
    "timer_per_window_s": re.compile(
        rf"^timer per window output.*?:\s*{FLOAT_CAPTURE}$",
        re.IGNORECASE,
    ),
    "wake_recovery_us": re.compile(rf"^wake recovery us:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "wake_overshoot_us": re.compile(rf"^wake overshoot us:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "slept_slots": re.compile(rf"^slept slots:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "stop_entries": re.compile(rf"^stop entries:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "rtc_sleep_total_ms": re.compile(rf"^rtc sleep total ms:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "deadline_miss_count": re.compile(
        rf"^cadence deadline misses:\s*{FLOAT_CAPTURE}$",
        re.IGNORECASE,
    ),
    "rtc_clock_hz_nominal": re.compile(
        rf"^rtc clock hz nominal output:\s*{FLOAT_CAPTURE}$",
        re.IGNORECASE,
    ),
}
DUT_STRING_PATTERNS = {
    "phase": re.compile(r"^phase output:\s*(?P<value>.+)$", re.IGNORECASE),
    "rtc_clock_source": re.compile(
        r"^rtc clock source output:\s*(?P<value>.+)$",
        re.IGNORECASE,
    ),
    "cadence_timing_quality": re.compile(
        r"^cadence timing quality output:\s*(?P<value>.+)$",
        re.IGNORECASE,
    ),
    "stop_mode_variant": re.compile(
        r"^stop mode variant output:\s*(?P<value>.+)$",
        re.IGNORECASE,
    ),
}


def build_backend_error_metrics(kind: str, detail: str) -> dict[str, Any]:
    """Build a flattened backend-error payload for metrics propagation.

    Parameters
    ----------
    kind : str
        Stable backend-specific error kind.
    detail : str
        Human-readable diagnostic detail.

    Returns
    -------
    dict[str, Any]
        Flat key/value payload that can travel through ``power_metrics`` and be
        copied into the final metrics dict.
    """
    return {
        "backend_error_kind": str(kind),
        "backend_error_detail": str(detail),
    }


class STM32RuntimeProtocolError(RuntimeError):
    """Raised when STM32 UART output does not satisfy the runtime protocol."""

    def __init__(self, *, kind: str, detail: str, serial_log: Sequence[str]) -> None:
        """Capture protocol failure context.

        Parameters
        ----------
        kind : str
            Stable backend-specific error kind.
        detail : str
            Human-readable failure detail.
        serial_log : Sequence[str]
            Serial lines captured up to the point of failure.
        """
        super().__init__(detail)
        self.kind = kind
        self.detail = detail
        self.serial_log = list(serial_log)


@dataclass(frozen=True)
class STM32RuntimeTelemetry:
    """Parsed direct-serial telemetry from one STM32 runtime attempt.

    Parameters
    ----------
    latency_s : float
        Selected canonical latency in seconds.
    serial_log : list[str]
        Complete DUT serial log captured for the session.
    power_metrics : dict[str, Any]
        Parsed telemetry fields and optional backend detail keys.

    Attributes
    ----------
    latency_s : float
        Selected canonical latency in seconds.
    serial_log : list[str]
        Complete DUT serial log captured for the session.
    power_metrics : dict[str, Any]
        Parsed telemetry fields and optional backend detail keys.
    """

    latency_s: float
    serial_log: list[str]
    power_metrics: dict[str, Any]


class SerialMonitor:
    """Background serial reader with replayable line matching and writes."""

    def __init__(self, port: str, baud: int, label: str) -> None:
        """Open the serial port and start the background reader thread.

        Parameters
        ----------
        port : str
            Serial device path.
        baud : int
            Serial baud rate.
        label : str
            Human-readable stream label used in logs and errors.
        """
        self.port = port
        self.baud = baud
        self.label = label
        self._serial = serial.Serial(port, baud, timeout=0.2)
        reset_input = getattr(self._serial, "reset_input_buffer", None)
        if callable(reset_input):
            try:
                reset_input()
            except serial.SerialException:
                logger.debug("Unable to reset input buffer for %s.", label, exc_info=True)
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._reader_error: str | None = None
        self._reader_closed = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def __enter__(self) -> "SerialMonitor":
        """Return ``self`` for context-manager use.

        Returns
        -------
        "SerialMonitor"
            Active monitor instance.
        """
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Close the serial session when leaving a context-manager block.

        Parameters
        ----------
        exc_type : object
            Exception type received from the context manager protocol.
        exc : object
            Exception instance received from the context manager protocol.
        traceback : object
            Traceback object received from the context manager protocol.
        """
        del exc_type, exc, traceback
        self.close()

    def _reader(self) -> None:
        """Continuously read serial lines into the in-memory log."""
        try:
            while not self._stop.is_set():
                raw = self._serial.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    continue
                with self._condition:
                    self._lines.append(line)
                    self._condition.notify_all()
        except (serial.SerialException, OSError, TypeError) as exc:
            with self._condition:
                if not self._stop.is_set():
                    self._reader_error = str(exc)
                self._reader_closed = True
                self._condition.notify_all()
        finally:
            with self._condition:
                self._reader_closed = True
                self._condition.notify_all()

    def write_line(self, text: str) -> None:
        """Write one line to the monitored serial port.

        Parameters
        ----------
        text : str
            Command payload without a trailing newline.
        """
        payload = f"{text}\n".encode("utf-8")
        self._serial.write(payload)
        self._serial.flush()

    def wait_for_match(
        self,
        predicate: Callable[[str], bool],
        timeout_s: float,
        stage: str,
        *,
        start_index: int = 0,
    ) -> tuple[str | None, int]:
        """Wait for a matching line while preserving already-captured history.

        Parameters
        ----------
        predicate : callable[[str], bool]
            Predicate applied to each decoded serial line.
        timeout_s : float
            Maximum time to wait before returning ``None``.
        stage : str
            Human-readable label for the awaited event.
        start_index : int, default=0
            History offset from which matching should begin.

        Returns
        -------
        tuple[str | None, int]
            Matched line and the next cursor index to use for subsequent waits.

        Raises
        ------
        RuntimeError
            If the serial reader dies unexpectedly while waiting.
        """
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        cursor = max(int(start_index), 0)
        with self._condition:
            while True:
                while cursor < len(self._lines):
                    line = self._lines[cursor]
                    cursor += 1
                    logger.info("[%s] %s", self.label, line)
                    if predicate(line):
                        return line, cursor
                if self._reader_error is not None and self._reader_closed:
                    raise RuntimeError(
                        f"{self.label} serial reader exited while waiting for {stage}: {self._reader_error}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None, cursor
                self._condition.wait(timeout=min(0.25, remaining))

    def wait_for(self, predicate: Callable[[str], bool], timeout_s: float, stage: str) -> str | None:
        """Compatibility wrapper that waits from the start of the log.

        Parameters
        ----------
        predicate : callable[[str], bool]
            Predicate applied to each decoded serial line.
        timeout_s : float
            Maximum time to wait before returning ``None``.
        stage : str
            Human-readable label for the awaited event.

        Returns
        -------
        str | None
            The first matching line or ``None`` on timeout.
        """
        match, _ = self.wait_for_match(predicate, timeout_s, stage, start_index=0)
        return match

    def snapshot_lines(self) -> list[str]:
        """Return a thread-safe copy of all captured serial lines.

        Returns
        -------
        list[str]
            Complete serial history up to the moment of the call.
        """
        with self._lock:
            return list(self._lines)

    def close(self) -> None:
        """Stop the reader thread and close the serial port."""
        self._stop.set()
        try:
            if getattr(self._serial, "is_open", False):
                self._serial.close()
        finally:
            self._thread.join(timeout=2.0)


def _parse_runs(lines: Sequence[str]) -> int | None:
    """Return the last parsed ``runs:`` value from a serial log.

    Parameters
    ----------
    lines : Sequence[str]
        Serial lines to inspect.

    Returns
    -------
    int | None
        Parsed run count when present, otherwise ``None``.
    """
    parsed: int | None = None
    for line in lines:
        match = RUNS_RE.match(str(line).strip())
        if match is not None:
            parsed = int(match.group("value"))
    return parsed


def _parse_last_float_fields(
    lines: Sequence[str],
    patterns: dict[str, re.Pattern[str]],
) -> dict[str, float]:
    """Parse the last matching float for each telemetry field.

    Parameters
    ----------
    lines : Sequence[str]
        Serial lines to inspect.
    patterns : dict[str, re.Pattern[str]]
        Field-name to regex mapping.

    Returns
    -------
    dict[str, float]
        Parsed float values keyed by field name.
    """
    parsed: dict[str, float] = {}
    for line in lines:
        stripped = str(line).strip()
        for key, pattern in patterns.items():
            match = pattern.match(stripped)
            if match is not None:
                parsed[key] = float(match.group("value"))
    return parsed


def _parse_last_string_fields(
    lines: Sequence[str],
    patterns: dict[str, re.Pattern[str]],
) -> dict[str, str]:
    """Parse the last matching string for each telemetry field.

    Parameters
    ----------
    lines : Sequence[str]
        Serial lines to inspect.
    patterns : dict[str, re.Pattern[str]]
        Field-name to regex mapping.

    Returns
    -------
    dict[str, str]
        Parsed string values keyed by field name.
    """
    parsed: dict[str, str] = {}
    for line in lines:
        stripped = str(line).strip()
        for key, pattern in patterns.items():
            match = pattern.match(stripped)
            if match is not None:
                parsed[key] = match.group("value").strip()
    return parsed


def _is_finite_positive(value: float | None) -> bool:
    """Return whether a parsed float is present and strictly positive.

    Parameters
    ----------
    value : float | None
        Parsed float candidate.

    Returns
    -------
    bool
        ``True`` when the value is finite and positive.
    """
    return value is not None and math.isfinite(value) and value > 0.0


def _init_terminal(line: str) -> bool:
    """Return whether a line ends the init stage.

    Parameters
    ----------
    line : str
        Serial line to classify.

    Returns
    -------
    bool
        ``True`` when the line is an init success or failure token.
    """
    stripped = line.strip()
    return stripped == "STM32_AI_INIT=OK" or stripped.startswith("STM32_AI_INIT=FAIL")


def _run_terminal(line: str) -> bool:
    """Return whether a line ends the measured runtime stage.

    Parameters
    ----------
    line : str
        Serial line to classify.

    Returns
    -------
    bool
        ``True`` when the line is a runtime success/failure terminal token.
    """
    stripped = line.strip()
    return (
        stripped == "STM32_AI_RUN=OK"
        or stripped.startswith("STM32_AI_RUN=FAIL")
        or stripped == "STM32_AI_START=TIMEOUT"
    )


def _raise_failure_from_lines(lines: Sequence[str]) -> None:
    """Raise a protocol error when captured serial lines already show failure.

    Parameters
    ----------
    lines : Sequence[str]
        Serial lines to inspect.

    Raises
    ------
    STM32RuntimeProtocolError
        If a known failure token is present.
    """
    for raw_line in lines:
        line = str(raw_line).strip()
        if line.startswith("STM32_AI_INIT=FAIL"):
            raise STM32RuntimeProtocolError(
                kind="runtime_model",
                detail=line,
                serial_log=lines,
            )
        if line == "STM32_AI_RUN=FAIL measurement_integrity:stop_sleep":
            raise STM32RuntimeProtocolError(
                kind="measurement_integrity", detail=line, serial_log=lines,
            )
        if line.startswith("STM32_AI_RUN=FAIL"):
            raise STM32RuntimeProtocolError(
                kind="runtime_model",
                detail=line,
                serial_log=lines,
            )
        if line == "STM32_AI_START=TIMEOUT":
            raise STM32RuntimeProtocolError(
                kind="runtime_start_timeout",
                detail=line,
                serial_log=lines,
            )


def parse_stm32_runtime_lines(lines: Sequence[str]) -> STM32RuntimeTelemetry:
    """Parse one STM32 runtime telemetry session.

    Parameters
    ----------
    lines : Sequence[str]
        Complete serial log captured from the DUT.

    Returns
    -------
    STM32RuntimeTelemetry
        Parsed latency plus flattened telemetry fields.

    Raises
    ------
    STM32RuntimeProtocolError
        If required success tokens or required telemetry fields are missing or
        inconsistent with the STM32 runtime contract.
    """
    serial_log = [str(line) for line in lines]
    _raise_failure_from_lines(serial_log)
    floats = _parse_last_float_fields(serial_log, DUT_FLOAT_PATTERNS)
    strings = _parse_last_string_fields(serial_log, DUT_STRING_PATTERNS)
    # Validate the coarse session shape first so later field-specific failures
    # can assume the runtime reached a known protocol phase.
    phase = strings.get("phase", "").strip().lower()
    if phase not in {"back_to_back", "cadenced"}:
        raise STM32RuntimeProtocolError(
            kind="runtime_protocol",
            detail=(
                "Expected 'phase output: back_to_back' or 'phase output: cadenced' "
                f"but saw {phase or '<missing>'}."
            ),
            serial_log=serial_log,
        )
    runs = _parse_runs(serial_log)
    if runs is None or runs <= 0:
        raise STM32RuntimeProtocolError(
            kind="runtime_protocol",
            detail="Missing or invalid 'runs:' telemetry.",
            serial_log=serial_log,
        )
    clock_hz = floats.get("clock_hz")
    if not _is_finite_positive(clock_hz):
        raise STM32RuntimeProtocolError(
            kind="runtime_protocol",
            detail="Missing or invalid 'clock hz output' telemetry.",
            serial_log=serial_log,
        )
    timer_output_s = floats.get("timer_output_s")
    timer_per_inference_s = floats.get("timer_per_inference_s")
    # Back-to-back mode reports aggregate timer output for the whole run,
    # whereas cadenced mode may only expose per-inference timing.
    if phase == "back_to_back" and _is_finite_positive(timer_output_s):
        latency_s = float(timer_output_s)
    elif _is_finite_positive(timer_per_inference_s):
        latency_s = float(timer_per_inference_s)
    else:
        raise STM32RuntimeProtocolError(
            kind="runtime_latency",
            detail="Missing required latency telemetry for the reported STM32 phase.",
            serial_log=serial_log,
        )
    power_metrics = {
        "clock_hz": float(clock_hz),
        "sequence": float(floats.get("inference_seq", -1.0)),
        "dwt_cycles_per_inference": float(floats.get("dwt_cycles_per_inference", -1.0)),
        "timer_output_s": float(timer_output_s) if timer_output_s is not None else -1.0,
        "timer_per_inference_s": (
            float(timer_per_inference_s) if timer_per_inference_s is not None else -1.0
        ),
        "timer_per_window_s": float(floats.get("timer_per_window_s", -1.0)),
        "wake_recovery_us": float(floats.get("wake_recovery_us", -1.0)),
        "wake_overshoot_us": float(floats.get("wake_overshoot_us", -1.0)),
        "slept_slots": float(floats.get("slept_slots", -1.0)),
        "stop_entries": float(floats.get("stop_entries", -1.0)),
        "rtc_sleep_total_ms": float(floats.get("rtc_sleep_total_ms", -1.0)),
        "deadline_miss_count": float(floats.get("deadline_miss_count", -1.0)),
        "rtc_clock_hz_nominal": float(floats.get("rtc_clock_hz_nominal", -1.0)),
        "phase": phase,
        "rtc_clock_source": strings.get("rtc_clock_source"),
        "cadence_timing_quality": strings.get("cadence_timing_quality"),
        "stop_mode_variant": strings.get("stop_mode_variant"),
        "runs": runs,
    }
    return STM32RuntimeTelemetry(
        latency_s=latency_s,
        serial_log=serial_log,
        power_metrics=power_metrics,
    )


def execute_runtime_session(
    monitor: SerialMonitor,
    *,
    boot_timeout_s: float,
    run_timeout_s: float,
) -> STM32RuntimeTelemetry:
    """Drive one direct-serial runtime session using an open monitor.

    Parameters
    ----------
    monitor : SerialMonitor
        Already-open serial monitor that started capturing before firmware ran.
    boot_timeout_s : float
        Timeout for pre-START protocol stages.
    run_timeout_s : float
        Timeout for post-START runtime completion.

    Returns
    -------
    STM32RuntimeTelemetry
        Parsed runtime telemetry for the session.

    Raises
    ------
    STM32RuntimeProtocolError
        If the DUT fails to reach the required protocol milestones.
    RuntimeError
        If the serial reader exits unexpectedly.

    Notes
    -----
    The expected token order is ``STM32_AI_INIT=*`` followed by ``DUT READY``,
    then a host-sent ``START`` line, and finally ``STM32_AI_RUN=*`` plus the
    telemetry fields parsed by :func:`parse_stm32_runtime_lines`.
    """
    # Phase 1: wait for the firmware to boot and publish its init status.
    init_line, cursor = monitor.wait_for_match(
        _init_terminal,
        boot_timeout_s,
        "STM32_AI_INIT result",
        start_index=0,
    )
    if init_line is None:
        raise STM32RuntimeProtocolError(
            kind="runtime_start_timeout",
            detail="Timed out waiting for STM32_AI_INIT result.",
            serial_log=monitor.snapshot_lines(),
        )
    _raise_failure_from_lines([init_line])
    # Phase 2: wait for the DUT to signal that it is armed for the run.
    ready_line, cursor = monitor.wait_for_match(
        lambda line: line.strip() == "DUT READY",
        boot_timeout_s,
        "DUT READY",
        start_index=cursor,
    )
    if ready_line is None:
        raise STM32RuntimeProtocolError(
            kind="runtime_start_timeout",
            detail="Timed out waiting for DUT READY.",
            serial_log=monitor.snapshot_lines(),
        )
    # Phase 3: explicitly start the measurement window from the host side.
    monitor.write_line("START")
    # Phase 4: wait for the terminal run token and parse the full session log.
    run_line, _ = monitor.wait_for_match(
        _run_terminal,
        run_timeout_s,
        "STM32_AI_RUN result",
        start_index=cursor,
    )
    if run_line is None:
        # START was sent: a slow/hung measured run remains a candidate timeout.
        # Missing terminal telemetry alone cannot distinguish a later transport loss.
        raise STM32RuntimeProtocolError(
            kind="runtime_timeout",
            detail="Timed out waiting for STM32_AI_RUN result.",
            serial_log=monitor.snapshot_lines(),
        )
    _raise_failure_from_lines([run_line])
    return parse_stm32_runtime_lines(monitor.snapshot_lines())


def measure_direct_serial(
    *,
    serial_port: str,
    baud_rate: int,
    boot_timeout_s: float | None,
    run_timeout_s: float,
) -> STM32RuntimeTelemetry:
    """Open a DUT serial session and run the direct-serial protocol.

    Parameters
    ----------
    serial_port : str
        DUT serial port path.
    baud_rate : int
        DUT UART baud rate.
    boot_timeout_s : float | None
        Timeout for init and ready tokens. ``None`` falls back to 5 seconds.
    run_timeout_s : float
        Timeout for the post-``START`` runtime stage.

    Returns
    -------
    STM32RuntimeTelemetry
        Parsed runtime telemetry for the session.
    """
    with SerialMonitor(serial_port, baud_rate, "dut") as monitor:
        return execute_runtime_session(
            monitor,
            boot_timeout_s=DEFAULT_BOOT_TIMEOUT_S if boot_timeout_s is None else float(boot_timeout_s),
            run_timeout_s=float(run_timeout_s),
        )
