# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for STM32 runtime parsing and serial-session helpers."""

from __future__ import annotations

import queue
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.microcontrollers import stm32_runtime  # noqa: E402


class _FakeSerial:
    """Small serial-port double used by STM32 runtime session tests."""

    def __init__(self, port: str, baud: int, timeout: float) -> None:
        """Capture serial settings and initialize an in-memory line queue.

        Parameters
        ----------
        port : str
            Serial port name used by the fake connection.
        baud : int
            Serial baud rate used by the fake connection.
        timeout : float
            Read timeout for the fake serial connection.
        """
        del timeout
        self.port = port
        self.baud = baud
        self.is_open = True
        self._lines: queue.Queue[bytes] = queue.Queue()
        self.writes: list[bytes] = []

    def readline(self) -> bytes:
        """Return the next queued line or an empty payload on timeout.

        Returns
        -------
        bytes
            Next byte response from the dummy serial stream.
        """
        if not self.is_open:
            return b""
        try:
            return self._lines.get(timeout=0.05)
        except queue.Empty:
            return b""

    def write(self, payload: bytes) -> int:
        """Record writes and report the written byte count.

        Parameters
        ----------
        payload : bytes
            Serial payload returned by the fake connection.

        Returns
        -------
        int
            Number of bytes accepted by the fake serial stream.
        """
        self.writes.append(payload)
        return len(payload)

    def flush(self) -> None:
        """Match the serial API without performing extra work."""
        return None

    def close(self) -> None:
        """Mark the fake port closed."""
        self.is_open = False

    def push_line(self, text: str) -> None:
        """Queue one newline-terminated UTF-8 line for later reads.

        Parameters
        ----------
        text : str
            Text returned by the fake serial connection.
        """
        self._lines.put(f"{text}\r\n".encode("utf-8"))


class _FakeMonitor:
    """History-based serial monitor double for runtime session tests."""

    def __init__(self, lines: list[str]) -> None:
        """Seed the monitor with a fixed transcript.

        Parameters
        ----------
        lines : list[str]
            Initial serial lines stored by the monitor test double.
        """
        self._lines = lines
        self.writes: list[str] = []

    def wait_for_match(self, predicate, timeout_s: float, stage: str, *, start_index: int = 0):
        """Scan transcript lines from ``start_index`` and return the next match.

        Parameters
        ----------
        predicate : object
            Predicate used to match serial monitor lines.
        timeout_s : float
            Timeout in seconds for the serial wait helper.
        stage : str
            Serial monitor stage label used in timeout messages.
        start_index : int
            Line index where serial matching should begin.

        Returns
        -------
        object
            Matched serial line returned by the wait helper.
        """
        del timeout_s, stage
        # The cursor-style ``start_index`` contract lets tests replay one shared
        # transcript across multiple sequential wait phases.
        for index in range(start_index, len(self._lines)):
            if predicate(self._lines[index]):
                return self._lines[index], index + 1
        return None, len(self._lines)

    def write_line(self, text: str) -> None:
        """Record writes emitted back to the DUT monitor.

        Parameters
        ----------
        text : str
            Text returned by the fake serial connection.
        """
        self.writes.append(text)

    def snapshot_lines(self) -> list[str]:
        """Return a shallow copy of the transcript history.

        Returns
        -------
        list[str]
            Serial lines returned by the snapshot helper.
        """
        return list(self._lines)


class STM32RuntimeParserTests(unittest.TestCase):
    """Tests covering STM32 runtime parser behavior."""

    def test_parse_runtime_lines_accepts_required_back_to_back_grammar(self) -> None:
        # Back-to-back runtime parsing should accept the required telemetry grammar.
        """Validate parse runtime lines accepts required back to back grammar."""
        lines = [
            "STM32_BOOT=START",
            "STM32_AI_INIT=OK",
            "DUT READY",
            "phase output: back_to_back",
            "runs: 10",
            "clock hz output: 600000000",
            "dwt cycles per inference output: 120000",
            "timer per inference output: 0.000200",
            "timer output: 0.000200",
            "inference seq output: 1",
            "STM32_AI_RUN=OK",
        ]

        telemetry = stm32_runtime.parse_stm32_runtime_lines(lines)

        self.assertAlmostEqual(telemetry.latency_s, 0.0002)
        self.assertEqual(telemetry.power_metrics["phase"], "back_to_back")
        self.assertEqual(telemetry.power_metrics["runs"], 10)
        self.assertAlmostEqual(telemetry.power_metrics["clock_hz"], 600000000.0)
        self.assertAlmostEqual(telemetry.power_metrics["dwt_cycles_per_inference"], 120000.0)
        self.assertEqual(telemetry.power_metrics["wake_recovery_us"], -1.0)
        self.assertEqual(telemetry.power_metrics["wake_overshoot_us"], -1.0)
        self.assertEqual(telemetry.power_metrics["rtc_sleep_total_ms"], -1.0)
        self.assertEqual(telemetry.power_metrics["deadline_miss_count"], -1)

    def test_parse_runtime_lines_rejects_fail_tokens(self) -> None:
        # Runtime parsing should reject explicit FAIL tokens instead of returning partial telemetry.
        """Validate parse runtime lines rejects fail tokens."""
        with self.assertRaises(stm32_runtime.STM32RuntimeProtocolError) as context:
            stm32_runtime.parse_stm32_runtime_lines(
                [
                    "STM32_AI_INIT=OK",
                    "DUT READY",
                    "STM32_AI_RUN=FAIL batch=0",
                ]
            )

        self.assertEqual(context.exception.kind, "runtime_model")

    def test_parse_runtime_lines_requires_latency_token(self) -> None:
        # Back-to-back runtime parsing should require the latency token before returning success.
        """Validate parse runtime lines requires latency token."""
        with self.assertRaises(stm32_runtime.STM32RuntimeProtocolError) as context:
            stm32_runtime.parse_stm32_runtime_lines(
                [
                    "STM32_AI_INIT=OK",
                    "DUT READY",
                    "phase output: back_to_back",
                    "runs: 10",
                    "clock hz output: 600000000",
                    "STM32_AI_RUN=OK",
                ]
            )

        self.assertEqual(context.exception.kind, "runtime_latency")

    def test_parse_runtime_lines_accepts_cadenced_grammar(self) -> None:
        # Cadenced runtime parsing should accept the extended cadence telemetry grammar.
        """Validate parse runtime lines accepts cadenced grammar."""
        lines = [
            "STM32_BOOT=START",
            "STM32_AI_INIT=OK",
            "phase output: cadenced",
            "rtc clock source output: LSE",
            "rtc clock hz nominal output: 32768",
            "cadence timing quality output: crystal",
            "stop mode variant output: system_stop_mainreg_wfi",
            "DUT READY",
            "runs: 10",
            "clock hz output: 600000000",
            "dwt cycles per inference output: 120000",
            "timer per inference output: 0.080000",
            "timer per window output: 2.000000",
            "wake recovery us: 1200",
            "wake overshoot us: 35",
            "rtc sleep total ms: 1500.000",
            "slept slots: 9",
            "stop entries: 9",
            "cadence deadline misses: 0",
            "inference seq output: 1",
            "STM32_AI_RUN=OK",
        ]

        telemetry = stm32_runtime.parse_stm32_runtime_lines(lines)

        self.assertAlmostEqual(telemetry.latency_s, 0.08)
        self.assertEqual(telemetry.power_metrics["phase"], "cadenced")
        self.assertEqual(telemetry.power_metrics["rtc_clock_source"], "LSE")
        self.assertEqual(telemetry.power_metrics["cadence_timing_quality"], "crystal")
        self.assertEqual(telemetry.power_metrics["stop_mode_variant"], "system_stop_mainreg_wfi")
        self.assertAlmostEqual(telemetry.power_metrics["rtc_sleep_total_ms"], 1500.0)
        self.assertEqual(telemetry.power_metrics["deadline_miss_count"], 0)
        self.assertEqual(telemetry.power_metrics["slept_slots"], 9)
        self.assertEqual(telemetry.power_metrics["stop_entries"], 9)


class SerialMonitorTests(unittest.TestCase):
    """Tests covering serial monitor behavior."""

    def test_serial_monitor_replays_history_and_writes_lines(self) -> None:
        # The serial monitor should replay prior history and capture new lines into the live log stream.
        """Validate serial monitor replays history and writes lines."""
        fake_serial = _FakeSerial("/dev/ttyACM0", 115200, 0.2)

        with patch("crest.microcontrollers.stm32_runtime.serial.Serial", return_value=fake_serial):
            monitor = stm32_runtime.SerialMonitor("/dev/ttyACM0", 115200, "dut")
            try:
                fake_serial.push_line("STM32_AI_INIT=OK")
                time.sleep(0.05)
                line, cursor = monitor.wait_for_match(
                    lambda value: value == "STM32_AI_INIT=OK",
                    0.2,
                    "STM32_AI_INIT=OK",
                )
                self.assertEqual(line, "STM32_AI_INIT=OK")
                fake_serial.push_line("DUT READY")
                line, _ = monitor.wait_for_match(
                    lambda value: value == "DUT READY",
                    0.2,
                    "DUT READY",
                    start_index=cursor,
                )
                self.assertEqual(line, "DUT READY")
                monitor.write_line("START")
            finally:
                monitor.close()

        self.assertEqual(fake_serial.writes, [b"START\n"])


class STM32RuntimeSessionTests(unittest.TestCase):
    """Tests covering STM32 runtime session behavior."""

    def test_execute_runtime_session_waits_for_init_then_ready_then_run(self) -> None:
        # Runtime sessions should wait for INIT, READY, then RUN so the harness protocol stays ordered.
        """Validate execute runtime session waits for init then ready then run."""
        monitor = _FakeMonitor(
            [
                "STM32_BOOT=START",
                "STM32_AI_INIT=OK",
                "phase output: back_to_back",
                "DUT READY",
                "runs: 5",
                "clock hz output: 600000000",
                "timer output: 0.000500",
                "STM32_AI_RUN=OK",
            ]
        )

        telemetry = stm32_runtime.execute_runtime_session(
            monitor,
            boot_timeout_s=1.0,
            run_timeout_s=1.0,
        )

        self.assertEqual(monitor.writes, ["START"])
        self.assertAlmostEqual(telemetry.latency_s, 0.0005)

    def test_execute_runtime_session_times_out_before_init(self) -> None:
        # Runtime sessions should time out cleanly if INIT never appears.
        """Validate execute runtime session times out before init."""
        monitor = _FakeMonitor(["STM32_BOOT=START"])

        with self.assertRaises(stm32_runtime.STM32RuntimeProtocolError) as context:
            stm32_runtime.execute_runtime_session(
                monitor,
                boot_timeout_s=0.1,
                run_timeout_s=0.1,
            )

        self.assertEqual(context.exception.kind, "runtime_start_timeout")


if __name__ == "__main__":
    unittest.main()
