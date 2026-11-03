# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Board selection shared by the MCU bench scripts (``--board``).

Both boards use the same staged latency sketch and results format; this
module maps ``--board`` to the device class, board id and per-board
constants (static RAM without the arena, measured on each board).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass(frozen=True)
class BenchBoard:
    """One benchable MCU board."""

    key: str
    board_id: str
    factory: Callable[..., Any]
    # Static RAM of the latency sketch without the tensor arena (bytes),
    # measured from a compile; used by the arena/RAM predictions.
    static_ram_base: int
    core_version: str
    clock_hz: int


def _rp2040(**kwargs):
    from crest.microcontrollers.arduino_nano_rp2040 import ArduinoNanoRP2040Device

    return ArduinoNanoRP2040Device(**kwargs)


def _esp32s3(**kwargs):
    from crest.microcontrollers.xiao_esp32s3 import XiaoESP32S3Device

    return XiaoESP32S3Device(**kwargs)


BOARDS: Dict[str, BenchBoard] = {
    "rp2040": BenchBoard("rp2040", "nano_rp2040_connect", _rp2040, 50_512, "arduino:mbed_nano@4.6.0", 125_000_000),
    # ESP32-S3: static base measured from a compile of the latency sketch
    # (internal-SRAM arena) on 2026-11-02, see memory.md section 14.
    "xiao_esp32s3": BenchBoard("xiao_esp32s3", "xiao_esp32s3", _esp32s3, 45_780, "esp32:esp32@3.3.12", 240_000_000),
}


def make_device(board: str, *, serial_port: Optional[str] = None, device_options: Optional[dict] = None):
    """Construct the device wrapper for ``board``."""
    return BOARDS[board].factory(serial_port=serial_port, device_options=dict(device_options or {}))
