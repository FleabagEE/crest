# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Measure a USB-powered board's VBUS power with an INA228 read by the Pi 4.

Wiring (see README.md, "Power measurement"):

- Only USB **VBUS** goes through the INA228: laptop-side USB-C breakout
  VBUS -> INA228 VIN+ -> shunt -> VIN- -> board-side breakout (Adafruit 5978)
  VBUS. D+, D- and GND pass straight through, so the board still enumerates,
  uploads and prints serial markers on the laptop.
- The INA228's VBUS sense pin must see the load side (VIN-).
- I2C to the Pi 4 header: 3.3 V (pin 1), SDA (pin 3), SCL (pin 5), GND
  (pin 6). The Pi GND and the breakout GND are common.

This script runs **on the Pi**, uses only the standard library (Linux
``/dev/i2c-N`` via ``fcntl``), and reads the raw ``VSHUNT`` and ``VBUS``
registers. Current is ``V_shunt / R_shunt`` and power is ``V_bus * I``,
computed per sample, so ``SHUNT_CAL`` is not needed. ADC range is
+-40.96 mV (78.125 nV/LSB; with a 15 mOhm shunt that is 5.21 uA/LSB and a
2.73 A full scale).

Output: one ``key=value`` summary line per window (``--windows``, each
``--seconds`` long), optionally all samples as CSV (``--csv``). The summary
is what ``record_power_measurement.py`` stores in
``board_power_measured.yaml``.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import statistics
import sys
import time
from typing import Dict, List, Tuple

I2C_SLAVE = 0x0703

REG_CONFIG = 0x00
REG_ADC_CONFIG = 0x01
REG_VSHUNT = 0x04
REG_VBUS = 0x05
REG_DIETEMP = 0x06
REG_MANUFACTURER_ID = 0x3E
REG_DEVICE_ID = 0x3F

MANUFACTURER_TI = 0x5449  # "TI"
DEVICE_ID_INA228 = 0x228  # upper 12 bits of DEVICE_ID

VSHUNT_LSB_V = {0: 312.5e-9, 1: 78.125e-9}  # by CONFIG.ADCRANGE
VBUS_LSB_V = 195.3125e-6
DIETEMP_LSB_C = 7.8125e-3

# ADC_CONFIG: MODE=0xB (continuous shunt + bus), VBUSCT=VSHCT=1052 us, VTCT=50 us, AVG=16.
ADC_MODE_CONT_SHUNT_BUS = 0xB
CT_1052_US = 0x5
AVG_16 = 0x2


def twos_complement(value: int, bits: int) -> int:
    """Interpret ``value`` as a ``bits``-wide two's-complement integer."""
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def decode_vshunt(raw24: int, adc_range: int) -> float:
    """VSHUNT register (24 bits, data in bits 23..4, signed) to volts."""
    return twos_complement(raw24 >> 4, 20) * VSHUNT_LSB_V[adc_range]


def decode_vbus(raw24: int) -> float:
    """VBUS register (24 bits, data in bits 23..4) to volts."""
    return twos_complement(raw24 >> 4, 20) * VBUS_LSB_V


def adc_config_word(mode: int = ADC_MODE_CONT_SHUNT_BUS, vbusct: int = CT_1052_US, vshct: int = CT_1052_US,
                    vtct: int = 0, avg: int = AVG_16) -> int:
    """ADC_CONFIG register value (MODE[15:12] VBUSCT[11:9] VSHCT[8:6] VTCT[5:3] AVG[2:0])."""
    return (mode << 12) | (vbusct << 9) | (vshct << 6) | (vtct << 3) | avg


def summarize(samples: List[Tuple[float, float, float]], shunt_ohms: float, adc_range: int) -> Dict[str, float]:
    """Summary of ``(t, v_bus, v_shunt)`` samples."""
    volts = [v for _, v, _ in samples]
    amps = [s / shunt_ohms for _, _, s in samples]
    watts = [v * i for v, i in zip(volts, amps)]
    seconds = samples[-1][0] - samples[0][0] if len(samples) > 1 else 0.0
    current_lsb = VSHUNT_LSB_V[adc_range] / shunt_ohms
    return {
        "n": len(samples),
        "seconds": round(seconds, 3),
        "rate_hz": round((len(samples) - 1) / seconds, 2) if seconds > 0 else 0.0,
        "volts_mean": statistics.fmean(volts),
        "amps_mean": statistics.fmean(amps),
        "amps_std": statistics.pstdev(amps),
        "amps_min": min(amps),
        "amps_max": max(amps),
        "power_w_mean": statistics.fmean(watts),
        "power_w_std": statistics.pstdev(watts),
        "current_lsb_a": current_lsb,
        # Resolution of the mean power: one current LSB at the mean bus voltage.
        "power_resolution_w": current_lsb * statistics.fmean(volts),
    }


class INA228:
    """Minimal INA228 access over Linux i2c-dev (standard library only)."""

    def __init__(self, bus: int, address: int) -> None:
        """Open ``/dev/i2c-<bus>`` and select ``address``."""
        self.fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
        fcntl.ioctl(self.fd, I2C_SLAVE, address)

    def close(self) -> None:
        """Close the bus."""
        os.close(self.fd)

    def read(self, register: int, nbytes: int) -> int:
        """Read a big-endian register of ``nbytes``."""
        os.write(self.fd, bytes([register]))
        return int.from_bytes(os.read(self.fd, nbytes), "big")

    def write16(self, register: int, value: int) -> None:
        """Write a 16-bit register."""
        os.write(self.fd, bytes([register, (value >> 8) & 0xFF, value & 0xFF]))

    def identify(self) -> Tuple[int, int]:
        """Return ``(manufacturer_id, device_id >> 4)``."""
        return self.read(REG_MANUFACTURER_ID, 2), self.read(REG_DEVICE_ID, 2) >> 4

    def configure(self, adc_range: int) -> None:
        """Set ADCRANGE and continuous shunt+bus conversion with averaging."""
        self.write16(REG_CONFIG, (adc_range & 1) << 4)
        self.write16(REG_ADC_CONFIG, adc_config_word())

    def sample(self) -> Tuple[float, int, int]:
        """Return ``(timestamp, raw_vbus, raw_vshunt)``."""
        return time.time(), self.read(REG_VBUS, 3), self.read(REG_VSHUNT, 3)

    def die_temp_c(self) -> float:
        """Die temperature in degrees C."""
        return twos_complement(self.read(REG_DIETEMP, 2), 16) * DIETEMP_LSB_C


def main() -> int:
    """Measure ``--windows`` windows and print one summary line each."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bus", type=int, default=1, help="I2C bus (Pi header pins 3/5 = bus 1).")
    parser.add_argument("--address", type=lambda s: int(s, 0), default=0x40, help="INA228 address (default 0x40).")
    parser.add_argument("--shunt-ohms", type=float, default=0.015, help="Shunt resistance (Adafruit INA228: 0.015).")
    parser.add_argument("--adc-range", type=int, choices=(0, 1), default=1, help="1: +-40.96 mV (default), 0: +-163.84 mV.")
    parser.add_argument("--seconds", type=float, default=60.0, help="Length of one window.")
    parser.add_argument("--windows", type=int, default=3, help="Consecutive windows (repeats).")
    parser.add_argument("--wait-s", type=float, default=0.0, help="Wait before the first window.")
    parser.add_argument("--label", default="", help="Free text echoed in every summary line (e.g. variant).")
    parser.add_argument("--csv", default=None, help="Optional path for all samples (t, v_bus, v_shunt, amps, watts).")
    args = parser.parse_args()

    dev = INA228(args.bus, args.address)
    try:
        manufacturer, device = dev.identify()
        if manufacturer != MANUFACTURER_TI or device != DEVICE_ID_INA228:
            print(f"ERROR not an INA228 at {args.address:#04x}: manufacturer={manufacturer:#06x} device={device:#05x}")
            return 2
        dev.configure(args.adc_range)
        time.sleep(0.2)
        if args.wait_s > 0:
            time.sleep(args.wait_s)
        csv_handle = open(args.csv, "w", encoding="utf-8") if args.csv else None
        if csv_handle:
            csv_handle.write("window,t,v_bus,v_shunt,amps,watts\n")
        for window in range(1, args.windows + 1):
            samples: List[Tuple[float, float, float]] = []
            end = time.time() + args.seconds
            while time.time() < end:
                t, raw_bus, raw_shunt = dev.sample()
                v_bus, v_shunt = decode_vbus(raw_bus), decode_vshunt(raw_shunt, args.adc_range)
                samples.append((t, v_bus, v_shunt))
                if csv_handle:
                    amps = v_shunt / args.shunt_ohms
                    csv_handle.write(f"{window},{t:.4f},{v_bus:.6f},{v_shunt:.9f},{amps:.7f},{v_bus * amps:.7f}\n")
                time.sleep(0.02)
            s = summarize(samples, args.shunt_ohms, args.adc_range)
            fields = " ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items())
            print(f"INA228_WINDOW label={args.label or '-'} window={window} t_start={samples[0][0]:.3f} "
                  f"die_temp_c={dev.die_temp_c():.2f} shunt_ohms={args.shunt_ohms} adc_range={args.adc_range} {fields}",
                  flush=True)
        if csv_handle:
            csv_handle.close()
    finally:
        dev.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
