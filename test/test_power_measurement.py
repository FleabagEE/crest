# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the INA228 power-measurement tools and the measured power source."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
ENERGY_DIR = ROOT_DIR / "analysis_scripts" / "energy"
if str(ENERGY_DIR) not in sys.path:
    sys.path.insert(0, str(ENERGY_DIR))

import estimate_energy as ee  # noqa: E402
import ina228_reader as ina  # noqa: E402
import record_power_measurement as rec  # noqa: E402


class INA228DecodeTests(unittest.TestCase):
    """Tests covering INA228 register decoding (no hardware)."""

    def test_register_decoding(self):
        # 20-bit signed data sits in bits 23..4; LSBs follow the INA228 datasheet.
        """Validate VSHUNT/VBUS decoding and ADC_CONFIG packing."""
        self.assertAlmostEqual(ina.decode_vbus(25600 << 4), 25600 * 195.3125e-6)  # 5.0 V
        self.assertAlmostEqual(ina.decode_vshunt(1000 << 4, 1), 1000 * 78.125e-9)
        self.assertAlmostEqual(ina.decode_vshunt(((1 << 20) - 1000) << 4, 1), -1000 * 78.125e-9)
        self.assertAlmostEqual(ina.decode_vshunt(1000 << 4, 0), 1000 * 312.5e-9)
        self.assertEqual(ina.adc_config_word(), (0xB << 12) | (5 << 9) | (5 << 6) | (0 << 3) | 2)

    def test_summarize(self):
        # Current = V_shunt / R, power = V_bus * I, resolution = one current LSB at the mean voltage.
        """Validate window summaries."""
        samples = [(0.0, 5.0, 0.0003), (1.0, 5.0, 0.0003), (2.0, 5.0, 0.0003)]
        s = ina.summarize(samples, 0.015, 1)
        self.assertAlmostEqual(s["amps_mean"], 0.02)
        self.assertAlmostEqual(s["power_w_mean"], 0.1)
        self.assertAlmostEqual(s["current_lsb_a"], 78.125e-9 / 0.015)
        self.assertAlmostEqual(s["power_resolution_w"], 5.0 * 78.125e-9 / 0.015)
        self.assertEqual((s["n"], s["seconds"]), (3, 2.0))


class RecordAndMeasuredSourceTests(unittest.TestCase):
    """Tests covering recording windows and using them in the estimator."""

    LINE = ("INA228_WINDOW label=rp2040_inference_nina_reset window=1 t_start=1.000 die_temp_c=30.00 "
            "shunt_ohms=0.015 adc_range=1 n=1800 seconds=60 rate_hz=30 volts_mean=5.05 amps_mean=0.02 "
            "amps_std=0.0001 amps_min=0.019 amps_max=0.021 power_w_mean=0.101 power_w_std=0.0005 "
            "current_lsb_a=5.2e-06 power_resolution_w=2.6e-05")

    def test_record_round_trip_keeps_header(self):
        # Recording appends a sample, keeps the comment header, and rejects unknown states.
        """Validate record()."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yaml"
            shutil.copyfile(ENERGY_DIR / "board_power_measured.yaml", path)
            window = rec.parse_window_line(self.LINE)
            self.assertEqual((window["label"], window["power_w_mean"]), ("rp2040_inference_nina_reset", 0.101))
            rec.record(path, "rp2040", "inference_nina_reset", [window], "test")
            text = path.read_text()
            self.assertTrue(text.startswith("# Copyright"))
            data = yaml.safe_load(text)
            self.assertEqual(len(data["boards"]["rp2040"]["states"]["inference_nina_reset"]), 1)
            with self.assertRaises(KeyError):
                rec.record(path, "rp2040", "deep_sleep", [window])
        with self.assertRaises(ValueError):
            rec.parse_window_line("runs: 1")

    def test_apply_measured_overrides_only_measured_values(self):
        # Measured states replace their datasheet value; everything else keeps the datasheet and says so.
        """Validate apply_measured()."""
        power = ee.load_power()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yaml"
            shutil.copyfile(ENERGY_DIR / "board_power_measured.yaml", path)
            merged, provenance = ee.apply_measured(power, path)
            self.assertEqual(merged, power)
            self.assertTrue(all("datasheet" in line or "not measured" in line for line in provenance))
            window = rec.parse_window_line(self.LINE)
            rec.record(path, "rp2040", "inference_nina_reset", [window, dict(window, power_w_mean=0.105)])
            rec.record(path, "rpi4", "idle", [dict(window, power_w_mean=2.75, power_resolution_w=0.0)])
            merged, provenance = ee.apply_measured(power, path)
        low, high = merged["rp2040"].active[1]
        self.assertAlmostEqual(low, 0.101 - 2.6e-05)
        self.assertAlmostEqual(high, 0.105 + 2.6e-05)
        self.assertEqual(merged["rp2040"].listen, power["rp2040"].listen)
        self.assertEqual(merged["rp2040"].sleep, power["rp2040"].sleep)
        self.assertEqual(merged["rpi4"].listen, (2.75, 2.75))
        self.assertEqual(merged["rpi4"].sleep, (2.75, 2.75))
        self.assertIn("rp2040.active: measured inference_nina_reset (2 windows)", provenance)
        self.assertIn("rp2040.listen: datasheet (no measurement)", provenance)


if __name__ == "__main__":
    unittest.main()
