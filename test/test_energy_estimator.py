# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the energy estimator and board power parameters."""

import math
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

POWER = {
    "rp2040": ee.BoardPower(active={1: (0.05, 0.4)}, listen=(0.04, 0.4), sleep=(0.004, 0.16)),
    "rpi4": ee.BoardPower(active={1: (3.6, 5.1), 4: (6.4, 7.3)}, listen=(2.7, 2.9), sleep=(2.7, 2.9)),
}


def _pair(model="m", board="rp2040", threads=1, window=1.0, t=0.1, status="ok", acc=0.7):
    """Build a Pair with defaults."""
    return ee.Pair(model=model, board=board, threads=threads, window_s=window, latency_s=t, status=status,
                   val_acc=acc, val_macro_f1=acc)


class EnergyModelTests(unittest.TestCase):
    """Tests covering the per-inference energy formula."""

    def test_formula_branches(self):
        # S <= L listens for the rest of the stride; S > L listens for L and sleeps the remainder.
        """Validate both branches of the energy model."""
        self.assertAlmostEqual(ee.energy_per_inference(0.2, 1.0, 1.0, 2.0, 1.0, 0.1), 2.0 * 0.2 + 1.0 * 0.8)
        self.assertAlmostEqual(ee.energy_per_inference(0.2, 2.0, 1.0, 2.0, 1.0, 0.1), 2.0 * 0.2 + 1.0 * 0.8)
        self.assertAlmostEqual(ee.energy_per_inference(0.5, 1.0, 4.0, 2.0, 1.0, 0.1), 2.0 * 0.5 + 1.0 * 1.0 + 0.1 * 2.5)
        with self.assertRaises(ValueError):
            ee.energy_per_inference(1.5, 1.0, 1.0, 1, 1, 1)
        with self.assertRaises(ValueError):
            ee.energy_per_inference(1.5, 1.0, 2.0, 1, 1, 1)  # t <= S but t > S - L

    def test_estimate_feasibility_and_bounds(self):
        # Window mismatch, bad status and slow pairs are excluded; feasible pairs get low/high energy and E/S power.
        """Validate estimate()."""
        self.assertFalse(ee.estimate(_pair(window=2.0), 1.0, 1.0, POWER).applicable)
        self.assertEqual(ee.estimate(_pair(status="infeasible:arena"), 1.0, 1.0, POWER).reason,
                         "status infeasible:arena")
        self.assertEqual(ee.estimate(_pair(t=1.2), 1.0, 1.0, POWER).reason, "too slow (t > S)")
        self.assertEqual(ee.estimate(_pair(t=1.2), 1.0, 2.0, POWER).reason,
                         "too slow (t > S - L, no compute/listen overlap)")
        ok = ee.estimate(_pair(board="rpi4", threads=4, t=0.01), 1.0, 2.0, POWER)
        self.assertTrue(ok.feasible)
        self.assertAlmostEqual(ok.energy_j["low"], 6.4 * 0.01 + 2.7 * 1.0 + 2.7 * 0.99)
        self.assertAlmostEqual(ok.energy_j["high"], 7.3 * 0.01 + 2.9 * 1.0 + 2.9 * 0.99)
        self.assertAlmostEqual(ok.avg_power_w["low"], ok.energy_j["low"] / 2.0)


class ResponseLimitTests(unittest.TestCase):
    """Tests covering the maximum response latency R."""

    def test_response_limit(self):
        # R excludes pairs with t > R independently of the stride; None means no limit; t == R is allowed.
        """Validate the response-latency limit."""
        self.assertTrue(ee.estimate(_pair(t=0.3), 1.0, 1.0, POWER).feasible)
        limited = ee.estimate(_pair(t=0.3), 1.0, 1.0, POWER, max_response_s=0.2)
        self.assertEqual((limited.feasible, limited.reason, limited.max_response_s), (False, "too slow (t > R)", 0.2))
        self.assertTrue(ee.estimate(_pair(t=0.2), 1.0, 1.0, POWER, max_response_s=0.2).feasible)
        self.assertEqual(ee.estimate(_pair(t=1.5), 1.0, 1.0, POWER, max_response_s=0.2).reason, "too slow (t > S)")

    def test_response_summary_switches_winner(self):
        # When R excludes the MCU, the summary shows the Pi; when low/high power disagree it shows both.
        """Validate response_summary()."""
        pairs = [_pair(model="mcu", t=0.5, acc=0.68), _pair(model="pi", board="rpi4", threads=4, t=0.001, acc=0.68)]
        estimates = [ee.estimate(p, 1.0, 1.0, POWER, r) for r in (None, 0.2) for p in pairs]
        table = ee.response_summary(estimates, [(1.0, 1.0)], [None, 0.2], [0.67, 0.99])
        rows = table.splitlines()[2:]
        self.assertIn("| no limit | mcu @ RP2040 | none feasible |", rows[0])
        self.assertIn("| 200 ms | pi @ Pi 4T | none feasible |", rows[1])
        self.assertEqual(ee._parse_response("none"), None)
        self.assertEqual(ee._parse_response("0.5"), 0.5)


class SelectionTests(unittest.TestCase):
    """Tests covering winner/runner-up selection."""

    def test_select_orders_by_energy_and_filters_accuracy(self):
        # Pairs below the accuracy floor or without accuracy are dropped; the rest sort by energy for the bound.
        """Validate select()."""
        estimates = [
            ee.estimate(_pair(model="cheap_inaccurate", t=0.01, acc=0.60), 1.0, 1.0, POWER),
            ee.estimate(_pair(model="no_acc", t=0.01, acc=None), 1.0, 1.0, POWER),
            ee.estimate(_pair(model="mcu", t=0.5, acc=0.68), 1.0, 1.0, POWER),
            ee.estimate(_pair(model="pi", board="rpi4", threads=4, t=0.001, acc=0.70), 1.0, 1.0, POWER),
        ]
        ranked = ee.select(estimates, 0.67, "high")
        self.assertEqual([e.pair.model for e in ranked], ["mcu", "pi"])
        self.assertEqual(ee.select(estimates, 0.69, "low")[0].pair.model, "pi")
        self.assertEqual(ee.select(estimates, 0.99, "low"), [])


class PowerFileTests(unittest.TestCase):
    """Tests covering board_power.yaml consistency."""

    def setUp(self):
        """Load the YAML once."""
        self.data = yaml.safe_load((ENERGY_DIR / "board_power.yaml").read_text(encoding="utf-8"))

    def test_every_value_has_source_and_confidence(self):
        # Each watt range must carry its provenance so estimates stay auditable.
        """Validate sources and confidence notes."""
        boards = self.data["boards"]
        pi = boards["rpi4"]
        for node in (pi["P_active"]["threads_1"], pi["P_active"]["threads_4"], pi["P_listen"]):
            self.assertIn("low_source", node)
            self.assertIn("high_source", node)
            self.assertIn("confidence", node)
        self.assertEqual(pi["P_sleep"]["same_as"], "P_listen")
        rp = boards["rp2040"]
        self.assertIn("ESTIMATE", rp["description"])
        for key in ("P_active", "P_listen", "P_sleep"):
            self.assertIn("confidence", rp[key])
        for name, component in rp["components"].items():
            self.assertIn("source", component, name)
        self.assertIn("one-time", rp["components"]["nina_w102"]["uncertainty_note"].lower())

    def test_rp2040_totals_match_components(self):
        # The stated RP2040 board watts must equal 3.3 V * sum(component mA) / efficiency for each bound.
        """Recompute RP2040 totals from the component table."""
        rp = self.data["boards"]["rp2040"]
        volts = rp["rail_voltage"]
        eff = rp["buck_efficiency"]
        for state, key in (("active", "P_active"), ("listen", "P_listen"), ("sleep", "P_sleep")):
            for i in (0, 1):
                current_ma = sum(c[state][i] for c in rp["components"].values())
                expected = volts * current_ma / 1000.0 / eff[i]
                self.assertTrue(math.isclose(rp[key]["watts"][i], expected, abs_tol=5e-4), (state, i, expected))

    def test_load_power_pi_sleep_equals_listen(self):
        # The Pi has no board sleep, so P_sleep must resolve to P_listen; ranges are ordered.
        """Validate load_power()."""
        power = ee.load_power()
        self.assertEqual(power["rpi4"].sleep, power["rpi4"].listen)
        self.assertEqual(set(power["rpi4"].active), {1, 4})
        self.assertEqual(power["rp2040"].active_for(4), power["rp2040"].active[1])
        for board in power.values():
            for low, high in [*board.active.values(), board.listen, board.sleep]:
                self.assertLessEqual(low, high)


class CsvLoadingTests(unittest.TestCase):
    """Tests covering summary CSV loading."""

    def test_load_pairs(self):
        # RP2040 rows (no threads column) count as 1 thread; board names map to power keys; blanks become None.
        """Validate load_pairs()."""
        with tempfile.TemporaryDirectory() as tmp:
            rp = Path(tmp) / "rp.csv"
            rp.write_text("model,board,stride_s,median_us,status,val_acc,val_macro_f1\n"
                          "a,nano_rp2040_connect,1.0,179197.0,ok,0.646,0.658\n"
                          "b,nano_rp2040_connect,2.0,,infeasible:arena,,\n")
            pi = Path(tmp) / "pi.csv"
            pi.write_text("model,board,stride_s,median_us,status,val_acc,val_macro_f1,threads,xnnpack\n"
                          "a,rpi4,1.0,174.0,ok,0.646,0.658,4,default\n")
            pairs = ee.load_pairs([rp, pi])
        self.assertEqual([(p.board, p.threads) for p in pairs], [("rp2040", 1), ("rp2040", 1), ("rpi4", 4)])
        self.assertAlmostEqual(pairs[0].latency_s, 0.179197)
        self.assertIsNone(pairs[1].latency_s)
        self.assertIsNone(pairs[1].val_acc)
        self.assertEqual(pairs[2].xnnpack, "default")


if __name__ == "__main__":
    unittest.main()
