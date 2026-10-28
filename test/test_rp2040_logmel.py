# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the RP2040 integer log-mel front end and its golden harness."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
LOGMEL_DIR = ROOT_DIR / "analysis_scripts" / "rp2040_logmel"
if str(LOGMEL_DIR) not in sys.path:
    sys.path.insert(0, str(LOGMEL_DIR))

import export_logmel_tables  # noqa: E402
import golden_logmel  # noqa: E402

TRAINING_REPO = golden_logmel.DEFAULT_TRAINING_REPO


class TableTests(unittest.TestCase):
    """Tests covering the generated tables header."""

    def test_committed_tables_match_generator(self):
        # The committed header must be exactly what the librosa-based generator writes.
        """Validate the tables header is up to date."""
        committed = (ROOT_DIR / "sketches" / "common" / "crest_logmel_tables.h").read_text(encoding="utf-8")
        self.assertEqual(export_logmel_tables.render(export_logmel_tables.mel_matrix()), committed)

    def test_sparse_mel_reconstructs_matrix(self):
        # Sparse runs times the global scale must reproduce the float matrix to uint16 precision.
        """Validate sparse mel encoding."""
        weights = export_logmel_tables.mel_matrix()
        starts, lengths, offsets, values, scale = export_logmel_tables.sparse_mel(weights)
        rebuilt = np.zeros_like(weights, dtype=np.float64)
        for m, (s, n, o) in enumerate(zip(starts, lengths, offsets)):
            rebuilt[m, s:s + n] = values[o:o + n] * scale
        self.assertLess(np.abs(rebuilt - weights).max(), scale)


@unittest.skipUnless(shutil.which("g++") and (TRAINING_REPO / "training" / "features.py").exists(),
                     "needs g++ and the course training repo next to crest")
class HostGoldenTests(unittest.TestCase):
    """Tests running the host build of crest_logmel.h against the training front end."""

    def test_q31_matches_training_front_end(self):
        # q31 must stay within int8 rounding of the training log_mel on the synthetic clips.
        """Validate q31 host accuracy."""
        with tempfile.TemporaryDirectory() as tmp:
            binary = golden_logmel.build_host(32, Path(tmp))
            for name, pcm in golden_logmel.synthetic_clips(1.0).items():
                ref = golden_logmel.reference(pcm, TRAINING_REPO)
                stats = golden_logmel.compare(ref, golden_logmel.reference_int8(ref), golden_logmel.run_host(binary, pcm))
                with self.subTest(clip=name):
                    self.assertLess(stats["max_abs_err"], 0.01)
                    self.assertLessEqual(stats["int8_max_diff"], 1)
                    self.assertLess(stats["int8_diff_fraction"], 0.02)


if __name__ == "__main__":
    unittest.main()
