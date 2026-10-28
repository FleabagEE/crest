<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# RP2040 Log-Mel Front End

Integer-only log-mel for the Nano RP2040 Connect (Cortex-M0+, no FPU),
matching the course-project training front end
(`embedded-systme-project/training/features.py`, `log_mel`): 16 kHz, frame
400 / hop 160, periodic Hann, 512-point real FFT power spectrum, 64 Slaney mel
bins (`librosa.filters.mel(..., htk=False, norm='slaney')`),
`ln(max(mel, 1e-6))`, then int8 with `rint` like `quantization/validate_int8.py`.

**Hardware requirement:** none for the host scripts; a Nano RP2040 Connect for
`run_logmel_golden_board.py`.

- [`export_logmel_tables.py`](export_logmel_tables.py) regenerates
  [`../../sketches/common/crest_logmel_tables.h`](../../sketches/common/crest_logmel_tables.h)
  (Hann Q16, sparse uint16 mel matrix, log2 LUT) with librosa 0.11.0 / numpy
  1.26.4, the versions pinned by the training repo.
- [`../../sketches/common/crest_logmel.h`](../../sketches/common/crest_logmel.h)
  is the header-only front end. It uses a kissfft fixed-point real FFT,
  `CREST_LOGMEL_FFT_BITS` 32 (q31, default for the golden sketch) or 16 (q15).
- [`golden_logmel.py`](golden_logmel.py) compares the host build (g++) and,
  with `--board-dump`, the board against the training code's `log_mel`. It
  reports max/mean absolute log-mel error and the fraction of int8 values
  that differ. Add real clips with `--wav`.
- [`run_logmel_golden_board.py`](run_logmel_golden_board.py) stages, flashes
  and runs [`../../sketches/logmel_golden`](../../sketches/logmel_golden).
  It dumps every frame, checks bit-exactness against the host build, and
  times one frame and full 1 s / 2 s windows.

Results (2026-10-28, five synthetic 1 s clips):

| FFT | max abs err | mean abs err | int8 differ | board |
|---|---|---|---|---|
| q15 | 2.48 | 0.049 | 11.7% (up to 33 levels) | not used |
| q31 | 0.0049 | 0.000053 | 0.32% (all +-1) | bit-exact with host |

q31 timing on the RP2040 (125 MHz, PCM in RAM): 4.63 ms per frame (one 10 ms
hop), 460.5 ms for a 1 s window in batch, 930.4 ms for 2 s.

```bash
python analysis_scripts/rp2040_logmel/export_logmel_tables.py
python analysis_scripts/rp2040_logmel/golden_logmel.py [--wav clip1.wav ...]
python analysis_scripts/rp2040_logmel/run_logmel_golden_board.py --port /dev/ttyACM0
```
