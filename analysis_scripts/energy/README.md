<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Energy Estimator

Turns measured latency into energy per inference and picks the best
(model, board, threads) pair for a sensing spec.

**Hardware requirement: none** (reads existing summary CSVs).

- [`board_power.yaml`](board_power.yaml): `P_active`, `P_listen`, `P_sleep`
  per board as `[low, high]` watt ranges, each with sources and a confidence
  note.
  - Pi 4: `P_active` is given separately for 1 and 4 threads, and
    `P_sleep = P_listen` because the board has no whole-board sleep.
  - Nano RP2040 Connect: a board-level estimate built from component
    datasheets, to be replaced by a USB power measurement.
- [`estimate_energy.py`](estimate_energy.py): reads `rp2040_summary.csv` and
  `rpi4_summary.csv` and applies

  ```
  S <= L: E = P_active * t + P_listen * (S - t)
  S >  L: E = P_active * t + P_listen * L + P_sleep * (S - L - t)
  ```

  A pair counts only if its input window equals `L`. It is infeasible if its
  benchmark status is not `ok`, if `t > S`, or (for `S > L`) if `t > S - L`.
  It writes `results/energy_estimates.csv` (every pair, low/high energy and
  average power) and `results/energy_selection.md` (per spec: all pairs, then
  winner and runner-up for each minimum **validation-fold** accuracy, low vs.
  high power).

```bash
python analysis_scripts/energy/estimate_energy.py                                   # default specs
python analysis_scripts/energy/estimate_energy.py --spec 1,1 2,4 --min-acc 0.67     # custom
```
