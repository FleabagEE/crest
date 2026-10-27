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

  A pair counts only if its input window equals `L`. It is infeasible if:
  - its benchmark status is not `ok`;
  - `t > S`;
  - for `S > L`, `t > S - L`. This is a **conservative assumption**: compute
    does not overlap listening, so each period is listen `L`, compute `t`,
    then sleep. Revisit it if a pair ever fails only this rule;
  - `t > R` when a maximum response latency `R` is given (`--max-response`).
    `R` is separate from the stride. Response latency is currently the
    `invoke()` time only; on-device log-mel preprocessing will add to it.
  It writes `results/energy_estimates.csv` (every pair, low/high energy and
  average power) and `results/energy_selection.md` (per spec: all pairs, then
  winner and runner-up for each minimum **validation-fold** accuracy, low vs.
  high power), starting with a compact winners table over all specs and
  response limits.

```bash
python analysis_scripts/energy/estimate_energy.py                                   # default specs
python analysis_scripts/energy/estimate_energy.py --spec 1,1 2,4 --min-acc 0.67     # custom
python analysis_scripts/energy/estimate_energy.py --max-response none 0.2 0.5 1.0  # response limits
```
