<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Analysis Scripts

This folder contains public-facing analysis utilities and publication-figure
generators that sit outside the core CREST runtime.

Run these scripts from the repository root with the `crest` Conda environment
active unless a package README says otherwise. Most scripts consume existing
NAS, replay, or measurement artifacts and do not run NAS or touch hardware.
The micro-workload energy probe is the exception: it stages a synthetic
workload and uses the CREST HIL harness.

## Folders

- `compare_pareto_front_calcs/`
  - **Hardware requirement: no hardware required.** Generic CSV-derived
    Pareto-front comparison helper with explicit quality/cost columns,
    feasibility filters, matching rules, and reduction denominators.

- `cs3_audio_sensitivity/`
  - **Hardware requirement: no hardware required.** Case Study 3 post-hoc
    score-sensitivity analysis over audio NAS logs.

- `micro_workload_energy_probe/`
  - **Hardware requirement: development board and HIL harness required.**
    Synthetic phase-energy probe for CREST-compatible MCU targets, reusing the
    existing HIL harness and INA228 telemetry path.

- `rp2040_latency_bench/`
  - **Hardware requirement: Arduino Nano RP2040 Connect required** (export
    and compile-only steps need no board). Latency/memory-only bench for the
    RP2040 backend: smallest DS-CNN export, per-run timing results, and an
    upload-robustness loop.

- `paper_plots/`
  - **Hardware requirement: no hardware required.** Publication figure
    generators plus sidecar plotted-point and summary outputs from explicit
    input CSV/replay paths.
