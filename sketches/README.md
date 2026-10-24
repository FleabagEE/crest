<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Sketches

This directory contains the repo-owned Arduino and STM32 sketch/workspace
sources used by CREST runtime, HIL, and analysis flows.

Hardware requirement depends on the workflow. Reading or modifying these
sources is software-only; compiling, uploading, or measuring them requires the
matching development board, and harness-assisted energy measurement requires
the CREST HIL harness.

For board bring-up details, see
[`../src/crest/microcontrollers/README.md`](../src/crest/microcontrollers/README.md).
For config semantics, see
[`../src/config/README.md`](../src/config/README.md).

## Directory Map

- [`crest_inference_energy.ino`](crest_inference_energy.ino)
  Shared uniform-input Arduino runtime sketch used when
  `training.input_mode: uniform` and `training.energy_aware: true`.
- [`crest_inference_no_energy.ino`](crest_inference_no_energy.ino)
  Shared uniform-input Arduino runtime sketch used when
  `training.input_mode: uniform` and `training.energy_aware: false`.
- [`crest_inference_energy_cadenced.ino`](crest_inference_energy_cadenced.ino)
  Shared Arduino cadenced-pass sketch.
- [`crest_inference_latency.ino`](crest_inference_latency.ino)
  Latency/memory-only sketch with warm-up and per-run `key=value` telemetry.
  Always used for `ARDUINO_NANO_RP2040_CONNECT`; never drives harness GPIO.
- [`analysis_sketches/`](analysis_sketches)
  Analysis-oriented sketch variants and data headers for non-uniform input
  modes.
- [`common/`](common)
  Shared headers copied into staged Arduino candidate directories.
- [`harness/`](harness)
  The energy/runtime harness sketch and its local support files.
- [`boot_m4_helper/`](boot_m4_helper)
  Portenta CM7 helper sketch used to bring up CM4 in the harness-only path.
- [`stm32/`](stm32)
  STM32 workspaces. The committed LRUN workspace is
  [`stm32/crest_stm32_lrun/README.md`](stm32/crest_stm32_lrun/README.md).

## Runtime Sketch Selection

The Arduino runtime-selection path described in
[`../src/crest/microcontrollers/README.md`](../src/crest/microcontrollers/README.md)
uses this layout:

1. Uniform input mode:
   - [`crest_inference_energy.ino`](crest_inference_energy.ino)
   - [`crest_inference_no_energy.ino`](crest_inference_no_energy.ino)
   - [`crest_inference_latency.ino`](crest_inference_latency.ino)
     (`ARDUINO_NANO_RP2040_CONNECT` only, regardless of `energy_aware`)
2. Cadenced second pass:
   - [`crest_inference_energy_cadenced.ino`](crest_inference_energy_cadenced.ino)
3. Analysis input modes:
   - [`analysis_sketches/crest_inference_representative.ino`](analysis_sketches/crest_inference_representative.ino)
   - [`analysis_sketches/crest_inference_real_data.ino`](analysis_sketches/crest_inference_real_data.ino)
   - [`analysis_sketches/oxiod_input_data.h`](analysis_sketches/oxiod_input_data.h)
   - [`analysis_sketches/urbansound8k_input_data.h`](analysis_sketches/urbansound8k_input_data.h)
4. Shared copied headers:
   - [`common/crest_hil_config.h`](common/crest_hil_config.h)
   - [`common/crest_power.h`](common/crest_power.h)
   - [`common/crest_clock_telemetry.h`](common/crest_clock_telemetry.h)

## Harness And Helper Sketches

- [`harness/harness.ino`](harness/harness.ino)
  Used when runtime measurement is harness-assisted or harness-only.
- [`boot_m4_helper/boot_m4_helper.ino`](boot_m4_helper/boot_m4_helper.ino)
  Used by the current Portenta CM4 path to bring up CM4 before uploading the
  DUT sketch.

## STM32 Workspace

STM32 content is not organized as one Arduino sketch per board.

The current committed workspace is:

- [`stm32/crest_stm32_lrun/README.md`](stm32/crest_stm32_lrun/README.md)

See that README for the LRUN workspace internals, staged/generated file
layout, and STM32-specific workflow details.

## Generated Candidate Workspaces

CREST stages candidate-specific sketches and build products outside these
source templates during NAS, replay, and probe runs. Treat this directory as
the committed template and support-code layout; generated candidate workspaces
and measurement outputs should remain local artifacts.
