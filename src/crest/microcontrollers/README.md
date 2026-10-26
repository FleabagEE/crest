<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Microcontroller Backends

Microcontroller backends turn one trained/model-candidate export into
something CREST can compile, upload, run on a board, and measure.

For the broader source map, see [`../../README.md`](../../README.md). If you
are working on dataset loading, task semantics, or model architecture, use the
neighboring component-layer READMEs instead.

## What This Layer Owns

This layer owns:

- target/device metadata and arena limits;
- candidate staging for board-specific build systems;
- compile/upload/load tooling;
- runtime telemetry parsing and error classification;
- power/energy measurement integration through the HIL harness;
- normalized `DeviceMetrics` returned to the shared HIL controller.

The backend boundary is important because Arduino boards, STM32 Cube projects,
and future non-Arduino vendors do not all use the same build and runtime flow.

This README covers:

1. Adding a new Arduino board.
2. Adding another STM32 board that is similar to the shipped STM32 flow.
3. Adding a new non-Arduino vendor backend such as NXP or Nordic.

### Hardware Requirements

Reading or extending backend code is software-only. Compile-only checks require
the target toolchain but do not require a connected board. Uploading or runtime
measurement requires the target development board. Energy measurement requires
the CREST HIL harness and INA228 telemetry path.

## Contents

- [What This Layer Owns](#what-this-layer-owns)
- [Shipped Implementations](#shipped-implementations)
- [Key Files](#key-files)
- [Runtime Flow](#runtime-flow)
- [Add A New Hardware Backend](#add-a-new-hardware-backend)
- [Contract In Practice](#contract-in-practice)
- [Registration And Config Selection](#registration-and-config-selection)
- [Arduino Bring-Up Guide](#arduino-bring-up-guide)
- [STM32 Bring-Up Guide](#stm32-bring-up-guide)
- [New-Vendor Bring-Up Guide](#new-vendor-bring-up-guide)
- [Runtime Measurement Notes](#runtime-measurement-notes)
- [Nano RP2040 Connect](#nano-rp2040-connect)
- [Raspberry Pi 4](#raspberry-pi-4)
- [Test Plan](#test-plan)
- [Contributor Checklist](#contributor-checklist)
- [Definition Of Done](#definition-of-done)
- [Troubleshooting](#troubleshooting)

## Shipped Implementations

### Arduino CLI Backends

Arduino boards are the most standardized path in this repo.

Shared behavior comes from [`ArduinoDevice`](../devices.py) and
[`arduino_base.py`](arduino_base.py).

That shared path already handles:

1. TFLite export and `model.cc` / `model.h` generation during candidate prep.
2. Sketch synchronization into the active output directory.
3. Arduino CLI compile command construction.
4. Arduino CLI upload command construction.
5. Flash/RAM parsing from Arduino summary lines.
6. Fallback flash/RAM parsing from the board package `platform.txt` size regexes.
7. Direct-serial runtime measurement and harness-assisted measurement.

Current examples:

1. [`arduino_ble33.py`](arduino_ble33.py)
2. [`arduino_portenta_h7.py`](arduino_portenta_h7.py)
3. [`arduino_nano_rp2040.py`](arduino_nano_rp2040.py)
   (latency/memory only; see [Nano RP2040 Connect](#nano-rp2040-connect))

### STM32 Cube Backend

The STM32 implementation is intentionally split across three modules because
they own different responsibilities and failure domains.

1. [`stm32_nucleo_n657x0.py`](stm32_nucleo_n657x0.py)
   The concrete CREST board backend. It implements `DeviceInterface`,
   resolves STM options, stages candidate-specific projects, invokes ST Edge AI
   generation, and translates compile/upload/runtime results into CREST
   dataclasses and error codes.
2. [`stm32_cube_clt.py`](stm32_cube_clt.py)
   The STM32 toolchain/helper layer. It owns build, ELF discovery, size
   parsing, ST-LINK debug-load workflow, and external flash programming
   helpers. It does not implement `DeviceInterface`.
3. [`stm32_runtime.py`](stm32_runtime.py)
   The STM32 runtime protocol layer. It owns serial monitoring, READY/START
   handshake handling, telemetry parsing, and runtime protocol error typing. It
   does not implement `DeviceInterface`.

Important current limitation:

1. This is a concrete STM32 implementation, not yet a generic `stm_base.py`.
2. It assumes a specific project template shape, ST Edge AI generation flow,
   ST-LINK load path, and serial telemetry contract.
3. A second STM32 board can likely reuse parts of this stack, but should not
   assume every STM32 board is a small metadata-only addition.

### Other-Vendor Backends

For NXP, Nordic, RP2040 SDK, Zephyr, or any other non-Arduino/non-STM32-Cube
target, you should treat the work as a new backend implementation.

That usually means:

1. Implement `DeviceInterface` directly.
2. Add a registry entry in
   [`__init__.py`](__init__.py).
3. Add config-option resolution in
   [`resolve_device_options(...)`](__init__.py)
   when needed.
4. Provide your own build, flash/load, and runtime telemetry path.

## Key Files

These files are the main integration points in this package.

1. [`../devices.py`](../devices.py)
   Defines the shared dataclasses and the `DeviceInterface` contract. Also
   contains the shared Arduino bridge class `ArduinoDevice`.
2. [`__init__.py`](__init__.py)
   Registry, factory, and config-option resolution for known devices.
3. [`arduino_base.py`](arduino_base.py)
   Shared Arduino CLI compile/upload/measurement helpers.
4. `arduino_<board>.py`
   One concrete Arduino-backed board wrapper per board family.
5. [`stm32_nucleo_n657x0.py`](stm32_nucleo_n657x0.py)
   The concrete STM32 backend currently shipped in this repo.
6. [`stm32_cube_clt.py`](stm32_cube_clt.py)
   STM32 Cube / build / ELF / ST-LINK / CubeProgrammer helper layer.
7. [`stm32_runtime.py`](stm32_runtime.py)
   STM32 runtime serial protocol parser and runtime error classifier.

## Runtime Flow

CREST has two major candidate-prep families today.

### Arduino Candidate Staging

Arduino backends use the shared candidate path in
[`ArduinoDevice.prepare_candidate(...)`](../devices.py).

That path:

1. Converts the Keras model to TFLite.
2. Converts the TFLite file to `model.cc` / `model.h`.
3. Copies the selected sketch variant into the active output directory.

Runtime sketch selection is handled by
[`_sync_arduino_sketch_variant_for_config(...)`](../devices.py)
and uses a `runtime_phase` selector plus this layout:

For sketch ownership and directory roles, see
[`../../../sketches/README.md`](../../../sketches/README.md).

1. Shared uniform variants:
   - `sketches/crest_inference_energy.ino`
   - `sketches/crest_inference_no_energy.ino`
   - `sketches/crest_inference_energy_cadenced.ino`
2. Shared analysis variants:
   - `sketches/analysis_sketches/crest_inference_representative.ino`
   - `sketches/analysis_sketches/crest_inference_real_data.ino`
   - `sketches/analysis_sketches/oxiod_input_data.h`
   - `sketches/analysis_sketches/urbansound8k_input_data.h`
3. Shared headers copied from `sketches/common/`:
   - `crest_hil_config.h`
   - `crest_power.h`
   - `crest_clock_telemetry.h`

Board/core behavior should be selected in Python wrappers and compile-time
defines, not by duplicating the shared uniform sketch sources.

Cadenced Arduino runs are intentionally staged as a second pass above arena
search. The base pass keeps using `runtime_phase="back_to_back"`, then the
prepared candidate directory is restaged via `set_input_mode(...,
runtime_phase="cadenced")` before one extra fixed-arena evaluation.

### STM32 Candidate Staging

STM32 does not use Arduino sketch staging.

The current backend in
[`stm32_nucleo_n657x0.py`](stm32_nucleo_n657x0.py) instead:

1. Copies a canonical STM32 workspace overlay into a candidate-specific staging root.
2. Exports the Keras model to TFLite.
3. Runs ST Edge AI analyze/generate steps.
4. Stages generated outputs into the project.
5. Writes manifest/config artifacts needed for later compile/upload phases.

Important boundary:

1. STM32 does not consume Arduino `model.cc` / `model.h` artifacts.
2. STM32 does not use the Arduino sketch variant sync path.
3. Arduino and STM32 candidate prep already follow separate backend-owned paths
   under the shared HIL server flow.

## Add A New Hardware Backend

Choose the path that matches your board.

### Use the Arduino path when:

1. The board has an Arduino FQBN.
2. The board builds with `arduino-cli compile`.
3. The board uploads with `arduino-cli upload`.

### Use the STM32 path when:

1. The board uses the STM32 Cube / ST Edge AI / ST-LINK workflow already used
   in this repo.
2. The board can be staged as a project directory, built with the generated
   `Debug/makefile`, loaded with ST-LINK tooling, and measured through the
   shipped runtime protocol.

### Use the new-vendor path when:

1. The board does not use Arduino CLI.
2. The board does not fit the shipped STM32 Cube flow.
3. You need a new toolchain, flash/load method, or runtime telemetry contract.

## Contract In Practice

All microcontroller backends must satisfy the `DeviceInterface` contract in
[`../devices.py`](../devices.py).

In plain English, a backend owns these responsibilities:

1. `spec`
   Returns the device limits and arena search space exposed to the rest of the
   system through `DeviceSpec`.
2. `prepare_candidate(...)`
   Creates the board-specific build directory or project for one model
   candidate.
3. `compile(...)`
   Builds that candidate and returns normalized flash/RAM diagnostics through
   `CompileResult`.
4. `upload(...)`
   Flashes or loads the built image onto the device and returns `UploadResult`.
5. `measure(...)`
   Captures runtime telemetry and returns normalized latency / power / error
   data through `MeasureResult`.
6. `evaluate(...)`
   Orchestrates compile, upload, and runtime measurement, then translates
   backend-specific behavior into CREST `DeviceMetrics`.

Backends also declare behavior through capability flags:

1. `requires_candidate_model()`
   Whether the backend consumes generated model artifacts.
2. `requires_training_data()`
   Whether candidate preparation needs calibration/training data.
3. `requires_arena_validation()`
   Whether arena search should treat runtime arena validation as required.
4. `supports_energy_measurement()`
   Whether real power/energy telemetry is supported.
5. `supports_runtime_measurement()`
   Whether upload/runtime HIL passes are supported.
6. `runtime_measure_mode()`
   Whether runtime is measured via direct DUT serial or via a harness-led flow.

If you are adding a new backend family, start by understanding this contract.

## Registration And Config Selection

All backends, regardless of family, need to integrate with the shared factory
and config plumbing.

### Registry

Update [`__init__.py`](__init__.py).

Required work:

1. Add an entry to `_registry_entries()`.
2. Add lazy export support in `__getattr__` if the backend class should be
   importable from the package root.
3. Add the class name to `__all__`.

Why this matters:

1. `get_device(...)` instantiates backend wrappers.
2. `list_device_specs()` projects default specs for compatibility paths.
3. `devices.DEVICE_SPECS` is rebuilt from this registry plus legacy entries.

### Device Options

If the board has no custom options:

1. Set `device.name` in config.
2. Set `device.serial_port`.

If the board has custom options:

1. Add a nested config block such as `device.portenta.*` or `device.stm32.*`.
2. Parse it in
   [`resolve_device_options(...)`](__init__.py).
3. Ensure those options flow through
   [`build_collect_metrics_request(...)`](../hil_runtime.py)
   and the HIL server path in
   [`../../hil_server.py`](../../hil_server.py).

Do not bypass `collect_metrics(...)` or `HIL_controller(...)`. Device options
must flow through the shared controller path.

## Arduino Bring-Up Guide

Use this path only for boards built and flashed through `arduino-cli`.

### Files You Will Usually Touch

1. `src/crest/microcontrollers/arduino_<board>.py`
2. `src/crest/microcontrollers/__init__.py`
3. `setup_arduino.sh` when a new Arduino core package is required
4. `src/config/` example configs and any board-specific config variants
5. Tests in `test/test_hardware.py` and `test/test_model.py`
6. `src/crest/model.py` only when you are changing shared request plumbing,
   which should be unusual for a normal Arduino board bring-up

### Board Module Contract

Create `src/crest/microcontrollers/arduino_<board>.py` with this shape:

1. Define board identity constants:
   - `BOARD_NAME`
   - `BOARD_FQBN`
2. Define resolver/spec symbols:
   - `resolve_<board>_options(...)`
   - `build_<board>_spec(...)`
   - `BOARD_DEFAULT_SPEC`
3. Define a default `DeviceSpec` with:
   - explicit `arena_sizes_kb`
   - explicit `max_ram_bytes`
   - explicit `max_flash_bytes`
   - `toolchain="arduino-cli"`
4. If the board has options, define:
   - an options dataclass if useful
   - a resolver/validator
   - a spec builder using resolved options
5. Implement a wrapper class that extends
   [`ArduinoDevice`](../devices.py)

Current in-repo examples:

1. BLE33:
   [`arduino_ble33.py`](arduino_ble33.py)
2. Portenta H7:
   [`arduino_portenta_h7.py`](arduino_portenta_h7.py)

Example minimal static board pattern:

```python
from __future__ import annotations

from typing import Mapping, Optional

from ..devices import ArduinoDevice, DeviceSpec

BOARD_NAME = "ARDUINO_GIGA_R1"
BOARD_FQBN = "arduino:mbed_giga:giga"


def resolve_giga_r1_options(
    device_options: Optional[Mapping[str, object]],
) -> None:
    del device_options
    return None


def build_giga_r1_spec(options: None = None) -> DeviceSpec:
    del options
    return DeviceSpec(
        name=BOARD_NAME,
        arena_sizes_kb=[32, 64, 96, 128],
        max_ram_bytes=512_000,
        max_flash_bytes=2_000_000,
        fqbn=BOARD_FQBN,
        toolchain="arduino-cli",
    )


BOARD_DEFAULT_SPEC = build_giga_r1_spec()


class ArduinoGigaR1Device(ArduinoDevice):
    def __init__(
        self,
        *,
        serial_port: Optional[str] = None,
        device_options: Optional[dict[str, str]] = None,
    ) -> None:
        resolved_options = resolve_giga_r1_options(device_options)
        super().__init__(
            BOARD_NAME,
            serial_port=serial_port,
            device_options=resolved_options,
            spec_override=build_giga_r1_spec(resolved_options),
        )
```

### Arduino Core Setup

If the board needs a new Arduino core package:

1. Update `setup_arduino.sh`.
2. Add the correct `arduino-cli core install ...` line.
3. Keep existing core installs intact.

### Memory Parsing Notes

[`arduino_base.compile_sketch(...)`](arduino_base.py)
parses memory in two stages:

1. Primary: Arduino CLI `Sketch uses ...` output.
2. Fallback: board-defined `platform.txt` regexes
   (`recipe.size.regex` and `recipe.size.regex.data`) over
   `arm-none-eabi-size -A` output.

Why the fallback exists:

1. Some boards may not print summary lines.
2. `platform.txt` is the board package's own source of truth for Arduino-style
   memory accounting.
3. It avoids counting linker-reserved sections such as `.heap`, `.stack`, or
   board-reserved regions as "used RAM".

If a board does not print `Sketch uses ...`:

1. Inspect resolved properties with:
   `arduino-cli compile --show-properties --fqbn <fqbn> <sketch>`
2. Confirm these properties exist:
   - `recipe.size.pattern`
   - `recipe.size.regex`
   - `recipe.size.regex.data`
3. Confirm board limits resolve:
   - `upload.maximum_size`
   - `upload.maximum_data_size`
4. If needed, inject `upload.maximum_size` via `--build-property`.
5. Re-run compile and verify either:
   - Arduino CLI prints summary lines, or
   - the fallback returns correct flash/RAM values.

### Upload Error Notes

[`arduino_base.upload_sketch(...)`](arduino_base.py)
appends Linux DFU permission guidance for known `LIBUSB_ERROR_ACCESS` patterns.

If your board uses DFU on Linux:

1. Add or update root `README.md` udev instructions.
2. Keep Linux and macOS notes explicit.

## STM32 Bring-Up Guide

This section documents the shipped `STM32_NUCLEO_N657X0_Q` backend
and the starting point for adding a second STM32 backend.

### Shipped Config Shape

Example config block:

```yaml
device:
  name: STM32_NUCLEO_N657X0_Q
  serial_port: /dev/ttyACM0
  measured_inference_runs: 10
  runtime_mode: cadenced
  cpu_clock_mhz_options: [200, 300, 400, 600, 800]
  stm32:
    project_root: sketches/stm32/crest_stm32_lrun
    appli_flash_address: 0x70100000
    weight_storage_mode: external_flash
    weights_memory_pool: src/config/stm32_nucleo_mypool.json
    wake_margin_us: 5000
    min_sleep_us: 5000
```

Current default behavior:

1. If `device.stm32.project_root` is omitted, the backend defaults to
   `sketches/stm32/crest_stm32_lrun`.
2. LRUN `dev_boot` is the only supported layout shipped for this board and is
   implied automatically.

Supported STM options today:

1. Shared runtime knob: `device.measured_inference_runs` controls how many on-device
   inference invokes are averaged into one measured HIL attempt. This is not an
   STM-only board option.
2. Shared cadence/runtime knobs:
   - `device.runtime_mode`: `back_to_back` or `cadenced`
   - `device.latency_budget_ms`: optional cadence-budget override
3. Shared CPU clock sweep knob (if allowed by the DUT):
   - `device.cpu_clock_mhz_options`: optional per-trial presets
     `200`, `300`, `400`, `600`, `800`
4. STM32 workspace/layout options:
   - `template_root`
   - `project_root`
5. STM32 runtime/storage options:
   - `weight_storage_mode`: `embedded` or `external_flash`
   - `appli_flash_address`
   - `weights_flash_address`
   - `weights_memory_pool`
   - `weights_external_loader`
   - `wake_margin_us`
   - `min_sleep_us`
   - `max_external_flash_bytes`
6. Optional tool/signing overrides:
   - `gdbserver`
   - `gdb`
   - `cubeprog_bin`
   - `signing_tool`
   - `gdb_port`
   - `apid`
   - `server_ready_timeout_s`
   - `signing_load_offset`
   - `signing_header_version`

Important current caveats:

1. `device.stm32.cpu_clock_mhz` is not the current config shape. Use the
   shared top-level `device.cpu_clock_mhz_options` instead.
2. `device.stm32.project_layout` is no longer accepted. LRUN `dev_boot` is
   implied automatically for this board.
3. `device.stm32.runtime_mode` is no longer supported. Use the shared
   top-level `device.runtime_mode` instead.

### Required Host Tools

1. `ST-LINK_gdbserver`
2. `arm-none-eabi-gdb`
3. `STM32_Programmer_CLI`
4. `arm-none-eabi-gcc`
5. `arm-none-eabi-size`
6. `arm-none-eabi-objdump`
7. `STM32_SigningTool_CLI` or `STM32TrustedPackageCreator_CLI`
8. ST Edge AI CLI (`stedgeai`) reachable from the host build environment for
   normal STM32 NAS/HIL candidate generation

`STM32_Programmer_CLI` is required both for external-flash programming and for
the current ST-LINK debug-load helper flow, because the GDB server is launched
with the STM32CubeProgrammer `bin` directory.

### What The Current STM Backend Assumes

This matters if you want to add a second STM board.

The current backend assumes:

1. A template-root project that can be copied into a candidate-specific staging
   directory.
2. A `Debug` directory with a generated `makefile` build path.
3. ST Edge AI generated outputs with the expected filenames.
4. An ST-LINK based load flow via GDB server + GDB.
5. A serial runtime protocol compatible with
   [`stm32_runtime.py`](stm32_runtime.py).
6. Optional harness-assisted energy measurement when `harness_serial_port` is
   configured during HIL runs.

If your new STM board does not meet those assumptions, you are doing more than
board bring-up. You are defining a new STM backend shape.

### Recommended Path For A Second STM32 Board

Today, the practical path is:

1. Treat
   [`stm32_nucleo_n657x0.py`](stm32_nucleo_n657x0.py)
   as the reference implementation.
2. Reuse
   [`stm32_cube_clt.py`](stm32_cube_clt.py)
   if your board still uses the same Cube/ST-LINK workflow.
3. Reuse
   [`stm32_runtime.py`](stm32_runtime.py)
   only if your firmware emits the same runtime tokens.
4. Add a new concrete STM backend module when the board-specific differences are
   too large to express as metadata only.

There is no generic `stm_base.py` yet.

## New-Vendor Bring-Up Guide

Use this path for NXP, Nordic, Zephyr-based targets, or any board that does not
fit the Arduino path or the shipped STM32 path.

### Mindset

You are building a new backend, not adding a board definition.

### Minimum Deliverables

1. A new module that implements `DeviceInterface`.
2. A registry entry in
   [`__init__.py`](__init__.py).
3. An option resolver in
   [`resolve_device_options(...)`](__init__.py)
   when the backend needs config-owned options.
4. A candidate preparation path.
5. A compile/build path and size parser.
6. A flash/load path.
7. A runtime telemetry parser and error-classification path.
8. Tests and at least one smoke workflow.

### Questions To Answer Before You Start

1. What tool builds the firmware?
2. What tool flashes or loads the device?
3. How will flash and RAM usage be parsed?
4. What serial or harness protocol defines a successful runtime run?
5. How are compile overflow, upload failure, runtime timeout, and runtime
   protocol errors classified?

If those answers are not yet clear, the backend boundary is not ready.

## Runtime Measurement Notes

### Direct Serial

This is the default mode. The DUT is uploaded, opened over serial, and CREST
waits for the backend-specific runtime contract to produce latency and optional
power data.

### Harness-Only

Some targets cannot provide reliable host-visible DUT serial during runtime.
When that happens, `runtime_measure_mode()` may return `harness_only`, which
requires `device.harness_serial_port` and the harness flow in the shared
controller path.

### Portenta CM4 Limitation

For `PORTENTA_H7` with `target_core=cm4`, CREST uses `harness_only`
measurement.

Important notes:

1. Harness metrics remain the source of truth for energy and harness latency.
2. DUT-side clock telemetry may still be merged in when available.
3. CREST uploads a CM7 boot-helper sketch before the CM4 DUT upload:
   `sketches/boot_m4_helper/boot_m4_helper.ino`
4. `serial_port` is still required because it is the upload path for both the
   helper and the CM4 DUT sketch.

Current limitation:

1. The Arduino `mbed_portenta` CM4 path does not expose host USB CDC `Serial`
   on the primary ACM port used by CREST uploads.
2. Harness telemetry is therefore authoritative for runtime latency and energy.
3. DUT-side arena/runtime error lines are not reliably visible from the host.

Practical implication:

1. A CM4 runtime arena allocation failure can surface as a generic
   `HIL_ERROR_LATENCY` rather than `HIL_ERROR_UNDER_SIZED`.
2. Deterministic CM4 runtime-fault classification would require a separate
   fault channel and is out of scope for the current backend.

## Nano RP2040 Connect

`ARDUINO_NANO_RP2040_CONNECT` (FQBN `arduino:mbed_nano:nanorp2040connect`)
uses the same `arduino:mbed_nano` core and `ArduinoDevice` flow as BLE33, but
it only measures **latency and memory**. There is no harness or INA228 path:
`supports_energy_measurement()` returns `False`, and config validation
rejects `training.energy_aware: true`, `device.runtime_mode: cadenced` and
`device.harness_serial_port` for this board.

What is different from BLE33:

1. Sketch: candidates are staged with
   [`../../../sketches/crest_inference_latency.ino`](../../../sketches/crest_inference_latency.ino)
   whatever `training.energy_aware` says. It runs `device.rp2040.warmup_runs`
   untimed `Invoke()` calls, times each of `device.measured_inference_runs`
   calls with `micros()` (nothing is printed inside the timed window), and
   then prints one line per run:
   `board=nano_rp2040_connect model=<dir> run=<k> invoke_us=<int> preprocess_us=-1 arena_used_bytes=<int> status=ok`.
   It then prints the usual `runs:` / power-placeholder / `timer output:`
   lines, so `arduino_base` parses it unchanged. `timer output` is the mean of
   the timed runs (warm-up excluded, unlike the BLE33 sketch).
   `preprocess_us` is `-1` because log-mel features are computed on the host.
2. Failures: `AllocateTensors()` runs after `START`, so TFLM's
   `Failed to allocate ... missing: N` message reaches the host and maps to
   `HIL_ERROR_UNDER_SIZED` as before. The sketch also prints
   `status=infeasible reason=<reason>` and never prints `timer output:`.
   Flash/RAM overflow is detected at compile time (`HIL_ERROR_*_OVERFLOW`).
3. Arena sweep: set per target with `device.rp2040.arena_sizes_kb`
   (default `[10, 25, 40, 70, 95, 120, 140, 160, 180, 200]`, capped below
   the 225 000-byte RAM limit).
4. Upload: each `arduino-cli upload` attempt has a timeout
   (`device.rp2040.upload_timeout_s`). Failed attempts are retried
   (`upload_retries`) after an explicit 1200-bps touch, and the backend waits
   up to `port_wait_s` for `/dev/ttyACM*` to come back.

### Build, Upload, Run

Example config:
[`../../config/nas_config_audio_rp2040.yaml`](../../config/nas_config_audio_rp2040.yaml)
(pinned to the smallest DS-CNN).

```bash
make arduino-setup                  # installs arduino:mbed_nano, same core as BLE33
make rp2040-export                  # smallest int8 DS-CNN -> models/rp2040_bench/...
make rp2040-compile                 # compile-only memory check, no board needed
make rp2040-bench PORT=/dev/ttyACM0 # upload + 5 warm-up + 100 timed runs -> results file
make rp2040-upload-stress PORT=/dev/ttyACM0
make start-hil                      # NAS: run with --config src/config/nas_config_audio_rp2040.yaml
```

See
[`../../../analysis_scripts/rp2040_latency_bench/README.md`](../../../analysis_scripts/rp2040_latency_bench/README.md)
for script options and the results-file format.

### Known Bootloader Issue

Every upload goes through the RP2040 ROM USB bootloader. The 1200-bps touch
resets the board, and it re-enumerates with a different USB ID
(`2e8a:0003`, "RP2 Boot") before `picotool` flashes it; afterwards it comes
back as `2341:005e` with a new `/dev/ttyACM*` node.

1. Linux needs the mbed core's udev rules for `2e8a:0003`. Otherwise
   `picotool` cannot open the device and waits forever.
2. Under WSL2 the bootloader device is a new USB device. It has to be
   attached with `usbipd` (`--auto-attach`) too, or the upload hangs at the
   same point.
3. If the touch does not reach the board (for example a sketch that crashed
   with USB down), force the ROM bootloader by hand. The Nano RP2040 Connect
   has no BOOTSEL button: jumper the `REC` pin to `GND`, press reset, remove
   the jumper, then retry the upload.

The backend's timeout and retry loop turn a hang into `HIL_ERROR_UPLOAD`
instead of blocking NAS. It cannot fix a missing udev rule or USB attachment.

## Raspberry Pi 4

`RASPBERRY_PI_4` ([`raspberry_pi4.py`](raspberry_pi4.py)) is a new-vendor
backend. It implements `DeviceInterface` directly and runs LiteRT on the Pi
over SSH, measuring **latency and memory** only.

1. `prepare_candidate` exports `g_model.tflite` (float or full-int8) on the
   CREST host. There is no C array and no sketch.
2. `compile` builds nothing; `flash_bytes` is the `.tflite` size.
3. `upload` runs `ssh mkdir -p <remote_dir>/<candidate>`, then `scp` copies
   the model and
   [`../../../sketches/rpi4/crest_litert_runner.py`](../../../sketches/rpi4/crest_litert_runner.py).
4. `measure` runs the runner over `ssh`. It does `warmup_runs` untimed and
   `measured_inference_runs` timed `invoke()` calls, prints the RP2040-format
   per-run lines and `timer output:`, and reports `peak_rss_delta_bytes`
   (`VmHWM` window from model load to the last run), which becomes
   `ram_bytes`.
5. The arena sweep is the single-shot `[-1]` sentinel and
   `requires_arena_validation()` is `False`, as for STM32.
   `supports_energy_measurement()` is `False`. Config validation rejects
   `energy_aware: true`, cadenced mode, a harness port, and `hil: true`
   without `device.rpi4.host`.

Config lives under `device.rpi4.*` (`host`, `user`, `ssh_port`, `remote_dir`,
`python`, `num_threads`, `xnnpack`, `warmup_runs`, `ssh_connect_timeout_s`,
`upload_timeout_s`). See
[`../../config/nas_config_audio_rpi4.yaml`](../../config/nas_config_audio_rpi4.yaml).

Requirements on the Pi:

1. Key-based SSH login from the CREST host. `BatchMode=yes` is used, and
   the host key must already be in `known_hosts`.
2. A Python with a LiteRT runtime at `device.rpi4.python`. The runner tries
   `ai_edge_litert`, then `tflite_runtime`, then `tensorflow.lite`.
   `tflite_runtime` has no wheel for Python 3.13 (Raspberry Pi OS / Debian
   13), so use a venv with `ai-edge-litert`.
3. For stable latency, use the `performance` CPU governor and watch the
   reported temperature and `throttled` flags. The runner records these but
   never changes them.

Bench tool:
[`../../../analysis_scripts/rpi4_latency_bench/run_rpi4_latency_bench.py`](../../../analysis_scripts/rpi4_latency_bench/run_rpi4_latency_bench.py).

## Test Plan

Most backend work should have a hardware-free preflight path first. Use
connected hardware only after config resolution, request construction, staging,
and parser behavior are covered locally.

### Software Tests

Add or update tests in the backend-specific and orchestration test surfaces,
not just the older `test/test_hardware.py` and `test/test_model.py` pair.

Important current test files include:

1. `test/test_hardware.py`
2. `test/test_model.py`
3. `test/test_hil_server.py`
4. `test/test_nas_model_client.py`
5. `test/test_odom_tcn.py`
6. `test/test_stm32_backend.py`
7. `test/test_stm32_runtime.py`
8. `test/integration/test_hardware_integration.py`

For a new Arduino board, cover:

1. Board option validation if applicable.
2. Command construction including expected `--fqbn` and `--board-options`.
3. Memory parsing returning usable values.
4. Config -> `device_options` -> `collect_metrics` -> `HIL_controller`
   plumbing.
5. Compatibility behavior for `DEVICE_SPECS` helpers.

For a new STM or non-Arduino backend, cover:

1. Option validation and config resolution.
2. Candidate staging behavior.
3. Build/size parsing behavior.
4. Upload failure classification.
5. Runtime protocol success and error parsing.
6. End-to-end `evaluate(...)` normalization into CREST metrics and error
   codes.
7. Compile-only smoke (`run_hil=False`) on the new board/backend config when
   the required toolchain is available.

## Contributor Checklist

Before opening a PR:

1. Keep board/toolchain code in the microcontroller backend.
2. Keep architecture/search logic in model families.
3. Keep task metrics and evaluation logic in task adapters.
4. Add or update NumPy-style docstrings for changed functions, classes,
   dataclasses, and tests.
5. Toolchain preflight: compile-only smoke (`run_hil=False`) on the new
   board/backend config.
6. Upload smoke on connected hardware.
7. Runtime smoke on connected hardware when HIL is supported.
8. Confirm serial port defaults in config are correct.
9. Confirm harness pins/settings when energy-aware mode or harness-only mode is
   enabled.
10. Confirm generated output filenames and staging layout are what the backend
   expects.

## Definition Of Done

A backend bring-up is done when:

1. The device can be constructed through `get_device(...)`.
2. `prepare_candidate(...)` produces the correct build input directory.
3. `compile(...)` returns real flash/RAM diagnostics or correctly typed
   compile-overflow failures.
4. `upload(...)` works on connected hardware or returns a clear actionable
   failure.
5. `measure(...)` returns real telemetry or correctly typed runtime failures.
6. `evaluate(...)` returns normalized `DeviceMetrics` that the shared HIL
   controller can consume.
7. Tests cover the backend-specific control path.
8. This README and the root README document any required setup or platform
   caveats.

## Troubleshooting

If bring-up stalls, check these layers in order:

1. Registry:
   Is the backend reachable through `get_device(...)`?
2. Config:
   Are `device.name`, `device.serial_port`, and any backend options resolving
   correctly?
3. Candidate prep:
   Is the backend producing the expected build directory or project layout?
4. Compile:
   Are flash/RAM diagnostics being parsed, not silently returned as sentinels?
5. Upload:
   Is the correct tool and transport being used for the target?
6. Runtime:
   Does the firmware emit the tokens the backend parser expects?

If you are unsure whether your target is "just another board" or "a new
backend", the answer is usually:

1. Arduino FQBN + Arduino CLI compile/upload means board bring-up.
2. Same STM32 Cube/ST Edge AI/ST-LINK/runtime contract may be close to board
   bring-up.
3. New toolchain or new runtime contract means new backend.
