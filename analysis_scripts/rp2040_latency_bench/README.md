<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# RP2040 Latency Bench

Latency/memory-only bring-up tools for the `ARDUINO_NANO_RP2040_CONNECT`
backend ([`arduino_nano_rp2040.py`](../../src/crest/microcontrollers/arduino_nano_rp2040.py)).

**Hardware requirement:** an Arduino Nano RP2040 Connect on USB for the bench
and upload-stress scripts. Export and `--compile-only` need no board. No
harness or INA228 is used.

Run everything from the repository root with the `crest` Conda environment
active and `make arduino-setup` done (it installs `arduino:mbed_nano`, the
same core as the Nano 33 BLE Sense).

## Scripts

- [`export_smallest_dscnn.py`](export_smallest_dscnn.py)
  Builds the smallest `audio_dscnn` configuration (2 blocks, 4 base channels
  with 0.75 pointwise scale, 3x3 kernels, `aggressive` strides, no dense
  head) for the UrbanSound8K 201x64 log-mel input. It exports full-int8 PTQ
  and stages `model.cc`, `model.h` and `crest_inference_latency.ino` into
  `models/rp2040_bench/dscnn_us8k_min_int8/`. The model is untrained, which
  does not affect latency or memory. Calibration uses random data unless
  `--calibration-npy` is given.
- [`run_rp2040_latency_bench.py`](run_rp2040_latency_bench.py)
  Compiles, uploads and measures through `ArduinoNanoRP2040Device`. With
  `--arena-kb auto` it walks the RP2040 arena sweep upward until
  `AllocateTensors()` succeeds. It writes `results/<model>_<timestamp>.txt`.
- [`rp2040_upload_stress.py`](rp2040_upload_stress.py)
  Compiles once, then uploads `--count` times (default 10). It writes
  `results/upload_stress_<mode>_<timestamp>.tsv` with one row per attempt.
  `--mode plain` runs bare `arduino-cli upload` with no retries;
  `--mode robust` adds the backend's 1200-bps touch and retries.
- [`stage_tflite_candidate.py`](stage_tflite_candidate.py)
  Stages existing `.tflite` files (for example the course-project subnets at
  `<subnet>/exports/<1s|2s>/int8.tflite`) as candidates under
  `models/rp2040_bench/<subnet>_<duration>_<variant>/`. When the subnet's
  `run.json` is present, `candidate.json` also records width, block channels,
  parameter count and int8 validation accuracy/macro-F1.
- [`analyze_tflite_memory.py`](analyze_tflite_memory.py)
  Per-layer MACs, peak live activations, and a predicted TFLM arena and
  static RAM, computed before flashing. The arena estimate is peak activations
  + CMSIS-NN conv scratch + 8 B per conv output channel + 2 KiB. On the
  Nano RP2040 Connect it was 0.5-3.5% above the measured `arena_used_bytes`.
- [`run_rp2040_bench_plan.py`](run_rp2040_bench_plan.py)
  Runs several candidates in order. Each gets a short probe (10 runs), then a
  full run sized to `--budget-s` (default 120 s, at least 30 and at most 100
  timed runs), with serial timeouts derived from the measured latency. The
  arena grows in 5 KiB steps on `arena_too_small`. `--skip-done` resumes an
  interrupted plan.
- [`summarize_rp2040_results.py`](summarize_rp2040_results.py)
  Writes `results/rp2040_summary.csv`, one row per (model, board): width,
  window, MACs, params, flash, static RAM, `arena_used_bytes`, median/p95,
  runs, MAC/s, stride fit, validation accuracy/macro-F1, core version and
  clock.

## Make Targets

```bash
make rp2040-export
make rp2040-compile
make rp2040-bench PORT=/dev/ttyACM0
make rp2040-upload-stress PORT=/dev/ttyACM0                               # plain
make rp2040-upload-stress PORT=/dev/ttyACM0 RP2040_STRESS_ARGS="--mode robust"
```

Benchmarking external models end to end:

```bash
python analysis_scripts/rp2040_latency_bench/stage_tflite_candidate.py <run>/subnets/*/exports/{1s,2s}/int8.tflite
python analysis_scripts/rp2040_latency_bench/analyze_tflite_memory.py models/rp2040_bench/DSCNN_64_SC_*_int8
python analysis_scripts/rp2040_latency_bench/run_rp2040_bench_plan.py --skip-done models/rp2040_bench/DSCNN_64_SC_*_int8
python analysis_scripts/rp2040_latency_bench/summarize_rp2040_results.py
```

On a WSL host with limited memory, set `ARDUINO_CLI_BIN` to a wrapper that
adds `--jobs 4` to `arduino-cli compile`. Unrestricted parallel mbed + TFLM
builds crashed a 9 GB WSL VM twice.

`RP2040_BENCH_ARGS` passes extra flags, for example
`RP2040_BENCH_ARGS="--runs 200 --warmup 10 --arena-kb 40"`.

## Results File

Header lines start with `#`: the arena sweep, then flash/RAM from the compile
summary (the arena is a static array, so it is inside `ram_bytes`), then
`arena_used_bytes` from the interpreter after `AllocateTensors()`, then
`n`, `median_invoke_us`, `p95_invoke_us` (nearest rank), min/max/mean and
CREST's own `timer output` mean. After the header come the raw per-run lines,
exactly as the board printed them:

```
board=nano_rp2040_connect model=dscnn_us8k_min_int8 run=1 invoke_us=<int> preprocess_us=-1 arena_used_bytes=<int> status=ok
```

An infeasible candidate gives one line with `status=infeasible
reason=<arena_too_small|flash_overflow|ram_overflow|compile_failed|upload_failed|invoke_failed|timeout_no_telemetry|...>`.

`preprocess_us` is always `-1`: the UrbanSound8K pipeline computes log-mel
features on the host, so the runner does no on-device preprocessing.
