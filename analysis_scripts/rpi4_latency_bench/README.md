<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Raspberry Pi 4 Latency Bench

Latency/memory bench for the `RASPBERRY_PI_4` backend
([`raspberry_pi4.py`](../../src/crest/microcontrollers/raspberry_pi4.py)).

**Hardware requirement:** a Raspberry Pi 4 reachable over SSH with key-based
login, and a Python with a LiteRT runtime on it (see the backend section in
[`../../src/crest/microcontrollers/README.md`](../../src/crest/microcontrollers/README.md#raspberry-pi-4)).
The script installs nothing on the Pi.

```bash
python analysis_scripts/rpi4_latency_bench/run_rpi4_latency_bench.py \
  models/rp2040_bench/DSCNN_64_SC_*_int8 \
  --host <pi-host> --user <pi-user> --python crest-litert/bin/python \
  --runs 100 --warmup 5 --threads 1
```

- Candidates are any directories with `g_model.tflite`, e.g. the ones staged
  by `analysis_scripts/rp2040_latency_bench/stage_tflite_candidate.py`.
- Each run writes `results/<model>_rpi4_t<threads>_<timestamp>.txt` in the
  RP2040 bench's results format. Its `ram_bytes` is the runner's peak RSS
  increase and `flash_bytes` the `.tflite` size, followed by the raw per-run
  lines and the environment (runtime version, threads, CPU governor and
  frequency, temperature, throttle flags).
- Use `--threads 4` for all cores, and `--no-xnnpack` to measure the
  reference kernels instead of the XNNPACK delegate.
- Before any run, the script checks that every core uses the
  `--require-governor` cpufreq governor (default `performance`). If not, it
  stops without running anything.
- Each results file records `vcgencmd get_throttled` before and after the
  timed runs, plus the temperature. `flags` marks under-voltage, frequency
  capping, throttling or soft temperature limits that are active after the
  run or were newly set during it, and temperatures >= 80 C (`none` when
  clean).

Summary CSV (`results/rpi4_summary.csv`, one row per model and thread count;
the `rp2040_summary.csv` columns plus `threads`, `xnnpack`, `governor`,
`temp_after_c`, `throttled_before`, `throttled_after` and `flags`):

```bash
python analysis_scripts/rpi4_latency_bench/summarize_rpi4_results.py --threads 1 4 \
  --models dscnn_us8k_min_int8 DSCNN_64_SC_25_1s_int8 DSCNN_64_SC_25_2s_int8 \
    DSCNN_64_SC_5_1s_int8 DSCNN_64_SC_5_2s_int8 DSCNN_64_SC_1_1s_int8 DSCNN_64_SC_1_2s_int8
```

On the Pi, `static_ram_bytes` is the runner's peak RSS increase. It is mostly
interpreter/XNNPACK setup (~10-11.5 MB for every model), so it is not
comparable to the RP2040's static RAM.
