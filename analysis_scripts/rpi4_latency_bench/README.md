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
