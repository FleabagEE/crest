# Model-grid triage (offline)

`triage_model_grid.py` checks every int8 export of the course-project model grid
before any hardware run. It covers operator support, the RP2040 arena/RAM
prediction, the ESP32-S3 arena range, a latency triage estimate and a
measurement plan. It needs no board.

```bash
# Course repo is read-only: extract its models/ instead of checking out a branch.
git -C ~/m202a/embedded-systme-project fetch
mkdir -p "$SCRATCH/grid" && git -C ~/m202a/embedded-systme-project archive origin/main models | tar -x -C "$SCRATCH/grid"

# Optional LiteRT check (Pi runtime, ai-edge-litert 2.3.0) in a throwaway venv:
python -m venv --system-site-packages "$SCRATCH/litert_venv"
"$SCRATCH/litert_venv/bin/pip" install --no-deps ai-edge-litert==2.3.0

LD_LIBRARY_PATH=$CONDA_PREFIX/lib python analysis_scripts/model_grid/triage_model_grid.py \
    --models-root "$SCRATCH/grid/models" --litert-python "$SCRATCH/litert_venv/bin/python" \
    --source-label "embedded-systme-project origin/main c250591, models/"
```

`LD_LIBRARY_PATH` is only needed when ROS's old `libstdc++` leaks into the shell
(librosa/scipy then fail with a `CXXABI_1.3.15` error).

Outputs (committed, so the triage is reproducible): `model_grid_triage.csv`
(one row per model) and `model_grid_summary.md`, next to this script.

## What each column group means

- **Metadata**: comes from `report.json` (the run report, or a subnet's own
  `report.json`). Accuracy is the int8 TFLite validation fold 9. `test_acc` is
  filled only where the report has a test-fold (10) evaluation.
- **MACs**: counted per layer from the .tflite (conv, depthwise and FC), and
  checked against the report's `estimated_macs`. All 48 match.
- **Operator support**:
  - `rp2040_ops`: checked against Chirale's `AllOpsResolver`.
  - `esp32_ops_runtime`: checked against the esp32 core's esp-tflite-micro.
  - `esp32_sketch_resolver_missing`: ops the latency sketch's
    `MicroMutableOpResolver` doesn't register yet.
  - `litert_*`: a real load + invoke under LiteRT, with the number of XNNPACK
    delegate partitions.
- **Memory**:
  - RP2040 uses the calibrated `analyze_tflite_memory.py` estimate (+10%), so
    static RAM = 50,512 B + arena. `fits` means within the CREST cap of
    225,000 B. `borderline` means it fits only without the margin, or only up
    to the 270,336 B linker limit. `no` means over the linker limit.
  - ESP32-S3: the arena is the RP2040 estimate x [0.99, 2.08], the measured
    ESP-NN ratio range. Static RAM = 45,780 B + arena, against a 327,680 B
    limit. `fits` means the high end fits, `borderline` means only the low end
    fits, and `no` means use a PSRAM arena.
  - The estimator was calibrated on standard-conv DS-CNNs only
    (`estimator_calibrated_for_graph`).
- **Latency (TRIAGE ESTIMATE, not a result)**:
  - Conv and FC MACs are divided by the measured MAC/s of the standard-conv
    DS-CNN with the same width and window.
  - Depthwise MACs use a range: that same rate (optimistic) down to the
    overhead-bound depthwise baseline (pessimistic).
  - `miss` means even the optimistic estimate exceeds the stride. `at_risk`
    means only the pessimistic one does.
  - `*_logmel_same_core` repeats the check with the 64-mel streaming log-mel
    duty (RP2040 46%, ESP32-S3 6.2%) taken off the stride, i.e. log-mel on the
    same core.
- **Plan**:
  - Skips models that are already measured (identical graph signature to the
    measured DSCNN_64_SC exports), and on the RP2040, models predicted to
    overflow RAM or to miss the stride.
  - Timed runs: 100 below 0.3 s, ~60 s of runs up to 1 s, and 10-30 above 1 s
    (run-to-run spread has been under 0.1%).
  - Minutes come from the earlier bench logs: compile + upload + probe, plus
    one arena retry for uncalibrated graphs and for the ESP32.

## Running the ESP32-S3 part of the plan

[`run_esp32_grid.py`](run_esp32_grid.py) runs every planned ESP32-S3 model
through `run_rp2040_bench_plan.py --board xiao_esp32s3 --skip-done`, so it
resumes after an interruption:

```bash
ARDUINO_CLI_BIN=<arduino-cli wrapper with --jobs 4> \
  python analysis_scripts/model_grid/run_esp32_grid.py [MODEL ...]
```

- Runs: the planned timed / warm-up runs from `model_grid_triage.csv`.
- Arena: starts at max(1.7 x RP2040 estimate, estimate + largest conv filter)
  + 4 KiB and grows in 32 KiB steps (`--arena-step-kb`). Internal SRAM is
  capped at 232 KiB, the largest arena that links (233 KiB overflows
  `dram0_0_seg`), so the ESP32 static-RAM limit is ~283.5 KB in practice,
  not the 327,680 B the triage assumes.
- PSRAM: used where the plan says so. A model that fails in internal SRAM is
  retried with a PSRAM arena starting at 232 KiB.
