<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Config Reference

This directory documents the runtime configuration surface for CREST.

For the source architecture and extension map, see
[`../README.md`](../README.md).

Case-study reproducibility configs live under
[`case_study_configs/`](case_study_configs/).

## Example Configs

| If You Want To... | Start With | Hardware Requirement |
|-------------------|------------|----------------------|
| Train or search OxIOD models without hardware energy metrics | [`nas_config_flops_rmse.yaml`](nas_config_flops_rmse.yaml) | No hardware required |
| Run STM32 OxIOD HIL experiments | [`nas_config_stm32.yaml`](nas_config_stm32.yaml) | NUCLEO-N657X0-Q board required; HIL harness required for energy measurement |
| Run Arduino OxIOD HIL experiments | [`nas_config_ble.yaml`](nas_config_ble.yaml) or [`nas_config_portenta.yaml`](nas_config_portenta.yaml) | Target board required; HIL harness required for energy measurement |
| Run UrbanSound8K audio HIL experiments | [`nas_config_audio_stm32.yaml`](nas_config_audio_stm32.yaml) or [`nas_config_audio_portenta.yaml`](nas_config_audio_portenta.yaml) | Target board and HIL harness required as shipped; use these as templates with HIL/measured scoring disabled for hardware-free training |

- [`nas_config_stm32.yaml`](nas_config_stm32.yaml)
  NUCLEO-N657X0-Q OxIOD measured-energy NAS example and the most complete
  commented reference for score/prune policy shapes.
- [`nas_config_ble.yaml`](nas_config_ble.yaml)
  Arduino Nano 33 BLE Sense OxIOD measured-energy NAS example.
- [`nas_config_portenta.yaml`](nas_config_portenta.yaml)
  Portenta H7 CM7 OxIOD measured-energy NAS example.
- [`nas_config_audio_stm32.yaml`](nas_config_audio_stm32.yaml)
  NUCLEO-N657X0-Q UrbanSound8K / DS-CNN audio NAS example.
- [`nas_config_audio_portenta.yaml`](nas_config_audio_portenta.yaml)
  Portenta H7 CM7 UrbanSound8K / DS-CNN audio NAS example.
- [`nas_config_flops_rmse.yaml`](nas_config_flops_rmse.yaml)
  Pure desktop OxIOD NAS example that optimizes validation RMSE and FLOPs.
- [`nas_config_memory_proxy.yaml`](nas_config_memory_proxy.yaml)
  Pure desktop OxIOD NAS example that optimizes validation RMSE and static
  memory traffic.

## Supporting Files

- [`stm32_nucleo_mypool.json`](stm32_nucleo_mypool.json)
  ST Edge AI memory-pool description used by STM32 configs when
  `device.stm32.weight_storage_mode: external_flash` is enabled. This is a
  repo-local version modified from the STM32 LRUN example for the
  NUCLEO-N657X0-Q externalized-weights flow.

## Audio Fold Rotation

Audio configs can optionally run final reporting across UrbanSound8K folds:

- `task.params.evaluation.protocol: fixed_split | fold_rotation`
- `task.params.evaluation.fold_rotation.test_folds`, defaulting to all 10
  UrbanSound8K folds.
- `dataset.params.fold_rotation_cache_dir`, required only when fold rotation is
  enabled.

The fixed `dataset.params.cache_dir` remains the source for NAS, HIL runs, and
deployable export. Fold rotation runs after the fixed-split final checkpoint and
does not export per-fold models.
`task.params.evaluation.protocol: fold_rotation` is single-objective only;
multi-objective NAS fails during config validation. The dataset owns where
fold-specific caches live; the task owns whether final reporting rotates
through those folds.

The runtime loader and validator live in
[`../crest/model.py`](../crest/model.py), while the task-aware bootstrap
that resolves components and validates NAS policy against the active task lives
in [`../crest/runtime_bootstrap.py`](../crest/runtime_bootstrap.py).

## Top-Level Blocks

The main top-level blocks are:

- `device`
  Hardware target, HIL runtime behavior, timing, harness options, and
  backend-owned device options.
- `dataset`
  Required dataset selection block. The built-in OxIOD dataset uses
  `dataset.name: oxiod` plus the keys under `dataset.params`.
- `task`
  Required task selection block. Task-owned final reporting policy, such as
  audio fold rotation, lives under `task.params.evaluation`.
- `model`
  Required model-family selection block.
- `training`
  NAS and training limits plus runtime-side training switches such as
  `energy_aware` and `input_mode`.
- `nas`
  Scoring and pruning policy.
- `outputs`
  Output directories and derived artifact naming inputs.
- `network`
  HIL server/client socket settings.
- `logging`
  Runtime log level.

Those component blocks are resolved by
[`../crest/component_selection.py`](../crest/component_selection.py), and
they are mandatory. Use `dataset`, `task`, and `model`; the old top-level
`data` block is not supported.

## `device`

The `device` block owns target selection and runtime behavior.

Common keys:

- `device.name`
  Target device identifier.
- `device.hil`
  Enables or disables hardware-in-the-loop measurement.
- `device.compile_when_hil_disabled`
  Controls whether `device.hil: false` still calls the HIL server for
  compile-only proxy metrics. Valid values are `auto`, `true`, and `false`;
  YAML booleans are accepted and normalized to `true`/`false`. The default
  `auto` compiles only when the active score/prune policy references
  compile-derived metrics such as `ram_bytes`, `flash_bytes`,
  `external_flash_bytes`, or `arena_bytes`.
- `device.runtime_mode`
  `back_to_back` or `cadenced`.
- `device.latency_budget_ms`
  Optional shared cadence-budget override. When omitted, the runtime derives
  it from the active dataset cadence: first `dataset.params.batch_period_ms`,
  then `dataset.params.stride / dataset.params.sampling_rate_hz * 1000` for
  the built-in `oxiod` dataset.
- `device.serial_port`
  DUT serial port.
- `device.measured_inference_runs`
  Number of repeated inference invokes averaged into one measured pass.
- `device.serial_timeout_s`
- `device.dut_ready_timeout_s`
- `device.cpu_clock_mhz_options`
  Optional per-trial CPU presets for boards that support runtime clock
  selection.

Harness-related keys:

- `device.harness_serial_port`
- `device.harness_fqbn`
- `device.harness_auto_flash`
- `device.harness_arm_pin`
- `device.harness_trigger_pin`
- `device.dut_arm_hold_ms`
- `device.harness_stable_low_ms`
- `device.harness_ready_timeout_s`
- `device.harness_arm_timeout_s`
- `device.harness_active_timeout_s`
- `device.harness_done_timeout_s`

Per-backend nested blocks include:

- `device.portenta.*`
- `device.stm32.*`
- `device.rp2040.*` (Nano RP2040 Connect, latency/memory only):
  `arena_sizes_kb` (ascending arena sweep in KiB), `warmup_runs` (default 5),
  `upload_timeout_s` (default 90), `upload_retries` (default 2), and
  `port_wait_s` (default 10). See
  [`nas_config_audio_rp2040.yaml`](nas_config_audio_rp2040.yaml).

STM32 option plumbing is resolved by
[`../crest/microcontrollers/__init__.py`](../crest/microcontrollers/__init__.py).
Examples of STM32-owned keys include:

- `template_root`
- `project_root`
- `gdbserver`
- `gdb`
- `cubeprog_bin`
- `signing_tool`
- `gdb_port`
- `apid`
- `server_ready_timeout_s`
- `wake_margin_us`
- `min_sleep_us`
- `weight_storage_mode`
- `appli_flash_address`
- `weights_flash_address`
- `weights_memory_pool`
- `weights_external_loader`
- `signing_load_offset`
- `signing_header_version`
- `max_external_flash_bytes`

Validation notes:

- `device.runtime_mode` must be `back_to_back` or `cadenced`.
- `device.stm32.runtime_mode` is no longer supported. Use
  `device.runtime_mode` instead.
- `device.stm32.project_layout` is no longer supported. LRUN `dev_boot` is
  implicit for the STM32 backend.
- `device.latency_budget_ms` must be positive when set.
- `device.measured_inference_runs` must be an integer `>= 1`.
- `device.cpu_clock_mhz_options` must be a non-empty integer list when set.
- For `STM32_NUCLEO_N657X0_Q`, `device.cpu_clock_mhz_options` is validated
  against the backend-supported set in code.
- For `PORTENTA_H7` and `ARDUINO_NANO_33_BLE_SENSE`, cadenced mode requires
  `training.input_mode: uniform`.

## `training`

The `training` block owns the main NAS/training runtime switches.

Common keys in the shipped config:

- `training.nas_epochs`
- `training.model_epochs`
- `training.nas_trials`
- `training.nas_multiobjective_population_size`
- `training.max_total_trials`
- `training.quantization`
- `training.latency_proxy_max_flops`
- `training.train`
- `training.energy_aware`
- `training.input_mode`

Runtime behavior:

- `training.energy_aware` defaults to `false` when omitted
- `training.quantization` is required and must use the mapping shape:
  `mode`, `search`, and non-empty `choices`. The main measured-board configs
  search `choices: [float, int8_ptq]` so float32 and int8 PTQ exports can be
  compared on the same backend.
- Supported quantization modes are `float` and `int8_ptq`. Enabling
  `training.quantization.search: true` samples `quantization_mode` from
  `choices`; this expands the effective NAS search space and usually needs a
  larger trial budget. Mixed `float`/`int8_ptq` studies also conflate
  architecture quality with quantization effects, so compare them deliberately.
- `int8_ptq` export uses full-integer TensorFlow Lite conversion with int8
  inputs and outputs. Representative calibration data is taken from the
  explicit calibration split when available, otherwise the training split, and
  conversion samples up to 1000 windows evenly across that split. This avoids
  calibrating on one contiguous prefix of an ordered time-series split.
- HIL metrics are deployment-mode preflight metrics collected before training.
  Per-trial NAS scoring evaluates the trained checkpoint with host-side TFLite
  on the validation split using interpreter-provided input/output
  scale/zero-point values and TFLite signature output order; final fixed-split
  reporting exports/evaluates the trained TFLite on the test split after
  `train_best_trial`.
- Some closeout artifact paths are Keras-derived unless the path explicitly
  requests TFLite evaluation.
- `training.input_mode` defaults to `uniform` when omitted
- `training.input_mode` supports dataset-agnostic `uniform` plus
  dataset-specific analysis modes: `oxiod_representative`, `oxiod_real`,
  `urbansound8k_representative`, and `urbansound8k_real`
- `training.max_total_trials` defaults to `training.nas_trials * 2` when
  omitted
- `training.nas_trials` is the target number of feasible completed trials when
  `nas.feasibility.rules` is enabled. Infeasible, failed, and pruned attempts
  still count against `training.max_total_trials`, so constrained hardware
  runs usually need a larger total-attempt budget than the feasible target.

## `dataset`, `task`, and `model`

The modular component-selection surface is resolved by
[`../crest/component_selection.py`](../crest/component_selection.py).

Keys:

- `dataset.name`
- `dataset.params`
- `task.name`
- `task.params`
- `model.family`
- `model.params`
- `model.search`

For `audio_dscnn`, `model.search: {}` means "use the model-family default
search surface" from `AudioDSCNNFamily.AUDIO_DSCNN_SEARCH_CHOICES`. Add keys
under `model.search` only when you want to narrow that default surface.

Validation notes:

- `dataset`, `task`, and `model` are required top-level blocks
- `dataset.params` is required for the built-in `oxiod` dataset path
- dataset classes are instantiated as zero-argument classes
- model family classes are instantiated as zero-argument classes
- task classes are expected to use the explicit keyword-only constructor
  contract `__init__(*, checkpoint_path, early_stopping_patience)`; the runtime
  does not probe constructor signatures or provide compatibility shims for
  older task classes

Minimal example:

```yaml
dataset:
  name: oxiod
  params:
    directory: "data/oxiod/"
    sampling_rate_hz: 100
    window_size: 200
    stride: 20
    calibration_windows: 10000

task:
  name: odometry_regression
  params:
    early_stopping_patience: 40

model:
  family: odom_tcn
  params:
    export_variant: approx_trained
  search: {}
```

## `nas`

The `nas` block owns scoring, pruning, and constrained-feasibility policy.

Structure:

- `nas.score.type`
  `scoring-function` or `multi-objective`
- `nas.score.metrics`
  Optional derived metrics
- `nas.score.params`
  Score terms or objectives depending on `score.type`
- `nas.prune.rules`
  Optional post-build/pre-fit hard termination gates
- `nas.feasibility.train_if_infeasible`
  Whether trials that violate feasibility constraints should still train and
  return real objective values. Defaults to `false`.
- `nas.feasibility.rules`
  Optional post-build/pre-fit deployability constraints using the same rule
  shape as `nas.prune.rules`: `rule`, `metric`, `condition`, `reference`, and
  `reason`.

Built-in derived metric types validated in code include:

- `add`
- `energy-budget-from-power`

Scalar term types include:

- `weighted`
- `normalized-weighted`
- `boundary`
- `target`

Practical guidance:

- use `scoring-function` when you want one scalar score
- use `multi-objective` when you want a Pareto front instead of one scalar
- use `nas.prune.rules` with either score type when a pre-fit metric should
  hard-stop a trial before feasibility is evaluated. Rules run after model
  build/compile, FLOP counting,
  and HIL/compile metric collection, but before `task.build_fit_plan`,
  `model.fit`, TFLite validation, or Keras validation. Multi-objective gate
  hits are logged with `pruned=True` but remain Optuna COMPLETE trials with
  direction-aware penalty values.
- use `nas.feasibility.rules` for deployability constraints that should be
  visible to Optuna constrained samplers. Feasibility is evaluated after hard
  failures and `nas.prune.rules`; each rule persists one signed constraint
  where `<= 0` is feasible and `> 0` is infeasible. With
  `train_if_infeasible: false`, infeasible trials skip training and return
  penalties while still consuming `training.max_total_trials`. With
  `train_if_infeasible: true`, trials train normally but remain
  constrained-infeasible for samplers, CSV filtering, and plots.
- move latency/deadline budget gates such as `latency_ms > latency_budget_ms`
  or `cadenced_deadline_miss_count > 0` to `nas.feasibility.rules` when you
  want constrained dominance. Keep the same metric in `nas.prune.rules` only
  when you deliberately want hard early termination for debugging or resource
  protection.
- keep non-HIL configs away from score/prune terms that require measured
  latency or energy. `latency_ms`, energy/power/current/voltage metrics,
  `clock_hz`, `harness_latency_ms`, and `cadenced_*` metrics require
  `device.hil: true`.
- set `device.compile_when_hil_disabled: false` for pure desktop scores such
  as RMSE/FLOPs; leave it as `auto` when non-HIL score/prune terms still need
  compile-derived resource metrics.
- in cadenced multi-objective runs, overload is telemetry unless you add a
  `nas.feasibility.rules` constraint such as
  `cadenced_deadline_miss_count > 0`

The most complete commented examples remain in
[`nas_config_stm32.yaml`](nas_config_stm32.yaml) itself.

## `outputs`

The `outputs` block controls directory roots and naming inputs.

Keys:

- `outputs.models_dir`
- `outputs.candidate_dir`
- `outputs.artifact_stem`
- `outputs.log_file_name`

Runtime notes:

- `load_config(...)` derives read-only runtime fields `model_name` and
  `checkpoint_name` from `outputs.artifact_stem` and `device.name`, then
  populates derived paths such as `tflite_model_path` and `checkpoint_path`
- YAML-authored `outputs.model_name` and `outputs.checkpoint_name` are rejected
  because artifact names now follow the shared `{artifact_stem}_{device.name}`
  rule
- `models_dir` and `candidate_dir` are resolved into absolute paths in memory

So the final in-memory values may differ from the literal YAML text.

## `network`

The `network` block owns HIL socket settings.

Keys:

- `network.host`
- `network.port`
- `network.recv_timeout_sec`
- `network.send_timeout_sec`

These must match the HIL client/server deployment you actually run.

## `logging`

The `logging` block exposes:

- `logging.level`

Valid values are:

- `CRITICAL`
- `ERROR`
- `WARNING`
- `INFO`
- `DEBUG`
- `NOTSET`

## Where To Look Next

Use these files together:

- [`nas_config_stm32.yaml`](nas_config_stm32.yaml) for the STM32 commented example
- [`../crest/model.py`](../crest/model.py) for validation and derived
  runtime behavior
- [`../README.md`](../README.md) for source architecture
- [`../crest/model_families/README.md`](../crest/model_families/README.md)
  for model-family selection and extension
- [`../crest/microcontrollers/README.md`](../crest/microcontrollers/README.md)
  for backend-owned hardware options
