# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Shared CREST scoring, config-validation, and model helpers.

This module carries the shared NAS scoring DSL, YAML config validation,
runtime metric normalization, CSV trial logging, and the built-in CREST
model helpers that are still reused across the componentized runtime.
"""

import csv
import fcntl
import json
import itertools
import logging
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import tensorflow as tf
import yaml
from addict import Dict

# import optuna
from tcn import TCN
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, MaxPooling1D, Reshape

from .model_metrics import count_flops_keras
from .hardware import (
    HIL_controller,
    describe_error_code,
    normalize_power_metrics,
)
from .microcontrollers import (
    get_device as get_microcontroller_device,
    resolve_device_options,
)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "nas_config_stm32.yaml"
REPO_ROOT = Path(__file__).resolve().parents[2]
STM32_DEFAULT_PROJECT_ROOT = (
    REPO_ROOT
    / "sketches"
    / "stm32"
    / "crest_stm32_lrun"
)
MIN_TCN_LAYERS = 3
MAX_TCN_LAYERS = 8
DILATION_POOL = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DILATION_CANDIDATES = [
    list(combo)
    for layer_count in range(MIN_TCN_LAYERS, MAX_TCN_LAYERS + 1)
    for combo in itertools.combinations(DILATION_POOL, layer_count)
]
DROP_RATE_CHOICES = [0.0, 0.1, 0.2, 0.3, 0.4]
logger = logging.getLogger(__name__)

VALID_SCORE_TYPES = {"scoring-function", "multi-objective"}
VALID_QUANTIZATION_MODES = {"float", "int8_ptq"}
VALID_COMPILE_WHEN_HIL_DISABLED = {"auto", "true", "false"}
BOARD_QUANTIZATION_CAPABILITIES = {
    "STM32_NUCLEO_N657X0_Q": {"float", "int8_ptq"},
    "PORTENTA_H7": {"float", "int8_ptq"},
    "ARDUINO_NANO_33_BLE_SENSE": {"float", "int8_ptq"},
    "ARDUINO_NANO_RP2040_CONNECT": {"float", "int8_ptq"},
}
VALID_DERIVED_METRIC_TYPES = {"add", "energy-budget-from-power"}
VALID_TERM_TYPES = {"weighted", "normalized-weighted", "boundary", "target"}
VALID_OBJECTIVE_DIRECTIONS = {"maximize", "minimize"}
VALID_PRUNE_CONDITIONS = {"gt", "gte", "lt", "lte"}
FEASIBILITY_EQUALITY_EPSILON = 1e-12
NONNEGATIVE_METRICS = {
    "measurement_deployment_eligible",
    "ram_bytes",
    "flash_bytes",
    "max_ram_bytes",
    "max_flash_bytes",
    "external_flash_bytes",
    "flops",
    "weight_bytes",
    "activation_bytes",
    "memory_traffic_bytes",
    "memory_proxy_dtype_bytes",
    "memory_proxy_warning_count",
    "latency_ms",
    "energy_mj_per_inference",
    "avg_power_mw",
    "avg_current_ma",
    "bus_voltage_v",
    "cpu_clock_mhz_requested",
    "clock_hz",
    "latency_budget_ms",
    "arena_bytes",
    "cadenced_error_code",
    "cadenced_active_inference_latency_ms",
    "cadenced_window_latency_ms",
    "cadenced_energy_mj_per_window",
    "cadenced_energy_mj_per_trial",
    "cadenced_rtc_sleep_ms",
    "cadenced_deadline_miss_count",
}
INFRASTRUCTURE_SCORE_METRICS = {
    "measurement_deployment_eligible",
    "ram_bytes",
    "flash_bytes",
    "max_ram_bytes",
    "max_flash_bytes",
    "external_flash_bytes",
    "flops",
    "weight_bytes",
    "activation_bytes",
    "memory_traffic_bytes",
    "memory_proxy_dtype_bytes",
    "memory_proxy_warning_count",
    "latency_ms",
    "energy_mj_per_inference",
    "avg_power_mw",
    "avg_current_ma",
    "bus_voltage_v",
    "cpu_clock_mhz_requested",
    "clock_hz",
    "latency_budget_ms",
    "arena_bytes",
    "error_code",
    "cadenced_error_code",
    "cadenced_active_inference_latency_ms",
    "cadenced_window_latency_ms",
    "cadenced_energy_mj_per_window",
    "cadenced_energy_mj_per_trial",
    "cadenced_rtc_sleep_ms",
    "cadenced_deadline_miss_count",
}
DEFAULT_TASK_SCORE_METRICS = {"rmse_vel_x", "rmse_vel_y", "rmse_total"}
DEFAULT_TRAINING_ONLY_TASK_METRICS = {"rmse_vel_x", "rmse_vel_y", "rmse_total"}
# These constants describe the built-in odometry task only. Task-aware NAS
# validation must not treat them as the implicit contract for arbitrary tasks.
TRAINING_ONLY_METRICS = DEFAULT_TRAINING_ONLY_TASK_METRICS
BUILTIN_SCORE_METRICS = INFRASTRUCTURE_SCORE_METRICS | DEFAULT_TASK_SCORE_METRICS

CADENCED_NUMERIC_FIELD_DEFAULTS = {
    "cadenced_error_code": -1,
    "cadenced_active_inference_latency_ms": -1.0,
    "cadenced_window_latency_ms": -1.0,
    "cadenced_energy_mj_per_window": -1.0,
    "cadenced_energy_mj_per_trial": -1.0,
    "cadenced_avg_power_mw": -1.0,
    "cadenced_avg_current_ma": -1.0,
    "cadenced_bus_voltage_v": -1.0,
    "cadenced_idle_power_mw": -1.0,
    "cadenced_harness_latency_ms": -1.0,
    "cadenced_clock_hz": -1.0,
    "cadenced_dwt_cycles_per_inference": -1.0,
    "cadenced_rtc_sleep_ms": -1.0,
    "cadenced_deadline_miss_count": -1,
    "cadenced_wake_recovery_us_mean": -1.0,
    "cadenced_wake_overshoot_us_mean": -1.0,
    "cadenced_rtc_clock_hz_nominal": -1.0,
}
CADENCED_STRING_FIELDS = (
    "cadenced_error_label",
    "cadenced_rtc_clock_source",
    "cadenced_timing_quality",
    "cadenced_stop_mode_variant",
)
CADENCED_ALL_FIELDS = (
    "runtime_mode",
    *tuple(CADENCED_NUMERIC_FIELD_DEFAULTS.keys()),
    *CADENCED_STRING_FIELDS,
)
CADENCED_CSV_FIELDS = (
    "runtime_mode",
    "cadenced_active_inference_latency_ms",
    "cadenced_window_latency_ms",
    "cadenced_energy_mj_per_window",
    "cadenced_energy_mj_per_trial",
    "cadenced_rtc_sleep_ms",
    "cadenced_deadline_miss_count",
    "cadenced_error_code",
    "cadenced_error_label",
)
TRIAL_LOG_STABLE_COLUMNS = (
    "study_name",
    "timestamp_unix",
    "timestamp_readable",
    "score",
    "quantization_mode",
    "ram_bytes",
    "flash_bytes",
    "external_flash_bytes",
    "weight_storage_mode",
    "flops",
    "weight_bytes",
    "activation_bytes",
    "memory_traffic_bytes",
    "memory_proxy_dtype_bytes",
    "memory_proxy_warning_count",
    "latency_ms",
    "latency_budget_ms",
    "energy_mj_per_inference",
    "avg_power_mw",
    "avg_current_ma",
    "bus_voltage_v",
    "cpu_clock_mhz_requested",
    "clock_hz",
    "error_code",
    "error_label",
    "score_type",
    "objective_names_json",
    "objective_values_json",
    "trial_outcome",
    "objective_directions_json",
    "artifact_summary_json",
    "pruned",
    "prune_reason",
    "prune_rule",
    "feasible",
    "feasibility_status",
    "feasibility_rule",
    "feasibility_reason",
    "feasibility_metric",
    "feasibility_value",
    "feasibility_reference",
    "feasibility_violation",
    "feasibility_constraints_json",
    *CADENCED_CSV_FIELDS,
)


def _minimum_stm32_serial_timeout_s(
    *,
    runtime_mode: str,
    latency_budget_ms: float,
    measured_inference_runs: int,
) -> float:
    """Return the minimum practical STM32 serial timeout for one HIL attempt.

    Parameters
    ----------
    runtime_mode : str
        Configured STM32 runtime mode, usually ``"back_to_back"`` or
        ``"cadenced"``.
    latency_budget_ms : float
        Per-slot cadence budget in milliseconds.
    measured_inference_runs : int
        Number of measured on-device runs in the attempt.

    Returns
    -------
    float
        Minimum timeout in seconds that should be allowed for the STM32 HIL
        attempt.
    """
    normalized_runtime_mode = str(runtime_mode).strip().lower()
    safe_runs = max(1, int(measured_inference_runs))
    if normalized_runtime_mode == "cadenced":
        return max(30.0, (float(latency_budget_ms) * safe_runs) / 1000.0 + 10.0)
    return 30.0


class TrialLike(Protocol):
    """Minimal Optuna Trial interface used by log_trial."""

    def set_user_attr(self, key: str, value: Any) -> None:
        """Attach auxiliary metadata to a trial.

        Parameters
        ----------
        key : str
            Attribute name to store.
        value : Any
            JSON-serializable value associated with ``key``.

        Returns
        -------
        None
        """
        ...


@dataclass(frozen=True)
class ScoreEvaluationResult:
    """Resolved score/objective values for one evaluated trial.

    Parameters
    ----------
    score : float | None
        Scalar score for single-objective runs. ``None`` for multi-objective
        runs.
    objective_names : list[str]
        Ordered objective labels.
    objective_values : list[float]
        Ordered objective values.
    objective_directions : list[str]
        Ordered objective directions matching Optuna conventions.

    Attributes
    ----------
    score : float | None
        Scalar score for single-objective runs. ``None`` for multi-objective
        runs.
    objective_names : list[str]
        Ordered objective labels.
    objective_values : list[float]
        Ordered objective values.
    objective_directions : list[str]
        Ordered objective directions matching Optuna conventions.
    """

    score: float | None
    objective_names: list[str]
    objective_values: list[float]
    objective_directions: list[str]


@dataclass(frozen=True)
class TrialOutcome:
    """Generic trial outcome used by NAS logging and pruning paths.

    Parameters
    ----------
    score : float | None
        Scalar score for single-objective runs. ``None`` for multi-objective
        runs.
    objective_names : list[str]
        Ordered objective labels.
    objective_values : list[float | None]
        Ordered objective values; null entries mark failed infrastructure attempts.
    objective_directions : list[str]
        Ordered objective directions matching Optuna conventions.
    task_metrics : dict[str, Any]
        Task-owned metric payload recorded for this trial.
    hyperparams : dict[str, Any]
        Resolved trial hyperparameters used for HIL, scoring, and logging.
    artifact_summary : dict[str, Any] | None
        JSON-safe task-owned artifact summary associated with the trial.
    quantization_mode : str
        Deployment quantization mode used for this trial.

    Attributes
    ----------
    score : float | None
        Scalar score for single-objective runs. ``None`` for multi-objective
        runs.
    objective_names : list[str]
        Ordered objective labels.
    objective_values : list[float | None]
        Ordered objective values; null entries mark failed infrastructure attempts.
    objective_directions : list[str]
        Ordered objective directions matching Optuna conventions.
    task_metrics : dict[str, Any]
        Task-owned metric payload recorded for this trial.
    hyperparams : dict[str, Any]
        Resolved trial hyperparameters used for HIL, scoring, and logging.
    artifact_summary : dict[str, Any] | None
        JSON-safe task-owned artifact summary associated with the trial.
    quantization_mode : str
        Deployment quantization mode used for this trial.
    """

    score: float | None
    objective_names: list[str]
    objective_values: list[float | None]
    objective_directions: list[str]
    task_metrics: dict[str, Any]
    hyperparams: dict[str, Any]
    artifact_summary: dict[str, Any] | None = None
    quantization_mode: str = "int8_ptq"


@dataclass(frozen=True)
class FeasibilityEvaluation:
    """Resolved feasibility policy outcome for one pre-training trial.

    Parameters
    ----------
    feasible : bool
        Whether every configured feasibility constraint is satisfied.
    status : str
        Stable status label: ``"feasible"``, ``"infeasible"``, or
        ``"not_evaluated"``.
    constraints : list[float]
        Signed Optuna constraint values in configuration order.
    first_violation : dict[str, Any] | None
        Metadata for the first violating rule in configuration order.

    Attributes
    ----------
    feasible : bool
        Whether every configured feasibility constraint is satisfied.
    status : str
        Stable status label: ``"feasible"``, ``"infeasible"``, or
        ``"not_evaluated"``.
    constraints : list[float]
        Signed Optuna constraint values in configuration order.
    first_violation : dict[str, Any] | None
        Metadata for the first violating rule in configuration order.
    """

    feasible: bool
    status: str
    constraints: list[float]
    first_violation: dict[str, Any] | None = None


class ScoreConfigEvaluationError(ValueError):
    """Raised when a configured score cannot be evaluated at runtime.

    This is reserved for score-config resolution failures such as unavailable
    metrics, invalid runtime references, or derived metrics that cannot be
    computed from the current trial context.
    """

def set_error_code(metrics: dict, code: int) -> None:
    """Attach a numeric error code and its descriptive label to `metrics`.

    Parameters
    ----------
    metrics : dict
        Mutable metric dictionary updated in place.
    code : int
        CREST HIL error code to attach to the metrics.
    """
    metrics["error_code"] = code
    metrics["error_label"] = describe_error_code(code)


def apply_cadenced_metric_defaults(
    metrics: dict[str, Any],
    power_metrics: dict[str, Any] | None,
) -> None:
    """Populate cadenced metric fields from raw backend metrics plus defaults.

    Parameters
    ----------
    metrics : dict[str, Any]
        Final normalized metrics dictionary being assembled for the caller.
    power_metrics : dict[str, Any] | None
        Raw backend metrics payload that may contain ``runtime_mode`` and
        flattened ``cadenced_*`` fields.

    Returns
    -------
    None
        The function mutates ``metrics`` in place.
    """
    raw_power_metrics = dict(power_metrics or {})
    raw_runtime_mode = raw_power_metrics.get("runtime_mode")
    if raw_runtime_mode in (None, ""):
        metrics["runtime_mode"] = "back_to_back"
    else:
        normalized_runtime_mode = str(raw_runtime_mode).strip().lower()
        metrics["runtime_mode"] = (
            normalized_runtime_mode
            if normalized_runtime_mode in {"back_to_back", "cadenced"}
            else "back_to_back"
        )
    for field_name, default_value in CADENCED_NUMERIC_FIELD_DEFAULTS.items():
        raw_value = raw_power_metrics.get(field_name, default_value)
        try:
            if isinstance(default_value, int) and not isinstance(default_value, bool):
                metrics[field_name] = int(raw_value)
            else:
                metrics[field_name] = float(raw_value)
        except (TypeError, ValueError):
            metrics[field_name] = default_value
    if metrics["cadenced_error_code"] >= 0:
        metrics["cadenced_error_label"] = describe_error_code(
            metrics["cadenced_error_code"],
            prefer_master=False,
        )
    else:
        metrics["cadenced_error_label"] = None
    for field_name in CADENCED_STRING_FIELDS[1:]:
        raw_value = raw_power_metrics.get(field_name)
        metrics[field_name] = None if raw_value in (None, "") else str(raw_value)


def validate_model_input_shape(
    model: tf.keras.Model,
    input_shape: tuple[int, ...] | None,
) -> None:
    """Validate that a model input shape matches HIL/export expectations.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded or newly built Keras model.
    input_shape : tuple[int, ...] | None
        Expected logical input shape excluding the batch dimension.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the model has an unexpected number of inputs or incompatible input
        dimensions.
    """
    expected_timesteps, expected_input_dim = require_logical_input_shape(input_shape)

    resolved_input_shape = model.input_shape
    if isinstance(resolved_input_shape, list):
        if len(resolved_input_shape) != 1:
            raise ValueError(
                f"Expected a single-input model but checkpoint exposes {len(resolved_input_shape)} inputs."
            )
        resolved_input_shape = resolved_input_shape[0]
    if not isinstance(resolved_input_shape, tuple) or len(resolved_input_shape) != 3:
        raise ValueError(f"Unexpected checkpoint input shape: {resolved_input_shape}")

    actual_timesteps = resolved_input_shape[1]
    actual_input_dim = resolved_input_shape[2]
    if (
        actual_timesteps not in (None, expected_timesteps)
        or actual_input_dim not in (None, expected_input_dim)
    ):
        raise ValueError(
            "Checkpoint input shape mismatch: "
            f"expected (None, {expected_timesteps}, {expected_input_dim}), "
            f"got {resolved_input_shape}."
        )


def validate_loaded_model_input_shape(model: tf.keras.Model, hyperparams: Dict) -> None:
    """Validate that a loaded checkpoint input shape matches HIL expectations.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded Keras model from a checkpoint.
    hyperparams : Dict
        Hyperparameter set that provides expected ``timesteps`` and
        ``input_dim``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the checkpoint has an unexpected number of inputs or incompatible
        input dimensions.
    """
    validate_model_input_shape(
        model,
        (int(hyperparams.timesteps), int(hyperparams.input_dim)),
    )


def iter_layers(model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
    """Flatten model layers across TensorFlow/Keras versions.

    Parameters
    ----------
    model : tf.keras.Model
        Model whose layers should be traversed.

    Returns
    -------
    list[tf.keras.layers.Layer]
        De-duplicated list of all layers reachable from the model graph.
    """
    layers: list[tf.keras.layers.Layer] = []
    seen_nodes: set[int] = set()

    if hasattr(model, "_flatten_layers"):
        try:
            for layer in model._flatten_layers(include_self=False, recursive=True):
                if id(layer) in seen_nodes:
                    continue
                seen_nodes.add(id(layer))
                layers.append(layer)
            if layers:
                return layers
        except TypeError:
            pass

    queue: deque[object] = deque([model])
    while queue:
        node = queue.popleft()
        node_id = id(node)
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        if isinstance(node, tf.keras.layers.Layer):
            layers.append(node)
        for child in getattr(node, "layers", []) or []:
            queue.append(child)
    return layers


def collect_bn_layers(model: tf.keras.Model) -> list[tf.keras.layers.BatchNormalization]:
    """Collect all BatchNormalization layers from a model.

    Parameters
    ----------
    model : tf.keras.Model
        Model to inspect.

    Returns
    -------
    list[tf.keras.layers.BatchNormalization]
        BatchNormalization layers found in the model graph.
    """
    out: list[tf.keras.layers.BatchNormalization] = []
    for layer in iter_layers(model):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            out.append(layer)
    return out


def collect_non_bn_bias_layers(model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
    """Collect non-BN layers that expose a ``bias`` variable.

    Parameters
    ----------
    model : tf.keras.Model
        Model to inspect.

    Returns
    -------
    list[tf.keras.layers.Layer]
        Layers that are not BatchNormalization and have a bias tensor.
    """
    out: list[tf.keras.layers.Layer] = []
    for layer in iter_layers(model):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            continue
        bias = getattr(layer, "bias", None)
        if bias is None:
            continue
        out.append(layer)
    return out


def apply_combined_perturbation(
    model: tf.keras.Model,
    seed: int = 1337,
) -> tuple[int, int]:
    """Apply BN-full + non-BN-bias perturbations.

    Parameters
    ----------
    model : tf.keras.Model
        Model whose BN statistics/affine parameters and non-BN biases are
        perturbed in place.
    seed : int, optional
        Random seed used to make perturbation deterministic.

    Returns
    -------
    tuple[int, int]
        (number of BN layers touched, number of non-BN bias layers touched)
    """
    rng = np.random.default_rng(int(seed))
    bn_touched = 0
    for layer in collect_bn_layers(model):
        if layer.gamma is not None:
            gamma_values = rng.uniform(0.7, 1.3, size=layer.gamma.shape).astype(np.float32)
            layer.gamma.assign(gamma_values)
        if layer.beta is not None:
            beta_values = rng.uniform(-0.3, 0.3, size=layer.beta.shape).astype(np.float32)
            layer.beta.assign(beta_values)
        if layer.moving_mean is not None:
            mean_values = rng.uniform(-0.5, 0.5, size=layer.moving_mean.shape).astype(np.float32)
            layer.moving_mean.assign(mean_values)
        if layer.moving_variance is not None:
            var_values = rng.uniform(0.3, 2.0, size=layer.moving_variance.shape).astype(np.float32)
            layer.moving_variance.assign(var_values)
        bn_touched += 1

    bias_touched = 0
    for layer in collect_non_bn_bias_layers(model):
        bias = getattr(layer, "bias", None)
        if bias is None:
            continue
        bias_values = rng.uniform(-0.3, 0.3, size=bias.shape).astype(np.float32)
        bias.assign(bias_values)
        bias_touched += 1
    return bn_touched, bias_touched


def is_multiobjective_score_config(score_config: Any) -> bool:
    """Return whether a resolved score config uses multi-objective optimization.

    Parameters
    ----------
    score_config : object
        Score configuration object with a ``type`` field.

    Returns
    -------
    bool
        ``True`` when ``score.type == "multi-objective"``.
    """
    score_type = getattr(score_config, "type", None)
    if score_type is None and hasattr(score_config, "get"):
        score_type = score_config.get("type")
    return str(score_type).strip().lower() == "multi-objective"


def get_score_config_directions(score_config: Any) -> list[str]:
    """Return Optuna directions for a resolved score configuration.

    Parameters
    ----------
    score_config : object
        Score configuration tree.

    Returns
    -------
    list[str]
        One direction for scalar mode or one per objective for multi-objective mode.
    """
    if not is_multiobjective_score_config(score_config):
        return ["maximize"]
    return [
        str(objective.direction)
        for objective in getattr(score_config.params, "objectives", [])
    ]


def _resolve_nonnegative_metric_names(
    task_nonnegative_metric_names: set[str] | None = None,
) -> set[str]:
    """Return the effective nonnegative metric set for score evaluation.

    Parameters
    ----------
    task_nonnegative_metric_names : set[str] | None, optional
        Task-declared metric names that are guaranteed to stay nonnegative.
        When omitted, the current odometry RMSE metrics are used.

    Returns
    -------
    set[str]
        Union of infrastructure nonnegative metrics and task-declared
        nonnegative metrics.
    """
    effective_task_nonnegative = set(
        DEFAULT_TASK_SCORE_METRICS
        if task_nonnegative_metric_names is None
        else task_nonnegative_metric_names
    )
    return set(NONNEGATIVE_METRICS) | effective_task_nonnegative


def _metric_unavailable(
    metric_name: str,
    value: Any,
    task_nonnegative_metric_names: set[str] | None = None,
) -> bool:
    """Return whether a metric value is unavailable for scoring.

    Parameters
    ----------
    metric_name : str
        Metric registry name.
    value : object
        Candidate runtime value for the metric.

    Returns
    -------
    bool
        ``True`` when the metric is missing, non-finite, or a negative
        sentinel for a metric that is expected to stay non-negative.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return False
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return True
    if not np.isfinite(numeric_value):
        return True
    return (
        metric_name in _resolve_nonnegative_metric_names(task_nonnegative_metric_names)
        and numeric_value < 0.0
    )


def _normalize_json_safe(value: Any) -> Any:
    """Convert one value into a JSON-safe representation.

    Parameters
    ----------
    value : Any
        Arbitrary runtime value.

    Returns
    -------
    Any
        Recursively normalized JSON-safe value.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {
            str(key): _normalize_json_safe(val)
            for key, val in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_normalize_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def normalize_artifact_summary(artifact_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize task-owned artifact metadata into a JSON-safe mapping.

    Parameters
    ----------
    artifact_summary : dict[str, Any] | None
        Optional raw artifact payload returned by a task.

    Returns
    -------
    dict[str, Any] | None
        Normalized JSON-safe artifact summary.
    """
    if artifact_summary is None:
        return None
    return dict(_normalize_json_safe(artifact_summary))


def build_trial_outcome(
    *,
    score_result: ScoreEvaluationResult,
    task_metrics: dict[str, Any],
    hyperparams: dict[str, Any],
    artifact_summary: dict[str, Any] | None = None,
    quantization_mode: str = "int8_ptq",
) -> TrialOutcome:
    """Assemble a generic trial outcome from score, metric, and artifact data.

    Parameters
    ----------
    score_result : ScoreEvaluationResult
        Evaluated scalar or multi-objective score payload.
    task_metrics : dict[str, Any]
        Task-owned metrics recorded for the trial.
    hyperparams : dict[str, Any]
        Resolved trial hyperparameters.
    artifact_summary : dict[str, Any] | None, optional
        Optional task-owned artifact metadata.
    quantization_mode : str, optional
        Deployment quantization mode used for this trial.

    Returns
    -------
    TrialOutcome
        Normalized generic trial outcome.
    """
    normalized_metrics = dict(_normalize_json_safe(task_metrics))
    normalized_hyperparams = dict(_normalize_json_safe(hyperparams))
    normalized_artifacts = normalize_artifact_summary(artifact_summary)
    return TrialOutcome(
        score=score_result.score,
        objective_names=list(score_result.objective_names),
        objective_values=list(score_result.objective_values),
        objective_directions=list(score_result.objective_directions),
        task_metrics=normalized_metrics,
        hyperparams=normalized_hyperparams,
        artifact_summary=normalized_artifacts,
        quantization_mode=str(quantization_mode),
    )


def quantization_requires_calibration(quantization_mode: str) -> bool:
    """Return whether a deployment mode needs representative calibration data.

    Parameters
    ----------
    quantization_mode : str
        Normalized deployment quantization mode.

    Returns
    -------
    bool
        ``True`` for modes that use representative post-training
        quantization.
    """
    return str(quantization_mode).strip().lower() == "int8_ptq"


def _normalize_quantization_config(
    training: Dict,
    normalized_device_name: str,
) -> Dict:
    """Validate and normalize ``training.quantization``.

    Parameters
    ----------
    training : addict.Dict
        Training subtree from the parsed config.
    normalized_device_name : str
        Uppercase board name used for capability validation.

    Returns
    -------
    addict.Dict
        Normalized quantization subtree with ``mode``, ``search``, and
        ``choices`` fields.

    Raises
    ------
    KeyError
        If the quantization section is missing.
    ValueError
        If the section is legacy boolean/string syntax, malformed, empty, or
        unsupported by the selected board.
    """
    if "quantization" not in training:
        raise KeyError("Expected 'training.quantization' to be set in the configuration.")
    raw_quantization = training.quantization
    if raw_quantization in (None, ""):
        raise ValueError("training.quantization must be a mapping with mode, search, and choices.")
    if isinstance(raw_quantization, bool) or not isinstance(raw_quantization, (dict, Dict)):
        raise ValueError("training.quantization must be a mapping; legacy boolean/string values are not supported.")

    quantization = Dict(raw_quantization)
    raw_mode = quantization.get("mode", None)
    if raw_mode in (None, "") or isinstance(raw_mode, bool):
        raise ValueError("training.quantization.mode must be one of: float, int8_ptq.")
    mode = str(raw_mode).strip().lower()
    if mode not in VALID_QUANTIZATION_MODES:
        raise ValueError("training.quantization.mode must be one of: float, int8_ptq.")

    raw_search = quantization.get("search", False)
    if isinstance(raw_search, bool):
        search = bool(raw_search)
    else:
        raise ValueError("training.quantization.search must be a boolean.")

    raw_choices = quantization.get("choices", None)
    if not isinstance(raw_choices, list) or len(raw_choices) == 0:
        raise ValueError("training.quantization.choices must be a non-empty list.")
    choices: list[str] = []
    for raw_choice in raw_choices:
        if raw_choice in (None, "") or isinstance(raw_choice, bool):
            raise ValueError("training.quantization.choices entries must be one of: float, int8_ptq.")
        choice = str(raw_choice).strip().lower()
        if choice not in VALID_QUANTIZATION_MODES:
            raise ValueError("training.quantization.choices entries must be one of: float, int8_ptq.")
        if choice not in choices:
            choices.append(choice)
    if mode not in choices:
        raise ValueError("training.quantization.mode must be included in training.quantization.choices.")

    supported_modes = BOARD_QUANTIZATION_CAPABILITIES.get(normalized_device_name)
    if supported_modes is not None:
        unsupported = [choice for choice in choices if choice not in supported_modes]
        if unsupported:
            raise ValueError(
                f"Device {normalized_device_name} does not support quantization mode(s): "
                + ", ".join(unsupported)
            )

    return Dict(mode=mode, search=search, choices=choices)


def configured_quantization_mode(config_or_training: Any) -> str:
    """Return the fixed configured quantization mode from a config object.

    Parameters
    ----------
    config_or_training : Any
        Full config object with ``training`` or the training subtree itself.

    Returns
    -------
    str
        Normalized configured quantization mode.

    Raises
    ------
    ValueError
        If the runtime object does not expose the normalized mapping shape.
    """
    training = getattr(config_or_training, "training", config_or_training)
    quantization = getattr(training, "quantization", None)
    if isinstance(quantization, Mapping):
        mode = quantization.get("mode")
    else:
        mode = getattr(quantization, "mode", None)
    if mode in (None, "") or isinstance(mode, bool):
        raise ValueError("training.quantization.mode must be normalized before use.")
    normalized = str(mode).strip().lower()
    if normalized not in VALID_QUANTIZATION_MODES:
        raise ValueError("training.quantization.mode must be one of: float, int8_ptq.")
    return normalized


def _resolve_metric_value(
    metric_name: str,
    context: dict[str, Any],
    score_config: Dict,
    task_nonnegative_metric_names: set[str] | None = None,
    stack: tuple[str, ...] = (),
) -> float:
    """Resolve a metric from the runtime scoring context.

    Parameters
    ----------
    metric_name : str
        Metric registry key to resolve.
    context : dict[str, Any]
        Runtime scoring context populated from metrics, hyperparameters, and
        previously resolved derived metrics.
    score_config : addict.Dict
        Validated score configuration tree.
    task_nonnegative_metric_names : set[str] | None, optional
        Task-declared metric names that are guaranteed to stay nonnegative.
    stack : tuple[str, ...], optional
        Active recursion stack used to detect cycles in derived metrics.

    Returns
    -------
    float
        Resolved metric value.

    Raises
    ------
    ValueError
        If the metric is unavailable, undefined, or participates in a cyclic
        derived-metric graph.
    """
    if metric_name in context:
        value = context[metric_name]
        if _metric_unavailable(metric_name, value, task_nonnegative_metric_names):
            raise ValueError(f"Metric '{metric_name}' is unavailable for scoring.")
        return float(value)

    score_metrics = getattr(score_config, "metrics", Dict())
    if metric_name not in score_metrics:
        raise ValueError(f"Metric '{metric_name}' is not defined in the scoring registry.")
    if metric_name in stack:
        raise ValueError(f"Cycle detected while resolving score metric '{metric_name}'.")

    # `context` doubles as both the input metric namespace and a per-trial
    # memoization cache for derived metrics so repeated references resolve
    # consistently and cheaply.
    metric_config = score_metrics[metric_name]
    try:
        if metric_config.type == "add":
            # Derived metrics intentionally stay simple in v1: resolve each
            # child metric and sum the results.
            resolved = sum(
                _resolve_metric_value(
                    child_name,
                    context,
                    score_config,
                    task_nonnegative_metric_names,
                    stack + (metric_name,),
                )
                for child_name in metric_config.metrics
            )
        elif metric_config.type == "energy-budget-from-power":
            power_mw = _typed_reference_value(
                metric_config.power_mw,
                context,
                score_config,
                task_nonnegative_metric_names,
            )
            duration_ms = _typed_reference_value(
                metric_config.duration_ms,
                context,
                score_config,
                task_nonnegative_metric_names,
            )
            if duration_ms <= 0.0:
                raise ValueError(
                    f"Derived metric '{metric_name}' requires a positive duration reference."
                )
            if power_mw < 0.0:
                raise ValueError(
                    f"Derived metric '{metric_name}' requires a non-negative power reference."
                )
            # Power is expressed in mW and duration in ms, so dividing by 1000
            # yields energy in mJ.
            resolved = (power_mw * duration_ms) / 1000.0
        else:
            raise ValueError(f"Unsupported derived metric type '{metric_config.type}'.")
    except ValueError:
        # Cache the standardized unavailable sentinel so repeated lookups for the
        # same derived metric stay consistent within one trial.
        context[metric_name] = -1.0
        raise
    context[metric_name] = float(resolved)
    return float(resolved)


def _typed_reference_value(
    reference: Dict,
    context: dict[str, Any],
    score_config: Dict,
    task_nonnegative_metric_names: set[str] | None = None,
) -> float:
    """Resolve a typed literal-or-metric reference entry.

    Parameters
    ----------
    reference : addict.Dict
        Typed reference config with ``type`` equal to ``literal`` or ``metric``.
    context : dict[str, Any]
        Runtime scoring context.
    score_config : addict.Dict
        Validated score configuration tree.
    task_nonnegative_metric_names : set[str] | None, optional
        Task-declared metric names that are guaranteed to stay nonnegative.

    Returns
    -------
    float
        Numeric reference value used by scalar score terms that compare or
        normalize against another value.
    """
    if reference.type == "literal":
        return float(reference.value)
    return float(
        _resolve_metric_value(
            reference.metric,
            context,
            score_config,
            task_nonnegative_metric_names,
        )
    )


def _evaluate_score_config(
    metrics: dict[str, Any],
    hyperparams: Dict,
    score_config: Dict,
    task_nonnegative_metric_names: set[str] | None = None,
) -> ScoreEvaluationResult:
    """Evaluate scalar or multi-objective scores from the resolved config.

    Parameters
    ----------
    metrics : dict[str, Any]
        Runtime metrics collected for the trial.
    hyperparams : addict.Dict
        Sampled hyperparameters for the trial.
    score_config : addict.Dict
        Validated score configuration tree.
    task_nonnegative_metric_names : set[str] | None, optional
        Task-declared metric names that are guaranteed to stay nonnegative.

    Returns
    -------
    ScoreEvaluationResult
        Structured scalar or multi-objective score result.

    Raises
    ------
    ScoreConfigEvaluationError
        If any configured metric or derived metric is unavailable, or if a
        normalized reference resolves to a non-positive value.
    """
    context = dict(metrics)
    context["flops"] = hyperparams["flops"]

    def _resolve_score_metric(metric_name: str) -> float:
        """Resolve one configured metric and normalize evaluation errors.

        Parameters
        ----------
        metric_name : str
            Metric name requested by the score config.

        Returns
        -------
        float
            Resolved metric value from the current evaluation context.

        Raises
        ------
        ScoreConfigEvaluationError
            If the metric cannot be resolved for the current context.
        """
        try:
            return _resolve_metric_value(
                metric_name,
                context,
                score_config,
                task_nonnegative_metric_names,
            )
        except ValueError as exc:
            raise ScoreConfigEvaluationError(str(exc)) from exc

    def _resolve_score_reference(reference: Dict) -> float:
        """Resolve a typed score reference and normalize evaluation errors.

        Parameters
        ----------
        reference : Dict
            Typed reference specification from the score config.

        Returns
        -------
        float
            Reference value derived from the current evaluation context.

        Raises
        ------
        ScoreConfigEvaluationError
            If the reference cannot be evaluated for the current context.
        """
        try:
            return _typed_reference_value(
                reference,
                context,
                score_config,
                task_nonnegative_metric_names,
            )
        except ValueError as exc:
            raise ScoreConfigEvaluationError(str(exc)) from exc

    if not is_multiobjective_score_config(score_config):
        score_total = 0.0
        # Scalar scoring uses a small DSL: weighted terms add directly,
        # normalized-weighted terms divide by a positive reference, and
        # boundary/target terms subtract penalties from the maximize score.
        for term in score_config.params.terms:
            metric_value = _resolve_score_metric(term.metric)
            weight = float(term.get("weight", 1.0))
            if term.type == "weighted":
                score_total += weight * metric_value
            elif term.type == "normalized-weighted":
                reference_value = _resolve_score_reference(term.reference)
                if reference_value <= 0.0:
                    raise ScoreConfigEvaluationError(
                        f"Normalized reference for metric '{term.metric}' must be greater than zero."
                    )
                score_total += weight * (metric_value / reference_value)
            elif term.type == "boundary":
                reference_value = _resolve_score_reference(term.reference)
                score_total -= weight * max(0.0, metric_value - reference_value)
            elif term.type == "target":
                reference_value = _resolve_score_reference(term.reference)
                score_total -= weight * abs(metric_value - reference_value)
            else:
                raise ValueError(f"Unsupported scalar score term '{term.type}'.")
        return ScoreEvaluationResult(
            score=float(score_total),
            objective_names=["score"],
            objective_values=[float(score_total)],
            objective_directions=["maximize"],
        )

    objective_names: list[str] = []
    objective_values: list[float] = []
    objective_directions: list[str] = []
    for objective in score_config.params.objectives:
        objective_names.append(str(objective.metric))
        objective_values.append(_resolve_score_metric(objective.metric))
        objective_directions.append(str(objective.direction))
    return ScoreEvaluationResult(
        score=None,
        objective_names=objective_names,
        objective_values=objective_values,
        objective_directions=objective_directions,
    )


def evaluate_score_config(
    *,
    metrics: dict[str, Any],
    hyperparams: Dict,
    score_config: Dict,
    task_nonnegative_metric_names: set[str] | None = None,
) -> ScoreEvaluationResult:
    """Evaluate one validated score configuration against runtime metrics.

    Parameters
    ----------
    metrics : dict[str, Any]
        Runtime metrics collected for the trial.
    hyperparams : Dict
        Resolved hyperparameter payload used for HIL, logging, and scoring.
    score_config : Dict
        Validated score configuration tree.
    task_nonnegative_metric_names : set[str] | None, optional
        Task-declared metric names that are guaranteed to stay nonnegative.

    Returns
    -------
    ScoreEvaluationResult
        Structured scalar or multi-objective score result.
    """
    return _evaluate_score_config(
        metrics=metrics,
        hyperparams=hyperparams,
        score_config=score_config,
        task_nonnegative_metric_names=task_nonnegative_metric_names,
    )


def _validate_typed_reference(
    reference: Any,
    allowed_metrics: set[str],
    context_name: str,
    *,
    allow_unknown_metric_names: bool = False,
) -> Dict:
    """Validate and normalize a typed reference entry.

    Parameters
    ----------
    reference : object
        Raw reference configuration.
    allowed_metrics : set[str]
        Metric names that are valid in the current score configuration.
    context_name : str
        Human-readable label used in validation errors.

    Returns
    -------
    addict.Dict
        Normalized typed reference configuration.

    Raises
    ------
    ValueError
        If the reference shape or values are invalid.
    """
    if not isinstance(reference, (dict, Dict)):
        raise ValueError(f"{context_name} reference must be a mapping.")
    normalized_reference = Dict(reference)
    ref_type = str(normalized_reference.get("type", "")).strip().lower()
    if ref_type not in {"metric", "literal"}:
        raise ValueError(f"{context_name} reference.type must be 'metric' or 'literal'.")
    normalized_reference.type = ref_type
    if ref_type == "metric":
        metric_name = str(normalized_reference.get("metric", "")).strip()
        if metric_name not in allowed_metrics and not allow_unknown_metric_names:
            raise ValueError(f"{context_name} references unknown metric '{metric_name}'.")
        normalized_reference.metric = metric_name
    else:
        try:
            normalized_reference.value = float(normalized_reference["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{context_name} literal reference must define a numeric value.") from exc
    return normalized_reference


def _resolve_validation_metric_sets(
    task_metric_names: set[str] | None = None,
    training_only_task_metric_names: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    """Resolve the metric sets used by task-aware score and prune validation.

    Parameters
    ----------
    task_metric_names : set[str] | None, optional
        Task-owned metric names that may appear in score or prune configs.
        When omitted, task-aware validation is disabled and only
        infrastructure metrics are considered known.
    training_only_task_metric_names : set[str] | None, optional
        Task-owned metric names that are only available after training. When
        omitted, no task metrics are treated as training-only.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(allowed_base_metric_names, effective_training_only_metric_names)``
        after applying defaults and compatibility checks.

    Raises
    ------
    ValueError
        If task metrics overlap infrastructure metrics, if the training-only
        set is supplied without task metrics, or if the training-only set is
        not a subset of the task metric set.
    """
    if task_metric_names is None:
        if training_only_task_metric_names is not None:
            raise ValueError(
                "training_only_task_metric_names requires task_metric_names for task-aware validation."
            )
        effective_task_metric_names = set()
        effective_training_only_metric_names = set()
    else:
        effective_task_metric_names = set(task_metric_names)
        effective_training_only_metric_names = set(
            set() if training_only_task_metric_names is None else training_only_task_metric_names
        )

    overlap = INFRASTRUCTURE_SCORE_METRICS & effective_task_metric_names
    if overlap:
        overlapping_metric = sorted(overlap)[0]
        raise ValueError(
            f"Task metric '{overlapping_metric}' overlaps a reserved infrastructure metric name."
        )

    if not effective_training_only_metric_names <= effective_task_metric_names:
        missing_metric = sorted(
            effective_training_only_metric_names - effective_task_metric_names
        )[0]
        raise ValueError(
            "training_only_task_metric_names must be a subset of task_metric_names; "
            f"'{missing_metric}' is missing from task_metric_names."
        )

    return (
        INFRASTRUCTURE_SCORE_METRICS | effective_task_metric_names,
        effective_training_only_metric_names,
    )


def _metric_depends_on_training(
    metric_name: str,
    score_metrics: Dict,
    training_only_metric_names: set[str] | None = None,
    stack: tuple[str, ...] = (),
) -> bool:
    """Return whether a metric depends on training-only quantities.

    Parameters
    ----------
    metric_name : str
        Metric name to inspect.
    score_metrics : addict.Dict
        Normalized derived metrics from ``nas.score.metrics``.
    training_only_metric_names : set[str] | None, optional
        Task-owned metric names that are only available after training. When
        omitted, the current odometry RMSE set is used.
    stack : tuple[str, ...], optional
        Active recursion stack used to detect cycles.

    Returns
    -------
    bool
        ``True`` when the metric or any derived dependency requires
        post-training values such as RMSE.
    """
    effective_training_only_metric_names = set(
        DEFAULT_TRAINING_ONLY_TASK_METRICS
        if training_only_metric_names is None
        else training_only_metric_names
    )
    if metric_name in effective_training_only_metric_names:
        return True
    if metric_name not in score_metrics or metric_name in stack:
        return False

    metric_cfg = score_metrics[metric_name]
    child_metrics: list[str] = []
    if metric_cfg.type == "add":
        child_metrics = list(metric_cfg.metrics)
    elif metric_cfg.type == "energy-budget-from-power":
        if metric_cfg.power_mw.type == "metric":
            child_metrics.append(str(metric_cfg.power_mw.metric))
        if metric_cfg.duration_ms.type == "metric":
            child_metrics.append(str(metric_cfg.duration_ms.metric))
    return any(
        _metric_depends_on_training(
            child_metric,
            score_metrics,
            effective_training_only_metric_names,
            stack + (metric_name,),
        )
        for child_metric in child_metrics
    )


def score_config_uses_training_metrics(
    score_config: Dict,
    training_only_metric_names: set[str] | None = None,
) -> bool:
    """Return whether a score config depends on post-training RMSE metrics.

    Parameters
    ----------
    score_config : addict.Dict
        Normalized score configuration.
    training_only_metric_names : set[str] | None, optional
        Task-owned metric names that are only available after training. When
        omitted, the current odometry RMSE set is used.

    Returns
    -------
    bool
        ``True`` when any active term, objective, or typed metric reference
        depends directly or indirectly on training-only metrics.
    """
    score_metrics = getattr(score_config, "metrics", Dict())
    effective_training_only_metric_names = set(
        DEFAULT_TRAINING_ONLY_TASK_METRICS
        if training_only_metric_names is None
        else training_only_metric_names
    )

    def _reference_depends_on_training(reference: Any) -> bool:
        """Check whether a typed reference reads a training-derived metric.

        Parameters
        ----------
        reference : Any
            Typed metric reference from the score config.

        Returns
        -------
        bool
            ``True`` when ``reference`` depends on a training-only metric.
        """
        return (
            reference is not None
            and getattr(reference, "type", None) == "metric"
            and _metric_depends_on_training(
                str(reference.metric),
                score_metrics,
                effective_training_only_metric_names,
            )
        )

    if is_multiobjective_score_config(score_config):
        return any(
            _metric_depends_on_training(
                str(objective.metric),
                score_metrics,
                effective_training_only_metric_names,
            )
            for objective in getattr(score_config.params, "objectives", [])
        )

    for term in getattr(score_config.params, "terms", []):
        if _metric_depends_on_training(
            str(term.metric),
            score_metrics,
            effective_training_only_metric_names,
        ):
            return True
        if _reference_depends_on_training(getattr(term, "reference", None)):
            return True
    return False


def _validate_score_config(
    score_input: Any,
    allowed_base_metric_names: set[str],
    has_legacy_multiobjective: bool = False,
    *,
    allow_unknown_metric_names: bool = False,
) -> Dict:
    """Validate and normalize the NAS score configuration.

    Parameters
    ----------
    score_input : object
        Raw ``nas.score`` configuration.
    allowed_base_metric_names : set[str]
        Infrastructure and task-owned metric names that are valid for the
        current validation context.
    has_legacy_multiobjective : bool, optional
        Whether the deprecated ``training.nas_multiobjective`` field is still
        present in the source configuration.

    Returns
    -------
    addict.Dict
        Normalized score configuration.

    Raises
    ------
    KeyError
        If the required score section is missing or the deprecated
        ``training.nas_multiobjective`` field is still present.
    ValueError
        If any score type, metric, term, reference, or objective entry is
        invalid.
    """
    if score_input is None:
        raise KeyError("Missing required 'nas.score' section in the configuration.")
    if has_legacy_multiobjective:
        raise KeyError(
            "training.nas_multiobjective is no longer supported; define nas.score.type instead."
        )

    score_config = Dict(score_input)
    score_type = str(score_config.get("type", "")).strip().lower()
    if score_type not in VALID_SCORE_TYPES:
        raise ValueError("score.type must be one of: scoring-function, multi-objective.")
    score_config.type = score_type
    score_config.metrics = Dict(score_config.get("metrics", {}))
    score_config.params = Dict(score_config.get("params", {}))

    custom_metric_names = set(score_config.metrics.keys())
    duplicate_names = allowed_base_metric_names & custom_metric_names
    if duplicate_names:
        duplicate_name = sorted(duplicate_names)[0]
        raise ValueError(
            "score.metrics may not redefine a built-in or task-declared metric "
            f"'{duplicate_name}'."
        )
    allowed_metric_names = allowed_base_metric_names | custom_metric_names

    # Keep the two metric namespaces distinct: base metrics come from the
    # infrastructure/task contract, while score.metrics introduces only derived
    # metrics layered on top.
    normalized_metrics = Dict()
    for metric_name, raw_metric in score_config.metrics.items():
        metric_cfg = Dict(raw_metric)
        metric_type = str(metric_cfg.get("type", "")).strip().lower()
        if metric_type not in VALID_DERIVED_METRIC_TYPES:
            raise ValueError(
                f"score.metrics.{metric_name}.type must be one of: {sorted(VALID_DERIVED_METRIC_TYPES)}."
            )
        metric_cfg.type = metric_type
        if metric_type == "add":
            metric_list = metric_cfg.get("metrics")
            if not isinstance(metric_list, list) or len(metric_list) == 0:
                raise ValueError(f"score.metrics.{metric_name}.metrics must be a non-empty list.")
            normalized_metric_list = []
            for child_metric in metric_list:
                child_name = str(child_metric).strip()
                if child_name not in allowed_metric_names and not allow_unknown_metric_names:
                    raise ValueError(
                        f"score.metrics.{metric_name} references unknown metric '{child_name}'."
                    )
                normalized_metric_list.append(child_name)
            metric_cfg.metrics = normalized_metric_list
        elif metric_type == "energy-budget-from-power":
            if "power" in metric_cfg and "power_mw" in metric_cfg:
                raise ValueError(
                    f"score.metrics.{metric_name} may define only one of power or power_mw."
                )
            if "duration" in metric_cfg and "duration_ms" in metric_cfg:
                raise ValueError(
                    f"score.metrics.{metric_name} may define only one of duration or duration_ms."
                )

            # Keep the runtime representation explicit about units even when
            # reading older configs that still use the shorter key names.
            metric_cfg.power_mw = _validate_typed_reference(
                metric_cfg.get("power_mw", metric_cfg.get("power")),
                allowed_metric_names,
                f"score.metrics.{metric_name}.power_mw",
                allow_unknown_metric_names=allow_unknown_metric_names,
            )
            metric_cfg.duration_ms = _validate_typed_reference(
                metric_cfg.get("duration_ms", metric_cfg.get("duration")),
                allowed_metric_names,
                f"score.metrics.{metric_name}.duration_ms",
                allow_unknown_metric_names=allow_unknown_metric_names,
            )
            if (
                metric_cfg.duration_ms.type == "literal"
                and float(metric_cfg.duration_ms.value) <= 0.0
            ):
                raise ValueError(
                    f"score.metrics.{metric_name}.duration_ms literal reference must be greater than zero."
                )
            if metric_cfg.power_mw.type == "literal" and float(metric_cfg.power_mw.value) < 0.0:
                raise ValueError(
                    f"score.metrics.{metric_name}.power_mw literal reference must be non-negative."
                )
        normalized_metrics[metric_name] = metric_cfg
    score_config.metrics = normalized_metrics

    if score_type == "scoring-function":
        raw_terms = score_config.params.get("terms")
        if not isinstance(raw_terms, list) or len(raw_terms) == 0:
            raise ValueError("score.params.terms must be a non-empty list for scoring-function mode.")
        normalized_terms = []
        for idx, raw_term in enumerate(raw_terms):
            term = Dict(raw_term)
            term_type = str(term.get("type", "")).strip().lower()
            if term_type not in VALID_TERM_TYPES:
                raise ValueError(
                    f"score.params.terms[{idx}].type must be one of: {sorted(VALID_TERM_TYPES)}."
                )
            metric_name = str(term.get("metric", "")).strip()
            term.type = term_type
            term.metric = metric_name
            term.weight = float(term.get("weight", 1.0))
            if metric_name not in allowed_metric_names and not allow_unknown_metric_names:
                raise ValueError(f"score.params.terms[{idx}] references unknown metric '{metric_name}'.")
            if term_type in {"normalized-weighted", "boundary", "target"}:
                term.reference = _validate_typed_reference(
                    term.get("reference"),
                    allowed_metric_names,
                    f"score.params.terms[{idx}]",
                    allow_unknown_metric_names=allow_unknown_metric_names,
                )
                if (
                    term_type == "normalized-weighted"
                    and term.reference.type == "literal"
                    and float(term.reference.value) <= 0.0
                ):
                    raise ValueError(
                        f"score.params.terms[{idx}] normalized-weighted literal reference must be greater than zero."
                    )
            normalized_terms.append(term)
        score_config.params.terms = normalized_terms
    else:
        raw_objectives = score_config.params.get("objectives")
        if not isinstance(raw_objectives, list) or len(raw_objectives) == 0:
            raise ValueError("score.params.objectives must be a non-empty list for multi-objective mode.")
        normalized_objectives = []
        for idx, raw_objective in enumerate(raw_objectives):
            objective = Dict(raw_objective)
            metric_name = str(objective.get("metric", "")).strip()
            direction = str(objective.get("direction", "")).strip().lower()
            if metric_name not in allowed_metric_names and not allow_unknown_metric_names:
                raise ValueError(f"score.params.objectives[{idx}] references unknown metric '{metric_name}'.")
            if direction not in VALID_OBJECTIVE_DIRECTIONS:
                raise ValueError(
                    f"score.params.objectives[{idx}].direction must be one of: {sorted(VALID_OBJECTIVE_DIRECTIONS)}."
                )
            objective.metric = metric_name
            objective.direction = direction
            normalized_objectives.append(objective)
        score_config.params.objectives = normalized_objectives
    return score_config


def _validate_prune_config(
    prune_input: Any,
    score_config: Dict,
    allowed_base_metric_names: set[str],
    training_only_metric_names: set[str],
    *,
    allow_unknown_metric_names: bool = False,
) -> Dict:
    """Validate and normalize NAS prune policy.

    Parameters
    ----------
    prune_input : object
        Raw ``nas.prune`` configuration.
    score_config : addict.Dict
        Normalized ``nas.score`` configuration.
    allowed_base_metric_names : set[str]
        Infrastructure and task-owned metric names that are valid for the
        current validation context.
    training_only_metric_names : set[str]
        Task-owned metric names that are only available after training.

    Returns
    -------
    addict.Dict
        Normalized prune configuration with a ``rules`` list.

    Raises
    ------
    ValueError
        If any prune rule is invalid or references metrics that are unavailable
        before task training.
    """
    prune_config = Dict(prune_input or {})
    raw_rules = prune_config.get("rules", [])
    if raw_rules is None:
        raw_rules = []
    if not isinstance(raw_rules, list):
        raise ValueError("nas.prune.rules must be a list when provided.")
    allowed_metric_names = allowed_base_metric_names | set(getattr(score_config, "metrics", Dict()).keys())
    normalized_rules = []
    for idx, raw_rule in enumerate(raw_rules):
        rule_cfg = Dict(raw_rule)
        metric_name = str(rule_cfg.get("metric", "")).strip()
        condition = str(rule_cfg.get("condition", "")).strip().lower()
        if metric_name not in allowed_metric_names and not allow_unknown_metric_names:
            raise ValueError(f"nas.prune.rules[{idx}] references unknown metric '{metric_name}'.")
        if condition not in VALID_PRUNE_CONDITIONS:
            raise ValueError(
                f"nas.prune.rules[{idx}].condition must be one of: {sorted(VALID_PRUNE_CONDITIONS)}."
            )
        if _metric_depends_on_training(
            metric_name,
            getattr(score_config, "metrics", Dict()),
            training_only_metric_names,
        ):
            raise ValueError(
                f"nas.prune.rules[{idx}] may not use training-only metric '{metric_name}'."
            )
        reference = _validate_typed_reference(
            rule_cfg.get("reference"),
            allowed_metric_names,
            f"nas.prune.rules[{idx}]",
            allow_unknown_metric_names=allow_unknown_metric_names,
        )
        if reference.type == "metric" and _metric_depends_on_training(
            str(reference.metric),
            getattr(score_config, "metrics", Dict()),
            training_only_metric_names,
        ):
            raise ValueError(
                f"nas.prune.rules[{idx}] may not reference training-only metric '{reference.metric}'."
            )
        rule_cfg.metric = metric_name
        rule_cfg.condition = condition
        rule_cfg.reference = reference
        rule_cfg.reason = str(rule_cfg.get("reason", "")).strip()
        raw_rule_id = str(rule_cfg.get("rule", "")).strip()
        rule_cfg.rule = raw_rule_id if raw_rule_id else f"rule_{idx}"
        normalized_rules.append(rule_cfg)
    prune_config.rules = normalized_rules
    return prune_config


def _validate_feasibility_config(
    feasibility_input: Any,
    score_config: Dict,
    allowed_base_metric_names: set[str],
    training_only_metric_names: set[str],
    *,
    allow_unknown_metric_names: bool = False,
) -> Dict:
    """Validate and normalize NAS feasibility constraints.

    Parameters
    ----------
    feasibility_input : object
        Raw ``nas.feasibility`` configuration.
    score_config : addict.Dict
        Normalized ``nas.score`` configuration.
    allowed_base_metric_names : set[str]
        Infrastructure and task-owned metric names that are valid for the
        current validation context.
    training_only_metric_names : set[str]
        Task-owned metric names that are only available after training.
    allow_unknown_metric_names : bool, optional
        Whether generic config loading should defer unknown task metric checks.

    Returns
    -------
    addict.Dict
        Normalized feasibility configuration with ``train_if_infeasible`` and
        ``rules`` fields.

    Raises
    ------
    ValueError
        If any feasibility rule is invalid or references metrics unavailable
        before task training.
    """
    feasibility_config = Dict(feasibility_input or {})
    raw_train_if_infeasible = feasibility_config.get("train_if_infeasible", False)
    if not isinstance(raw_train_if_infeasible, bool):
        raise ValueError("nas.feasibility.train_if_infeasible must be a boolean.")
    feasibility_config.train_if_infeasible = bool(raw_train_if_infeasible)
    feasibility_config.rules = _validate_prefit_rule_config(
        raw_rules=feasibility_config.get("rules", []),
        rule_path="nas.feasibility.rules",
        score_config=score_config,
        allowed_base_metric_names=allowed_base_metric_names,
        training_only_metric_names=training_only_metric_names,
        allow_unknown_metric_names=allow_unknown_metric_names,
    )
    return feasibility_config


def _validate_prefit_rule_config(
    *,
    raw_rules: Any,
    rule_path: str,
    score_config: Dict,
    allowed_base_metric_names: set[str],
    training_only_metric_names: set[str],
    allow_unknown_metric_names: bool = False,
) -> list[Dict]:
    """Validate and normalize a list of pre-training NAS rules.

    Parameters
    ----------
    raw_rules : Any
        Raw rule list from ``nas.prune.rules`` or ``nas.feasibility.rules``.
    rule_path : str
        Human-readable config path used in error messages.
    score_config : addict.Dict
        Normalized ``nas.score`` configuration.
    allowed_base_metric_names : set[str]
        Infrastructure and task-owned metric names allowed by the active task.
    training_only_metric_names : set[str]
        Task metric names unavailable before training.
    allow_unknown_metric_names : bool, optional
        Whether generic config loading should defer unknown task metric checks.

    Returns
    -------
    list[addict.Dict]
        Normalized rule entries preserving configuration order.

    Raises
    ------
    ValueError
        If rules are malformed or reference metrics unavailable pre-training.
    """
    if raw_rules is None:
        raw_rules = []
    if not isinstance(raw_rules, list):
        raise ValueError(f"{rule_path} must be a list when provided.")
    allowed_metric_names = allowed_base_metric_names | set(getattr(score_config, "metrics", Dict()).keys())
    normalized_rules = []
    for idx, raw_rule in enumerate(raw_rules):
        rule_cfg = Dict(raw_rule)
        metric_name = str(rule_cfg.get("metric", "")).strip()
        condition = str(rule_cfg.get("condition", "")).strip().lower()
        if metric_name not in allowed_metric_names and not allow_unknown_metric_names:
            raise ValueError(f"{rule_path}[{idx}] references unknown metric '{metric_name}'.")
        if condition not in VALID_PRUNE_CONDITIONS:
            raise ValueError(
                f"{rule_path}[{idx}].condition must be one of: {sorted(VALID_PRUNE_CONDITIONS)}."
            )
        if _metric_depends_on_training(
            metric_name,
            getattr(score_config, "metrics", Dict()),
            training_only_metric_names,
        ):
            raise ValueError(f"{rule_path}[{idx}] may not use training-only metric '{metric_name}'.")
        reference = _validate_typed_reference(
            rule_cfg.get("reference"),
            allowed_metric_names,
            f"{rule_path}[{idx}]",
            allow_unknown_metric_names=allow_unknown_metric_names,
        )
        if reference.type == "metric" and _metric_depends_on_training(
            str(reference.metric),
            getattr(score_config, "metrics", Dict()),
            training_only_metric_names,
        ):
            raise ValueError(
                f"{rule_path}[{idx}] may not reference training-only metric '{reference.metric}'."
            )
        rule_cfg.metric = metric_name
        rule_cfg.condition = condition
        rule_cfg.reference = reference
        rule_cfg.reason = str(rule_cfg.get("reason", "")).strip()
        raw_rule_id = str(rule_cfg.get("rule", "")).strip()
        rule_cfg.rule = raw_rule_id if raw_rule_id else f"rule_{idx}"
        normalized_rules.append(rule_cfg)
    return normalized_rules


def _validate_nas_config(
    config: Dict,
    task_metric_names: set[str] | None = None,
    training_only_task_metric_names: set[str] | None = None,
    *,
    allow_unknown_metric_names: bool = False,
) -> Dict:
    """Validate and normalize the top-level NAS policy configuration.

    Parameters
    ----------
    config : addict.Dict
        Parsed YAML configuration tree.
    task_metric_names : set[str] | None, optional
        Task-owned metric names that may appear in score or prune configs.
    training_only_task_metric_names : set[str] | None, optional
        Task-owned metric names that are only available after training.

    Returns
    -------
    addict.Dict
        Normalized ``nas`` subtree containing ``score`` and ``prune``.

    Raises
    ------
    KeyError
        If the NAS policy section is missing or legacy score fields are used.
    """
    if "score" in config:
        raise KeyError("Top-level 'score' is no longer supported; move it to 'nas.score'.")
    if "nas" not in config:
        raise KeyError("Missing required top-level 'nas' section in the configuration.")

    nas_config = Dict(config.nas)
    allowed_base_metric_names, effective_training_only_metric_names = _resolve_validation_metric_sets(
        task_metric_names=task_metric_names,
        training_only_task_metric_names=training_only_task_metric_names,
    )
    score_config = _validate_score_config(
        nas_config.get("score"),
        allowed_base_metric_names=allowed_base_metric_names,
        has_legacy_multiobjective="nas_multiobjective" in config.training,
        allow_unknown_metric_names=allow_unknown_metric_names,
    )
    prune_config = _validate_prune_config(
        nas_config.get("prune", {}),
        score_config,
        allowed_base_metric_names,
        effective_training_only_metric_names,
        allow_unknown_metric_names=allow_unknown_metric_names,
    )
    feasibility_config = _validate_feasibility_config(
        nas_config.get("feasibility", {}),
        score_config,
        allowed_base_metric_names,
        effective_training_only_metric_names,
        allow_unknown_metric_names=allow_unknown_metric_names,
    )
    nas_config.score = score_config
    nas_config.prune = prune_config
    nas_config.feasibility = feasibility_config
    return nas_config


def validate_nas_policy_for_task(
    config: Dict,
    *,
    task_metric_names: set[str],
    training_only_task_metric_names: set[str],
) -> Dict:
    """Validate and attach the NAS policy for one concrete task contract.

    Parameters
    ----------
    config : Dict
        Loaded runtime configuration.
    task_metric_names : set[str]
        Task-owned metric names that may appear in score or prune configs.
    training_only_task_metric_names : set[str]
        Task-owned metric names that are only available after training.

    Returns
    -------
    Dict
        The same config object with its ``nas`` subtree normalized against the
        supplied task metric contract.
    """
    config.nas = _validate_nas_config(
        config,
        task_metric_names=task_metric_names,
        training_only_task_metric_names=training_only_task_metric_names,
    )
    return config


def evaluate_prune_rules(
    metrics: dict[str, Any],
    hyperparams: Dict,
    score_config: Dict,
    prune_config: Dict,
    task_nonnegative_metric_names: set[str] | None = None,
) -> tuple[str, str] | None:
    """Evaluate post-build feasibility rules against the current trial context.

    Parameters
    ----------
    metrics : dict[str, Any]
        Runtime metrics available before training.
    hyperparams : addict.Dict
        Trial hyperparameters.
    score_config : addict.Dict
        Normalized score configuration that owns the derived metric registry.
    prune_config : addict.Dict
        Normalized prune configuration.
    task_nonnegative_metric_names : set[str] | None, optional
        Task-declared metric names that must not use negative unavailable
        sentinels.

    Returns
    -------
    tuple[str, str] | None
        ``(prune_rule, prune_reason)`` for the first matching rule, otherwise
        ``None``.

    Notes
    -----
    Rules are evaluated in configuration order after model build/compile, FLOP
    counting, and HIL/compile metrics are available, but before task fitting or
    validation evaluation. If a configured metric or reference is unavailable
    at gate time, that unavailability is itself treated as a prune outcome for
    the current rule.
    """
    context = dict(metrics)
    context["flops"] = hyperparams["flops"]

    for rule_cfg in getattr(prune_config, "rules", []):
        try:
            metric_value = _resolve_metric_value(
                rule_cfg.metric,
                context,
                score_config,
                task_nonnegative_metric_names,
            )
        except ValueError:
            return rule_cfg.rule, f"Configured prune metric unavailable: {rule_cfg.metric}"

        try:
            reference_value = _typed_reference_value(
                rule_cfg.reference,
                context,
                score_config,
                task_nonnegative_metric_names,
            )
        except ValueError:
            if rule_cfg.reference.type == "metric":
                return (
                    rule_cfg.rule,
                    f"Configured prune reference metric unavailable: {rule_cfg.reference.metric}",
                )
            return rule_cfg.rule, "Configured prune reference unavailable"

        condition_matched = {
            "gt": metric_value > reference_value,
            "gte": metric_value >= reference_value,
            "lt": metric_value < reference_value,
            "lte": metric_value <= reference_value,
        }[rule_cfg.condition]
        if condition_matched:
            reason = (
                rule_cfg.reason
                or f"Prune rule '{rule_cfg.rule}' matched: {rule_cfg.metric} {rule_cfg.condition} {reference_value}"
            )
            return rule_cfg.rule, reason
    return None


def _signed_feasibility_constraint(
    *,
    condition: str,
    metric_value: float,
    reference_value: float,
) -> float:
    """Return an Optuna-compatible signed constraint value.

    Parameters
    ----------
    condition : str
        Feasibility condition from the rule config.
    metric_value : float
        Resolved metric value for the current trial.
    reference_value : float
        Resolved numeric reference value for the current trial.

    Returns
    -------
    float
        Signed constraint where values ``<= 0`` are feasible and values ``> 0``
        are infeasible.

    Notes
    -----
    Inclusive ``gte``/``lte`` rules treat equality as a violation because the
    rule text says the forbidden region includes equality. A tiny positive
    value preserves that distinction without changing ordinary violation
    magnitudes.

    Raises
    ------
    ValueError
        If existing validation or execution checks fail.
    """
    if condition in {"gt", "gte"}:
        signed = metric_value - reference_value
    elif condition in {"lt", "lte"}:
        signed = reference_value - metric_value
    else:
        raise ValueError(f"Unsupported feasibility condition '{condition}'.")
    if condition in {"gte", "lte"} and signed == 0.0:
        return FEASIBILITY_EQUALITY_EPSILON
    return float(signed)


def evaluate_feasibility_rules(
    metrics: dict[str, Any],
    hyperparams: Dict,
    score_config: Dict,
    feasibility_config: Dict,
    task_nonnegative_metric_names: set[str] | None = None,
) -> FeasibilityEvaluation:
    """Evaluate NAS feasibility rules against the current pre-training context.

    Parameters
    ----------
    metrics : dict[str, Any]
        Runtime and compile metrics available before task training.
    hyperparams : addict.Dict
        Trial hyperparameters, including ``flops``.
    score_config : addict.Dict
        Normalized score configuration that owns derived metric definitions.
    feasibility_config : addict.Dict
        Normalized ``nas.feasibility`` policy.
    task_nonnegative_metric_names : set[str] | None, optional
        Task-declared metric names that must not use negative unavailable
        sentinels.

    Returns
    -------
    FeasibilityEvaluation
        Signed constraints plus first-violation metadata.

    Notes
    -----
    Feasibility is intentionally separate from pruning: every rule contributes
    a signed Optuna constraint value, and the first positive value in
    configuration order is the CSV/user-attribute summary.
    """
    context = dict(metrics)
    context["flops"] = hyperparams["flops"]
    constraints: list[float] = []
    first_violation: dict[str, Any] | None = None

    for rule_cfg in getattr(feasibility_config, "rules", []):
        try:
            metric_value = _resolve_metric_value(
                rule_cfg.metric,
                context,
                score_config,
                task_nonnegative_metric_names,
            )
        except ValueError:
            metric_value = 1e12
            reference_value = 0.0
            violation = 1e12
        else:
            try:
                reference_value = _typed_reference_value(
                    rule_cfg.reference,
                    context,
                    score_config,
                    task_nonnegative_metric_names,
                )
            except ValueError:
                reference_value = 0.0
                violation = 1e12
            else:
                violation = _signed_feasibility_constraint(
                    condition=str(rule_cfg.condition),
                    metric_value=float(metric_value),
                    reference_value=float(reference_value),
                )

        constraints.append(float(violation))
        if violation > 0.0 and first_violation is None:
            reason = (
                rule_cfg.reason
                or f"Feasibility rule '{rule_cfg.rule}' matched: {rule_cfg.metric} {rule_cfg.condition} {reference_value}"
            )
            first_violation = {
                "rule": str(rule_cfg.rule),
                "reason": reason,
                "metric": str(rule_cfg.metric),
                "value": float(metric_value),
                "reference": float(reference_value),
                "violation": float(violation),
            }

    if first_violation is None:
        return FeasibilityEvaluation(
            feasible=True,
            status="feasible",
            constraints=constraints,
        )
    return FeasibilityEvaluation(
        feasible=False,
        status="infeasible",
        constraints=constraints,
        first_violation=first_violation,
    )


def load_config(
    config_path: str | Path | None = None,
    *,
    task_metric_names: set[str] | None = None,
    training_only_task_metric_names: set[str] | None = None,
) -> Dict:
    """Load the NAS configuration from YAML, derive convenience paths/names,.

    and return an addict.Dict for dot-attribute access.

    Parameters
    ----------
    config_path : str | Path | None
        Optional override for the YAML location. Defaults to
        ``src/config/nas_config_stm32.yaml``.
    task_metric_names : set[str] | None, optional
        Task-owned metric names that may appear in score or prune configs. When
        omitted, generic NAS-policy validation may preserve metric names that
        are unknown until a task contract is available.
    training_only_task_metric_names : set[str] | None, optional
        Task-owned metric names that are only available after training. When
        omitted, task-aware validation treats no task metrics as training-only.

    Returns
    -------
    addict.Dict
        Configuration tree with derived paths (models_dir, checkpoint paths,
        etc.) and a generically validated NAS policy.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    KeyError
        If required config sections or mandatory fields are missing.
    ValueError
        If runtime-mode, timing, harness, CPU-clock, or NAS policy fields are
        malformed.

    Notes
    -----
    Besides parsing YAML, this helper normalizes derived artifact paths,
    validates board/runtime policy, and injects harness defaults. It always
    performs the generic NAS score/prune validation pass, preserving potential
    task-owned metric names when no task context exists. When
    ``task_metric_names`` are supplied, it rejects truly unknown metrics and
    validates the NAS policy against that concrete task contract instead of
    deferring task-owned metric checks to the later bootstrap step.
    """
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    config = Dict(yaml.safe_load(cfg_path.read_text()))

    # Validate required sections early so callers get actionable errors.
    for section in ("device", "outputs", "training"):
        if section not in config:
            raise KeyError(f"Missing '{section}' section in {cfg_path}")

    device_name = config.device.get("name")
    if not device_name:
        raise KeyError("Expected 'device.name' to be set in the configuration.")
    normalized_device_name = str(device_name).strip().upper()

    # Derive paths and names for models/checkpoints based on device name.
    outputs = config.outputs
    models_dir = Path(outputs.get("models_dir", "../models")).resolve()
    if "tcn_dir" in outputs:
        raise ValueError(
            "outputs.tcn_dir has been removed. Use outputs.candidate_dir instead."
        )
    if "model_name" in outputs or "checkpoint_name" in outputs:
        raise ValueError(
            "outputs.model_name and outputs.checkpoint_name are derived runtime fields. "
            "Use outputs.artifact_stem instead."
        )
    if "candidate_dir" not in outputs:
        raise ValueError("Expected 'outputs.candidate_dir' to be set in the configuration.")
    raw_artifact_stem = outputs.get("artifact_stem", None)
    if raw_artifact_stem is None or not isinstance(raw_artifact_stem, str):
        raise ValueError("Expected 'outputs.artifact_stem' to be a non-empty string.")
    artifact_stem = raw_artifact_stem.strip()
    if not artifact_stem or "/" in artifact_stem or "\\" in artifact_stem:
        raise ValueError(
            "Expected 'outputs.artifact_stem' to be a non-empty filename stem without path separators."
        )
    candidate_dir = Path(outputs.candidate_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    # Stores derived model/checkpoint names and paths into the config
    model_stem = f"{artifact_stem}_{device_name}"
    outputs.model_name = f"{model_stem}.tflite"
    outputs.checkpoint_name = f"{model_stem}.keras"
    outputs.models_dir = models_dir
    outputs.candidate_dir = candidate_dir
    outputs.tflite_model_path = models_dir / outputs.model_name
    outputs.checkpoint_path = models_dir / outputs.checkpoint_name

    # Populate training choices from constants
    training = config.training
    if "nas_trials" not in training:
        raise KeyError("Expected 'training.nas_trials' to be set in the configuration.")
    if "max_total_trials" not in training:
        # Allow pruned/failed runs without risking an infinite loop.
        training.max_total_trials = int(training.nas_trials * 2)
    # If not explicitly set, disable training by default for faster debugging.
    training.energy_aware = bool(training.get("energy_aware", False))
    training.quantization = _normalize_quantization_config(
        training,
        normalized_device_name,
    )
    # Input mode selects which Arduino sketch variant is used during HIL runs.
    training.input_mode = str(training.get("input_mode", "uniform")).lower()
    config.training.drop_rate_choices = DROP_RATE_CHOICES

    device = config.device

    raw_compile_when_hil_disabled = device.get("compile_when_hil_disabled", "auto")
    if isinstance(raw_compile_when_hil_disabled, bool):
        device.compile_when_hil_disabled = "true" if raw_compile_when_hil_disabled else "false"
    else:
        compile_when_hil_disabled = str(raw_compile_when_hil_disabled).strip().lower()
        if compile_when_hil_disabled not in VALID_COMPILE_WHEN_HIL_DISABLED:
            raise ValueError("device.compile_when_hil_disabled must be one of: auto, true, false.")
        device.compile_when_hil_disabled = compile_when_hil_disabled

    def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
        """Read a value from mapping-like or namespace-like config objects.

        Parameters
        ----------
        container : Any
            Config subtree or namespace to inspect.
        key : str
            Field name to resolve.
        default : Any, optional
            Fallback value when the field is missing.

        Returns
        -------
        Any
            Resolved field value or ``default``.
        """
        getter = getattr(container, "get", None)
        if callable(getter):
            return getter(key, default)
        return getattr(container, key, default)

    runtime_mode = str(device.get("runtime_mode", "back_to_back")).strip().lower()
    if runtime_mode not in {"back_to_back", "cadenced"}:
        raise ValueError("device.runtime_mode must be one of: back_to_back, cadenced.")
    device.runtime_mode = runtime_mode

    latency_budget_override_raw = device.get("latency_budget_ms", None)
    if latency_budget_override_raw in (None, ""):
        device.latency_budget_ms = None
    else:
        if isinstance(latency_budget_override_raw, bool):
            raise ValueError("device.latency_budget_ms must be a positive number or null.")
        try:
            device.latency_budget_ms = float(latency_budget_override_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("device.latency_budget_ms must be a positive number or null.") from exc
        if device.latency_budget_ms <= 0.0:
            raise ValueError("device.latency_budget_ms must be a positive number or null.")

    if normalized_device_name == "STM32_NUCLEO_N657X0_Q":
        stm32_cfg = device.get("stm32", None)
        if _cfg_get(stm32_cfg, "runtime_mode", None) not in (None, ""):
            raise ValueError(
                "device.stm32.runtime_mode is no longer supported. Use device.runtime_mode instead."
            )

    if (
        normalized_device_name in {"PORTENTA_H7", "ARDUINO_NANO_33_BLE_SENSE"}
        and device.runtime_mode == "cadenced"
        and training.input_mode != "uniform"
    ):
        raise ValueError(
            "Cadenced runtime for Portenta H7 and Arduino Nano 33 BLE Sense only supports "
            "training.input_mode='uniform'."
        )

    if normalized_device_name == "ARDUINO_NANO_RP2040_CONNECT":
        # This target is latency/memory-only: no harness, INA228 or cadenced pass.
        if device.runtime_mode == "cadenced":
            raise ValueError("ARDUINO_NANO_RP2040_CONNECT only supports device.runtime_mode='back_to_back'.")
        if bool(training.get("energy_aware", False)):
            raise ValueError(
                "ARDUINO_NANO_RP2040_CONNECT is latency/memory-only; set training.energy_aware: false."
            )
        if device.get("harness_serial_port", None) not in (None, ""):
            raise ValueError(
                "ARDUINO_NANO_RP2040_CONNECT has no measurement harness; remove device.harness_serial_port."
            )

    measured_inference_runs_raw = device.get("measured_inference_runs", 10)
    if isinstance(measured_inference_runs_raw, bool):
        raise ValueError("device.measured_inference_runs must be an integer >= 1.")
    if isinstance(measured_inference_runs_raw, float) and not measured_inference_runs_raw.is_integer():
        raise ValueError("device.measured_inference_runs must be an integer >= 1.")
    try:
        device.measured_inference_runs = int(measured_inference_runs_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("device.measured_inference_runs must be an integer >= 1.") from exc
    if device.measured_inference_runs < 1:
        raise ValueError("device.measured_inference_runs must be >= 1.")

    cpu_clock_mhz_options_raw = device.get("cpu_clock_mhz_options", None)
    if cpu_clock_mhz_options_raw is None:
        device.cpu_clock_mhz_options = None
    else:
        if not isinstance(cpu_clock_mhz_options_raw, list) or len(cpu_clock_mhz_options_raw) == 0:
            raise ValueError("device.cpu_clock_mhz_options must be a non-empty list of integers or null.")
        normalized_cpu_clock_mhz_options = []
        for value in cpu_clock_mhz_options_raw:
            if isinstance(value, bool):
                raise ValueError("device.cpu_clock_mhz_options entries must be integers, not booleans.")
            if isinstance(value, float) and not value.is_integer():
                raise ValueError("device.cpu_clock_mhz_options entries must be integers.")
            try:
                normalized_value = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("device.cpu_clock_mhz_options entries must be integers.") from exc
            normalized_cpu_clock_mhz_options.append(normalized_value)
        if normalized_device_name == "STM32_NUCLEO_N657X0_Q":
            from .microcontrollers.stm32_nucleo_n657x0 import SUPPORTED_CPU_CLOCK_MHZ

            allowed_cpu_clock_mhz_options = sorted(SUPPORTED_CPU_CLOCK_MHZ)
            for normalized_value in normalized_cpu_clock_mhz_options:
                if normalized_value not in SUPPORTED_CPU_CLOCK_MHZ:
                    raise ValueError(
                        "device.cpu_clock_mhz_options entries for STM32_NUCLEO_N657X0_Q "
                        f"must be one of: {allowed_cpu_clock_mhz_options}."
                    )
        device.cpu_clock_mhz_options = normalized_cpu_clock_mhz_options

    device.harness_fqbn = device.get("harness_fqbn", "arduino:mbed_nano:nano33ble")
    device.harness_auto_flash = str(device.get("harness_auto_flash", "once")).lower()
    device.harness_arm_pin = int(device.get("harness_arm_pin", 3))
    device.harness_trigger_pin = int(device.get("harness_trigger_pin", 2))
    device.dut_arm_hold_ms = int(device.get("dut_arm_hold_ms", 600))
    device.harness_stable_low_ms = int(device.get("harness_stable_low_ms", 500))
    device.harness_ready_timeout_s = float(device.get("harness_ready_timeout_s", 5.0))
    device.harness_arm_timeout_s = float(device.get("harness_arm_timeout_s", 5.0))
    device.harness_active_timeout_s = float(device.get("harness_active_timeout_s", 30.0))
    device.harness_done_timeout_s = float(device.get("harness_done_timeout_s", 5.0))
    device.dut_ready_timeout_s = float(device.get("dut_ready_timeout_s", 5.0))

    if device.harness_arm_pin <= 0:
        raise ValueError("device.harness_arm_pin must be a positive integer.")
    if device.harness_trigger_pin <= 0:
        raise ValueError("device.harness_trigger_pin must be a positive integer.")
    if device.dut_arm_hold_ms <= 0:
        raise ValueError("device.dut_arm_hold_ms must be a positive integer.")
    if device.harness_stable_low_ms <= 0:
        raise ValueError("device.harness_stable_low_ms must be a positive integer.")
    if device.dut_arm_hold_ms <= device.harness_stable_low_ms:
        raise ValueError(
            "device.dut_arm_hold_ms must be greater than device.harness_stable_low_ms."
        )
    if device.harness_auto_flash not in {"once", "always", "never"}:
        raise ValueError(
            "device.harness_auto_flash must be one of: once, always, never."
        )

    # Logging level for runtime observability (used by hil_server.py and scripts).
    logging_section = Dict(config.get("logging", {}))
    level_name = str(logging_section.get("level", "INFO")).upper()
    valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
    if level_name not in valid_levels:
        raise ValueError(
            "logging.level must be one of: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET."
        )
    config.logging = Dict()
    config.logging.level = level_name
    if "score" in config:
        raise KeyError("Top-level 'score' is no longer supported; move it to 'nas.score'.")
    if "nas" not in config:
        raise KeyError("Missing required top-level 'nas' section in the configuration.")
    config.nas = Dict(config.nas)
    has_task_context = (
        task_metric_names is not None
        or training_only_task_metric_names is not None
    )
    config.nas = _validate_nas_config(
        config,
        task_metric_names=task_metric_names,
        training_only_task_metric_names=training_only_task_metric_names,
        allow_unknown_metric_names=not has_task_context,
    )
    _validate_evaluation_config(config)

    return config


def _validate_evaluation_config(config: Dict) -> None:
    """Validate and normalize task-owned final evaluation/reporting settings.

    Parameters
    ----------
    config : addict.Dict
        Loaded configuration tree.

    Returns
    -------
    None
        Mutates ``config.task.params.evaluation`` with normalized defaults or
        raises.

    Raises
    ------
    KeyError
        If existing validation or execution checks fail.
    ValueError
        If existing validation or execution checks fail.
    """
    if "evaluation" in config:
        raise KeyError(
            "Top-level 'evaluation' is no longer supported; move it to "
            "'task.params.evaluation'."
        )

    task = Dict(config.get("task", {}))
    task_params = Dict(task.get("params", {}))
    evaluation = Dict(task_params.get("evaluation", {}))
    protocol = str(evaluation.get("protocol", "fixed_split")).strip().lower()
    if protocol not in {"fixed_split", "fold_rotation"}:
        raise ValueError("task.params.evaluation.protocol must be one of: fixed_split, fold_rotation.")
    fold_rotation = Dict(evaluation.get("fold_rotation", {}))
    raw_test_folds = fold_rotation.get("test_folds", list(range(1, 11)))
    if not isinstance(raw_test_folds, list) or not raw_test_folds:
        raise ValueError("task.params.evaluation.fold_rotation.test_folds must be a non-empty list.")
    test_folds: list[int] = []
    for raw_fold in raw_test_folds:
        if isinstance(raw_fold, bool):
            raise ValueError(
                "task.params.evaluation.fold_rotation.test_folds entries must be integers from 1 to 10."
            )
        try:
            fold = int(raw_fold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "task.params.evaluation.fold_rotation.test_folds entries must be integers from 1 to 10."
            ) from exc
        if fold < 1 or fold > 10:
            raise ValueError(
                "task.params.evaluation.fold_rotation.test_folds entries must be integers from 1 to 10."
            )
        if fold in test_folds:
            raise ValueError("task.params.evaluation.fold_rotation.test_folds entries must be unique.")
        test_folds.append(fold)
    fold_rotation.test_folds = test_folds
    evaluation.protocol = protocol
    evaluation.fold_rotation = fold_rotation
    task_params.evaluation = evaluation
    task.params = task_params
    config.task = task

    if protocol != "fold_rotation":
        return
    if is_multiobjective_score_config(config.nas.score):
        raise ValueError("task.params.evaluation.protocol=fold_rotation is only supported for single-objective NAS.")
    dataset_params = Dict(config.get("dataset", {}).get("params", {}))
    if dataset_params.get("cache_dir", None) in (None, ""):
        raise ValueError(
            "task.params.evaluation.protocol=fold_rotation requires dataset.params.cache_dir "
            "for fixed-split NAS."
        )
    if dataset_params.get("fold_rotation_cache_dir", None) in (None, ""):
        raise ValueError(
            "task.params.evaluation.protocol=fold_rotation requires dataset.params.fold_rotation_cache_dir."
        )


def count_flops(model, input_shape):
    """Estimate model FLOPs by profiling a frozen forward graph.

    Parameters
    ----------
    model : tf.keras.Model
        Keras model with defined input signatures.
    input_shape : tuple[int]
        Input tensor shape excluding the batch dimension.

    Returns
    -------
    int
        Static graph FLOP proxy for a single forward pass with batch size 1.

    Notes
    -----
    Compatibility wrapper around :func:`crest.model_metrics.count_flops_keras`.
    """
    return count_flops_keras(model, tuple(int(dim) for dim in input_shape))


def require_logical_input_shape(input_shape: Any) -> tuple[int, int]:
    """Resolve one concrete 2D logical input shape from build-context metadata.

    Parameters
    ----------
    input_shape : Any
        Logical input shape excluding the batch dimension.

    Returns
    -------
    tuple[int, int]
        Concrete ``(timesteps, input_dim)`` pair.

    Raises
    ------
    ValueError
        If the shape is missing, not 2D, non-numeric, boolean, or contains
        non-positive dimensions.
    """
    error_message = (
        "The active model build context must expose a 2D logical input shape "
        "with positive integer timesteps and input_dim."
    )
    if input_shape is None:
        raise ValueError(error_message)
    try:
        if len(input_shape) != 2:
            raise ValueError(error_message)
    except TypeError as exc:
        raise ValueError(error_message) from exc

    resolved_dims: list[int] = []
    for raw_dim in (input_shape[0], input_shape[1]):
        if raw_dim is None or isinstance(raw_dim, bool):
            raise ValueError(error_message)
        try:
            resolved_dim = int(raw_dim)
        except (TypeError, ValueError) as exc:
            raise ValueError(error_message) from exc
        if resolved_dim <= 0:
            raise ValueError(error_message)
        resolved_dims.append(resolved_dim)
    return resolved_dims[0], resolved_dims[1]

def _serialize_csv_cell(value: Any) -> Any:
    """Return a CSV-friendly representation for one cell value.

    Parameters
    ----------
    value : Any
        Raw cell value.

    Returns
    -------
    Any
        Primitive value or JSON string suitable for CSV writing.
    """
    normalized = _normalize_json_safe(value)
    if normalized is None:
        return ""
    if isinstance(normalized, (str, int, float, bool)):
        return normalized
    return json.dumps(normalized, sort_keys=True)


def _trial_log_row_mapping(
    *,
    trial_outcome: TrialOutcome,
    metrics: dict[str, Any],
    study_name: str,
    pruned: bool,
    prune_reason: str,
    prune_rule: str,
) -> dict[str, Any]:
    """Build one row mapping for the shared trial CSV.

    Parameters
    ----------
    trial_outcome : TrialOutcome
        Generic trial outcome to log.
    metrics : dict[str, Any]
        Shared infrastructure metrics dictionary.
    study_name : str
        Name of the active study.
    pruned : bool
        Whether the trial was pruned.
    prune_reason : str
        Human-readable pruning reason.
    prune_rule : str
        Stable pruning rule identifier.

    Returns
    -------
    dict[str, Any]
        Flat row mapping keyed by CSV column name.
    """
    score_type = metrics.get("score_type", "multi-objective" if trial_outcome.score is None else "scoring-function")
    mapping: dict[str, Any] = {
        "study_name": study_name,
        "timestamp_unix": time.time(),
        "timestamp_readable": time.strftime("%m-%d-%Y %H:%M:%S"),
        "score": "" if trial_outcome.score is None else trial_outcome.score,
        "quantization_mode": trial_outcome.quantization_mode,
        "ram_bytes": metrics["ram_bytes"],
        "flash_bytes": metrics["flash_bytes"],
        "external_flash_bytes": metrics.get("external_flash_bytes", -1),
        "weight_storage_mode": metrics.get("weight_storage_mode", "embedded"),
        "flops": trial_outcome.hyperparams["flops"],
        "weight_bytes": metrics.get("weight_bytes", trial_outcome.hyperparams.get("weight_bytes", -1)),
        "activation_bytes": metrics.get("activation_bytes", trial_outcome.hyperparams.get("activation_bytes", -1)),
        "memory_traffic_bytes": metrics.get(
            "memory_traffic_bytes",
            trial_outcome.hyperparams.get("memory_traffic_bytes", -1),
        ),
        "memory_proxy_dtype_bytes": metrics.get(
            "memory_proxy_dtype_bytes",
            trial_outcome.hyperparams.get("memory_proxy_dtype_bytes", -1),
        ),
        "memory_proxy_warning_count": metrics.get(
            "memory_proxy_warning_count",
            trial_outcome.hyperparams.get("memory_proxy_warning_count", -1),
        ),
        "latency_ms": metrics["latency_ms"],
        "latency_budget_ms": metrics.get("latency_budget_ms", -1.0),
        "energy_mj_per_inference": metrics["energy_mj_per_inference"],
        "avg_power_mw": metrics["avg_power_mw"],
        "avg_current_ma": metrics["avg_current_ma"],
        "bus_voltage_v": metrics["bus_voltage_v"],
        "cpu_clock_mhz_requested": metrics.get("cpu_clock_mhz_requested", -1),
        "clock_hz": metrics.get("clock_hz", -1.0),
        "error_code": metrics["error_code"],
        "error_label": metrics.get("error_label", describe_error_code(metrics["error_code"])),
        "score_type": score_type,
        "objective_names_json": json.dumps(trial_outcome.objective_names),
        "objective_values_json": json.dumps(trial_outcome.objective_values),
        "trial_outcome": metrics.get("trial_outcome", ""),
        "objective_directions_json": json.dumps(trial_outcome.objective_directions),
        "artifact_summary_json": json.dumps(trial_outcome.artifact_summary, sort_keys=True),
        "pruned": pruned,
        "prune_reason": prune_reason,
        "prune_rule": prune_rule,
        "feasible": metrics.get("feasible", not pruned),
        "feasibility_status": metrics.get(
            "feasibility_status",
            "not_evaluated" if pruned else "feasible",
        ),
        "feasibility_rule": metrics.get("feasibility_rule", ""),
        "feasibility_reason": metrics.get("feasibility_reason", ""),
        "feasibility_metric": metrics.get("feasibility_metric", ""),
        "feasibility_value": metrics.get("feasibility_value", ""),
        "feasibility_reference": metrics.get("feasibility_reference", ""),
        "feasibility_violation": metrics.get("feasibility_violation", ""),
        "feasibility_constraints_json": metrics.get("feasibility_constraints_json", ""),
    }
    for field_name in CADENCED_CSV_FIELDS:
        mapping[field_name] = metrics.get(field_name)
    for metric_name, value in trial_outcome.task_metrics.items():
        mapping[f"metric__{metric_name}"] = value
    for hyperparam_name, value in trial_outcome.hyperparams.items():
        if hyperparam_name in {
            "flops",
            "weight_bytes",
            "activation_bytes",
            "memory_traffic_bytes",
            "memory_proxy_dtype_bytes",
            "memory_proxy_warning_count",
        }:
            continue
        mapping[f"hparam__{hyperparam_name}"] = value
    return mapping


def _read_trial_log(log_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read one existing trial log file.

    Parameters
    ----------
    log_path : Path
        CSV log path.

    Returns
    -------
    tuple[list[str], list[dict[str, str]]]
        Existing header plus all row dictionaries.
    """
    with log_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader.fieldnames or []), list(reader)


def _write_trial_log_atomic(
    log_path: Path,
    header: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """Rewrite one trial log file atomically.

    Parameters
    ----------
    log_path : Path
        Destination CSV path.
    header : list[str]
        Full header to write.
    rows : list[dict[str, Any]]
        Row mappings to serialize.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If existing validation or execution checks fail.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        delete=False,
        dir=log_path.parent,
        prefix=f"{log_path.name}.",
        suffix=".tmp",
    ) as tmpfile:
        writer = csv.DictWriter(tmpfile, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _serialize_csv_cell(row.get(column)) for column in header})
        temp_path = Path(tmpfile.name)
    try:
        os.replace(temp_path, log_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _write_trial_log_append(log_path: Path, header: list[str], row: dict[str, Any]) -> None:
    """Append one row to a trial log whose header already matches.

    Parameters
    ----------
    log_path : Path
        Destination CSV path.
    header : list[str]
        Existing CSV header.
    row : dict[str, Any]
        Row mapping to append.

    Returns
    -------
    None
    """
    with log_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header, extrasaction="ignore")
        writer.writerow({column: _serialize_csv_cell(row.get(column)) for column in header})


def _record_measurement_evidence(metrics: dict[str, Any], trial: TrialLike) -> dict[str, Any]:
    """Preserve raw-record links and acceptance reasons in trial attrs and CSV.

    Parameters
    ----------
    metrics : dict[str, Any]
        Metrics whose measurement-prefixed entries contain acceptance evidence.
    trial : TrialLike
        Trial receiving JSON-safe measurement user attributes.

    Returns
    -------
    dict[str, Any]
        CSV cells keyed by metric-prefixed column names, with lists and mappings
        serialized as JSON strings.
    """
    columns = {}
    for name, value in metrics.items():
        if name.startswith("measurement_"):
            normalized = _normalize_json_safe(value)
            trial.set_user_attr(name, normalized)
            columns[f"metric__{name}"] = (
                json.dumps(normalized, sort_keys=True)
                if isinstance(normalized, (dict, list)) else normalized
            )
    return columns


def log_trial(
    trial_outcome: TrialOutcome,
    metrics: dict,
    trial: TrialLike,
    log_file_name: str,
    study_name: str = "",
    pruned: bool = False,
    prune_reason: str = "",
    prune_rule: str = "",
):
    """Write one trial summary row to CSV and annotate the Optuna trial.

    Parameters
    ----------
    trial_outcome : TrialOutcome
        Generic trial outcome to write.
    metrics : dict
        Resource metrics dict.
    trial : TrialLike
        Trial-like object (e.g., optuna.Trial) to annotate.
    log_file_name : str
        Path to the CSV log file.
    study_name : str, optional
        Name of the Optuna study.
    pruned : bool, optional
        Whether the trial was pruned.
    prune_reason : str, optional
        Reason for pruning.
    prune_rule : str, optional
        Stable machine-readable pruning rule identifier.

    Returns
    -------
    None
        Writes one CSV row and mutates ``trial`` user attributes in place.

    Notes
    -----
    The CSV schema keeps a stable leading column set, then expands
    ``metric__*`` and ``hparam__*`` columns as needed while preserving older
    rows through lock-protected header migration.
    """
    log_path = Path(log_file_name)
    row_mapping = _trial_log_row_mapping(
        trial_outcome=trial_outcome,
        metrics=metrics,
        study_name=study_name,
        pruned=pruned,
        prune_reason=prune_reason,
        prune_rule=prune_rule,
    )
    row_mapping.update(_record_measurement_evidence(metrics, trial))
    metric_columns = sorted(
        column_name
        for column_name in row_mapping
        if column_name.startswith("metric__")
    )
    hyperparam_columns = sorted(
        column_name
        for column_name in row_mapping
        if column_name.startswith("hparam__")
    )
    lock_path = log_path.with_name(f"{log_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if not log_path.exists() or log_path.stat().st_size == 0:
            header = [
                *TRIAL_LOG_STABLE_COLUMNS,
                *metric_columns,
                *hyperparam_columns,
            ]
            _write_trial_log_atomic(log_path, header, [row_mapping])
        else:
            existing_header, existing_rows = _read_trial_log(log_path)
            existing_metric_columns = sorted(
                column_name
                for column_name in existing_header
                if column_name.startswith("metric__")
            )
            existing_hparam_columns = sorted(
                column_name
                for column_name in existing_header
                if column_name.startswith("hparam__")
            )
            legacy_columns = [
                column_name
                for column_name in existing_header
                if (
                    column_name not in TRIAL_LOG_STABLE_COLUMNS
                    and not column_name.startswith("metric__")
                    and not column_name.startswith("hparam__")
                )
            ]
            # Preserve the stable core schema first, retain any legacy unknown
            # columns already on disk, then union the dynamic metric/hparam
            # columns so older rows can be rewritten with backfilled blanks.
            header = [
                *TRIAL_LOG_STABLE_COLUMNS,
                *legacy_columns,
                *sorted(set(existing_metric_columns) | set(metric_columns)),
                *sorted(set(existing_hparam_columns) | set(hyperparam_columns)),
            ]
            if existing_header == header:
                _write_trial_log_append(log_path, header, row_mapping)
            else:
                _write_trial_log_atomic(
                    log_path,
                    header,
                    [*existing_rows, row_mapping],
                )

    trial.set_user_attr("ram_bytes", metrics["ram_bytes"])
    trial.set_user_attr("quantization_mode", trial_outcome.quantization_mode)
    trial.set_user_attr("flash_bytes", metrics["flash_bytes"])
    trial.set_user_attr("external_flash_bytes", metrics.get("external_flash_bytes", -1))
    trial.set_user_attr("weight_storage_mode", metrics.get("weight_storage_mode", "embedded"))
    trial.set_user_attr("latency_ms", metrics["latency_ms"])
    trial.set_user_attr("latency_budget_ms", metrics.get("latency_budget_ms", -1.0))
    trial.set_user_attr("energy_mj_per_inference", metrics["energy_mj_per_inference"])
    trial.set_user_attr("cpu_clock_mhz_requested", metrics.get("cpu_clock_mhz_requested", -1))
    trial.set_user_attr("clock_hz", metrics.get("clock_hz", -1.0))
    trial.set_user_attr("hil_error_code", metrics["error_code"])
    trial.set_user_attr("arena_bytes", metrics["arena_bytes"])
    trial.set_user_attr("flops", trial_outcome.hyperparams["flops"])
    trial.set_user_attr("weight_bytes", metrics.get("weight_bytes", trial_outcome.hyperparams.get("weight_bytes", -1)))
    trial.set_user_attr(
        "activation_bytes",
        metrics.get("activation_bytes", trial_outcome.hyperparams.get("activation_bytes", -1)),
    )
    trial.set_user_attr(
        "memory_traffic_bytes",
        metrics.get("memory_traffic_bytes", trial_outcome.hyperparams.get("memory_traffic_bytes", -1)),
    )
    trial.set_user_attr(
        "memory_proxy_dtype_bytes",
        metrics.get(
            "memory_proxy_dtype_bytes",
            trial_outcome.hyperparams.get("memory_proxy_dtype_bytes", -1),
        ),
    )
    trial.set_user_attr(
        "memory_proxy_warning_count",
        metrics.get(
            "memory_proxy_warning_count",
            trial_outcome.hyperparams.get("memory_proxy_warning_count", -1),
        ),
    )
    trial.set_user_attr("error_code", metrics["error_code"])
    trial.set_user_attr(
        "score_type",
        metrics.get("score_type", "multi-objective" if trial_outcome.score is None else "scoring-function"),
    )
    if metrics.get("trial_outcome"):
        trial.set_user_attr("trial_outcome", metrics["trial_outcome"])
    trial.set_user_attr("objective_names", list(trial_outcome.objective_names))
    trial.set_user_attr("objective_values", list(trial_outcome.objective_values))
    trial.set_user_attr("objective_directions", list(trial_outcome.objective_directions))
    error_label = metrics.get("error_label", describe_error_code(metrics["error_code"]))
    trial.set_user_attr("error_code_label", error_label)
    trial.set_user_attr("pruned", pruned)
    trial.set_user_attr("prune_reason", prune_reason)
    trial.set_user_attr("prune_rule", prune_rule)
    trial.set_user_attr("feasible", metrics.get("feasible", not pruned))
    trial.set_user_attr(
        "feasibility_status",
        metrics.get("feasibility_status", "not_evaluated" if pruned else "feasible"),
    )
    trial.set_user_attr("feasibility_rule", metrics.get("feasibility_rule", ""))
    trial.set_user_attr("feasibility_reason", metrics.get("feasibility_reason", ""))
    trial.set_user_attr("feasibility_metric", metrics.get("feasibility_metric", ""))
    trial.set_user_attr("feasibility_value", metrics.get("feasibility_value", ""))
    trial.set_user_attr("feasibility_reference", metrics.get("feasibility_reference", ""))
    trial.set_user_attr("feasibility_violation", metrics.get("feasibility_violation", ""))
    trial.set_user_attr("feasibility_constraints_json", metrics.get("feasibility_constraints_json", ""))
    if "feasibility_constraints" in metrics:
        trial.set_user_attr("feasibility_constraints", metrics["feasibility_constraints"])
    trial.set_user_attr("task_metrics", dict(trial_outcome.task_metrics))
    trial.set_user_attr("hyperparameters", dict(trial_outcome.hyperparams))
    trial.set_user_attr("artifact_summary", trial_outcome.artifact_summary)
    for metric_name, value in trial_outcome.task_metrics.items():
        trial.set_user_attr(f"metric__{metric_name}", value)
    for hyperparam_name, value in trial_outcome.hyperparams.items():
        if hyperparam_name in {
            "flops",
            "weight_bytes",
            "activation_bytes",
            "memory_traffic_bytes",
            "memory_proxy_dtype_bytes",
            "memory_proxy_warning_count",
        }:
            continue
        trial.set_user_attr(f"hparam__{hyperparam_name}", value)
    for field_name in CADENCED_ALL_FIELDS:
        trial.set_user_attr(field_name, metrics.get(field_name))

def build_odom_tcn_model(hyperparams: Dict) -> Model:
    """Build a CREST Keras model based on given hyperparameters.

    Parameters
    ----------
    hyperparams : addict.Dict
        Model hyperparameters containing at least ``timesteps``,
        ``input_dim``, ``nb_filters``, ``kernel_size``, ``dilations``,
        ``dropout_rate``, ``use_skip_connections``, and ``norm_flag``.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model with TCN layers and post-processing for velocity prediction.
    """
    inputs = Input(shape=(hyperparams.timesteps, hyperparams.input_dim))

    features = TCN(
        nb_filters=hyperparams.nb_filters,
        kernel_size=hyperparams.kernel_size,
        dilations=hyperparams.dilations,
        dropout_rate=hyperparams.dropout_rate,
        use_skip_connections=hyperparams.use_skip_connections,
        use_batch_norm=hyperparams.norm_flag,
    )(inputs)

    # Mirror the Odom TCN post-processing stack.
    # Each step updates `features` to the next layer's output
    features = Reshape((hyperparams.nb_filters, 1))(features)  # preserves symbolic tensor by staying in Keras space
    features = MaxPooling1D(pool_size=2)(features)
    features = Flatten()(features)
    features = Dense(32, activation="linear", name="pre")(features)

    # Outputs for velocity in X and Y directions
    vel_x = Dense(1, activation="linear", name="velx")(features)
    vel_y = Dense(1, activation="linear", name="vely")(features)

    # Define the Keras Model with specified inputs and outputs. Traces the layers from the given outputs
    model = Model(inputs=[inputs], outputs=[vel_x, vel_y])
    return model
