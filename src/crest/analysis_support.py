# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Shared helpers for analysis scripts that exercise the modular pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from addict import Dict

from .builtin_components import ensure_builtin_components_registered
from .cadence import resolve_batch_period_ms
from .component_selection import resolve_component_selection
from .model_families.odom_tcn import OdomTCNFamily
from .pipeline_types import DatasetBundle, ModelBuildContext, TargetSpec
from .registry import model_family_registry
from .runtime_bootstrap import instantiate_task_component

FIXED_ODOM_TCN_FAMILY_DEFAULTS = {
    "nb_filters": 10,
    "kernel_size": 12,
    "dilations": [1, 4, 8, 64],
    "dropout_rate": 0.0,
    "use_skip_connections": False,
    "norm_flag": True,
}
HIL_RUNTIME_METADATA_KEYS = ("flops", "batch_size", "timesteps", "input_dim")


def _build_odometry_target_spec() -> TargetSpec:
    """Return the fixed odometry regression target contract.

    Returns
    -------
    TargetSpec
        Two-output velocity regression contract used by the legacy odometry
        model family.
    """
    return TargetSpec(
        task_type="regression",
        output_names=["velx", "vely"],
        output_shapes=[(1,), (1,)],
        metadata={},
    )


def build_fixed_crest_hyperparams(
    *,
    window_size: int,
    input_dim: int,
    batch_size: int = 256,
    nb_filters: int | None = None,
) -> Dict:
    """Build the fixed Odom TCN family hyperparameters plus FLOP estimate.

    Parameters
    ----------
    window_size : int
        Logical input window length.
    input_dim : int
        Logical input channel count.
    batch_size : int, optional
        Runtime batch-size metadata stored alongside the family parameters.
    nb_filters : int | None, optional
        Optional override for the fixed Odom TCN filter count.

    Returns
    -------
    addict.Dict
        Fixed family parameters plus runtime metadata fields required by the
        HIL request boundary.
    """
    resolved_window_size = int(window_size)
    resolved_input_dim = int(input_dim)
    resolved_nb_filters = (
        int(FIXED_ODOM_TCN_FAMILY_DEFAULTS["nb_filters"])
        if nb_filters is None
        else int(nb_filters)
    )
    hyperparams = Dict(
        **FIXED_ODOM_TCN_FAMILY_DEFAULTS,
        nb_filters=resolved_nb_filters,
        batch_size=int(batch_size),
        timesteps=resolved_window_size,
        input_dim=resolved_input_dim,
    )

    # Analysis scripts need a stable FLOP count without routing through the
    # legacy shared model helper. Build and profile through the family.
    family = OdomTCNFamily()
    ctx = ModelBuildContext(
        input_shape=(resolved_window_size, resolved_input_dim),
        input_dtype="float32",
        target_spec=_build_odometry_target_spec(),
        dataset_metadata={},
        task_metadata={},
    )
    model = family.build_model(dict(hyperparams), ctx, {})
    hyperparams.flops = int(family.count_flops(model, ctx, {}))
    return hyperparams


def split_hil_request_hyperparams(
    hyperparams: Mapping[str, Any],
) -> tuple[Dict, Dict]:
    """Split a mixed hyperparameter payload into family and runtime sections.

    Parameters
    ----------
    hyperparams : Mapping[str, Any]
        Mixed payload containing both family hyperparameters and runtime-owned
        request metadata.

    Returns
    -------
    tuple[addict.Dict, addict.Dict]
        ``(family_hparams, runtime_metadata)`` ready for
        ``HILServer.determine_metrics(...)``.
    """
    family_hparams = Dict()
    runtime_metadata = Dict()
    for key, value in dict(hyperparams).items():
        if key in HIL_RUNTIME_METADATA_KEYS:
            runtime_metadata[key] = value
        else:
            family_hparams[key] = value
    return family_hparams, runtime_metadata


def derive_latency_budget_ms(dataset_config: Any, device_config: Any = None) -> float:
    """Resolve the logical latency budget, including explicit device overrides.

    Parameters
    ----------
    dataset_config : Any
        ``config.dataset.params`` subtree. Explicit ``batch_period_ms`` takes
        precedence over the legacy ``stride / sampling_rate_hz`` derivation.
    device_config : Any, optional
        Device subtree; explicit latency budget overrides dataset cadence.

    Returns
    -------
    float
        Positive finite logical per-batch cadence in milliseconds.
    """
    return resolve_batch_period_ms(dataset_config, device_config=device_config)


def require_calibration_inputs(calibration_inputs: np.ndarray | None) -> np.ndarray:
    """Require representative calibration inputs for export-oriented scripts.

    Parameters
    ----------
    calibration_inputs : np.ndarray | None
        Representative input samples used for export calibration.

    Returns
    -------
    np.ndarray
        Calibration samples guaranteed to be present.

    Raises
    ------
    ValueError
        If existing validation or execution checks fail.
    """
    if calibration_inputs is None:
        raise ValueError(
            "This analysis workflow requires representative calibration inputs, "
            "but the active dataset does not provide a calibration split."
        )
    return calibration_inputs


def resolve_task_contract(
    config: Any,
    bundle: DatasetBundle,
    *,
    checkpoint_path: Path | None = None,
    early_stopping_patience: int | None = None,
) -> tuple[Any, Any, TargetSpec]:
    """Resolve the active task component and target contract for one bundle.

    Parameters
    ----------
    config : Any
        Loaded runtime config.
    bundle : DatasetBundle
        Dataset bundle the script is operating on.
    checkpoint_path : Path | None, optional
        Optional checkpoint path override for task construction.
    early_stopping_patience : int | None, optional
        Optional early-stopping override for task construction.

    Returns
    -------
    tuple[Any, Any, TargetSpec]
        ``(task, task_config, target_spec)`` for the active config.
    """
    ensure_builtin_components_registered()
    selection = resolve_component_selection(config)
    task = instantiate_task_component(
        selection["task_name"],
        config,
        selection["task_config"],
        checkpoint_path=checkpoint_path,
        early_stopping_patience=early_stopping_patience,
    )
    task.validate_config(selection["task_config"])
    target_spec = task.build_target_spec(bundle, selection["task_config"])
    return task, selection["task_config"], target_spec


def build_model_context(bundle: DatasetBundle, target_spec: TargetSpec) -> ModelBuildContext:
    """Build the normalized model-family context for one ad hoc dataset bundle.

    Parameters
    ----------
    bundle : DatasetBundle
        Dataset bundle containing calibration data and metadata.
    target_spec : TargetSpec
        Task output contract associated with the dataset bundle.

    Returns
    -------
    ModelBuildContext
        Model-family build context assembled from dataset and task contracts.
    """
    return ModelBuildContext(
        input_shape=bundle.input_shape,
        input_dtype=bundle.input_dtype,
        target_spec=target_spec,
        dataset_metadata=dict(bundle.metadata),
        task_metadata=dict(target_spec.metadata),
    )


def resolve_model_family_contract(config: Any) -> tuple[Any, Any]:
    """Resolve the active model-family component and validated local config.

    Parameters
    ----------
    config : Any
        Loaded CREST configuration object or mapping.

    Returns
    -------
    tuple[Any, Any]
        Instantiated model-family component and its validated local config.
    """
    ensure_builtin_components_registered()
    selection = resolve_component_selection(config)
    model_family = model_family_registry.get(selection["model_family_name"])()
    model_family.validate_config(selection["model_config"])
    return model_family, selection["model_config"]
