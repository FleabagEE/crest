# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Describe zeroMQ REP server entry point for CREST HIL metric requests.

This module owns the host-side HIL server surface, including import-time
TensorFlow/absl logging suppression, exported sketch-variant name constants,
and the lazy bootstrap path for dataset, task, and model-family components.
"""

import argparse
import json
import uuid
import sys
from collections.abc import Mapping
import logging
import math
import shutil
from pathlib import Path
from typing import Any

import absl.logging
# import optuna
import serial
import tensorflow as tf
# import tensorflow_model_optimization as tfmot
import zmq
from addict import Dict
import numpy as np

from crest.builtin_components import ensure_builtin_components_registered
from crest.cadence import resolve_batch_period_ms
from crest.component_selection import cfg_get, resolve_component_selection
from crest.devices import (
    CandidatePrepareRequest,
    arduino_staged_sketch_path,
    _sync_arduino_sketch_variant_for_config,
)
from crest.hardware import (
    HIL_MASTER_DEVICE_NOT_FOUND,
    describe_error_code,
)
from crest.hil_runtime import build_collect_metrics_request, collect_metrics
from crest.errors import HIL_ERROR_OK, HIL_MASTER_FATAL
from crest.microcontrollers import (
    get_device as get_microcontroller_device,
    resolve_device_options,
)
from crest.microcontrollers.stm32_nucleo_n657x0 import (
    BOARD_NAME as STM32_BOARD_NAME,
    classify_stm32_backend_error,
)
from crest.microcontrollers import stm32_cube_clt, stm32_runtime
from crest.model import (
    DEFAULT_CONFIG_PATH,
    VALID_QUANTIZATION_MODES,
    configured_quantization_mode,
    load_config,
    quantization_requires_calibration,
    require_logical_input_shape,
    set_error_code,
    validate_model_input_shape,
)
from crest.pipeline_types import DataSplit, DatasetBundle, ModelBuildContext
from crest.registry import dataset_registry, model_family_registry
from crest.runtime_bootstrap import bootstrap_pipeline, instantiate_task_component

# Reduce noisy TensorFlow / absl logging at import time for CLI and server use.
tf.get_logger().setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

logger = logging.getLogger(__name__)
APPROX_TRAINED_VARIANT_NAME = "approx_trained"
REPRESENTATIVE_VARIANT_LEGACY_NAME = "representative"
PERTURBED_VARIANT_LEGACY_NAME = "bn_full_plus_non_bn_bias_perturbed"
# Backward-compatible alias used by existing scripts/imports.
PERTURBED_VARIANT_NAME = APPROX_TRAINED_VARIANT_NAME


def _configure_logging(level_name: str) -> None:
    """Configure root logging for the HIL server process.

    Parameters
    ----------
    level_name : str
        Logging level name (e.g., INFO, DEBUG). Unknown names fall back to
        ``INFO``.

    Returns
    -------
    None
        Configures the root logger through :func:`logging.basicConfig`.
    """
    level_value = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level_value,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _build_backend_failure_metrics(
    *,
    error_kind: str,
    error_detail: str,
    latency_budget_ms: float,
    hil_enabled: bool,
    energy_aware: bool,
) -> dict:
    """Return a structured metrics dict for backend/request boundary failures.

    Parameters
    ----------
    error_kind : str
        Stable backend error kind.
    error_detail : str
        Human-readable backend failure detail.
    latency_budget_ms : float
        Current trial latency budget in milliseconds.
    hil_enabled : bool
        Whether the request intended to run HIL.
    energy_aware : bool
        Whether the request intended energy-aware execution.

    Returns
    -------
    dict
        Metrics-shaped failure payload compatible with the HIL server contract.
        Numeric metrics are populated with sentinel values, backend error
        fields are stringified, ``latency_budget_ms`` is preserved only for
        HIL-enabled requests, and the fatal master error code is stamped into
        the result.
    """
    metrics = {
        "ram_bytes": -1,
        "flash_bytes": -1,
        "external_flash_bytes": -1,
        "latency_ms": -1,
        "latency_budget_ms": latency_budget_ms if hil_enabled else -1,
        "arena_bytes": -1,
        "hil_enabled": hil_enabled,
        "energy_aware": energy_aware,
        "inference_seq": -1,
        "energy_mj_per_inference": -1.0,
        "avg_power_mw": -1.0,
        "avg_current_ma": -1.0,
        "bus_voltage_v": -1.0,
        "idle_power_mw": -1.0,
        "clock_hz": -1.0,
        "harness_latency_ms": -1.0,
        "weight_storage_mode": "embedded",
        "backend_error_kind": str(error_kind),
        "backend_error_detail": str(error_detail),
    }
    set_error_code(metrics, HIL_MASTER_FATAL)
    return metrics

class HILServer:
    """Serve HIL metric requests over a ZeroMQ REP socket.

    Construction resolves the active dataset/task/model-family selection but
    defers registry-backed component instantiation and dataset loading until
    the pipeline is first needed. The REP socket is created in ``__init__``
    and bound later by :meth:`start`.

    Attributes
    ----------
    config : Dict
        Loaded NAS and device configuration.
    repo_root : Path
        Repository root used to locate sketch variants and other artifacts.
    sketch_variants_dir : Path
        Directory containing Arduino sketch templates selected per request.
    active_sketch_path : Path | None
        Most recently staged sketch path, when one has been selected.
    calibration_split : DataSplit | None
        Lazily resolved calibration split for model export workflows.
    context : zmq.Context
        Shared ZeroMQ context backing the server socket.
    socket : zmq.Socket
        REP socket bound by :meth:`start`.
    """

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        config: Dict | None = None,
    ) -> None:
        """Initialize the HIL server state and modular pipeline components.

        Parameters
        ----------
        config_path : Path, optional
            Path to the NAS/HIL YAML configuration file. Used when ``config`` is
            not provided.
        config : Dict | None, optional
            Pre-loaded configuration dictionary. When provided, this takes
            precedence over ``config_path``.

        Returns
        -------
        None
            Resolves component-selection state immediately, initializes lazy
            pipeline caches, derives sketch-path locations, and creates the
            shared ZeroMQ context / REP socket.
        """
        ensure_builtin_components_registered()
        self.config = config if config is not None else load_config(config_path)
        selection = self._resolve_component_selection(self.config)
        # Cache the selected component names and config blocks up front; the
        # registry-backed component objects are instantiated lazily later.
        self.dataset_name = selection["dataset_name"]
        self.task_name = selection["task_name"]
        self.model_family_name = selection["model_family_name"]
        self.dataset_config = selection["dataset_config"]
        self.task_config = selection["task_config"]
        self.model_config = selection["model_config"]
        self.dataset = None
        self.task = None
        self.model_family = None
        self.dataset_bundle = None
        self.target_spec = None
        self.model_build_context = None
        self._pipeline_bootstrapped = False

        # Resolve repository root once so sketch variants can be copied before each compile.
        self.repo_root = Path(__file__).resolve().parent.parent
        self.sketch_variants_dir = self.repo_root / "sketches"
        self.active_sketch_path: Path | None = None
        self.calibration_split: DataSplit | None = None
        self._calibration_split_resolved = False

        if self.config.device.hil is False:
            logger.warning("HIL is disabled in the configuration.")

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)

    @staticmethod
    def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
        """Read one field from a mapping-like or namespace-like container.

        Parameters
        ----------
        container : Any
            Object exposing either ``get`` or attribute access.
        key : str
            Field name to resolve.
        default : Any, optional
            Fallback value returned when ``key`` is unavailable.

        Returns
        -------
        Any
            Resolved configuration value or ``default``.
        """
        return cfg_get(container, key, default)

    def _ensure_pipeline_bootstrapped(self) -> None:
        """Load and cache the active dataset/task/model pipeline on first use.

        Returns
        -------
        None
            Subsequent calls are no-ops after the first successful bootstrap.
            When ``dataset_bundle`` has already been injected, that bundle is
            validated and reused instead of reloading the dataset.
        """
        if self._pipeline_bootstrapped:
            return
        pipeline = bootstrap_pipeline(
            self.config,
            dataset=self.dataset,
            bundle=self.dataset_bundle,
        )
        self.dataset_name = pipeline.selection["dataset_name"]
        self.task_name = pipeline.selection["task_name"]
        self.model_family_name = pipeline.selection["model_family_name"]
        self.dataset = pipeline.dataset
        self.task = pipeline.task
        self.model_family = pipeline.model_family
        self.dataset_bundle = pipeline.bundle
        self.target_spec = pipeline.target_spec
        self.metric_contract = pipeline.metric_contract
        self.model_build_context = pipeline.model_build_context
        self.dataset_config = pipeline.selection["dataset_config"]
        self.task_config = pipeline.selection["task_config"]
        self.model_config = pipeline.selection["model_config"]
        self._pipeline_bootstrapped = True

    def _resolve_component_selection(self, config: Any) -> dict[str, Any]:
        """Resolve active dataset, task, and model-family selections.

        Parameters
        ----------
        config : Any
            Loaded HIL/NAS configuration object.

        Returns
        -------
        dict[str, Any]
            Resolved component names and local configuration subtrees.
        """
        return resolve_component_selection(config)

    def _instantiate_task(
        self,
        task_name: str,
        task_config: Any,
    ) -> Any:
        """Instantiate the configured task using the explicit runtime contract.

        Parameters
        ----------
        task_name : str
            Registered task name.
        task_config : Any
            Task-local configuration subtree.

        Returns
        -------
        Any
            Instantiated task component.
        """
        return instantiate_task_component(
            task_name,
            self.config,
            task_config,
        )

    def _load_dataset_bundle(
        self,
        dataset_name: str,
        dataset_config: Any,
    ) -> tuple[Any, DatasetBundle]:
        """Instantiate the configured dataset and load its normalized bundle.

        Parameters
        ----------
        dataset_name : str
            Registered dataset name.
        dataset_config : Any
            Dataset-local configuration subtree.

        Returns
        -------
        tuple[Any, DatasetBundle]
            Instantiated dataset and its loaded normalized bundle.
        """
        dataset_cls = dataset_registry.get(dataset_name)
        dataset = dataset_cls()
        dataset.validate_config(dataset_config)
        bundle = dataset.load(dataset_config)
        return dataset, bundle

    def _build_model_context(
        self,
        bundle: DatasetBundle,
        target_spec: Any,
    ) -> ModelBuildContext:
        """Build the normalized model-family context for one dataset/task pair.

        Parameters
        ----------
        bundle : DatasetBundle
            Active dataset bundle.
        target_spec : Any
            Task-owned target specification.

        Returns
        -------
        ModelBuildContext
            Normalized build context for model-family operations.
        """
        return ModelBuildContext(
            input_shape=bundle.input_shape,
            input_dtype=bundle.input_dtype,
            target_spec=target_spec,
            dataset_metadata=dict(bundle.metadata),
            task_metadata=dict(target_spec.metadata),
        )

    def _initialize_component_state(
        self,
        selection: dict[str, Any],
        dataset: Any,
        bundle: DatasetBundle,
    ) -> None:
        """Instantiate and validate the active modular pipeline components.

        Parameters
        ----------
        selection : dict[str, Any]
            Resolved component names and local configuration subtrees.
        dataset : Any
            Instantiated dataset component that loaded ``bundle``.
        bundle : DatasetBundle
            Normalized dataset bundle backing this server.

        Returns
        -------
        None
        """
        model_family_cls = model_family_registry.get(selection["model_family_name"])
        task = self._instantiate_task(selection["task_name"], selection["task_config"])
        model_family = model_family_cls()

        task.validate_config(selection["task_config"])
        model_family.validate_config(selection["model_config"])

        target_spec = task.build_target_spec(bundle, selection["task_config"])
        model_build_context = self._build_model_context(bundle, target_spec)

        self.dataset_name = selection["dataset_name"]
        self.task_name = selection["task_name"]
        self.model_family_name = selection["model_family_name"]
        self.dataset = dataset
        self.task = task
        self.model_family = model_family
        self.dataset_bundle = bundle
        self.target_spec = target_spec
        self.model_build_context = model_build_context
        self.dataset_config = selection["dataset_config"]
        self.task_config = selection["task_config"]
        self.model_config = selection["model_config"]
        self._pipeline_bootstrapped = True

    def get_runtime_dimensions(self) -> tuple[int, int]:
        """Return the resolved runtime `(window_size, input_dim)` pair.

        Returns
        -------
        tuple[int, int]
            Resolved runtime dimensions for the active dataset/model context.
        """
        return self._resolve_runtime_dimensions()

    def get_calibration_inputs(self) -> np.ndarray | None:
        """Return representative calibration inputs when available.

        Returns
        -------
        np.ndarray | None
            Calibration inputs for export-capable backends, or ``None`` when
            the active dataset does not expose calibration data.
        """
        calibration_split = self._ensure_calibration_split()
        if calibration_split is None:
            return None
        return calibration_split.inputs

    @staticmethod
    def _parse_required_runtime_int(
        runtime_metadata: Mapping[str, Any],
        field_name: str,
    ) -> int:
        """Parse one required integer field from runtime metadata.

        Parameters
        ----------
        runtime_metadata : Mapping[str, Any]
            Runtime-owned request metadata.
        field_name : str
            Required field name.

        Returns
        -------
        int
            Parsed integer value.

        Raises
        ------
        ValueError
            If the field is missing or not a valid integer.
        """
        if field_name not in runtime_metadata:
            raise ValueError(f"HIL request hyperparams must include '{field_name}'.")
        raw_value = runtime_metadata[field_name]
        if isinstance(raw_value, bool):
            raise ValueError(f"HIL request hyperparams field '{field_name}' must be an integer.")
        try:
            return int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"HIL request hyperparams field '{field_name}' must be an integer."
            ) from exc

    @classmethod
    def _normalize_runtime_metadata(cls, runtime_metadata: Mapping[str, Any]) -> Dict:
        """Normalize required runtime metadata into typed integer values.

        Parameters
        ----------
        runtime_metadata : Mapping[str, Any]
            Raw runtime-owned request metadata.

        Returns
        -------
        Dict
            Normalized runtime metadata with validated integer values.
        """
        normalized = Dict(dict(runtime_metadata))
        normalized.flops = cls._parse_required_runtime_int(runtime_metadata, "flops")
        normalized.timesteps = cls._parse_required_runtime_int(runtime_metadata, "timesteps")
        normalized.input_dim = cls._parse_required_runtime_int(runtime_metadata, "input_dim")
        if "batch_size" in runtime_metadata:
            normalized.batch_size = cls._parse_required_runtime_int(runtime_metadata, "batch_size")
        return normalized

    def _validate_request_dimensions(
        self,
        *,
        requested_timesteps: int,
        requested_input_dim: int,
    ) -> None:
        """Reject HIL requests whose model dimensions diverge from the server.

        Parameters
        ----------
        requested_timesteps : int
            Request-owned logical timestep count.
        requested_input_dim : int
            Request-owned logical input dimension.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the bootstrapped model context is unavailable or mismatched.
        """
        self._ensure_pipeline_bootstrapped()
        input_shape = None if self.model_build_context is None else self.model_build_context.input_shape
        expected_timesteps, expected_input_dim = require_logical_input_shape(input_shape)
        if requested_timesteps != expected_timesteps or requested_input_dim != expected_input_dim:
            raise ValueError(
                "HIL request model dimensions do not match the active server configuration: "
                f"expected timesteps={expected_timesteps}, input_dim={expected_input_dim}; "
                f"got timesteps={requested_timesteps}, input_dim={requested_input_dim}."
            )

    def _resolve_dataset_numeric_setting(self, key: str) -> float:
        """Resolve one numeric dataset setting from metadata or config.

        Parameters
        ----------
        key : str
            Numeric dataset field name to resolve.

        Returns
        -------
        float
            Resolved numeric value.

        Raises
        ------
        ValueError
            If the field cannot be resolved to a finite positive number.
        """
        self._ensure_pipeline_bootstrapped()
        metadata = {} if self.dataset_bundle is None else dict(self.dataset_bundle.metadata)
        raw_value = metadata.get(key, self._cfg_get(self.dataset_config, key, None))
        if raw_value in (None, "") or isinstance(raw_value, bool):
            raise ValueError(f"Unable to resolve dataset setting '{key}' for the active configuration.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Dataset setting '{key}' must be numeric for the active configuration."
            ) from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Dataset setting '{key}' must be a positive finite value for the active configuration."
            )
        return value

    def _ensure_calibration_split(self) -> DataSplit | None:
        """Resolve and cache calibration data only when a backend needs it.

        Returns
        -------
        DataSplit | None
            Cached calibration split, or ``None`` when the dataset does not
            expose calibration data.
        """
        self._ensure_pipeline_bootstrapped()
        if not self._calibration_split_resolved:
            self.calibration_split = self.dataset.make_calibration_data(
                self.dataset_bundle,
                self.dataset_config,
            )
            self._calibration_split_resolved = True
        return self.calibration_split

    def _require_calibration_split(self) -> DataSplit:
        """Return calibration data required by export-capable backends.

        Returns
        -------
        DataSplit
            Resolved calibration split.

        Raises
        ------
        ValueError
            If the active dataset does not provide calibration data.
        """
        calibration_split = self._ensure_calibration_split()
        if calibration_split is None:
            raise ValueError(
                "The active dataset does not provide calibration data required for export preparation."
            )
        return calibration_split

    def _resolve_runtime_dimensions(self) -> tuple[int, int]:
        """Resolve runtime window size and input dimension from standardized context.

        Returns
        -------
        tuple[int, int]
            Resolved ``(window_size, input_dim)`` pair.

        Raises
        ------
        ValueError
            If the required dimensions cannot be derived from dataset metadata
            or model build context.
        """
        self._ensure_pipeline_bootstrapped()
        metadata = {} if self.dataset_bundle is None else dict(self.dataset_bundle.metadata)
        input_shape = None if self.model_build_context is None else self.model_build_context.input_shape
        resolved_input_shape = None
        raw_window_size = metadata.get("window_size")
        raw_input_dim = metadata.get("input_dim")
        if raw_window_size is None or raw_input_dim is None:
            resolved_input_shape = require_logical_input_shape(input_shape)
        if raw_window_size is None:
            raw_window_size = resolved_input_shape[0]
        if raw_input_dim is None:
            raw_input_dim = resolved_input_shape[1]
        if raw_window_size is None or raw_input_dim is None:
            raise ValueError(
                "Unable to resolve runtime window_size/input_dim from dataset metadata or model context."
            )
        return int(raw_window_size), int(raw_input_dim)

    def _materialize_candidate_model(
        self,
        family_hparams: dict[str, Any],
        *,
        model_variant: str,
        checkpoint_path: Path | str | None,
    ) -> Any:
        """Materialize and validate the model used for export preparation.

        Parameters
        ----------
        family_hparams : dict[str, Any]
            Model-family-owned hyperparameters.
        model_variant : str
            Requested export variant name.
        checkpoint_path : Path | str | None
            Optional checkpoint path for trained variants.

        Returns
        -------
        Any
            Validated model instance ready for export preparation.
        """
        self._ensure_pipeline_bootstrapped()
        model = self.model_family.materialize_export_model(
            family_hparams,
            self.model_build_context,
            self.model_config,
            model_variant=model_variant,
            checkpoint_path=checkpoint_path,
        )
        validate_model_input_shape(model, self.model_build_context.input_shape)
        self.task.validate_model_outputs(model, self.target_spec)
        return model

    def _build_candidate_prepare_request(
        self,
        *,
        model: Any,
        model_variant: str,
        checkpoint_path: Path | str | None,
        calibration_split: DataSplit | None,
        quantization_mode: str,
    ) -> CandidatePrepareRequest:
        """Assemble the typed backend request for candidate preparation.

        Parameters
        ----------
        model : Any
            Model instance to export and stage.
        model_variant : str
            Human-readable export variant label.
        checkpoint_path : Path | str | None
            Optional checkpoint path associated with the candidate.
        calibration_split : DataSplit | None
            Calibration split supplied to backends that require representative
            export data.
        quantization_mode : str
            Deployment quantization mode selected by the NAS client.

        Returns
        -------
        CandidatePrepareRequest
            Typed backend preparation request.
        """
        return CandidatePrepareRequest(
            config=self.config,
            model=model,
            model_variant=model_variant,
            artifact_root=Path(self.config.outputs.candidate_dir),
            tflite_model_path=Path(self.config.outputs.tflite_model_path),
            calibration_split=calibration_split,
            quantization_mode=quantization_mode,
            input_shape=self.model_build_context.input_shape,
            checkpoint_path=checkpoint_path,
        )

    def _resolve_export_model_variant(self, model_variant: str | None) -> str:
        """Resolve the export model variant for one HIL request.

        Parameters
        ----------
        model_variant : str | None
            Explicit caller override, or ``None`` to use
            ``model.params.export_variant`` from the active config.

        Returns
        -------
        str
            Non-empty model variant name.

        Raises
        ------
        ValueError
            If no usable variant is available.
        """
        if model_variant is not None:
            if not isinstance(model_variant, str) or not model_variant.strip():
                raise ValueError("model_variant must be a non-empty string when provided.")
            return model_variant.strip()
        params = getattr(self.model_config, "params", None)
        raw_variant = None if params is None else cfg_get(params, "export_variant", None)
        if not isinstance(raw_variant, str) or not raw_variant.strip():
            raise ValueError("Expected model.params.export_variant to be a non-empty string.")
        return raw_variant.strip()

    def _resolve_export_checkpoint_path(
        self,
        *,
        model_variant: str,
        checkpoint_path: Path | str | None,
    ) -> Path | str | None:
        """Resolve checkpoint fallback for trained export variants.

        Parameters
        ----------
        model_variant : str
            Resolved model variant name.
        checkpoint_path : pathlib.Path | str | None
            Explicit checkpoint path supplied by the caller.

        Returns
        -------
        pathlib.Path | str | None
            Explicit checkpoint path, config-derived checkpoint path for
            trained variants, or ``None`` for variants that do not require a
            checkpoint.
        """
        if checkpoint_path is not None:
            return checkpoint_path
        if model_variant.strip().lower().startswith("trained"):
            return Path(self.config.outputs.checkpoint_path)
        return None

    def _normalized_device_name(self) -> str:
        """Return the normalized configured device name.

        Returns
        -------
        str
            Upper-cased device name from the loaded configuration.
        """
        return str(getattr(self.config.device, "name", "")).strip().upper()

    def _effective_latency_budget_ms(self) -> float:
        """Return the resolved logical latency budget.

        Returns
        -------
        float
            Positive finite cadence budget in milliseconds. Already-loaded
            dataset metadata is used when available, but malformed request
            handling must not bootstrap the dataset just to build failure
            metrics.
        """
        metadata = {} if self.dataset_bundle is None else dict(self.dataset_bundle.metadata)
        return resolve_batch_period_ms(
            self.dataset_config,
            dataset_metadata=metadata,
            device_config=self.config.device,
        )

    def _configured_runtime_mode(self) -> str:
        """Return the normalized configured runtime mode.

        Returns
        -------
        str
            Lower-case runtime mode requested by the device configuration.
        """
        return str(getattr(self.config.device, "runtime_mode", "back_to_back")).strip().lower()

    def _run_arduino_cadenced_second_pass(
        self,
        *,
        runtime_device: object,
        prepared_dir: Path,
        base_metrics: dict,
        request_metrics_args,
        window_size: int,
        input_dim: int,
    ) -> dict:
        """Run and merge the Arduino-only cadenced second pass.

        Parameters
        ----------
        runtime_device : object
            Active runtime device.
        prepared_dir : Path
            Prepared candidate directory.
        base_metrics : dict
            Metrics emitted by the primary runtime pass.
        request_metrics_args : CollectMetricsRequest
            Shared primary metrics request payload.
        window_size : int
            Resolved runtime window size.
        input_dim : int
            Resolved runtime input dimension.

        Returns
        -------
        dict
            Updated metrics dictionary after the optional cadenced pass.
        """
        normalized_device_name = self._normalized_device_name()
        if self._configured_runtime_mode() != "cadenced":
            return base_metrics
        if normalized_device_name not in {"PORTENTA_H7", "ARDUINO_NANO_33_BLE_SENSE"}:
            return base_metrics
        if not request_metrics_args.hil_enabled:
            return base_metrics
        if int(base_metrics.get("error_code", -1)) != HIL_ERROR_OK:
            return base_metrics

        arena_bytes = int(base_metrics.get("arena_bytes", -1))
        if arena_bytes <= 0:
            return base_metrics

        runtime_device.set_input_mode(
            str(self.config.training.input_mode),
            outputs_dir=prepared_dir,
            config=self.config,
            sketches_dir=self.sketch_variants_dir,
            runtime_phase="cadenced",
        )
        sketch_candidate = arduino_staged_sketch_path(prepared_dir)
        if sketch_candidate.is_file():
            self.active_sketch_path = sketch_candidate
            logger.info("Using sketch variant: %s", self.active_sketch_path)

        arena_kb = max(1, int(math.ceil(float(arena_bytes) / 1024.0)))
        harness = request_metrics_args.harness
        cadenced_result = runtime_device.evaluate(
            dirpath=prepared_dir,
            arena_kb=arena_kb,
            window_size=int(window_size),
            num_channels=int(input_dim),
            serial_port=request_metrics_args.serial_port,
            run_hil=True,
            serial_timeout_s=float(request_metrics_args.serial_timeout_s or 12.0),
            measured_inference_runs=int(request_metrics_args.measured_inference_runs),
            dut_ready_timeout_s=request_metrics_args.dut_ready_timeout_s,
            harness_serial_port=None if harness is None else harness.harness_serial_port,
            harness_fqbn=None if harness is None else harness.harness_fqbn,
            harness_auto_flash=None if harness is None else harness.harness_auto_flash,
            harness_arm_pin=None if harness is None else harness.harness_arm_pin,
            harness_trigger_pin=None if harness is None else harness.harness_trigger_pin,
            dut_arm_hold_ms=None if harness is None else harness.dut_arm_hold_ms,
            harness_stable_low_ms=None if harness is None else harness.harness_stable_low_ms,
            harness_ready_timeout_s=None if harness is None else harness.harness_ready_timeout_s,
            harness_arm_timeout_s=None if harness is None else harness.harness_arm_timeout_s,
            harness_active_timeout_s=None if harness is None else harness.harness_active_timeout_s,
            harness_done_timeout_s=None if harness is None else harness.harness_done_timeout_s,
        )

        base_metrics["runtime_mode"] = "cadenced"
        base_metrics["cadenced_error_code"] = int(cadenced_result.error_code)
        base_metrics["cadenced_error_label"] = describe_error_code(
            int(cadenced_result.error_code),
            prefer_master=False,
        )
        if int(cadenced_result.error_code) != HIL_ERROR_OK:
            return base_metrics

        raw_energy = -1.0
        if cadenced_result.power_metrics is not None:
            try:
                raw_energy = float(
                    cadenced_result.power_metrics.get("energy_mj_per_inference", -1.0)
                )
            except (AttributeError, TypeError, ValueError):
                raw_energy = -1.0
        energy_per_slot = raw_energy if math.isfinite(raw_energy) and raw_energy >= 0.0 else -1.0
        base_metrics["cadenced_energy_mj_per_window"] = energy_per_slot
        base_metrics["cadenced_energy_mj_per_trial"] = (
            energy_per_slot * float(request_metrics_args.measured_inference_runs)
            if energy_per_slot >= 0.0
            else -1.0
        )
        return base_metrics

    def start(self) -> None:
        """Start the ZeroMQ REP loop that evaluates incoming hyperparameters.

        The server blocks waiting for JSON messages, runs
        :meth:`determine_metrics`, and responds with a metrics dictionary for
        each request.

        Returns
        -------
        None
        """
        endpoint = f"tcp://{self.config.network.host}:{self.config.network.port}"
        self.socket.bind(endpoint)
        logger.info("[HIL REP] Listening for hyperparameters on %s", endpoint)

        try:
            while True:
                payload = self.socket.recv_json()
                logger.debug("[HIL REP] Received payload: %s", payload)
                try:
                    family_hparams, runtime_metadata, quantization_mode, device_options_overrides = self._normalize_request_payload(payload)
                    metrics = self.determine_metrics(
                        family_hparams=family_hparams,
                        runtime_metadata=runtime_metadata,
                        quantization_mode=quantization_mode,
                        device_options_overrides=device_options_overrides,
                    )
                except ValueError as exc:
                    logger.error("Invalid HIL request payload: %s", exc)
                    metrics = self._request_boundary_failure_metrics(
                        error_kind="request",
                        error_detail=str(exc),
                    )

                logger.debug("[HIL REP] Sending metrics: %s", metrics)
                self.socket.send_json(metrics)
                if metrics.get("error_code") == HIL_MASTER_DEVICE_NOT_FOUND:
                    logger.error(
                        "Upload failed (device not found); stopping HIL server so the NAS run can be restarted."
                    )
                    break
        except KeyboardInterrupt:
            logger.info("[HIL REP] Shutting down HIL REP server.")
        finally:
            self.socket.close(linger=0)
            self.context.term()

    @staticmethod
    def _normalize_request_payload(payload: object) -> tuple[Dict, Dict, str, dict | None]:
        """Normalize one structured REQ payload into internal request parts.

        Parameters
        ----------
        payload : object
            Inbound JSON-decoded request payload.

        Returns
        -------
        tuple[Dict, Dict, str, dict | None]
            Family hyperparameters, runtime metadata, deployment quantization
            mode, and optional device option overrides.

        Raises
        ------
        ValueError
            If existing validation or execution checks fail.
        """
        if not isinstance(payload, Mapping):
            raise ValueError("HIL request payload must be a JSON object.")
        raw_family_hparams = payload.get("family_hparams", None)
        raw_runtime_metadata = payload.get("runtime_metadata", None)
        if not isinstance(raw_family_hparams, Mapping):
            raise ValueError("HIL request field `family_hparams` must be a JSON object.")
        if not isinstance(raw_runtime_metadata, Mapping):
            raise ValueError("HIL request field `runtime_metadata` must be a JSON object.")
        raw_quantization_mode = payload.get("quantization_mode", None)
        if raw_quantization_mode in (None, "") or isinstance(raw_quantization_mode, bool):
            raise ValueError("HIL request field `quantization_mode` must be one of: float, int8_ptq.")
        quantization_mode = str(raw_quantization_mode).strip().lower()
        if quantization_mode not in VALID_QUANTIZATION_MODES:
            raise ValueError("HIL request field `quantization_mode` must be one of: float, int8_ptq.")
        raw_overrides = payload.get("device_options_overrides", None)
        if raw_overrides is not None and not isinstance(raw_overrides, Mapping):
            raise ValueError(
                "HIL request field `device_options_overrides` must be a JSON object when provided."
            )
        overrides = dict(raw_overrides) if raw_overrides is not None else None
        return Dict(dict(raw_family_hparams)), Dict(dict(raw_runtime_metadata)), quantization_mode, overrides

    def _request_boundary_failure_metrics(
        self,
        *,
        error_kind: str,
        error_detail: str,
    ) -> dict:
        """Build failure metrics for malformed or invalid client requests.

        Parameters
        ----------
        error_kind : str
            Error category reported by the HIL server.
        error_detail : str
            Detailed error text reported by the HIL server.

        Returns
        -------
        dict
            Failure metrics payload that can be returned before request
            execution starts.
        """
        try:
            latency_budget_ms = self._effective_latency_budget_ms()
        except ValueError:
            latency_budget_ms = -1.0
        effective_hil_enabled = bool(self.config.device.hil)
        effective_energy_aware = bool(self.config.training.energy_aware and effective_hil_enabled)
        return _build_backend_failure_metrics(
            error_kind=error_kind,
            error_detail=error_detail,
            latency_budget_ms=latency_budget_ms,
            hil_enabled=effective_hil_enabled,
            energy_aware=effective_energy_aware,
        )

    def determine_metrics(
        self,
        family_hparams: Mapping[str, Any],
        runtime_metadata: Mapping[str, Any],
        quantization_mode: str | None = None,
        device_options_overrides: dict | None = None,
        checkpoint_path: Path | str | None = None,
        model_variant: str | None = None,
    ) -> dict:
        """Build/select a model variant, compile/export it, and collect HIL metrics.

        Parameters
        ----------
        family_hparams : Mapping[str, Any]
            Model-family-owned hyperparameters for candidate construction.
        runtime_metadata : Mapping[str, Any]
            Runtime-owned request metadata such as ``flops``, ``timesteps``,
            ``input_dim``, and ``batch_size``.
        quantization_mode : str | None, optional
            Deployment quantization mode for candidate export. Defaults to the
            fixed configured mode when called directly.
        checkpoint_path : Path | str | None, optional
            Checkpoint path used when ``model_variant`` starts with
            ``"trained"``. When omitted for a trained variant, the server uses
            ``config.outputs.checkpoint_path``.
        model_variant : str | None, optional
            Explicit model source/variant selector. When ``None``, the server
            uses ``model.params.export_variant`` from the active config.

        Returns
        -------
        dict
            Metrics dictionary returned by ``collect_metrics``.

        Raises
        ------
        ValueError
            If ``model_variant`` is unsupported or a trained variant is selected
            without a checkpoint path.
        FileNotFoundError
            If a requested trained checkpoint does not exist.
        """
        try:
            runtime_metadata = self._normalize_runtime_metadata(runtime_metadata)
        except ValueError as exc:
            return self._request_boundary_failure_metrics(
                error_kind="request",
                error_detail=str(exc),
            )
        try:
            resolved_quantization_mode = (
                configured_quantization_mode(self.config)
                if quantization_mode is None
                else str(quantization_mode).strip().lower()
            )
            if resolved_quantization_mode not in VALID_QUANTIZATION_MODES:
                raise ValueError("quantization_mode must be one of: float, int8_ptq.")
            allowed_modes = set(getattr(self.config.training.quantization, "choices", []))
            if resolved_quantization_mode not in allowed_modes:
                raise ValueError(
                    f"quantization_mode '{resolved_quantization_mode}' is not allowed by training.quantization.choices."
                )
        except ValueError as exc:
            return self._request_boundary_failure_metrics(
                error_kind="request",
                error_detail=str(exc),
            )
        try:
            self._ensure_pipeline_bootstrapped()
            latency_budget_ms = self._effective_latency_budget_ms()
            resolved_model_variant = self._resolve_export_model_variant(model_variant)
            resolved_checkpoint_path = self._resolve_export_checkpoint_path(
                model_variant=resolved_model_variant,
                checkpoint_path=checkpoint_path,
            )
        except ValueError as exc:
            return self._request_boundary_failure_metrics(
                error_kind="config",
                error_detail=str(exc),
            )
        try:
            self._validate_request_dimensions(
                requested_timesteps=runtime_metadata.timesteps,
                requested_input_dim=runtime_metadata.input_dim,
            )
        except ValueError as exc:
            return self._request_boundary_failure_metrics(
                error_kind="request",
                error_detail=str(exc),
            )
        resolved_device_options = resolve_device_options(self._normalized_device_name(), self.config.device) or {}
        filtered_device_options_overrides = {
            key: value
            for key, value in dict(device_options_overrides or {}).items()
            if value is not None
        }
        requested_cpu_clock_mhz = -1
        if "cpu_clock_mhz" in filtered_device_options_overrides:
            raw_requested_cpu_clock_mhz = filtered_device_options_overrides["cpu_clock_mhz"]
            if isinstance(raw_requested_cpu_clock_mhz, bool):
                return self._request_boundary_failure_metrics(
                    error_kind="request",
                    error_detail="device_options_overrides.cpu_clock_mhz must be an integer.",
                )
            try:
                requested_cpu_clock_mhz = int(raw_requested_cpu_clock_mhz)
            except (TypeError, ValueError):
                return self._request_boundary_failure_metrics(
                    error_kind="request",
                    error_detail="device_options_overrides.cpu_clock_mhz must be an integer.",
                )
            filtered_device_options_overrides["cpu_clock_mhz"] = requested_cpu_clock_mhz
        merged_device_options = {
            **resolved_device_options,
            **filtered_device_options_overrides,
        }
        merged_device_options["latency_budget_ms"] = float(latency_budget_ms)
        if self._normalized_device_name() == STM32_BOARD_NAME:
            merged_device_options["measurement_records_dir"] = str(
                (Path(getattr(self.config.outputs, "models_dir", "outputs")) / "hil_measurements").resolve())
        try:
            runtime_device = get_microcontroller_device(
                self._normalized_device_name(),
                serial_port=getattr(self.config.device, "serial_port", None),
                device_options=merged_device_options,
            )
        except ValueError as exc:
            logger.error("Invalid runtime device options: %s", exc)
            return self._request_boundary_failure_metrics(
                error_kind="request" if filtered_device_options_overrides else "config",
                error_detail=str(exc),
            )

        window_size, input_dim = self._resolve_runtime_dimensions()
        model = None
        calibration_split: DataSplit | None = None
        if runtime_device.requires_candidate_model():
            model = self._materialize_candidate_model(
                family_hparams,
                model_variant=resolved_model_variant,
                checkpoint_path=resolved_checkpoint_path,
            )
            if runtime_device.requires_training_data() and quantization_requires_calibration(resolved_quantization_mode):
                try:
                    calibration_split = self._require_calibration_split()
                except ValueError as exc:
                    logger.error("Export calibration data unavailable: %s", exc)
                    return self._request_boundary_failure_metrics(
                        error_kind="config",
                        error_detail=str(exc),
                    )

        effective_hil_enabled = bool(
            self.config.device.hil and runtime_device.supports_runtime_measurement()
        )
        harness_serial_port = getattr(self.config.device, "harness_serial_port", None)
        energy_requires_explicit_harness = self._normalized_device_name() in {
            "PORTENTA_H7",
            "ARDUINO_NANO_33_BLE_SENSE",
        }
        effective_energy_aware = bool(
            self.config.training.energy_aware
            and effective_hil_enabled
            and runtime_device.supports_energy_measurement()
            and (
                bool(harness_serial_port)
                or not energy_requires_explicit_harness
            )
        )
        prepared_dir: Path | None = None
        metrics = {}
        try:
            prepared_dir = runtime_device.prepare_candidate(
                request=self._build_candidate_prepare_request(
                    model=model,
                    model_variant=resolved_model_variant,
                    checkpoint_path=resolved_checkpoint_path,
                    calibration_split=calibration_split,
                    quantization_mode=resolved_quantization_mode,
                ),
            )
            prepared_dir = Path(prepared_dir)
            sketch_candidate = arduino_staged_sketch_path(prepared_dir)
            if runtime_device.requires_candidate_model() and sketch_candidate.is_file():
                self.active_sketch_path = sketch_candidate
                logger.info("Using sketch variant: %s", self.active_sketch_path)
            elif runtime_device.requires_candidate_model():
                self.active_sketch_path = None

            logger.info("Starting metric collection")
            try:
                request_metrics_args = build_collect_metrics_request(
                    config=self.config,
                    runtime_metadata=runtime_metadata,
                    latency_budget_ms=latency_budget_ms,
                    dirpath=prepared_dir,
                    device_options=merged_device_options,
                    hil_enabled=effective_hil_enabled,
                    energy_aware=effective_energy_aware,
                    window_size=window_size,
                    input_dim=input_dim,
                )
            except RuntimeError as exc:
                logger.error("Runtime metrics request failure: %s", exc)
                metrics = _build_backend_failure_metrics(
                    error_kind="config",
                    error_detail=str(exc),
                    latency_budget_ms=latency_budget_ms,
                    hil_enabled=effective_hil_enabled,
                    energy_aware=effective_energy_aware,
                )
            else:
                metrics = collect_metrics(request_metrics_args)
                metrics = self._run_arduino_cadenced_second_pass(
                    runtime_device=runtime_device,
                    prepared_dir=prepared_dir,
                    base_metrics=metrics,
                    request_metrics_args=request_metrics_args,
                    window_size=window_size,
                    input_dim=input_dim,
                )
        except (
            stm32_cube_clt.WorkflowError,
            stm32_runtime.STM32RuntimeProtocolError,
            serial.SerialException,
        ) as exc:
            if self._normalized_device_name() != STM32_BOARD_NAME:
                raise
            logger.error("STM backend failure: %s", exc)
            metrics = _build_backend_failure_metrics(
                error_kind=classify_stm32_backend_error(str(exc)),
                error_detail=str(exc),
                latency_budget_ms=latency_budget_ms,
                hil_enabled=effective_hil_enabled,
                energy_aware=effective_energy_aware,
            )
        finally:
            active_exception = sys.exc_info()[1]
            try:
                if self._normalized_device_name() == STM32_BOARD_NAME:
                    # Persist request identity outside disposable candidate builds.
                    records_dir = Path(getattr(self.config.outputs, "models_dir", "outputs")) / "hil_requests"
                    records_dir.mkdir(parents=True, exist_ok=True)
                    request_record = records_dir / f"{uuid.uuid4().hex}.json"
                    metrics["measurement_request_record"] = str(request_record.resolve())
                    active_exception = sys.exc_info()[1]
                    with request_record.open("x") as handle:
                        json.dump({"family_hparams": family_hparams,
                                   "runtime_metadata": runtime_metadata,
                                   "quantization_mode": resolved_quantization_mode,
                                   "device_options": merged_device_options,
                                   "prepared_dir": str(prepared_dir),
                                   "exception": None if active_exception is None else str(active_exception),
                                   "metrics": metrics}, handle, indent=2, default=str)
            except OSError:
                logger.exception("Failed to persist STM32 request evidence")
                # Preserve any active candidate exception. A successful request
                # must still fail closed if its required evidence cannot be saved.
                if active_exception is None:
                    raise
            finally:
                runtime_device.cleanup_prepared_candidate(prepared_dir)

        metrics["cpu_clock_mhz_requested"] = requested_cpu_clock_mhz
        if self.config.device.hil:
            metrics["latency_budget_ms"] = latency_budget_ms

        logger.debug(
            "[HIL REP] Runtime clock details: "
            "requested_cpu_clock_mhz=%s, "
            "effective_cpu_clock_mhz=%s, "
            "reported_clock_hz=%s",
            requested_cpu_clock_mhz,
            merged_device_options.get("cpu_clock_mhz", -1),
            metrics.get("clock_hz", -1.0),
        )
        logger.info("Metric collection complete")
        return metrics

    def _sync_sketch_variant(self) -> Path:
        """Copy the selected Arduino sketch variant into the active build directory.

        Selection depends on ``device.name``, optional Portenta
        ``device.portenta.target_core``, ``training.energy_aware``, and
        ``training.input_mode`` in the loaded config. Uniform sketches use the
        shared root ``sketches/crest_inference_*.ino`` assets, while
        dataset-specific representative/real analysis variants continue to use
        ``sketches/analysis_sketches``.

        Returns
        -------
        Path
            Path to the synchronized ``{candidate_dir.name}.ino`` sketch.

        Raises
        ------
        ValueError
            If the configured input mode is unsupported.
        FileNotFoundError
            If a required variant sketch or input header is missing.
        """
        return _sync_arduino_sketch_variant_for_config(
            self.config,
            Path(self.config.outputs.candidate_dir),
            sketches_dir=self.sketch_variants_dir,
        )

    def set_input_mode(self, input_mode: str, *, runtime_phase: str = "back_to_back") -> Path:
        """Set the input mode and resynchronize the active Arduino sketch variant.

        Parameters
        ----------
        input_mode : str
            Desired input mode, such as ``"uniform"``,
            ``"oxiod_representative"``, ``"oxiod_real"``,
            ``"urbansound8k_representative"``, or ``"urbansound8k_real"``.
        runtime_phase : str, optional
            Runtime sketch phase (``"back_to_back"`` or ``"cadenced"``).

        Returns
        -------
        Path
            Path to the active synchronized sketch file.
        """
        runtime_device = get_microcontroller_device(
            self._normalized_device_name(),
            serial_port=getattr(self.config.device, "serial_port", None),
            device_options=resolve_device_options(self._normalized_device_name(), self.config.device),
        )
        self.active_sketch_path = runtime_device.set_input_mode(
            input_mode,
            outputs_dir=Path(self.config.outputs.candidate_dir),
            config=self.config,
            sketches_dir=self.sketch_variants_dir,
            runtime_phase=runtime_phase,
        )
        logger.info("Using sketch variant: %s", self.active_sketch_path)
        return self.active_sketch_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CREST HIL server.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config YAML.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    config = load_config(cfg_path)
    _configure_logging(config.logging.level)

    server = HILServer(config_path=cfg_path, config=config)
    server.start()
