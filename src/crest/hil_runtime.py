# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Runtime-owned HIL request construction and metric collection helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from addict import Dict

from .hardware import HIL_controller, describe_error_code
from .microcontrollers import get_device as get_microcontroller_device
from .microcontrollers.arduino_base import normalize_power_metrics

logger = logging.getLogger(__name__)

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
        Runtime profile requested for the backend measurement pass.
    latency_budget_ms : float
        Per-window latency budget in milliseconds.
    measured_inference_runs : int
        Number of inferences averaged by the firmware for one measurement.

    Returns
    -------
    float
        Timeout in seconds used for the STM32 runtime serial session.
    """
    normalized_runtime_mode = str(runtime_mode).strip().lower()
    safe_runs = max(1, int(measured_inference_runs))
    if normalized_runtime_mode == "cadenced":
        return max(30.0, (float(latency_budget_ms) * safe_runs) / 1000.0 + 10.0)
    return 30.0


def set_error_code(metrics: dict[str, Any], code: int) -> None:
    """Attach a numeric error code and its descriptive label to ``metrics``.

    Parameters
    ----------
    metrics : dict[str, Any]
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
    """Populate cadenced metric defaults from raw backend telemetry.

    Parameters
    ----------
    metrics : dict[str, Any]
        Mutable metric dictionary updated in place.
    power_metrics : dict[str, Any] | None
        Raw power and timing telemetry parsed from the backend log.
    """
    raw_power_metrics = dict(power_metrics or {})
    # Backends may omit runtime-mode metadata entirely on failure paths, so
    # normalize to the stable non-cadenced default before filling the rest.
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
    # Only synthesize the label when the backend actually reported a cadenced
    # error code. The negative sentinel means "cadenced pass did not run".
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


@dataclass(frozen=True)
class HarnessConfig:
    """Energy-aware harness settings forwarded to ``HIL_controller``.

    Parameters
    ----------
    harness_serial_port : str | None
        Serial port for the INA228 harness.
    harness_fqbn : str | None
        FQBN used to compile/upload the harness sketch.
    harness_auto_flash : str | None
        Harness flashing policy (``once``, ``always``, ``never``).
    harness_arm_pin : int | None
        Harness arming GPIO pin.
    harness_trigger_pin : int | None
        Harness trigger GPIO pin.
    dut_arm_hold_ms : int | None
        Time to hold DUT arm low before trigger observation.
    harness_stable_low_ms : int | None
        Required stable-low arming duration.
    harness_ready_timeout_s : float | None
        Timeout waiting for ``HARNESS READY``.
    harness_arm_timeout_s : float | None
        Timeout waiting for a valid arm/trigger edge.
    harness_active_timeout_s : float | None
        Maximum active measurement window.
    harness_done_timeout_s : float | None
        Timeout waiting for ``DONE``.

    Attributes
    ----------
    harness_serial_port : str | None
        Serial port for the INA228 harness.
    harness_fqbn : str | None
        FQBN used to compile/upload the harness sketch.
    harness_auto_flash : str | None
        Harness flashing policy (``once``, ``always``, ``never``).
    harness_arm_pin : int | None
        Harness arming GPIO pin.
    harness_trigger_pin : int | None
        Harness trigger GPIO pin.
    dut_arm_hold_ms : int | None
        Time to hold DUT arm low before trigger observation.
    harness_stable_low_ms : int | None
        Required stable-low arming duration.
    harness_ready_timeout_s : float | None
        Timeout waiting for ``HARNESS READY``.
    harness_arm_timeout_s : float | None
        Timeout waiting for a valid arm/trigger edge.
    harness_active_timeout_s : float | None
        Maximum active measurement window.
    harness_done_timeout_s : float | None
        Timeout waiting for ``DONE``.
    """

    harness_serial_port: str | None
    harness_fqbn: str | None
    harness_auto_flash: str | None
    harness_arm_pin: int | None
    harness_trigger_pin: int | None
    dut_arm_hold_ms: int | None
    harness_stable_low_ms: int | None
    harness_ready_timeout_s: float | None
    harness_arm_timeout_s: float | None
    harness_active_timeout_s: float | None
    harness_done_timeout_s: float | None


@dataclass(frozen=True)
class CollectMetricsRequest:
    """Normalized request used by :func:`collect_metrics`.

    Parameters
    ----------
    hil_enabled : bool
        Whether to run HIL upload/measurement (vs compile-only proxy mode).
    energy_aware : bool
        Whether harness-assisted power measurement is enabled.
    flops : float
        Model FLOP estimate for trial bookkeeping.
    device_name : str
        Target hardware name.
    window_size : int
        Input window length compiled into firmware.
    input_dim : int
        Number of input channels compiled into firmware.
    dirpath : pathlib.Path
        Firmware project directory containing generated model artifacts.
    latency_proxy_max_flops : float
        Maximum FLOPs used by proxy latency normalization.
    serial_port : str | None
        DUT serial port used for upload/latency capture during HIL runs.
    latency_budget_ms : float | None, optional
        Target inference cadence in milliseconds for normalized latency checks.
    dut_ready_timeout_s : float | None, optional
        Timeout waiting for DUT ready handshake.
    serial_timeout_s : float | None, optional
        Post-``START`` runtime timeout forwarded to direct-serial backends.
    measured_inference_runs : int, optional
        Number of on-device inference invokes averaged into one measured HIL
        attempt.
    harness : HarnessConfig | None, optional
        Harness settings for energy-aware runs. ``None`` for non-energy-aware runs.
    device_options : dict[str, Any] | None, optional
        Optional board-specific options forwarded to the device factory.

    Attributes
    ----------
    hil_enabled : bool
        Whether to run HIL upload/measurement (vs compile-only proxy mode).
    energy_aware : bool
        Whether harness-assisted power measurement is enabled.
    flops : float
        Model FLOP estimate for trial bookkeeping.
    device_name : str
        Target hardware name.
    window_size : int
        Input window length compiled into firmware.
    input_dim : int
        Number of input channels compiled into firmware.
    dirpath : Path
        Firmware project directory containing generated model artifacts.
    latency_proxy_max_flops : float
        Maximum FLOPs used by proxy latency normalization.
    serial_port : str | None
        DUT serial port used for upload/latency capture during HIL runs.
    latency_budget_ms : float | None
        Target inference cadence in milliseconds for normalized latency checks.
    dut_ready_timeout_s : float | None
        Timeout waiting for DUT ready handshake.
    serial_timeout_s : float | None
        Post-``START`` runtime timeout forwarded to direct-serial backends.
    measured_inference_runs : int
        Number of on-device inference invokes averaged into one measured HIL
        attempt.
    harness : HarnessConfig | None
        Harness settings for energy-aware runs. ``None`` for non-energy-aware runs.
    device_options : dict[str, Any] | None
        Optional board-specific options forwarded to the device factory.
    """

    hil_enabled: bool
    energy_aware: bool
    flops: float
    device_name: str
    window_size: int
    input_dim: int
    dirpath: Path
    latency_proxy_max_flops: float
    serial_port: str | None
    latency_budget_ms: float | None = None
    dut_ready_timeout_s: float | None = None
    serial_timeout_s: float | None = None
    measured_inference_runs: int = 10
    harness: HarnessConfig | None = None
    device_options: dict[str, Any] | None = None


def build_collect_metrics_request(
    config: Dict,
    runtime_metadata: Dict,
    latency_budget_ms: float,
    *,
    dirpath: Path,
    device_options: dict[str, Any] | None,
    hil_enabled: bool | None = None,
    energy_aware: bool | None = None,
    window_size: int | None = None,
    input_dim: int | None = None,
) -> CollectMetricsRequest:
    """Build a :class:`CollectMetricsRequest` from runtime metadata and config.

    Parameters
    ----------
    config : addict.Dict
        Loaded runtime configuration.
    runtime_metadata : addict.Dict
        Runtime-owned metadata containing at least ``flops`` and, when
        ``input_dim`` is not passed explicitly, ``input_dim``.
    latency_budget_ms : float
        Logical per-batch latency budget in milliseconds. The caller owns
        resolving this from an explicit device override, dataset batch period,
        or legacy stride cadence.
    dirpath : Path
        Candidate sketch or project directory passed through to the controller.
    device_options : dict[str, Any] | None
        Optional backend-specific device overrides that should accompany the
        request.
    hil_enabled : bool | None, optional
        Explicit override for whether runtime HIL should be used.
    energy_aware : bool | None, optional
        Explicit override for whether harness energy measurement is required.
    window_size : int | None, optional
        Explicit resolved window-size override.
    input_dim : int | None, optional
        Explicit resolved input-dimension override.

    Returns
    -------
    CollectMetricsRequest
        Normalized request payload for :func:`collect_metrics`.

    Raises
    ------
    RuntimeError
        If runtime measurement requires a harness but ``device.harness_serial_port``
        is not configured.
    ValueError
        If runtime dimensions cannot be resolved from the passed context.
    """
    def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
        """Read a value from either an ``addict.Dict`` or a namespace-like object.

        Parameters
        ----------
        container : Any
            Mapping or object that may contain the requested key.
        key : str
            Field name to read from the mapping or object.
        default : Any
            Fallback returned when the field is absent.

        Returns
        -------
        Any
            Resolved value from ``container`` or ``default`` when absent.
        """
        getter = getattr(container, "get", None)
        if callable(getter):
            return getter(key, default)
        return getattr(container, key, default)

    training_config = _cfg_get(config, "training", Dict())
    device_config = _cfg_get(config, "device", Dict())
    effective_energy_aware = bool(_cfg_get(training_config, "energy_aware", False)) if energy_aware is None else bool(energy_aware)
    harness = None
    normalized_device_name = str(_cfg_get(device_config, "name", "")).strip().upper()
    effective_hil_enabled = bool(_cfg_get(device_config, "hil", False)) if hil_enabled is None else bool(hil_enabled)
    request_device_options = {} if device_options is None else dict(device_options)
    request_device_options["latency_budget_ms"] = float(latency_budget_ms)

    runtime_mode = "direct_serial"
    if effective_hil_enabled:
        # Ask the backend which runtime path it actually uses so harness-only
        # boards can force the harness contract even when energy accounting is off.
        try:
            runtime_device = get_microcontroller_device(
                normalized_device_name,
                serial_port=_cfg_get(device_config, "serial_port", None),
                device_options=request_device_options,
            )
        except ValueError:
            runtime_device = None
        if runtime_device is not None:
            runtime_mode_fn = getattr(runtime_device, "runtime_measure_mode", None)
            if callable(runtime_mode_fn):
                runtime_mode = str(runtime_mode_fn())

    # Some boards still require the harness even for non-energy runs because
    # their runtime telemetry is only reliable through the harness path.
    if effective_energy_aware or runtime_mode == "harness_only":
        harness_serial_port = _cfg_get(device_config, "harness_serial_port", None)
        if not harness_serial_port:
            raise RuntimeError(
                "Set device.harness_serial_port when runtime measurement requires the harness."
            )
        harness = HarnessConfig(
            harness_serial_port=harness_serial_port,
            harness_fqbn=_cfg_get(device_config, "harness_fqbn", None),
            harness_auto_flash=_cfg_get(device_config, "harness_auto_flash", None),
            harness_arm_pin=_cfg_get(device_config, "harness_arm_pin", None),
            harness_trigger_pin=_cfg_get(device_config, "harness_trigger_pin", None),
            dut_arm_hold_ms=_cfg_get(device_config, "dut_arm_hold_ms", None),
            harness_stable_low_ms=_cfg_get(device_config, "harness_stable_low_ms", None),
            harness_ready_timeout_s=_cfg_get(device_config, "harness_ready_timeout_s", None),
            harness_arm_timeout_s=_cfg_get(device_config, "harness_arm_timeout_s", None),
            harness_active_timeout_s=_cfg_get(device_config, "harness_active_timeout_s", None),
            harness_done_timeout_s=_cfg_get(device_config, "harness_done_timeout_s", None),
        )

    dut_ready_timeout = _cfg_get(device_config, "dut_ready_timeout_s", 5.0)
    if dut_ready_timeout is None:
        dut_ready_timeout = 5.0
    measured_inference_runs = int(_cfg_get(device_config, "measured_inference_runs", 10))
    serial_timeout = _cfg_get(device_config, "serial_timeout_s", 12.0)
    if serial_timeout is None:
        serial_timeout = 12.0
    configured_runtime_mode = _cfg_get(device_config, "runtime_mode", "back_to_back")
    if effective_hil_enabled and normalized_device_name == "STM32_NUCLEO_N657X0_Q":
        # STM32 bring-up has a nontrivial floor even when the config asks for a
        # shorter timeout, so clamp to the backend-owned minimum here.
        serial_timeout = max(
            float(serial_timeout),
            _minimum_stm32_serial_timeout_s(
                runtime_mode=str(configured_runtime_mode),
                latency_budget_ms=float(latency_budget_ms),
                measured_inference_runs=measured_inference_runs,
            ),
        )

    resolved_window_size = window_size
    if resolved_window_size is None:
        dataset_config = _cfg_get(config, "dataset", Dict())
        dataset_params = _cfg_get(dataset_config, "params", Dict())
        resolved_window_size = _cfg_get(dataset_params, "window_size", None)
    if resolved_window_size is None:
        raise ValueError("build_collect_metrics_request requires an explicit window_size or dataset.params.window_size.")

    raw_input_dim = _cfg_get(runtime_metadata, "input_dim", None) if input_dim is None else input_dim
    try:
        resolved_input_dim = int(raw_input_dim)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "build_collect_metrics_request requires runtime_metadata.input_dim to be present and integer-like."
        ) from exc

    raw_flops = _cfg_get(runtime_metadata, "flops", None)
    try:
        resolved_flops = float(raw_flops)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "build_collect_metrics_request requires runtime_metadata.flops to be present and numeric."
        ) from exc

    return CollectMetricsRequest(
        hil_enabled=effective_hil_enabled,
        energy_aware=effective_energy_aware,
        flops=resolved_flops,
        device_name=normalized_device_name,
        window_size=int(resolved_window_size),
        input_dim=resolved_input_dim,
        dirpath=Path(dirpath).resolve(),
        latency_proxy_max_flops=_cfg_get(training_config, "latency_proxy_max_flops", None),
        serial_port=_cfg_get(device_config, "serial_port", None),
        latency_budget_ms=latency_budget_ms,
        dut_ready_timeout_s=float(dut_ready_timeout),
        serial_timeout_s=float(serial_timeout),
        measured_inference_runs=measured_inference_runs,
        harness=harness,
        device_options=request_device_options,
    )


def collect_metrics(request: CollectMetricsRequest) -> dict:
    """Gather RAM/flash/latency metrics from the controller for HIL or proxy runs.

    Parameters
    ----------
    request : CollectMetricsRequest
        Normalized request containing all required and optional controller inputs.

    Returns
    -------
    dict
        RAM/flash/latency/arena metrics plus shared error-code fields, using
        stable sentinels when backend values are unavailable.

    Raises
    ------
    RuntimeError
        If runtime measurement requires a harness but ``request.harness`` is missing.
    """
    controller_kwargs = {
        "dirpath": request.dirpath,
        "chosen_device": request.device_name,
        "window_size": request.window_size,
        "number_of_channels": request.input_dim,
        "measured_inference_runs": request.measured_inference_runs,
    }
    if request.device_options is not None:
        controller_kwargs["device_options"] = request.device_options

    runtime_mode = "direct_serial"
    if request.hil_enabled:
        # The request already carries normalized config, but the runtime mode is
        # still backend-owned because some boards route measurement through a
        # harness while others can run direct-serial.
        try:
            runtime_device = get_microcontroller_device(
                str(request.device_name),
                serial_port=request.serial_port,
                device_options=request.device_options,
            )
        except ValueError:
            runtime_device = None
        if runtime_device is not None:
            runtime_mode_fn = getattr(runtime_device, "runtime_measure_mode", None)
            if callable(runtime_mode_fn):
                runtime_mode = str(runtime_mode_fn())

    if request.energy_aware and request.harness is None:
        raise RuntimeError(
            "energy_aware=True requires harness configuration; do not run without harness."
        )
    if request.hil_enabled and runtime_mode == "harness_only" and request.harness is None:
        raise RuntimeError(
            "Runtime mode requires harness configuration. Set device.harness_serial_port."
        )

    if request.hil_enabled and request.serial_port is not None:
        controller_kwargs["serial_port"] = request.serial_port
    elif request.hil_enabled and request.serial_port is None:
        raise RuntimeError(
            "Set serial_port before enabling HIL runs so uploads know which DUT to target."
        )

    if request.hil_enabled and request.dut_ready_timeout_s is not None:
        controller_kwargs["dut_ready_timeout_s"] = request.dut_ready_timeout_s
    if request.hil_enabled and request.serial_timeout_s is not None:
        controller_kwargs["serial_timeout_s"] = request.serial_timeout_s

    if (
        request.hil_enabled
        and request.harness is not None
        and (request.energy_aware or runtime_mode == "harness_only")
    ):
        # Only forward harness wiring when the backend will actually consume it.
        # That keeps proxy and direct-serial runs from carrying irrelevant fields.
        controller_kwargs["harness_serial_port"] = request.harness.harness_serial_port
        controller_kwargs["harness_fqbn"] = request.harness.harness_fqbn
        controller_kwargs["harness_auto_flash"] = request.harness.harness_auto_flash
        controller_kwargs["harness_arm_pin"] = request.harness.harness_arm_pin
        controller_kwargs["harness_trigger_pin"] = request.harness.harness_trigger_pin
        controller_kwargs["dut_arm_hold_ms"] = request.harness.dut_arm_hold_ms
        controller_kwargs["harness_stable_low_ms"] = request.harness.harness_stable_low_ms
        controller_kwargs["harness_ready_timeout_s"] = request.harness.harness_ready_timeout_s
        controller_kwargs["harness_arm_timeout_s"] = request.harness.harness_arm_timeout_s
        controller_kwargs["harness_active_timeout_s"] = request.harness.harness_active_timeout_s
        controller_kwargs["harness_done_timeout_s"] = request.harness.harness_done_timeout_s

    logger.info(
        "collect_metrics: invoking HIL_controller (hil=%s, serial_port=%s, harness_port=%s)",
        request.hil_enabled,
        request.serial_port,
        request.harness.harness_serial_port if request.harness is not None else None,
    )
    (
        ram_bytes,
        flash_bytes,
        latency_s,
        arena_bytes,
        error_code,
        power_metrics,
    ) = HIL_controller(
        run_hil=request.hil_enabled,
        **controller_kwargs,
    )
    logger.info(
        "collect_metrics: HIL_controller finished (error_code=%s, latency_s=%s, arena_bytes=%s)",
        error_code,
        latency_s,
        arena_bytes,
    )

    # Convert raw controller outputs into the stable metrics schema expected by
    # NAS, reporting, and integration tests. Missing values become explicit
    # sentinels instead of leaking backend-specific `None` handling upward.
    ram_bytes = ram_bytes if ram_bytes is not None else -1
    flash_bytes = flash_bytes if flash_bytes is not None else -1
    latency_ms = latency_s * 1000.0 if latency_s is not None else -1

    latency_budget_entry = -1.0
    if request.hil_enabled:
        if request.latency_budget_ms is None:
            raise ValueError(
                "latency_budget_ms must be provided when hil_enabled is True so the normalized latency penalty has consistent units."
            )
        if request.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be a positive value")
        latency_budget_entry = request.latency_budget_ms
    elif request.latency_proxy_max_flops <= 0:
        raise ValueError("latency_proxy_max_flops must be a positive value")

    backend_error_kind = None
    backend_error_detail = None
    external_flash_bytes = -1
    weight_storage_mode = "embedded"
    if power_metrics:
        backend_error_kind = power_metrics.get("backend_error_kind")
        backend_error_detail = power_metrics.get("backend_error_detail")
        raw_external_flash_bytes = power_metrics.get("external_flash_bytes")
        if raw_external_flash_bytes is not None:
            try:
                parsed_external_flash_bytes = int(float(raw_external_flash_bytes))
            except (TypeError, ValueError):
                parsed_external_flash_bytes = -1
            if parsed_external_flash_bytes >= 0:
                external_flash_bytes = parsed_external_flash_bytes
        raw_weight_storage_mode = power_metrics.get("weight_storage_mode")
        if raw_weight_storage_mode:
            weight_storage_mode = str(raw_weight_storage_mode)
    # Power/runtime telemetry comes back as a partially populated backend blob.
    # Normalize it once here so every caller sees one stable field set.
    normalized_power = normalize_power_metrics(power_metrics)
    harness_latency_ms = -1.0
    if normalized_power.get("harness_latency_s", -1.0) >= 0:
        harness_latency_ms = normalized_power["harness_latency_s"] * 1000.0
    metrics = {
        "ram_bytes": ram_bytes,
        "flash_bytes": flash_bytes,
        "external_flash_bytes": external_flash_bytes,
        "latency_ms": latency_ms if request.hil_enabled else -1,
        "latency_budget_ms": latency_budget_entry,
        "arena_bytes": arena_bytes,
        "hil_enabled": request.hil_enabled,
        "energy_aware": request.energy_aware,
        "weight_storage_mode": weight_storage_mode,
        "inference_seq": int(normalized_power["sequence"]) if normalized_power["sequence"] >= 0 else -1,
        "energy_mj_per_inference": normalized_power["energy_mj_per_inference"],
        "avg_power_mw": normalized_power["avg_power_mw"],
        "avg_current_ma": normalized_power["avg_current_ma"],
        "bus_voltage_v": normalized_power["bus_voltage_v"],
        "idle_power_mw": normalized_power["idle_power_mw"],
        "clock_hz": normalized_power["clock_hz"],
        "harness_latency_ms": harness_latency_ms,
    }
    set_error_code(metrics, error_code)
    apply_cadenced_metric_defaults(metrics, power_metrics)
    # Preserve STM32 acceptance provenance through the generic metric normalizer.
    for key, value in (power_metrics or {}).items():
        if key.startswith("measurement_"):
            metrics[key] = value
    if backend_error_kind is not None:
        metrics["backend_error_kind"] = str(backend_error_kind)
    if backend_error_detail is not None:
        metrics["backend_error_detail"] = str(backend_error_detail)
    return metrics
