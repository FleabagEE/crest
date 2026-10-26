# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Lazy registry helpers for CREST microcontroller backends.

This package surface keeps board wrapper imports lightweight for callers that
only need registry metadata, default specs, or config-to-backend option
normalization. Dedicated backend modules still own the compile, upload, and
runtime logic for each board family.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional


def _registry_entries() -> dict[str, tuple[str, str, str]]:
    """Return lazy registry metadata for supported microcontroller wrappers.

    Returns
    -------
    dict[str, tuple[str, str, str]]
        Mapping of ``device_name -> (module_path, class_name, default_spec_symbol)``.
    """
    return {
        "ARDUINO_NANO_33_BLE_SENSE": (
            "crest.microcontrollers.arduino_ble33",
            "ArduinoBLE33Device",
            "BOARD_DEFAULT_SPEC",
        ),
        "ARDUINO_NANO_RP2040_CONNECT": (
            "crest.microcontrollers.arduino_nano_rp2040",
            "ArduinoNanoRP2040Device",
            "BOARD_DEFAULT_SPEC",
        ),
        "RASPBERRY_PI_4": (
            "crest.microcontrollers.raspberry_pi4",
            "RaspberryPi4Device",
            "BOARD_DEFAULT_SPEC",
        ),
        "PORTENTA_H7": (
            "crest.microcontrollers.arduino_portenta_h7",
            "ArduinoPortentaH7Device",
            "BOARD_DEFAULT_SPEC",
        ),
        "STM32_NUCLEO_N657X0_Q": (
            "crest.microcontrollers.stm32_nucleo_n657x0",
            "STM32NucleoN657X0QDevice",
            "BOARD_DEFAULT_SPEC",
        ),
    }


def _load_symbol(module_path: str, symbol_name: str) -> Any:
    """Import and return a symbol from a module path.

    Parameters
    ----------
    module_path : str
        Fully-qualified module path.
    symbol_name : str
        Symbol defined in ``module_path``.

    Returns
    -------
    Any
        Imported symbol object.
    """
    module = __import__(module_path, fromlist=[symbol_name])
    return getattr(module, symbol_name)


def list_device_specs() -> dict[str, dict[str, Any]]:
    """Return registry-backed default specs for supported board classes.

    Returns
    -------
    dict[str, dict[str, Any]]
        Compatibility-style metadata with keys:
        ``arena_sizes``, ``max_ram``, ``max_flash``, and ``fqbn``.
    """
    specs: dict[str, dict[str, Any]] = {}
    entries = _registry_entries()
    for device_name, (module_path, _class_name, spec_symbol) in entries.items():
        spec = _load_symbol(module_path, spec_symbol)
        specs[device_name] = {
            "arena_sizes": list(spec.arena_sizes_kb),
            "max_ram": int(spec.max_ram_bytes),
            "max_flash": int(spec.max_flash_bytes),
            "max_external_flash": (
                int(spec.max_external_flash_bytes)
                if spec.max_external_flash_bytes is not None
                else None
            ),
            "fqbn": spec.fqbn,
        }
    return specs


def get_device(
    name: str,
    *,
    device_options: Optional[dict[str, Any]] = None,
    serial_port: Optional[str] = None,
):
    """Construct a board-specific device wrapper when available.

    Parameters
    ----------
    name : str
        Device identifier.
    device_options : dict[str, Any] | None, optional
        Optional board-specific options (for example Portenta core split).
    serial_port : str | None, optional
        Serial port used for upload/measure operations.

    Returns
    -------
    Any
        Device wrapper implementing the ``DeviceInterface`` contract.

    Raises
    ------
    ValueError
        If existing validation or execution checks fail.
    Exception
        If existing validation or execution checks fail.
    """
    normalized_name = name.strip().upper()
    entries = _registry_entries()
    if normalized_name in entries:
        module_path, class_name, _spec_symbol = entries[normalized_name]
        device_cls = _load_symbol(module_path, class_name)
        return device_cls(
            serial_port=serial_port,
            device_options=device_options,
        )

    # Fallback keeps compatibility with legacy dictionary-only board entries.
    from ..devices import ArduinoDevice

    try:
        return ArduinoDevice(
            normalized_name,
            serial_port=serial_port,
            device_options=device_options,
        )
    except ValueError as exc:
        # Legacy catalog entries without FQBN are not Arduino-backed and should
        # instruct callers to register a dedicated backend implementation.
        if "has no Arduino FQBN" in str(exc):
            raise ValueError(
                f"Device '{normalized_name}' has no registered backend in crest.microcontrollers "
                "and no Arduino FQBN fallback. Register a DeviceInterface-backed "
                "microcontroller class for this board."
            ) from exc
        raise


def resolve_device_options(
    device_name: str,
    device_config: Any,
) -> dict[str, Any] | None:
    """Resolve backend-owned device options from a config subtree.

    Parameters
    ----------
    device_name : str
        Target device identifier.
    device_config : Any
        Device config subtree or namespace-like object.

    Returns
    -------
    dict[str, Any] | None
        Normalized backend options, or ``None`` when the backend does not use
        explicit option fields.

    Notes
    -----
    Shared runtime knobs such as ``runtime_mode`` and ``latency_budget_ms``
    are merged here so higher-level orchestration code can stay board-agnostic.
    Backend-owned fields still come from per-board config subtrees because the
    Portenta and STM32 wrappers expose different option schemas.
    """
    def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
        """Read a config field from mapping-like or attribute-style objects.

        Parameters
        ----------
        container : Any
            Config object or mapping to inspect.
        key : str
            Field name to retrieve.
        default : Any, optional
            Fallback value returned when the field is missing.

        Returns
        -------
        Any
            Resolved field value or ``default``.
        """
        getter = getattr(container, "get", None)
        if callable(getter):
            return getter(key, default)
        return getattr(container, key, default)

    normalized_name = device_name.strip().upper()
    shared_runtime_mode = _cfg_get(device_config, "runtime_mode", None)
    shared_latency_budget_ms = _cfg_get(device_config, "latency_budget_ms", None)
    if normalized_name == "PORTENTA_H7":
        from .arduino_portenta_h7 import resolve_portenta_h7_options

        portenta_cfg = _cfg_get(device_config, "portenta", None)
        # Portenta exposes a compact board-option tuple, but shared runtime
        # fields are layered on afterward because they are not Arduino CLI
        # board options.
        resolved = resolve_portenta_h7_options(
            {
                "target_core": _cfg_get(portenta_cfg, "target_core", None) if portenta_cfg is not None else None,
                "split": _cfg_get(portenta_cfg, "split", None) if portenta_cfg is not None else None,
                "security": _cfg_get(portenta_cfg, "security", None) if portenta_cfg is not None else None,
            }
        )
        options = resolved.to_board_options()
        if shared_runtime_mode not in (None, ""):
            options["runtime_mode"] = shared_runtime_mode
        if shared_latency_budget_ms not in (None, ""):
            options["latency_budget_ms"] = shared_latency_budget_ms
        return options

    if normalized_name == "ARDUINO_NANO_33_BLE_SENSE":
        options: dict[str, Any] = {}
        if shared_runtime_mode not in (None, ""):
            options["runtime_mode"] = shared_runtime_mode
        if shared_latency_budget_ms not in (None, ""):
            options["latency_budget_ms"] = shared_latency_budget_ms
        return options or None

    if normalized_name == "ARDUINO_NANO_RP2040_CONNECT":
        from .arduino_nano_rp2040 import RP2040_OPTION_KEYS, resolve_rp2040_options

        rp2040_cfg = _cfg_get(device_config, "rp2040", None)
        # RP2040 keeps its latency-only knobs under `device.rp2040`; validate
        # them here so config errors surface before any board is touched.
        options = {}
        if rp2040_cfg is not None:
            for key in RP2040_OPTION_KEYS:
                value = _cfg_get(rp2040_cfg, key, None)
                if value not in (None, ""):
                    options[key] = list(value) if key == "arena_sizes_kb" else value
        resolve_rp2040_options(options)
        if shared_runtime_mode not in (None, ""):
            options["runtime_mode"] = shared_runtime_mode
        if shared_latency_budget_ms not in (None, ""):
            options["latency_budget_ms"] = shared_latency_budget_ms
        return options or None

    if normalized_name == "RASPBERRY_PI_4":
        from .raspberry_pi4 import RPI4_OPTION_KEYS, resolve_rpi4_options

        rpi4_cfg = _cfg_get(device_config, "rpi4", None)
        # The Pi backend keeps its SSH/runtime knobs under `device.rpi4`.
        options = {}
        if rpi4_cfg is not None:
            for key in RPI4_OPTION_KEYS:
                value = _cfg_get(rpi4_cfg, key, None)
                if value not in (None, ""):
                    options[key] = value
        resolve_rpi4_options(options)
        if shared_runtime_mode not in (None, ""):
            options["runtime_mode"] = shared_runtime_mode
        if shared_latency_budget_ms not in (None, ""):
            options["latency_budget_ms"] = shared_latency_budget_ms
        return options or None

    if normalized_name == "STM32_NUCLEO_N657X0_Q":
        from .stm32_nucleo_n657x0 import resolve_stm32_nucleo_n657x0_q_options

        stm32_cfg = _cfg_get(device_config, "stm32", None)
        # STM32 keeps most of its backend contract under `device.stm32`, but a
        # few runtime knobs are shared with other boards and therefore sourced
        # from the top-level device block.
        raw_options = {
            "template_root": _cfg_get(stm32_cfg, "template_root", None) if stm32_cfg is not None else None,
            "project_root": _cfg_get(stm32_cfg, "project_root", None) if stm32_cfg is not None else None,
            "project_layout": _cfg_get(stm32_cfg, "project_layout", None) if stm32_cfg is not None else None,
            "gdbserver": _cfg_get(stm32_cfg, "gdbserver", None) if stm32_cfg is not None else None,
            "gdb": _cfg_get(stm32_cfg, "gdb", None) if stm32_cfg is not None else None,
            "cubeprog_bin": _cfg_get(stm32_cfg, "cubeprog_bin", None) if stm32_cfg is not None else None,
            "signing_tool": _cfg_get(stm32_cfg, "signing_tool", None) if stm32_cfg is not None else None,
            "gdb_port": _cfg_get(stm32_cfg, "gdb_port", None) if stm32_cfg is not None else None,
            "apid": _cfg_get(stm32_cfg, "apid", None) if stm32_cfg is not None else None,
            "server_ready_timeout_s": _cfg_get(stm32_cfg, "server_ready_timeout_s", None)
            if stm32_cfg is not None
            else None,
            "runtime_mode": shared_runtime_mode,
            "latency_budget_ms": shared_latency_budget_ms,
            "wake_margin_us": _cfg_get(stm32_cfg, "wake_margin_us", None)
            if stm32_cfg is not None
            else None,
            "min_sleep_us": _cfg_get(stm32_cfg, "min_sleep_us", None)
            if stm32_cfg is not None
            else None,
            "weight_storage_mode": _cfg_get(stm32_cfg, "weight_storage_mode", None)
            if stm32_cfg is not None
            else None,
            "appli_flash_address": _cfg_get(stm32_cfg, "appli_flash_address", None)
            if stm32_cfg is not None
            else None,
            "weights_flash_address": _cfg_get(stm32_cfg, "weights_flash_address", None)
            if stm32_cfg is not None
            else None,
            "weights_memory_pool": _cfg_get(stm32_cfg, "weights_memory_pool", None)
            if stm32_cfg is not None
            else None,
            "weights_external_loader": _cfg_get(stm32_cfg, "weights_external_loader", None)
            if stm32_cfg is not None
            else None,
            "signing_load_offset": _cfg_get(stm32_cfg, "signing_load_offset", None)
            if stm32_cfg is not None
            else None,
            "signing_header_version": _cfg_get(stm32_cfg, "signing_header_version", None)
            if stm32_cfg is not None
            else None,
            "max_external_flash_bytes": _cfg_get(stm32_cfg, "max_external_flash_bytes", None)
            if stm32_cfg is not None
            else None,
        }
        resolved = resolve_stm32_nucleo_n657x0_q_options(
            {
                key: value
                for key, value in raw_options.items()
                if value not in (None, "")
            }
        )
        return asdict(resolved)

    return None


def __getattr__(name: str) -> Any:
    """Lazily expose board classes to avoid import cycles.

    Parameters
    ----------
    name : str
        Requested module attribute.

    Returns
    -------
    Any
        Loaded class object for known names.

    Raises
    ------
    AttributeError
        If ``name`` is not a supported lazy export.
    """
    if name == "ArduinoBLE33Device":
        return _load_symbol(
            "crest.microcontrollers.arduino_ble33",
            "ArduinoBLE33Device",
        )
    if name == "ArduinoNanoRP2040Device":
        return _load_symbol(
            "crest.microcontrollers.arduino_nano_rp2040",
            "ArduinoNanoRP2040Device",
        )
    if name == "ArduinoPortentaH7Device":
        return _load_symbol(
            "crest.microcontrollers.arduino_portenta_h7",
            "ArduinoPortentaH7Device",
        )
    if name == "RaspberryPi4Device":
        return _load_symbol(
            "crest.microcontrollers.raspberry_pi4",
            "RaspberryPi4Device",
        )
    if name == "STM32NucleoN657X0QDevice":
        return _load_symbol(
            "crest.microcontrollers.stm32_nucleo_n657x0",
            "STM32NucleoN657X0QDevice",
        )
    raise AttributeError(name)


__all__ = [
    "ArduinoBLE33Device",
    "ArduinoNanoRP2040Device",
    "ArduinoPortentaH7Device",
    "RaspberryPi4Device",
    "STM32NucleoN657X0QDevice",
    "get_device",
    "list_device_specs",
    "resolve_device_options",
]
