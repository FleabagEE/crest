# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Describe sTM32 Nucleo N657X0-Q backend for staged CREST evaluation.

This module owns the LRUN workspace contract, ST Edge AI code-generation flow,
staged manifest bookkeeping, and runtime/programming behavior for the
production STM32 backend.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import shutil
import subprocess
import uuid
import sys
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Mapping, Optional

import serial

from ..devices import (
    CandidatePrepareRequest,
    CompileResult,
    DeviceInterface,
    DeviceMetrics,
    DeviceSpec,
    MeasureResult,
    UploadResult,
)
from ..errors import (
    HIL_ERROR_COMPILE,
    HIL_ERROR_FLASH_OVERFLOW,
    HIL_ERROR_LATENCY,
    HIL_ERROR_INTEGRITY,
    HIL_ERROR_OK,
    HIL_ERROR_RAM_OVERFLOW,
    HIL_ERROR_UPLOAD,
)
from .. import hil_protocol
from . import arduino_base
from .stm32_measurement import validate_measurement, positive, firmware_period_ms
from . import stm32_cube_clt
from . import stm32_runtime


# Backend failures that indicate instrumentation failure and permit one retry.
MEASUREMENT_INTEGRITY_KINDS = frozenset({
    "measurement_integrity",
    "runtime_protocol",
    "runtime_latency",
    "harness_metrics_missing",
    "harness_run_count_mismatch",
    "harness_ready_timeout",
    "harness_done_timeout",
    "runtime_io",
    "runtime_start_timeout",
})

# Public device identifier used throughout config parsing and backend selection.
BOARD_NAME = "STM32_NUCLEO_N657X0_Q"
# Repository root used to resolve checked-in STM32 templates and helper assets.
REPO_ROOT = Path(__file__).resolve().parents[3]
# Canonical project root copied into a per-candidate staging directory.
DEFAULT_TEMPLATE_ROOT = REPO_ROOT / "sketches" / "stm32" / "crest_stm32_lrun"
# Internal STM32 workspace layout for the production N657X0 backend.
LRUN_PROJECT_LAYOUT = "lrun_dev_boot"
# ST STM32N657X0 product docs describe the part as having 4.2-Mbyte contiguous
# SRAM.
DEFAULT_MAX_RAM_BYTES = 4_194_304
# The LRUN path uses Boot at 0x70000000 and Appli before the weights region at
# 0x71000000, leaving 16 MiB for code images.
DEFAULT_MAX_FLASH_BYTES = 16_777_216
# ST UM3417 section 7.10 says the NUCLEO-N657X0-Q carries 512-Mbit Octo-SPI
# flash, which is 64 MiB usable capacity.
DEFAULT_MAX_EXTERNAL_FLASH_BYTES = 67_108_864
# Default fixed CPU clock preset written into the generated STM runtime config.
DEFAULT_CPU_CLOCK_MHZ = 600
DEFAULT_RUNTIME_MODE = "back_to_back"
DEFAULT_LATENCY_BUDGET_MS = 200.0
DEFAULT_WAKE_MARGIN_US = 5000
DEFAULT_MIN_SLEEP_US = 5000
# Weight placement mode used when callers do not override STM storage policy.
DEFAULT_WEIGHT_STORAGE_MODE = "embedded"
DEFAULT_APPLI_FLASH_ADDRESS = "0x70100000"
DEFAULT_SIGNING_HEADER_VERSION = stm32_cube_clt.DEFAULT_SIGNING_HEADER_VERSION
DEFAULT_SIGNING_LOAD_OFFSET = stm32_cube_clt.DEFAULT_SIGNING_LOAD_OFFSET
# Matches ST Edge AI STM32N6 CM55-validation examples, which generate external
# weights with `--address 0x71000000`; this is a workflow default, not a
# silicon-defined constant.
DEFAULT_WEIGHTS_FLASH_ADDRESS = "0x71000000"
# Default ST Edge AI memory-pool description for externalized weights.
DEFAULT_WEIGHTS_MEMORY_POOL = (
    REPO_ROOT / "src" / "config" / "stm32_nucleo_mypool.json"
)
# Supported fixed clock presets accepted by the STM staging pipeline.
SUPPORTED_CPU_CLOCK_MHZ = frozenset({200, 300, 400, 600, 800})
# Matches the checked-in LRUN AppS linker's `_Min_Heap_Size` reservation.
MIN_HEAP_BYTES = 0x2000
# Matches the checked-in LRUN AppS linker's `_Min_Stack_Size` reservation.
MIN_STACK_BYTES = 0x4000
# Safer default timeout for LRUN dev_boot, where the bootloader must copy the
# trusted app into AXISRAM before control transfers to the application.
DEFAULT_LRUN_BOOT_TIMEOUT_S = 12.0
# Generated ST Edge AI sources/headers that must exist after codegen.
EXPECTED_GENERATED_OUTPUTS = (
    "network.c",
    "network.h",
    "network_config.h",
    "network_data.c",
    "network_data.h",
    "network_data_params.c",
    "network_data_params.h",
)
# Filename emitted for externalized weight blobs.
WEIGHTS_BLOB_NAME = "network_data.bin"
# Default STM32CubeProgrammer external loader for the Nucleo flash device.
DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME = "MX25UM51245G_STM32N6570-NUCLEO.stldr"
# Manifest persisted in staged project roots to carry STM backend metadata.
STAGED_MANIFEST_NAME = "crest_stm32_manifest.json"
# Environment variable that preserves staged STM32 candidate roots for debugging.
KEEP_STAGED_CANDIDATES_ENV = "CREST_KEEP_STM32_CANDIDATES"
# Build-recipe files expected at the staged project root after makefile generation.
DEBUG_RECIPE_ROOT_FILENAMES = frozenset(
    {
        "makefile",
        "sources.mk",
        "objects.mk",
        "objects.list",
        "stedgeai.mk",
    }
)
# Accept both the checked-in ST Edge AI header shape ``(47792)`` and the
# simpler bare integer form used by lightweight test doubles.
ARENA_RE = re.compile(
    r"#define\s+AI_NETWORK_DATA_ACTIVATIONS_SIZE\s+\(?(?P<value>\d+)\)?"
)
# Accept the real linker-script assignment syntax as well as the temporary
# preprocessor-style fixtures that older tests used.
HEX_DEFINE_RE = re.compile(
    r"(?m)(?:#define\s+)?(?P<name>_Min_(?:Heap|Stack)_Size)"
    r"(?:\s+\(\(size_t\)\))?\s*(?:=)?\s*0x(?P<value>[0-9A-Fa-f]+)\s*;?"
)
MEASURED_RUNS_RE = re.compile(r"(?m)^#define\s+CREST_DUT_MEASURED_RUNS\s+(?P<value>\d+)\s*$")
COPY_WINDOW_DEFINE_RE = re.compile(
    r"^(#define\s+EXTMEM_LRUN_SOURCE_SIZE\s+)0x[0-9A-Fa-f]+\s*$",
    re.MULTILINE,
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class STM32NucleoN657X0QOptions:
    """Normalized STM32 backend options.

    Parameters
    ----------
    project_root : pathlib.Path
        Canonical STM32 template/workspace root. The backend copies this root
        into a per-candidate staging directory before build/upload.
    gdbserver : pathlib.Path | None
        Optional explicit path to ``ST-LINK_gdbserver``.
    gdb : pathlib.Path | None
        Optional explicit path to ``arm-none-eabi-gdb``.
    cubeprog_bin : pathlib.Path | None
        Optional explicit path to the STM32CubeProgrammer ``bin`` directory.
    gdb_port : int
        TCP port used by the ST-LINK GDB server.
    apid : int
        ST-LINK access port identifier.
    server_ready_timeout_s : float
        Timeout waiting for the GDB server to report that it is ready.
    cpu_clock_mhz : int
        Requested fixed CPU clock preset written into the generated phase
        configuration header.
    runtime_mode : str
        Requested STM32 runtime mode (`back_to_back` or `cadenced`).
    latency_budget_ms : float
        Cadence budget written into the generated phase configuration header.
    wake_margin_us : int
        Wake margin written into the generated phase configuration header.
    min_sleep_us : int
        Minimum stop-sleep duration written into the generated phase
        configuration header.
    weight_storage_mode : str
        Weight placement policy (`embedded` or `external_flash`).
    weights_flash_address : str
        Absolute external flash address used when externalizing weights.
    weights_memory_pool : pathlib.Path
        ST Edge AI memory-pool JSON used when externalizing weights.
    weights_external_loader : pathlib.Path | None
        Required `.stldr` loader path used to program external weight blobs.
    max_external_flash_bytes : int
        Maximum usable external flash bytes for staged weight blobs.

    Attributes
    ----------
    project_root : Path
        Canonical STM32 template/workspace root. The backend copies this root
        into a per-candidate staging directory before build/upload.
    gdbserver : Path | None
        Optional explicit path to ``ST-LINK_gdbserver``.
    gdb : Path | None
        Optional explicit path to ``arm-none-eabi-gdb``.
    cubeprog_bin : Path | None
        Optional explicit path to the STM32CubeProgrammer ``bin`` directory.
    gdb_port : int
        TCP port used by the ST-LINK GDB server.
    apid : int
        ST-LINK access port identifier.
    server_ready_timeout_s : float
        Timeout waiting for the GDB server to report that it is ready.
    cpu_clock_mhz : int
        Requested fixed CPU clock preset written into the generated phase
        configuration header.
    runtime_mode : str
        Requested STM32 runtime mode (`back_to_back` or `cadenced`).
    latency_budget_ms : float
        Cadence budget written into the generated phase configuration header.
    wake_margin_us : int
        Wake margin written into the generated phase configuration header.
    min_sleep_us : int
        Minimum stop-sleep duration written into the generated phase
        configuration header.
    weight_storage_mode : str
        Weight placement policy (`embedded` or `external_flash`).
    appli_flash_address : str
        External-flash base address used for the signed application image.
    weights_flash_address : str
        Absolute external flash address used when externalizing weights.
    weights_memory_pool : Path
        ST Edge AI memory-pool JSON used when externalizing weights.
    weights_external_loader : Path | None
        Required `.stldr` loader path used to program external weight blobs.
    signing_tool : Path | None
        Optional signing-tool executable used to wrap the trusted application.
    signing_load_offset : str
        Load offset embedded in the generated signed application header.
    signing_header_version : str
        Header-format version passed to the STM32 signing tool.
    max_external_flash_bytes : int
        Maximum usable external flash bytes for staged weight blobs.
    """

    project_root: Path
    gdbserver: Path | None
    gdb: Path | None
    cubeprog_bin: Path | None
    gdb_port: int
    apid: int
    server_ready_timeout_s: float
    cpu_clock_mhz: int
    runtime_mode: str
    latency_budget_ms: float
    wake_margin_us: int
    min_sleep_us: int
    weight_storage_mode: str
    appli_flash_address: str
    weights_flash_address: str
    weights_memory_pool: Path
    weights_external_loader: Path | None
    signing_tool: Path | None
    signing_load_offset: str
    signing_header_version: str
    max_external_flash_bytes: int
    measurement_records_dir: Path | None = None


@dataclass(frozen=True)
class STM32WorkspacePaths:
    """Resolved layout-aware STM32 workspace paths.

    Attributes
    ----------
    layout : str
        STM32 project layout identifier used to select path conventions.
    root : Path
        Root of the staged STM32 workspace.
    manifest_root : Path
        Directory where CREST staging metadata is persisted.
    source_root : Path
        Root containing generated ST Edge AI network sources.
    inc_dir : Path
        Include directory populated by ST Edge AI generation.
    src_dir : Path
        Source directory populated by ST Edge AI generation.
    linker_project_root : Path
        CubeIDE project root used for linker-script discovery.
    boot_project_root : Path
        CubeIDE Boot project root for LRUN dev_boot builds.
    app_project_root : Path
        CubeIDE App project root for trusted-application builds.
    boot_debug_dir : Path
        Boot project Debug directory containing make recipes and artifacts.
    app_debug_dir : Path
        App project Debug directory containing make recipes and artifacts.
    boot_elf_path : Path | None
        Boot ELF artifact path when the layout emits a separate boot image.
    app_elf_path : Path | None
        App ELF artifact path when the layout emits a separate application image.
    signed_app_bin_path : Path | None
        Signed trusted application binary ready for STM32 programming.
    boot_copy_window_header : Path | None
        Header containing the LRUN bootloader copy-window definition.
    """

    layout: str
    root: Path
    manifest_root: Path
    source_root: Path
    inc_dir: Path
    src_dir: Path
    linker_project_root: Path
    boot_project_root: Path
    app_project_root: Path
    boot_debug_dir: Path
    app_debug_dir: Path
    boot_elf_path: Path | None
    app_elf_path: Path | None
    signed_app_bin_path: Path | None
    boot_copy_window_header: Path | None


def _resolve_optional_path(value: object | None, *, base_dir: Path = REPO_ROOT) -> Path | None:
    """Normalize an optional filesystem path.

    Parameters
    ----------
    value : object | None
        Raw config or runtime value that may contain a path-like string.

    Returns
    -------
    pathlib.Path | None
        Resolved path when a non-empty value is provided, otherwise ``None``.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _coerce_int_with_default(value: object | None, default: int) -> int:
    """Return ``default`` for missing values, otherwise coerce to ``int``.

    Parameters
    ----------
    value : object | None
        Raw numeric value from config or request plumbing.
    default : int
        Fallback used when ``value`` is omitted.

    Returns
    -------
    int
        Normalized integer value.
    """
    if value in (None, ""):
        return int(default)
    return int(value)


def _coerce_float_with_default(value: object | None, default: float) -> float:
    """Return ``default`` for missing values, otherwise coerce to ``float``.

    Parameters
    ----------
    value : object | None
        Raw numeric value from config or request plumbing.
    default : float
        Fallback used when ``value`` is omitted.

    Returns
    -------
    float
        Normalized floating-point value.
    """
    if value in (None, ""):
        return float(default)
    return float(value)


def _resolve_cpu_clock_mhz(raw_value: object | None) -> int:
    """Validate and normalize the requested STM CPU clock preset.

    Parameters
    ----------
    raw_value : object | None
        Raw clock preset value from config or runtime options.

    Returns
    -------
    int
        Supported CPU clock preset in MHz.

    Raises
    ------
    ValueError
        If the requested preset is not one of the supported STM clock values.
    """
    clock_mhz = _coerce_int_with_default(raw_value, DEFAULT_CPU_CLOCK_MHZ)
    if clock_mhz not in SUPPORTED_CPU_CLOCK_MHZ:
        allowed = ", ".join(str(value) for value in sorted(SUPPORTED_CPU_CLOCK_MHZ))
        raise ValueError(f"Unsupported STM CPU clock preset {clock_mhz}. Expected one of: {allowed}.")
    return clock_mhz


def _resolve_weight_storage_mode(raw_value: object | None) -> str:
    """Validate and normalize the requested STM weight storage mode.

    Parameters
    ----------
    raw_value : object | None
        Raw storage-mode value from config or runtime options.

    Returns
    -------
    str
        Normalized storage mode.

    Raises
    ------
    ValueError
        If existing validation or execution checks fail.
    """
    mode = str(raw_value or DEFAULT_WEIGHT_STORAGE_MODE).strip().lower()
    if mode not in {"embedded", "external_flash"}:
        raise ValueError(
            "Unsupported STM weight storage mode "
            f"{mode!r}. Expected 'embedded' or 'external_flash'."
        )
    return mode


def _resolve_runtime_mode(raw_value: object | None) -> str:
    """Validate and normalize the requested STM runtime mode.

    Parameters
    ----------
    raw_value : object | None
        Raw runtime-mode value from config or runtime options.

    Returns
    -------
    str
        Normalized runtime mode.

    Raises
    ------
    ValueError
        If the requested runtime mode is unsupported.
    """
    runtime_mode = str(raw_value or DEFAULT_RUNTIME_MODE).strip().lower()
    if runtime_mode not in {"back_to_back", "cadenced"}:
        raise ValueError(
            "Unsupported STM runtime mode "
            f"{runtime_mode!r}. Expected 'back_to_back' or 'cadenced'."
        )
    return runtime_mode

def _resolve_cubeprog_cli_path(cubeprog_bin: Path | None) -> Path:
    """Resolve the STM32CubeProgrammer CLI executable path.

    Parameters
    ----------
    cubeprog_bin : pathlib.Path | None
        Optional configured STM32CubeProgrammer ``bin`` directory.

    Returns
    -------
    pathlib.Path
        Resolved ``STM32_Programmer_CLI`` executable path.
    """
    if cubeprog_bin is None:
        return stm32_cube_clt.resolve_required_tool_path(
            None,
            label="STM32_Programmer_CLI",
            hint="STM32_Programmer_CLI",
        )
    return stm32_cube_clt.resolve_required_tool_path(
        cubeprog_bin / "STM32_Programmer_CLI",
        label="STM32_Programmer_CLI",
        hint="STM32_Programmer_CLI",
    )


def resolve_stm32_nucleo_n657x0_q_options(
    device_options: Optional[Mapping[str, object]],
) -> STM32NucleoN657X0QOptions:
    """Resolve STM32 backend options from config/runtime input.

    Parameters
    ----------
    device_options : Mapping[str, object] | None
        Raw STM32 backend options supplied by config or request plumbing.

    Returns
    -------
    STM32NucleoN657X0QOptions
        Normalized options with defaults applied.

    Raises
    ------
    ValueError
        If existing validation or execution checks fail.
    """
    options = dict(device_options or {})
    raw_template_root = options.get("template_root")
    raw_project_root = options.get("project_root")
    if raw_template_root not in (None, ""):
        warnings.warn(
            "STM32 device option 'template_root' is deprecated; use 'project_root' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    project_root = _resolve_optional_path(raw_template_root)
    if project_root is None:
        project_root = _resolve_optional_path(raw_project_root)
    if project_root is None:
        project_root = DEFAULT_TEMPLATE_ROOT
    if options.get("project_layout") not in (None, ""):
        raise ValueError(
            "STM32 device option 'project_layout' is no longer supported for "
            f"{BOARD_NAME}; LRUN dev_boot is implicit."
        )
    project_root = Path(project_root).expanduser().resolve()
    gdbserver = _resolve_optional_path(options.get("gdbserver"))
    gdb = _resolve_optional_path(options.get("gdb"))
    cubeprog_bin = _resolve_optional_path(options.get("cubeprog_bin"))
    signing_tool = _resolve_optional_path(options.get("signing_tool"))
    gdb_port = _coerce_int_with_default(options.get("gdb_port"), stm32_cube_clt.DEFAULT_GDB_PORT)
    apid = _coerce_int_with_default(options.get("apid"), stm32_cube_clt.DEFAULT_APID)
    server_ready_timeout_s = _coerce_float_with_default(
        options.get("server_ready_timeout_s"),
        stm32_cube_clt.SERVER_READY_TIMEOUT_S,
    )
    cpu_clock_mhz = _resolve_cpu_clock_mhz(options.get("cpu_clock_mhz"))
    runtime_mode = _resolve_runtime_mode(options.get("runtime_mode"))
    latency_budget_ms = _coerce_float_with_default(
        options.get("latency_budget_ms"),
        DEFAULT_LATENCY_BUDGET_MS,
    )
    if latency_budget_ms <= 0.0:
        raise ValueError("STM latency_budget_ms must be a positive value.")
    wake_margin_us = _coerce_int_with_default(
        options.get("wake_margin_us"),
        DEFAULT_WAKE_MARGIN_US,
    )
    min_sleep_us = _coerce_int_with_default(
        options.get("min_sleep_us"),
        DEFAULT_MIN_SLEEP_US,
    )
    weight_storage_mode = _resolve_weight_storage_mode(options.get("weight_storage_mode"))
    appli_flash_address = str(
        options.get("appli_flash_address", DEFAULT_APPLI_FLASH_ADDRESS)
    ).strip()
    weights_flash_address = str(
        options.get("weights_flash_address", DEFAULT_WEIGHTS_FLASH_ADDRESS)
    ).strip()
    weights_memory_pool = (
        _resolve_optional_path(options.get("weights_memory_pool")) or DEFAULT_WEIGHTS_MEMORY_POOL
    )
    weights_external_loader = _resolve_optional_path(options.get("weights_external_loader"))
    max_external_flash_bytes = _coerce_int_with_default(
        options.get("max_external_flash_bytes"),
        DEFAULT_MAX_EXTERNAL_FLASH_BYTES,
    )
    return STM32NucleoN657X0QOptions(
        measurement_records_dir=(Path(str(options["measurement_records_dir"])).expanduser().resolve() if options.get("measurement_records_dir") else None),
        project_root=project_root,
        gdbserver=gdbserver,
        gdb=gdb,
        cubeprog_bin=cubeprog_bin,
        gdb_port=gdb_port,
        apid=apid,
        server_ready_timeout_s=server_ready_timeout_s,
        cpu_clock_mhz=cpu_clock_mhz,
        runtime_mode=runtime_mode,
        latency_budget_ms=latency_budget_ms,
        wake_margin_us=max(0, int(wake_margin_us)),
        min_sleep_us=max(0, int(min_sleep_us)),
        weight_storage_mode=weight_storage_mode,
        appli_flash_address=appli_flash_address,
        weights_flash_address=weights_flash_address,
        weights_memory_pool=Path(weights_memory_pool).expanduser().resolve(),
        weights_external_loader=weights_external_loader,
        signing_tool=signing_tool,
        signing_load_offset=str(options.get("signing_load_offset", DEFAULT_SIGNING_LOAD_OFFSET)),
        signing_header_version=str(options.get("signing_header_version", DEFAULT_SIGNING_HEADER_VERSION)),
        max_external_flash_bytes=max_external_flash_bytes,
    )


def build_stm32_nucleo_n657x0_q_spec(
    options: STM32NucleoN657X0QOptions | None = None,
) -> DeviceSpec:
    """Build the STM32 ``DeviceSpec`` exposed through the device registry.

    Parameters
    ----------
    options : STM32NucleoN657X0QOptions | None, optional
        Optional resolved board options that may override external-flash
        capacity metadata.

    Returns
    -------
    DeviceSpec
        Static STM32 board metadata used by the controller.
    """
    return DeviceSpec(
        name=BOARD_NAME,
        arena_sizes_kb=[-1],
        max_ram_bytes=DEFAULT_MAX_RAM_BYTES,
        max_flash_bytes=DEFAULT_MAX_FLASH_BYTES,
        max_external_flash_bytes=(
            DEFAULT_MAX_EXTERNAL_FLASH_BYTES
            if options is None
            else int(options.max_external_flash_bytes)
        ),
        toolchain="stm32-cube",
    )


def _resolve_workspace_paths(
    *,
    project_root: Path | str,
) -> STM32WorkspacePaths:
    """Resolve STM32 LRUN workspace paths.

    Parameters
    ----------
    project_root : Path | str
        Staged LRUN workspace root.

    Returns
    -------
    STM32WorkspacePaths
        Resolved path bundle for the staged workspace.

    Notes
    -----
    The LRUN flow validates both the Boot and AppS subprojects up front
    because later build, signing, and debug-load steps span both halves of the
    staged workspace.

    Raises
    ------
    WorkflowError
        If existing validation or execution checks fail.
    """
    root = Path(project_root).expanduser().resolve()
    resolved_root = root
    # The staged LRUN workspace must retain both the trusted boot path and the
    # application project because candidate generation touches both.
    for required_dir in (
        resolved_root / "FSBL",
        resolved_root / "Appli",
        resolved_root / "STM32CubeIDE" / "Boot",
        resolved_root / "STM32CubeIDE" / "AppS",
    ):
        if not required_dir.is_dir():
            raise stm32_cube_clt.WorkflowError(
                "STM32 project_root must point to the LRUN workspace; "
                f"missing required path: {required_dir}"
            )
    boot_project_root = stm32_cube_clt.validate_project_root(resolved_root / "STM32CubeIDE" / "Boot")
    app_project_root = stm32_cube_clt.validate_project_root(resolved_root / "STM32CubeIDE" / "AppS")
    return STM32WorkspacePaths(
        layout=LRUN_PROJECT_LAYOUT,
        root=resolved_root,
        manifest_root=resolved_root,
        source_root=resolved_root / "Appli",
        inc_dir=resolved_root / "Appli" / "Inc",
        src_dir=resolved_root / "Appli" / "Src",
        linker_project_root=app_project_root,
        boot_project_root=boot_project_root,
        app_project_root=app_project_root,
        boot_debug_dir=boot_project_root / "Debug",
        app_debug_dir=app_project_root / "Debug",
        boot_elf_path=boot_project_root / "Debug" / "Template_LRUN_FSBL.elf",
        app_elf_path=app_project_root / "Debug" / "Template_LRUN_AppS.elf",
        signed_app_bin_path=app_project_root / "Debug" / "Template_LRUN_AppS-trusted.bin",
        boot_copy_window_header=resolved_root / "FSBL" / "Inc" / "stm32_extmem_conf.h",
    )


def _find_linker_script(paths: STM32WorkspacePaths) -> Path:
    """Locate the primary linker script for the active STM32 workspace layout.

    Parameters
    ----------
    paths : STM32WorkspacePaths
        Layout-aware STM32 workspace paths.

    Returns
    -------
    pathlib.Path
        First linker script found under ``project_root``.

    Raises
    ------
    crest.microcontrollers.stm32_cube_clt.WorkflowError
        If no linker script exists in the project root.
    """
    linker_scripts = sorted(paths.linker_project_root.glob("*.ld"))
    if not linker_scripts:
        raise stm32_cube_clt.WorkflowError(
            f"Unable to locate linker script under {paths.linker_project_root}"
        )
    return linker_scripts[0]


def _find_boot_linker_script(paths: STM32WorkspacePaths) -> Path:
    """Locate the first boot linker script under an STM32 workspace.

    Parameters
    ----------
    paths : STM32WorkspacePaths
        Resolved workspace paths for the staged candidate.

    Returns
    -------
    Path
        Boot linker script path for the active workspace.

    Raises
    ------
    WorkflowError
        If existing validation or execution checks fail.
    """
    linker_scripts = sorted(paths.boot_project_root.glob("*.ld"))
    if not linker_scripts:
        raise stm32_cube_clt.WorkflowError(
            f"Unable to locate boot linker script under {paths.boot_project_root}"
        )
    return linker_scripts[0]


def _parse_linker_reservations(linker_script: Path) -> dict[str, int]:
    """Parse linker-reserved heap and stack bytes.

    Parameters
    ----------
    linker_script : pathlib.Path
        STM32 linker script containing ``_Min_Heap_Size`` and
        ``_Min_Stack_Size`` declarations.

    Returns
    -------
    dict[str, int]
        Parsed ``heap_bytes`` and ``stack_bytes`` values when present.
    """
    text = linker_script.read_text(encoding="utf-8")
    values: dict[str, int] = {}
    for match in HEX_DEFINE_RE.finditer(text):
        key = "heap_bytes" if "Heap" in match.group("name") else "stack_bytes"
        values[key] = int(match.group("value"), 16)
    return values


def _parse_arena_bytes(network_data_params: Path) -> int:
    """Parse the ST Edge AI activation arena size from a generated header.

    Parameters
    ----------
    network_data_params : pathlib.Path
        Generated ``network_data_params.h`` header emitted by ST Edge AI.

    Returns
    -------
    int
        Parsed activation arena size in bytes.

    Raises
    ------
    crest.microcontrollers.stm32_cube_clt.WorkflowError
        If the header is missing or does not contain the expected macro.
    """
    if not network_data_params.is_file():
        raise stm32_cube_clt.WorkflowError(f"Missing generated header: {network_data_params}")
    text = network_data_params.read_text(encoding="utf-8")
    match = ARENA_RE.search(text)
    if not match:
        raise stm32_cube_clt.WorkflowError(
            f"Unable to parse AI_NETWORK_DATA_ACTIVATIONS_SIZE from {network_data_params}"
        )
    return int(match.group("value"))


def _write_phase_config_header(
    *,
    paths: STM32WorkspacePaths,
    cpu_clock_mhz: int,
    selected_phase: str = DEFAULT_RUNTIME_MODE,
    latency_budget_ms: float = DEFAULT_LATENCY_BUDGET_MS,
    measured_runs: int = 10,
    wake_margin_us: int = DEFAULT_WAKE_MARGIN_US,
    min_sleep_us: int = DEFAULT_MIN_SLEEP_US,
) -> Path:
    """Write the generated STM phase-config header.

    Parameters
    ----------
    paths : STM32WorkspacePaths
        Layout-aware staged STM32 workspace paths.
    cpu_clock_mhz : int
        Requested fixed CPU clock preset.
    selected_phase : str, default="back_to_back"
        Runtime phase written into the generated header.
    latency_budget_ms : float, default=200.0
        Cadence budget written into the generated header.
    measured_runs : int, default=10
        Measured-run count written into the staged DUT phase configuration.
    wake_margin_us : int, default=5000
        Wake margin in microseconds.
    min_sleep_us : int, default=5000
        Minimum sleep duration in microseconds.

    Returns
    -------
    pathlib.Path
        Path to the generated phase-config header.
    """
    # Choose the runtime phase macro first so validation and file generation
    # share one normalized policy decision.
    normalized_phase = _resolve_runtime_mode(selected_phase)
    selected_phase_macro = (
        "CREST_DUT_PHASE_CADENCED"
        if normalized_phase == "cadenced"
        else "CREST_DUT_PHASE_BACK_TO_BACK"
    )
    header_path = paths.inc_dir / "crest_dut_phase_config.h"
    # Render one self-contained header because downstream STM32 builds consume
    # these runtime settings strictly through generated preprocessor macros.
    header_text = (
        "#ifndef CREST_DUT_PHASE_CONFIG_H\n"
        "#define CREST_DUT_PHASE_CONFIG_H\n\n"
        "#define CREST_DUT_PHASE_BACK_TO_BACK 0\n"
        "#define CREST_DUT_PHASE_CADENCED 1\n\n"
        f"#define CREST_DUT_SELECTED_PHASE {selected_phase_macro}\n"
        f"#define CREST_DUT_LATENCY_BUDGET_MS {firmware_period_ms(latency_budget_ms)}\n"
        f"#define CREST_DUT_MEASURED_RUNS {max(1, int(measured_runs))}\n"
        f"#define CREST_DUT_CPU_CLOCK_MHZ {int(cpu_clock_mhz)}\n"
        f"#define CREST_DUT_WAKE_MARGIN_US {max(0, int(wake_margin_us))}\n"
        f"#define CREST_DUT_MIN_SLEEP_US {max(0, int(min_sleep_us))}\n\n"
        "#endif /* CREST_DUT_PHASE_CONFIG_H */\n"
    )
    header_path.write_text(header_text, encoding="utf-8")
    return header_path


def _validate_phase_config_header(
    header_path: Path,
    *,
    cpu_clock_mhz: int,
    selected_phase: str,
    measured_runs: int,
    latency_budget_ms: float,
    wake_margin_us: int,
    min_sleep_us: int,
) -> None:
    """Confirm the generated phase-config header contains the expected values.

    Parameters
    ----------
    header_path : pathlib.Path
        Generated phase-config header path.
    cpu_clock_mhz : int
        Requested CPU clock preset expected in the generated header.
    selected_phase : str
        Requested runtime phase expected in the generated header.
    measured_runs : int
        Requested measured-run count expected in the generated header.
    latency_budget_ms : float
        Requested cadence budget expected in the generated header.
    wake_margin_us : int
        Requested wake margin expected in the generated header.
    min_sleep_us : int
        Requested minimum sleep duration expected in the generated header.

    Returns
    -------
    None

    Raises
    ------
    crest.microcontrollers.stm32_cube_clt.WorkflowError
        If the header does not contain the required STM runtime settings.
    """
    text = header_path.read_text(encoding="utf-8")
    normalized_phase = _resolve_runtime_mode(selected_phase)
    selected_phase_macro = (
        "CREST_DUT_PHASE_CADENCED"
        if normalized_phase == "cadenced"
        else "CREST_DUT_PHASE_BACK_TO_BACK"
    )
    expected_phase = f"#define CREST_DUT_SELECTED_PHASE {selected_phase_macro}"
    if expected_phase not in text:
        raise stm32_cube_clt.WorkflowError(
            "Generated phase config did not select the requested runtime phase "
            f"{normalized_phase}: {header_path}"
        )
    expected_clock = f"#define CREST_DUT_CPU_CLOCK_MHZ {int(cpu_clock_mhz)}"
    if expected_clock not in text:
        raise stm32_cube_clt.WorkflowError(
            f"Generated phase config is missing requested clock preset {cpu_clock_mhz}: {header_path}"
        )
    expected_runs = f"#define CREST_DUT_MEASURED_RUNS {max(1, int(measured_runs))}"
    if expected_runs not in text:
        raise stm32_cube_clt.WorkflowError(
            f"Generated phase config is missing requested measured-runs {measured_runs}: {header_path}"
        )
    expected_budget = f"#define CREST_DUT_LATENCY_BUDGET_MS {firmware_period_ms(latency_budget_ms)}"
    if expected_budget not in text:
        raise stm32_cube_clt.WorkflowError(
            "Generated phase config is missing requested cadence budget "
            f"{latency_budget_ms}: {header_path}"
        )
    expected_wake_margin = f"#define CREST_DUT_WAKE_MARGIN_US {max(0, int(wake_margin_us))}"
    if expected_wake_margin not in text:
        raise stm32_cube_clt.WorkflowError(
            "Generated phase config is missing requested wake margin "
            f"{wake_margin_us}: {header_path}"
        )
    expected_min_sleep = f"#define CREST_DUT_MIN_SLEEP_US {max(0, int(min_sleep_us))}"
    if expected_min_sleep not in text:
        raise stm32_cube_clt.WorkflowError(
            "Generated phase config is missing requested minimum sleep "
            f"{min_sleep_us}: {header_path}"
        )


def _read_phase_config_measured_runs(paths: STM32WorkspacePaths) -> int:
    """Read the staged DUT measured-run count from the generated phase header.

    Parameters
    ----------
    paths : STM32WorkspacePaths
        Layout-aware staged STM32 workspace paths.

    Returns
    -------
    int
        Positive DUT run count compiled into the staged project.

    Raises
    ------
    crest.microcontrollers.stm32_cube_clt.WorkflowError
        If the generated phase-config header is missing or does not declare
        ``CREST_DUT_MEASURED_RUNS``.
    """
    header_path = paths.inc_dir / "crest_dut_phase_config.h"
    if not header_path.is_file():
        raise stm32_cube_clt.WorkflowError(f"Missing generated phase config header: {header_path}")
    header_text = header_path.read_text(encoding="utf-8")
    match = MEASURED_RUNS_RE.search(header_text)
    if not match:
        raise stm32_cube_clt.WorkflowError(
            f"Generated phase config is missing CREST_DUT_MEASURED_RUNS: {header_path}"
        )
    return max(1, int(match.group("value")))


def _stage_generated_outputs(generated_output_dir: Path, paths: STM32WorkspacePaths) -> None:
    """Copy generated ST Edge AI sources and headers into the staged FSBL tree.

    Parameters
    ----------
    generated_output_dir : pathlib.Path
        Directory containing generated ``network*.c`` and ``network*.h`` files.
    paths : STM32WorkspacePaths
        Layout-aware staged STM32 workspace paths.

    Returns
    -------
    None

    Raises
    ------
    crest.microcontrollers.stm32_cube_clt.WorkflowError
        If any required generated output is missing.
    """
    src_dir = paths.src_dir
    inc_dir = paths.inc_dir
    src_dir.mkdir(parents=True, exist_ok=True)
    inc_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in EXPECTED_GENERATED_OUTPUTS if not (generated_output_dir / name).is_file()]
    if missing:
        missing_text = ", ".join(missing)
        raise stm32_cube_clt.WorkflowError(f"Missing generated outputs: {missing_text}")
    for name in EXPECTED_GENERATED_OUTPUTS:
        destination_dir = src_dir if name.endswith(".c") else inc_dir
        shutil.copy2(generated_output_dir / name, destination_dir / name)


def _generated_weights_blob_path(
    generated_output_dir: Path,
    *,
    weight_storage_mode: str,
) -> Path | None:
    """Return the generated external-weight blob path when present.

    Parameters
    ----------
    generated_output_dir : pathlib.Path
        Directory containing ST Edge AI outputs.
    weight_storage_mode : str
        Weight placement policy for the staged candidate.

    Returns
    -------
    pathlib.Path | None
        Generated blob path in `external_flash` mode, otherwise `None`.

    Raises
    ------
    WorkflowError
        If existing validation or execution checks fail.
    """
    if weight_storage_mode != "external_flash":
        return None
    blob_path = generated_output_dir / WEIGHTS_BLOB_NAME
    if not blob_path.is_file():
        raise stm32_cube_clt.WorkflowError(f"Missing generated weights blob: {blob_path}")
    return blob_path


def _resolve_weights_external_loader(
    cubeprog_bin: Path | None,
    weights_external_loader: Path | None,
) -> Path:
    """Resolve the Nucleo N657 external-loader path.

    Parameters
    ----------
    cubeprog_bin : pathlib.Path | None
        Optional STM32CubeProgrammer ``bin`` directory from config.
    weights_external_loader : pathlib.Path | None
        Explicit override from config or CLI, when provided.

    Returns
    -------
    pathlib.Path
        Resolved `.stldr` path for the board's external NOR flash.

    Raises
    ------
    crest.microcontrollers.stm32_cube_clt.WorkflowError
        If STM32CubeProgrammer is unavailable or the required loader cannot be
        found under the expected `ExternalLoader` directories.
    """
    if weights_external_loader is not None:
        return stm32_cube_clt.resolve_required_file_path(
            weights_external_loader,
            label="STM external loader",
        )
    cubeprog_dir = cubeprog_bin or stm32_cube_clt.default_cubeprog_bin()
    if cubeprog_dir is None:
        raise stm32_cube_clt.WorkflowError(
            "STM32CubeProgrammer not found; required for external_flash weight storage mode."
        )
    candidate_paths = (
        cubeprog_dir / "ExternalLoader" / DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME,
        cubeprog_dir.parent / "ExternalLoader" / DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME,
    )
    for loader_path in candidate_paths:
        if loader_path.is_file():
            return loader_path.resolve()
    candidate_text = "\n".join(f"  - {path}" for path in candidate_paths)
    raise stm32_cube_clt.WorkflowError(
        "Required Nucleo external loader was not found in the STM32CubeProgrammer install.\n"
        f"Searched:\n{candidate_text}\n"
        "Install STM32CubeProgrammer with ExternalLoader files, add "
        "`STM32_Programmer_CLI` to PATH, or configure device.stm32.weights_external_loader."
    )


def _manifest_path(paths: STM32WorkspacePaths) -> Path:
    """Return the fixed path of the staged STM manifest.

    Parameters
    ----------
    paths : STM32WorkspacePaths
        Layout-aware staged STM32 workspace paths.

    Returns
    -------
    pathlib.Path
        Sidecar manifest path stored inside the staged project root.
    """
    return paths.manifest_root / STAGED_MANIFEST_NAME


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a single file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_paths(entries: list[Path], *, root: Path) -> str:
    """Return a stable SHA-256 digest for a collection of files/directories.

    Parameters
    ----------
    entries : list[Path]
        Files or directories to include in the composite hash.
    root : Path
        Root used to normalize relative names before hashing.

    Returns
    -------
    str
        Hex digest that changes when file contents or relative paths change.
    """
    digest = hashlib.sha256()
    seen: set[Path] = set()
    resolved_root = root.resolve()
    for entry in sorted({item.resolve() for item in entries}):
        if not entry.exists():
            continue
        if entry.is_file():
            files = [entry]
        else:
            files = sorted(path for path in entry.rglob("*") if path.is_file())
        for file_path in files:
            resolved_file = file_path.resolve()
            if resolved_file in seen:
                continue
            seen.add(resolved_file)
            relative = resolved_file.relative_to(resolved_root)
            digest.update(str(relative).encode("utf-8"))
            digest.update(b"\0")
            with resolved_file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def _debug_recipe_inputs(debug_dir: Path) -> list[Path]:
    """Return checked-in CubeIDE recipe files that affect a staged build.

    Parameters
    ----------
    debug_dir : Path
        CubeIDE Debug directory containing generated make recipes.

    Returns
    -------
    list[Path]
        Recipe files that should participate in STM32 build-cache hashing.
    """
    recipe_files = [debug_dir / name for name in DEBUG_RECIPE_ROOT_FILENAMES if (debug_dir / name).is_file()]
    recipe_files.extend(sorted(path for path in debug_dir.rglob("subdir.mk") if path.is_file()))
    return recipe_files


def _lrun_app_build_input_hash(paths: STM32WorkspacePaths) -> str:
    """Return a stable digest for LRUN AppS build inputs.

    Parameters
    ----------
    paths : STM32WorkspacePaths
        Resolved STM32 workspace paths for the staged candidate.

    Returns
    -------
    str
        Digest that changes when LRUN App build inputs change.
    """
    entries = [
        paths.inc_dir,
        paths.src_dir,
        paths.root / "Secure_nsclib",
        paths.root / "Drivers",
        paths.root / "Middlewares",
        paths.app_project_root / "Src",
        paths.app_project_root / "Startup",
        _find_linker_script(paths),
        *_debug_recipe_inputs(paths.app_debug_dir),
    ]
    return _hash_paths(entries, root=paths.root)


def _lrun_boot_build_input_hash(paths: STM32WorkspacePaths) -> str:
    """Return a stable digest for LRUN Boot build inputs.

    Parameters
    ----------
    paths : STM32WorkspacePaths
        Resolved STM32 workspace paths for the staged candidate.

    Returns
    -------
    str
        Digest that changes when LRUN Boot build inputs change.
    """
    entries = [
        paths.root / "FSBL",
        paths.root / "Drivers",
        paths.root / "Middlewares",
        paths.boot_project_root / "Src",
        paths.boot_project_root / "Startup",
        _find_boot_linker_script(paths),
        *_debug_recipe_inputs(paths.boot_debug_dir),
    ]
    return _hash_paths(entries, root=paths.root)


def _lrun_app_sign_input_hash(
    app_bin: Path,
    *,
    signing_load_offset: str,
    signing_header_version: str,
) -> str:
    """Return a stable digest for signed-App generation inputs.

    Parameters
    ----------
    app_bin : Path
        Unsigned application binary emitted by the STM32 App build.
    signing_load_offset : str
        Load offset embedded into the STM32 signed application header.
    signing_header_version : str
        STM32 signing header version passed to the signing tool.

    Returns
    -------
    str
        Digest that changes when signed-App generation inputs change.
    """
    digest = hashlib.sha256()
    digest.update(_sha256_file(app_bin).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(signing_load_offset).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(signing_header_version).encode("utf-8"))
    return digest.hexdigest()


def _write_staged_manifest(
    *,
    paths: STM32WorkspacePaths,
    candidate_root: Path,
    generated_output_dir: Path,
    weight_storage_mode: str,
    weights_blob_path: Path | None,
    appli_signed_image_path: Path | None,
    boot_elf_path: Path | None,
    fsbl_copy_window_bytes: int | None,
    weights_flash_address: str,
    appli_flash_address: str,
    weights_external_loader: Path | None,
    existing_manifest: Mapping[str, object] | None = None,
    extra_fields: Mapping[str, object] | None = None,
) -> Path:
    """Write the staged STM manifest used across compile and evaluate.

    The manifest is valid only for the lifetime of one staged candidate root.
    Callers must keep the staged root intact until `evaluate()` finishes.

    Parameters
    ----------
    paths : STM32WorkspacePaths
        Layout-aware staged STM32 workspace paths.
    candidate_root : pathlib.Path
        Per-candidate root directory that contains generated outputs.
    generated_output_dir : pathlib.Path
        ST Edge AI output directory for this candidate.
    weight_storage_mode : str
        Weight placement policy.
    weights_blob_path : pathlib.Path | None
        Generated external-weight blob path when present.
    weights_flash_address : str
        Absolute external flash address for the blob.
        weights_external_loader : pathlib.Path | None
        Optional explicit `.stldr` loader path from config. When omitted, the
        backend auto-discovers the board-specific loader under the detected
        STM32CubeProgrammer install.

    Returns
    -------
    pathlib.Path
        Written manifest path.
    """
    manifest: dict[str, object] = dict(existing_manifest or {})
    manifest.update(
        {
            "staged_workspace_root": str(paths.root.resolve()),
            "candidate_root": str(candidate_root.resolve()),
            "generated_output_dir": str(generated_output_dir.resolve()),
            "weight_storage_mode": str(weight_storage_mode),
            "appli_signed_image_path": (
                str(appli_signed_image_path.resolve()) if appli_signed_image_path is not None else None
            ),
            "boot_elf_path": str(boot_elf_path.resolve()) if boot_elf_path is not None else None,
            "fsbl_copy_window_bytes": fsbl_copy_window_bytes,
            "appli_flash_address": str(appli_flash_address),
            "weights_blob_path": str(weights_blob_path.resolve()) if weights_blob_path is not None else None,
            "weights_blob_size": weights_blob_path.stat().st_size if weights_blob_path is not None else None,
            "weights_flash_address": str(weights_flash_address),
            "weights_external_loader": (
                str(weights_external_loader.resolve()) if weights_external_loader is not None else None
            ),
        }
    )
    if extra_fields:
        manifest.update(dict(extra_fields))
    manifest_path = _manifest_path(paths)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _update_staged_manifest(paths: STM32WorkspacePaths, **updates: object) -> Path:
    """Update selected manifest fields without dropping existing metadata.

    Parameters
    ----------
    paths : STM32WorkspacePaths
        Resolved STM32 workspace paths for the staged candidate.
    **updates : dict[str, object]
        Manifest fields to merge into the existing JSON payload.

    Returns
    -------
    Path
        Path to the updated staged manifest file.
    """
    manifest = _read_staged_manifest(paths)
    manifest.update(updates)
    manifest_path = _manifest_path(paths)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _read_staged_manifest(paths: STM32WorkspacePaths) -> dict[str, object]:
    """Read the staged STM manifest from a staged project root.

    The manifest is expected to survive only for one staged candidate
    lifecycle. Cleanup must not remove it before `evaluate()` finishes.

    Parameters
    ----------
    paths : STM32WorkspacePaths
        Layout-aware staged STM32 workspace paths.

    Returns
    -------
    dict[str, object]
        Parsed manifest payload.

    Raises
    ------
    WorkflowError
        If existing validation or execution checks fail.
    """
    manifest_path = _manifest_path(paths)
    if not manifest_path.is_file():
        raise stm32_cube_clt.WorkflowError(f"STM staged manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise stm32_cube_clt.WorkflowError(f"Unable to parse STM staged manifest: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise stm32_cube_clt.WorkflowError(f"Unexpected STM staged manifest payload: {manifest_path}")
    stored_workspace_root = payload.get("staged_workspace_root")
    if stored_workspace_root in (None, ""):
        raise stm32_cube_clt.WorkflowError(
            "STM LRUN staged manifest is missing the required staged_workspace_root field."
        )
    if stored_workspace_root not in (None, ""):
        resolved_manifest_root = Path(str(stored_workspace_root)).expanduser().resolve()
        resolved_workspace_root = paths.root.resolve()
        if resolved_manifest_root != resolved_workspace_root:
            raise stm32_cube_clt.WorkflowError(
                "STM staged manifest does not belong to the current staged workspace "
                f"({resolved_manifest_root} != {resolved_workspace_root})."
            )
    return payload


def _validate_project_structure(paths: STM32WorkspacePaths) -> STM32WorkspacePaths:
    """Validate the staged STM32 workspace shape and required headers.

    Parameters
    ----------
    paths : STM32WorkspacePaths
        Layout-aware STM32 workspace paths.

    Returns
    -------
    STM32WorkspacePaths
        Validated workspace paths.

    Raises
    ------
    crest.microcontrollers.stm32_cube_clt.WorkflowError
        If the project root is missing required directories, headers, or the
        generated CubeIDE makefile.
    """
    for candidate in (paths.src_dir, paths.inc_dir):
        if not candidate.is_dir():
            raise stm32_cube_clt.WorkflowError(f"STM32 project is missing required directory: {candidate}")
    required_makefiles = {paths.app_debug_dir / "makefile", paths.boot_debug_dir / "makefile"}
    for makefile in required_makefiles:
        if not makefile.is_file():
            raise stm32_cube_clt.WorkflowError(
                f"STM32 project is missing required CubeIDE makefile: {makefile}"
            )
    required_headers = ["stm32n6xx_hal_conf.h", "stm32n6xx_nucleo_conf.h", "crest_dut_runner.h"]
    required_headers.append("main.h")
    for required_header in required_headers:
        candidate = paths.inc_dir / required_header
        if not candidate.is_file():
            raise stm32_cube_clt.WorkflowError(f"STM32 project is missing required CubeN6 header: {candidate}")
    system_init = paths.src_dir / "system_stm32n6xx_s.c"
    if not system_init.is_file():
        raise stm32_cube_clt.WorkflowError(
            f"STM32 LRUN Appli is missing the required system init file: {system_init}"
        )
    _find_linker_script(paths)
    return paths


def _validate_memory_reservations(paths: STM32WorkspacePaths) -> tuple[int, int]:
    """Validate linker-reserved heap and stack sizes against minimum floors.

    Parameters
    ----------
    paths : STM32WorkspacePaths
        Layout-aware STM32 workspace paths.

    Returns
    -------
    tuple[int, int]
        Parsed ``(heap_bytes, stack_bytes)`` tuple.

    Raises
    ------
    crest.microcontrollers.stm32_cube_clt.WorkflowError
        If the reservations cannot be parsed or fall below the accepted floor.
    """
    min_heap_bytes = MIN_HEAP_BYTES
    min_stack_bytes = MIN_STACK_BYTES
    linker_scripts = [_find_linker_script(paths)]
    linker_scripts.append(_find_boot_linker_script(paths))

    app_heap_bytes: int | None = None
    app_stack_bytes: int | None = None
    for index, linker_script in enumerate(linker_scripts):
        reservations = _parse_linker_reservations(linker_script)
        heap_bytes = reservations.get("heap_bytes")
        stack_bytes = reservations.get("stack_bytes")
        if heap_bytes is None or stack_bytes is None:
            raise stm32_cube_clt.WorkflowError(
                f"Unable to parse linker heap/stack reservations from {linker_script}"
            )
        if heap_bytes < min_heap_bytes:
            raise stm32_cube_clt.WorkflowError(
                f"STM32 linker heap reservation is too small ({heap_bytes} bytes < {min_heap_bytes}) in {linker_script}."
            )
        if stack_bytes < min_stack_bytes:
            raise stm32_cube_clt.WorkflowError(
                f"STM32 linker stack reservation is too small ({stack_bytes} bytes < {min_stack_bytes}) in {linker_script}."
            )
        if index == 0:
            app_heap_bytes = heap_bytes
            app_stack_bytes = stack_bytes

    assert app_heap_bytes is not None
    assert app_stack_bytes is not None
    return app_heap_bytes, app_stack_bytes


def _validate_lrun_boot_include_path(paths: STM32WorkspacePaths) -> None:
    """Ensure LRUN Boot recipes still include the FSBL header directory.

    Parameters
    ----------
    paths : STM32WorkspacePaths
        Resolved STM32 workspace paths for the staged candidate.

    Raises
    ------
    WorkflowError
        If existing validation or execution checks fail.
    """
    if paths.boot_copy_window_header is None:
        return
    mk_files = sorted(paths.boot_debug_dir.rglob("*.mk"))
    if not mk_files:
        raise stm32_cube_clt.WorkflowError(
            f"STM32 LRUN Boot debug recipes are missing under {paths.boot_debug_dir}."
        )
    for mk_file in mk_files:
        text = mk_file.read_text(encoding="utf-8")
        if "FSBL/Inc" in text:
            return
    raise stm32_cube_clt.WorkflowError(
        "STM32 LRUN Boot recipes no longer reference the FSBL include path; "
        "EXTMEM_LRUN_SOURCE_SIZE updates would not affect the build."
    )


def _update_lrun_copy_window(
    *,
    paths: STM32WorkspacePaths,
    trusted_app_size: int,
    alignment: int = 0x400,
) -> tuple[Path, bool, int]:
    """Rewrite the staged LRUN copy-window define from the trusted app size.

    Parameters
    ----------
    paths : STM32WorkspacePaths
        Resolved STM32 workspace paths for the staged candidate.
    trusted_app_size : int
        Signed application size in bytes before copy-window alignment.
    alignment : int
        Byte alignment applied when rounding the copy window upward.

    Returns
    -------
    tuple[Path, bool, int]
        Updated header path, whether it changed, and aligned copy-window size.

    Raises
    ------
    WorkflowError
        If existing validation or execution checks fail.
    """
    if paths.boot_copy_window_header is None:
        raise stm32_cube_clt.WorkflowError(
            "LRUN copy-window updates require a boot copy-window header path."
        )
    header_path = paths.boot_copy_window_header
    if not header_path.is_file():
        raise stm32_cube_clt.WorkflowError(f"Missing LRUN copy-window header: {header_path}")
    aligned_size = ((int(trusted_app_size) + alignment - 1) // alignment) * alignment
    original_text = header_path.read_text(encoding="utf-8")
    replacement = rf"\g<1>0x{aligned_size:08X}"
    updated_text, replacements = COPY_WINDOW_DEFINE_RE.subn(replacement, original_text, count=1)
    if replacements != 1:
        raise stm32_cube_clt.WorkflowError(
            f"Could not update EXTMEM_LRUN_SOURCE_SIZE in {header_path}"
        )
    changed = updated_text != original_text
    if changed:
        header_path.write_text(updated_text, encoding="utf-8")
    return header_path, changed, aligned_size


def _validate_lrun_flash_layout(
    *,
    copy_window_bytes: int,
    appli_flash_address: str,
    weights_flash_address: str,
) -> None:
    """Validate that the LRUN boot copy window does not overlap the weights region.

    Parameters
    ----------
    copy_window_bytes : int
        Bootloader copy-window size in bytes for the trusted application.
    appli_flash_address : str
        Application image base address in external flash.
    weights_flash_address : str
        External-flash base address used for the weight blob.

    Raises
    ------
    WorkflowError
        If existing validation or execution checks fail.
    """
    appli_addr = int(appli_flash_address, 16)
    weights_addr = int(weights_flash_address, 16)
    if appli_addr >= weights_addr:
        raise stm32_cube_clt.WorkflowError("Flash layout must satisfy Appli < weights addresses.")
    if appli_addr + copy_window_bytes > weights_addr:
        raise stm32_cube_clt.WorkflowError("LRUN copy window overlaps the weights region.")


def _resolve_bin_artifact(elf_path: Path) -> Path:
    """Return the BIN artifact emitted alongside one STM32 ELF.

    Parameters
    ----------
    elf_path : Path
        ELF file whose section sizes are parsed.

    Returns
    -------
    Path
        Binary path derived from the ELF location.

    Raises
    ------
    WorkflowError
        If existing validation or execution checks fail.
    """
    candidate = elf_path.with_suffix(".bin")
    if candidate.is_file():
        return candidate
    siblings = sorted(elf_path.parent.glob("*.bin"))
    if len(siblings) == 1:
        return siblings[0]
    if siblings:
        raise stm32_cube_clt.WorkflowError(
            f"Could not determine matching BIN artifact for {elf_path}; found: "
            f"{', '.join(path.name for path in siblings)}"
        )
    raise stm32_cube_clt.WorkflowError(f"Missing BIN artifact after build: {candidate}")


def _strip_stale_debug_artifacts(paths: STM32WorkspacePaths) -> None:
    """Remove copied STM build outputs while preserving committed recipes.

    Parameters
    ----------
    paths : STM32WorkspacePaths
        Layout-aware STM32 workspace paths.
    """
    def _prune_recipe_tree(directory: Path) -> None:
        """Prune recipe tree.

        Parameters
        ----------
        directory : Path
            Directory to prune or inspect.
        """
        for nested in directory.iterdir():
            if nested.is_dir():
                _prune_recipe_tree(nested)
                continue
            if nested.name == "subdir.mk":
                continue
            nested.unlink()

    for debug_dir in {paths.boot_debug_dir, paths.app_debug_dir}:
        if not debug_dir.is_dir():
            continue
        for child in debug_dir.iterdir():
            if child.name in DEBUG_RECIPE_ROOT_FILENAMES:
                continue
            if child.is_dir():
                _prune_recipe_tree(child)
            else:
                child.unlink()


def _keep_staged_candidates_enabled() -> bool:
    """Return whether staged STM32 candidate roots should be preserved.

    Returns
    -------
    bool
        ``True`` when the debug environment variable requests that staged
        candidate workspaces stay on disk after evaluation.
    """
    keep_candidates = str(os.environ.get(KEEP_STAGED_CANDIDATES_ENV, "")).strip().lower()
    return keep_candidates in {"1", "true", "yes", "on"}


def _ensure_staging_tools() -> None:
    """Fail early when STM staging tools are unavailable.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Raises immediately when required staging tools are unavailable.
    """
    stm32_cube_clt.resolve_required_tool_path(
        None,
        label="stedgeai",
        hint="stedgeai",
    )


def _run_stedgeai_analyze(
    *,
    model_path: Path,
    workspace_dir: Path,
    output_dir: Path,
) -> None:
    """Run ST Edge AI analyze as a cheap compatibility preflight.

    Parameters
    ----------
    model_path : pathlib.Path
        Candidate-specific exported TFLite model.
    workspace_dir : pathlib.Path
        Per-candidate ST Edge AI workspace directory.
    output_dir : pathlib.Path
        Per-candidate output directory receiving analyze artifacts.

    Raises
    ------
    WorkflowError
        If existing validation or execution checks fail.
    """
    stedgeai = stm32_cube_clt.resolve_required_tool_path(
        None,
        label="stedgeai",
        hint="stedgeai",
    )
    workspace_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Analyze is a preflight step: it validates the candidate and surfaces
    # compatibility or memory issues before the heavier code-generation phase.
    cmd = [
        str(stedgeai),
        "analyze",
        "-m",
        str(model_path),
        "-t",
        "tflite",
        "--target",
        "stm32n6",
        "--quiet",
        "--workspace",
        str(workspace_dir),
        "--output",
        str(output_dir),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise stm32_cube_clt.WorkflowError(
            f"Failed to execute ST Edge AI analyze command: {' '.join(cmd)}\n\n{exc}"
        ) from exc
    output = f"{proc.stdout}{proc.stderr}".strip()
    if proc.returncode != 0:
        detail = output if output else "No ST Edge AI output captured."
        raise stm32_cube_clt.WorkflowError(
            f"ST Edge AI analyze failed with exit code {proc.returncode}.\n\n{detail}"
        )


def _run_stedgeai_generate(
    *,
    model_path: Path,
    workspace_dir: Path,
    output_dir: Path,
    weight_storage_mode: str,
    weights_flash_address: str,
    weights_memory_pool: Path,
) -> None:
    """Run ST Edge AI code generation for one staged candidate model.

    Parameters
    ----------
    model_path : pathlib.Path
        Candidate-specific exported TFLite model.
    workspace_dir : pathlib.Path
        Per-candidate ST Edge AI workspace directory.
    output_dir : pathlib.Path
        Per-candidate output directory receiving generated network files.
    weight_storage_mode : str
        Weight placement policy (`embedded` or `external_flash`).
    weights_flash_address : str
        Absolute external flash address used when externalizing weights.
    weights_memory_pool : pathlib.Path
        Memory-pool JSON used when externalizing weights.

    Returns
    -------
    None

    Raises
    ------
    crest.microcontrollers.stm32_cube_clt.WorkflowError
        If the ST Edge AI CLI cannot be started or generation fails.
    """
    stedgeai = stm32_cube_clt.resolve_required_tool_path(
        None,
        label="stedgeai",
        hint="stedgeai",
    )
    workspace_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Inject external-flash arguments only when the staged workspace uses
    # split weight storage; embedded builds keep the simpler default command.
    cmd = [
        str(stedgeai),
        "generate",
        "-m",
        str(model_path),
        "-t",
        "tflite",
        "--target",
        "stm32n6",
        "--c-api",
        "legacy",
    ]
    if weight_storage_mode == "external_flash":
        cmd.extend(
            [
                "--binary",
                "--address",
                str(weights_flash_address),
                "--memory-pool",
                str(weights_memory_pool),
            ]
        )
    cmd.extend(
        [
            "--quiet",
            "--workspace",
            str(workspace_dir),
            "--output",
            str(output_dir),
        ]
    )
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise stm32_cube_clt.WorkflowError(
            f"Failed to execute ST Edge AI generation command: {' '.join(cmd)}\n\n{exc}"
        ) from exc
    output = f"{proc.stdout}{proc.stderr}".strip()
    if proc.returncode != 0:
        detail = output if output else "No ST Edge AI output captured."
        raise stm32_cube_clt.WorkflowError(
            f"ST Edge AI generation failed with exit code {proc.returncode}.\n\n{detail}"
        )


def classify_stm32_backend_error(detail: str | BaseException) -> str:
    """Return a stable STM backend error kind for a diagnostic message.

    Parameters
    ----------
    detail : str
        Diagnostic text from a staging, toolchain, upload, or runtime failure.

    Returns
    -------
    str
        Stable STM-specific backend error kind.
    """
    if isinstance(detail, stm32_cube_clt.SigningWorkflowError):
        return "binary_signing"
    lowered = str(detail).lower()
    if "st edge ai" in lowered or "stedgeai" in lowered:
        unsupported_markers = (
            "unsupported",
            "not supported",
            "not yet supported",
            "operator",
            "op type",
            "layer",
            "incompatible",
            "not compatible",
        )
        if any(marker in lowered for marker in unsupported_markers):
            return "unsupported_model"
        return "codegen"
    if "external loader" in lowered or ".stldr" in lowered:
        return "external_flash_loader"
    if "binary signing failed" in lowered or "signingtool" in lowered:
        return "binary_signing"
    if "boot recipes no longer reference the fsbl include path" in lowered:
        return "boot_include_path"
    if "copy-window" in lowered or "extmem_lrun_source_size" in lowered:
        return "boot_copy_window_update"
    if "stm32_rtc_" in lowered:
        return "rtc_init"
    if "stm32_xspi_" in lowered or "external_weight_mapping" in lowered:
        return "external_weight_mapping"
    if "weights blob" in lowered or "external weight" in lowered or "network_data.bin" in lowered:
        if "overflow" in lowered or "too large" in lowered:
            return "external_flash_overflow"
        return "external_flash_programming"
    if "harness ready" in lowered:
        return "harness_ready_timeout"
    if "harness done" in lowered:
        return "harness_done_timeout"
    if "run-count mismatch" in lowered or "run count mismatch" in lowered:
        return "harness_run_count_mismatch"
    if "gdb" in lowered or "st-link" in lowered or "cubeprog" in lowered:
        return "upload"
    if "dut ready" in lowered or "stm32_ai_" in lowered or "runtime" in lowered:
        return "runtime_protocol"
    return "toolchain"


def _manifest_weight_storage_mode(manifest: Mapping[str, object]) -> str:
    """Return the normalized weight storage mode from a staged manifest.

    Parameters
    ----------
    manifest : Mapping[str, object]
        Parsed staged-manifest payload.

    Returns
    -------
    str
        Normalized staged weight storage mode.
    """
    return _resolve_weight_storage_mode(manifest.get("weight_storage_mode"))


def _manifest_external_flash_bytes(manifest: Mapping[str, object]) -> int | None:
    """Return staged external-weight bytes from a manifest.

    Parameters
    ----------
    manifest : Mapping[str, object]
        Parsed staged-manifest payload.

    Returns
    -------
    int | None
        Staged external-weight blob size in bytes when present.
    """
    value = manifest.get("weights_blob_size")
    if value in (None, ""):
        return None
    return int(value)


def _build_storage_power_metrics(
    *,
    weight_storage_mode: str,
    external_flash_bytes: int | None,
    appli_signed_image_path: str | None = None,
    fsbl_copy_window_bytes: int | None = None,
    appli_programmed: bool | None = None,
    weights_programmed: bool | None = None,
) -> dict[str, Any]:
    """Return storage-related telemetry fields for downstream metrics.

    Parameters
    ----------
    weight_storage_mode : str
        Effective weight-placement policy.
    external_flash_bytes : int | None
        External weight-blob size in bytes when present.

    Returns
    -------
    dict[str, Any]
        Storage-related telemetry fields merged into final device metrics.
    """
    return {
        "weight_storage_mode": str(weight_storage_mode),
        "external_flash_bytes": (
            -1.0 if external_flash_bytes is None else float(external_flash_bytes)
        ),
        "appli_signed_image_path": appli_signed_image_path,
        "fsbl_copy_window_bytes": (
            -1.0 if fsbl_copy_window_bytes is None else float(fsbl_copy_window_bytes)
        ),
        "appli_programmed": bool(appli_programmed) if appli_programmed is not None else False,
        "weights_programmed": bool(weights_programmed) if weights_programmed is not None else False,
    }


BOARD_DEFAULT_SPEC = build_stm32_nucleo_n657x0_q_spec()


class STM32NucleoN657X0QDevice(DeviceInterface):
    """Describe CREST-facing STM32 backend for the Nucleo N657x0 board.

    Notes
    -----
    The production backend supports compile, upload, back-to-back runtime
    measurement, harness-assisted energy capture, optional external-flash
    weight staging, and an optional dual-phase cadenced mode that runs a
    canonical back-to-back pass followed by a cadenced pass.
    """

    def __init__(
        self,
        *,
        serial_port: Optional[str] = None,
        device_options: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the STM32 backend.

        Parameters
        ----------
        serial_port : str | None, optional
            Serial port used for upload/runtime measurement when callers do not
            override it per request.
        device_options : dict[str, Any] | None, optional
            Raw STM32 backend options from config or request plumbing.
        """
        self._serial_port = serial_port
        self._options = resolve_stm32_nucleo_n657x0_q_options(device_options)
        self._spec = build_stm32_nucleo_n657x0_q_spec(self._options)

    @property
    def spec(self) -> DeviceSpec:
        """Return the STM32 board specification.

        Returns
        -------
        DeviceSpec
            Static STM32 board metadata.
        """
        return self._spec

    @property
    def resolved_options(self) -> STM32NucleoN657X0QOptions:
        """Expose the normalized STM32 backend options.

        Returns
        -------
        STM32NucleoN657X0QOptions
            Resolved STM32 backend options.
        """
        return self._options

    def _resolve_paths(self, project_root: Path | str) -> STM32WorkspacePaths:
        """Return validated layout-aware workspace paths.

        Parameters
        ----------
        project_root : Path | str
            STM32 project root containing the CubeIDE workspace to inspect.

        Returns
        -------
        STM32WorkspacePaths
            Validated layout-aware paths rooted at ``project_root``.
        """
        resolved_root = Path(project_root).expanduser().resolve()
        return _validate_project_structure(
            _resolve_workspace_paths(
                project_root=resolved_root,
            )
        )

    def requires_candidate_model(self) -> bool:
        """Return whether the STM backend stages candidate-specific model artifacts.

        Returns
        -------
        bool
            Always ``True`` for STM candidate staging.
        """
        return True

    def requires_training_data(self) -> bool:
        """Return whether candidate export needs training/calibration data.

        Returns
        -------
        bool
            Always ``True`` for STM candidate export.
        """
        return True

    def requires_arena_validation(self) -> bool:
        """Return whether the backend requires arena-sentinel validation.

        Returns
        -------
        bool
            Always ``False`` because STM uses a single-shot arena sentinel
            contract and surfaces the real arena size from generated headers.
        """
        return False

    def supports_energy_measurement(self) -> bool:
        """Return whether the backend can produce real energy metrics.

        Returns
        -------
        bool
            ``True`` because STM can produce harness-assisted energy metrics.
        """
        return True

    def supports_runtime_measurement(self) -> bool:
        """Return whether the backend can run HIL latency/energy passes.

        Returns
        -------
        bool
            ``True`` because direct-serial runtime measurement is available.
        """
        return True

    def _build_candidate_root(self, outputs_dir: Path, model_variant: str) -> Path:
        """Return a unique per-candidate staging root.

        Parameters
        ----------
        outputs_dir : pathlib.Path
            Root output directory owned by the current CREST run.
        model_variant : str
            Human-readable model variant name used to label the staging root.

        Returns
        -------
        pathlib.Path
            Unique per-candidate staging directory.
        """
        variant = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(model_variant).strip() or "candidate").strip("-")
        unique_suffix = uuid.uuid4().hex[:10]
        return outputs_dir.resolve() / "stm32" / f"{variant}-{unique_suffix}"

    def _read_storage_manifest(self, project_root: Path | STM32WorkspacePaths) -> dict[str, object]:
        """Read and validate the staged storage manifest.

        Parameters
        ----------
        project_root : pathlib.Path
            Staged STM32 workspace root.

        Returns
        -------
        dict[str, object]
            Parsed manifest payload.
        """
        paths = project_root if isinstance(project_root, STM32WorkspacePaths) else self._resolve_paths(project_root)
        return _read_staged_manifest(paths)

    def _storage_power_metrics(self, project_root: Path | STM32WorkspacePaths) -> dict[str, Any]:
        """Return storage metadata flattened for downstream metrics.

        Parameters
        ----------
        project_root : pathlib.Path
            Staged STM32 workspace root.

        Returns
        -------
        dict[str, Any]
            Weight storage mode plus optional external flash bytes.
        """
        try:
            manifest = self._read_storage_manifest(project_root)
        except stm32_cube_clt.WorkflowError:
            return _build_storage_power_metrics(
                weight_storage_mode=self._options.weight_storage_mode,
                external_flash_bytes=None,
            )
        return _build_storage_power_metrics(
            weight_storage_mode=_manifest_weight_storage_mode(manifest),
            external_flash_bytes=_manifest_external_flash_bytes(manifest),
            appli_signed_image_path=(
                str(manifest.get("appli_signed_image_path"))
                if manifest.get("appli_signed_image_path")
                else None
            ),
            fsbl_copy_window_bytes=(
                int(manifest["fsbl_copy_window_bytes"])
                if manifest.get("fsbl_copy_window_bytes") not in (None, "")
                else None
            ),
        )

    def _program_weight_blob_if_needed(
        self,
        project_root: Path | STM32WorkspacePaths,
        *,
        recover_first: bool = True,
    ) -> dict[str, Any]:
        """Program and verify staged external weights on every deployment.

        Parameters
        ----------
        project_root : pathlib.Path
            Staged STM32 workspace root.

        Returns
        -------
        dict[str, Any]
            Storage metadata to merge into final metrics.

        Raises
        ------
        WorkflowError
            If existing validation or execution checks fail.
        """
        paths = project_root if isinstance(project_root, STM32WorkspacePaths) else self._resolve_paths(project_root)
        try:
            manifest = self._read_storage_manifest(paths)
        except stm32_cube_clt.WorkflowError:
            return _build_storage_power_metrics(
                weight_storage_mode=self._options.weight_storage_mode,
                external_flash_bytes=None,
            )
        weight_storage_mode = _manifest_weight_storage_mode(manifest)
        external_flash_bytes = _manifest_external_flash_bytes(manifest)
        power_metrics = _build_storage_power_metrics(
            weight_storage_mode=weight_storage_mode,
            external_flash_bytes=external_flash_bytes,
            appli_signed_image_path=(
                str(manifest.get("appli_signed_image_path"))
                if manifest.get("appli_signed_image_path")
                else None
            ),
            fsbl_copy_window_bytes=(
                int(manifest["fsbl_copy_window_bytes"])
                if manifest.get("fsbl_copy_window_bytes") not in (None, "")
                else None
            ),
            weights_programmed=False,
        )
        if weight_storage_mode != "external_flash":
            return power_metrics
        blob_path = manifest.get("weights_blob_path")
        if not blob_path:
            raise stm32_cube_clt.WorkflowError(
                "STM staged manifest is missing weights_blob_path for external_flash mode."
            )
        if external_flash_bytes is None or external_flash_bytes <= 0:
            raise stm32_cube_clt.WorkflowError(
                "STM staged manifest is missing a valid weights_blob_size for external_flash mode."
            )
        if external_flash_bytes > self._spec.max_external_flash_bytes:
            raise stm32_cube_clt.WorkflowError(
                "STM external weight blob exceeds available external flash "
                f"({external_flash_bytes} > {self._spec.max_external_flash_bytes})."
            )
        current_blob_path = Path(str(blob_path)).expanduser().resolve()
        # Workspace history cannot establish the contents of a shared board.
        # Hash the bytes being sent, never a potentially stale manifest digest.
        current_blob_hash = _sha256_file(current_blob_path)
        actual_blob_bytes = current_blob_path.stat().st_size
        if actual_blob_bytes <= 0 or actual_blob_bytes > self._spec.max_external_flash_bytes:
            raise stm32_cube_clt.WorkflowError(
                f"STM external weight blob has invalid actual size ({actual_blob_bytes})."
            )
        current_flash_address = str(manifest.get("weights_flash_address") or self._options.weights_flash_address)
        loader_path = manifest.get("weights_external_loader")
        resolved_loader = (
            _resolve_weights_external_loader(
                self._options.cubeprog_bin,
                self._options.weights_external_loader,
            )
            if not loader_path
            else Path(str(loader_path)).expanduser().resolve()
        )
        program_log = stm32_cube_clt.program_external_flash_blob(
            cubeprog_bin=self._options.cubeprog_bin,
            apid=self._options.apid,
            weights_blob_path=current_blob_path,
            weights_flash_address=current_flash_address,
            external_loader=resolved_loader,
            recover_first=recover_first,
        )
        power_metrics["external_flash_program_log"] = program_log
        power_metrics["weights_programmed"] = True
        power_metrics["external_flash_bytes"] = actual_blob_bytes
        _update_staged_manifest(
            paths,
            weights_blob_sha256=current_blob_hash,
            weights_blob_size=actual_blob_bytes,
            last_programmed_weights_sha256=current_blob_hash,
            last_programmed_weights_flash_address=current_flash_address,
        )
        return power_metrics

    def _program_runtime_images(
        self,
        paths: STM32WorkspacePaths,
        *,
        compile_result: CompileResult,
    ) -> dict[str, Any]:
        """Program the LRUN signed app and optional weights before boot.

        Parameters
        ----------
        paths : STM32WorkspacePaths
            Resolved STM32 workspace paths for the staged candidate.
        compile_result : CompileResult
            Compile result carrying paths and size accounting for the staged build.

        Returns
        -------
        dict[str, Any]
            Programming metadata recorded for the staged runtime images.

        Raises
        ------
        WorkflowError
            If existing validation or execution checks fail.
        """
        signed_app_path = compile_result.signed_app_bin_path
        if signed_app_path is None:
            raise stm32_cube_clt.WorkflowError(
                "STM LRUN upload requires a signed_app_bin_path from the compile step."
            )
        manifest = self._read_storage_manifest(paths)
        # A different candidate/workspace may have programmed the board since
        # this manifest was written. Always program; CubeProgrammer uses -v.
        current_app_hash = _sha256_file(signed_app_path)
        appli_flash_address = str(manifest.get("appli_flash_address") or self._options.appli_flash_address)
        loader_path = manifest.get("weights_external_loader")
        resolved_loader = (
            _resolve_weights_external_loader(
                self._options.cubeprog_bin,
                self._options.weights_external_loader,
            )
            if not loader_path
            else Path(str(loader_path)).expanduser().resolve()
        )
        app_program_log = stm32_cube_clt.program_external_image(
            cubeprog_bin=self._options.cubeprog_bin,
            apid=self._options.apid,
            image_path=signed_app_path,
            flash_address=appli_flash_address,
            external_loader=resolved_loader,
            recover_first=True,
        )
        appli_programmed = True
        _update_staged_manifest(
            paths,
            signed_app_sha256=current_app_hash,
            last_programmed_appli_sha256=current_app_hash,
            last_programmed_appli_flash_address=appli_flash_address,
        )
        storage_metrics = self._program_weight_blob_if_needed(
            paths,
            recover_first=not appli_programmed,
        )
        merged = dict(storage_metrics)
        merged["appli_program_log"] = app_program_log
        merged["appli_programmed"] = appli_programmed
        return merged

    def prepare_candidate(
        self,
        *,
        request: CandidatePrepareRequest,
    ) -> Path:
        """Stage a candidate-specific STM32 workspace and return its root.

        Parameters
        ----------
        request : CandidatePrepareRequest
            Typed request describing the model, calibration data, and
            orchestration-selected artifact locations for one candidate.

        Returns
        -------
        pathlib.Path
            Per-candidate staged STM32 workspace root.

        Raises
        ------
        ValueError
            If the candidate model or training data is missing.
        crest.microcontrollers.stm32_cube_clt.WorkflowError
            If template validation, compatibility preflight, ST Edge AI
            generation, or staged-file validation fails.
        """
        if request.model is None:
            raise ValueError("STM32 candidate preparation requires a built Keras model.")
        if request.quantization_mode == "int8_ptq" and request.calibration_split is None:
            raise ValueError("STM32 candidate preparation requires calibration/training data for int8_ptq.")

        from crest.hardware import convert_to_tflite_model

        _ensure_staging_tools()
        canonical_paths = self._resolve_paths(self._options.project_root)
        _validate_memory_reservations(canonical_paths)
        resolved_external_loader: Path | None = None
        if self._options.weight_storage_mode == "external_flash":
            if (
                self._options.cubeprog_bin is None
                and stm32_cube_clt.default_cubeprog_bin() is None
            ):
                raise stm32_cube_clt.WorkflowError(
                    "STM32CubeProgrammer not found; required for external_flash weight storage mode."
                )
            if not self._options.weights_memory_pool.is_file():
                raise stm32_cube_clt.WorkflowError(
                    f"STM weights memory-pool JSON was not found: {self._options.weights_memory_pool}"
                )
            resolved_external_loader = _resolve_weights_external_loader(
                self._options.cubeprog_bin,
                self._options.weights_external_loader,
            )

        candidate_root = self._build_candidate_root(Path(request.artifact_root), request.model_variant)
        staged_dir_name = canonical_paths.root.name
        staged_project_root = candidate_root / staged_dir_name
        model_dir = candidate_root / "model"
        analyze_workspace_dir = candidate_root / "stedgeai_analyze_ws"
        analyze_output_dir = candidate_root / "stedgeai_analyze_out"
        stedgeai_workspace_dir = candidate_root / "stedgeai_ws"
        generated_output_dir = candidate_root / "stedgeai_out"
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(canonical_paths.root, staged_project_root)
            staged_paths = self._resolve_paths(staged_project_root)
            _strip_stale_debug_artifacts(staged_paths)
            staged_paths = self._resolve_paths(staged_project_root)
            _validate_memory_reservations(staged_paths)

            tflite_path = model_dir / "crest_candidate.tflite"
            convert_to_tflite_model(
                model=request.model,
                training_data=None if request.calibration_split is None else request.calibration_split.inputs,
                quantization_mode=request.quantization_mode,
                output_name=tflite_path,
            )
            _run_stedgeai_analyze(
                model_path=tflite_path,
                workspace_dir=analyze_workspace_dir,
                output_dir=analyze_output_dir,
            )
            _run_stedgeai_generate(
                model_path=tflite_path,
                workspace_dir=stedgeai_workspace_dir,
                output_dir=generated_output_dir,
                weight_storage_mode=self._options.weight_storage_mode,
                weights_flash_address=self._options.weights_flash_address,
                weights_memory_pool=self._options.weights_memory_pool,
            )
            _stage_generated_outputs(generated_output_dir, staged_paths)
            weights_blob_path = _generated_weights_blob_path(
                generated_output_dir,
                weight_storage_mode=self._options.weight_storage_mode,
            )
            _write_staged_manifest(
                paths=staged_paths,
                candidate_root=candidate_root,
                generated_output_dir=generated_output_dir,
                weight_storage_mode=self._options.weight_storage_mode,
                weights_blob_path=weights_blob_path,
                appli_signed_image_path=None,
                boot_elf_path=None,
                fsbl_copy_window_bytes=None,
                appli_flash_address=self._options.appli_flash_address,
                weights_flash_address=self._options.weights_flash_address,
                weights_external_loader=resolved_external_loader,
                extra_fields={
                    "weights_blob_sha256": (
                        _sha256_file(weights_blob_path) if weights_blob_path is not None and weights_blob_path.is_file() else None
                    ),
                    "last_programmed_appli_sha256": None,
                    "last_programmed_appli_flash_address": None,
                    "last_programmed_weights_sha256": None,
                    "last_programmed_weights_flash_address": None,
                },
            )
            _parse_arena_bytes(staged_paths.inc_dir / "network_data_params.h")
            config_device = getattr(request.config, "device", None)
            header_path = _write_phase_config_header(
                paths=staged_paths,
                cpu_clock_mhz=self._options.cpu_clock_mhz,
                selected_phase=self._options.runtime_mode,
                latency_budget_ms=self._options.latency_budget_ms,
                measured_runs=int(getattr(config_device, "measured_inference_runs", 10)),
                wake_margin_us=self._options.wake_margin_us,
                min_sleep_us=self._options.min_sleep_us,
            )
            _validate_phase_config_header(
                header_path,
                cpu_clock_mhz=self._options.cpu_clock_mhz,
                selected_phase=self._options.runtime_mode,
                measured_runs=int(getattr(config_device, "measured_inference_runs", 10)),
                latency_budget_ms=self._options.latency_budget_ms,
                wake_margin_us=self._options.wake_margin_us,
                min_sleep_us=self._options.min_sleep_us,
            )
            return staged_paths.root
        except Exception:
            if candidate_root.exists() and not _keep_staged_candidates_enabled():
                try:
                    shutil.rmtree(candidate_root)
                except OSError:
                    logger.warning(
                        "Failed to remove staged STM32 candidate root after prepare_candidate failure: %s",
                        candidate_root,
                        exc_info=True,
                    )
            raise

    def set_input_mode(
        self,
        input_mode: str,
        *,
        outputs_dir: Path,
        config: Any,
        sketches_dir: Path | None = None,
        runtime_phase: str = "back_to_back",
    ) -> Path | None:
        """Validate sTM32 ignores input-mode selection and returns the canonical template.

        Parameters
        ----------
        input_mode : str
            Requested input mode.
        outputs_dir : pathlib.Path
            Runtime outputs directory.
        config : Any
            Loaded CREST config.
        sketches_dir : pathlib.Path | None, optional
            Optional sketch-root override retained for interface compatibility.

        Returns
        -------
        pathlib.Path | None
            Canonical STM32 template root.
        """
        del input_mode, outputs_dir, config, sketches_dir, runtime_phase
        return self._options.project_root

    def cleanup_prepared_candidate(self, prepared_dir: Path | None) -> None:
        """Delete staged STM32 candidate roots once evaluation is complete.

        Parameters
        ----------
        prepared_dir : Path | None
            Path previously returned by ``prepare_candidate()``.
        """
        if prepared_dir is None:
            return
        if _keep_staged_candidates_enabled():
            logger.info(
                "Preserving staged STM32 candidate at %s because %s is enabled.",
                prepared_dir,
                KEEP_STAGED_CANDIDATES_ENV,
            )
            return

        project_root = Path(prepared_dir).expanduser().resolve()
        try:
            manifest_path = _manifest_path(self._resolve_paths(project_root))
        except stm32_cube_clt.WorkflowError:
            manifest_path = project_root / STAGED_MANIFEST_NAME
        candidate_root = project_root.parent
        if not manifest_path.is_file():
            return
        if not candidate_root.exists():
            return
        try:
            shutil.rmtree(candidate_root)
        except OSError:
            logger.warning("Failed to remove staged STM32 candidate root: %s", candidate_root, exc_info=True)

    def compile(
        self,
        *,
        sketch_path: Path,
        arena_kb: int,
        window_size: int,
        num_channels: int,
        build_defines: Optional[dict[str, int]] = None,
    ) -> CompileResult:
        """Build the staged STM32 project and parse compile-time diagnostics.

        Parameters
        ----------
        sketch_path : pathlib.Path
            Staged STM32 workspace root to compile.
        arena_kb : int
            Shared interface argument retained for compatibility.
        window_size : int
            Shared interface argument retained for compatibility.
        num_channels : int
            Shared interface argument retained for compatibility.
        build_defines : dict[str, int] | None, optional
            Shared interface argument retained for compatibility.

        Returns
        -------
        CompileResult
            Parsed build result including flash, RAM, activation arena, and
            linker heap/stack diagnostics when successful.

        Raises
        ------
        WorkflowError
            If existing validation or execution checks fail.
        """
        del arena_kb, window_size, num_channels, build_defines
        project_root = Path(sketch_path).expanduser().resolve()
        external_flash_bytes: int | None = None
        weight_storage_mode = self._options.weight_storage_mode
        build_dir_fallback = project_root / "Debug"
        try:
            paths = self._resolve_paths(project_root)
            build_dir_fallback = paths.boot_debug_dir
            heap_bytes, stack_bytes = _validate_memory_reservations(paths)
            _validate_lrun_boot_include_path(paths)
            try:
                manifest = _read_staged_manifest(paths)
            except stm32_cube_clt.WorkflowError:
                manifest = None
            if manifest is not None:
                external_flash_bytes = _manifest_external_flash_bytes(manifest)
                weight_storage_mode = _manifest_weight_storage_mode(manifest)
            if external_flash_bytes is not None and external_flash_bytes > self._spec.max_external_flash_bytes:
                raise stm32_cube_clt.WorkflowError(
                    "STM external weight blob exceeds available external flash "
                    f"({external_flash_bytes} > {self._spec.max_external_flash_bytes})."
                )

            jobs = os.cpu_count() or 1
            candidate_root = (
                Path(str(manifest.get("candidate_root")))
                if manifest is not None and manifest.get("candidate_root")
                else paths.root.parent
            )
            generated_output_dir = (
                Path(str(manifest.get("generated_output_dir")))
                if manifest is not None and manifest.get("generated_output_dir")
                else paths.root
            )
            app_build_input_hash = _lrun_app_build_input_hash(paths)
            app_build_log = ""
            app_elf_path = paths.app_elf_path
            app_bin: Path | None = None
            if (
                manifest is not None
                and manifest.get("app_build_input_hash") == app_build_input_hash
                and app_elf_path is not None
                and app_elf_path.is_file()
            ):
                try:
                    app_bin = _resolve_bin_artifact(app_elf_path)
                    app_build_log = (
                        "STM LRUN AppS build reused from staged workspace; App inputs are unchanged."
                    )
                except stm32_cube_clt.WorkflowError:
                    app_bin = None
            if app_bin is None:
                app_build = stm32_cube_clt.build_project(
                    project_root=paths.app_project_root,
                    jobs=jobs,
                    clean=True,
                )
                app_elf_path = app_build.elf_path
                app_build_log = app_build.log.strip()
                app_bin = _resolve_bin_artifact(app_build.elf_path)
            assert app_elf_path is not None
            app_size = stm32_cube_clt.parse_size_output(app_elf_path)
            signed_app_path = paths.signed_app_bin_path or app_bin.with_name(
                f"{app_bin.stem}-trusted{app_bin.suffix}"
            )
            app_sign_input_hash = _lrun_app_sign_input_hash(
                app_bin,
                signing_load_offset=self._options.signing_load_offset,
                signing_header_version=self._options.signing_header_version,
            )
            signing_log = ""
            if (
                manifest is not None
                and manifest.get("app_sign_input_hash") == app_sign_input_hash
                and signed_app_path.is_file()
            ):
                signed_output_bin = signed_app_path
                signing_log = (
                    "STM LRUN signed App reused from staged workspace; signing inputs are unchanged."
                )
            else:
                signing_result = stm32_cube_clt.sign_binary(
                    signing_tool=self._options.signing_tool,
                    input_bin=app_bin,
                    output_bin=signed_app_path,
                    load_offset=self._options.signing_load_offset,
                    header_version=self._options.signing_header_version,
                )
                signed_output_bin = signing_result.output_bin
                signing_log = signing_result.log.strip()
            trusted_app_size = signed_output_bin.stat().st_size
            if trusted_app_size > self._spec.max_flash_bytes:
                raise stm32_cube_clt.WorkflowError(
                    "STM trusted App image exceeds available LRUN code-image budget "
                    f"({trusted_app_size} > {self._spec.max_flash_bytes})."
                )
            _header_path, copy_window_changed, copy_window_bytes = _update_lrun_copy_window(
                paths=paths,
                trusted_app_size=trusted_app_size,
            )
            boot_build_input_hash = _lrun_boot_build_input_hash(paths)
            boot_build_log = ""
            boot_elf_path = paths.boot_elf_path
            if (
                manifest is not None
                and manifest.get("boot_build_input_hash") == boot_build_input_hash
                and boot_elf_path is not None
                and boot_elf_path.is_file()
            ):
                boot_build_log = (
                    "STM LRUN Boot build reused from staged workspace; Boot inputs are unchanged."
                )
            else:
                boot_build = stm32_cube_clt.build_project(
                    project_root=paths.boot_project_root,
                    jobs=jobs,
                    clean=True,
                )
                boot_elf_path = boot_build.elf_path
                boot_build_log = boot_build.log.strip()
            assert boot_elf_path is not None
            boot_size = stm32_cube_clt.parse_size_output(boot_elf_path)
            flash_bytes = trusted_app_size
            arena_bytes = _parse_arena_bytes(paths.inc_dir / "network_data_params.h")
            _validate_lrun_flash_layout(
                copy_window_bytes=copy_window_bytes,
                appli_flash_address=self._options.appli_flash_address,
                weights_flash_address=self._options.weights_flash_address,
            )
            _write_staged_manifest(
                paths=paths,
                candidate_root=candidate_root,
                generated_output_dir=generated_output_dir,
                weight_storage_mode=weight_storage_mode,
                weights_blob_path=(
                    Path(str(manifest.get("weights_blob_path")))
                    if manifest is not None and manifest.get("weights_blob_path")
                    else None
                ),
                appli_signed_image_path=signed_output_bin,
                boot_elf_path=boot_elf_path,
                fsbl_copy_window_bytes=copy_window_bytes,
                appli_flash_address=self._options.appli_flash_address,
                weights_flash_address=self._options.weights_flash_address,
                weights_external_loader=(
                    Path(str(manifest.get("weights_external_loader"))).expanduser().resolve()
                    if manifest is not None and manifest.get("weights_external_loader")
                    else self._options.weights_external_loader
                ),
                existing_manifest=manifest,
                extra_fields={
                    "app_build_input_hash": app_build_input_hash,
                    "app_sign_input_hash": app_sign_input_hash,
                    "signed_app_sha256": _sha256_file(signed_output_bin),
                    "boot_build_input_hash": boot_build_input_hash,
                    "weights_blob_sha256": (
                        _sha256_file(Path(str(manifest["weights_blob_path"])).expanduser().resolve())
                        if manifest is not None and manifest.get("weights_blob_path")
                        else None
                    ),
                },
            )
            diagnostics = (
                f"stm32_arena_bytes={arena_bytes}\n"
                f"stm32_heap_bytes={heap_bytes}\n"
                f"stm32_stack_bytes={stack_bytes}\n"
                f"stm32_weight_storage_mode={weight_storage_mode}\n"
                f"stm32_external_flash_bytes={external_flash_bytes if external_flash_bytes is not None else -1}\n"
                f"stm32_appli_flash_address={self._options.appli_flash_address}\n"
                f"stm32_fsbl_copy_window_bytes={copy_window_bytes}"
            )
            return CompileResult(
                success=True,
                log="\n".join(
                    text
                    for text in (
                        app_build_log,
                        app_size.raw_output.strip(),
                        signing_log,
                        boot_build_log,
                        boot_size.raw_output.strip(),
                        diagnostics,
                    )
                    if text
                ),
                flash_bytes=flash_bytes,
                ram_bytes=app_size.ram_bytes,
                overflow_kind=None,
                build_dir=paths.boot_debug_dir,
                boot_build_dir=paths.boot_debug_dir,
                boot_elf_path=boot_elf_path,
                app_build_dir=paths.app_debug_dir,
                app_elf_path=app_elf_path,
                signed_app_bin_path=signed_output_bin,
                fsbl_copy_window_bytes=copy_window_bytes,
                arena_bytes=arena_bytes,
                heap_bytes=heap_bytes,
                stack_bytes=stack_bytes,
                external_flash_bytes=external_flash_bytes,
            )
        except stm32_cube_clt.WorkflowError as exc:
            log_text = str(exc)
            return CompileResult(
                success=False,
                log=log_text,
                flash_bytes=None,
                ram_bytes=None,
                overflow_kind=stm32_cube_clt.classify_build_failure(log_text),
                build_dir=build_dir_fallback,
                external_flash_bytes=external_flash_bytes,
            )

    def upload(
        self,
        *,
        sketch_path: Path,
        build_dir: Optional[Path],
        serial_port: Optional[str],
    ) -> UploadResult:
        """Debug-load the staged STM32 ELF through ST-LINK.

        Parameters
        ----------
        sketch_path : pathlib.Path
            Shared interface argument retained for compatibility.
        build_dir : pathlib.Path | None
            Staged ``Debug`` directory produced by the compile step.
        serial_port : str | None
            Shared interface argument retained for compatibility.

        Returns
        -------
        UploadResult
            Result of the ST-LINK debug-load workflow.
        """
        project_root = Path(sketch_path).expanduser().resolve()
        del serial_port
        if build_dir is None:
            return UploadResult(
                success=False,
                log="STM32 upload requires a staged build directory from the preceding compile step.",
            )
        try:
            paths = self._resolve_paths(project_root)
            manifest = self._read_storage_manifest(paths)
            signed_app_path = None
            if manifest.get("appli_signed_image_path"):
                signed_app_path = Path(str(manifest["appli_signed_image_path"])).expanduser().resolve()
            boot_elf_path = None
            if manifest.get("boot_elf_path"):
                boot_elf_path = stm32_cube_clt.resolve_required_file_path(
                    manifest.get("boot_elf_path"),
                    label="STM32 Boot ELF",
                )
            storage_metrics = self._program_runtime_images(
                paths,
                compile_result=CompileResult(
                    success=True,
                    log="",
                    flash_bytes=None,
                    ram_bytes=None,
                    overflow_kind=None,
                    build_dir=build_dir,
                    boot_build_dir=build_dir,
                    signed_app_bin_path=signed_app_path,
                    boot_elf_path=boot_elf_path,
                ),
            )
            elf_path = boot_elf_path if boot_elf_path is not None else stm32_cube_clt.resolve_elf_path(build_dir)
            log = "\n".join(
                text
                for text in (
                    storage_metrics.get("appli_program_log"),
                    storage_metrics.get("external_flash_program_log"),
                )
                if text
            )
            load_log = stm32_cube_clt.debug_load_elf(
                elf_path=elf_path,
                gdbserver=self._options.gdbserver,
                gdb=self._options.gdb,
                cubeprog_bin=self._options.cubeprog_bin,
                gdb_port=self._options.gdb_port,
                apid=self._options.apid,
                server_ready_timeout_s=self._options.server_ready_timeout_s,
                run_after_load=True,
            )
            storage_lines = [
                f"{key}={value}"
                for key, value in storage_metrics.items()
                if key in {"weight_storage_mode", "external_flash_bytes", "appli_signed_image_path", "fsbl_copy_window_bytes"}
            ]
            return UploadResult(
                success=True,
                log="\n".join(
                    text for text in (log, load_log, "\n".join(storage_lines)) if text
                ),
            )
        except stm32_cube_clt.WorkflowError as exc:
            return UploadResult(success=False, log=str(exc))

    def measure(
        self,
        *,
        serial_port: Optional[str],
        baud_rate: int,
        serial_timeout_s: float,
        measured_inference_runs: int | None = None,
        dut_ready_timeout_s: Optional[float] = None,
        harness_serial_port: Optional[str] = None,
        harness_fqbn: Optional[str] = None,
        harness_auto_flash: Optional[str] = None,
        harness_arm_pin: Optional[int] = None,
        harness_trigger_pin: Optional[int] = None,
        dut_arm_hold_ms: Optional[int] = None,
        harness_stable_low_ms: Optional[int] = None,
        harness_ready_timeout_s: Optional[float] = None,
        harness_arm_timeout_s: Optional[float] = None,
        harness_active_timeout_s: Optional[float] = None,
        harness_done_timeout_s: Optional[float] = None,
    ) -> MeasureResult:
        """Capture direct-serial runtime telemetry from an already-loaded DUT.

        Parameters
        ----------
        serial_port : str | None
            Serial port associated with the target.
        baud_rate : int
            Serial baud rate.
        serial_timeout_s : float
            Post-``START`` runtime timeout in seconds.
        dut_ready_timeout_s : float | None, optional
            Pre-``START`` timeout for init and ready tokens.
        harness_serial_port : str | None, optional
            Retained for interface compatibility.
        harness_fqbn : str | None, optional
            Retained for interface compatibility.
        harness_auto_flash : str | None, optional
            Retained for interface compatibility.
        harness_arm_pin : int | None, optional
            Retained for interface compatibility.
        harness_trigger_pin : int | None, optional
            Retained for interface compatibility.
        dut_arm_hold_ms : int | None, optional
            Retained for interface compatibility.
        harness_stable_low_ms : int | None, optional
            Retained for interface compatibility.
        harness_ready_timeout_s : float | None, optional
            Retained for interface compatibility.
        harness_arm_timeout_s : float | None, optional
            Retained for interface compatibility.
        harness_active_timeout_s : float | None, optional
            Retained for interface compatibility.
        harness_done_timeout_s : float | None, optional
            Retained for interface compatibility.

        Returns
        -------
        MeasureResult
            Parsed latency, serial log, and backend error details when present.

        Raises
        ------
        ValueError
            If existing validation or execution checks fail.
        """
        del (
            measured_inference_runs,
            harness_serial_port,
            harness_fqbn,
            harness_auto_flash,
            harness_arm_pin,
            harness_trigger_pin,
            dut_arm_hold_ms,
            harness_stable_low_ms,
            harness_ready_timeout_s,
            harness_arm_timeout_s,
            harness_active_timeout_s,
            harness_done_timeout_s,
        )
        use_serial_port = self._serial_port if serial_port is None else serial_port
        if use_serial_port is None:
            raise ValueError("serial_port must be provided for STM32 runtime measurement.")
        try:
            telemetry = stm32_runtime.measure_direct_serial(
                serial_port=use_serial_port,
                baud_rate=baud_rate,
                boot_timeout_s=dut_ready_timeout_s,
                run_timeout_s=serial_timeout_s,
            )
        except stm32_runtime.STM32RuntimeProtocolError as exc:
            return MeasureResult(
                latency_s=None,
                arena_error_line=None,
                serial_log=exc.serial_log,
                power_metrics=stm32_runtime.build_backend_error_metrics(exc.kind, exc.detail),
            )
        except (serial.SerialException, RuntimeError) as exc:
            return MeasureResult(
                latency_s=None,
                arena_error_line=None,
                serial_log=[],
                power_metrics=stm32_runtime.build_backend_error_metrics("runtime_io", str(exc)),
            )
        return MeasureResult(
            latency_s=telemetry.latency_s,
            arena_error_line=None,
            serial_log=telemetry.serial_log,
            power_metrics=telemetry.power_metrics,
        )

    def _evaluate_single_phase(self, **kwargs) -> DeviceMetrics:
        """Measure a phase, allowing one integrity retry of the same build.

        Parameters
        ----------
        **kwargs : dict
            Phase and candidate arguments forwarded to `_evaluate_phase_attempt`.

        Returns
        -------
        DeviceMetrics
            Final attempt metrics with record paths from both attempts when retried.

        Raises
        ------
        Exception
            If phase execution or durable evidence persistence raises an exception.
        """
        retry_state = {}
        first = self._evaluate_phase_attempt(**kwargs, _retry_state=retry_state)
        compile_result = retry_state.get("compile_result")
        if first.error_code != HIL_ERROR_INTEGRITY or compile_result is None:
            return first
        second = self._evaluate_phase_attempt(
            **kwargs, _retry_state=retry_state, _reuse_compile_result=compile_result)
        metrics = dict(second.power_metrics or {})
        metrics["measurement_record_paths"] = (list((first.power_metrics or {}).get("measurement_record_paths", []))
                                               + list(metrics.get("measurement_record_paths", [])))
        metrics["measurement_remeasure_count"] = 1
        return replace(second, power_metrics=metrics)

    def _evaluate_phase_attempt(self, *, _retry_state: dict | None = None, **kwargs) -> DeviceMetrics:
        """Measure and validate one phase while persisting durable raw evidence.

        Parameters
        ----------
        _retry_state : dict | None, optional
            State local to one phase, carrying the build into a bounded retry.
        **kwargs : dict
            Candidate and runtime arguments forwarded to `_evaluate_single_phase_unchecked`.
            With HIL disabled, execution bypasses measurement evidence and acceptance.

        Returns
        -------
        DeviceMetrics
            Phase metrics augmented with acceptance flags and a durable record path.

        Raises
        ------
        OSError
            If the evidence directory or record cannot be written and no earlier
            execution exception is already being propagated.
        Exception
            If the underlying phase execution fails; its evidence is retained.
        """
        if not kwargs.get("run_hil", True):
            return self._evaluate_single_phase_unchecked(**kwargs, _retry_state=_retry_state)
        record = {"started_utc": datetime.now(timezone.utc).isoformat(), "request": {key: str(value) if isinstance(value, Path) else value
                              for key, value in kwargs.items()},
                  "dut_log": [], "harness_log": []}
        record_path = (self._options.measurement_records_dir or (Path.cwd() / "outputs" / "hil_measurements")) / f"{uuid.uuid4().hex}.json"
        # Fail before touching hardware if the durable evidence directory is unavailable.
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("x") as handle:
            json.dump(record, handle, indent=2, default=str)
        result = None
        try:
            result = self._evaluate_single_phase_unchecked(**kwargs, _record=record, _retry_state=_retry_state)
            metrics = dict(result.power_metrics or {})
            if result.error_code == HIL_ERROR_OK:
                if kwargs.get("phase", DEFAULT_RUNTIME_MODE) == "back_to_back" and not kwargs.get("harness_serial_port"):
                    rtc_window = metrics.get("timer_per_window_s")
                    if positive(rtc_window):
                        result = replace(result, latency_s=rtc_window / record["expected_runs"])
                acceptance = validate_measurement(
                    metrics, latency_s=result.latency_s,
                    phase=kwargs.get("phase", DEFAULT_RUNTIME_MODE),
                    clock_mhz=self._options.cpu_clock_mhz,
                    runs=record["expected_runs"], period_ms=self._options.latency_budget_ms,
                    dut_log=record["dut_log"], harness_log=record["harness_log"],
                    require_harness=bool(kwargs.get("harness_serial_port")))
                record["acceptance"] = acceptance
                metrics.update(measurement_instrumentation_valid=acceptance["instrumentation_valid"],
                               measurement_deployment_eligible=acceptance["deployment_eligible"],
                               measurement_infeasibility_reasons=acceptance["infeasibility_reasons"],
                               measurement_instrumentation_reasons=acceptance["instrumentation_reasons"])
                if not acceptance["instrumentation_valid"]:
                    metrics.update(stm32_runtime.build_backend_error_metrics(
                        "measurement_integrity", ", ".join(acceptance["instrumentation_reasons"])))
                    result = replace(result, error_code=HIL_ERROR_INTEGRITY, latency_s=-1.0)
            else:
                record["acceptance"] = {"instrumentation_valid": False,
                                        "instrumentation_reasons": [metrics.get("backend_error_kind", "backend_failure")]}
            metrics.setdefault("measurement_instrumentation_valid", False)
            metrics.setdefault("measurement_deployment_eligible", False)
            metrics.setdefault("measurement_instrumentation_reasons", record["acceptance"]["instrumentation_reasons"])
            metrics["measurement_record_path"] = str(record_path)
            metrics["measurement_record_paths"] = [str(record_path)]
            result = replace(result, power_metrics=metrics)
            record["result"] = asdict(result)
            return result
        except Exception as exc:
            record["exception"] = {"type": type(exc).__name__, "detail": str(exc)}
            raise
        finally:
            # Unique names prevent a retry/candidate from overwriting prior evidence.
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            active_exception = sys.exc_info()[1]
            try:
                with record_path.open("w") as handle:
                    json.dump(record, handle, indent=2, default=str)
            except OSError:
                logger.exception("Failed to finalize measurement record %s", record_path)
                if active_exception is None:
                    raise
            self.measurement_records = getattr(self, "measurement_records", []) + [str(record_path)]

    def _evaluate_single_phase_unchecked(
        self,
        *,
        dirpath: str | Path,
        phase: str = DEFAULT_RUNTIME_MODE,
        arena_kb: int,
        window_size: int,
        num_channels: int,
        serial_port: Optional[str] = None,
        run_hil: bool = True,
        baud_rate: int = 115200,
        serial_timeout_s: float = 12.0,
        measured_inference_runs: int | None = None,
        dut_ready_timeout_s: Optional[float] = None,
        harness_serial_port: Optional[str] = None,
        harness_fqbn: Optional[str] = None,
        harness_auto_flash: Optional[str] = None,
        harness_arm_pin: Optional[int] = None,
        harness_trigger_pin: Optional[int] = None,
        dut_arm_hold_ms: Optional[int] = None,
        harness_stable_low_ms: Optional[int] = None,
        harness_ready_timeout_s: Optional[float] = None,
        harness_arm_timeout_s: Optional[float] = None,
        harness_active_timeout_s: Optional[float] = None,
        harness_done_timeout_s: Optional[float] = None,
        _record: dict | None = None,
        _retry_state: dict | None = None,
        _reuse_compile_result: Any = None,
    ) -> DeviceMetrics:
        """Run one STM32 compile/upload/runtime phase and return metrics.

        Parameters
        ----------
        dirpath : str | pathlib.Path
            Staged STM32 workspace root to compile.
        phase : str, default="back_to_back"
            Runtime phase compiled and executed for this pass.
        arena_kb : int
            Shared interface argument retained for compatibility.
        window_size : int
            Shared interface argument retained for compatibility.
        num_channels : int
            Shared interface argument retained for compatibility.
        serial_port : str | None, optional
            Serial port retained for interface compatibility.
        run_hil : bool, default=True
            Whether the caller requested runtime measurement.
        baud_rate : int, default=115200
            Serial baud rate retained for interface compatibility.
        serial_timeout_s : float, default=12.0
            Serial timeout retained for interface compatibility.
        measured_inference_runs : int | None, optional
            Number of measured invocations; defaults to the device setting.
        dut_ready_timeout_s : float | None, optional
            DUT-ready timeout retained for interface compatibility.
        harness_serial_port : str | None, optional
            Harness serial port retained for interface compatibility.
        harness_fqbn : str | None, optional
            Harness FQBN retained for interface compatibility.
        harness_auto_flash : str | None, optional
            Harness flashing policy retained for interface compatibility.
        harness_arm_pin : int | None, optional
            Harness arm pin retained for interface compatibility.
        harness_trigger_pin : int | None, optional
            Harness trigger pin retained for interface compatibility.
        dut_arm_hold_ms : int | None, optional
            DUT arm hold retained for interface compatibility.
        harness_stable_low_ms : int | None, optional
            Stable-low period retained for interface compatibility.
        harness_ready_timeout_s : float | None, optional
            Harness ready timeout retained for interface compatibility.
        harness_arm_timeout_s : float | None, optional
            Harness arm timeout retained for interface compatibility.
        harness_active_timeout_s : float | None, optional
            Harness active timeout retained for interface compatibility.
        harness_done_timeout_s : float | None, optional
            Harness done timeout retained for interface compatibility.

        _record : dict | None, optional
            Mutable evidence record populated with raw logs and expected run count.
        _retry_state : dict | None, optional
            Mutable state local to the phase, populated with its compile result.
        _reuse_compile_result : Any, optional
            Existing compile result reused for a bounded instrumentation retry.

        Returns
        -------
        DeviceMetrics
            Normalized STM32 metrics with compile-time sizes and optional runtime
            latency when HIL is enabled.

        Raises
        ------
        Exception
            If existing validation or execution checks fail.
        ValueError
            If existing validation or execution checks fail.
        """
        del arena_kb
        _record = {} if _record is None else _record
        project_root = Path(dirpath).expanduser().resolve()
        paths: STM32WorkspacePaths | None = None
        normalized_phase = _resolve_runtime_mode(phase)
        try:
            paths = self._resolve_paths(project_root)
        except stm32_cube_clt.WorkflowError:
            if project_root.exists():
                raise
        if measured_inference_runs is None:
            if paths is None:
                effective_measured_runs = 10
            else:
                try:
                    effective_measured_runs = _read_phase_config_measured_runs(paths)
                except stm32_cube_clt.WorkflowError:
                    effective_measured_runs = 10
        else:
            effective_measured_runs = max(1, int(measured_inference_runs))
        _record["expected_runs"] = effective_measured_runs
        if paths is not None:
            self._write_runtime_phase_config(
                paths=paths,
                selected_phase=phase,
                measured_runs=effective_measured_runs,
            )
        try:
            storage_metrics = self._storage_power_metrics(paths if paths is not None else project_root)
        except stm32_cube_clt.WorkflowError:
            storage_metrics = _build_storage_power_metrics(
                weight_storage_mode=self._options.weight_storage_mode,
                external_flash_bytes=None,
            )

        compile_result = _reuse_compile_result or self.compile(
            sketch_path=project_root if paths is None else paths.root,
            arena_kb=-1,
            window_size=window_size,
            num_channels=num_channels,
            build_defines=None,
        )
        if _retry_state is not None:
            _retry_state["compile_result"] = compile_result
        compile_external_flash_bytes = getattr(compile_result, "external_flash_bytes", None)

        def _merge_metrics(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
            """Merge storage metrics with optional runtime/backend detail fields.

            Parameters
            ----------
            extra : Mapping[str, Any] | None, optional
                Additional metrics to merge over the storage baseline.

            Returns
            -------
            dict[str, Any]
                Combined metrics payload.
            """
            merged = dict(storage_metrics)
            if extra:
                merged.update(extra)
            return merged

        def _latency_failure(kind: str, detail: str) -> DeviceMetrics:
            """Build a normalized latency-failure result for the current candidate.

            Parameters
            ----------
            kind : str
                Backend-specific latency failure category.
            detail : str
                Human-readable failure detail.

            Returns
            -------
            DeviceMetrics
                Failure metrics tagged as measurement infrastructure or candidate latency.
            """
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=compile_result.arena_bytes or -1,
                error_code=(HIL_ERROR_INTEGRITY if kind in MEASUREMENT_INTEGRITY_KINDS else HIL_ERROR_LATENCY),
                power_metrics=_merge_metrics(
                    stm32_runtime.build_backend_error_metrics(kind, detail)
                ),
                external_flash_bytes=compile_external_flash_bytes,
            )

        def _upload_failure(kind: str, detail: str) -> DeviceMetrics:
            """Build a normalized upload-failure result for the current candidate.

            Parameters
            ----------
            kind : str
                Backend-specific upload failure category.
            detail : str
                Human-readable failure detail.

            Returns
            -------
            DeviceMetrics
                Failure metrics tagged with ``HIL_ERROR_UPLOAD``.
            """
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=compile_result.arena_bytes or -1,
                error_code=HIL_ERROR_UPLOAD,
                power_metrics=_merge_metrics(
                    stm32_runtime.build_backend_error_metrics(kind, detail)
                ),
                external_flash_bytes=compile_external_flash_bytes,
            )

        overflow_kind = compile_result.overflow_kind
        if overflow_kind is not None:
            error_code = (
                HIL_ERROR_FLASH_OVERFLOW if overflow_kind == "flash" else HIL_ERROR_RAM_OVERFLOW
            )
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=compile_result.arena_bytes or -1,
                error_code=error_code,
                power_metrics=_merge_metrics(),
                external_flash_bytes=compile_external_flash_bytes,
            )
        if not compile_result.success:
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=compile_result.arena_bytes or -1,
                error_code=HIL_ERROR_COMPILE,
                power_metrics=_merge_metrics(),
                external_flash_bytes=compile_external_flash_bytes,
            )
        if not run_hil:
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=compile_result.arena_bytes or -1,
                error_code=HIL_ERROR_OK,
                power_metrics=_merge_metrics(),
                external_flash_bytes=compile_external_flash_bytes,
            )
        use_serial_port = self._serial_port if serial_port is None else serial_port
        if use_serial_port is None:
            raise ValueError("serial_port must be provided when running STM32 HIL uploads.")
        try:
            elf_path = (
                getattr(compile_result, "boot_elf_path", None)
                if getattr(compile_result, "boot_elf_path", None) is not None
                else stm32_cube_clt.resolve_elf_path(compile_result.build_dir)
            )
        except stm32_cube_clt.WorkflowError as exc:
            return _upload_failure("upload", str(exc))
        harness_enabled = bool(harness_serial_port)
        if harness_enabled:
            try:
                arduino_base.ensure_harness_firmware(
                    harness_serial_port=str(harness_serial_port),
                    harness_fqbn="arduino:mbed_nano:nano33ble"
                    if harness_fqbn is None
                    else str(harness_fqbn),
                    harness_auto_flash="once"
                    if harness_auto_flash is None
                    else str(harness_auto_flash),
                    build_defines={
                        "CREST_HARNESS_ARM_PIN": 3 if harness_arm_pin is None else int(harness_arm_pin),
                        "CREST_HARNESS_TRIGGER_PIN": 2
                        if harness_trigger_pin is None
                        else int(harness_trigger_pin),
                        "CREST_DUT_ARM_HOLD_MS": 600 if dut_arm_hold_ms is None else int(dut_arm_hold_ms),
                        "CREST_HARNESS_STABLE_LOW_MS": 500
                        if harness_stable_low_ms is None
                        else int(harness_stable_low_ms),
                        "CREST_HARNESS_ARM_TIMEOUT_MS": max(
                            0,
                            int(round((5.0 if harness_arm_timeout_s is None else float(harness_arm_timeout_s)) * 1000.0)),
                        ),
                        "CREST_HARNESS_ACTIVE_TIMEOUT_MS": max(
                            0,
                            int(round((30.0 if harness_active_timeout_s is None else float(harness_active_timeout_s)) * 1000.0)),
                        ),
                        "CREST_INFERENCE_RUNS": effective_measured_runs,
                    },
                )
            except (RuntimeError, stm32_cube_clt.WorkflowError) as exc:
                return _upload_failure("harness_prepare", str(exc))
        monitor = None
        try:
            # Canonical combined ordering:
            # 1. build
            # 2. program runtime images (signed App + optional weights)
            # 3. open DUT serial
            # 4. open harness serial
            # 5. prime harness / send PING
            # 6. ELF load / resume
            # 7. wait for DUT READY
            # 8. send DUT START
            # 9. wait for harness DONE
            runtime_storage_metrics = (
                self._program_runtime_images(paths, compile_result=compile_result)
                if paths is not None
                else self._storage_power_metrics(project_root)
            )
            with stm32_runtime.SerialMonitor(use_serial_port, baud_rate, "dut") as monitor:
                if harness_enabled:
                    harness_log: list[str] = []
                    _record["harness_log"] = harness_log
                    with serial.Serial(
                        str(harness_serial_port),
                        baudrate=baud_rate,
                        timeout=0.1,
                    ) as harness:
                        prime_result = hil_protocol.prime_harness_session(
                            harness=harness,
                            harness_ready_timeout_s=(
                                5.0 if harness_ready_timeout_s is None else float(harness_ready_timeout_s)
                            ),
                            harness_log=harness_log,
                            flush_input=True,
                        )
                        if not prime_result.harness_ready:
                            return _latency_failure(
                                "harness_ready_timeout",
                                "Timed out waiting for HARNESS READY.",
                            )
                        stm32_cube_clt.debug_load_elf(
                            elf_path=elf_path,
                            gdbserver=self._options.gdbserver,
                            gdb=self._options.gdb,
                            cubeprog_bin=self._options.cubeprog_bin,
                            gdb_port=self._options.gdb_port,
                            apid=self._options.apid,
                            server_ready_timeout_s=self._options.server_ready_timeout_s,
                            run_after_load=True,
                        )
                        telemetry = stm32_runtime.execute_runtime_session(
                            monitor,
                            boot_timeout_s=(
                                DEFAULT_LRUN_BOOT_TIMEOUT_S
                                if dut_ready_timeout_s is None
                                else (
                                    stm32_runtime.DEFAULT_BOOT_TIMEOUT_S
                                    if dut_ready_timeout_s is None
                                    else float(dut_ready_timeout_s)
                                )
                            ),
                            run_timeout_s=float(serial_timeout_s),
                        )
                        _record["dut_log"] = list(telemetry.serial_log)
                        _record["harness_log"] = prime_result.harness_log
                        harness_result = hil_protocol.wait_for_harness_done(
                            harness=harness,
                            harness_active_timeout_s=(
                                30.0 if harness_active_timeout_s is None else float(harness_active_timeout_s)
                            ),
                            harness_done_timeout_s=(
                                5.0 if harness_done_timeout_s is None else float(harness_done_timeout_s)
                            ),
                            harness_log=prime_result.harness_log,
                        )
                    _record["harness_log"] = list(harness_result.harness_log)
                    if not harness_result.harness_done:
                        return _latency_failure(
                            "harness_done_timeout",
                            "Timed out waiting for harness DONE.",
                        )
                    harness_power = arduino_base._parse_power_metrics(harness_result.harness_log)
                    if harness_power is None:
                        return _latency_failure(
                            "harness_metrics_missing",
                            "Harness completed without reporting energy telemetry.",
                        )
                    dut_runs = int(telemetry.power_metrics.get("runs", -1))
                    if harness_result.runs_harness is None or harness_result.runs_harness != dut_runs:
                        return _latency_failure(
                            "harness_run_count_mismatch",
                            f"Run-count mismatch: dut={dut_runs} harness={harness_result.runs_harness}",
                        )
                    merged_power_metrics = arduino_base._merge_power_metrics(
                        primary=harness_power,
                        secondary=telemetry.power_metrics,
                    ) or dict(telemetry.power_metrics)
                    merged_power_metrics["phase"] = telemetry.power_metrics.get("phase", "back_to_back")
                    merged_power_metrics["runs"] = telemetry.power_metrics.get("runs", dut_runs)
                    for timer_field in (
                        "timer_output_s",
                        "timer_per_inference_s",
                        "timer_per_window_s",
                    ):
                        timer_value = telemetry.power_metrics.get(timer_field)
                        if timer_value is not None:
                            merged_power_metrics[timer_field] = timer_value
                    # Preserve STM32-specific cadence telemetry that the generic
                    # harness merge helper does not know about.
                    for passthrough_field in (
                        "slept_slots",
                        "stop_entries",
                        "wake_recovery_us",
                        "wake_overshoot_us",
                        "rtc_sleep_total_ms",
                        "deadline_miss_count",
                        "rtc_clock_hz_nominal",
                        "rtc_clock_source",
                        "cadence_timing_quality",
                        "stop_mode_variant",
                    ):
                        passthrough_value = telemetry.power_metrics.get(passthrough_field)
                        if passthrough_value is not None:
                            merged_power_metrics[passthrough_field] = passthrough_value
                    merged_power_metrics.update(runtime_storage_metrics)
                    result_latency_s = telemetry.latency_s
                    try:
                        harness_latency_s = float(merged_power_metrics.get("harness_latency_s", -1.0))
                    except (TypeError, ValueError):
                        harness_latency_s = -1.0
                    phase_label = str(merged_power_metrics.get("phase", "")).strip().lower()
                    if phase_label == "back_to_back" and harness_latency_s > 0.0:
                        result_latency_s = harness_latency_s
                    return DeviceMetrics(
                        ram_bytes=compile_result.ram_bytes or -1,
                        flash_bytes=compile_result.flash_bytes or -1,
                        latency_s=result_latency_s,
                        arena_bytes=compile_result.arena_bytes or -1,
                        error_code=HIL_ERROR_OK,
                        power_metrics=merged_power_metrics,
                        external_flash_bytes=compile_external_flash_bytes,
                    )
                stm32_cube_clt.debug_load_elf(
                    elf_path=elf_path,
                    gdbserver=self._options.gdbserver,
                    gdb=self._options.gdb,
                    cubeprog_bin=self._options.cubeprog_bin,
                    gdb_port=self._options.gdb_port,
                    apid=self._options.apid,
                    server_ready_timeout_s=self._options.server_ready_timeout_s,
                    run_after_load=True,
                )
                telemetry = stm32_runtime.execute_runtime_session(
                    monitor,
                    boot_timeout_s=(
                        DEFAULT_LRUN_BOOT_TIMEOUT_S
                        if dut_ready_timeout_s is None
                        else (
                            stm32_runtime.DEFAULT_BOOT_TIMEOUT_S
                            if dut_ready_timeout_s is None
                            else float(dut_ready_timeout_s)
                        )
                    ),
                    run_timeout_s=float(serial_timeout_s),
                )
        except stm32_cube_clt.WorkflowError as exc:
            return _upload_failure(classify_stm32_backend_error(exc), str(exc))
        except stm32_runtime.STM32RuntimeProtocolError as exc:
            _record["dut_log"] = list(exc.serial_log)
            return _latency_failure(exc.kind, exc.detail)
        except (serial.SerialException, RuntimeError) as exc:
            if monitor is not None:
                try:
                    _record["dut_log"] = list(monitor.snapshot_lines())
                except Exception:
                    logger.warning("Unable to snapshot DUT log after runtime I/O failure", exc_info=True)
            return _latency_failure("runtime_io", str(exc))
        _record["dut_log"] = list(telemetry.serial_log)
        return DeviceMetrics(
            ram_bytes=compile_result.ram_bytes or -1,
            flash_bytes=compile_result.flash_bytes or -1,
            latency_s=telemetry.latency_s,
            arena_bytes=compile_result.arena_bytes or -1,
            error_code=HIL_ERROR_OK,
            power_metrics=_merge_metrics(telemetry.power_metrics),
            external_flash_bytes=compile_external_flash_bytes,
        )

    def _write_runtime_phase_config(
        self,
        *,
        paths: STM32WorkspacePaths,
        selected_phase: str,
        measured_runs: int,
    ) -> None:
        """Write and validate the staged STM32 phase configuration.

        Parameters
        ----------
        paths : STM32WorkspacePaths
            Layout-aware staged STM32 workspace paths.
        selected_phase : str
            Runtime phase to compile into the staged project.
        measured_runs : int
            Requested on-device run count.

        Returns
        -------
        None
            The method rewrites the generated phase-config header in place.
        """
        header_path = _write_phase_config_header(
            paths=paths,
            cpu_clock_mhz=self._options.cpu_clock_mhz,
            selected_phase=selected_phase,
            latency_budget_ms=self._options.latency_budget_ms,
            measured_runs=max(1, int(measured_runs)),
            wake_margin_us=self._options.wake_margin_us,
            min_sleep_us=self._options.min_sleep_us,
        )
        _validate_phase_config_header(
            header_path,
            cpu_clock_mhz=self._options.cpu_clock_mhz,
            selected_phase=selected_phase,
            measured_runs=max(1, int(measured_runs)),
            latency_budget_ms=self._options.latency_budget_ms,
            wake_margin_us=self._options.wake_margin_us,
            min_sleep_us=self._options.min_sleep_us,
        )

    def _cadenced_power_metrics_from_phase_result(
        self,
        phase_result: DeviceMetrics,
    ) -> dict[str, Any]:
        """Convert one cadenced phase result into flattened metrics extras.

        Parameters
        ----------
        phase_result : DeviceMetrics
            Result returned by the cadenced second pass.

        Returns
        -------
        dict[str, Any]
            Flattened cadenced metrics ready to merge into ``power_metrics``.
        """
        power_metrics = dict(phase_result.power_metrics or {})

        def _safe_float(value: Any, default: float = -1.0) -> float:
            """Coerce a raw metrics value to ``float`` or return ``default``.

            Parameters
            ----------
            value : Any
                Raw metrics payload value.
            default : float, default=-1.0
                Fallback value used when ``value`` is unavailable.

            Returns
            -------
            float
                Parsed floating-point value or ``default``.
            """
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def _safe_int(value: Any, default: int = -1) -> int:
            """Coerce a raw metrics value to ``int`` or return ``default``.

            Parameters
            ----------
            value : Any
                Raw metrics payload value.
            default : int, default=-1
                Fallback value used when ``value`` is unavailable.

            Returns
            -------
            int
                Parsed integer value or ``default``.
            """
            try:
                return int(value)
            except (TypeError, ValueError):
                return int(default)

        runs = max(1, _safe_int(power_metrics.get("runs", 1), default=1))
        energy_per_inference = _safe_float(power_metrics.get("energy_mj_per_inference", -1.0))
        energy_per_window = (
            float(energy_per_inference) * float(runs)
            if energy_per_inference >= 0.0
            else -1.0
        )
        harness_latency_s = _safe_float(power_metrics.get("harness_latency_s", -1.0))
        window_latency_s = _safe_float(power_metrics.get("timer_per_window_s", -1.0))
        return {
            "cadenced_error_code": int(phase_result.error_code),
            "cadenced_active_inference_latency_ms": (
                phase_result.latency_s * 1000.0 if phase_result.latency_s >= 0.0 else -1.0
            ),
            "cadenced_window_latency_ms": (
                window_latency_s * 1000.0
                if window_latency_s >= 0.0
                else -1.0
            ),
            "cadenced_energy_mj_per_window": energy_per_inference,
            "cadenced_energy_mj_per_trial": energy_per_window,
            "cadenced_avg_power_mw": _safe_float(power_metrics.get("avg_power_mw", -1.0)),
            "cadenced_avg_current_ma": _safe_float(power_metrics.get("avg_current_ma", -1.0)),
            "cadenced_bus_voltage_v": _safe_float(power_metrics.get("bus_voltage_v", -1.0)),
            "cadenced_idle_power_mw": _safe_float(power_metrics.get("idle_power_mw", -1.0)),
            "cadenced_harness_latency_ms": (
                harness_latency_s * 1000.0
                if harness_latency_s >= 0.0
                else -1.0
            ),
            "cadenced_clock_hz": _safe_float(power_metrics.get("clock_hz", -1.0)),
            "cadenced_dwt_cycles_per_inference": _safe_float(
                power_metrics.get("dwt_cycles_per_inference", -1.0)
            ),
            "cadenced_rtc_sleep_ms": _safe_float(power_metrics.get("rtc_sleep_total_ms", -1.0)),
            "cadenced_deadline_miss_count": _safe_int(power_metrics.get("deadline_miss_count", -1)),
            "cadenced_wake_recovery_us_mean": _safe_float(power_metrics.get("wake_recovery_us", -1.0)),
            "cadenced_wake_overshoot_us_mean": _safe_float(power_metrics.get("wake_overshoot_us", -1.0)),
            "cadenced_rtc_clock_source": power_metrics.get("rtc_clock_source"),
            "cadenced_rtc_clock_hz_nominal": _safe_float(power_metrics.get("rtc_clock_hz_nominal", -1.0)),
            "cadenced_timing_quality": power_metrics.get("cadence_timing_quality"),
            "cadenced_stop_mode_variant": power_metrics.get("stop_mode_variant"),
        }

    def evaluate(
        self,
        *,
        dirpath: str | Path,
        arena_kb: int,
        window_size: int,
        num_channels: int,
        serial_port: Optional[str] = None,
        run_hil: bool = True,
        baud_rate: int = 115200,
        serial_timeout_s: float = 12.0,
        measured_inference_runs: int | None = None,
        dut_ready_timeout_s: Optional[float] = None,
        harness_serial_port: Optional[str] = None,
        harness_fqbn: Optional[str] = None,
        harness_auto_flash: Optional[str] = None,
        harness_arm_pin: Optional[int] = None,
        harness_trigger_pin: Optional[int] = None,
        dut_arm_hold_ms: Optional[int] = None,
        harness_stable_low_ms: Optional[int] = None,
        harness_ready_timeout_s: Optional[float] = None,
        harness_arm_timeout_s: Optional[float] = None,
        harness_active_timeout_s: Optional[float] = None,
        harness_done_timeout_s: Optional[float] = None,
    ) -> DeviceMetrics:
        """Run the staged STM32 evaluation path and return canonical metrics.

        Parameters
        ----------
        dirpath : str | pathlib.Path
            Staged STM32 workspace root to compile.
        arena_kb : int
            Shared interface argument retained for compatibility.
        window_size : int
            Shared interface argument retained for compatibility.
        num_channels : int
            Shared interface argument retained for interface compatibility.
        serial_port : str | None, optional
            Serial port retained for interface compatibility.
        run_hil : bool, default=True
            Whether the caller requested runtime measurement.
        baud_rate : int, default=115200
            Serial baud rate retained for interface compatibility.
        serial_timeout_s : float, default=12.0
            Serial timeout retained for interface compatibility.
        measured_inference_runs : int | None, optional
            Requested on-device run count. When omitted, staged workspaces keep
            their existing configured value and unstaged evaluation falls back
            to ``10``.
        dut_ready_timeout_s : float | None, optional
            DUT-ready timeout retained for interface compatibility.
        harness_serial_port : str | None, optional
            Harness serial port retained for interface compatibility.
        harness_fqbn : str | None, optional
            Harness FQBN retained for interface compatibility.
        harness_auto_flash : str | None, optional
            Harness flashing policy retained for interface compatibility.
        harness_arm_pin : int | None, optional
            Harness arm pin retained for interface compatibility.
        harness_trigger_pin : int | None, optional
            Harness trigger pin retained for interface compatibility.
        dut_arm_hold_ms : int | None, optional
            DUT arm hold retained for interface compatibility.
        harness_stable_low_ms : int | None, optional
            Stable-low period retained for interface compatibility.
        harness_ready_timeout_s : float | None, optional
            Harness ready timeout retained for interface compatibility.
        harness_arm_timeout_s : float | None, optional
            Harness arm timeout retained for interface compatibility.
        harness_active_timeout_s : float | None, optional
            Harness active timeout retained for interface compatibility.
        harness_done_timeout_s : float | None, optional
            Harness done timeout retained for interface compatibility.

        Returns
        -------
        DeviceMetrics
            Canonical STM32 metrics from the back-to-back pass, plus optional
            cadenced extras stored in ``power_metrics``.
        """
        base_result = self._evaluate_single_phase(
            dirpath=dirpath,
            phase="back_to_back",
            arena_kb=arena_kb,
            window_size=window_size,
            num_channels=num_channels,
            serial_port=serial_port,
            run_hil=run_hil,
            baud_rate=baud_rate,
            serial_timeout_s=serial_timeout_s,
            measured_inference_runs=measured_inference_runs,
            dut_ready_timeout_s=dut_ready_timeout_s,
            harness_serial_port=harness_serial_port,
            harness_fqbn=harness_fqbn,
            harness_auto_flash=harness_auto_flash,
            harness_arm_pin=harness_arm_pin,
            harness_trigger_pin=harness_trigger_pin,
            dut_arm_hold_ms=dut_arm_hold_ms,
            harness_stable_low_ms=harness_stable_low_ms,
            harness_ready_timeout_s=harness_ready_timeout_s,
            harness_arm_timeout_s=harness_arm_timeout_s,
            harness_active_timeout_s=harness_active_timeout_s,
            harness_done_timeout_s=harness_done_timeout_s,
        )
        merged_power_metrics = dict(base_result.power_metrics or {})
        if self._options.runtime_mode != "cadenced" or not run_hil or base_result.error_code != HIL_ERROR_OK:
            merged_power_metrics["runtime_mode"] = "back_to_back"
            return DeviceMetrics(
                ram_bytes=base_result.ram_bytes,
                flash_bytes=base_result.flash_bytes,
                latency_s=base_result.latency_s,
                arena_bytes=base_result.arena_bytes,
                error_code=base_result.error_code,
                power_metrics=merged_power_metrics,
                external_flash_bytes=base_result.external_flash_bytes,
                retry_hint_bytes=base_result.retry_hint_bytes,
            )

        cadenced_result = self._evaluate_single_phase(
            dirpath=dirpath,
            phase="cadenced",
            arena_kb=arena_kb,
            window_size=window_size,
            num_channels=num_channels,
            serial_port=serial_port,
            run_hil=run_hil,
            baud_rate=baud_rate,
            serial_timeout_s=serial_timeout_s,
            measured_inference_runs=measured_inference_runs,
            dut_ready_timeout_s=dut_ready_timeout_s,
            harness_serial_port=harness_serial_port,
            harness_fqbn=harness_fqbn,
            harness_auto_flash=harness_auto_flash,
            harness_arm_pin=harness_arm_pin,
            harness_trigger_pin=harness_trigger_pin,
            dut_arm_hold_ms=dut_arm_hold_ms,
            harness_stable_low_ms=harness_stable_low_ms,
            harness_ready_timeout_s=harness_ready_timeout_s,
            harness_arm_timeout_s=harness_arm_timeout_s,
            harness_active_timeout_s=harness_active_timeout_s,
            harness_done_timeout_s=harness_done_timeout_s,
        )
        merged_power_metrics["runtime_mode"] = "cadenced"
        merged_power_metrics.update(self._cadenced_power_metrics_from_phase_result(cadenced_result))
        for key, value in (cadenced_result.power_metrics or {}).items():
            if key.startswith("measurement_") or key.startswith("backend_error_"):
                merged_power_metrics[key] = value
        merged_power_metrics["measurement_record_paths"] = (
            list((base_result.power_metrics or {}).get("measurement_record_paths", []))
            + list((cadenced_result.power_metrics or {}).get("measurement_record_paths", [])))
        base_acceptance = base_result.power_metrics or {}
        cadence_acceptance = cadenced_result.power_metrics or {}
        merged_power_metrics["measurement_deployment_eligible"] = (
            base_acceptance.get("measurement_deployment_eligible") is True
            and cadence_acceptance.get("measurement_deployment_eligible") is True)
        merged_power_metrics["measurement_infeasibility_reasons"] = list(dict.fromkeys(
            list(base_acceptance.get("measurement_infeasibility_reasons", []))
            + list(cadence_acceptance.get("measurement_infeasibility_reasons", []))))
        result_latency_s = cadenced_result.latency_s
        return DeviceMetrics(
            ram_bytes=base_result.ram_bytes,
            flash_bytes=base_result.flash_bytes,
            latency_s=result_latency_s,
            arena_bytes=base_result.arena_bytes,
            error_code=cadenced_result.error_code,
            power_metrics=merged_power_metrics,
            external_flash_bytes=base_result.external_flash_bytes,
            retry_hint_bytes=base_result.retry_hint_bytes,
        )
