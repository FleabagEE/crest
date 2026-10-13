# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the STM32 backend and LRUN workspace helpers."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.errors import (  # noqa: E402
    HIL_ERROR_INTEGRITY,
    HIL_ERROR_COMPILE,
    HIL_ERROR_FLASH_OVERFLOW,
    HIL_ERROR_LATENCY,
    HIL_ERROR_OK,
    HIL_ERROR_RAM_OVERFLOW,
    HIL_ERROR_UPLOAD,
)
from crest.devices import CandidatePrepareRequest, DeviceMetrics  # noqa: E402
from crest.microcontrollers import get_device, list_device_specs, resolve_device_options  # noqa: E402
from crest.microcontrollers import stm32_cube_clt  # noqa: E402
from crest.microcontrollers import stm32_nucleo_n657x0 as stm32_n657_backend  # noqa: E402
from crest.microcontrollers import stm32_runtime  # noqa: E402
from crest.microcontrollers.stm32_nucleo_n657x0 import (  # noqa: E402
    BOARD_NAME,
    MEASUREMENT_INTEGRITY_KINDS,
    DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME,
    DEFAULT_MAX_EXTERNAL_FLASH_BYTES,
    DEFAULT_TEMPLATE_ROOT,
    DEFAULT_WEIGHTS_FLASH_ADDRESS,
    DEFAULT_WEIGHTS_MEMORY_POOL,
    EXPECTED_GENERATED_OUTPUTS,
    STAGED_MANIFEST_NAME,
    STM32NucleoN657X0QDevice,
)


def _write_text(path: Path, text: str) -> None:
    """Write text to a file, creating parent directories first.

    Parameters
    ----------
    path : pathlib.Path
        Destination file path.
    text : str
        File contents to write.

    Returns
    -------
    None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_lrun_project_tree(root: Path) -> Path:
    """Build a minimal STM32 LRUN workspace for unit tests.

    Parameters
    ----------
    root : Path
        Root directory where the synthetic LRUN workspace should be created.

    Returns
    -------
    Path
        The same ``root`` path after the required template files exist.
    """
    # Create just enough of the canonical tree for template parsing, compile
    # staging, and manifest tests without copying a real Cube workspace.
    _write_text(root / "FSBL" / "Inc" / "stm32_extmem_conf.h", "#define EXTMEM_LRUN_SOURCE_SIZE 0x00020000\n")
    _write_text(root / "Secure_nsclib" / "secure_nsc.h", "#pragma once\n")
    _write_text(root / "Drivers" / "STM32N6xx_HAL_Driver" / "Src" / "vendor.c", "void vendor(void) {}\n")
    _write_text(root / "Drivers" / "STM32N6xx_HAL_Driver" / "Inc" / "vendor.h", "#pragma once\n")
    _write_text(
        root / "Middlewares" / "ST" / "STM32_ExtMem_Manager" / "boot_stub.c",
        "void extmem_boot_stub(void) {}\n",
    )
    _write_text(root / "Appli" / "Inc" / "main.h", "#pragma once\n")
    _write_text(root / "Appli" / "Src" / "main.c", "int main(void) { return 0; }\n")
    _write_text(root / "Appli" / "Src" / "secure_nsc.c", "void SecureGateway(void) {}\n")
    _write_text(root / "Appli" / "Src" / "system_stm32n6xx_s.c", "void SystemInit(void) {}\n")
    _write_text(root / "Appli" / "Inc" / "stm32n6xx_hal_conf.h", "#pragma once\n")
    _write_text(root / "Appli" / "Inc" / "stm32n6xx_nucleo_conf.h", "#pragma once\n")
    _write_text(root / "Appli" / "Inc" / "partition_stm32n657xx.h", "#pragma once\n")
    _write_text(root / "Appli" / "Inc" / "crest_dut_runner.h", "#pragma once\n")
    _write_text(
        root / "Appli" / "Inc" / "network_data_params.h",
        "#define AI_NETWORK_DATA_ACTIVATIONS_SIZE (16384)\n",
    )
    _write_text(
        root / "STM32CubeIDE" / "AppS" / "STM32N657XX_LRUN.ld",
        "\n".join(
            [
                "_Min_Heap_Size = 0x2000;",
                "_Min_Stack_Size = 0x4000;",
                "",
            ]
        ),
    )
    _write_text(
        root / "STM32CubeIDE" / "Boot" / "STM32N657XX_AXISRAM2_fsbl.ld",
        "\n".join(
            [
                "_Min_Heap_Size = 0x2000;",
                "_Min_Stack_Size = 0x4000;",
                "",
            ]
        ),
    )
    _write_text(
        root / "STM32CubeIDE" / "AppS" / "Debug" / "makefile",
        "\n".join(
            [
                "BUILD_ARTIFACT_NAME := Template_LRUN_AppS",
                "Template_LRUN_AppS.elf:",
                "",
            ]
        ),
    )
    _write_text(
        root / "STM32CubeIDE" / "Boot" / "Debug" / "makefile",
        "\n".join(
            [
                "BUILD_ARTIFACT_NAME := Template_LRUN_FSBL",
                "Template_LRUN_FSBL.elf:",
                "",
            ]
        ),
    )
    _write_text(
        root / "STM32CubeIDE" / "Boot" / "Debug" / "Src" / "subdir.mk",
        "arm-none-eabi-gcc -I../../../FSBL/Inc\n",
    )
    _write_text(
        root / "STM32CubeIDE" / "AppS" / "Debug" / "Src" / "subdir.mk",
        "\n".join(
            [
                "C_SRCS += \\",
                "../../../Appli/Src/secure_nsc.c",
                "",
            ]
        ),
    )
    _write_text(root / "STM32CubeIDE" / "Boot" / "Debug" / "objects.list", "obj\n")
    _write_text(root / "STM32CubeIDE" / "AppS" / "Debug" / "objects.list", "obj\n")
    return root


def _make_prepare_request(
    *,
    config: object,
    outputs_dir: Path,
    model: object,
    calibration_inputs: object,
    model_variant: str = "approx_trained",
    checkpoint_path: Path | str | None = None,
) -> CandidatePrepareRequest:
    """Build a typed backend preparation request for STM32 tests.

    Parameters
    ----------
    config : object
        Lightweight config object supplied to the backend.
    outputs_dir : Path
        Artifact root used by the staged candidate.
    model : object
        Model object forwarded to candidate preparation.
    calibration_inputs : object
        Representative calibration inputs exposed through ``.inputs``.
    model_variant : str, optional
        Export variant label.
    checkpoint_path : Path | str | None, optional
        Optional checkpoint path for trained variants.

    Returns
    -------
    CandidatePrepareRequest
        Typed request object accepted by ``prepare_candidate(...)``.
    """
    calibration_split = type("CalibrationSplit", (), {"inputs": calibration_inputs})()
    return CandidatePrepareRequest(
        config=config,
        model=model,
        model_variant=model_variant,
        artifact_root=outputs_dir,
        tflite_model_path=outputs_dir / "ignored.tflite",
        calibration_split=calibration_split,
        quantization_mode="int8_ptq",
        input_shape=(4, 6),
        checkpoint_path=checkpoint_path,
    )


class _FakeSerialMonitor:
    """Context-manager double for STM direct-serial tests."""

    def __init__(self, port: str, baud: int, label: str) -> None:
        """Initialize the fake serial monitor.

        Parameters
        ----------
        port : str
            Serial device path associated with the monitor.
        baud : int
            Baud rate requested by the caller.
        label : str
            Human-readable channel label used in runtime logging.
        """
        self.port = port
        self.baud = baud
        self.label = label

    def __enter__(self):
        """Open the fake monitor context.

        Returns
        -------
        _FakeSerialMonitor
            The active fake serial monitor instance.
        """
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Close the fake monitor context.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type raised inside the context, if any.
        exc : BaseException | None
            Exception instance raised inside the context, if any.
        traceback : types.TracebackType | None
            Traceback associated with the exception, if one occurred.

        Returns
        -------
        None
            Always returns ``None`` so exceptions propagate normally.
        """
        del exc_type, exc, traceback

    def write_line(self, text: str) -> None:
        """Accept a serial log line emitted by the DUT.

        Parameters
        ----------
        text : str
            Serial log line that would have been written to the monitor.

        Returns
        -------
        None
            This test double stores no output and returns nothing.
        """
        del text


class _FakeHarnessSerial:
    """Context-manager double for open harness serial sessions."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the fake harness serial session.

        Parameters
        ----------
        *args
            Positional arguments passed to the real harness serial class.
        **kwargs
            Keyword arguments passed to the real harness serial class.
        """
        del args, kwargs

    def __enter__(self):
        """Open the fake harness serial session.

        Returns
        -------
        _FakeHarnessSerial
            The active fake harness serial session.
        """
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Close the fake harness serial session.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type raised inside the context, if any.
        exc : BaseException | None
            Exception instance raised inside the context, if any.
        traceback : types.TracebackType | None
            Traceback associated with the exception, if one occurred.

        Returns
        -------
        None
            Always returns ``None`` so exceptions propagate normally.
        """
        del exc_type, exc, traceback


class STM32RegistryTests(unittest.TestCase):
    """Validate STM32 registry wiring and default metadata."""

    def test_registry_resolves_stm_device_without_explicit_project_root(self) -> None:
        """Ensure registry construction uses the canonical STM template by default.

        Returns
        -------
        None
        """
        # The STM32 registry should still resolve the default device when callers omit a custom project root.
        device = get_device("STM32_NUCLEO_N657X0_Q")
        self.assertIsInstance(device, STM32NucleoN657X0QDevice)
        self.assertEqual(device.spec.name, BOARD_NAME)
        self.assertEqual(device.spec.arena_sizes_kb, [-1])
        self.assertEqual(device.resolved_options.project_root, DEFAULT_TEMPLATE_ROOT.resolve())

    def test_list_device_specs_includes_stm_entry(self) -> None:
        """Ensure the public device spec listing includes the STM board.

        Returns
        -------
        None
        """
        # Listing device specs should include the STM32 entry so the public hardware catalog stays complete.
        specs = list_device_specs()
        self.assertIn("STM32_NUCLEO_N657X0_Q", specs)
        self.assertEqual(specs["STM32_NUCLEO_N657X0_Q"]["arena_sizes"], [-1])
        self.assertEqual(
            specs["STM32_NUCLEO_N657X0_Q"]["max_external_flash"],
            DEFAULT_MAX_EXTERNAL_FLASH_BYTES,
        )


class STM32BackendBehaviorTests(unittest.TestCase):
    """Validate STM Phase 2 option resolution, staging, and compile behavior."""

    def setUp(self) -> None:
        """Keep durable measurement records inside a per-test temporary directory."""
        evidence_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch("crest.microcontrollers.stm32_nucleo_n657x0.Path.cwd", return_value=Path(evidence_dir)))

    def test_resolve_device_options_defaults_partial_stm_numeric_block(self) -> None:
        """Ensure partial STM config blocks inherit defaults from the backend.

        Returns
        -------
        None
        """
        # Partial STM32 numeric option blocks should be defaulted deterministically before staging begins.
        resolved = resolve_device_options(
            "STM32_NUCLEO_N657X0_Q",
            type(
                "DeviceConfigDouble",
                (),
                {"stm32": type("STMConfigDouble", (), {"gdb_port": 61235})()},
            )(),
        )
        self.assertEqual(resolved["project_root"], DEFAULT_TEMPLATE_ROOT.resolve())
        self.assertEqual(resolved["gdb_port"], 61235)
        self.assertEqual(resolved["apid"], stm32_cube_clt.DEFAULT_APID)
        self.assertEqual(
            resolved["server_ready_timeout_s"],
            stm32_cube_clt.SERVER_READY_TIMEOUT_S,
        )
        self.assertEqual(resolved["cpu_clock_mhz"], 600)
        self.assertEqual(resolved["runtime_mode"], "back_to_back")
        self.assertEqual(resolved["wake_margin_us"], 5000)
        self.assertEqual(resolved["min_sleep_us"], 5000)
        self.assertEqual(resolved["weight_storage_mode"], "embedded")
        self.assertEqual(resolved["weights_flash_address"], DEFAULT_WEIGHTS_FLASH_ADDRESS)
        self.assertEqual(resolved["weights_memory_pool"], DEFAULT_WEIGHTS_MEMORY_POOL.resolve())
        self.assertIsNone(resolved["weights_external_loader"])
        self.assertEqual(resolved["max_external_flash_bytes"], DEFAULT_MAX_EXTERNAL_FLASH_BYTES)

    def test_resolve_device_options_warns_for_template_root_override(self) -> None:
        """Ensure the deprecated ``template_root`` override emits a warning.

        Returns
        -------
        None
        """
        # Template-root overrides should emit a warning so callers know they are leaving the canonical workspace layout.
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = _build_lrun_project_tree(Path(tmpdir) / "template")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                resolved = resolve_device_options(
                    "STM32_NUCLEO_N657X0_Q",
                    type(
                        "DeviceConfigDouble",
                        (),
                        {
                            "stm32": type(
                                "STMConfigDouble",
                                (),
                                {
                                    "template_root": str(template_root),
                                },
                            )()
                        },
                    )(),
                )
        self.assertEqual(resolved["project_root"], template_root.resolve())
        self.assertTrue(any(issubclass(item.category, DeprecationWarning) for item in caught))

    def test_resolve_device_options_defaults_without_materialized_lrun_workspace(self) -> None:
        """Ensure default STM option resolution does not require setup-generated paths."""
        # LRUN option resolution should synthesize defaults even before the workspace exists on disk.
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_root = Path(tmpdir) / "crest_stm32_lrun"
            with patch.object(stm32_n657_backend, "DEFAULT_TEMPLATE_ROOT", missing_root):
                resolved = resolve_device_options(
                    "STM32_NUCLEO_N657X0_Q",
                    type("DeviceConfigDouble", (), {})(),
                )
        self.assertEqual(resolved["project_root"], missing_root.resolve())

    def test_resolve_device_options_accepts_custom_lrun_root(self) -> None:
        """Ensure custom STM roots resolve when they match the LRUN workspace shape."""
        # Custom LRUN roots should resolve cleanly so staging can target an alternate workspace.
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = _build_lrun_project_tree(Path(tmpdir) / "template")
            resolved = resolve_device_options(
                "STM32_NUCLEO_N657X0_Q",
                type(
                    "DeviceConfigDouble",
                    (),
                    {"stm32": type("STMConfigDouble", (), {"project_root": str(template_root)})()},
                )(),
            )
        self.assertEqual(resolved["project_root"], template_root.resolve())

    def test_resolve_device_options_rejects_project_layout_override(self) -> None:
        """Ensure callers cannot force a legacy STM layout anymore."""
        # Project-layout overrides should fail early so the backend only stages supported LRUN layouts.
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = _build_lrun_project_tree(Path(tmpdir) / "template")
            with self.assertRaisesRegex(ValueError, "project_layout'.*no longer supported"):
                resolve_device_options(
                    "STM32_NUCLEO_N657X0_Q",
                    type(
                        "DeviceConfigDouble",
                        (),
                        {
                            "stm32": type(
                                "STMConfigDouble",
                                (),
                                {"project_root": str(template_root), "project_layout": "lrun_dev_boot"},
                            )()
                        },
                    )(),
                )

    def test_compile_fails_fast_when_stack_is_too_small(self) -> None:
        """Ensure undersized LRUN linker stack reservations fail before build.

        Returns
        -------
        None
        """
        # Compile-time stack checks should fail before flashing so impossible runtime layouts never stage artifacts.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_lrun_project_tree(Path(tmpdir) / "staged")
            _write_text(
                staged_root / "STM32CubeIDE" / "AppS" / "STM32N657XX_LRUN.ld",
                "\n".join(
                    [
                        "_Min_Heap_Size = 0x2000;",
                        "_Min_Stack_Size = 0x0400;",
                        "",
                    ]
                ),
            )
            device = STM32NucleoN657X0QDevice(device_options={"project_root": str(staged_root)})
            result = device.compile(
                sketch_path=staged_root,
                arena_kb=-1,
                window_size=200,
                num_channels=6,
            )

        self.assertFalse(result.success)
        self.assertIn("stack reservation is too small", result.log)

    def test_upload_requires_staged_build_dir(self) -> None:
        """Ensure upload does not silently fall back to the template Debug dir.

        Returns
        -------
        None
        """
        # Runtime upload should require a staged build directory so it cannot program an undefined workspace.
        device = STM32NucleoN657X0QDevice()
        result = device.upload(
            sketch_path=Path("ignored"),
            build_dir=None,
            serial_port="ttyACM0",
        )
        self.assertFalse(result.success)
        self.assertIn("requires a staged build directory", result.log)

    def test_evaluate_compile_only_reports_real_arena_bytes(self) -> None:
        """Ensure compile-only evaluate returns real parsed arena bytes.

        Returns
        -------
        None
        """
        # Compile-only evaluation should only report real arena bytes when the backend actually knows them.
        device = STM32NucleoN657X0QDevice()
        project_root = Path("stm")
        build_dir = project_root / "Debug"
        with patch.object(
            device,
            "compile",
            return_value=type(
                "CompileResultDouble",
                (),
                {
                    "success": True,
                    "log": "ok",
                    "flash_bytes": 2222,
                    "ram_bytes": 1111,
                    "overflow_kind": None,
                    "build_dir": build_dir,
                    "arena_bytes": 4096,
                },
            )(),
        ):
            metrics = device.evaluate(
                dirpath=project_root,
                arena_kb=-1,
                window_size=200,
                num_channels=6,
                run_hil=False,
            )

        self.assertEqual(metrics.error_code, HIL_ERROR_OK)
        self.assertEqual(metrics.arena_bytes, 4096)
        self.assertEqual(metrics.flash_bytes, 2222)
        self.assertEqual(metrics.ram_bytes, 1111)

    def test_supports_runtime_measurement_is_enabled(self) -> None:
        """Ensure STM direct-serial runtime measurement is advertised.

        Returns
        -------
        None
            This test asserts that runtime measurement support is enabled.
        """
        # The STM32 device contract should advertise runtime measurement support.
        device = STM32NucleoN657X0QDevice()
        self.assertTrue(device.supports_runtime_measurement())

    def test_supports_energy_measurement_is_enabled(self) -> None:
        """Ensure STM harness-assisted energy measurement is advertised.

        Returns
        -------
        None
            This test asserts that energy measurement support is enabled.
        """
        # The STM32 device contract should advertise energy measurement support.
        device = STM32NucleoN657X0QDevice()
        self.assertTrue(device.supports_energy_measurement())

    def test_evaluate_run_hil_success_uses_runtime_session(self) -> None:
        """Ensure HIL evaluation returns parsed runtime latency on success.

        Returns
        -------
        None
            This test asserts that successful runtime telemetry reaches the caller.
        """
        # Successful STM32 HIL runs should use the runtime session path instead of falling back to proxy behavior.
        device = STM32NucleoN657X0QDevice(serial_port="/dev/ttyACM0")
        project_root = Path("stm")
        build_dir = project_root / "Debug"
        compile_result = type(
            "CompileResultDouble",
            (),
            {
                "success": True,
                "log": "ok",
                "flash_bytes": 2222,
                "ram_bytes": 1111,
                "overflow_kind": None,
                "build_dir": build_dir,
                "arena_bytes": 4096,
            },
        )()
        telemetry = stm32_runtime.STM32RuntimeTelemetry(
            latency_s=0.0025,
            serial_log=["STM32_AI_INIT=OK", "DUT READY", "STM32_AI_RUN=OK"],
            power_metrics={"clock_hz": 600000000.0, "sequence": 1.0},
        )
        # Repaired DUT telemetry: RTC aggregate covers all reported releases.
        telemetry.serial_log.extend([
            "measurement protocol output: rtc_wall_v4_dwt_active_stop_evidence_awake_tail",
            "cadence budget (ms): 200",
        ])
        telemetry.power_metrics.update(runs=10, phase="back_to_back",
            timer_per_window_s=0.025, rtc_clock_source="LSE", rtc_clock_hz_nominal=32768)

        with patch.object(device, "compile", return_value=compile_result), patch(
            "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.resolve_elf_path",
            return_value=build_dir / "app.elf",
        ), patch(
            "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
            _FakeSerialMonitor,
        ), patch(
            "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.debug_load_elf",
            return_value="upload ok",
        ) as load_mock, patch(
            "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.execute_runtime_session",
            return_value=telemetry,
        ) as runtime_mock:
            metrics = device.evaluate(
                dirpath=project_root,
                arena_kb=-1,
                window_size=200,
                num_channels=6,
                serial_port="/dev/ttyACM0",
                run_hil=True,
                serial_timeout_s=7.5,
                dut_ready_timeout_s=3.5,
            )

        load_mock.assert_called_once()
        runtime_mock.assert_called_once()
        self.assertEqual(metrics.error_code, HIL_ERROR_OK)
        self.assertAlmostEqual(metrics.latency_s, 0.0025)
        self.assertEqual(metrics.power_metrics["clock_hz"], 600000000.0)
        self.assertEqual(metrics.power_metrics["sequence"], 1.0)
        self.assertEqual(metrics.power_metrics["weight_storage_mode"], "embedded")
        self.assertEqual(metrics.power_metrics["external_flash_bytes"], -1.0)

    def test_evaluate_back_to_back_runtime_mode_skips_second_phase(self) -> None:
        # Back-to-back STM32 runs should skip the cadenced second phase entirely.
        """Validate evaluate back to back runtime mode skips second phase."""
        device = STM32NucleoN657X0QDevice(
            serial_port="/dev/ttyACM0",
            device_options={"runtime_mode": "back_to_back"},
        )
        base_result = DeviceMetrics(
            ram_bytes=1111,
            flash_bytes=2222,
            latency_s=0.0025,
            arena_bytes=4096,
            error_code=HIL_ERROR_OK,
            power_metrics={"clock_hz": 600000000.0, "sequence": 1.0},
        )

        with patch.object(device, "_evaluate_single_phase", return_value=base_result) as phase_mock:
            metrics = device.evaluate(
                dirpath=Path("stm"),
                arena_kb=-1,
                window_size=200,
                num_channels=6,
                serial_port="/dev/ttyACM0",
                run_hil=True,
            )

        phase_mock.assert_called_once()
        self.assertEqual(metrics.power_metrics["runtime_mode"], "back_to_back")
        self.assertNotIn("cadenced_active_inference_latency_ms", metrics.power_metrics)

    def test_evaluate_cadenced_runtime_mode_merges_second_pass_metrics(self) -> None:
        # Cadenced STM32 runs should merge second-pass telemetry into the first-pass result instead of overwriting it.
        """Validate evaluate cadenced runtime mode merges second pass metrics."""
        device = STM32NucleoN657X0QDevice(
            serial_port="/dev/ttyACM0",
            device_options={"runtime_mode": "cadenced", "latency_budget_ms": 200.0},
        )
        base_result = DeviceMetrics(
            ram_bytes=1111,
            flash_bytes=2222,
            latency_s=0.0025,
            arena_bytes=4096,
            error_code=HIL_ERROR_OK,
            power_metrics={"clock_hz": 600000000.0, "sequence": 1.0},
        )
        cadenced_result = DeviceMetrics(
            ram_bytes=1111,
            flash_bytes=2222,
            latency_s=0.080,
            arena_bytes=4096,
            error_code=HIL_ERROR_OK,
            power_metrics={
                "runs": 10,
                "energy_mj_per_inference": 1.25,
                "avg_power_mw": 2.5,
                "avg_current_ma": 0.5,
                "bus_voltage_v": 5.0,
                "idle_power_mw": 0.1,
                "harness_latency_s": 0.2,
                "clock_hz": 600000000.0,
                "dwt_cycles_per_inference": 120000.0,
                "timer_per_window_s": 20.0,
                "rtc_sleep_total_ms": 1500.0,
                "deadline_miss_count": 0,
                "wake_recovery_us": 1200.0,
                "wake_overshoot_us": 35.0,
                "rtc_clock_source": "LSE",
                "rtc_clock_hz_nominal": 32768.0,
                "cadence_timing_quality": "crystal",
                "stop_mode_variant": "system_stop_mainreg_wfi",
            },
        )

        with patch.object(
            device,
            "_evaluate_single_phase",
            side_effect=[base_result, cadenced_result],
        ) as phase_mock:
            metrics = device.evaluate(
                dirpath=Path("stm"),
                arena_kb=-1,
                window_size=200,
                num_channels=6,
                serial_port="/dev/ttyACM0",
                run_hil=True,
            )

        self.assertEqual(phase_mock.call_count, 2)
        self.assertEqual(phase_mock.call_args_list[0].kwargs["phase"], "back_to_back")
        self.assertEqual(phase_mock.call_args_list[1].kwargs["phase"], "cadenced")
        self.assertEqual(metrics.power_metrics["runtime_mode"], "cadenced")
        self.assertEqual(metrics.power_metrics["cadenced_error_code"], HIL_ERROR_OK)
        self.assertAlmostEqual(metrics.latency_s, 0.080)
        self.assertAlmostEqual(metrics.power_metrics["cadenced_active_inference_latency_ms"], 80.0)
        self.assertAlmostEqual(metrics.power_metrics["cadenced_window_latency_ms"], 20000.0)
        self.assertAlmostEqual(metrics.power_metrics["cadenced_energy_mj_per_window"], 1.25)
        self.assertAlmostEqual(metrics.power_metrics["cadenced_energy_mj_per_trial"], 12.5)
        self.assertAlmostEqual(metrics.power_metrics["cadenced_rtc_sleep_ms"], 1500.0)

    def test_evaluate_cadenced_runtime_mode_reports_back_to_back_when_second_phase_never_runs(self) -> None:
        # If cadenced phase two never starts, report back-to-back mode so partial telemetry does not masquerade as a full cadenced run.
        """Validate evaluate cadenced runtime mode reports back to back when second phase never runs."""
        device = STM32NucleoN657X0QDevice(
            serial_port="/dev/ttyACM0",
            device_options={"runtime_mode": "cadenced", "latency_budget_ms": 200.0},
        )
        failed_base_result = DeviceMetrics(
            ram_bytes=1111,
            flash_bytes=2222,
            latency_s=0.0025,
            arena_bytes=4096,
            error_code=HIL_ERROR_LATENCY,
            power_metrics={"clock_hz": 600000000.0},
        )

        with patch.object(
            device,
            "_evaluate_single_phase",
            return_value=failed_base_result,
        ) as phase_mock:
            metrics = device.evaluate(
                dirpath=Path("stm"),
                arena_kb=-1,
                window_size=200,
                num_channels=6,
                serial_port="/dev/ttyACM0",
                run_hil=True,
            )

        phase_mock.assert_called_once()
        self.assertEqual(metrics.power_metrics["runtime_mode"], "back_to_back")
        self.assertNotIn("cadenced_energy_mj_per_trial", metrics.power_metrics)

    def test_cadenced_energy_window_uses_stable_sentinel_when_energy_is_unavailable(self) -> None:
        # Cadenced energy windows should emit a stable sentinel when energy is unavailable so CSV rows keep the same schema.
        """Validate cadenced energy window uses stable sentinel when energy is unavailable."""
        device = STM32NucleoN657X0QDevice(
            serial_port="/dev/ttyACM0",
            device_options={"runtime_mode": "cadenced", "latency_budget_ms": 200.0},
        )
        phase_result = DeviceMetrics(
            ram_bytes=1111,
            flash_bytes=2222,
            latency_s=0.080,
            arena_bytes=4096,
            error_code=HIL_ERROR_OK,
            power_metrics={"runs": 10, "energy_mj_per_inference": -1.0, "timer_per_window_s": 20.0},
        )

        metrics = device._cadenced_power_metrics_from_phase_result(phase_result)

        self.assertEqual(metrics["cadenced_window_latency_ms"], 20000.0)
        self.assertEqual(metrics["cadenced_energy_mj_per_window"], -1.0)
        self.assertEqual(metrics["cadenced_energy_mj_per_trial"], -1.0)

    def test_evaluate_run_hil_runtime_failure_maps_to_latency_error(self) -> None:
        """Ensure protocol failures become ``HIL_ERROR_LATENCY`` with backend detail.

        Returns
        -------
        None
            This test asserts that runtime protocol failures map to latency errors.
        """
        # Runtime-session failures after bring-up should classify as latency errors rather than upload failures.
        device = STM32NucleoN657X0QDevice(serial_port="/dev/ttyACM0")
        project_root = Path("stm")
        build_dir = project_root / "Debug"
        compile_result = type(
            "CompileResultDouble",
            (),
            {
                "success": True,
                "log": "ok",
                "flash_bytes": 2222,
                "ram_bytes": 1111,
                "overflow_kind": None,
                "build_dir": build_dir,
                "arena_bytes": 4096,
            },
        )()
        protocol_error = stm32_runtime.STM32RuntimeProtocolError(
            kind="runtime_timeout",
            detail="Timed out waiting for DUT READY.",
            serial_log=["STM32_AI_INIT=OK"],
        )

        with patch.object(device, "compile", return_value=compile_result), patch(
            "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.resolve_elf_path",
            return_value=build_dir / "app.elf",
        ), patch(
            "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
            _FakeSerialMonitor,
        ), patch(
            "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.debug_load_elf",
            return_value="upload ok",
        ), patch(
            "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.execute_runtime_session",
            side_effect=protocol_error,
        ):
            metrics = device.evaluate(
                dirpath=project_root,
                arena_kb=-1,
                window_size=200,
                num_channels=6,
                serial_port="/dev/ttyACM0",
                run_hil=True,
            )

        self.assertEqual(metrics.error_code, HIL_ERROR_LATENCY)
        self.assertEqual(metrics.power_metrics["backend_error_kind"], "runtime_timeout")

    def test_evaluate_run_hil_upload_failure_maps_to_upload_error(self) -> None:
        """Ensure debug-load failures become ``HIL_ERROR_UPLOAD``.

        Returns
        -------
        None
            This test asserts that upload failures map to upload errors.
        """
        # Failures before runtime execution should stay in the upload bucket so callers can distinguish staging from timing issues.
        device = STM32NucleoN657X0QDevice(serial_port="/dev/ttyACM0")
        project_root = Path("stm")
        build_dir = project_root / "Debug"
        compile_result = type(
            "CompileResultDouble",
            (),
            {
                "success": True,
                "log": "ok",
                "flash_bytes": 2222,
                "ram_bytes": 1111,
                "overflow_kind": None,
                "build_dir": build_dir,
                "arena_bytes": 4096,
            },
        )()

        with patch.object(device, "compile", return_value=compile_result), patch(
            "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.resolve_elf_path",
            return_value=build_dir / "app.elf",
        ), patch(
            "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
            _FakeSerialMonitor,
        ), patch(
            "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.debug_load_elf",
            side_effect=stm32_cube_clt.WorkflowError("ST-LINK failed"),
        ):
            metrics = device.evaluate(
                dirpath=project_root,
                arena_kb=-1,
                window_size=200,
                num_channels=6,
                serial_port="/dev/ttyACM0",
                run_hil=True,
            )

        self.assertEqual(metrics.error_code, HIL_ERROR_UPLOAD)
        self.assertEqual(metrics.power_metrics["backend_error_kind"], "upload")

    def test_evaluate_maps_overflow_kinds(self) -> None:
        """Ensure compile overflow classifications map to shared HIL codes.

        Returns
        -------
        None
        """
        # Overflow classification should map backend-specific failure kinds onto the stable HIL error codes.
        device = STM32NucleoN657X0QDevice()
        project_root = Path("stm")
        build_dir = project_root / "Debug"
        for overflow_kind, expected_error in (
            ("flash", HIL_ERROR_FLASH_OVERFLOW),
            ("ram", HIL_ERROR_RAM_OVERFLOW),
        ):
            compile_result = type(
                "CompileResultDouble",
                (),
                {
                    "success": False,
                    "log": "overflow",
                    "flash_bytes": 100,
                    "ram_bytes": 200,
                    "overflow_kind": overflow_kind,
                    "build_dir": build_dir,
                    "arena_bytes": 4096,
                },
            )()
            with patch.object(device, "compile", return_value=compile_result):
                metrics = device.evaluate(
                    dirpath=project_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                    run_hil=False,
                )
            self.assertEqual(metrics.error_code, expected_error)

    def test_prepare_candidate_lrun_copies_workspace_and_stages_generated_outputs(self) -> None:
        """Ensure LRUN candidate prep stages generated outputs into Appli paths."""
        # LRUN candidate staging should copy the workspace and stage generated outputs into the candidate root.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            template_root = _build_lrun_project_tree(tmp_path / "crest_stm32_lrun")
            _write_text(template_root / "STM32CubeIDE" / "AppS" / "Debug" / "Src" / "stale.d", "old dep\n")
            _write_text(template_root / "STM32CubeIDE" / "Boot" / "Debug" / "Src" / "stale.o", "old obj\n")
            outputs_dir = tmp_path / "outputs"
            device = STM32NucleoN657X0QDevice(
                device_options={
                    "project_root": str(template_root),
                    "cpu_clock_mhz": 400,
                }
            )

            def _fake_generate(
                *,
                workspace_dir: Path,
                output_dir: Path,
                model_path: Path,
                **kwargs,
            ) -> None:
                """Create fake generated STM32 source output.

                Parameters
                ----------
                workspace_dir : Path
                    Directory used for workspace artifacts.
                output_dir : Path
                    Directory used for output artifacts.
                model_path : Path
                    Path to the model used by the helper.
                **kwargs : dict[str, object]
                    Keyword arguments forwarded to the helper.
                """
                del kwargs
                output_dir.mkdir(parents=True, exist_ok=True)
                workspace_dir.mkdir(parents=True, exist_ok=True)
                model_path.write_bytes(model_path.read_bytes())
                for name in EXPECTED_GENERATED_OUTPUTS:
                    target = output_dir / name
                    if name.endswith(".h"):
                        target.write_text(
                            "#define AI_NETWORK_DATA_ACTIVATIONS_SIZE (32768)\n",
                            encoding="utf-8",
                        )
                    else:
                        target.write_text("/* generated */\n", encoding="utf-8")

            fake_model = object()
            fake_training_data = type("TrainingData", (), {"inputs": [[[0.0] * 6] * 4]})()

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0._ensure_staging_tools"
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0._run_stedgeai_analyze"
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0._run_stedgeai_generate",
                side_effect=_fake_generate,
            ):
                def _write_tflite(*, output_name, **kwargs):
                    """Write tflite.

                    Parameters
                    ----------
                    output_name : object
                        Destination path where the fake exporter writes the TFLite payload.
                    **kwargs : dict[str, object]
                        Keyword arguments forwarded to the helper.
                    """
                    del kwargs
                    Path(output_name).write_bytes(b"tflite")

                with patch("crest.hardware.convert_to_tflite_model", side_effect=_write_tflite):
                    staged_root = device.prepare_candidate(
                        request=_make_prepare_request(
                            config=type(
                                "Config",
                                (),
                                {
                                    "training": type("Training", (), {"quantization": True})(),
                                    "device": type("Device", (), {"measured_inference_runs": 9})(),
                                },
                            )(),
                            outputs_dir=outputs_dir,
                            model=fake_model,
                            calibration_inputs=fake_training_data.inputs,
                        )
                    )

            self.assertTrue(staged_root.is_dir())
            self.assertEqual(staged_root.name, template_root.name)
            self.assertTrue((staged_root / "Appli" / "Src" / "network.c").is_file())
            self.assertTrue((staged_root / "Appli" / "Inc" / "network_data_params.h").is_file())
            self.assertTrue((staged_root / "Appli" / "Inc" / "crest_dut_phase_config.h").is_file())
            self.assertFalse((staged_root / "STM32CubeIDE" / "AppS" / "Debug" / "Src" / "stale.d").exists())
            self.assertFalse((staged_root / "STM32CubeIDE" / "Boot" / "Debug" / "Src" / "stale.o").exists())
            manifest = json.loads((staged_root / STAGED_MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["staged_workspace_root"], str(staged_root.resolve()))
            header_text = (
                staged_root / "Appli" / "Inc" / "crest_dut_phase_config.h"
            ).read_text(encoding="utf-8")
            self.assertIn("CREST_DUT_MEASURED_RUNS 9", header_text)

    def test_prepare_candidate_runs_analyze_before_generate(self) -> None:
        """Ensure ST Edge AI analyze preflights compatibility before generate.

        Returns
        -------
        None
        """
        # Candidate staging should analyze the workspace before code generation so generation can reuse validated metadata.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            template_root = _build_lrun_project_tree(tmp_path / "template")
            outputs_dir = tmp_path / "outputs"
            device = STM32NucleoN657X0QDevice(device_options={"template_root": str(template_root)})
            call_order: list[str] = []

            def _fake_analyze(*, workspace_dir: Path, output_dir: Path, model_path: Path) -> None:
                """Record the analyze step without touching disk.

                Parameters
                ----------
                workspace_dir : pathlib.Path
                    Ignored workspace path.
                output_dir : pathlib.Path
                    Ignored output path.
                model_path : pathlib.Path
                    Ignored model path.

                Returns
                -------
                None
                """
                del workspace_dir, output_dir, model_path
                call_order.append("analyze")

            def _fake_generate(
                *,
                workspace_dir: Path,
                output_dir: Path,
                model_path: Path,
                **kwargs,
            ) -> None:
                """Emit minimal generated outputs and record generation order.

                Parameters
                ----------
                workspace_dir : pathlib.Path
                    Ignored workspace path.
                output_dir : pathlib.Path
                    Destination directory populated with fake outputs.
                model_path : pathlib.Path
                    Ignored model path.
                **kwargs
                    Ignored code-generation keyword arguments.

                Returns
                -------
                None
                """
                del workspace_dir, model_path, kwargs
                call_order.append("generate")
                output_dir.mkdir(parents=True, exist_ok=True)
                for name in EXPECTED_GENERATED_OUTPUTS:
                    target = output_dir / name
                    if name.endswith(".h"):
                        target.write_text(
                            "#define AI_NETWORK_DATA_ACTIVATIONS_SIZE (8192)\n",
                            encoding="utf-8",
                        )
                    else:
                        target.write_text("/* generated */\n", encoding="utf-8")

            def _write_tflite(*, output_name, **kwargs):
                """Write a placeholder TFLite file for candidate staging.

                Parameters
                ----------
                output_name : str | pathlib.Path
                    Output file written by the test double.
                **kwargs
                    Ignored conversion keyword arguments.

                Returns
                -------
                None
                """
                del kwargs
                Path(output_name).write_bytes(b"tflite")

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0._ensure_staging_tools"
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0._run_stedgeai_analyze",
                side_effect=_fake_analyze,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0._run_stedgeai_generate",
                side_effect=_fake_generate,
            ), patch(
                "crest.hardware.convert_to_tflite_model",
                side_effect=_write_tflite,
            ):
                staged_root = device.prepare_candidate(
                    request=_make_prepare_request(
                        config=type(
                            "Config",
                            (),
                            {"training": type("Training", (), {"quantization": True})()},
                        )(),
                        outputs_dir=outputs_dir,
                        model=object(),
                        calibration_inputs=[[[0.0] * 6] * 4],
                    )
                )
                self.assertTrue(staged_root.is_dir())

            self.assertEqual(call_order, ["analyze", "generate"])

    def test_cleanup_prepared_candidate_removes_staged_root(self) -> None:
        """Ensure staged STM32 candidate directories are deleted after use.

        Returns
        -------
        None
        """
        # Candidate cleanup should remove the staged root so repeated runs do not accumulate stale workspaces.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = _build_lrun_project_tree(tmp_path / "candidate" / "crest_stm32_lrun")
            _write_text(
                project_root / STAGED_MANIFEST_NAME,
                "{}\n",
            )
            device = STM32NucleoN657X0QDevice()

            device.cleanup_prepared_candidate(project_root)

            self.assertFalse(project_root.parent.exists())

    def test_prepare_candidate_cleans_up_candidate_root_after_generate_failure(self) -> None:
        """Ensure failed STM32 staging does not leak per-candidate directories.

        Returns
        -------
        None
        """
        # Failed candidate generation should clean up the partially staged root before the error returns.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            template_root = _build_lrun_project_tree(tmp_path / "template")
            outputs_dir = tmp_path / "outputs"
            device = STM32NucleoN657X0QDevice(device_options={"template_root": str(template_root)})
            fake_training_data = type("TrainingData", (), {"inputs": [[[0.0] * 6] * 4]})()

            def _write_tflite(*, output_name, **kwargs):
                """Write tflite.

                Parameters
                ----------
                output_name : object
                    Destination path where the fake exporter writes the TFLite payload.
                **kwargs : dict[str, object]
                    Keyword arguments forwarded to the helper.
                """
                del kwargs
                Path(output_name).write_bytes(b"tflite")

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0._ensure_staging_tools"
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0._run_stedgeai_analyze"
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0._run_stedgeai_generate",
                side_effect=stm32_cube_clt.WorkflowError("generate failed"),
            ), patch(
                "crest.hardware.convert_to_tflite_model",
                side_effect=_write_tflite,
            ):
                with self.assertRaisesRegex(stm32_cube_clt.WorkflowError, "generate failed"):
                    device.prepare_candidate(
                        request=_make_prepare_request(
                            config=type(
                                "Config",
                                (),
                                {"training": type("Training", (), {"quantization": True})()},
                            )(),
                            outputs_dir=outputs_dir,
                            model=object(),
                            calibration_inputs=fake_training_data.inputs,
                        )
                    )

            stm32_outputs_root = outputs_dir / "stm32"
            self.assertTrue(stm32_outputs_root.exists())
            self.assertEqual(list(stm32_outputs_root.iterdir()), [])


class STM32HelperTests(unittest.TestCase):
    """Validate lower-level STM32 helper behavior."""

    def setUp(self) -> None:
        """Keep durable measurement records inside a per-test temporary directory."""
        evidence_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch("crest.microcontrollers.stm32_nucleo_n657x0.Path.cwd", return_value=Path(evidence_dir)))

    def test_resolve_weights_external_loader_autodiscovers_from_cubeprog_bin(self) -> None:
        """Ensure the backend finds the shipped Nucleo loader without a user path.

        Returns
        -------
        None
        """
        # External-weight programming should autodiscover its loader from the CubeProgrammer output when one is staged.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cubeprog_bin = tmp_path / "bin"
            loader_path = cubeprog_bin / "ExternalLoader" / DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME
            _write_text(loader_path, "loader")

            resolved = stm32_n657_backend._resolve_weights_external_loader(cubeprog_bin, None)

        self.assertEqual(resolved, loader_path.resolve())

    def test_resolve_elf_path_accepts_non_blink_name(self) -> None:
        """Ensure ELF discovery remains name-agnostic.

        Returns
        -------
        None
        """
        # ELF-path resolution should not assume a blink example name when locating staged runtime images.
        with tempfile.TemporaryDirectory() as tmpdir:
            debug_dir = Path(tmpdir)
            elf_path = debug_dir / "crest_phase2.elf"
            elf_path.write_bytes(b"elf")
            resolved = stm32_cube_clt.resolve_elf_path(debug_dir)
        self.assertEqual(resolved, elf_path.resolve())

    def test_parse_size_output_raises_workflow_error_when_size_tool_is_missing(self) -> None:
        """Ensure missing ``arm-none-eabi-size`` becomes ``WorkflowError``.

        Returns
        -------
        None
        """
        # Missing size tools should raise a workflow error so the backend reports a setup problem instead of bogus memory numbers.
        with tempfile.TemporaryDirectory() as tmpdir:
            elf_path = Path(tmpdir) / "crest_phase2.elf"
            elf_path.write_bytes(b"elf")
            with patch(
                "crest.microcontrollers.stm32_cube_clt.resolve_required_tool_path",
                side_effect=stm32_cube_clt.WorkflowError("arm-none-eabi-size was not provided."),
            ):
                with self.assertRaises(stm32_cube_clt.WorkflowError):
                    stm32_cube_clt.parse_size_output(elf_path)

    def test_classify_build_failure_treats_rom_overflow_as_flash(self) -> None:
        """Ensure ROM linker overflows map to flash overflow semantics.

        Returns
        -------
        None
        """
        # ROM-overflow diagnostics should map to flash overflow so pruning uses the correct memory dimension.
        classification = stm32_cube_clt.classify_build_failure(
            "ld: region `ROM' overflowed by 2048 bytes"
        )
        self.assertEqual(classification, "flash")

    def test_classify_build_failure_treats_lrun_code_image_budget_as_flash(self) -> None:
        """Ensure LRUN trusted-App budget overflows map to flash overflow semantics.

        Returns
        -------
        None
        """
        # LRUN code-image budget overflows should still prune as flash failures.
        classification = stm32_cube_clt.classify_build_failure(
            "STM trusted App image exceeds available LRUN code-image budget (130321 > 65536)."
        )
        self.assertEqual(classification, "flash")

    def test_run_command_wraps_host_os_errors_in_workflow_error(self) -> None:
        """Ensure host-side command launch failures are normalized.

        Returns
        -------
        None
        """
        # Host OS command failures should be wrapped in workflow errors so backend logs keep a stable failure type.
        with patch(
            "crest.microcontrollers.stm32_cube_clt.subprocess.run",
            side_effect=FileNotFoundError("make not found"),
        ):
            with self.assertRaises(stm32_cube_clt.WorkflowError):
                stm32_cube_clt._run_command(["make", "-C", "Debug", "all"])

    def test_run_command_prefers_stm32_programmer_bundled_qt_libs(self) -> None:
        """Ensure STM32CubeProgrammer does not inherit Conda Qt precedence.

        Returns
        -------
        None
        """
        # The Python process may need CONDA_PREFIX/lib, but the Qt-based ST
        # programmer must load its own bundled Qt libraries before Conda's.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cubeprog_root = tmp_path / "STM32CubeProgrammer"
            cubeprog_bin = cubeprog_root / "bin"
            cubeprog_lib = cubeprog_root / "lib"
            cubeprog_bin.mkdir(parents=True)
            cubeprog_lib.mkdir()
            programmer = cubeprog_bin / "STM32_Programmer_CLI"
            programmer.touch()
            captured_env = {}
            inherited_conda_lib = tmp_path / "inherited" / "conda" / "lib"
            inherited_usr_lib = tmp_path / "inherited" / "usr" / "lib"
            inherited_ld_library_path = os.pathsep.join(
                [str(inherited_conda_lib), str(inherited_usr_lib)]
            )

            def _fake_run(argv, **kwargs):
                """Capture the subprocess environment for assertions.

                Parameters
                ----------
                argv : list[str]
                    Command arguments provided to ``subprocess.run``.
                **kwargs : dict
                    Keyword arguments provided to ``subprocess.run``.

                Returns
                -------
                subprocess.CompletedProcess
                    Successful fake process result.
                """
                captured_env.update(kwargs.get("env") or {})
                return subprocess.CompletedProcess(argv, 0, "", "")

            with patch.dict(os.environ, {"LD_LIBRARY_PATH": inherited_ld_library_path}), patch(
                "crest.microcontrollers.stm32_cube_clt.subprocess.run",
                side_effect=_fake_run,
            ):
                stm32_cube_clt._run_command([str(programmer), "-q"])

            search_path = captured_env["LD_LIBRARY_PATH"].split(os.pathsep)
            self.assertEqual(search_path[0], str(cubeprog_lib))
            self.assertIn(str(inherited_conda_lib), search_path)

    def test_prepare_candidate_only_requires_staging_tools(self) -> None:
        """Ensure Phase 2 candidate prep does not demand upload/debug tools.

        Returns
        -------
        None
        """
        # Candidate staging should only require the tools needed for staging, not the full runtime toolchain.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            template_root = _build_lrun_project_tree(tmp_path / "template")
            outputs_dir = tmp_path / "outputs"
            device = STM32NucleoN657X0QDevice(device_options={"template_root": str(template_root)})

            def _write_tflite(*, output_name, **kwargs):
                """Write a placeholder TFLite file for staging tests.

                Parameters
                ----------
                output_name : str | os.PathLike[str]
                    Destination path where the TFLite model should be written.
                **kwargs
                    Additional conversion arguments ignored by this fake helper.

                Returns
                -------
                None
                    The helper writes a minimal TFLite payload to disk.
                """
                Path(output_name).write_bytes(b"tflite")

            def _fake_generate(
                *,
                workspace_dir: Path,
                output_dir: Path,
                model_path: Path,
                **kwargs,
            ) -> None:
                """Create placeholder generated files for staged candidate tests.

                Parameters
                ----------
                workspace_dir : Path
                    ST Edge AI workspace directory used during generation.
                output_dir : Path
                    Destination directory for generated C artifacts.
                model_path : Path
                    TFLite model path supplied to the generator.
                **kwargs
                    Additional generation options ignored by this fake helper.

                Returns
                -------
                None
                    The helper writes placeholder generated outputs to disk.
                """
                del model_path, kwargs
                workspace_dir.mkdir(parents=True, exist_ok=True)
                output_dir.mkdir(parents=True, exist_ok=True)
                for name in EXPECTED_GENERATED_OUTPUTS:
                    target = output_dir / name
                    if name.endswith(".h"):
                        target.write_text(
                            "#define AI_NETWORK_DATA_ACTIVATIONS_SIZE (2048)\n",
                            encoding="utf-8",
                        )
                    else:
                        target.write_text("/* generated */\n", encoding="utf-8")

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0._ensure_staging_tools"
            ) as staging_tools_mock, patch(
                "crest.microcontrollers.stm32_nucleo_n657x0._run_stedgeai_analyze"
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0._run_stedgeai_generate",
                side_effect=_fake_generate,
            ), patch(
                "crest.hardware.convert_to_tflite_model",
                side_effect=_write_tflite,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.resolve_required_tool_path"
            ) as tool_mock:
                staged_root = device.prepare_candidate(
                    request=_make_prepare_request(
                        config=type(
                            "Config",
                            (),
                            {"training": type("Training", (), {"quantization": True})()},
                        )(),
                        outputs_dir=outputs_dir,
                        model=object(),
                        calibration_inputs=[[[0.0] * 6] * 4],
                    )
                )

                self.assertTrue(staged_root.is_dir())
                staging_tools_mock.assert_called_once_with()
                tool_mock.assert_not_called()

    def test_evaluate_combined_external_flash_and_harness_uses_canonical_order(self) -> None:
        """Ensure the combined STM path preserves the documented operation order.

        Returns
        -------
        None
        """
        # Combined external-flash and harness flows should program and measure in the canonical order.
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = _build_lrun_project_tree(Path(tmpdir) / "stm")
            _write_text(
                project_root / "Appli" / "Inc" / "crest_dut_phase_config.h",
                "\n".join(
                    [
                        "#ifndef CREST_DUT_PHASE_CONFIG_H",
                        "#define CREST_DUT_PHASE_CONFIG_H",
                        "#define CREST_DUT_MEASURED_RUNS 5",
                        "#endif",
                        "",
                    ]
                ),
            )
            device = STM32NucleoN657X0QDevice(serial_port="/dev/ttyACM0", device_options={"project_root": str(project_root)})
            compile_result = type(
                "CompileResultDouble",
                (),
                {
                    "success": True,
                    "log": "ok",
                    "flash_bytes": 2222,
                    "ram_bytes": 1111,
                    "overflow_kind": None,
                    "build_dir": project_root / "STM32CubeIDE" / "Boot" / "Debug",
                    "arena_bytes": 4096,
                    "external_flash_bytes": 4096,
                },
            )()
            telemetry = stm32_runtime.STM32RuntimeTelemetry(
                latency_s=0.001,
                serial_log=["STM32_AI_INIT=OK", "DUT READY", "STM32_AI_RUN=OK"],
                power_metrics={
                    "clock_hz": 600000000.0,
                    "sequence": 1.0,
                    "runs": 5,
                    "phase": "back_to_back",
                    "timer_output_s": 0.015,
                    "timer_per_inference_s": 0.003,
                    "timer_per_window_s": 0.003,
                    "wake_recovery_us": 1200.0,
                    "wake_overshoot_us": 35.0,
                    "rtc_sleep_total_ms": 1500.0,
                    "deadline_miss_count": 2,
                    "slept_slots": 4,
                    "stop_entries": 4,
                    "rtc_clock_hz_nominal": 32768.0,
                    "rtc_clock_source": "LSE",
                    "cadence_timing_quality": "crystal",
                    "stop_mode_variant": "system_stop_mainreg_wfi",
                },
            )
            # Repaired DUT telemetry: RTC aggregate covers all reported releases.
            telemetry.serial_log.extend([
                "measurement protocol output: rtc_wall_v4_dwt_active_stop_evidence_awake_tail",
                "cadence budget (ms): 200",
            ])
            telemetry.power_metrics.update(runs=5, phase="back_to_back",
                timer_per_window_s=0.015, rtc_clock_source="LSE", rtc_clock_hz_nominal=32768)
            trace: list[str] = []

            class _TracingMonitor(_FakeSerialMonitor):
                """Tests covering tracing monitor behavior."""

                def __init__(self, port: str, baud: int, label: str) -> None:
                    """Record DUT monitor creation for ordering assertions.

                    Parameters
                    ----------
                    port : str
                        DUT serial port.
                    baud : int
                        UART baud rate.
                    label : str
                        Human-readable monitor label.

                    Returns
                    -------
                    None
                    """
                    trace.append("open_dut")
                    super().__init__(port, baud, label)

            def _open_harness(*args, **kwargs):
                """Record harness-open ordering and return a fake serial session.

                Parameters
                ----------
                *args
                    Ignored serial constructor positional arguments.
                **kwargs
                    Ignored serial constructor keyword arguments.

                Returns
                -------
                _FakeHarnessSerial
                    Fake harness serial session.
                """
                del args, kwargs
                trace.append("open_harness")
                return _FakeHarnessSerial()

            def _prime_harness(**kwargs):
                """Return a ready harness session and record ordering.

                Parameters
                ----------
                **kwargs
                    Ignored harness priming keyword arguments.

                Returns
                -------
                object
                    Prime-result stand-in with ``harness_ready=True``.
                """
                del kwargs
                trace.append("prime_harness")
                return type(
                    "PrimeResult",
                    (),
                    {"harness_ready": True, "harness_log": ["HARNESS READY"], "error": None},
                )()

            def _wait_done(**kwargs):
                """Return a completed harness result and record ordering.

                Parameters
                ----------
                **kwargs
                    Ignored harness wait keyword arguments.

                Returns
                -------
                object
                    Done-result stand-in with harness telemetry.
                """
                del kwargs
                trace.append("wait_done")
                return type(
                    "DoneResult",
                    (),
                    {
                        "harness_done": True,
                        "runs_harness": 5,
                        "harness_log": [
                            "HARNESS READY",
                            "runs: 5",
                            "energy output: 1.25",
                            "avg power output: 2.5",
                            "avg current output: 0.5",
                            "bus voltage output: 5.0",
                            "idle power baseline: 0.1",
                            "harness timer output: 0.003",
                            "DONE",
                        ],
                        "error": None,
                    },
                )()

            with patch.object(device, "compile", return_value=compile_result), patch.object(
                device,
                "_storage_power_metrics",
                return_value={"weight_storage_mode": "external_flash", "external_flash_bytes": 4096.0},
            ), patch.object(
                device,
                "_program_runtime_images",
                side_effect=lambda *_args, **_kwargs: trace.append("program_external") or {
                    "weight_storage_mode": "external_flash",
                    "external_flash_bytes": 4096.0,
                },
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.resolve_elf_path",
                return_value=project_root / "STM32CubeIDE" / "Boot" / "Debug" / "Template_LRUN_FSBL.elf",
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.arduino_base.ensure_harness_firmware"
            ) as harness_mock, patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
                _TracingMonitor,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.serial.Serial",
                side_effect=_open_harness,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.hil_protocol.prime_harness_session",
                side_effect=_prime_harness,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.hil_protocol.wait_for_harness_done",
                side_effect=_wait_done,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.debug_load_elf",
                side_effect=lambda **kwargs: trace.append("debug_load") or "upload ok",
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.execute_runtime_session",
                side_effect=lambda *args, **kwargs: trace.append("runtime_session") or telemetry,
            ):
                metrics = device.evaluate(
                    dirpath=project_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                    serial_port="/dev/ttyACM0",
                    run_hil=True,
                    measured_inference_runs=5,
                    harness_serial_port="/dev/ttyACM1",
                )

            self.assertEqual(
                trace,
                [
                    "program_external",
                    "open_dut",
                    "open_harness",
                    "prime_harness",
                    "debug_load",
                    "runtime_session",
                    "wait_done",
                ],
            )
            self.assertEqual(
                harness_mock.call_args.kwargs["build_defines"]["CREST_INFERENCE_RUNS"],
                5,
            )
            self.assertEqual(metrics.error_code, HIL_ERROR_OK)
            self.assertEqual(metrics.latency_s, 0.003)
            self.assertEqual(metrics.external_flash_bytes, 4096)
            self.assertEqual(metrics.power_metrics["weight_storage_mode"], "external_flash")
            self.assertEqual(metrics.power_metrics["runs"], 5)
            self.assertEqual(metrics.power_metrics["timer_output_s"], 0.015)
            self.assertEqual(metrics.power_metrics["timer_per_inference_s"], 0.003)
            self.assertEqual(metrics.power_metrics["timer_per_window_s"], 0.015)
            self.assertEqual(metrics.power_metrics["wake_recovery_us"], 1200.0)
            self.assertEqual(metrics.power_metrics["wake_overshoot_us"], 35.0)
            self.assertEqual(metrics.power_metrics["rtc_sleep_total_ms"], 1500.0)
            self.assertEqual(metrics.power_metrics["deadline_miss_count"], 2)
            self.assertEqual(metrics.power_metrics["slept_slots"], 4)
            self.assertEqual(metrics.power_metrics["stop_entries"], 4)
            self.assertEqual(metrics.power_metrics["rtc_clock_hz_nominal"], 32768.0)
            self.assertEqual(metrics.power_metrics["rtc_clock_source"], "LSE")
            self.assertEqual(metrics.power_metrics["cadence_timing_quality"], "crystal")
            self.assertEqual(
                metrics.power_metrics["stop_mode_variant"],
                "system_stop_mainreg_wfi",
            )

    def test_evaluate_harness_ready_timeout_maps_to_integrity_error(self) -> None:
        """Ensure missing HARNESS READY becomes a stable latency failure.

        Returns
        -------
        None
        """
        # If the harness never reports READY after runtime starts, classify it as a latency-side timeout rather than upload failure.
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = _build_lrun_project_tree(Path(tmpdir) / "stm")
            _write_text(
                project_root / "Appli" / "Inc" / "crest_dut_phase_config.h",
                "#define CREST_DUT_MEASURED_RUNS 1\n",
            )
            device = STM32NucleoN657X0QDevice(serial_port="/dev/ttyACM0", device_options={"project_root": str(project_root)})
            compile_result = type(
                "CompileResultDouble",
                (),
                {
                    "success": True,
                    "log": "ok",
                    "flash_bytes": 2222,
                    "ram_bytes": 1111,
                    "overflow_kind": None,
                    "build_dir": project_root / "STM32CubeIDE" / "Boot" / "Debug",
                    "arena_bytes": 4096,
                    "external_flash_bytes": None,
                },
            )()

            with patch.object(device, "compile", return_value=compile_result), patch.object(
                device,
                "_storage_power_metrics",
                return_value={"weight_storage_mode": "embedded", "external_flash_bytes": -1.0},
            ), patch.object(
                device,
                "_program_runtime_images",
                return_value={"weight_storage_mode": "embedded", "external_flash_bytes": -1.0},
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.resolve_elf_path",
                return_value=project_root / "STM32CubeIDE" / "Boot" / "Debug" / "Template_LRUN_FSBL.elf",
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.arduino_base.ensure_harness_firmware"
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
                _FakeSerialMonitor,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.serial.Serial",
                return_value=_FakeHarnessSerial(),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.hil_protocol.prime_harness_session",
                return_value=type(
                    "PrimeResult",
                    (),
                    {"harness_ready": False, "harness_log": [], "error": "timeout"},
                )(),
            ):
                metrics = device.evaluate(
                    dirpath=project_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                    serial_port="/dev/ttyACM0",
                    run_hil=True,
                    harness_serial_port="/dev/ttyACM1",
                )

            self.assertEqual(metrics.error_code, HIL_ERROR_INTEGRITY)
            self.assertEqual(metrics.power_metrics["backend_error_kind"], "harness_ready_timeout")

    def test_evaluate_harness_done_timeout_maps_to_integrity_error(self) -> None:
        """Ensure missing harness DONE becomes a stable latency failure.

        Returns
        -------
        None
        """
        # If the harness starts but never reports DONE, keep the failure in the latency bucket.
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = _build_lrun_project_tree(Path(tmpdir) / "stm")
            _write_text(
                project_root / "Appli" / "Inc" / "crest_dut_phase_config.h",
                "#define CREST_DUT_MEASURED_RUNS 5\n",
            )
            device = STM32NucleoN657X0QDevice(serial_port="/dev/ttyACM0", device_options={"project_root": str(project_root)})
            compile_result = type(
                "CompileResultDouble",
                (),
                {
                    "success": True,
                    "log": "ok",
                    "flash_bytes": 2222,
                    "ram_bytes": 1111,
                    "overflow_kind": None,
                    "build_dir": project_root / "STM32CubeIDE" / "Boot" / "Debug",
                    "arena_bytes": 4096,
                    "external_flash_bytes": None,
                },
            )()
            telemetry = stm32_runtime.STM32RuntimeTelemetry(
                latency_s=0.003,
                serial_log=["STM32_AI_INIT=OK", "DUT READY", "STM32_AI_RUN=OK"],
                power_metrics={
                    "clock_hz": 600000000.0,
                    "sequence": 1.0,
                    "runs": 5,
                    "phase": "back_to_back",
                },
            )

            with patch.object(device, "compile", return_value=compile_result), patch.object(
                device,
                "_storage_power_metrics",
                return_value={"weight_storage_mode": "embedded", "external_flash_bytes": -1.0},
            ), patch.object(
                device,
                "_program_runtime_images",
                return_value={"weight_storage_mode": "embedded", "external_flash_bytes": -1.0},
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.resolve_elf_path",
                return_value=project_root / "STM32CubeIDE" / "Boot" / "Debug" / "Template_LRUN_FSBL.elf",
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.arduino_base.ensure_harness_firmware"
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
                _FakeSerialMonitor,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.serial.Serial",
                return_value=_FakeHarnessSerial(),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.hil_protocol.prime_harness_session",
                return_value=type(
                    "PrimeResult",
                    (),
                    {"harness_ready": True, "harness_log": ["HARNESS READY"], "error": None},
                )(),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.debug_load_elf",
                return_value="upload ok",
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.execute_runtime_session",
                return_value=telemetry,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.hil_protocol.wait_for_harness_done",
                return_value=type(
                    "DoneResult",
                    (),
                    {
                        "harness_done": False,
                        "runs_harness": None,
                        "harness_log": ["HARNESS READY"],
                        "error": "timeout",
                    },
                )(),
            ):
                metrics = device.evaluate(
                    dirpath=project_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                    serial_port="/dev/ttyACM0",
                    run_hil=True,
                    harness_serial_port="/dev/ttyACM1",
                )

            self.assertEqual(metrics.error_code, HIL_ERROR_INTEGRITY)
            self.assertEqual(metrics.power_metrics["backend_error_kind"], "harness_done_timeout")

    def test_compile_external_flash_overflow_maps_to_flash_failure(self) -> None:
        """Ensure oversized staged external weights fail as flash overflow.

        Returns
        -------
        None
        """
        # External-flash overflows should still surface as flash failures for pruning and reporting.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = _build_lrun_project_tree(tmp_path / "crest_stm32_lrun")
            weights_blob_path = tmp_path / "network_data.bin"
            manifest_path = project_root / STAGED_MANIFEST_NAME
            manifest_path.write_text(
                "{\n"
                f'  "staged_workspace_root": "{project_root.resolve()}",\n'
                '  "weight_storage_mode": "external_flash",\n'
                '  "weights_blob_size": 4097,\n'
                f'  "weights_blob_path": "{weights_blob_path}"\n'
                "}\n",
                encoding="utf-8",
            )
            device = STM32NucleoN657X0QDevice(
                device_options={
                    "template_root": str(project_root),
                    "weight_storage_mode": "external_flash",
                    "max_external_flash_bytes": 4096,
                }
            )

            result = device.compile(
                sketch_path=project_root,
                arena_kb=-1,
                window_size=200,
                num_channels=6,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.overflow_kind, "flash")
        self.assertEqual(result.external_flash_bytes, 4097)

    def test_compile_success_lrun_returns_artifacts_and_copy_window(self) -> None:
        """Ensure LRUN compile reports app/boot artifacts and copy-window metadata."""
        # Successful LRUN compiles should report both staged artifacts and the copy window used to place weights.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_lrun_project_tree(Path(tmpdir) / "staged")
            device = STM32NucleoN657X0QDevice(device_options={"project_root": str(staged_root)})

            def _fake_build(*, project_root: Path, jobs: int, clean: bool):
                """Build a fake STM32 project artifact.

                Parameters
                ----------
                project_root : Path
                    Root directory for project artifacts.
                jobs : int
                    Parallel job count received by the fake build step.
                clean : bool
                    Whether the fake build was requested as a clean build.

                Returns
                -------
                object
                    Build result returned by the fake STM32 build step.
                """
                del jobs, clean
                debug_dir = project_root / "Debug"
                if project_root.name == "AppS":
                    elf_path = debug_dir / "Template_LRUN_AppS.elf"
                    _write_text(debug_dir / "Template_LRUN_AppS.bin", "appbin")
                else:
                    elf_path = debug_dir / "Template_LRUN_FSBL.elf"
                _write_text(elf_path, "elf")
                return stm32_cube_clt.BuildResult(log="build ok", debug_dir=debug_dir, elf_path=elf_path)

            def _fake_size(elf_path: Path):
                """Return fake STM32 size data.

                Parameters
                ----------
                elf_path : Path
                    Path to the ELF used by the helper.

                Returns
                -------
                object
                    Size result returned by the fake size parser.
                """
                if "AppS" in elf_path.name:
                    return stm32_cube_clt.SizeResult(elf_flash_bytes=120000, ram_bytes=64000, raw_output="app")
                return stm32_cube_clt.SizeResult(elf_flash_bytes=32000, ram_bytes=12000, raw_output="boot")

            def _fake_sign(**kwargs):
                """Write and report a fake signed binary.

                Parameters
                ----------
                **kwargs : dict[str, object]
                    Keyword arguments forwarded to the helper.

                Returns
                -------
                object
                    Signed-binary result returned by the fake signing step.
                """
                output_bin = Path(kwargs["output_bin"])
                output_bin.parent.mkdir(parents=True, exist_ok=True)
                output_bin.write_bytes(b"x" * 130321)
                return stm32_cube_clt.SignedBinaryResult(log="sign ok", output_bin=output_bin)

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.build_project",
                side_effect=_fake_build,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.parse_size_output",
                side_effect=_fake_size,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.sign_binary",
                side_effect=_fake_sign,
            ):
                result = device.compile(
                    sketch_path=staged_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                )

        self.assertTrue(result.success)
        self.assertEqual(result.heap_bytes, 0x2000)
        self.assertEqual(result.stack_bytes, 0x4000)
        self.assertEqual(result.build_dir, (staged_root / "STM32CubeIDE" / "Boot" / "Debug").resolve())
        self.assertEqual(
            result.boot_elf_path,
            (staged_root / "STM32CubeIDE" / "Boot" / "Debug" / "Template_LRUN_FSBL.elf").resolve(),
        )
        self.assertEqual(
            result.app_elf_path,
            (staged_root / "STM32CubeIDE" / "AppS" / "Debug" / "Template_LRUN_AppS.elf").resolve(),
        )
        self.assertEqual(
            result.signed_app_bin_path,
            (staged_root / "STM32CubeIDE" / "AppS" / "Debug" / "Template_LRUN_AppS-trusted.bin").resolve(),
        )
        self.assertEqual(result.flash_bytes, 130321)
        self.assertEqual(result.fsbl_copy_window_bytes, 131072)
        self.assertIn("boot", result.log)

    def test_compile_lrun_reuses_unchanged_artifacts(self) -> None:
        """Ensure repeated LRUN compile skips rebuild/sign when inputs are unchanged."""
        # Unchanged LRUN inputs should reuse existing artifacts so repeat builds do not restage identical binaries.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_lrun_project_tree(Path(tmpdir) / "staged")
            manifest_path = staged_root / STAGED_MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(
                    {
                        "candidate_root": str(staged_root.parent),
                        "generated_output_dir": str(staged_root / "generated"),
                        "weight_storage_mode": "embedded",
                        "staged_workspace_root": str(staged_root.resolve()),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            device = STM32NucleoN657X0QDevice(device_options={"project_root": str(staged_root)})

            def _fake_build(*, project_root: Path, jobs: int, clean: bool):
                """Build a fake STM32 project artifact.

                Parameters
                ----------
                project_root : Path
                    Root directory for project artifacts.
                jobs : int
                    Parallel job count received by the fake build step.
                clean : bool
                    Whether the fake build was requested as a clean build.

                Returns
                -------
                object
                    Build result returned by the fake STM32 build step.
                """
                del jobs, clean
                debug_dir = project_root / "Debug"
                if project_root.name == "AppS":
                    elf_path = debug_dir / "Template_LRUN_AppS.elf"
                    _write_text(debug_dir / "Template_LRUN_AppS.bin", "appbin")
                else:
                    elf_path = debug_dir / "Template_LRUN_FSBL.elf"
                _write_text(elf_path, "elf")
                return stm32_cube_clt.BuildResult(log="build ok", debug_dir=debug_dir, elf_path=elf_path)

            def _fake_size(elf_path: Path):
                """Return fake STM32 size data.

                Parameters
                ----------
                elf_path : Path
                    Path to the ELF used by the helper.

                Returns
                -------
                object
                    Size result returned by the fake size parser.
                """
                if "AppS" in elf_path.name:
                    return stm32_cube_clt.SizeResult(elf_flash_bytes=120000, ram_bytes=64000, raw_output="app")
                return stm32_cube_clt.SizeResult(elf_flash_bytes=32000, ram_bytes=12000, raw_output="boot")

            def _fake_sign(**kwargs):
                """Write and report a fake signed binary.

                Parameters
                ----------
                **kwargs : dict[str, object]
                    Keyword arguments forwarded to the helper.

                Returns
                -------
                object
                    Signed-binary result returned by the fake signing step.
                """
                output_bin = Path(kwargs["output_bin"])
                output_bin.parent.mkdir(parents=True, exist_ok=True)
                output_bin.write_bytes(b"x" * 130321)
                return stm32_cube_clt.SignedBinaryResult(log="sign ok", output_bin=output_bin)

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.build_project",
                side_effect=_fake_build,
            ) as build_mock, patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.parse_size_output",
                side_effect=_fake_size,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.sign_binary",
                side_effect=_fake_sign,
            ) as sign_mock:
                first = device.compile(
                    sketch_path=staged_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                )

            self.assertTrue(first.success, msg=first.log)
            self.assertEqual(build_mock.call_count, 2)
            self.assertEqual(sign_mock.call_count, 1)

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.build_project",
                side_effect=AssertionError("build should be reused"),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.parse_size_output",
                side_effect=_fake_size,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.sign_binary",
                side_effect=AssertionError("sign should be reused"),
            ):
                second = device.compile(
                    sketch_path=staged_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                )

            self.assertTrue(second.success, msg=second.log)
            self.assertIn("AppS build reused", second.log)
            self.assertIn("signed App reused", second.log)
            self.assertIn("Boot build reused", second.log)

    def test_lrun_build_input_hashes_include_setup_managed_vendor_trees(self) -> None:
        """Ensure LRUN reuse invalidates when setup-managed vendor inputs change."""
        # LRUN input hashes should include setup-managed vendor trees so cache invalidation follows the real source set.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_lrun_project_tree(Path(tmpdir) / "staged")
            paths = stm32_n657_backend._resolve_workspace_paths(project_root=staged_root)

            app_hash_before = stm32_n657_backend._lrun_app_build_input_hash(paths)
            boot_hash_before = stm32_n657_backend._lrun_boot_build_input_hash(paths)

            _write_text(staged_root / "Secure_nsclib" / "secure_nsc.h", "#define SECURE_GATEWAY 1\n")
            _write_text(
                staged_root / "Middlewares" / "ST" / "STM32_ExtMem_Manager" / "boot_stub.c",
                "void extmem_boot_stub(void) { int updated = 1; }\n",
            )

            app_hash_after = stm32_n657_backend._lrun_app_build_input_hash(paths)
            boot_hash_after = stm32_n657_backend._lrun_boot_build_input_hash(paths)

            self.assertNotEqual(app_hash_before, app_hash_after)
            self.assertNotEqual(boot_hash_before, boot_hash_after)

    def test_compile_lrun_rebuilds_boot_cleanly_when_boot_inputs_change(self) -> None:
        """Ensure Boot rebuilds cleanly when Boot inputs change without a copy-window change."""
        # Boot artifacts should rebuild when their inputs change so stale secure-boot binaries do not get reused.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_lrun_project_tree(Path(tmpdir) / "staged")
            manifest_path = staged_root / STAGED_MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(
                    {
                        "candidate_root": str(staged_root.parent),
                        "generated_output_dir": str(staged_root / "generated"),
                        "weight_storage_mode": "embedded",
                        "staged_workspace_root": str(staged_root.resolve()),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            device = STM32NucleoN657X0QDevice(device_options={"project_root": str(staged_root)})

            def _fake_build(*, project_root: Path, jobs: int, clean: bool):
                """Build a fake STM32 project artifact.

                Parameters
                ----------
                project_root : Path
                    Root directory for project artifacts.
                jobs : int
                    Parallel job count received by the fake build step.
                clean : bool
                    Whether the fake build was requested as a clean build.

                Returns
                -------
                object
                    Build result returned by the fake STM32 build step.
                """
                del jobs, clean
                debug_dir = project_root / "Debug"
                if project_root.name == "AppS":
                    elf_path = debug_dir / "Template_LRUN_AppS.elf"
                    _write_text(debug_dir / "Template_LRUN_AppS.bin", "appbin")
                else:
                    elf_path = debug_dir / "Template_LRUN_FSBL.elf"
                _write_text(elf_path, "elf")
                return stm32_cube_clt.BuildResult(log="build ok", debug_dir=debug_dir, elf_path=elf_path)

            def _fake_size(elf_path: Path):
                """Return fake STM32 size data.

                Parameters
                ----------
                elf_path : Path
                    Path to the ELF used by the helper.

                Returns
                -------
                object
                    Size result returned by the fake size parser.
                """
                if "AppS" in elf_path.name:
                    return stm32_cube_clt.SizeResult(elf_flash_bytes=120000, ram_bytes=64000, raw_output="app")
                return stm32_cube_clt.SizeResult(elf_flash_bytes=32000, ram_bytes=12000, raw_output="boot")

            def _fake_sign(**kwargs):
                """Write and report a fake signed binary.

                Parameters
                ----------
                **kwargs : dict[str, object]
                    Keyword arguments forwarded to the helper.

                Returns
                -------
                object
                    Signed-binary result returned by the fake signing step.
                """
                output_bin = Path(kwargs["output_bin"])
                output_bin.parent.mkdir(parents=True, exist_ok=True)
                output_bin.write_bytes(b"x" * 130321)
                return stm32_cube_clt.SignedBinaryResult(log="sign ok", output_bin=output_bin)

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.build_project",
                side_effect=_fake_build,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.parse_size_output",
                side_effect=_fake_size,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.sign_binary",
                side_effect=_fake_sign,
            ):
                first = device.compile(
                    sketch_path=staged_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                )

            self.assertTrue(first.success, msg=first.log)
            _write_text(
                staged_root / "FSBL" / "Inc" / "stm32_extmem_conf.h",
                "#define EXTMEM_LRUN_SOURCE_SIZE 0x00020000\n/* boot source changed */\n",
            )

            build_calls: list[tuple[str, bool]] = []

            def _second_build(*, project_root: Path, jobs: int, clean: bool):
                """Exercise second build behavior.

                Parameters
                ----------
                project_root : Path
                    Root directory for project artifacts.
                jobs : int
                    Parallel job count received by the second fake build step.
                clean : bool
                    Whether the second fake build was requested as clean.

                Returns
                -------
                object
                    Build result returned by the second fake build step.
                """
                del jobs
                build_calls.append((project_root.name, clean))
                elf_path = project_root / "Debug" / "Template_LRUN_FSBL.elf"
                _write_text(elf_path, "elf")
                return stm32_cube_clt.BuildResult(
                    log="boot rebuilt",
                    debug_dir=project_root / "Debug",
                    elf_path=elf_path,
                )

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.build_project",
                side_effect=_second_build,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.parse_size_output",
                side_effect=_fake_size,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.sign_binary",
                side_effect=AssertionError("sign should be reused"),
            ):
                second = device.compile(
                    sketch_path=staged_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                )

            self.assertTrue(second.success, msg=second.log)
            self.assertEqual(build_calls, [("Boot", True)])

    def test_compile_lrun_rejects_copy_window_overlap_with_weights_region(self) -> None:
        """Ensure LRUN validates the aligned Boot copy window, not the raw signed App size."""
        # Copy-window overlap with the weights region should fail before flashing a corrupt memory layout.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_lrun_project_tree(Path(tmpdir) / "staged")
            device = STM32NucleoN657X0QDevice(
                device_options={"project_root": str(staged_root), "appli_flash_address": "0x70FFFF00"}
            )

            def _fake_build(*, project_root: Path, jobs: int, clean: bool):
                """Build a fake STM32 project artifact.

                Parameters
                ----------
                project_root : Path
                    Root directory for project artifacts.
                jobs : int
                    Parallel job count received by the fake build step.
                clean : bool
                    Whether the fake build was requested as a clean build.

                Returns
                -------
                object
                    Build result returned by the fake STM32 build step.
                """
                del jobs, clean
                debug_dir = project_root / "Debug"
                if project_root.name == "AppS":
                    elf_path = debug_dir / "Template_LRUN_AppS.elf"
                    _write_text(debug_dir / "Template_LRUN_AppS.bin", "appbin")
                else:
                    elf_path = debug_dir / "Template_LRUN_FSBL.elf"
                _write_text(elf_path, "elf")
                return stm32_cube_clt.BuildResult(log="build ok", debug_dir=debug_dir, elf_path=elf_path)

            def _fake_size(elf_path: Path):
                """Return fake STM32 size data.

                Parameters
                ----------
                elf_path : Path
                    Path to the ELF used by the helper.

                Returns
                -------
                object
                    Size result returned by the fake size parser.
                """
                if "AppS" in elf_path.name:
                    return stm32_cube_clt.SizeResult(elf_flash_bytes=256, ram_bytes=64000, raw_output="app")
                return stm32_cube_clt.SizeResult(elf_flash_bytes=32000, ram_bytes=12000, raw_output="boot")

            def _fake_sign(**kwargs):
                """Write and report a fake signed binary.

                Parameters
                ----------
                **kwargs : dict[str, object]
                    Keyword arguments forwarded to the helper.

                Returns
                -------
                object
                    Signed-binary result returned by the fake signing step.
                """
                output_bin = Path(kwargs["output_bin"])
                output_bin.parent.mkdir(parents=True, exist_ok=True)
                output_bin.write_bytes(b"x" * 128)
                return stm32_cube_clt.SignedBinaryResult(log="sign ok", output_bin=output_bin)

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.build_project",
                side_effect=_fake_build,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.parse_size_output",
                side_effect=_fake_size,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.sign_binary",
                side_effect=_fake_sign,
            ):
                result = device.compile(
                    sketch_path=staged_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                )

        self.assertFalse(result.success)
        self.assertIn("LRUN copy window overlaps the weights region.", result.log)

    def test_compile_lrun_classifies_trusted_app_budget_overflow_as_flash(self) -> None:
        """Ensure LRUN trusted-App budget failures surface as flash overflow."""
        # Trusted-app budget overflows should classify as flash so memory pruning stays consistent across security modes.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_lrun_project_tree(Path(tmpdir) / "staged")
            device = STM32NucleoN657X0QDevice(device_options={"project_root": str(staged_root)})

            def _fake_build(*, project_root: Path, jobs: int, clean: bool):
                """Build a fake STM32 project artifact.

                Parameters
                ----------
                project_root : Path
                    Root directory for project artifacts.
                jobs : int
                    Parallel job count received by the fake build step.
                clean : bool
                    Whether the fake build was requested as a clean build.

                Returns
                -------
                object
                    Build result returned by the fake STM32 build step.
                """
                del jobs, clean
                debug_dir = project_root / "Debug"
                if project_root.name == "AppS":
                    elf_path = debug_dir / "Template_LRUN_AppS.elf"
                    _write_text(debug_dir / "Template_LRUN_AppS.bin", "appbin")
                else:
                    elf_path = debug_dir / "Template_LRUN_FSBL.elf"
                _write_text(elf_path, "elf")
                return stm32_cube_clt.BuildResult(log="build ok", debug_dir=debug_dir, elf_path=elf_path)

            def _fake_size(elf_path: Path):
                """Return fake STM32 size data.

                Parameters
                ----------
                elf_path : Path
                    Path to the ELF used by the helper.

                Returns
                -------
                object
                    Size result returned by the fake size parser.
                """
                if "AppS" in elf_path.name:
                    return stm32_cube_clt.SizeResult(elf_flash_bytes=120000, ram_bytes=64000, raw_output="app")
                return stm32_cube_clt.SizeResult(elf_flash_bytes=32000, ram_bytes=12000, raw_output="boot")

            def _fake_sign(**kwargs):
                """Write and report a fake signed binary.

                Parameters
                ----------
                **kwargs : dict[str, object]
                    Keyword arguments forwarded to the helper.

                Returns
                -------
                object
                    Signed-binary result returned by the fake signing step.
                """
                output_bin = Path(kwargs["output_bin"])
                output_bin.parent.mkdir(parents=True, exist_ok=True)
                output_bin.write_bytes(b"x" * (device.spec.max_flash_bytes + 1))
                return stm32_cube_clt.SignedBinaryResult(log="sign ok", output_bin=output_bin)

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.build_project",
                side_effect=_fake_build,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.parse_size_output",
                side_effect=_fake_size,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.sign_binary",
                side_effect=_fake_sign,
            ):
                result = device.compile(
                    sketch_path=staged_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                )

        self.assertFalse(result.success)
        self.assertEqual(result.overflow_kind, "flash")
        self.assertIn("exceeds available LRUN code-image budget", result.log)

    def test_compile_lrun_flash_accounting_excludes_debug_loaded_boot(self) -> None:
        """Ensure LRUN flash accounting only tracks the trusted App image."""
        # Flash accounting should exclude debug-loaded boot images so reported usage matches deployment reality.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_lrun_project_tree(Path(tmpdir) / "staged")
            device = STM32NucleoN657X0QDevice(device_options={"project_root": str(staged_root)})

            def _fake_build(*, project_root: Path, jobs: int, clean: bool):
                """Build a fake STM32 project artifact.

                Parameters
                ----------
                project_root : Path
                    Root directory for project artifacts.
                jobs : int
                    Parallel job count received by the fake build step.
                clean : bool
                    Whether the fake build was requested as a clean build.

                Returns
                -------
                object
                    Build result returned by the fake STM32 build step.
                """
                del jobs, clean
                debug_dir = project_root / "Debug"
                if project_root.name == "AppS":
                    elf_path = debug_dir / "Template_LRUN_AppS.elf"
                    _write_text(debug_dir / "Template_LRUN_AppS.bin", "appbin")
                else:
                    elf_path = debug_dir / "Template_LRUN_FSBL.elf"
                _write_text(elf_path, "elf")
                return stm32_cube_clt.BuildResult(log="build ok", debug_dir=debug_dir, elf_path=elf_path)

            def _fake_size(elf_path: Path):
                """Return fake STM32 size data.

                Parameters
                ----------
                elf_path : Path
                    Path to the ELF used by the helper.

                Returns
                -------
                object
                    Size result returned by the fake size parser.
                """
                if "AppS" in elf_path.name:
                    return stm32_cube_clt.SizeResult(elf_flash_bytes=4096, ram_bytes=64000, raw_output="app")
                return stm32_cube_clt.SizeResult(
                    elf_flash_bytes=device.spec.max_flash_bytes,
                    ram_bytes=12000,
                    raw_output="boot",
                )

            def _fake_sign(**kwargs):
                """Write and report a fake signed binary.

                Parameters
                ----------
                **kwargs : dict[str, object]
                    Keyword arguments forwarded to the helper.

                Returns
                -------
                object
                    Signed-binary result returned by the fake signing step.
                """
                output_bin = Path(kwargs["output_bin"])
                output_bin.parent.mkdir(parents=True, exist_ok=True)
                output_bin.write_bytes(b"x" * 4096)
                return stm32_cube_clt.SignedBinaryResult(log="sign ok", output_bin=output_bin)

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.build_project",
                side_effect=_fake_build,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.parse_size_output",
                side_effect=_fake_size,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.sign_binary",
                side_effect=_fake_sign,
            ):
                result = device.compile(
                    sketch_path=staged_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                )

        self.assertTrue(result.success, msg=result.log)
        self.assertEqual(result.flash_bytes, 4096)

    def test_read_storage_manifest_rejects_workspace_root_mismatch(self) -> None:
        """Ensure staged manifests cannot be reused from another workspace root."""
        # Storage manifests should reject mismatched workspace roots so staging never reuses artifacts from another workspace.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_lrun_project_tree(Path(tmpdir) / "staged")
            wrong_root = Path(tmpdir) / "other-staged-root"
            (staged_root / STAGED_MANIFEST_NAME).write_text(
                json.dumps(
                    {
                        "staged_workspace_root": str(wrong_root.resolve()),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            device = STM32NucleoN657X0QDevice(device_options={"project_root": str(staged_root)})

            with self.assertRaises(stm32_cube_clt.WorkflowError):
                device._read_storage_manifest(staged_root)

    def test_read_storage_manifest_requires_workspace_root_for_lrun(self) -> None:
        """Ensure LRUN staged manifests always declare their staged workspace root."""
        # LRUN manifest reads should require a workspace root because the stored paths are workspace-relative.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_lrun_project_tree(Path(tmpdir) / "staged")
            (staged_root / STAGED_MANIFEST_NAME).write_text(
                json.dumps(
                    {
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            device = STM32NucleoN657X0QDevice(device_options={"project_root": str(staged_root)})

            with self.assertRaises(stm32_cube_clt.WorkflowError):
                device._read_storage_manifest(staged_root)

    def test_external_programming_verifies_both_app_and_weights(self) -> None:
        """App and weight writes must request verification and propagate failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            image = directory / "payload.bin"
            loader = directory / "flash.stldr"
            programmer = directory / "STM32_Programmer_CLI"
            for path in (image, loader, programmer):
                path.write_bytes(b"fixture")
            programmer.chmod(0o755)
            for weights in (False, True):
                with self.subTest(weights=weights):
                    kwargs = dict(cubeprog_bin=directory, apid=1, external_loader=loader, recover_first=False)
                    if weights:
                        function = stm32_cube_clt.program_external_flash_blob
                        kwargs.update(weights_blob_path=image, weights_flash_address="0x71000000")
                    else:
                        function = stm32_cube_clt.program_external_image
                        kwargs.update(image_path=image, flash_address="0x70100000")
                    with patch.object(stm32_cube_clt, "_run_command", return_value="verified") as command:
                        self.assertEqual(function(**kwargs), "verified")
                    argv = command.call_args.args[0]
                    self.assertIn("-v", argv)
                    self.assertEqual(argv[argv.index("-d") + 1], str(image.resolve()))
                    with patch.object(stm32_cube_clt, "_run_command", side_effect=stm32_cube_clt.WorkflowError("verification failed")):
                        with self.assertRaisesRegex(stm32_cube_clt.WorkflowError, "verification failed"):
                            function(**kwargs)

    def test_program_runtime_images_reprograms_shared_board_despite_workspace_history(self) -> None:
        """Revisiting a workspace must restore its app and weights on a shared board."""
        # Another workspace may have replaced both images since the previous upload.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_lrun_project_tree(Path(tmpdir) / "staged")
            signed_app = staged_root / "STM32CubeIDE" / "AppS" / "Debug" / "Template_LRUN_AppS-trusted.bin"
            signed_app.parent.mkdir(parents=True, exist_ok=True)
            signed_app.write_bytes(b"signed-app")
            weights_blob = Path(tmpdir) / "network_data.bin"
            weights_blob.write_bytes(b"weights")
            cubeprog_bin = Path(tmpdir) / "cubeprog" / "bin"
            loader_path = cubeprog_bin / "ExternalLoader" / DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME
            _write_text(loader_path, "loader")
            manifest_path = staged_root / STAGED_MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(
                    {
                        "candidate_root": str(staged_root.parent),
                        "generated_output_dir": str(staged_root / "generated"),
                        "staged_workspace_root": str(staged_root.resolve()),
                        "weight_storage_mode": "external_flash",
                        "appli_signed_image_path": str(signed_app),
                        "signed_app_sha256": stm32_n657_backend._sha256_file(signed_app),
                        "appli_flash_address": stm32_n657_backend.DEFAULT_APPLI_FLASH_ADDRESS,
                        "weights_blob_path": str(weights_blob),
                        "weights_blob_size": weights_blob.stat().st_size,
                        "weights_blob_sha256": stm32_n657_backend._sha256_file(weights_blob),
                        "weights_flash_address": DEFAULT_WEIGHTS_FLASH_ADDRESS,
                        "weights_external_loader": str(loader_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            device = STM32NucleoN657X0QDevice(
                device_options={
                    "project_root": str(staged_root),
                    "cubeprog_bin": str(cubeprog_bin),
                    "weight_storage_mode": "external_flash",
                }
            )
            compile_result = stm32_n657_backend.CompileResult(
                success=True,
                log="ok",
                flash_bytes=None,
                ram_bytes=None,
                overflow_kind=None,
                build_dir=staged_root / "STM32CubeIDE" / "Boot" / "Debug",
                signed_app_bin_path=signed_app,
            )
            paths = device._resolve_paths(staged_root)

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.program_external_image",
                return_value="app programmed",
            ) as app_program_mock, patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.program_external_flash_blob",
                return_value="weights programmed",
            ) as weight_program_mock:
                first = device._program_runtime_images(paths, compile_result=compile_result)

            self.assertTrue(first["appli_programmed"])
            self.assertTrue(first["weights_programmed"])
            self.assertEqual(app_program_mock.call_count, 1)
            self.assertEqual(weight_program_mock.call_count, 1)

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.program_external_image",
                return_value="app reprogrammed",
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.program_external_flash_blob",
                return_value="weights reprogrammed",
            ):
                second = device._program_runtime_images(paths, compile_result=compile_result)

            self.assertTrue(second["appli_programmed"])
            self.assertTrue(second["weights_programmed"])
            signed_app.write_bytes(b"changed-signed-app")
            weights_blob.write_bytes(b"changed-weights")
            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.program_external_image",
                return_value="app verified",
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.program_external_flash_blob",
                return_value="weights verified",
            ):
                device._program_runtime_images(paths, compile_result=compile_result)
            updated = json.loads(manifest_path.read_text())
            self.assertEqual(updated["last_programmed_appli_sha256"], stm32_n657_backend._sha256_file(signed_app))
            self.assertEqual(updated["last_programmed_weights_sha256"], stm32_n657_backend._sha256_file(weights_blob))
            self.assertEqual(updated["weights_blob_size"], weights_blob.stat().st_size)

    def test_program_runtime_images_stale_workspace_cache_still_requires_loader(self) -> None:
        """Cached workspace digests cannot bypass loader validation or programming."""
        # A stale workspace history must not bypass a missing programming prerequisite.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_lrun_project_tree(Path(tmpdir) / "staged")
            signed_app = staged_root / "STM32CubeIDE" / "AppS" / "Debug" / "Template_LRUN_AppS-trusted.bin"
            signed_app.parent.mkdir(parents=True, exist_ok=True)
            signed_app.write_bytes(b"signed-app")
            weights_blob = Path(tmpdir) / "network_data.bin"
            weights_blob.write_bytes(b"weights")
            manifest_path = staged_root / STAGED_MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(
                    {
                        "candidate_root": str(staged_root.parent),
                        "generated_output_dir": str(staged_root / "generated"),
                        "staged_workspace_root": str(staged_root.resolve()),
                        "weight_storage_mode": "external_flash",
                        "appli_signed_image_path": str(signed_app),
                        "signed_app_sha256": stm32_n657_backend._sha256_file(signed_app),
                        "appli_flash_address": stm32_n657_backend.DEFAULT_APPLI_FLASH_ADDRESS,
                        "last_programmed_appli_sha256": stm32_n657_backend._sha256_file(signed_app),
                        "last_programmed_appli_flash_address": stm32_n657_backend.DEFAULT_APPLI_FLASH_ADDRESS,
                        "weights_blob_path": str(weights_blob),
                        "weights_blob_size": weights_blob.stat().st_size,
                        "weights_blob_sha256": stm32_n657_backend._sha256_file(weights_blob),
                        "weights_flash_address": DEFAULT_WEIGHTS_FLASH_ADDRESS,
                        "last_programmed_weights_sha256": stm32_n657_backend._sha256_file(weights_blob),
                        "last_programmed_weights_flash_address": DEFAULT_WEIGHTS_FLASH_ADDRESS,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            device = STM32NucleoN657X0QDevice(
                device_options={
                    "project_root": str(staged_root),
                    "cubeprog_bin": str(Path(tmpdir) / "missing-cubeprog"),
                    "weight_storage_mode": "external_flash",
                }
            )
            compile_result = stm32_n657_backend.CompileResult(
                success=True,
                log="ok",
                flash_bytes=None,
                ram_bytes=None,
                overflow_kind=None,
                build_dir=staged_root / "STM32CubeIDE" / "Boot" / "Debug",
                signed_app_bin_path=signed_app,
            )
            paths = device._resolve_paths(staged_root)

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0._resolve_weights_external_loader",
                side_effect=stm32_cube_clt.WorkflowError("missing loader"),
            ) as resolve_loader:
                with self.assertRaisesRegex(stm32_cube_clt.WorkflowError, "missing loader"):
                    device._program_runtime_images(paths, compile_result=compile_result)
            resolve_loader.assert_called_once()

    def test_program_runtime_images_uses_manifest_loader_for_signed_app_programming(self) -> None:
        """Ensure signed-App programming honors the loader path staged in the manifest."""
        # Signed runtime images should use the manifest-selected loader so programming follows the staged security layout.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_lrun_project_tree(Path(tmpdir) / "staged")
            signed_app = staged_root / "STM32CubeIDE" / "AppS" / "Debug" / "Template_LRUN_AppS-trusted.bin"
            signed_app.parent.mkdir(parents=True, exist_ok=True)
            signed_app.write_bytes(b"signed-app")
            weights_blob = Path(tmpdir) / "network_data.bin"
            weights_blob.write_bytes(b"weights")
            loader_path = Path(tmpdir) / "cubeprog" / "bin" / "ExternalLoader" / DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME
            _write_text(loader_path, "loader")
            manifest_path = staged_root / STAGED_MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(
                    {
                        "candidate_root": str(staged_root.parent),
                        "generated_output_dir": str(staged_root / "generated"),
                        "staged_workspace_root": str(staged_root.resolve()),
                        "weight_storage_mode": "embedded",
                        "appli_signed_image_path": str(signed_app),
                        "appli_flash_address": stm32_n657_backend.DEFAULT_APPLI_FLASH_ADDRESS,
                        "weights_external_loader": str(loader_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            device = STM32NucleoN657X0QDevice(
                device_options={
                    "project_root": str(staged_root),
                    "cubeprog_bin": str(Path(tmpdir) / "missing-cubeprog"),
                }
            )
            compile_result = stm32_n657_backend.CompileResult(
                success=True,
                log="ok",
                flash_bytes=None,
                ram_bytes=None,
                overflow_kind=None,
                build_dir=staged_root / "STM32CubeIDE" / "Boot" / "Debug",
                signed_app_bin_path=signed_app,
            )
            paths = device._resolve_paths(staged_root)

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0._resolve_weights_external_loader",
                side_effect=AssertionError("manifest loader should be used directly"),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.program_external_image",
                return_value="app programmed",
            ) as app_program_mock:
                metrics = device._program_runtime_images(paths, compile_result=compile_result)

            self.assertTrue(metrics["appli_programmed"])
            self.assertEqual(app_program_mock.call_args.kwargs["external_loader"], loader_path.resolve())

    def test_evaluate_lrun_external_flash_programs_app_then_weights_then_boot(self) -> None:
        """Ensure LRUN happy-path upload order is app, weights, then Boot ELF load."""
        # External-flash LRUN runs should program the app, then weights, then boot image so the runtime layout is coherent.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_lrun_project_tree(Path(tmpdir) / "staged")
            signed_app = staged_root / "STM32CubeIDE" / "AppS" / "Debug" / "Template_LRUN_AppS-trusted.bin"
            signed_app.parent.mkdir(parents=True, exist_ok=True)
            signed_app.write_bytes(b"signed-app")
            boot_elf = staged_root / "STM32CubeIDE" / "Boot" / "Debug" / "Template_LRUN_FSBL.elf"
            _write_text(boot_elf, "boot-elf")
            weights_blob = Path(tmpdir) / "network_data.bin"
            weights_blob.write_bytes(b"weights")
            cubeprog_bin = Path(tmpdir) / "cubeprog" / "bin"
            loader_path = cubeprog_bin / "ExternalLoader" / DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME
            _write_text(loader_path, "loader")
            manifest_path = staged_root / STAGED_MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(
                    {
                        "candidate_root": str(staged_root.parent),
                        "generated_output_dir": str(staged_root / "generated"),
                        "staged_workspace_root": str(staged_root.resolve()),
                        "weight_storage_mode": "external_flash",
                        "appli_signed_image_path": str(signed_app),
                        "appli_flash_address": stm32_n657_backend.DEFAULT_APPLI_FLASH_ADDRESS,
                        "weights_blob_path": str(weights_blob),
                        "weights_blob_size": weights_blob.stat().st_size,
                        "weights_flash_address": DEFAULT_WEIGHTS_FLASH_ADDRESS,
                        "weights_external_loader": str(loader_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            device = STM32NucleoN657X0QDevice(
                serial_port="/dev/ttyACM0",
                device_options={
                    "project_root": str(staged_root),
                    "cubeprog_bin": str(cubeprog_bin),
                    "weight_storage_mode": "external_flash",
                },
            )
            compile_result = stm32_n657_backend.CompileResult(
                success=True,
                log="ok",
                flash_bytes=None,
                ram_bytes=None,
                overflow_kind=None,
                build_dir=staged_root / "STM32CubeIDE" / "Boot" / "Debug",
                boot_elf_path=boot_elf,
                signed_app_bin_path=signed_app,
            )
            telemetry = stm32_runtime.STM32RuntimeTelemetry(
                latency_s=0.003,
                serial_log=["STM32_AI_INIT=OK", "DUT READY", "STM32_AI_RUN=OK"],
                power_metrics={"clock_hz": 600000000.0, "sequence": 1.0, "runs": 1, "phase": "back_to_back"},
            )
            # Repaired DUT telemetry: RTC aggregate covers all reported releases.
            telemetry.serial_log.extend([
                "measurement protocol output: rtc_wall_v4_dwt_active_stop_evidence_awake_tail",
                "cadence budget (ms): 200",
            ])
            telemetry.power_metrics.update(runs=10, phase="back_to_back",
                timer_per_window_s=0.03, rtc_clock_source="LSE", rtc_clock_hz_nominal=32768)
            trace: list[str] = []

            with patch.object(device, "compile", return_value=compile_result), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.program_external_image",
                side_effect=lambda **kwargs: trace.append("app") or "app programmed",
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.program_external_flash_blob",
                side_effect=lambda **kwargs: trace.append("weights") or "weights programmed",
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
                _FakeSerialMonitor,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.debug_load_elf",
                side_effect=lambda **kwargs: trace.append("boot") or "upload ok",
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.execute_runtime_session",
                side_effect=lambda *args, **kwargs: trace.append("runtime") or telemetry,
            ):
                metrics = device.evaluate(
                    dirpath=staged_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                    serial_port="/dev/ttyACM0",
                    run_hil=True,
                )

            self.assertEqual(trace, ["app", "weights", "boot", "runtime"])
            self.assertEqual(metrics.error_code, HIL_ERROR_OK)

    def test_compile_lrun_requires_boot_include_path_for_copy_window_updates(self) -> None:
        """Ensure LRUN compile fails if Boot recipes stop including FSBL headers."""
        # Copy-window updates should require the boot include path because that header owns the final boot constants.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_lrun_project_tree(Path(tmpdir) / "staged")
            _write_text(
                staged_root / "STM32CubeIDE" / "Boot" / "Debug" / "Src" / "subdir.mk",
                "arm-none-eabi-gcc -I../Inc\n",
            )
            device = STM32NucleoN657X0QDevice(device_options={"project_root": str(staged_root)})

            result = device.compile(
                sketch_path=staged_root,
                arena_kb=-1,
                window_size=200,
                num_channels=6,
            )

        self.assertFalse(result.success)
        self.assertIn("FSBL include path", result.log)

    def test_classify_signing_failure_returns_binary_signing(self) -> None:
        """Ensure signing failures are classified separately from generic toolchain errors."""
        # Signing failures should map to the stable binary-signing error kind.
        exc = stm32_cube_clt.SigningWorkflowError("STM32 binary signing failed.")
        self.assertEqual(stm32_n657_backend.classify_stm32_backend_error(exc), "binary_signing")

    def test_classify_lrun_specific_failures_returns_stable_kinds(self) -> None:
        """Ensure LRUN-specific failure strings map to stable backend error kinds."""
        # LRUN-specific backend failures should collapse into stable public error kinds.
        self.assertEqual(
            stm32_n657_backend.classify_stm32_backend_error(
                "Could not update EXTMEM_LRUN_SOURCE_SIZE in staged/stm32_extmem_conf.h"
            ),
            "boot_copy_window_update",
        )
        self.assertEqual(
            stm32_n657_backend.classify_stm32_backend_error(
                "STM32 LRUN Boot recipes no longer reference the FSBL include path; EXTMEM_LRUN_SOURCE_SIZE updates would not affect the build."
            ),
            "boot_include_path",
        )
        self.assertEqual(
            stm32_n657_backend.classify_stm32_backend_error("STM32_RTC_INIT=FAIL"),
            "rtc_init",
        )
        self.assertEqual(
            stm32_n657_backend.classify_stm32_backend_error("STM32_XSPI_INIT=FAIL stage=init"),
            "external_weight_mapping",
        )

    def test_evaluate_uses_lrun_default_boot_timeout(self) -> None:
        """Ensure LRUN evaluation uses a safer default boot timeout when unspecified."""
        # Evaluation should use the LRUN default boot timeout when the caller does not override it.
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = _build_lrun_project_tree(Path(tmpdir) / "stm")
            _write_text(
                project_root / "Appli" / "Inc" / "crest_dut_phase_config.h",
                "#define CREST_DUT_MEASURED_RUNS 1\n",
            )
            device = STM32NucleoN657X0QDevice(
                serial_port="/dev/ttyACM0",
                device_options={"project_root": str(project_root)},
            )
            compile_result = type(
                "CompileResultDouble",
                (),
                {
                    "success": True,
                    "log": "ok",
                    "flash_bytes": 2222,
                    "ram_bytes": 1111,
                    "overflow_kind": None,
                    "build_dir": project_root / "STM32CubeIDE" / "Boot" / "Debug",
                    "boot_elf_path": project_root
                    / "STM32CubeIDE"
                    / "Boot"
                    / "Debug"
                    / "Template_LRUN_FSBL.elf",
                    "arena_bytes": 4096,
                    "external_flash_bytes": None,
                    "signed_app_bin_path": project_root
                    / "STM32CubeIDE"
                    / "AppS"
                    / "Debug"
                    / "Template_LRUN_AppS-trusted.bin",
                },
            )()
            telemetry = stm32_runtime.STM32RuntimeTelemetry(
                latency_s=0.003,
                serial_log=["STM32_AI_INIT=OK", "DUT READY", "STM32_AI_RUN=OK"],
                power_metrics={"clock_hz": 600000000.0, "sequence": 1.0, "runs": 1, "phase": "back_to_back"},
            )
            # Repaired DUT telemetry: RTC aggregate covers all reported releases.
            telemetry.serial_log.extend([
                "measurement protocol output: rtc_wall_v4_dwt_active_stop_evidence_awake_tail",
                "cadence budget (ms): 200",
            ])
            telemetry.power_metrics.update(runs=1, phase="back_to_back",
                timer_per_window_s=0.003, rtc_clock_source="LSE", rtc_clock_hz_nominal=32768)
            observed: dict[str, float] = {}

            def _runtime_session(*args, **kwargs):
                """Create a fake runtime session.

                Parameters
                ----------
                *args : tuple[object, ...]
                    Arguments supplied when opening the fake runtime session.
                **kwargs : dict[str, object]
                    Keyword arguments forwarded to the helper.

                Returns
                -------
                object
                    Runtime session test double returned by the factory.
                """
                del args
                observed["boot_timeout_s"] = kwargs["boot_timeout_s"]
                return telemetry

            with patch.object(device, "compile", return_value=compile_result), patch.object(
                device,
                "_storage_power_metrics",
                return_value={"weight_storage_mode": "embedded", "external_flash_bytes": -1.0},
            ), patch.object(
                device,
                "_program_runtime_images",
                return_value={"weight_storage_mode": "embedded", "external_flash_bytes": -1.0},
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
                _FakeSerialMonitor,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.debug_load_elf",
                return_value="upload ok",
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.execute_runtime_session",
                side_effect=_runtime_session,
            ):
                metrics = device.evaluate(
                    dirpath=project_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                    serial_port="/dev/ttyACM0",
                    run_hil=True,
                    dut_ready_timeout_s=None,
                )

        self.assertEqual(metrics.error_code, HIL_ERROR_OK)
        self.assertEqual(observed["boot_timeout_s"], stm32_n657_backend.DEFAULT_LRUN_BOOT_TIMEOUT_S)

    def test_evaluate_lrun_honors_requested_measured_runs_for_header_and_harness(self) -> None:
        """Ensure LRUN evaluate uses the request run count, not the stale staged header value."""
        # LRUN evaluation should propagate measured-run overrides into both the generated header and the harness workflow.
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = _build_lrun_project_tree(Path(tmpdir) / "stm")
            _write_text(
                project_root / "Appli" / "Inc" / "crest_dut_phase_config.h",
                "#define CREST_DUT_MEASURED_RUNS 1\n",
            )
            device = STM32NucleoN657X0QDevice(
                serial_port="/dev/ttyACM0",
                device_options={"project_root": str(project_root)},
            )
            compile_result = type(
                "CompileResultDouble",
                (),
                {
                    "success": True,
                    "log": "ok",
                    "flash_bytes": 2222,
                    "ram_bytes": 1111,
                    "overflow_kind": None,
                    "build_dir": project_root / "STM32CubeIDE" / "Boot" / "Debug",
                    "boot_elf_path": project_root
                    / "STM32CubeIDE"
                    / "Boot"
                    / "Debug"
                    / "Template_LRUN_FSBL.elf",
                    "arena_bytes": 4096,
                    "external_flash_bytes": None,
                    "signed_app_bin_path": project_root
                    / "STM32CubeIDE"
                    / "AppS"
                    / "Debug"
                    / "Template_LRUN_AppS-trusted.bin",
                },
            )()
            telemetry = stm32_runtime.STM32RuntimeTelemetry(
                latency_s=0.003,
                serial_log=["STM32_AI_INIT=OK", "DUT READY", "STM32_AI_RUN=OK"],
                power_metrics={
                    "clock_hz": 600000000.0,
                    "sequence": 1.0,
                    "runs": 7,
                    "phase": "back_to_back",
                },
            )
            # Repaired DUT telemetry: RTC aggregate covers all reported releases.
            telemetry.serial_log.extend([
                "measurement protocol output: rtc_wall_v4_dwt_active_stop_evidence_awake_tail",
                "cadence budget (ms): 200",
            ])
            telemetry.power_metrics.update(runs=7, phase="back_to_back",
                timer_per_window_s=0.021, rtc_clock_source="LSE", rtc_clock_hz_nominal=32768)

            with patch.object(device, "compile", return_value=compile_result), patch.object(
                device,
                "_storage_power_metrics",
                return_value={"weight_storage_mode": "embedded", "external_flash_bytes": -1.0},
            ), patch.object(
                device,
                "_program_runtime_images",
                return_value={"weight_storage_mode": "embedded", "external_flash_bytes": -1.0},
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.arduino_base.ensure_harness_firmware"
            ) as harness_mock, patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
                _FakeSerialMonitor,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.serial.Serial",
                return_value=_FakeHarnessSerial(),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.hil_protocol.prime_harness_session",
                return_value=type(
                    "PrimeResult",
                    (),
                    {"harness_ready": True, "harness_log": ["HARNESS READY"], "error": None},
                )(),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.hil_protocol.wait_for_harness_done",
                return_value=type(
                    "DoneResult",
                    (),
                    {
                        "harness_done": True,
                        "runs_harness": 7,
                        "harness_log": [
                            "HARNESS READY",
                            "runs: 7",
                            "energy output: 1.25",
                            "avg power output: 2.5",
                            "avg current output: 0.5",
                            "bus voltage output: 5.0",
                            "idle power baseline: 0.1",
                            "harness timer output: 0.003",
                            "DONE",
                        ],
                        "error": None,
                    },
                )(),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.debug_load_elf",
                return_value="upload ok",
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.execute_runtime_session",
                return_value=telemetry,
            ):
                metrics = device.evaluate(
                    dirpath=project_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                    serial_port="/dev/ttyACM0",
                    run_hil=True,
                    measured_inference_runs=7,
                    harness_serial_port="/dev/ttyACM1",
                )

            header_text = (
                project_root / "Appli" / "Inc" / "crest_dut_phase_config.h"
            ).read_text(encoding="utf-8")

        self.assertEqual(metrics.error_code, HIL_ERROR_OK)
        self.assertIn("CREST_DUT_MEASURED_RUNS 7", header_text)
        self.assertEqual(
            harness_mock.call_args.kwargs["build_defines"]["CREST_INFERENCE_RUNS"],
            7,
        )

    def test_evaluate_lrun_preserves_staged_measured_runs_when_override_is_omitted(self) -> None:
        """Ensure staged LRUN measured runs are preserved when evaluate omits an override."""
        # Omitted measured-run overrides should preserve the staged value instead of rewriting it with a new default.
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = _build_lrun_project_tree(Path(tmpdir) / "stm")
            _write_text(
                project_root / "Appli" / "Inc" / "crest_dut_phase_config.h",
                "\n".join(
                    [
                        "#ifndef CREST_DUT_PHASE_CONFIG_H",
                        "#define CREST_DUT_PHASE_CONFIG_H",
                        "#define CREST_DUT_PHASE_BACK_TO_BACK 0",
                        "#define CREST_DUT_PHASE_CADENCED 1",
                        "#define CREST_DUT_SELECTED_PHASE CREST_DUT_PHASE_BACK_TO_BACK",
                        "#define CREST_DUT_MEASURED_RUNS 4",
                        "#endif",
                        "",
                    ]
                ),
            )
            device = STM32NucleoN657X0QDevice(
                serial_port="/dev/ttyACM0",
                device_options={"project_root": str(project_root)},
            )
            compile_result = type(
                "CompileResultDouble",
                (),
                {
                    "success": True,
                    "log": "ok",
                    "flash_bytes": 2222,
                    "ram_bytes": 1111,
                    "overflow_kind": None,
                    "build_dir": project_root / "STM32CubeIDE" / "Boot" / "Debug",
                    "boot_elf_path": project_root
                    / "STM32CubeIDE"
                    / "Boot"
                    / "Debug"
                    / "Template_LRUN_FSBL.elf",
                    "arena_bytes": 4096,
                    "external_flash_bytes": None,
                    "signed_app_bin_path": project_root
                    / "STM32CubeIDE"
                    / "AppS"
                    / "Debug"
                    / "Template_LRUN_AppS-trusted.bin",
                },
            )()
            telemetry = stm32_runtime.STM32RuntimeTelemetry(
                latency_s=0.003,
                serial_log=["STM32_AI_INIT=OK", "DUT READY", "STM32_AI_RUN=OK"],
                power_metrics={
                    "clock_hz": 600000000.0,
                    "sequence": 1.0,
                    "runs": 4,
                    "phase": "back_to_back",
                },
            )
            # Repaired DUT telemetry: RTC aggregate covers all reported releases.
            telemetry.serial_log.extend([
                "measurement protocol output: rtc_wall_v4_dwt_active_stop_evidence_awake_tail",
                "cadence budget (ms): 200",
            ])
            telemetry.power_metrics.update(runs=4, phase="back_to_back",
                timer_per_window_s=0.012, rtc_clock_source="LSE", rtc_clock_hz_nominal=32768)

            with patch.object(device, "compile", return_value=compile_result), patch.object(
                device,
                "_storage_power_metrics",
                return_value={"weight_storage_mode": "embedded", "external_flash_bytes": -1.0},
            ), patch.object(
                device,
                "_program_runtime_images",
                return_value={"weight_storage_mode": "embedded", "external_flash_bytes": -1.0},
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.arduino_base.ensure_harness_firmware"
            ) as harness_mock, patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
                _FakeSerialMonitor,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.serial.Serial",
                return_value=_FakeHarnessSerial(),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.hil_protocol.prime_harness_session",
                return_value=type(
                    "PrimeResult",
                    (),
                    {"harness_ready": True, "harness_log": ["HARNESS READY"], "error": None},
                )(),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.hil_protocol.wait_for_harness_done",
                return_value=type(
                    "DoneResult",
                    (),
                    {
                        "harness_done": True,
                        "runs_harness": 4,
                        "harness_log": [
                            "HARNESS READY",
                            "runs: 4",
                            "energy output: 1.25",
                            "avg power output: 2.5",
                            "avg current output: 0.5",
                            "bus voltage output: 5.0",
                            "idle power baseline: 0.1",
                            "harness timer output: 0.003",
                            "DONE",
                        ],
                        "error": None,
                    },
                )(),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.debug_load_elf",
                return_value="upload ok",
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.execute_runtime_session",
                return_value=telemetry,
            ):
                metrics = device.evaluate(
                    dirpath=project_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                    serial_port="/dev/ttyACM0",
                    run_hil=True,
                    harness_serial_port="/dev/ttyACM1",
                )

            header_text = (
                project_root / "Appli" / "Inc" / "crest_dut_phase_config.h"
            ).read_text(encoding="utf-8")

        self.assertEqual(metrics.error_code, HIL_ERROR_OK)
        self.assertIn("CREST_DUT_MEASURED_RUNS 4", header_text)
        self.assertEqual(
            harness_mock.call_args.kwargs["build_defines"]["CREST_INFERENCE_RUNS"],
            4,
        )

    def test_evaluate_lrun_rewrites_stale_phase_config_fields_when_override_is_omitted(self) -> None:
        """Ensure staged LRUN evaluation refreshes generated phase-config macros."""
        # Evaluation should rewrite stale phase-config fields from the staged workspace so omitted overrides do not resurrect old run settings.
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = _build_lrun_project_tree(Path(tmpdir) / "stm")
            _write_text(
                project_root / "Appli" / "Inc" / "crest_dut_phase_config.h",
                "\n".join(
                    [
                        "#ifndef CREST_DUT_PHASE_CONFIG_H",
                        "#define CREST_DUT_PHASE_CONFIG_H",
                        "",
                        "#define CREST_DUT_PHASE_BACK_TO_BACK 0",
                        "#define CREST_DUT_PHASE_CADENCED 1",
                        "",
                        "#define CREST_DUT_SELECTED_PHASE CREST_DUT_PHASE_BACK_TO_BACK",
                        "#define CREST_DUT_LATENCY_BUDGET_MS 999",
                        "#define CREST_DUT_MEASURED_RUNS 4",
                        "#define CREST_DUT_CPU_CLOCK_MHZ 200",
                        "#define CREST_DUT_WAKE_MARGIN_US 111",
                        "#define CREST_DUT_MIN_SLEEP_US 222",
                        "",
                        "#endif /* CREST_DUT_PHASE_CONFIG_H */",
                        "",
                    ]
                ),
            )
            device = STM32NucleoN657X0QDevice(
                serial_port="/dev/ttyACM0",
                device_options={
                    "project_root": str(project_root),
                    "cpu_clock_mhz": 400,
                    "latency_budget_ms": 321.0,
                    "wake_margin_us": 4321,
                    "min_sleep_us": 5432,
                },
            )
            compile_result = type(
                "CompileResultDouble",
                (),
                {
                    "success": True,
                    "log": "ok",
                    "flash_bytes": 2222,
                    "ram_bytes": 1111,
                    "overflow_kind": None,
                    "build_dir": project_root / "STM32CubeIDE" / "Boot" / "Debug",
                    "boot_elf_path": project_root
                    / "STM32CubeIDE"
                    / "Boot"
                    / "Debug"
                    / "Template_LRUN_FSBL.elf",
                    "arena_bytes": 4096,
                    "external_flash_bytes": None,
                    "signed_app_bin_path": project_root
                    / "STM32CubeIDE"
                    / "AppS"
                    / "Debug"
                    / "Template_LRUN_AppS-trusted.bin",
                },
            )()
            telemetry = stm32_runtime.STM32RuntimeTelemetry(
                latency_s=0.003,
                serial_log=["STM32_AI_INIT=OK", "DUT READY", "STM32_AI_RUN=OK"],
                power_metrics={
                    "clock_hz": 400000000.0,
                    "sequence": 1.0,
                    "runs": 4,
                    "phase": "back_to_back",
                },
            )
            # Repaired DUT telemetry: RTC aggregate covers all reported releases.
            telemetry.serial_log.extend([
                "measurement protocol output: rtc_wall_v4_dwt_active_stop_evidence_awake_tail",
                "cadence budget (ms): 321",
            ])
            telemetry.power_metrics.update(runs=4, phase="back_to_back",
                timer_per_window_s=0.012, rtc_clock_source="LSE", rtc_clock_hz_nominal=32768)

            with patch.object(device, "compile", return_value=compile_result), patch.object(
                device,
                "_storage_power_metrics",
                return_value={"weight_storage_mode": "embedded", "external_flash_bytes": -1.0},
            ), patch.object(
                device,
                "_program_runtime_images",
                return_value={"weight_storage_mode": "embedded", "external_flash_bytes": -1.0},
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.arduino_base.ensure_harness_firmware"
            ) as harness_mock, patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
                _FakeSerialMonitor,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.serial.Serial",
                return_value=_FakeHarnessSerial(),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.hil_protocol.prime_harness_session",
                return_value=type(
                    "PrimeResult",
                    (),
                    {"harness_ready": True, "harness_log": ["HARNESS READY"], "error": None},
                )(),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.hil_protocol.wait_for_harness_done",
                return_value=type(
                    "DoneResult",
                    (),
                    {
                        "harness_done": True,
                        "runs_harness": 4,
                        "harness_log": [
                            "HARNESS READY",
                            "runs: 4",
                            "energy output: 1.25",
                            "avg power output: 2.5",
                            "avg current output: 0.5",
                            "bus voltage output: 5.0",
                            "idle power baseline: 0.1",
                            "harness timer output: 0.003",
                            "DONE",
                        ],
                        "error": None,
                    },
                )(),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.debug_load_elf",
                return_value="upload ok",
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.execute_runtime_session",
                return_value=telemetry,
            ):
                metrics = device.evaluate(
                    dirpath=project_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                    serial_port="/dev/ttyACM0",
                    run_hil=True,
                    harness_serial_port="/dev/ttyACM1",
                )

            header_text = (
                project_root / "Appli" / "Inc" / "crest_dut_phase_config.h"
            ).read_text(encoding="utf-8")

        self.assertEqual(metrics.error_code, HIL_ERROR_OK)
        self.assertIn("#define CREST_DUT_MEASURED_RUNS 4", header_text)
        self.assertIn("#define CREST_DUT_LATENCY_BUDGET_MS 321", header_text)
        self.assertIn("#define CREST_DUT_CPU_CLOCK_MHZ 400", header_text)
        self.assertIn("#define CREST_DUT_WAKE_MARGIN_US 4321", header_text)
        self.assertIn("#define CREST_DUT_MIN_SLEEP_US 5432", header_text)
        self.assertEqual(
            harness_mock.call_args.kwargs["build_defines"]["CREST_INFERENCE_RUNS"],
            4,
        )

    def test_evaluate_without_staged_paths_defaults_omitted_measured_runs_to_ten(self) -> None:
        """Ensure omitted measured runs still fall back to 10 for non-staged evaluation paths."""
        # Without staged paths, LRUN evaluation should fall back to the default measured-run count of ten.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            template_root = _build_lrun_project_tree(tmp_path / "template")
            build_dir = tmp_path / "build" / "Debug"
            build_dir.mkdir(parents=True)
            boot_elf_path = build_dir / "Template_LRUN_FSBL.elf"
            boot_elf_path.write_text("elf", encoding="utf-8")
            missing_root = tmp_path / "missing-staged-root"
            device = STM32NucleoN657X0QDevice(
                serial_port="/dev/ttyACM0",
                device_options={"project_root": str(template_root)},
            )
            compile_result = type(
                "CompileResultDouble",
                (),
                {
                    "success": True,
                    "log": "ok",
                    "flash_bytes": 2222,
                    "ram_bytes": 1111,
                    "overflow_kind": None,
                    "build_dir": build_dir,
                    "boot_elf_path": boot_elf_path,
                    "arena_bytes": 4096,
                    "external_flash_bytes": None,
                    "signed_app_bin_path": None,
                },
            )()
            telemetry = stm32_runtime.STM32RuntimeTelemetry(
                latency_s=0.003,
                serial_log=["STM32_AI_INIT=OK", "DUT READY", "STM32_AI_RUN=OK"],
                power_metrics={
                    "clock_hz": 600000000.0,
                    "sequence": 1.0,
                    "runs": 10,
                    "phase": "back_to_back",
                },
            )
            # Repaired DUT telemetry: RTC aggregate covers all reported releases.
            telemetry.serial_log.extend([
                "measurement protocol output: rtc_wall_v4_dwt_active_stop_evidence_awake_tail",
                "cadence budget (ms): 200",
            ])
            telemetry.power_metrics.update(runs=10, phase="back_to_back",
                timer_per_window_s=0.03, rtc_clock_source="LSE", rtc_clock_hz_nominal=32768)

            with patch.object(device, "compile", return_value=compile_result), patch.object(
                device,
                "_storage_power_metrics",
                return_value={"weight_storage_mode": "embedded", "external_flash_bytes": -1.0},
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.arduino_base.ensure_harness_firmware"
            ) as harness_mock, patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
                _FakeSerialMonitor,
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.serial.Serial",
                return_value=_FakeHarnessSerial(),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.hil_protocol.prime_harness_session",
                return_value=type(
                    "PrimeResult",
                    (),
                    {"harness_ready": True, "harness_log": ["HARNESS READY"], "error": None},
                )(),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.hil_protocol.wait_for_harness_done",
                return_value=type(
                    "DoneResult",
                    (),
                    {
                        "harness_done": True,
                        "runs_harness": 10,
                        "harness_log": [
                            "HARNESS READY",
                            "runs: 10",
                            "energy output: 1.25",
                            "avg power output: 2.5",
                            "avg current output: 0.5",
                            "bus voltage output: 5.0",
                            "idle power baseline: 0.1",
                            "harness timer output: 0.003",
                            "DONE",
                        ],
                        "error": None,
                    },
                )(),
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.debug_load_elf",
                return_value="upload ok",
            ), patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.execute_runtime_session",
                return_value=telemetry,
            ):
                metrics = device.evaluate(
                    dirpath=missing_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                    serial_port="/dev/ttyACM0",
                    run_hil=True,
                    harness_serial_port="/dev/ttyACM1",
                )

        self.assertEqual(metrics.error_code, HIL_ERROR_OK)
        self.assertEqual(
            harness_mock.call_args.kwargs["build_defines"]["CREST_INFERENCE_RUNS"],
            10,
        )

    def test_canonical_lrun_linkers_use_safe_heap_and_stack_floors(self) -> None:
        """Ensure the checked-in canonical LRUN linkers reserve safe heap/stack floors."""
        # Canonical LRUN linker templates should keep the checked-in minimum heap and stack floors.
        app_reservations = stm32_n657_backend._parse_linker_reservations(
            ROOT_DIR / "sketches" / "stm32" / "crest_stm32_lrun" / "STM32CubeIDE" / "AppS" / "STM32N657XX_LRUN.ld"
        )
        boot_reservations = stm32_n657_backend._parse_linker_reservations(
            ROOT_DIR / "sketches" / "stm32" / "crest_stm32_lrun" / "STM32CubeIDE" / "Boot" / "STM32N657XX_AXISRAM2_fsbl.ld"
        )
        self.assertEqual(app_reservations["heap_bytes"], 0x2000)
        self.assertEqual(app_reservations["stack_bytes"], 0x4000)
        self.assertEqual(boot_reservations["heap_bytes"], 0x2000)
        self.assertEqual(boot_reservations["stack_bytes"], 0x4000)

    def test_real_lrun_template_parsers_match_checked_in_files(self) -> None:
        """Ensure LRUN path resolution and parser expectations match the checked-in workspace."""
        # Template parsers should still match the checked-in LRUN files shipped with the repository.
        canonical_root = ROOT_DIR / "sketches" / "stm32" / "crest_stm32_lrun"
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = Path(tmpdir) / canonical_root.name
            shutil.copytree(canonical_root, staged_root)
            _write_text(
                staged_root / "FSBL" / "Inc" / "stm32_extmem_conf.h",
                "#define EXTMEM_LRUN_SOURCE_SIZE 0x00020000\n",
            )
            _write_text(staged_root / "Appli" / "Inc" / "main.h", "#pragma once\n")
            _write_text(staged_root / "Appli" / "Src" / "system_stm32n6xx_s.c", "void SystemInit(void) {}\n")
            _write_text(
                staged_root / "Appli" / "Inc" / "network_data_params.h",
                "#define AI_NETWORK_DATA_ACTIVATIONS_SIZE (47688)\n",
            )
            paths = stm32_n657_backend._resolve_workspace_paths(project_root=staged_root)
            validated = stm32_n657_backend._validate_project_structure(paths)

            self.assertEqual(validated.layout, "lrun_dev_boot")
            self.assertEqual(
                stm32_n657_backend._find_linker_script(paths).name,
                "STM32N657XX_LRUN.ld",
            )
            self.assertEqual(
                stm32_n657_backend._find_boot_linker_script(paths).name,
                "STM32N657XX_AXISRAM2_fsbl.ld",
            )
            self.assertEqual(
                stm32_n657_backend._parse_arena_bytes(staged_root / "Appli" / "Inc" / "network_data_params.h"),
                47688,
            )

    def test_canonical_lrun_app_recipe_includes_secure_nsc(self) -> None:
        """Ensure the checked-in AppS recipe compiles secure_nsc.c with its header path."""
        # The canonical LRUN app recipe should keep the secure NSC region enabled.
        recipe = (
            ROOT_DIR
            / "sketches"
            / "stm32"
            / "crest_stm32_lrun"
            / "STM32CubeIDE"
            / "AppS"
            / "Debug"
            / "Src"
            / "subdir.mk"
        ).read_text(encoding="utf-8")
        self.assertIn("../../../Appli/Src/secure_nsc.c", recipe)
        self.assertIn("-I../../../Secure_nsclib", recipe)

    def test_canonical_lrun_xspi2_irq_uses_second_hal_handle(self) -> None:
        """Ensure the checked-in LRUN IRQ file dispatches XSPI2 to hxspi_nor[1]."""
        # The canonical LRUN XSPI2 IRQ hook should keep using the second HAL handle.
        irq_text = (
            ROOT_DIR
            / "sketches"
            / "stm32"
            / "crest_stm32_lrun"
            / "Appli"
            / "Src"
            / "stm32n6xx_it.c"
        ).read_text(encoding="utf-8")
        self.assertIn("void XSPI2_IRQHandler(void)", irq_text)
        self.assertIn("HAL_XSPI_IRQHandler(&hxspi_nor[1]);", irq_text)

    def test_canonical_lrun_runner_fails_cadenced_init_on_rtc_selftest(self) -> None:
        """Ensure the checked-in LRUN runner treats cadenced RTC self-test failures as init failures."""
        # The canonical LRUN runner should fail cadenced init when the RTC self-test fails.
        runner_text = (
            ROOT_DIR
            / "sketches"
            / "stm32"
            / "crest_stm32_lrun"
            / "Appli"
            / "Src"
            / "crest_dut_runner.c"
        ).read_text(encoding="utf-8")
        self.assertIn('emit_line("STM32_AI_INIT=FAIL reason=rtc_wakeup_selftest");', runner_text)

    def test_program_weight_blob_autodiscovers_loader_when_manifest_omits_it(self) -> None:
        """Ensure runtime external-flash programming can recover without a stored loader path.

        Returns
        -------
        None
        """
        # Weight programming should autodiscover a loader when the manifest leaves that field unset.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = _build_lrun_project_tree(tmp_path / "crest_stm32_lrun")
            weights_blob = tmp_path / "network_data.bin"
            cubeprog_bin = tmp_path / "cubeprog" / "bin"
            loader_path = cubeprog_bin / "ExternalLoader" / DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME
            _write_text(weights_blob, "blob")
            _write_text(loader_path, "loader")
            (project_root / STAGED_MANIFEST_NAME).write_text(
                json.dumps(
                    {
                        "staged_workspace_root": str(project_root.resolve()),
                        "weight_storage_mode": "external_flash",
                        "weights_blob_path": str(weights_blob),
                        "weights_blob_size": weights_blob.stat().st_size,
                        "weights_flash_address": DEFAULT_WEIGHTS_FLASH_ADDRESS,
                        "weights_external_loader": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            device = STM32NucleoN657X0QDevice(
                device_options={
                    "template_root": str(project_root),
                    "weight_storage_mode": "external_flash",
                    "cubeprog_bin": str(cubeprog_bin),
                }
            )

            with patch(
                "crest.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.program_external_flash_blob",
                return_value="program ok",
            ) as program_mock:
                metrics = device._program_weight_blob_if_needed(project_root)

        self.assertEqual(metrics["weight_storage_mode"], "external_flash")
        self.assertEqual(program_mock.call_args.kwargs["external_loader"], loader_path.resolve())

    def test_parse_size_output_wraps_host_os_errors_in_workflow_error(self) -> None:
        """Ensure host OS errors from ``arm-none-eabi-size`` become ``WorkflowError``.

        Returns
        -------
        None
        """
        # Host OS failures from the size parser should still surface as workflow errors.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            elf_path = tmp_path / "crest_phase2.elf"
            size_binary = tmp_path / "toolchain" / "bin" / "arm-none-eabi-size"
            elf_path.write_bytes(b"elf")
            with patch(
                "crest.microcontrollers.stm32_cube_clt.resolve_required_tool_path",
                return_value=size_binary,
            ), patch(
                "crest.microcontrollers.stm32_cube_clt.subprocess.run",
                side_effect=OSError("exec format error"),
            ):
                with self.assertRaises(stm32_cube_clt.WorkflowError):
                    stm32_cube_clt.parse_size_output(elf_path)


class RuntimeIOEvidenceTests(unittest.TestCase):
    def test_partial_dut_log_survives_transport_failure_and_retry(self):
        """Partial DUT output must survive transport recovery in durable evidence."""
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = STM32NucleoN657X0QDevice(
                device_options={"measurement_records_dir": root / "records"})
            compiled = SimpleNamespace(success=True, log="ok", flash_bytes=2,
                ram_bytes=1, overflow_kind=None, build_dir=root / "Debug", arena_bytes=3)
            prefix = "crest.microcontrollers.stm32_nucleo_n657x0."
            with patch.object(device, "compile", return_value=compiled) as compile_mock, \
                 patch(prefix + "stm32_cube_clt.resolve_elf_path", return_value=root / "app.elf"), \
                 patch(prefix + "stm32_runtime.SerialMonitor") as monitor_class, \
                 patch(prefix + "stm32_cube_clt.debug_load_elf", return_value="ok"), \
                 patch(prefix + "stm32_runtime.execute_runtime_session",
                       side_effect=RuntimeError("serial reader failed")):
                monitor_class.return_value.__enter__.return_value.snapshot_lines.return_value = [
                    "STM32_AI_INIT=OK", "DUT READY"]
                result = device.evaluate(dirpath=root / "unstaged", arena_kb=-1,
                    window_size=200, num_channels=10, serial_port="fake")
            self.assertEqual(result.error_code, HIL_ERROR_INTEGRITY)
            # An integrity retry reuses its build without retaining hidden device state.
            self.assertEqual(compile_mock.call_count, 1)
            self.assertFalse(hasattr(device, "_phase_compile_result"))
            paths = result.power_metrics["measurement_record_paths"]
            self.assertEqual(len(paths), 2)
            for path in paths:
                record = json.loads(Path(path).read_text())
                self.assertEqual(record["dut_log"], ["STM32_AI_INIT=OK", "DUT READY"])
                self.assertEqual(record["result"]["power_metrics"]["backend_error_kind"], "runtime_io")


class MeasurementRetryPolicyTests(unittest.TestCase):
    def test_retry_compile_state_is_local_to_each_phase(self):
        # A later phase that fails before compiling must not reuse an earlier build.
        device = STM32NucleoN657X0QDevice()
        compiled = object()
        failure = DeviceMetrics(ram_bytes=1, flash_bytes=2, latency_s=-1.0,
                                arena_bytes=3, error_code=HIL_ERROR_INTEGRITY)
        states = []

        def attempt(*, _retry_state, **kwargs):
            if not states:
                _retry_state["compile_result"] = compiled
            states.append(_retry_state)
            if len(states) == 2:
                self.assertIs(kwargs["_reuse_compile_result"], compiled)
            return failure

        with patch.object(device, "_evaluate_phase_attempt", side_effect=attempt) as attempts:
            device._evaluate_single_phase()
            device._evaluate_single_phase()
        self.assertEqual(attempts.call_count, 3)
        self.assertIs(states[0], states[1])
        self.assertIsNot(states[1], states[2])
        self.assertEqual(states[2], {})
        self.assertFalse(hasattr(device, "_phase_compile_result"))

    def test_runtime_error_kinds_preserve_retry_and_timeout_policy(self):
        # Protocol/transport failures permit retry while an actual run timeout penalizes latency.
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = STM32NucleoN657X0QDevice()
            compiled = SimpleNamespace(success=True, log="ok", flash_bytes=2,
                ram_bytes=1, overflow_kind=None, build_dir=root / "Debug", arena_bytes=3)
            prefix = "crest.microcontrollers.stm32_nucleo_n657x0."
            with patch.object(device, "compile", return_value=compiled), \
                 patch(prefix + "stm32_cube_clt.resolve_elf_path", return_value=root / "app.elf"), \
                 patch(prefix + "stm32_runtime.SerialMonitor"), \
                 patch(prefix + "stm32_cube_clt.debug_load_elf", return_value="ok"):
                for kind in sorted(MEASUREMENT_INTEGRITY_KINDS | {"runtime_timeout"}):
                    with self.subTest(kind=kind), patch(
                        prefix + "stm32_runtime.execute_runtime_session",
                        side_effect=stm32_runtime.STM32RuntimeProtocolError(
                            kind=kind, detail="failed", serial_log=[])):
                        result = device._evaluate_single_phase_unchecked(
                            dirpath=root / "unstaged", arena_kb=-1, window_size=200,
                            num_channels=10, serial_port="fake")
                        expected = HIL_ERROR_LATENCY if kind == "runtime_timeout" else HIL_ERROR_INTEGRITY
                        self.assertEqual(result.error_code, expected)
                        self.assertEqual(result.power_metrics["backend_error_kind"], kind)
