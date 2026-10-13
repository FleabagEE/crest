# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the HIL server bootstrap and request pipeline."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from addict import Dict

# Ensure `src` is importable whenever this test module is executed via
# `python -m unittest`.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import hil_server as hil_server_module  # noqa: E402
from hil_server import HILServer  # noqa: E402
from crest.devices import CandidatePrepareRequest, arduino_staged_sketch_path  # noqa: E402
from crest.errors import HIL_ERROR_OK  # noqa: E402
from crest.hil_runtime import CollectMetricsRequest  # noqa: E402
from crest.pipeline_types import DataSplit, DatasetBundle, TargetSpec, TaskMetricContract  # noqa: E402


class FakeDataset:
    """Dataset double used to bootstrap ``HILServer`` tests."""

    bundle: DatasetBundle | None = None
    calibration_split_override = None
    validate_calls: list[object] = []
    load_calls: list[object] = []
    calibration_calls: list[tuple[DatasetBundle, object]] = []

    @classmethod
    def reset(cls) -> None:
        """Reset the requested operation."""
        cls.bundle = None
        cls.calibration_split_override = None
        cls.validate_calls = []
        cls.load_calls = []
        cls.calibration_calls = []

    def validate_config(self, dataset_config: object) -> None:
        """Validate config.

        Parameters
        ----------
        dataset_config : object
            Dataset configuration used to build the task bundle.
        """
        type(self).validate_calls.append(dataset_config)

    def load(self, dataset_config: object) -> DatasetBundle:
        """Load the requested operation.

        Parameters
        ----------
        dataset_config : object
            Dataset configuration used to build the task bundle.

        Returns
        -------
        DatasetBundle
            Dataset bundle returned by the fake loader.
        """
        type(self).load_calls.append(dataset_config)
        assert type(self).bundle is not None
        return type(self).bundle

    def make_calibration_data(
        self,
        bundle: DatasetBundle,
        dataset_config: object,
    ) -> DataSplit | None:
        """Return calibration data using the same fallback order as production.

        Parameters
        ----------
        bundle : DatasetBundle
            Loaded dataset bundle under test.
        dataset_config : object
            Dataset configuration object forwarded by the server.

        Returns
        -------
        DataSplit | None
            Explicit override when configured, otherwise the bundle's
            calibration split.
        """
        type(self).calibration_calls.append((bundle, dataset_config))
        if type(self).calibration_split_override is not None:
            return type(self).calibration_split_override
        return bundle.calibration


class FakeTask:
    """Task double used to bootstrap ``HILServer`` tests."""

    target_spec = TargetSpec(
        task_type="regression",
        output_names=["velx", "vely"],
        output_shapes=[(1,), (1,)],
        metadata={},
    )
    metric_contract_value = TaskMetricContract(
        available_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
        training_only_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
        nonnegative_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
        primary_metric_names={"rmse_total"},
    )
    init_kwargs: list[dict[str, object]] = []
    validate_calls: list[object] = []
    build_target_spec_calls: list[tuple[DatasetBundle, object]] = []
    validate_model_outputs_calls: list[tuple[object, TargetSpec]] = []

    @classmethod
    def reset(cls) -> None:
        """Reset the requested operation."""
        cls.init_kwargs = []
        cls.validate_calls = []
        cls.build_target_spec_calls = []
        cls.validate_model_outputs_calls = []

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        early_stopping_patience: int = 40,
    ) -> None:
        """Record the constructor arguments that the server passes through.

        Parameters
        ----------
        checkpoint_path : Path | None
            Checkpoint path passed through the training configuration.
        early_stopping_patience : int
            Early-stopping patience value passed through the training config.
        """
        type(self).init_kwargs.append(
            {
                "checkpoint_path": checkpoint_path,
                "early_stopping_patience": early_stopping_patience,
            }
        )

    def validate_config(self, task_config: object) -> None:
        """Validate config.

        Parameters
        ----------
        task_config : object
            Task configuration passed into the scaffolded method.
        """
        type(self).validate_calls.append(task_config)

    def build_target_spec(
        self,
        bundle: DatasetBundle,
        task_config: object,
    ) -> TargetSpec:
        """Return the shared target spec while recording the bootstrap inputs.

        Parameters
        ----------
        bundle : DatasetBundle
            Task bundle passed into the scaffolded task method.
        task_config : object
            Task configuration passed into the scaffolded method.

        Returns
        -------
        TargetSpec
            Constructed target spec.
        """
        type(self).build_target_spec_calls.append((bundle, task_config))
        return type(self).target_spec

    def metric_contract(
        self,
        target_spec: TargetSpec,
        task_config: object,
    ) -> TaskMetricContract:
        """Return the shared metric contract while recording bootstrap inputs.

        Parameters
        ----------
        target_spec : TargetSpec
            Target specification passed into the task method.
        task_config : object
            Task configuration passed into the scaffolded method.

        Returns
        -------
        TaskMetricContract
            Metric contract returned by the fake task.
        """
        del target_spec, task_config
        return type(self).metric_contract_value

    def validate_model_outputs(self, model: object, target_spec: TargetSpec) -> None:
        """Validate model outputs.

        Parameters
        ----------
        model : object
            Model instance supplied to the task method.
        target_spec : TargetSpec
            Target specification passed into the task method.
        """
        type(self).validate_model_outputs_calls.append((model, target_spec))


class FakeModelFamily:
    """Model-family double used to bootstrap ``HILServer`` tests."""

    model: object | None = None
    validate_calls: list[object] = []
    materialize_calls: list[dict[str, object]] = []

    @classmethod
    def reset(cls) -> None:
        """Reset the requested operation."""
        cls.model = None
        cls.validate_calls = []
        cls.materialize_calls = []

    def validate_config(self, model_config: object) -> None:
        """Validate config.

        Parameters
        ----------
        model_config : object
            Model configuration used by the request builder.
        """
        type(self).validate_calls.append(model_config)

    def materialize_export_model(
        self,
        hparams: dict[str, object],
        ctx: object,
        config: object,
        *,
        model_variant: str,
        checkpoint_path: Path | str | None = None,
    ) -> object:
        """Record export requests and return the configured fake model object.

        Parameters
        ----------
        hparams : dict[str, object]
            Hyperparameters supplied to the model-family helper.
        ctx : object
            Model context passed into the helper.
        config : object
            Configuration object used by the helper.
        model_variant : str
            Model variant name included in the request payload.
        checkpoint_path : Path | str | None
            Checkpoint path passed through the training configuration.

        Returns
        -------
        object
            Export model returned by the fake model family.
        """
        type(self).materialize_calls.append(
            {
                "hparams": dict(hparams),
                "ctx": ctx,
                "config": config,
                "model_variant": model_variant,
                "checkpoint_path": checkpoint_path,
            }
        )
        assert type(self).model is not None
        return type(self).model


class HILServerTestCase(unittest.TestCase):
    """Common test scaffolding for all HILServer unit tests.

    The real HIL server pulls configuration and datasets on construction, which
    would slow tests to a crawl. These helpers replace those heavy operations
    with small, deterministic doubles that still behave like the production
    objects.
    """

    def setUp(self) -> None:
        # Create a lightweight mock config object to avoid loading YAML files.
        """Prepare test fixtures."""
        evidence_dir = tempfile.TemporaryDirectory()
        self.addCleanup(evidence_dir.cleanup)
        self.config = Dict(
            network=SimpleNamespace(host="127.0.0.1", port=6000, recv_timeout_sec=1, send_timeout_sec=1),
            training=Dict(
                quantization=Dict(mode="int8_ptq", search=False, choices=["int8_ptq"]),
                latency_proxy_max_flops=5_000_000,
                energy_aware=False,  # default sketch variant for unit tests
            ),
            device=SimpleNamespace(
                hil=True,
                name="TEST_DEVICE",
                serial_port="ttyACM0",
                latency_budget_ms=None,
            ),
            outputs=SimpleNamespace(
                models_dir=Path(evidence_dir.name),
                tflite_model_path=Path("model.tflite"),
                candidate_dir=Path("odom_tcn"),
                checkpoint_path=Path("checkpoint.keras"),
            ),
            dataset=SimpleNamespace(
                name="oxiod",
                params=Dict(
                    directory="data",
                    sampling_rate_hz=100,
                    window_size=32,
                    stride=4,
                    sub_folders=["handheld/"],
                    calibration_windows=2048,
                ),
            ),
            task=SimpleNamespace(name="odometry_regression", params=Dict()),
            model=SimpleNamespace(family="odom_tcn", params=Dict(export_variant="approx_trained"), search=Dict()),
            nas=Dict(
                score=Dict(
                    type="scoring-function",
                    metrics=Dict(),
                    params=Dict(
                        terms=[
                            Dict(type="weighted", metric="rmse_total", weight=-1.0),
                        ]
                    ),
                ),
                prune=Dict(rules=[]),
            ),
        )

        self.train_split = DataSplit(
            inputs=np.zeros((1, 32, 6), dtype=np.float32),
            targets={"velx": np.zeros((1, 1), dtype=np.float32), "vely": np.zeros((1, 1), dtype=np.float32)},
        )
        self.calibration_split = DataSplit(
            inputs=np.ones((1, 32, 6), dtype=np.float32),
            targets={"velx": np.zeros((1, 1), dtype=np.float32), "vely": np.zeros((1, 1), dtype=np.float32)},
        )
        self.dataset_bundle = DatasetBundle(
            train=self.train_split,
            val=self.train_split,
            test=self.train_split,
            calibration=self.calibration_split,
            input_shape=(32, 6),
            input_dtype="float32",
            metadata={"window_size": 32, "input_dim": 6},
        )
        self.fake_model = MagicMock()
        self.fake_model.input_shape = (None, 32, 6)

        FakeDataset.reset()
        FakeDataset.bundle = self.dataset_bundle
        FakeTask.reset()
        FakeModelFamily.reset()
        FakeModelFamily.model = self.fake_model

        # Patch modular bootstrap seams so tests avoid the real registry-backed
        # dataset/task/model components while still exercising HILServer's
        # single-stage bootstrap behavior.
        self.load_settings_patcher = patch("hil_server.load_config", return_value=self.config)
        self.register_patcher = patch("hil_server.ensure_builtin_components_registered")
        self.dataset_registry_patcher = patch(
            "crest.runtime_bootstrap.dataset_registry.get",
            return_value=FakeDataset,
        )
        self.task_registry_patcher = patch(
            "crest.runtime_bootstrap.task_registry.get",
            return_value=FakeTask,
        )
        self.model_family_registry_patcher = patch(
            "crest.runtime_bootstrap.model_family_registry.get",
            return_value=FakeModelFamily,
        )
        self.context = MagicMock()
        self.socket = MagicMock()
        self.context.socket.return_value = self.socket
        self.zmq_patcher = patch("hil_server.zmq.Context.instance", return_value=self.context)

        self.load_settings_mock = self.load_settings_patcher.start()
        self.register_mock = self.register_patcher.start()
        self.dataset_registry_mock = self.dataset_registry_patcher.start()
        self.task_registry_mock = self.task_registry_patcher.start()
        self.model_family_registry_mock = self.model_family_registry_patcher.start()
        self.zmq_mock = self.zmq_patcher.start()

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        patch.stopall()

    def build_server(self) -> HILServer:
        """Return a configured ``HILServer`` using the mocked dependencies.

        Returns
        -------
        HILServer
            Server instance whose heavy filesystem and dataset dependencies have
            already been replaced with lightweight doubles.
        """
        server = HILServer()
        server._sync_sketch_variant = MagicMock(return_value=Path("odom_tcn/odom_tcn.ino"))
        return server

    @staticmethod
    def request_runtime_metadata(**overrides: object) -> Dict:
        """Build valid runtime metadata for direct `determine_metrics(...)` calls.

        Parameters
        ----------
        **overrides : object
            Field overrides merged into the default minimal request payload.

        Returns
        -------
        addict.Dict
            Runtime metadata containing the required FLOP/input-shape fields
            expected by ``HILServer.determine_metrics``.
        """
        payload = {"flops": 1, "timesteps": 32, "input_dim": 6}
        payload.update(overrides)
        return Dict(payload)

    @staticmethod
    def request_family_hparams(**overrides: object) -> Dict:
        """Build a minimal family-hyperparameter payload for HIL tests.

        Parameters
        ----------
        **overrides : dict[str, object]
            Configuration overrides merged for the test case.

        Returns
        -------
        Dict
            Family hyperparameters extracted from the sample request.
        """
        payload = {}
        payload.update(overrides)
        return Dict(payload)

    def request_payload(
        self,
        *,
        family_hparams: Dict | None = None,
        runtime_metadata: Dict | None = None,
        device_options_overrides: dict | None = None,
    ) -> dict[str, object]:
        """Build the structured REQ payload used by the server loop tests.

        Parameters
        ----------
        family_hparams : Dict | None
            Model-family hyperparameters used by the request builder.
        runtime_metadata : Dict | None
            Runtime metadata included in the HIL request.
        device_options_overrides : dict | None
            Device option overrides merged into the test configuration.

        Returns
        -------
        dict[str, object]
            Payload dictionary used by the HIL request test.
        """
        payload: dict[str, object] = {
            "family_hparams": self.request_family_hparams() if family_hparams is None else family_hparams,
            "runtime_metadata": self.request_runtime_metadata() if runtime_metadata is None else runtime_metadata,
            "quantization_mode": "int8_ptq",
        }
        if device_options_overrides is not None:
            payload["device_options_overrides"] = device_options_overrides
        return payload


class DetermineMetricsTests(HILServerTestCase):
    """Tests for the conversion + metrics pipeline in `determine_metrics`."""

    def test_conversion_pipeline_invoked_in_order(self) -> None:
        """Backend preparation should feed the request builder and metric collection."""
        # The conversion pipeline should run in a fixed order so later stages always see the expected intermediate artifact.
        server = self.build_server()
        fake_model = MagicMock()
        fake_model.input_shape = (None, 32, 6)
        FakeModelFamily.model = fake_model
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.candidate_dir
        fake_metrics = {"ram_bytes": 1024}

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.collect_metrics", return_value=fake_metrics) as collect_mock:
            result = server.determine_metrics(
                self.request_family_hparams(nb_filters=8),
                self.request_runtime_metadata(flops=123),
            )

        self.assertEqual(len(FakeModelFamily.materialize_calls), 1)
        self.assertEqual(
            FakeModelFamily.materialize_calls[0]["hparams"],
            {"nb_filters": 8},
        )
        fake_device.prepare_candidate.assert_called_once()
        prepare_request = fake_device.prepare_candidate.call_args.kwargs["request"]
        self.assertIsInstance(prepare_request, CandidatePrepareRequest)
        self.assertIs(prepare_request.model, fake_model)
        self.assertEqual(prepare_request.model_variant, "approx_trained")
        self.assertIsNone(prepare_request.checkpoint_path)
        self.assertIs(prepare_request.calibration_split, self.calibration_split)
        self.assertEqual(prepare_request.quantization_mode, "int8_ptq")
        self.assertEqual(FakeModelFamily.materialize_calls[0]["model_variant"], "approx_trained")
        self.assertIsNone(FakeModelFamily.materialize_calls[0]["checkpoint_path"])
        self.assertEqual(len(FakeTask.validate_model_outputs_calls), 1)
        collect_mock.assert_called_once()
        self.assertEqual(result, fake_metrics)

    def test_determine_metrics_uses_config_export_variant(self) -> None:
        """Default HIL export selection should come from model params."""
        self.config.model.params.export_variant = "untrained"
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.candidate_dir

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.collect_metrics", return_value={"ok": True}):
            server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(),
            )

        self.assertEqual(FakeModelFamily.materialize_calls[0]["model_variant"], "untrained")

    def test_determine_metrics_explicit_model_variant_overrides_config(self) -> None:
        """Explicit model variants should override config-owned defaults."""
        self.config.model.params.export_variant = "untrained"
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.candidate_dir

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.collect_metrics", return_value={"ok": True}):
            server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(),
                model_variant="approx_trained",
            )

        self.assertEqual(FakeModelFamily.materialize_calls[0]["model_variant"], "approx_trained")

    def test_determine_metrics_uses_config_checkpoint_for_trained_variant(self) -> None:
        """Trained variants should default to the config-derived checkpoint path."""
        self.config.model.params.export_variant = "trained"
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.candidate_dir

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.collect_metrics", return_value={"ok": True}):
            server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(),
            )

        self.assertEqual(
            FakeModelFamily.materialize_calls[0]["checkpoint_path"],
            self.config.outputs.checkpoint_path,
        )

    def test_determine_metrics_rejects_missing_config_export_variant(self) -> None:
        """Missing config export variants should return a structured config error."""
        server = self.build_server()
        server.model_config["params"] = Dict()
        server._pipeline_bootstrapped = True

        metrics = server.determine_metrics(
            self.request_family_hparams(),
            self.request_runtime_metadata(),
        )

        self.assertEqual(metrics["backend_error_kind"], "config")
        self.assertIn("model.params.export_variant", metrics["backend_error_detail"])
        self.assertEqual(FakeModelFamily.materialize_calls, [])

    def test_collect_metrics_receives_expected_fields(self) -> None:
        """Key hyperparameters should flow through untouched to the controller."""
        # The HIL request builder should hand collect_metrics the exact resolved fields needed for hardware evaluation.
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.candidate_dir
        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.collect_metrics", return_value={"ok": True}) as collect_mock:
            server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(flops=999, input_dim=6),
            )

        # The normalized request should carry the resolved dimensions and latency target into collect_metrics.
        request = collect_mock.call_args.args[0]
        self.assertEqual(request.flops, 999)
        self.assertEqual(request.input_dim, 6)
        self.assertEqual(request.device_name, self.config.device.name)
        self.assertEqual(request.dirpath, self.config.outputs.candidate_dir.resolve())
        self.assertAlmostEqual(
            request.latency_budget_ms,
            (self.config.dataset.params.stride / self.config.dataset.params.sampling_rate_hz) * 1000,
        )

    def test_collect_metrics_uses_device_latency_budget_override(self) -> None:
        """Device-level latency-budget overrides should win over dataset cadence.

        Returns
        -------
        None
            Asserts the normalized metrics request carries the explicit device
            latency budget.
        """
        # Device-level latency-budget overrides should win when the request explicitly provides them.
        server = self.build_server()
        self.config.device.latency_budget_ms = 75.0
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.candidate_dir

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.collect_metrics", return_value={"ok": True}) as collect_mock:
            server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(flops=999, input_dim=6),
            )

        request = collect_mock.call_args.args[0]
        self.assertEqual(request.latency_budget_ms, 75.0)

    def test_collect_metrics_uses_dataset_batch_period_metadata(self) -> None:
        """Synthetic feature-batch metadata should drive HIL cadence.

        Returns
        -------
        None
            Asserts metadata-owned batch periods flow into the collected
            metrics request without requiring legacy stride fields.
        """
        server = self.build_server()
        self.config.dataset.params = Dict(directory="data", calibration_windows=100)
        FakeDataset.bundle = DatasetBundle(
            train=DataSplit(
                inputs=np.zeros((1, 201, 64), dtype=np.float32),
                targets=np.zeros((1,), dtype=np.int64),
            ),
            input_shape=(201, 64),
            input_dtype="float32",
            metadata={"batch_period_ms": 2000, "window_size": 201, "input_dim": 64},
        )
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = False
        fake_device.requires_training_data.return_value = False
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.candidate_dir

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.collect_metrics", return_value={"ok": True}) as collect_mock:
            server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(flops=999, timesteps=201, input_dim=64),
            )

        request = collect_mock.call_args.args[0]
        self.assertEqual(request.latency_budget_ms, 2000.0)
        self.assertEqual(request.window_size, 201)
        self.assertEqual(request.input_dim, 64)

    def test_dataset_batch_period_failure_metrics_survive_request_errors(self) -> None:
        """Bootstrapped request failures should report loaded batch cadence.

        Returns
        -------
        None
            Asserts already-loaded dataset metadata is reused when malformed
            request metrics are generated.
        """
        server = self.build_server()
        self.config.dataset.params = Dict(directory="data")
        FakeDataset.bundle = DatasetBundle(
            train=self.train_split,
            input_shape=(32, 6),
            input_dtype="float32",
            metadata={"batch_period_ms": 1500, "window_size": 32, "input_dim": 6},
        )
        server._ensure_pipeline_bootstrapped()

        with patch("hil_server.get_microcontroller_device") as device_mock:
            metrics = server.determine_metrics(
                self.request_family_hparams(),
                Dict(flops=123, input_dim=6),
            )

        device_mock.assert_not_called()
        self.assertEqual(metrics["backend_error_kind"], "request")
        self.assertEqual(metrics["latency_budget_ms"], 1500.0)

    def test_unbootstrapped_request_failures_do_not_load_dataset(self) -> None:
        """Malformed request failures should not force dataset loading.

        Returns
        -------
        None
            Asserts config-owned batch cadence can populate failure metrics
            without bootstrapping the dataset.
        """
        self.config.dataset.params = Dict(directory="data", batch_period_ms=2000)
        server = self.build_server()
        FakeDataset.load_calls = []

        with patch("hil_server.get_microcontroller_device") as device_mock:
            metrics = server.determine_metrics(
                self.request_family_hparams(),
                Dict(flops=123, input_dim=6),
            )

        device_mock.assert_not_called()
        self.assertEqual(FakeDataset.load_calls, [])
        self.assertEqual(metrics["backend_error_kind"], "request")
        self.assertEqual(metrics["latency_budget_ms"], 2000.0)

    def test_determine_metrics_rejects_missing_timesteps(self) -> None:
        # Missing timesteps should be rejected before the server stages any backend work.
        """Validate determine metrics rejects missing timesteps."""
        server = self.build_server()

        with patch("hil_server.get_microcontroller_device") as device_mock:
            metrics = server.determine_metrics(
                self.request_family_hparams(),
                Dict(flops=123, input_dim=6),
            )

        device_mock.assert_not_called()
        self.assertEqual(metrics["backend_error_kind"], "request")
        self.assertIn("timesteps", metrics["backend_error_detail"])

    def test_determine_metrics_rejects_missing_input_dim(self) -> None:
        # Missing input_dim should be rejected before the server stages any backend work.
        """Validate determine metrics rejects missing input dim."""
        server = self.build_server()

        with patch("hil_server.get_microcontroller_device") as device_mock:
            metrics = server.determine_metrics(
                self.request_family_hparams(),
                Dict(flops=123, timesteps=32),
            )

        device_mock.assert_not_called()
        self.assertEqual(metrics["backend_error_kind"], "request")
        self.assertIn("input_dim", metrics["backend_error_detail"])

    def test_determine_metrics_rejects_non_integer_runtime_fields(self) -> None:
        # Non-integer runtime dimensions should fail request validation before any backend is touched.
        """Validate determine metrics rejects non integer runtime fields."""
        server = self.build_server()

        with patch("hil_server.get_microcontroller_device") as device_mock:
            metrics = server.determine_metrics(
                self.request_family_hparams(),
                Dict(flops=123, timesteps="abc", input_dim=6),
            )

        device_mock.assert_not_called()
        self.assertEqual(metrics["backend_error_kind"], "request")
        self.assertIn("timesteps", metrics["backend_error_detail"])

    def test_determine_metrics_rejects_dimension_mismatches(self) -> None:
        # Shape mismatches should be rejected before the server tries to prepare a candidate for hardware.
        """Validate determine metrics rejects dimension mismatches."""
        server = self.build_server()

        with patch("hil_server.get_microcontroller_device") as device_mock:
            metrics = server.determine_metrics(
                self.request_family_hparams(),
                Dict(flops=123, timesteps=16, input_dim=6),
            )

        device_mock.assert_not_called()
        self.assertEqual(metrics["backend_error_kind"], "request")
        self.assertIn("do not match", metrics["backend_error_detail"])

    def test_determine_metrics_rejects_invalid_model_build_context_input_shapes(self) -> None:
        # Invalid model-build context shapes should come back as request errors instead of backend failures.
        """Validate determine metrics rejects invalid model build context input shapes."""
        server = self.build_server()
        server._ensure_pipeline_bootstrapped()

        for invalid_shape in (None, (None, 6), ("abc", 6), (0, 6), (32, False)):
            with self.subTest(input_shape=invalid_shape):
                server.model_build_context.input_shape = invalid_shape
                with patch("hil_server.get_microcontroller_device") as device_mock:
                    metrics = server.determine_metrics(
                        self.request_family_hparams(),
                        self.request_runtime_metadata(),
                    )
                device_mock.assert_not_called()
                self.assertEqual(metrics["backend_error_kind"], "request")
                self.assertIn("2D logical input shape", metrics["backend_error_detail"])

    def test_determine_metrics_runs_arduino_cadenced_second_pass_after_successful_base_run(self) -> None:
        # Cadenced Arduino runs should only schedule the second pass after the base latency pass succeeds, otherwise follow-up telemetry hides the primary failure.
        """Validate determine metrics runs Arduino cadenced second pass after successful base run."""
        server = self.build_server()
        self.config.device.name = "ARDUINO_NANO_33_BLE_SENSE"
        self.config.device.runtime_mode = "cadenced"
        self.config.training.energy_aware = True
        self.config.training.input_mode = "uniform"
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.supports_energy_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.candidate_dir
        fake_device.evaluate.return_value = SimpleNamespace(
            error_code=HIL_ERROR_OK,
            power_metrics={"energy_mj_per_inference": 1.5},
        )

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.collect_metrics",
            return_value={
                "error_code": HIL_ERROR_OK,
                "arena_bytes": 4096,
                "runtime_mode": "back_to_back",
                "cadenced_error_code": -1,
                "cadenced_error_label": None,
                "cadenced_active_inference_latency_ms": -1.0,
                "cadenced_energy_mj_per_window": -1.0,
                "cadenced_energy_mj_per_trial": -1.0,
            },
        ):
            metrics = server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(flops=123),
            )

        fake_device.set_input_mode.assert_called_once()
        self.assertEqual(fake_device.set_input_mode.call_args.kwargs["runtime_phase"], "cadenced")
        fake_device.evaluate.assert_called_once()
        self.assertEqual(metrics["runtime_mode"], "cadenced")
        self.assertEqual(metrics["cadenced_error_code"], HIL_ERROR_OK)
        self.assertAlmostEqual(metrics["cadenced_energy_mj_per_window"], 1.5)
        self.assertAlmostEqual(metrics["cadenced_energy_mj_per_trial"], 15.0)

    def test_determine_metrics_discards_arduino_cadenced_second_pass_latency(self) -> None:
        # The Arduino cadenced follow-up should contribute energy data without overwriting the latency from the primary pass.
        """Validate determine metrics discards Arduino cadenced second pass latency."""
        server = self.build_server()
        self.config.device.name = "ARDUINO_NANO_33_BLE_SENSE"
        self.config.device.runtime_mode = "cadenced"
        self.config.training.energy_aware = True
        self.config.training.input_mode = "uniform"
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.supports_energy_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.candidate_dir
        fake_device.evaluate.return_value = SimpleNamespace(
            error_code=HIL_ERROR_OK,
            latency_s=0.2,
            power_metrics={"energy_mj_per_inference": 2.0},
        )

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.collect_metrics",
            return_value={
                "error_code": HIL_ERROR_OK,
                "arena_bytes": 4096,
                "runtime_mode": "back_to_back",
                "cadenced_error_code": -1,
                "cadenced_error_label": None,
                "cadenced_active_inference_latency_ms": -1.0,
                "cadenced_energy_mj_per_window": -1.0,
                "cadenced_energy_mj_per_trial": -1.0,
            },
        ):
            metrics = server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(flops=123),
            )

        self.assertEqual(metrics["cadenced_active_inference_latency_ms"], -1.0)

    def test_determine_metrics_uses_override_clock_for_runtime_options_only(self) -> None:
        # Clock overrides should affect runtime measurement options without mutating the static model-build context.
        """Validate determine metrics uses override clock for runtime options only."""
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.candidate_dir
        fake_metrics = {"ram_bytes": 1024, "clock_hz": 400000000.0}

        with patch("hil_server.resolve_device_options", return_value={"cpu_clock_mhz": 600}), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ) as device_mock, patch("hil_server.collect_metrics", return_value=fake_metrics) as collect_mock:
            result = server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(flops=123),
                device_options_overrides={"cpu_clock_mhz": 400},
            )

        self.assertEqual(result["cpu_clock_mhz_requested"], 400)
        self.assertEqual(device_mock.call_args.kwargs["device_options"]["cpu_clock_mhz"], 400)
        request = collect_mock.call_args.args[0]
        self.assertEqual(request.device_options["cpu_clock_mhz"], 400)

    def test_determine_metrics_unsampled_stm_clock_logs_minus_one(self) -> None:
        # If the STM32 backend does not report a sampled clock, the server should emit the stable sentinel rather than fabricating a value.
        """Validate determine metrics unsampled stm clock logs minus one."""
        server = self.build_server()
        self.config.device.name = "STM32_NUCLEO_N657X0_Q"
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.candidate_dir

        with patch("hil_server.resolve_device_options", return_value={"cpu_clock_mhz": 600}), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.collect_metrics", return_value={"ram_bytes": 1024, "clock_hz": 600000000.0}):
            metrics = server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(),
            )

        self.assertEqual(metrics["cpu_clock_mhz_requested"], -1)

    def test_determine_metrics_invalid_clock_override_returns_request_error(self) -> None:
        # Invalid clock overrides should return a structured request error before any backend work starts.
        """Validate determine metrics invalid clock override returns request error."""
        server = self.build_server()

        with patch("hil_server.resolve_device_options", return_value={"cpu_clock_mhz": 600}), patch(
            "hil_server.get_microcontroller_device"
        ) as device_mock:
            metrics = server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(flops=123),
                device_options_overrides={"cpu_clock_mhz": float("nan")},
            )

        device_mock.assert_not_called()
        self.assertEqual(metrics["error_code"], hil_server_module.HIL_MASTER_FATAL)
        self.assertEqual(metrics["backend_error_kind"], "request")
        self.assertIn("device_options_overrides.cpu_clock_mhz", metrics["backend_error_detail"])


class StartLoopTests(HILServerTestCase):
    """Validate the ZeroMQ REP loop implemented in `start`."""

    def test_start_binds_and_serves_single_request(self) -> None:
        """The server should bind, log, process one payload, and reply.

        Returns
        -------
        None
            The test passes when the REP loop emits request-loop diagnostics
            while preserving the existing request and cleanup behavior.
        """
        # The REP loop should bind, process one payload, and reply before cleanup so the happy path stays observable in tests.
        server = self.build_server()
        payload = self.request_payload(
            family_hparams=self.request_family_hparams(),
            runtime_metadata=self.request_runtime_metadata(flops=1, timesteps=32, input_dim=2),
        )
        metrics = {"flash_bytes": 2048}

        # Mock determine_metrics to return fake metrics, and simulate one request then interrupt.
        server.determine_metrics = MagicMock(return_value=metrics)
        self.socket.recv_json.side_effect = [payload, KeyboardInterrupt()]

        with self.assertLogs("hil_server", level="DEBUG") as captured:
            server.start()

        # Verify socket binding, message processing, and cleanup.
        endpoint = f"tcp://{self.config.network.host}:{self.config.network.port}"
        self.socket.bind.assert_called_once_with(endpoint)
        server.determine_metrics.assert_called_once_with(
            family_hparams=self.request_family_hparams(),
            runtime_metadata=self.request_runtime_metadata(flops=1, timesteps=32, input_dim=2),
            quantization_mode="int8_ptq",
            device_options_overrides=None,
        )
        self.socket.send_json.assert_called_once_with(metrics)
        self.socket.close.assert_called_once_with(linger=0)
        self.context.term.assert_called_once()
        joined_logs = "\n".join(captured.output)
        self.assertIn("Listening for hyperparameters", joined_logs)
        self.assertIn("Received payload", joined_logs)
        self.assertIn("Sending metrics", joined_logs)

    def test_start_normalizes_structured_payload(self) -> None:
        # Structured network payloads should normalize into the same call signature used by simpler requests.
        """Validate start normalizes structured payload."""
        server = self.build_server()
        payload = self.request_payload(
            runtime_metadata=self.request_runtime_metadata(flops=1, timesteps=32, input_dim=2),
            device_options_overrides={"cpu_clock_mhz": 400},
        )
        metrics = {"flash_bytes": 2048}
        server.determine_metrics = MagicMock(return_value=metrics)
        self.socket.recv_json.side_effect = [payload, KeyboardInterrupt()]

        server.start()

        server.determine_metrics.assert_called_once_with(
            family_hparams=self.request_family_hparams(),
            runtime_metadata=self.request_runtime_metadata(flops=1, timesteps=32, input_dim=2),
            quantization_mode="int8_ptq",
            device_options_overrides={"cpu_clock_mhz": 400},
        )

    def test_start_ignores_payload_model_variant_field(self) -> None:
        """Network payloads should not override config-owned export variants."""
        server = self.build_server()
        payload = self.request_payload(
            runtime_metadata=self.request_runtime_metadata(flops=1, timesteps=32, input_dim=2),
        )
        payload["model_variant"] = "trained"
        metrics = {"flash_bytes": 2048}
        server.determine_metrics = MagicMock(return_value=metrics)
        self.socket.recv_json.side_effect = [payload, KeyboardInterrupt()]

        server.start()

        server.determine_metrics.assert_called_once_with(
            family_hparams=self.request_family_hparams(),
            runtime_metadata=self.request_runtime_metadata(flops=1, timesteps=32, input_dim=2),
            quantization_mode="int8_ptq",
            device_options_overrides=None,
        )

    def test_start_returns_request_error_for_malformed_payload(self) -> None:
        """Malformed request payloads should log an error and return metrics.

        Returns
        -------
        None
            The test passes when malformed input is converted into structured
            failure metrics and the request boundary logs the validation error.
        """
        # Malformed network payloads should come back as structured request errors instead of exploding the server loop.
        server = self.build_server()
        payload = {
            "family_hparams": None,
            "runtime_metadata": {"flops": 1, "timesteps": 32, "input_dim": 2},
            "device_options_overrides": {"cpu_clock_mhz": 400},
        }
        self.socket.recv_json.side_effect = [payload, KeyboardInterrupt()]

        with self.assertLogs("hil_server", level="ERROR") as captured:
            server.start()

        self.socket.send_json.assert_called_once()
        metrics = self.socket.send_json.call_args.args[0]
        self.assertEqual(metrics["error_code"], hil_server_module.HIL_MASTER_FATAL)
        self.assertEqual(metrics["backend_error_kind"], "request")
        self.assertIn("family_hparams", metrics["backend_error_detail"])
        self.assertIn("Invalid HIL request payload", "\n".join(captured.output))

    def test_start_interrupt_cleans_up_resources(self) -> None:
        """If recv_json immediately raises, we should still close the socket."""
        # Keyboard interrupts should still close the REP socket and terminate the context cleanly.
        server = self.build_server()
        self.socket.recv_json.side_effect = KeyboardInterrupt()

        server.start()

        # Ensure no reply sent, but cleanup still happens.
        self.socket.send_json.assert_not_called()
        self.socket.close.assert_called_once_with(linger=0)
        self.context.term.assert_called_once()


class InitializationTests(HILServerTestCase):
    """Ensure constructor wiring uses modular bootstrap and lazy calibration."""

    def test_constructor_preserves_explicit_component_selection_without_bootstrap(self) -> None:
        # Constructor setup should defer bootstrap while keeping the explicit component contract intact.
        """Validate constructor preserves explicit component selection without bootstrap."""
        server = self.build_server()

        self.dataset_registry_mock.assert_not_called()
        self.task_registry_mock.assert_not_called()
        self.model_family_registry_mock.assert_not_called()
        self.assertEqual(FakeDataset.load_calls, [])
        self.assertEqual(server.dataset_name, "oxiod")
        self.assertEqual(server.task_name, "odometry_regression")
        self.assertEqual(server.model_family_name, "odom_tcn")
        self.assertEqual(server.model_config["params"].export_variant, "approx_trained")
        self.assertEqual(server.model_config["search"], Dict())

    def test_constructor_preserves_model_params_and_search_blocks(self) -> None:
        # Preloaded model params and search blocks should survive construction unchanged.
        """Validate constructor preserves model params and search blocks."""
        self.config.model = SimpleNamespace(
            family="custom_family",
            params=Dict(width=8, export_variant="untrained"),
            search=Dict(depth=[2, 3]),
        )

        server = HILServer(config=self.config)

        self.model_family_registry_mock.assert_not_called()
        self.assertEqual(server.model_config["params"].width, 8)
        self.assertEqual(server.model_config["params"].export_variant, "untrained")
        self.assertEqual(server.model_config["search"].depth, [2, 3])

    def test_constructor_requires_explicit_dataset_task_and_model_blocks(self) -> None:
        # The breaking contract rejects configs that omit component blocks.
        """Validate constructor requires explicit dataset task and model blocks."""
        delattr(self.config, "dataset")

        with self.assertRaisesRegex(KeyError, "dataset"):
            HILServer(config=self.config)

    def test_calibration_resolution_is_lazy_until_backend_requires_it(self) -> None:
        """Ensure the constructor does not eagerly resolve calibration data.

        Returns
        -------
        None
        """
        # Constructor setup should not resolve calibration data until a backend path actually needs it.
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.candidate_dir

        self.assertEqual(FakeDataset.calibration_calls, [])

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.collect_metrics", return_value={"ok": True}):
            server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(),
            )

        self.assertEqual(len(FakeDataset.calibration_calls), 1)

    def test_pipeline_bootstraps_once_across_multiple_requests(self) -> None:
        # The server should cache the modular bootstrap so repeated requests do not reload datasets or rebuild component state.
        """Validate pipeline bootstraps once across multiple requests."""
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.candidate_dir

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.collect_metrics", return_value={"ok": True}):
            server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(flops=1),
            )
            server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(flops=2),
            )

        self.assertEqual(len(FakeDataset.load_calls), 1)
        self.assertEqual(len(FakeTask.build_target_spec_calls), 1)

    def test_preloaded_config_uses_single_stage_bootstrap(self) -> None:
        """Ensure preloaded config bypasses ``load_config`` and stays lazy until needed."""
        # Passing a preloaded config should bypass load_config while keeping the lazy modular bootstrap behavior.
        server = HILServer(config=self.config)

        self.load_settings_mock.assert_not_called()
        self.assertIsNone(server.dataset_bundle)
        self.assertEqual(len(FakeTask.build_target_spec_calls), 0)

    def test_latency_budget_fallback_uses_dataset_config_without_legacy_data_block(self) -> None:
        # Latency-budget fallback should prefer dataset params directly from the explicit dataset block.
        """Validate latency budget fallback uses dataset config without legacy data block."""
        self.config.dataset = SimpleNamespace(
            name="oxiod",
            params=Dict(directory="data", sampling_rate_hz=100, window_size=32, stride=4),
        )
        server = HILServer(config=self.config)
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.candidate_dir

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.collect_metrics", return_value={"ok": True}) as collect_mock:
            server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(flops=123),
            )

        request = collect_mock.call_args.args[0]
        self.assertEqual(request.latency_budget_ms, 40.0)

    def test_require_calibration_split_raises_when_dataset_has_no_calibration_data(self) -> None:
        # Calibration-dependent requests should fail with a clear error when the dataset cannot supply any calibration split.
        """Validate require calibration split raises when dataset has no calibration data."""
        server = self.build_server()
        server.dataset_bundle = DatasetBundle(
            train=self.train_split,
            val=self.train_split,
            test=self.train_split,
            calibration=None,
            input_shape=(32, 6),
            input_dtype="float32",
            metadata={"window_size": 32, "input_dim": 6},
        )

        with self.assertRaisesRegex(ValueError, "does not provide calibration data"):
            server._require_calibration_split()

    def test_determine_metrics_returns_config_error_when_calibration_data_is_missing(self) -> None:
        # Missing calibration data should surface as a structured config error rather than a backend crash.
        """Validate determine metrics returns config error when calibration data is missing."""
        server = self.build_server()
        server.dataset_bundle = DatasetBundle(
            train=self.train_split,
            val=self.train_split,
            test=self.train_split,
            calibration=None,
            input_shape=(32, 6),
            input_dtype="float32",
            metadata={"window_size": 32, "input_dim": 6},
        )
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ):
            metrics = server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(),
            )

        self.assertEqual(metrics["error_code"], hil_server_module.HIL_MASTER_FATAL)
        self.assertEqual(metrics["backend_error_kind"], "config")
        self.assertIn("calibration data", metrics["backend_error_detail"])
        fake_device.prepare_candidate.assert_not_called()

    def test_determine_metrics_uses_dataset_calibration_fallback_when_bundle_calibration_missing(self) -> None:
        # If the bundle omits a calibration split, the server should ask the dataset adapter for one before failing the request.
        """Validate determine metrics uses dataset calibration fallback when bundle calibration missing."""
        FakeDataset.bundle = DatasetBundle(
            train=self.train_split,
            val=self.train_split,
            test=self.train_split,
            calibration=None,
            input_shape=(32, 6),
            input_dtype="float32",
            metadata={"window_size": 32, "input_dim": 6},
        )
        FakeDataset.calibration_split_override = self.train_split
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.candidate_dir

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.collect_metrics", return_value={"ok": True}):
            server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(),
            )

        request = fake_device.prepare_candidate.call_args.kwargs["request"]
        self.assertIs(request.calibration_split, self.train_split)

    def test_stm_runtime_backend_keeps_hil_enabled_when_supported(self) -> None:
        """Ensure STM runtime-capable backends keep HIL enabled.

        Returns
        -------
        None
        """
        # STM32 runtime backends should keep HIL enabled when the backend can measure on real hardware instead of silently downgrading to proxy mode.
        self.config.device.name = "STM32_NUCLEO_N657X0_Q"
        self.config.device.stm32 = SimpleNamespace(project_root=Path("stm_fsbl"))
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.supports_energy_measurement.return_value = True
        fake_device.prepare_candidate.return_value = Path("stm_fsbl")

        with patch("hil_server.resolve_device_options", return_value={"project_root": Path("stm_fsbl")}), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.build_collect_metrics_request") as request_mock, patch(
            "hil_server.collect_metrics",
            return_value={"ok": True},
        ) as collect_mock:
            request_mock.return_value = CollectMetricsRequest(
                hil_enabled=True,
                energy_aware=False,
                flops=1,
                device_name="STM32_NUCLEO_N657X0_Q",
                window_size=32,
                input_dim=6,
                dirpath=Path("stm_fsbl"),
                latency_proxy_max_flops=5_000_000,
                serial_port="ttyACM0",
                latency_budget_ms=40.0,
                dut_ready_timeout_s=5.0,
                serial_timeout_s=12.0,
                harness=None,
                device_options={},
            )
            result = server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(),
            )

        record_path = Path(result.pop("measurement_request_record"))
        self.assertTrue(record_path.is_file())
        self.assertEqual(result, {"ok": True, "cpu_clock_mhz_requested": -1, "latency_budget_ms": 40.0})
        self.assertEqual(len(FakeModelFamily.materialize_calls), 1)
        fake_device.prepare_candidate.assert_called_once()
        fake_device.cleanup_prepared_candidate.assert_called_once_with(Path("stm_fsbl"))
        collect_mock.assert_called_once()
        self.assertTrue(request_mock.call_args.kwargs["hil_enabled"])
        self.assertIsNone(server.active_sketch_path)

    def test_stm_backend_keeps_energy_aware_requests_when_supported(self) -> None:
        """Ensure STM energy-aware requests remain enabled once the backend supports them.

        Returns
        -------
        None
        """
        # STM32 backends that support energy measurement should preserve energy-aware requests instead of downgrading them.
        self.config.training.energy_aware = True
        self.config.device.name = "STM32_NUCLEO_N657X0_Q"
        self.config.device.stm32 = SimpleNamespace(project_root=Path("stm_fsbl"))
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.supports_energy_measurement.return_value = True
        fake_device.prepare_candidate.return_value = Path("stm_fsbl")

        with patch(
            "hil_server.resolve_device_options",
            return_value={"project_root": Path("stm_fsbl")},
        ), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch(
            "hil_server.collect_metrics",
            return_value={"ok": True},
        ), patch(
            "hil_server.build_collect_metrics_request"
        ) as request_mock:
            request_mock.return_value = CollectMetricsRequest(
                hil_enabled=True,
                energy_aware=True,
                flops=1,
                device_name="STM32_NUCLEO_N657X0_Q",
                window_size=32,
                input_dim=6,
                dirpath=Path("stm_fsbl"),
                latency_proxy_max_flops=5_000_000,
                serial_port="ttyACM0",
                latency_budget_ms=40.0,
                dut_ready_timeout_s=5.0,
                serial_timeout_s=12.0,
                harness=None,
                device_options={},
            )
            server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(),
            )

        self.assertTrue(request_mock.call_args.kwargs["energy_aware"])

    def test_arduino_and_stm_candidate_staging_diverge_at_active_sketch_boundary(self) -> None:
        """Ensure Arduino candidate prep activates a sketch while STM keeps project staging.

        Returns
        -------
        None
        """
        # Arduino and STM32 staging should split at the active-sketch boundary so each backend owns the files it is responsible for.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            arduino_server = self.build_server()
            arduino_server.active_sketch_path = None
            arduino_prepared_dir = tmp_path / "arduino"
            arduino_prepared_dir.mkdir()
            expected_sketch = arduino_staged_sketch_path(arduino_prepared_dir)
            expected_sketch.write_text("// staged sketch\n", encoding="utf-8")

            fake_arduino_device = MagicMock()
            fake_arduino_device.requires_candidate_model.return_value = True
            fake_arduino_device.requires_training_data.return_value = False
            fake_arduino_device.supports_runtime_measurement.return_value = False
            fake_arduino_device.supports_energy_measurement.return_value = False
            fake_arduino_device.prepare_candidate.return_value = arduino_prepared_dir

            with patch("hil_server.resolve_device_options", return_value=None), patch(
                "hil_server.get_microcontroller_device",
                return_value=fake_arduino_device,
            ), patch("hil_server.build_collect_metrics_request") as arduino_request_mock, patch(
                "hil_server.collect_metrics",
                return_value={"ok": True},
            ):
                arduino_request_mock.return_value = CollectMetricsRequest(
                    hil_enabled=False,
                    energy_aware=False,
                    flops=1,
                    device_name="ARDUINO_NANO_33_BLE_SENSE",
                    window_size=32,
                    input_dim=6,
                    dirpath=arduino_prepared_dir,
                    latency_proxy_max_flops=5_000_000,
                    serial_port="ttyACM0",
                    device_options=None,
                )
                arduino_server.determine_metrics(
                    self.request_family_hparams(),
                    self.request_runtime_metadata(),
                )

            self.assertEqual(arduino_server.active_sketch_path, expected_sketch)

            self.config.device.name = "STM32_NUCLEO_N657X0_Q"
            self.config.device.stm32 = SimpleNamespace(project_root=tmp_path / "stm32" / "FSBL")
            stm_server = self.build_server()
            stm_server.active_sketch_path = None
            stm_prepared_dir = tmp_path / "stm32" / "FSBL"
            stm_prepared_dir.mkdir(parents=True)

            fake_stm_device = MagicMock()
            fake_stm_device.requires_candidate_model.return_value = True
            fake_stm_device.requires_training_data.return_value = False
            fake_stm_device.supports_runtime_measurement.return_value = False
            fake_stm_device.supports_energy_measurement.return_value = False
            fake_stm_device.prepare_candidate.return_value = stm_prepared_dir

            with patch(
                "hil_server.resolve_device_options",
                return_value={"project_root": stm_prepared_dir},
            ), patch(
                "hil_server.get_microcontroller_device",
                return_value=fake_stm_device,
            ), patch("hil_server.build_collect_metrics_request") as stm_request_mock, patch(
                "hil_server.collect_metrics",
                return_value={"ok": True},
            ):
                stm_request_mock.return_value = CollectMetricsRequest(
                    hil_enabled=False,
                    energy_aware=False,
                    flops=1,
                    device_name="STM32_NUCLEO_N657X0_Q",
                    window_size=32,
                    input_dim=6,
                    dirpath=stm_prepared_dir,
                    latency_proxy_max_flops=5_000_000,
                    serial_port="ttyACM0",
                    device_options={"project_root": stm_prepared_dir},
                )
                stm_server.determine_metrics(
                    self.request_family_hparams(),
                    self.request_runtime_metadata(),
                )

            self.assertIsNone(stm_server.active_sketch_path)

    def test_stm_prepare_candidate_failure_returns_structured_backend_metrics(self) -> None:
        """STM staging failures should become metrics-shaped backend errors."""
        # Backend staging failures should come back as structured metrics so callers can surface the exact hardware-side failure without crashing the request.
        self.config.device.name = "STM32_NUCLEO_N657X0_Q"
        self.config.device.stm32 = SimpleNamespace(project_root=Path("stm_fsbl"))
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.supports_energy_measurement.return_value = True
        fake_device.prepare_candidate.side_effect = hil_server_module.stm32_cube_clt.WorkflowError(
            "ST Edge AI generation failed: unsupported operator"
        )

        with patch("hil_server.resolve_device_options", return_value={"project_root": Path("stm_fsbl")}), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ):
            metrics = server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(),
            )

        self.assertEqual(metrics["backend_error_kind"], "unsupported_model")
        self.assertIn("unsupported operator", metrics["backend_error_detail"])

    def test_request_build_failure_returns_structured_backend_metrics(self) -> None:
        """Request-building config failures should not crash the REP loop."""
        # Request-build failures should still return structured backend metrics so the caller can report the exact stage that failed.
        self.config.training.energy_aware = True
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.supports_energy_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.candidate_dir

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.build_collect_metrics_request",
            side_effect=RuntimeError(
                "Set device.harness_serial_port when runtime measurement requires the harness."
            ),
        ):
            metrics = server.determine_metrics(
                self.request_family_hparams(),
                self.request_runtime_metadata(),
            )

        self.assertEqual(metrics["error_code"], hil_server_module.HIL_MASTER_FATAL)
        self.assertEqual(metrics["backend_error_kind"], "config")
        self.assertIn("device.harness_serial_port", metrics["backend_error_detail"])
        fake_device.cleanup_prepared_candidate.assert_called_once_with(self.config.outputs.candidate_dir)

    def test_set_input_mode_delegates_to_backend_for_stm_phase1(self) -> None:
        """Ensure STM servers delegate input-mode changes to the backend.

        Returns
        -------
        None
        """
        # STM32 phase-one input-mode changes should delegate to the backend so sketch generation stays backend-owned.
        self.config.device.name = "STM32_NUCLEO_N657X0_Q"
        self.config.device.stm32 = SimpleNamespace(project_root=Path("stm_fsbl"))
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.set_input_mode.return_value = Path("stm_fsbl")

        with patch("hil_server.resolve_device_options", return_value={"project_root": Path("stm_fsbl")}), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ):
            out_path = server.set_input_mode("uniform")

        self.assertEqual(out_path, Path("stm_fsbl"))


class SketchVariantTests(unittest.TestCase):
    """Validate sketch variant selection and input-mode behaviors."""

    def _build_server(
        self,
        sketches_dir: Path,
        candidate_dir: Path,
        energy_aware: bool,
        input_mode: str,
        *,
        device_name: str = "ARDUINO_NANO_33_BLE_SENSE",
        target_core: str | None = None,
    ) -> HILServer:
        """Build a lightweight ``HILServer`` double for sketch-selection tests.

        Parameters
        ----------
        sketches_dir : Path
            Root sketches directory used by the selector under test.
        candidate_dir : Path
            Active output sketch directory.
        energy_aware : bool
            Whether energy-aware sketch variants should be selected.
        input_mode : str
            Requested input mode.
        device_name : str, optional
            Device profile used to route uniform sketch variants.
        target_core : str | None, optional
            Optional Portenta target core (``CM7``/``CM4``).

        Returns
        -------
        HILServer
            Server instance with just enough config for variant sync tests.
        """
        server = HILServer.__new__(HILServer)
        portenta_cfg = (
            SimpleNamespace(target_core=target_core) if target_core is not None else SimpleNamespace()
        )
        server.config = SimpleNamespace(
            training=SimpleNamespace(energy_aware=energy_aware, input_mode=input_mode),
            outputs=SimpleNamespace(candidate_dir=candidate_dir),
            device=SimpleNamespace(name=device_name, portenta=portenta_cfg),
        )
        server.sketch_variants_dir = sketches_dir
        server.active_sketch_path = None
        return server

    def _write_sketch(self, path: Path, label: str) -> None:
        """Write sketch.

        Parameters
        ----------
        path : Path
            Path to the file used by the helper.
        label : str
            Label written alongside the captured path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"// {label}\n")

    def test_selects_uniform_energy_sketch(self) -> None:
        """Energy-aware uniform staging should use the generic energy sketch."""
        # Uniform energy runs should pick the energy-enabled sketch variant so the DUT exports the expected telemetry.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            candidate_dir = Path(tmpdir) / "odom_tcn"
            self._write_sketch(sketches / "crest_inference_energy.ino", "uniform_shared")
            server = self._build_server(sketches, candidate_dir, energy_aware=True, input_mode="uniform")

            out_path = server._sync_sketch_variant()

            self.assertEqual(out_path, candidate_dir / "odom_tcn.ino")
            self.assertTrue(out_path.exists())
            self.assertIn("uniform_shared", out_path.read_text())

    def test_stages_sketch_using_candidate_dir_basename(self) -> None:
        """Arduino sketch staging should preserve the folder/sketch basename rule."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            candidate_dir = Path(tmpdir) / "audio_dscnn"
            self._write_sketch(sketches / "crest_inference_no_energy.ino", "audio_uniform")
            server = self._build_server(sketches, candidate_dir, energy_aware=False, input_mode="uniform")

            out_path = server._sync_sketch_variant()

            self.assertEqual(out_path, candidate_dir / "audio_dscnn.ino")
            self.assertTrue(out_path.exists())
            self.assertIn("audio_uniform", out_path.read_text())

    def test_selects_cadenced_uniform_sketch(self) -> None:
        """Cadenced staging should use the generic cadenced energy sketch."""
        # Cadenced energy runs must switch to the cadenced uniform sketch so the harness and DUT share the same timing protocol.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            candidate_dir = Path(tmpdir) / "odom_tcn"
            self._write_sketch(sketches / "crest_inference_energy_cadenced.ino", "cadenced_shared")
            server = self._build_server(sketches, candidate_dir, energy_aware=True, input_mode="uniform")

            out_path = server.set_input_mode("uniform", runtime_phase="cadenced")

            self.assertTrue(out_path.exists())
            self.assertIn("cadenced_shared", out_path.read_text())

    def test_selects_uniform_energy_sketch_for_portenta_cm7(self) -> None:
        """Portenta CM7 uniform staging should use the generic energy sketch."""
        # Uniform energy runs should pick the energy-enabled sketch variant so the DUT exports the expected telemetry.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            candidate_dir = Path(tmpdir) / "odom_tcn"
            self._write_sketch(sketches / "crest_inference_energy.ino", "uniform_shared_cm7")
            server = self._build_server(
                sketches,
                candidate_dir,
                energy_aware=True,
                input_mode="uniform",
                device_name="PORTENTA_H7",
                target_core="cm7",
            )

            out_path = server._sync_sketch_variant()

            self.assertTrue(out_path.exists())
            self.assertIn("uniform_shared_cm7", out_path.read_text())

    def test_selects_uniform_no_energy_sketch_for_portenta_cm4(self) -> None:
        """Portenta CM4 no-energy staging should use the generic no-energy sketch."""
        # Portenta CM4 uniform runs without energy measurement should select the no-energy sketch variant.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            candidate_dir = Path(tmpdir) / "odom_tcn"
            self._write_sketch(sketches / "crest_inference_no_energy.ino", "no_energy_shared_cm4")
            server = self._build_server(
                sketches,
                candidate_dir,
                energy_aware=False,
                input_mode="uniform",
                device_name="PORTENTA_H7",
                target_core="cm4",
            )

            out_path = server._sync_sketch_variant()

            self.assertTrue(out_path.exists())
            self.assertIn("no_energy_shared_cm4", out_path.read_text())

    def test_selects_oxiod_representative_variant_and_copies_header(self) -> None:
        """Representative OxIOD staging should copy the sketch and staged include header."""
        # Representative input-mode runs should switch sketches and copy the generated header that drives the sample payload.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            candidate_dir = Path(tmpdir) / "odom_tcn"
            self._write_sketch(
                sketches / "analysis_sketches/crest_inference_representative.ino",
                "representative",
            )
            header = sketches / "analysis_sketches/oxiod_input_data.h"
            header.write_text("// header\n")
            server = self._build_server(sketches, candidate_dir, energy_aware=True, input_mode="oxiod_representative")

            out_path = server._sync_sketch_variant()

            self.assertTrue(out_path.exists())
            self.assertIn("representative", out_path.read_text())
            self.assertTrue((candidate_dir / "crest_input_data.h").exists())
            self.assertEqual((candidate_dir / "crest_input_data.h").read_text(), "// header\n")

    def test_selects_oxiod_real_variant_and_copies_header(self) -> None:
        """Real-data OxIOD staging should copy the sketch and staged include header."""
        # Real-input runs should stage the real-data sketch and copy the matching generated header.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            candidate_dir = Path(tmpdir) / "odom_tcn"
            self._write_sketch(
                sketches / "analysis_sketches/crest_inference_real_data.ino",
                "real",
            )
            header = sketches / "analysis_sketches/oxiod_input_data.h"
            header.write_text("// header\n")
            server = self._build_server(sketches, candidate_dir, energy_aware=True, input_mode="oxiod_real")

            out_path = server._sync_sketch_variant()

            self.assertTrue(out_path.exists())
            self.assertIn("real", out_path.read_text())
            self.assertTrue((candidate_dir / "crest_input_data.h").exists())

    def test_selects_urbansound8k_variants_and_copies_audio_header(self) -> None:
        """UrbanSound8K modes should stage the shared sketches with the audio header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            candidate_dir = Path(tmpdir) / "audio_dscnn"
            self._write_sketch(
                sketches / "analysis_sketches/crest_inference_representative.ino",
                "representative",
            )
            self._write_sketch(
                sketches / "analysis_sketches/crest_inference_real_data.ino",
                "real",
            )
            header = sketches / "analysis_sketches/urbansound8k_input_data.h"
            header.write_text("// audio header\n")
            server = self._build_server(
                sketches,
                candidate_dir,
                energy_aware=True,
                input_mode="urbansound8k_representative",
            )

            representative_path = server._sync_sketch_variant()
            self.assertIn("representative", representative_path.read_text())
            self.assertEqual((candidate_dir / "crest_input_data.h").read_text(), "// audio header\n")

            real_path = server.set_input_mode("urbansound8k_real")
            self.assertIn("real", real_path.read_text())
            self.assertEqual((candidate_dir / "crest_input_data.h").read_text(), "// audio header\n")

    def test_missing_header_raises_for_representative(self) -> None:
        """Dataset representative staging should require its input header."""
        # Representative-mode staging should fail fast if the generated header is missing.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            candidate_dir = Path(tmpdir) / "odom_tcn"
            self._write_sketch(
                sketches / "analysis_sketches/crest_inference_representative.ino",
                "representative",
            )
            server = self._build_server(sketches, candidate_dir, energy_aware=True, input_mode="oxiod_representative")

            with self.assertRaises(FileNotFoundError):
                server._sync_sketch_variant()

    def test_invalid_input_mode_raises(self) -> None:
        # Unknown input modes should fail before sketch selection can stage the wrong artifact.
        """Validate invalid input mode raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            candidate_dir = Path(tmpdir) / "odom_tcn"
            server = self._build_server(sketches, candidate_dir, energy_aware=True, input_mode="bad_mode")

            with self.assertRaisesRegex(ValueError, "Unsupported input_mode"):
                server._sync_sketch_variant()

    def test_old_generic_input_modes_are_unsupported(self) -> None:
        """Old generic representative/real modes should fail through normal validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            candidate_dir = Path(tmpdir) / "odom_tcn"
            for mode in ("representative", "real"):
                with self.subTest(mode=mode):
                    server = self._build_server(sketches, candidate_dir, energy_aware=True, input_mode=mode)
                    with self.assertRaisesRegex(ValueError, "Unsupported input_mode"):
                        server._sync_sketch_variant()

    def test_cadenced_runtime_requires_uniform_input_mode(self) -> None:
        """Cadenced Arduino staging should still reject non-uniform inputs."""
        # Cadenced runtime mode should reject non-uniform input modes because the timing harness expects fixed windows.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            candidate_dir = Path(tmpdir) / "odom_tcn"
            self._write_sketch(
                sketches / "analysis_sketches/crest_inference_representative.ino",
                "representative",
            )
            header = sketches / "analysis_sketches/oxiod_input_data.h"
            header.write_text("// header\n")
            server = self._build_server(sketches, candidate_dir, energy_aware=True, input_mode="oxiod_representative")

            with self.assertRaises(ValueError):
                server.set_input_mode("oxiod_representative", runtime_phase="cadenced")

    def test_portenta_uniform_requires_target_core(self) -> None:
        # Uniform Portenta sketch selection needs an explicit target core so the server does not guess between CM4 and CM7.
        """Validate Portenta uniform requires target core."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            candidate_dir = Path(tmpdir) / "odom_tcn"
            server = self._build_server(
                sketches,
                candidate_dir,
                energy_aware=True,
                input_mode="uniform",
                device_name="PORTENTA_H7",
                target_core=None,
            )

            with self.assertRaises(ValueError):
                server._sync_sketch_variant()

    def test_energy_aware_false_uses_no_energy_sketch(self) -> None:
        """No-energy uniform staging should use the generic no-energy sketch."""
        # Turning off energy awareness should switch to the no-energy sketch variant to avoid unnecessary harness instrumentation.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            candidate_dir = Path(tmpdir) / "odom_tcn"
            self._write_sketch(sketches / "crest_inference_no_energy.ino", "no_energy_shared")
            server = self._build_server(sketches, candidate_dir, energy_aware=False, input_mode="uniform")

            out_path = server._sync_sketch_variant()

            self.assertTrue(out_path.exists())
            self.assertIn("no_energy_shared", out_path.read_text())

    def test_set_input_mode_updates_config_and_path(self) -> None:
        """Input-mode updates should stage the renamed generic energy sketch."""
        # Input-mode switching should update both the active config and the selected sketch path together.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            candidate_dir = Path(tmpdir) / "odom_tcn"
            self._write_sketch(sketches / "crest_inference_energy.ino", "uniform_shared")
            server = self._build_server(sketches, candidate_dir, energy_aware=True, input_mode="uniform")

            out_path = server.set_input_mode("uniform")

            self.assertEqual(server.config.training.input_mode, "uniform")
            self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()


class EvidenceFailureTests(HILServerTestCase):
    def test_request_archive_failure_preserves_candidate_exception_and_cleanup(self):
        """Archive failure must preserve the candidate exception and release request resources."""
        self.config.device.name = 'STM32_NUCLEO_N657X0_Q'
        server = self.build_server()
        device = MagicMock()
        device.requires_candidate_model.return_value = False
        device.prepare_candidate.return_value = Path('unused')
        with patch('hil_server.get_microcontroller_device', return_value=device), \
             patch('hil_server.resolve_device_options', return_value={}), \
             patch('hil_server.collect_metrics', side_effect=ValueError('original candidate error')), \
             patch('hil_server.Path.open', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(ValueError, 'original candidate error'):
                server.determine_metrics(self.request_family_hparams(), self.request_runtime_metadata())
        device.cleanup_prepared_candidate.assert_called_once_with(Path('unused'))
