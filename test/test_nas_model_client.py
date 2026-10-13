# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for the legacy NASModelClient orchestration surface.

This module covers bootstrap helpers, objective/pruning branches, smoke-test
behavior, final-training utilities, evaluation and trajectory reporting, and
resource-cleanup paths while replacing heavy production dependencies with test
doubles.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

import numpy as np
import optuna
import zmq
from addict import Dict
from optuna.trial import TrialState

# Ensure `src` is importable when the suite is launched via `python -m unittest`.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# The NASModelClient is the unit under test for this module.
from nas_model_client import (NASModelClient, MeasurementInfrastructureError,
                              MeasurementInfrastructureStop, _ConsecutiveMeasurementFailureGuard)  # noqa: E402
from crest.hardware import (
    HIL_MASTER_DEVICE_NOT_FOUND,
    HIL_MASTER_FLASH_OVERFLOW,
    HIL_MASTER_RAM_OVERFLOW,
    HIL_MASTER_FATAL,
    HIL_MASTER_INFRASTRUCTURE,
    HIL_MASTER_SUCCESS,
    TFLiteSubprocessError,
)  # noqa: E402
from crest.model import ScoreConfigEvaluationError, TrialOutcome  # noqa: E402
from crest.model_metrics import StaticMemoryEstimate  # noqa: E402
from crest.pipeline_types import (
    DataSplit,
    DatasetBundle,
    EvaluationResult,
    FitPlan,
    ModelBuildContext,
    TargetSpec,
    TaskMetricContract,
)  # noqa: E402


def _build_test_client(base_dir: Path | None = None) -> NASModelClient:
    """Construct a NASModelClient using lightweight stand-ins for config/data.

    The real class performs heavy dataset loading inside __init__. To keep the
    tests fast and deterministic, we bypass __init__ and fill in the handful of
    attributes the logic depends on.

    Notes
    -----
    This fixture preserves the production contracts around dataset bundles,
    target specs, metric contracts, model-family hooks, and output-path shape.
    It intentionally skips real dataset loading, real ZMQ sockets, and real
    filesystem/network side effects beyond temporary artifact directories.

    Parameters
    ----------
    base_dir : Path | None
        Base directory used to assemble temporary test files.

    Returns
    -------
    NASModelClient
        Constructed test client.
    """
    client = NASModelClient.__new__(NASModelClient)
    base_dir = Path(tempfile.mkdtemp()) if base_dir is None else Path(base_dir)

    window_size = 16
    input_dim = 3
    # Minimal dataset/task/model contract: enough structure for build, fit,
    # evaluate, and trajectory helpers to behave like production code.
    legacy_dataset = SimpleNamespace(
        inputs=np.zeros((2, window_size, input_dim), dtype=np.float32),
        x_vel=np.zeros((2, 1), dtype=np.float32),
        y_vel=np.zeros((2, 1), dtype=np.float32),
        size_of_each=[2],
        x0=[0.0],
        y0=[0.0],
    )
    bundle = DatasetBundle(
        train=DataSplit(
            inputs=legacy_dataset.inputs,
            targets={"velx": legacy_dataset.x_vel, "vely": legacy_dataset.y_vel},
            metadata={"size_of_each": legacy_dataset.size_of_each, "x0": legacy_dataset.x0, "y0": legacy_dataset.y0},
        ),
        val=DataSplit(
            inputs=legacy_dataset.inputs,
            targets={"velx": legacy_dataset.x_vel, "vely": legacy_dataset.y_vel},
            metadata={"size_of_each": legacy_dataset.size_of_each, "x0": legacy_dataset.x0, "y0": legacy_dataset.y0},
        ),
        test=DataSplit(
            inputs=legacy_dataset.inputs,
            targets={"velx": legacy_dataset.x_vel, "vely": legacy_dataset.y_vel},
            metadata={"size_of_each": legacy_dataset.size_of_each, "x0": legacy_dataset.x0, "y0": legacy_dataset.y0},
        ),
        input_shape=(window_size, input_dim),
        input_dtype="float32",
        metadata={"input_dim": input_dim},
    )
    target_spec = TargetSpec(
        task_type="regression",
        output_names=["velx", "vely"],
        output_shapes=[(1,), (1,)],
        metadata={},
    )
    metric_contract = TaskMetricContract(
        available_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
        training_only_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
        nonnegative_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
        primary_metric_names={"rmse_total"},
    )
    fake_built_model = MagicMock()
    fake_loaded_model = MagicMock()
    task = SimpleNamespace(
        validate_config=MagicMock(),
        validate_model_outputs=MagicMock(),
        compile_model=MagicMock(),
        build_fit_plan=MagicMock(
            return_value=FitPlan(
                fit_kwargs={
                    "x": bundle.train.inputs,
                    "y": [bundle.train.targets["velx"], bundle.train.targets["vely"]],
                    "validation_data": (
                        bundle.val.inputs,
                        [bundle.val.targets["velx"], bundle.val.targets["vely"]],
                    ),
                    "shuffle": True,
                },
                callbacks=["callback"],
                monitor_metric="val_loss",
            )
        ),
        history_component_keys=MagicMock(
            return_value=[("velx_loss", "val_velx_loss"), ("vely_loss", "val_vely_loss")]
        ),
        generate_closeout_artifacts=MagicMock(return_value={}),
        evaluate=MagicMock(
            return_value=EvaluationResult(
                metrics={"rmse_vel_x": 0.1, "rmse_vel_y": 0.2, "rmse_total": 0.3},
                predictions=[legacy_dataset.x_vel, legacy_dataset.y_vel],
            )
        ),
        evaluate_predictions=MagicMock(
            return_value=EvaluationResult(
                metrics={"rmse_vel_x": 0.1, "rmse_vel_y": 0.2, "rmse_total": 0.3},
                predictions=[legacy_dataset.x_vel, legacy_dataset.y_vel],
            )
        ),
    )
    model_family = SimpleNamespace(
        validate_config=MagicMock(),
        sample_hparams=MagicMock(
            return_value={
                "nb_filters": 2,
                "kernel_size": 2,
                "dropout_rate": 0.1,
                "use_skip_connections": True,
                "norm_flag": True,
                "dilations": [1, 2],
            }
        ),
        validate_hparams=MagicMock(),
        build_model=MagicMock(return_value=fake_built_model),
        load_model=MagicMock(return_value=fake_loaded_model),
        count_flops=MagicMock(return_value=1234),
        estimate_static_memory=MagicMock(
            return_value=StaticMemoryEstimate(
                weight_bytes=12,
                activation_bytes=34,
                memory_traffic_bytes=46,
                dtype_bytes=4,
                warning_count=0,
            )
        ),
        supports_tflite=MagicMock(return_value=True),
        default_seed_trial=MagicMock(
            return_value={
                "nb_filters": 10,
                "kernel_size": 12,
                "dropout_rate": 0.0,
                "use_skip_connections": False,
                "norm_flag": True,
                "dilations_index": 107,
            }
        ),
        decode_trial_hparams=MagicMock(
            side_effect=lambda raw_params, _ctx, _config: {
                **{key: value for key, value in raw_params.items() if key != "dilations_index"},
                "dilations": [1, 2] if "dilations_index" in raw_params else raw_params.get("dilations", [1, 2]),
            }
        ),
    )

    # Artifact, socket, and config defaults: mirror the production config shape
    # without paying the cost of real I/O-heavy initialization.
    client.config = SimpleNamespace(
        network=SimpleNamespace(host="localhost", port=5555, recv_timeout_sec=5, send_timeout_sec=5),
        training=SimpleNamespace(
            drop_rate_choices=[0.1, 0.2],
            train=True,
            nas_epochs=10,
            model_epochs=20,
            quantization=Dict(mode="float", search=False, choices=["float"]),
            latency_proxy_max_flops=1_000_000,
            nas_trials=2,
            max_total_trials=4,
            energy_aware=False,
            nas_multiobjective_population_size=8,
        ),
        device=SimpleNamespace(
            name="TEST_DEVICE",
            hil=True,
            compile_when_hil_disabled="auto",
            serial_port="ttyACM0",
        ),
        nas=SimpleNamespace(
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
        outputs=SimpleNamespace(
            log_file_name="test_log.csv",
            tflite_model_path=base_dir / "model.tflite",
            candidate_dir=base_dir / "odom_tcn",
            models_dir=base_dir / "models",
            checkpoint_path=base_dir / "model.keras",
        ),
        dataset=SimpleNamespace(
            name="oxiod",
            params=Dict(
                directory="data",
                sampling_rate_hz=100,
                window_size=window_size,
                stride=1,
            ),
        ),
        task=SimpleNamespace(name="odometry_regression", params=Dict()),
        model=SimpleNamespace(family="odom_tcn", params=Dict(export_variant="approx_trained"), search=Dict()),
    )
    client.config.outputs.models_dir.mkdir(parents=True, exist_ok=True)
    client.config_path = base_dir / "config.yaml"
    client.dataset = SimpleNamespace(validate_config=MagicMock())
    client.dataset_name = "oxiod"
    client.task = task
    client.task_name = "odometry_regression"
    client.model_family = model_family
    client.model_family_name = "odom_tcn"
    client.dataset_bundle = bundle
    client.target_spec = target_spec
    client.metric_contract = metric_contract
    client.model_build_context = ModelBuildContext(
        input_shape=bundle.input_shape,
        input_dtype=bundle.input_dtype,
        target_spec=target_spec,
        dataset_metadata=dict(bundle.metadata),
        task_metadata=dict(target_spec.metadata),
    )
    client.dataset_config = client.config.dataset.params
    client.task_config = Dict()
    client.model_config = Dict(family="odom_tcn", params=Dict(export_variant="approx_trained"), search=Dict())
    # Mirror the production default study name so log_trial calls succeed.
    client.study_name = "default_study"
    # Stub the ZMQ context/socket to avoid opening real network resources.
    client.socket = MagicMock()
    client.context = MagicMock()
    client._evaluate_model_with_backend = MagicMock(
        return_value=EvaluationResult(
            metrics={"rmse_vel_x": 0.1, "rmse_vel_y": 0.2, "rmse_total": 0.3},
            predictions=[legacy_dataset.x_vel, legacy_dataset.y_vel],
        )
    )
    return client


class DummyTrial:
    """Trivial Optuna Trial substitute that records suggestions and reports."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.report_calls = []
        self.params = {}
        self.user_attrs = {}

    def suggest_int(self, name, low, high):
        """Run suggest int.

        Parameters
        ----------
        name : object
            Parameter name requested by the fake sampler.
        low : object
            Lower bound used by the fake sampler.
        high : object
            Upper bound used by the fake sampler.

        Returns
        -------
        object
            Selected integer value from the fake sampler.
        """
        value = low
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        """Run suggest categorical.

        Parameters
        ----------
        name : object
            Parameter name requested by the fake sampler.
        choices : object
            Candidate values available to the fake sampler.

        Returns
        -------
        object
            Selected categorical value from the fake sampler.
        """
        value = choices[0]
        self.params[name] = value
        return value

    def report(self, value, step):
        """Record one Optuna-style intermediate report call.

        Parameters
        ----------
        value : object
            Value recorded by the test double.
        step : object
            Step index recorded by the fake trial.
        """
        self.report_calls.append((value, step))

    def set_user_attr(self, key, value):
        """Record one Optuna-style user attribute update.

        Parameters
        ----------
        key : object
            Dictionary key recorded by the test double.
        value : object
            Value recorded by the test double.
        """
        self.user_attrs[key] = value


class HILRequestTests(unittest.TestCase):
    """Validate the ZeroMQ request/response helper."""

    def test_hil_request_success(self) -> None:
        """A successful round-trip should log debug I/O and return metrics.

        Returns
        -------
        None
            The test passes when the mock socket is used once and request/
            response diagnostics are emitted as debug log records.
        """
        # A successful HIL round trip should deserialize cleanly into the NAS client's metrics payload.
        client = _build_test_client()
        metrics = {"ram_bytes": 1024}
        client.socket.recv_json.return_value = metrics

        payload = {"family_hparams": {"nb_filters": 4}, "runtime_metadata": {"flops": 1, "timesteps": 16, "input_dim": 3}}
        with self.assertLogs("nas_model_client", level="DEBUG") as captured:
            result = client._hil_request(payload)

        client.socket.send_json.assert_called_once_with(payload)
        client.socket.recv_json.assert_called_once()
        self.assertEqual(result, metrics)
        joined_logs = "\n".join(captured.output)
        self.assertIn("[REQ] Sending payload", joined_logs)
        self.assertIn("[REQ] Received metrics", joined_logs)

    def test_hil_request_timeout_raises(self) -> None:
        """Timeouts surfaced by pyzmq should be logged and wrapped.

        Returns
        -------
        None
            The test passes when a socket timeout logs a warning before the
            production RuntimeError is raised.
        """
        # Socket timeouts should surface as errors so stalled HIL servers do not look like valid trial failures.
        client = _build_test_client()
        client.socket.recv_json.side_effect = zmq.error.Again()

        with self.assertLogs("nas_model_client", level="WARNING") as captured:
            with self.assertRaises(RuntimeError):
                client._hil_request({"family_hparams": {"nb_filters": 8}, "runtime_metadata": {"flops": 1, "timesteps": 16, "input_dim": 3}})

        client.socket.send_json.assert_called_once()
        self.assertIn("Timed out waiting for metrics", "\n".join(captured.output))

    def test_hil_request_opens_socket_lazily(self) -> None:
        """The HIL socket should connect only when a HIL request is sent."""
        client = _build_test_client()
        fake_socket = MagicMock()
        fake_socket.recv_json.return_value = {"ram_bytes": 2048}
        fake_context = MagicMock()
        fake_context.socket.return_value = fake_socket
        client.socket = None
        client.context = None

        with patch("nas_model_client.zmq.Context.instance", return_value=fake_context):
            result = client._hil_request(
                {"family_hparams": {"nb_filters": 4}, "runtime_metadata": {"flops": 1}}
            )

        fake_context.socket.assert_called_once_with(zmq.REQ)
        fake_socket.connect.assert_called_once_with("tcp://localhost:5555")
        fake_socket.send_json.assert_called_once()
        self.assertEqual(result, {"ram_bytes": 2048})


class InitializationTests(unittest.TestCase):
    """Validate the Phase 4 bootstrap and component-selection helpers."""

    def test_resolve_component_selection_requires_explicit_component_blocks(self) -> None:
        # The breaking contract rejects configs that omit dataset/task/model blocks instead of inventing defaults.
        """Validate resolve component selection requires explicit component blocks."""
        client = _build_test_client()
        delattr(client.config, "dataset")

        with self.assertRaisesRegex(KeyError, "dataset"):
            client._resolve_component_selection(client.config)

    def test_resolve_component_selection_requires_dataset_params(self) -> None:
        # The breaking contract requires dataset params instead of falling back to the legacy top-level data block.
        """Validate resolve component selection requires dataset params."""
        client = _build_test_client()
        client.config.dataset = SimpleNamespace(name="custom_dataset")

        with self.assertRaisesRegex(KeyError, "dataset.params"):
            client._resolve_component_selection(client.config)

    def test_resolve_component_selection_honors_explicit_component_blocks(self) -> None:
        # Explicit component blocks should be the only supported path and should come back as native mappings.
        """Validate resolve component selection honors explicit component blocks."""
        client = _build_test_client()
        client.config.dataset = SimpleNamespace(name="custom_dataset", params=Dict(root="custom"))
        client.config.task = SimpleNamespace(name="custom_task", params=Dict(alpha=3))
        client.config.model = SimpleNamespace(
            family="custom_family",
            params=Dict(width=8, export_variant="untrained"),
            search=Dict(depth=[2, 3]),
        )

        selection = client._resolve_component_selection(client.config)

        self.assertEqual(selection["dataset_name"], "custom_dataset")
        self.assertEqual(selection["dataset_config"].root, "custom")
        self.assertEqual(selection["task_name"], "custom_task")
        self.assertEqual(selection["task_config"].alpha, 3)
        self.assertEqual(selection["model_family_name"], "custom_family")
        self.assertEqual(selection["model_config"]["family"], "custom_family")
        self.assertEqual(selection["model_config"]["params"].width, 8)
        self.assertEqual(selection["model_config"]["params"].export_variant, "untrained")
        self.assertEqual(selection["model_config"]["search"].depth, [2, 3])

    def test_resolve_component_selection_accepts_null_optional_task_and_search_blocks(self) -> None:
        # Optional task params and model search blocks should treat explicit null the same as omission.
        """Validate resolve component selection accepts null optional task and search blocks."""
        client = _build_test_client()
        client.config.dataset = SimpleNamespace(name="custom_dataset", params=Dict(root="custom"))
        client.config.task = SimpleNamespace(name="custom_task", params=None)
        client.config.model = SimpleNamespace(
            family="custom_family",
            params=Dict(export_variant="untrained"),
            search=None,
        )

        selection = client._resolve_component_selection(client.config)

        self.assertEqual(selection["task_config"], Dict())
        self.assertEqual(selection["model_config"]["params"].export_variant, "untrained")
        self.assertEqual(selection["model_config"]["search"], Dict())

    def test_resolve_component_selection_requires_export_variant(self) -> None:
        """Model params must provide the universal export variant field."""
        client = _build_test_client()
        client.config.model = SimpleNamespace(family="custom_family", params=Dict(), search=Dict())

        with self.assertRaisesRegex(KeyError, "model.params.export_variant"):
            client._resolve_component_selection(client.config)

    def test_resolve_component_selection_rejects_blank_export_variant(self) -> None:
        """Blank export variants should fail before component bootstrap."""
        client = _build_test_client()
        client.config.model = SimpleNamespace(
            family="custom_family",
            params=Dict(export_variant=" "),
            search=Dict(),
        )

        with self.assertRaisesRegex(ValueError, "model.params.export_variant"):
            client._resolve_component_selection(client.config)

    def test_init_reuses_preliminary_bundle_when_dataset_selection_matches(self) -> None:
        # Initialization should reuse a matching preliminary bundle so repeated setup does not reload the same dataset.
        """Validate init reuses preliminary bundle when dataset selection matches."""
        base = Path(tempfile.mkdtemp())
        config = SimpleNamespace(
            network=SimpleNamespace(host="localhost", port=5555, recv_timeout_sec=5, send_timeout_sec=5),
            device=SimpleNamespace(hil=True, name="TEST_DEVICE"),
            nas=SimpleNamespace(score=Dict(type="scoring-function", metrics=Dict(), params=Dict(terms=[]))),
            outputs=SimpleNamespace(
                models_dir=base / "models",
                checkpoint_path=base / "model.keras",
                log_file_name="log.csv",
            ),
            dataset=SimpleNamespace(
                name="oxiod",
                params=Dict(directory="data", sampling_rate_hz=100, window_size=16, stride=1),
            ),
            task=SimpleNamespace(name="odometry_regression", params=Dict()),
            model=SimpleNamespace(family="odom_tcn", params=Dict(export_variant="approx_trained"), search=Dict()),
        )
        fake_bundle = DatasetBundle(
            train=DataSplit(inputs=np.zeros((1, 16, 3)), targets={"velx": np.zeros((1, 1)), "vely": np.zeros((1, 1))}),
            val=DataSplit(inputs=np.zeros((1, 16, 3)), targets={"velx": np.zeros((1, 1)), "vely": np.zeros((1, 1))}),
            test=DataSplit(inputs=np.zeros((1, 16, 3)), targets={"velx": np.zeros((1, 1)), "vely": np.zeros((1, 1))}),
            input_shape=(16, 3),
            input_dtype="float32",
        )
        target_spec = TargetSpec(
            task_type="regression",
            output_names=["velx", "vely"],
            output_shapes=[(1,), (1,)],
            metadata={},
        )
        metric_contract = TaskMetricContract(
            available_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
            training_only_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
            nonnegative_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
            primary_metric_names={"rmse_total"},
        )
        fake_task = SimpleNamespace(
            build_target_spec=MagicMock(return_value=target_spec),
            metric_contract=MagicMock(return_value=metric_contract),
        )
        fake_socket = MagicMock()
        fake_context = MagicMock()
        fake_context.socket.return_value = fake_socket

        fake_pipeline = SimpleNamespace(
            selection={
                "dataset_name": "oxiod",
                "task_name": "odometry_regression",
                "model_family_name": "odom_tcn",
                "dataset_config": config.dataset.params,
                "task_config": config.task.params,
                "model_config": {"family": "odom_tcn", "params": {}, "search": {}},
            },
            dataset=sentinel.dataset,
            task=fake_task,
            model_family=sentinel.model_family,
            bundle=fake_bundle,
            target_spec=target_spec,
            metric_contract=metric_contract,
            model_build_context=ModelBuildContext(
                input_shape=fake_bundle.input_shape,
                input_dtype=fake_bundle.input_dtype,
                target_spec=target_spec,
            ),
        )

        with patch("nas_model_client.ensure_builtin_components_registered"), patch(
            "nas_model_client.load_config",
            return_value=config,
        ) as mock_load_config, patch(
            "nas_model_client.bootstrap_pipeline",
            return_value=fake_pipeline,
        ) as bootstrap_mock, patch(
            "nas_model_client.zmq.Context.instance",
            return_value=fake_context,
        ):
            client = NASModelClient(base / "config.yaml")

        self.assertIsInstance(client, NASModelClient)
        mock_load_config.assert_called_once_with(base / "config.yaml")
        bootstrap_mock.assert_called_once_with(config)
        fake_context.socket.assert_not_called()
        fake_socket.connect.assert_not_called()

class ObjectiveTests(unittest.TestCase):
    """Exercise Optuna objective branches with lightweight stubs."""

    def setUp(self) -> None:
        """Prepare test fixtures."""
        self.client = _build_test_client()

        self.hw_specs_patcher = patch("nas_model_client.return_hardware_specs", return_value=(2048, 4096))
        self.log_patcher = patch("nas_model_client.log_trial")

        self.mock_hw_specs = self.hw_specs_patcher.start()
        self.mock_log = self.log_patcher.start()

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        patch.stopall()

    def test_objective_classifies_exhausted_measurement_recovery(self) -> None:
        """Exhausted recovery must raise an infrastructure failure before training."""
        self.client._hil_request = MagicMock(return_value={"error_code": HIL_MASTER_INFRASTRUCTURE})
        with self.assertRaises(MeasurementInfrastructureError):
            self.client.objective(DummyTrial())
        logged = self.mock_log.call_args.kwargs
        self.assertFalse(logged["pruned"])
        self.assertEqual(logged["metrics"]["measurement_failure_class"], "infrastructure")
        self.assertEqual(logged["prune_rule"], "measurement_integrity")
        self.client.task.build_fit_plan.assert_not_called()

    def test_multiobjective_infrastructure_failure_has_no_penalty_return(self) -> None:
        """Multiobjective infrastructure failures must raise instead of returning penalties."""
        self.client.config.nas.score = Dict(
            type="multi-objective", metrics=Dict(), params=Dict(objectives=[
                Dict(metric="rmse_total", direction="minimize"),
                Dict(metric="latency_ms", direction="minimize")]))
        self.client._hil_request = MagicMock(return_value={"error_code": HIL_MASTER_INFRASTRUCTURE})
        with self.assertRaises(MeasurementInfrastructureError):
            self.client.objective(DummyTrial())
        self.assertFalse(self.mock_log.call_args.kwargs["pruned"])
        self.assertEqual(self.mock_log.call_args.kwargs["prune_rule"], "measurement_integrity")
        self.client.task.build_fit_plan.assert_not_called()

    def test_infrastructure_csv_has_null_objectives_and_explicit_failure(self):
        """Infrastructure CSV rows must identify FAIL and leave objective values null."""
        import csv
        import json
        from crest.model import log_trial as real_log_trial
        self.client.config.nas.score = Dict(
            type="multi-objective", metrics=Dict(), params=Dict(objectives=[
                Dict(metric="rmse_total", direction="minimize"),
                Dict(metric="latency_ms", direction="minimize")]))
        self.client._hil_request = MagicMock(return_value={"error_code": HIL_MASTER_INFRASTRUCTURE, "ram_bytes": 1024, "flash_bytes": 2048})
        with self.assertRaises(MeasurementInfrastructureError):
            self.client.objective(DummyTrial())
        kwargs = self.mock_log.call_args.kwargs.copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.csv"
            kwargs["log_file_name"] = str(path)
            real_log_trial(**kwargs)
            with path.open() as stream:
                row = next(csv.DictReader(stream))
        self.assertEqual(row["trial_outcome"], "FAIL_infrastructure")
        self.assertEqual(json.loads(row["objective_values_json"]), [None, None])
        self.assertEqual(row["score"], "")
        self.assertEqual(row["metric__measurement_failure_class"], "infrastructure")

    def test_objective_prunes_on_flash_overflow(self) -> None:
        """Flash overflow errors should prune the Optuna trial."""
        # Flash-overflow trials should prune immediately so obviously too-large candidates do not enter training.
        self.client._hil_request = MagicMock(return_value={"error_code": HIL_MASTER_FLASH_OVERFLOW})
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned):
            self.client.objective(trial)

        self.mock_log.assert_called_once()
        self.assertEqual(trial.report_calls, [(-float("inf"), 0)])
        self.client.task.build_fit_plan.assert_not_called()

    def test_objective_prune_logging_populates_cadenced_sentinels(self) -> None:
        """Early-pruned trials should still log stable cadenced metric defaults."""
        # Pruned trials should still log cadenced sentinel fields so CSV rows keep the same schema as successful runs.
        self.client._hil_request = MagicMock(return_value={"error_code": HIL_MASTER_FLASH_OVERFLOW})
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned):
            self.client.objective(trial)

        logged_metrics = self.mock_log.call_args.kwargs["metrics"]
        self.assertEqual(logged_metrics["runtime_mode"], "back_to_back")
        self.assertEqual(logged_metrics["cadenced_energy_mj_per_trial"], -1.0)
        self.assertEqual(logged_metrics["cadenced_error_code"], -1)
        self.assertIsNone(logged_metrics["cadenced_error_label"])

    def test_objective_prunes_on_ram_overflow(self) -> None:
        """RAM overflow errors should prune the trial to skip training."""
        # RAM-overflow trials should prune immediately so impossible memory footprints stop before fit().
        self.client._hil_request = MagicMock(return_value={"error_code": HIL_MASTER_RAM_OVERFLOW})
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned):
            self.client.objective(trial)

        self.mock_log.assert_called_once()
        self.assertEqual(trial.report_calls, [(-float("inf"), 0)])
        self.client.task.build_fit_plan.assert_not_called()

    def test_objective_raises_on_device_not_found(self) -> None:
        """Device-not-found errors should abort the NAS run instead of pruning every trial."""
        # Unknown devices should raise early so hardware catalog mistakes do not look like training failures.
        self.client._hil_request = MagicMock(return_value={"error_code": HIL_MASTER_DEVICE_NOT_FOUND})
        trial = DummyTrial()

        with self.assertRaises(RuntimeError):
            self.client.objective(trial)

    def test_objective_rejects_invalid_model_build_context_input_shapes(self) -> None:
        # Invalid model-build input shapes should fail before the NAS client tries to train or evaluate a broken graph.
        """Validate objective rejects invalid model build context input shapes."""
        self.client._hil_request = MagicMock()

        for invalid_shape in (None, (None, 3), ("abc", 3), (0, 3), (16, False)):
            with self.subTest(input_shape=invalid_shape):
                self.client.model_build_context.input_shape = invalid_shape
                with self.assertRaisesRegex(ValueError, "2D logical input shape"):
                    self.client.objective(DummyTrial())

        self.client._hil_request.assert_not_called()

        self.mock_log.assert_not_called()
        self.client.task.build_fit_plan.assert_not_called()

    def test_objective_handles_resource_failure(self) -> None:
        """Exceeding estimated resources should skip training and log a fatal code."""
        # Low-level resource failures should prune the trial with a stable penalty instead of leaking raw backend exceptions.
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            # Force RAM usage above the limit returned by return_hardware_specs.
            "ram_bytes": 8192,
            "flash_bytes": 1024,
            "arena_bytes": 2048,
        }
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned):
            self.client.objective(trial)

        self.client.task.build_fit_plan.assert_not_called()
        self.assertEqual(trial.report_calls, [(-float("inf"), 0)])


    def test_objective_happy_path_runs_training(self) -> None:
        """Valid metrics should flow into training and return the reported score."""
        # A healthy trial should still execute the full fit and task-evaluation path before scoring.
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
        }
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        result = self.client.objective(trial)

        self.client.model_family.count_flops.assert_called_once_with(
            self.client.model_family.build_model.return_value,
            self.client.model_build_context,
            self.client.model_config,
        )
        self.client.model_family.estimate_static_memory.assert_called_once_with(
            self.client.model_family.build_model.return_value,
            self.client.model_build_context,
            self.client.model_config,
            quantization_mode="float",
        )
        logged_hparams = self.mock_log.call_args.kwargs["trial_outcome"].hyperparams
        self.assertEqual(logged_hparams["memory_traffic_bytes"], 46)
        self.client.task.build_fit_plan.assert_called_once_with(
            self.client.dataset_bundle,
            self.client.task_config,
            self.client.target_spec,
            mode="search",
            combine_train_val=False,
        )
        self.assertEqual(self.client._evaluate_model_with_backend.call_count, 2)
        self.assertEqual(
            [call.kwargs["evaluation_backend"] for call in self.client._evaluate_model_with_backend.call_args_list],
            ["tflite", "keras"],
        )
        self.assertEqual(result, -0.3)
        self.mock_log.assert_called_once()
        logged_outcome = self.mock_log.call_args.kwargs["trial_outcome"]
        self.assertEqual(logged_outcome.task_metrics["rmse_total"], 0.3)
        self.assertEqual(logged_outcome.task_metrics["keras_rmse_total"], 0.3)

    def test_objective_multiobjective_tflite_failure_returns_penalty_tuple(self) -> None:
        """Multi-objective TFLite worker failures should return direction penalties."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
        }
        self.client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(
                objectives=[
                    Dict(metric="rmse_total", direction="minimize"),
                    Dict(metric="rmse_vel_x", direction="minimize"),
                ]
            ),
        )
        self.client._hil_request = MagicMock(return_value=metrics)
        self.client._evaluate_model_with_backend = MagicMock(
            side_effect=TFLiteSubprocessError(
                model_path="model.tflite",
                return_code=-6,
                timeout=False,
                stderr_tail="abort",
                command=["python", "-m", "crest.tflite_predict_worker"],
            )
        )
        trial = DummyTrial()

        result = self.client.objective(trial)

        self.assertEqual(result, (1e12, 1e12))
        self.mock_log.assert_called_once()
        self.assertEqual(self.mock_log.call_args.kwargs["prune_reason"], "TFLite evaluation failed")
        self.client.task.evaluate.assert_not_called()

    def test_objective_single_objective_tflite_failure_prunes(self) -> None:
        """Single-objective TFLite worker failures should raise TrialPruned."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
        }
        self.client._hil_request = MagicMock(return_value=metrics)
        self.client._evaluate_model_with_backend = MagicMock(
            side_effect=TFLiteSubprocessError(
                model_path="model.tflite",
                return_code=1,
                timeout=False,
                stderr_tail="worker failed",
                command=["python", "-m", "crest.tflite_predict_worker"],
            )
        )
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned) as raised:
            self.client.objective(trial)

        self.mock_log.assert_called_once()
        prune_reason = self.mock_log.call_args.kwargs["prune_reason"]
        self.assertIn("TFLite evaluation failed", prune_reason)
        self.assertIn("exited with code 1", prune_reason)
        self.assertIn("worker failed", prune_reason)
        self.assertIn("worker failed", str(raised.exception))
        self.assertEqual(trial.report_calls, [(-float("inf"), 0)])

    def test_objective_prunes_before_training_on_config_rule(self) -> None:
        # Config-level prune rules should short-circuit the trial before fit() so obviously bad candidates do not waste training time.
        """Validate objective prunes before training on config rule."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 25.0,
            "latency_budget_ms": 20.0,
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
        }
        self.client.config.nas.prune = Dict(
            rules=[
                Dict(
                    rule="latency_budget",
                    metric="latency_ms",
                    condition="gt",
                    reference=Dict(type="metric", metric="latency_budget_ms"),
                    reason="Latency exceeds deployment budget",
                )
            ]
        )
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned):
            self.client.objective(trial)

        self.client.task.build_fit_plan.assert_not_called()
        self.mock_log.assert_called_once()
        self.assertEqual(self.mock_log.call_args.kwargs["prune_rule"], "latency_budget")
        self.assertEqual(
            self.mock_log.call_args.kwargs["prune_reason"],
            "Latency exceeds deployment budget",
        )

    def test_objective_prunes_when_rule_metric_is_unavailable(self) -> None:
        # Unavailable prune metrics should fail closed and prune the trial instead of letting half-populated hardware results continue.
        """Validate objective prunes when rule metric is unavailable."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": -1.0,
            "latency_budget_ms": 20.0,
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
        }
        self.client.config.nas.prune = Dict(
            rules=[
                Dict(
                    metric="latency_ms",
                    condition="gt",
                    reference=Dict(type="metric", metric="latency_budget_ms"),
                    reason="Latency exceeds deployment budget",
                    rule="rule_0",
                )
            ]
        )
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned):
            self.client.objective(trial)

        self.client.task.build_fit_plan.assert_not_called()
        self.assertEqual(self.mock_log.call_args.kwargs["prune_rule"], "rule_0")
        self.assertIn("Configured prune metric unavailable", self.mock_log.call_args.kwargs["prune_reason"])

    def test_objective_infeasible_skips_training_and_logs_constraints(self) -> None:
        """Feasibility violations should complete with penalties when training is disabled."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 25.0,
            "latency_budget_ms": 20.0,
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
        }
        self.client.config.nas.feasibility = Dict(
            train_if_infeasible=False,
            rules=[
                Dict(
                    rule="latency_budget",
                    metric="latency_ms",
                    condition="gt",
                    reference=Dict(type="metric", metric="latency_budget_ms"),
                    reason="Latency exceeds cadence budget",
                )
            ],
        )
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        result = self.client.objective(trial)

        self.assertEqual(result, -100.0)
        self.client.task.build_fit_plan.assert_not_called()
        logged_metrics = self.mock_log.call_args.kwargs["metrics"]
        self.assertFalse(logged_metrics["feasible"])
        self.assertEqual(logged_metrics["feasibility_status"], "infeasible")
        self.assertEqual(logged_metrics["feasibility_rule"], "latency_budget")
        self.assertEqual(logged_metrics["feasibility_constraints"], [5.0])
        self.assertEqual(json.loads(logged_metrics["feasibility_constraints_json"]), [5.0])
        self.assertFalse(self.mock_log.call_args.kwargs.get("pruned", False))

    def test_objective_train_if_infeasible_trains_with_real_objectives(self) -> None:
        """Infeasible trials may still train while remaining constrained."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 25.0,
            "latency_budget_ms": 20.0,
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
        }
        self.client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(
                objectives=[
                    Dict(metric="latency_ms", direction="minimize"),
                    Dict(metric="rmse_total", direction="minimize"),
                ]
            ),
        )
        self.client.config.nas.feasibility = Dict(
            train_if_infeasible=True,
            rules=[
                Dict(
                    rule="latency_budget",
                    metric="latency_ms",
                    condition="gt",
                    reference=Dict(type="metric", metric="latency_budget_ms"),
                    reason="Latency exceeds cadence budget",
                )
            ],
        )
        self.client._hil_request = MagicMock(return_value=metrics)

        result = self.client.objective(DummyTrial())

        self.assertEqual(result, (25.0, 0.3))
        self.client.task.build_fit_plan.assert_called_once()
        logged_metrics = self.mock_log.call_args.kwargs["metrics"]
        self.assertFalse(logged_metrics["feasible"])
        self.assertEqual(logged_metrics["feasibility_status"], "infeasible")
        self.assertEqual(logged_metrics["feasibility_constraints"], [5.0])

    def test_objective_uses_negative_one_rmse_sentinels_for_failed_trials(self) -> None:
        # Failed trials should log stable RMSE sentinels so CSV summaries can distinguish a failure from missing training output.
        """Validate objective uses negative one rmse sentinels for failed trials."""
        metrics = {
            "error_code": HIL_MASTER_RAM_OVERFLOW,
            "ram_bytes": -1,
            "flash_bytes": -1,
            "arena_bytes": -1,
            "latency_ms": -1.0,
            "latency_budget_ms": -1.0,
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
        }
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned):
            self.client.objective(trial)

        trial_outcome = self.mock_log.call_args.kwargs["trial_outcome"]
        self.assertIsInstance(trial_outcome, TrialOutcome)
        self.assertEqual(trial_outcome.task_metrics, {})

    def test_objective_does_not_swallow_generic_training_value_errors(self) -> None:
        # Unexpected training ValueErrors should still escape so genuine code bugs are not hidden behind prune logic.
        """Validate objective does not swallow generic training value errors."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
            "latency_budget_ms": 20.0,
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
        }
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        with patch.object(self.client.task, "build_fit_plan", side_effect=ValueError("bad shape")):
            with self.assertRaisesRegex(ValueError, "bad shape"):
                self.client.objective(trial)

        self.mock_log.assert_not_called()

    def test_objective_converts_score_config_errors_into_prune_penalties(self) -> None:
        # Score-config evaluation errors should convert into prune penalties so search continues without masking the misconfiguration.
        """Validate objective converts score config errors into prune penalties."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
            "latency_budget_ms": 20.0,
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
        }
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        with patch(
            "nas_model_client.evaluate_score_config",
            side_effect=ScoreConfigEvaluationError("Metric 'latency_ms' is unavailable for scoring."),
        ):
            with self.assertRaises(optuna.TrialPruned):
                self.client.objective(trial)

        self.mock_log.assert_called_once()
        self.assertEqual(
            self.mock_log.call_args.kwargs["prune_reason"],
            "Training failed to produce valid metrics",
        )

    def test_objective_samples_cpu_clock_into_device_overrides(self) -> None:
        # Sampled CPU-clock choices should flow into device overrides so each trial evaluates the exact hardware point it sampled.
        """Validate objective samples cpu clock into device overrides."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
            "cpu_clock_mhz_requested": 400,
        }
        self.client.config.device.cpu_clock_mhz_options = [200, 400, 600]
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        self.client.objective(trial)

        sent_payload = self.client._hil_request.call_args.args[0]
        self.assertIn("family_hparams", sent_payload)
        self.assertIn("runtime_metadata", sent_payload)
        self.assertEqual(sent_payload["quantization_mode"], "float")
        self.assertEqual(sent_payload["device_options_overrides"], {"cpu_clock_mhz": 200})
        self.assertNotIn("model_variant", sent_payload)
        self.assertNotIn("cpu_clock_mhz", sent_payload["family_hparams"])
        self.assertEqual(sent_payload["runtime_metadata"]["flops"], 1234)
        self.assertEqual(sent_payload["runtime_metadata"]["memory_traffic_bytes"], 46)
        self.assertEqual(sent_payload["runtime_metadata"]["memory_proxy_dtype_bytes"], 4)
        self.assertEqual(trial.params["cpu_clock_mhz_index"], 0)

    def test_objective_omits_device_overrides_when_clock_options_are_null(self) -> None:
        # Null clock-option lists should avoid creating spurious overrides so default board clocks stay in control.
        """Validate objective omits device overrides when clock options are null."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
            "cpu_clock_mhz_requested": -1,
        }
        self.client.config.device.cpu_clock_mhz_options = None
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        self.client.objective(trial)

        sent_payload = self.client._hil_request.call_args.args[0]
        self.assertEqual(set(sent_payload.keys()), {"family_hparams", "runtime_metadata", "quantization_mode"})

    def test_objective_samples_quantization_mode_when_search_enabled(self) -> None:
        """Quantization search should sample before HIL and keep mode top-level."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
        }
        self.client.config.training.quantization = Dict(
            mode="int8_ptq",
            search=True,
            choices=["float", "int8_ptq"],
        )
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        self.client.objective(trial)

        sent_payload = self.client._hil_request.call_args.args[0]
        self.assertEqual(trial.params["quantization_mode"], "float")
        self.assertEqual(sent_payload["quantization_mode"], "float")
        self.assertNotIn("quantization_mode", sent_payload["family_hparams"])

    def test_objective_stm_phase1_allows_arena_sentinel(self) -> None:
        """Ensure STM Phase 1 does not get pruned solely for ``arena_bytes=-1``.

        Returns
        -------
        None
        """
        # STM32 phase-one trials should tolerate arena sentinel values because that stage may not know the final arena size yet.
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": -1,
            "latency_ms": 10.0,
        }
        self.client.config.device.name = "STM32_NUCLEO_N657X0_Q"
        self.client.config.device.stm32 = SimpleNamespace(project_root="stm_fsbl")
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        result = self.client.objective(trial)

        self.client.task.build_fit_plan.assert_called_once_with(
            self.client.dataset_bundle,
            self.client.task_config,
            self.client.target_spec,
            mode="search",
            combine_train_val=False,
        )
        self.assertEqual(result, -0.3)
        self.mock_log.assert_called_once()
        logged_outcome = self.mock_log.call_args.kwargs["trial_outcome"]
        self.assertEqual(logged_outcome.task_metrics["rmse_total"], 0.3)

    def test_objective_returns_configured_multiobjective_tuple(self) -> None:
        """Configured multi-objective runs should return all objective values.

        Returns
        -------
        None
        """
        # Configured multi-objective trials should return all objective values in the order Optuna expects.
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": -1,
            "latency_ms": 10.0,
            "energy_mj_per_inference": -1.0,
            "energy_aware": False,
            "hil_enabled": False,
        }
        self.client.config.device.name = "STM32_NUCLEO_N657X0_Q"
        self.client.config.device.stm32 = SimpleNamespace(project_root="stm_fsbl")
        self.client.config.training.energy_aware = True
        self.client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(
                objectives=[
                    Dict(metric="latency_ms", direction="minimize"),
                    Dict(metric="flash_bytes", direction="minimize"),
                ]
            ),
        )
        self.client.config.training.train = False
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        result = self.client.objective(trial)

        self.assertEqual(result, (10.0, 512.0))
        self.mock_log.assert_called_once()

    def test_objective_multiobjective_rule_hit_returns_penalty_tuple(self) -> None:
        """Multi-objective prune-rule hits should log and return signed penalties.

        Returns
        -------
        None
            Asserts that post-HIL gates skip fit/evaluation and return
            direction-aware penalties.
        """
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 25.0,
            "latency_budget_ms": 20.0,
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
        }
        self.client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(
                objectives=[
                    Dict(metric="latency_ms", direction="minimize"),
                    Dict(metric="rmse_total", direction="maximize"),
                ]
            ),
        )
        self.client.config.nas.prune = Dict(
            rules=[
                Dict(
                    rule="deadline_miss_limit",
                    metric="latency_ms",
                    condition="gt",
                    reference=Dict(type="metric", metric="latency_budget_ms"),
                    reason="Latency exceeds deployment budget",
                )
            ]
        )
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        result = self.client.objective(trial)

        self.assertEqual(result, (1e12, -1e12))
        self.client.model_family.build_model.assert_called_once()
        self.client.task.compile_model.assert_called_once()
        self.client.model_family.count_flops.assert_called_once()
        self.client._hil_request.assert_called_once()
        self.client.task.build_fit_plan.assert_not_called()
        self.client.model_family.build_model.return_value.fit.assert_not_called()
        self.client._evaluate_model_with_backend.assert_not_called()
        self.mock_log.assert_called_once()
        self.assertTrue(self.mock_log.call_args.kwargs["pruned"])
        self.assertEqual(self.mock_log.call_args.kwargs["prune_rule"], "deadline_miss_limit")
        self.assertEqual(
            self.mock_log.call_args.kwargs["prune_reason"],
            "Latency exceeds deployment budget",
        )

    def test_objective_multiobjective_rule_reference_unavailable_reason(self) -> None:
        """Unavailable prune reference metrics should be named in the reason.

        Returns
        -------
        None
            Asserts reference-metric unavailability is distinct from rule
            metric unavailability.
        """
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 25.0,
            "latency_budget_ms": -1.0,
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
        }
        self.client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(objectives=[Dict(metric="latency_ms", direction="minimize")]),
        )
        self.client.config.nas.prune = Dict(
            rules=[
                Dict(
                    rule="latency_reference",
                    metric="latency_ms",
                    condition="gt",
                    reference=Dict(type="metric", metric="latency_budget_ms"),
                    reason="Latency exceeds deployment budget",
                )
            ]
        )
        self.client._hil_request = MagicMock(return_value=metrics)

        result = self.client.objective(DummyTrial())

        self.assertEqual(result, (1e12,))
        self.assertEqual(self.mock_log.call_args.kwargs["prune_rule"], "latency_reference")
        self.assertEqual(
            self.mock_log.call_args.kwargs["prune_reason"],
            "Configured prune reference metric unavailable: latency_budget_ms",
        )

    def test_objective_multiobjective_custom_nonnegative_sentinel_fails_closed(self) -> None:
        """Task-owned nonnegative prune metrics should treat -1 as unavailable.

        Returns
        -------
        None
            Asserts task nonnegative sentinel handling is applied before fit.
        """
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
            "custom_feasibility": -1.0,
        }
        self.client.metric_contract = TaskMetricContract(
            available_metric_names={"custom_feasibility", "rmse_total"},
            training_only_metric_names={"rmse_total"},
            nonnegative_metric_names={"custom_feasibility", "rmse_total"},
            primary_metric_names={"rmse_total"},
        )
        self.client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(objectives=[Dict(metric="latency_ms", direction="minimize")]),
        )
        self.client.config.nas.prune = Dict(
            rules=[
                Dict(
                    rule="custom_prefit_gate",
                    metric="custom_feasibility",
                    condition="gt",
                    reference=Dict(type="literal", value=0.0),
                    reason="Custom feasibility failed",
                )
            ]
        )
        self.client._hil_request = MagicMock(return_value=metrics)

        result = self.client.objective(DummyTrial())

        self.assertEqual(result, (1e12,))
        self.client.task.build_fit_plan.assert_not_called()
        self.assertEqual(self.mock_log.call_args.kwargs["prune_rule"], "custom_prefit_gate")
        self.assertEqual(
            self.mock_log.call_args.kwargs["prune_reason"],
            "Configured prune metric unavailable: custom_feasibility",
        )

    def test_objective_multiobjective_rule_miss_trains_and_evaluates(self) -> None:
        """Non-matching multi-objective rules should continue through validation.

        Returns
        -------
        None
            Asserts rule misses continue into fit plus TFLite and Keras
            validation.
        """
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
            "latency_budget_ms": 20.0,
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
        }
        self.client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(
                objectives=[
                    Dict(metric="latency_ms", direction="minimize"),
                    Dict(metric="rmse_total", direction="minimize"),
                ]
            ),
        )
        self.client.config.nas.prune = Dict(
            rules=[
                Dict(
                    rule="latency_budget",
                    metric="latency_ms",
                    condition="gt",
                    reference=Dict(type="metric", metric="latency_budget_ms"),
                    reason="Latency exceeds deployment budget",
                )
            ]
        )
        self.client._hil_request = MagicMock(return_value=metrics)

        result = self.client.objective(DummyTrial())

        self.assertEqual(result, (10.0, 0.3))
        self.client.task.build_fit_plan.assert_called_once()
        self.assertEqual(self.client._evaluate_model_with_backend.call_count, 2)
        self.assertFalse(self.mock_log.call_args.kwargs.get("pruned", False))

    def test_objective_multiobjective_gated_optuna_trial_is_complete(self) -> None:
        """Optuna should record gated multi-objective trials as COMPLETE.

        Returns
        -------
        None
            Asserts a gated multi-objective objective return is not an Optuna
            PRUNED trial.
        """
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 25.0,
            "latency_budget_ms": 20.0,
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
        }
        self.client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(objectives=[Dict(metric="latency_ms", direction="minimize")]),
        )
        self.client.config.nas.prune = Dict(
            rules=[
                Dict(
                    rule="latency_budget",
                    metric="latency_ms",
                    condition="gt",
                    reference=Dict(type="metric", metric="latency_budget_ms"),
                    reason="Latency exceeds deployment budget",
                )
            ]
        )
        self.client._hil_request = MagicMock(return_value=metrics)
        study = optuna.create_study(directions=["minimize"])

        study.optimize(self.client.objective, n_trials=1)

        self.assertEqual(len(study.trials), 1)
        self.assertEqual(study.trials[0].state, TrialState.COMPLETE)
        self.assertEqual(study.trials[0].values, [1e12])

    def test_objective_train_false_uses_generic_metric_sentinels(self) -> None:
        # Train-disabled trials should still emit generic metric sentinels so downstream logs keep a stable shape.
        """Validate objective train false uses generic metric sentinels."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
        }
        self.client.metric_contract = TaskMetricContract(
            available_metric_names={"signed_bias", "signed_offset"},
            training_only_metric_names={"signed_bias", "signed_offset"},
            nonnegative_metric_names=set(),
            primary_metric_names={"signed_bias"},
        )
        self.client.config.training.train = False
        self.client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(
                objectives=[
                    Dict(metric="signed_bias", direction="maximize"),
                    Dict(metric="signed_offset", direction="maximize"),
                ]
            ),
        )
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        result = self.client.objective(trial)

        self.assertEqual(result, (-1.0, -1.0))
        logged_outcome = self.mock_log.call_args.kwargs["trial_outcome"]
        self.assertEqual(
            logged_outcome.task_metrics,
            {"signed_bias": -1.0, "signed_offset": -1.0},
        )

    def test_objective_portenta_forwards_device_options_to_hardware_specs(self) -> None:
        # Portenta trials should forward resolved board options when requesting hardware limits.
        """Validate objective Portenta forwards device options to hardware specs."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
        }
        self.client.config.device.name = "PORTENTA_H7"
        self.client.config.device.portenta = SimpleNamespace(
            target_core="cm4",
            split="50_50",
            security="none",
        )
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        self.client.objective(trial)

        self.mock_hw_specs.assert_called_once()
        _, kwargs = self.mock_hw_specs.call_args
        self.assertEqual(kwargs["device_options"]["target_core"], "cm4")
        self.assertEqual(kwargs["device_options"]["split"], "50_50")
        self.assertEqual(kwargs["device_options"]["security"], "none")

    def test_objective_portenta_requires_target_core_for_hardware_limits(self) -> None:
        # Portenta hardware-limit resolution should require an explicit target core instead of guessing one.
        """Validate objective Portenta requires target core for hardware limits."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
        }
        self.client.config.device.name = "PORTENTA_H7"
        self.client.config.device.portenta = SimpleNamespace()
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        with self.assertRaises(RuntimeError):
            self.client.objective(trial)

    def test_objective_auto_pure_desktop_skips_hil_for_rmse_flops(self) -> None:
        """Auto non-HIL mode should skip compile for local RMSE/FLOPs scoring."""
        self.client.config.device.hil = False
        self.client.config.device.compile_when_hil_disabled = "auto"
        self.client.config.device.cpu_clock_mhz_options = [200, 400]
        self.client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(
                objectives=[
                    Dict(metric="rmse_total", direction="minimize"),
                    Dict(metric="flops", direction="minimize"),
                ]
            ),
        )
        self.client._hil_request = MagicMock()
        trial = DummyTrial()

        result = self.client.objective(trial)

        self.assertEqual(result, (0.3, 1234.0))
        self.client._hil_request.assert_not_called()
        self.assertNotIn("cpu_clock_mhz_index", trial.params)
        logged_metrics = self.mock_log.call_args.kwargs["metrics"]
        self.assertFalse(logged_metrics["hil_enabled"])
        self.assertFalse(logged_metrics["energy_aware"])
        self.assertEqual(logged_metrics["error_code"], HIL_MASTER_SUCCESS)
        self.assertEqual(logged_metrics["ram_bytes"], -1)
        self.assertEqual(logged_metrics["cadenced_energy_mj_per_trial"], -1.0)

    def test_objective_false_pure_desktop_skips_hil_for_rmse_flops(self) -> None:
        """Explicit false non-HIL mode should run pure desktop scoring."""
        self.client.config.device.hil = False
        self.client.config.device.compile_when_hil_disabled = "false"
        self.client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(
                objectives=[
                    Dict(metric="rmse_total", direction="minimize"),
                    Dict(metric="flops", direction="minimize"),
                ]
            ),
        )
        self.client._hil_request = MagicMock()

        result = self.client.objective(DummyTrial())

        self.assertEqual(result, (0.3, 1234.0))
        self.client._hil_request.assert_not_called()

    def test_objective_auto_non_hil_compiles_for_flash_metric(self) -> None:
        """Auto non-HIL mode should preserve compile-only proxy behavior."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "external_flash_bytes": 64,
            "arena_bytes": 1024,
            "latency_ms": -1.0,
        }
        self.client.config.device.hil = False
        self.client.config.device.compile_when_hil_disabled = "auto"
        self.client.config.training.train = False
        self.client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(objectives=[Dict(metric="flash_bytes", direction="minimize")]),
        )
        self.client._hil_request = MagicMock(return_value=metrics)

        result = self.client.objective(DummyTrial())

        self.assertEqual(result, (512.0,))
        self.client._hil_request.assert_called_once()

    def test_objective_false_non_hil_rejects_compile_metrics(self) -> None:
        """Pure desktop mode should fail fast when compile metrics are required."""
        self.client.config.device.hil = False
        self.client.config.device.compile_when_hil_disabled = "false"
        self.client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(
                objectives=[
                    Dict(metric="flash_bytes", direction="minimize"),
                    Dict(metric="ram_bytes", direction="minimize"),
                    Dict(metric="external_flash_bytes", direction="minimize"),
                    Dict(metric="arena_bytes", direction="minimize"),
                ]
            ),
        )
        self.client._hil_request = MagicMock()

        with self.assertRaisesRegex(ValueError, "compile-derived metric"):
            self.client.objective(DummyTrial())

        self.client._hil_request.assert_not_called()
        self.client.model_family.sample_hparams.assert_not_called()

    def test_objective_non_hil_rejects_runtime_metrics_in_derived_score(self) -> None:
        """Runtime-only dependencies should fail before trial execution."""
        self.client.config.device.hil = False
        self.client.config.device.compile_when_hil_disabled = "auto"
        self.client.config.nas.score = Dict(
            type="scoring-function",
            metrics=Dict(
                active_energy=Dict(
                    type="energy-budget-from-power",
                    power_mw=Dict(type="metric", metric="avg_power_mw"),
                    duration_ms=Dict(type="metric", metric="latency_budget_ms"),
                )
            ),
            params=Dict(terms=[Dict(type="weighted", metric="active_energy", weight=-1.0)]),
        )
        self.client._hil_request = MagicMock()

        with self.assertRaisesRegex(ValueError, "runtime-only metric"):
            self.client.objective(DummyTrial())

        self.client._hil_request.assert_not_called()
        self.client.model_family.sample_hparams.assert_not_called()

    def test_objective_hil_true_ignores_compile_when_hil_disabled(self) -> None:
        """Real HIL mode should still request hardware metrics."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
        }
        self.client.config.device.hil = True
        self.client.config.device.compile_when_hil_disabled = "false"
        self.client._hil_request = MagicMock(return_value=metrics)

        self.client.objective(DummyTrial())

        self.client._hil_request.assert_called_once()

    def test_objective_pure_desktop_train_false_flops_omits_quantization_search(self) -> None:
        """No-op deployment choices should not expand pure desktop flops-only search."""
        self.client.config.device.hil = False
        self.client.config.device.compile_when_hil_disabled = "false"
        self.client.config.training.train = False
        self.client.config.training.quantization = Dict(
            mode="int8_ptq",
            search=True,
            choices=["float", "int8_ptq"],
        )
        self.client.config.nas.score = Dict(
            type="scoring-function",
            metrics=Dict(),
            params=Dict(terms=[Dict(type="weighted", metric="flops", weight=-1.0)]),
        )
        self.client._hil_request = MagicMock()
        trial = DummyTrial()

        result = self.client.objective(trial)

        self.assertEqual(result, -1234.0)
        self.client._hil_request.assert_not_called()
        self.client.task.build_fit_plan.assert_not_called()
        self.assertNotIn("quantization_mode", trial.params)

    def test_feasibility_metrics_participate_in_hil_dependency_classification(self) -> None:
        """Runtime-only feasibility metrics should force runtime dependency validation."""
        self.client.config.nas.score = Dict(
            type="scoring-function",
            metrics=Dict(),
            params=Dict(terms=[Dict(type="weighted", metric="flops", weight=-1.0)]),
        )
        self.client.config.nas.feasibility = Dict(
            train_if_infeasible=False,
            rules=[
                Dict(
                    rule="latency_budget",
                    metric="latency_ms",
                    condition="gt",
                    reference=Dict(type="metric", metric="latency_budget_ms"),
                    reason="Latency exceeds cadence budget",
                )
            ],
        )

        dependencies = self.client._classify_nas_metric_dependencies()

        self.assertIn("latency_ms", dependencies.metrics)
        self.assertIn("latency_budget_ms", dependencies.metrics)
        self.assertIn("latency_ms", dependencies.runtime_only)


class SmokeTestTests(unittest.TestCase):
    """Ensure the convenience smoke_test helper toggles config safely."""

    def test_smoke_test_restores_config_flags(self) -> None:
        """Config flags should be restored even if the study raises."""
        # Smoke tests should restore mutated config flags even when the validation run raises.
        client = _build_test_client()
        client.objective = MagicMock(return_value=0.1)

        class DummyStudy:
            """Simple Optuna study substitute that tracks optimize invocations."""

            def __init__(self):
                """Initialize the instance."""
                self.best_trial = SimpleNamespace(value=1.0, params={}, user_attrs={})
                self.optimize_calls = []
                self.metric_names_calls = []
                self.trials = []

            def set_metric_names(self, metric_names):
                """Set metric names.

                Parameters
                ----------
                metric_names : object
                    Metric names reported by the fake study.
                """
                self.metric_names_calls.append(list(metric_names))

            def optimize(self, func, n_trials, callbacks=None, catch=()):
                """Run optimize.

                Parameters
                ----------
                func : object
                    Objective callback invoked by the fake study.
                n_trials : object
                    Number of fake trials to execute.
                """
                self.optimize_calls.append((func, n_trials))

        fake_study = DummyStudy()

        with patch("nas_model_client.optuna.create_study", return_value=fake_study):
            with self.assertRaises(ValueError):
                client.smoke_test(train=False, hil=False, trials=2, epochs=1)

        self.assertEqual(client.config.device.hil, True)
        self.assertEqual(client.config.training.train, True)
        self.assertEqual(client.config.training.nas_epochs, 10)
        self.assertEqual(client.config.nas.score.type, "scoring-function")

    def test_smoke_test_uses_loaded_multiobjective_config(self) -> None:
        """Smoke test should honor multi-objective mode from the loaded config."""
        # Smoke tests should honor a loaded multi-objective config instead of forcing a scalar study shape.
        client = _build_test_client()
        client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(
                objectives=[
                    Dict(metric="rmse_total", direction="minimize"),
                    Dict(metric="latency_ms", direction="minimize"),
                ]
            ),
        )
        client.objective = MagicMock(return_value=(0.1, 1.0))

        class DummyStudy:
            """Simple Optuna study substitute for multi-objective runs."""

            def __init__(self):
                """Initialize the instance."""
                trial = SimpleNamespace(
                    state=TrialState.COMPLETE,
                    number=0,
                    values=(0.5, 1.5),
                    params={"foo": 1},
                    user_attrs={"latency_ms": 1.0},
                )
                self.best_trials = [trial]
                self.trials = [trial]
                self.optimize_calls = []
                self.metric_names_calls = []

            def set_metric_names(self, metric_names):
                """Set metric names.

                Parameters
                ----------
                metric_names : object
                    Metric names reported by the fake study.
                """
                self.metric_names_calls.append(list(metric_names))

            def optimize(self, func, n_trials, callbacks=None, catch=()):
                """Run optimize.

                Parameters
                ----------
                func : object
                    Objective callback invoked by the fake study.
                n_trials : object
                    Number of fake trials to execute.
                """
                self.optimize_calls.append((func, n_trials))

        fake_study = DummyStudy()

        with patch("nas_model_client.optuna.create_study", return_value=fake_study) as mock_create:
            client.smoke_test(train=True, hil=False, trials=1, epochs=1)

        self.assertEqual(fake_study.optimize_calls[0][1], 1)
        self.assertEqual(fake_study.metric_names_calls, [["rmse_total", "latency_ms"]])
        kwargs = mock_create.call_args.kwargs
        self.assertEqual(kwargs["directions"], ["minimize", "minimize"])
        self.assertTrue(kwargs["load_if_exists"])

    def test_smoke_test_uses_loaded_scalar_config(self) -> None:
        # Scalar smoke tests should preserve the loaded scalar study direction.
        """Validate smoke test uses loaded scalar config."""
        client = _build_test_client()
        client.objective = MagicMock(return_value=0.1)

        class DummyStudy:
            """Dummy Optuna study used by NAS client tests."""

            def __init__(self):
                """Initialize the instance."""
                self.best_trial = SimpleNamespace(value=1.0, params={}, user_attrs={})
                self.optimize_calls = []
                self.metric_names_calls = []
                self.trials = []

            def set_metric_names(self, metric_names):
                """Set metric names.

                Parameters
                ----------
                metric_names : object
                    Metric names reported by the fake study.
                """
                self.metric_names_calls.append(list(metric_names))

            def optimize(self, func, n_trials, callbacks=None, catch=()):
                """Run optimize.

                Parameters
                ----------
                func : object
                    Objective callback invoked by the fake study.
                n_trials : object
                    Number of fake trials to execute.
                """
                self.optimize_calls.append((func, n_trials))

        fake_study = DummyStudy()
        with patch("nas_model_client.optuna.create_study", return_value=fake_study) as mock_create:
            client.smoke_test(train=True, hil=False, trials=1, epochs=1)

        self.assertEqual(mock_create.call_args.kwargs["direction"], "maximize")
        self.assertTrue(mock_create.call_args.kwargs["load_if_exists"])
        self.assertEqual(fake_study.metric_names_calls, [["score"]])

    def test_smoke_test_defaults_to_loaded_hil_setting(self) -> None:
        # Smoke tests should inherit the loaded HIL flag so validation exercises the same execution mode the real run will use.
        """Validate smoke test defaults to loaded hil setting."""
        client = _build_test_client()
        observed_hil_values = []

        def _objective(_trial):
            """Evaluate the dummy objective.

            Parameters
            ----------
            _trial : object
                Trial object supplied by the test harness.

            Returns
            -------
            object
                Objective value returned by the dummy study.
            """
            observed_hil_values.append(client.config.device.hil)
            return 0.1

        client.objective = MagicMock(side_effect=_objective)

        class DummyStudy:
            """Dummy Optuna study used by NAS client tests."""

            def __init__(self):
                """Initialize the instance."""
                self.best_trial = SimpleNamespace(value=1.0, params={}, user_attrs={})
                self.optimize_calls = []
                self.metric_names_calls = []
                self.trials = []

            def set_metric_names(self, metric_names):
                """Set metric names.

                Parameters
                ----------
                metric_names : object
                    Metric names reported by the fake study.
                """
                self.metric_names_calls.append(list(metric_names))

            def optimize(self, func, n_trials, callbacks=None, catch=()):
                """Run optimize.

                Parameters
                ----------
                func : object
                    Objective callback invoked by the fake study.
                n_trials : object
                    Number of fake trials to execute.
                """
                self.optimize_calls.append((func, n_trials))
                func(SimpleNamespace())

        fake_study = DummyStudy()
        with patch("nas_model_client.optuna.create_study", return_value=fake_study):
            client.smoke_test(train=True, trials=1, epochs=1)

        self.assertEqual(observed_hil_values, [True])

    def test_smoke_test_rejects_derived_rmse_usage_when_training_disabled(self) -> None:
        # Derived RMSE metrics should be rejected when training is disabled because no training pass can produce them.
        """Validate smoke test rejects derived rmse usage when training disabled."""
        client = _build_test_client()
        client.config.nas.score = Dict(
            type="scoring-function",
            metrics=Dict(
                combined_error=Dict(
                    type="add",
                    metrics=["rmse_total", "flops"],
                )
            ),
            params=Dict(
                terms=[
                    Dict(type="weighted", metric="combined_error", weight=-1.0),
                ]
            ),
        )

        with patch("nas_model_client.optuna.create_study") as mock_create:
            with self.assertRaises(ValueError):
                client.smoke_test(train=False, trials=1, epochs=1)

        self.assertFalse(mock_create.called)

    def test_smoke_test_preserves_existing_db_and_log_before_validation(self) -> None:
        # Smoke-test validation should not clobber an existing DB or log before the run is known to be valid.
        """Validate smoke test preserves existing db and log before validation."""
        client = _build_test_client()
        client.objective = MagicMock(return_value=0.1)
        study_name = "stale_smoke"
        client.study_name = study_name

        class DummyStudy:
            """Dummy Optuna study used by NAS client tests."""

            def __init__(self):
                """Initialize the instance."""
                self.best_trial = SimpleNamespace(value=1.0, params={}, user_attrs={})
                self.optimize_calls = []
                self.trials = []

            def optimize(self, func, n_trials, callbacks=None, catch=()):
                """Run optimize.

                Parameters
                ----------
                func : object
                    Objective callback invoked by the fake study.
                n_trials : object
                    Number of fake trials to execute.
                """
                self.optimize_calls.append((func, n_trials))

        fake_study = DummyStudy()
        artifacts_dir = client._artifacts_dir()
        db_path = artifacts_dir / "optuna_smoke_test.db"
        db_path.write_text("stale-db", encoding="utf-8")
        log_path = artifacts_dir / client.config.outputs.log_file_name
        log_path.write_text("stale-log", encoding="utf-8")

        with patch("nas_model_client.optuna.create_study", return_value=fake_study) as mock_create:
            with self.assertRaises(ValueError):
                client.smoke_test(train=False, hil=False, trials=1, epochs=1, study_name=study_name)

        self.assertTrue(db_path.exists())
        self.assertTrue(log_path.exists())
        self.assertEqual(db_path.read_text(encoding="utf-8"), "stale-db")
        self.assertEqual(log_path.read_text(encoding="utf-8"), "stale-log")
        self.assertFalse(mock_create.called)

    def test_copy_run_config_skips_same_file(self) -> None:
        # Copying the run config should no-op when the source file already lives in the artifacts directory.
        """Validate copy run config skips same file."""
        client = _build_test_client()
        artifacts_dir = client._artifacts_dir()
        cfg_path = artifacts_dir / "config.yaml"
        cfg_path.write_text("device:\n  name: TEST_DEVICE\n", encoding="utf-8")
        client.config_path = cfg_path

        copied_path = client._copy_run_config(artifacts_dir)

        self.assertEqual(copied_path, cfg_path.resolve())
        self.assertEqual(cfg_path.read_text(encoding="utf-8"), "device:\n  name: TEST_DEVICE\n")


class MeasurementFailureGuardTests(unittest.TestCase):
    """Real Optuna lifecycle regressions; objectives never train or touch hardware."""

    @staticmethod
    def _fail(trial):
        """Record infrastructure evidence and raise the failure consumed by Optuna."""
        trial.set_user_attr("measurement_failure_class", "infrastructure")
        trial.set_user_attr("measurement_record_path", f"/evidence/{trial.number}.json")
        raise MeasurementInfrastructureError("bounded recovery exhausted")

    def test_three_failures_stop_run_and_resume_preserves_budget(self):
        """Three committed failures must stop proposals while resume preserves the trial budget."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = _build_test_client(root)
            client.config.training.nas_trials = 4
            client.config.training.max_total_trials = 4
            client.objective = MagicMock(side_effect=self._fail)
            storage = "sqlite:///" + str(root / "stop.db")
            with self.assertRaises(MeasurementInfrastructureStop):
                client.run_nas("stop", storage=storage)
            study = optuna.load_study(study_name="stop", storage=storage)
            self.assertEqual(client.objective.call_count, 3)
            self.assertEqual([t.state for t in study.trials], [TrialState.FAIL] * 3)
            self.assertEqual([t.user_attrs["measurement_record_path"] for t in study.trials],
                             [f"/evidence/{i}.json" for i in range(3)])
            # An explicit repaired invocation resets the streak, not the database/budget.
            client.objective = MagicMock(return_value=1.0)
            resumed = client.run_nas("stop", storage=storage)
            self.assertEqual(client.objective.call_count, 1)
            self.assertEqual([t.number for t in resumed.trials], [0, 1, 2, 3])
            self.assertEqual([t.state for t in resumed.trials], [TrialState.FAIL] * 3 + [TrialState.COMPLETE])

    def test_only_proven_measurement_success_resets_streak(self):
        """Only valid measurement evidence may reset the infrastructure failure streak."""
        study = optuna.create_study()
        guard = _ConsecutiveMeasurementFailureGuard()
        kwargs = dict(callbacks=[guard], catch=(MeasurementInfrastructureError,))
        study.optimize(self._fail, n_trials=2, **kwargs)
        def measured(trial):
            """Represent a valid measurement rejected only for deadline infeasibility."""
            trial.set_user_attr("measurement_instrumentation_valid", True)
            raise optuna.TrialPruned("valid measurement, deadline infeasible")
        study.optimize(measured, n_trials=1, **kwargs)
        self.assertEqual(guard.streak, 0)
        study.optimize(self._fail, n_trials=2, **kwargs)
        study.optimize(lambda trial: 10000., n_trials=1, **kwargs)
        def rejected(trial):
            """Represent a pre-measurement rejection that cannot reset the streak."""
            raise optuna.TrialPruned("pre-HIL compile rejection")
        study.optimize(rejected, n_trials=1, **kwargs)
        self.assertEqual(guard.streak, 2)
        with self.assertRaises(MeasurementInfrastructureStop):
            study.optimize(self._fail, n_trials=1, **kwargs)
        self.assertEqual(study.trials[-1].state, TrialState.FAIL)

    def test_each_transport_kind_commits_fail_and_advances_guard(self):
        """Each recoverable transport failure must commit FAIL and advance the guard."""
        study = optuna.create_study()
        guard = _ConsecutiveMeasurementFailureGuard()
        for kind in ("harness_ready_timeout", "harness_done_timeout", "runtime_io", "runtime_start_timeout"):
            with self.subTest(kind=kind):
                guard.streak = 0
                def failed(trial):
                    """Attach the transport category before producing an infrastructure failure."""
                    trial.set_user_attr("backend_error_kind", kind)
                    self._fail(trial)
                study.optimize(failed, n_trials=1, callbacks=[guard], catch=(MeasurementInfrastructureError,))
                self.assertEqual(study.trials[-1].state, TrialState.FAIL)
                self.assertEqual(guard.streak, 1)

    def test_smoke_stops_on_third_failure_and_restores_settings(self):
        """Smoke runs must stop after three failures and restore configuration settings."""
        with tempfile.TemporaryDirectory() as directory:
            client = _build_test_client(Path(directory))
            original = (client.config.device.hil, client.config.training.train,
                        client.config.training.nas_epochs)
            client.objective = MagicMock(side_effect=self._fail)
            with self.assertRaises(MeasurementInfrastructureStop):
                client.smoke_test(trials=5, train=True, hil=True, epochs=1, study_name="smoke_stop")
            database = client._artifacts_dir() / "optuna_smoke_test.db"
            study = optuna.load_study(study_name="smoke_stop", storage="sqlite:///" + str(database))
            self.assertEqual(client.objective.call_count, 3)
            self.assertEqual([t.state for t in study.trials], [TrialState.FAIL] * 3)
            self.assertEqual(original, (client.config.device.hil, client.config.training.train,
                                       client.config.training.nas_epochs))


class RunNASTests(unittest.TestCase):
    def test_real_nsga_infrastructure_failure_consumes_budget_and_continues(self):
        """NSGA must count an infrastructure failure toward budget and continue afterward."""
        with tempfile.TemporaryDirectory() as directory:
            client = _build_test_client(Path(directory))
            client.config.training.nas_trials = 2
            client.config.training.max_total_trials = 2
            client.config.training.nas_multiobjective_population_size = 2
            client.config.nas.score = Dict(
                type="multi-objective", metrics=Dict(), params=Dict(objectives=[
                    Dict(metric="rmse_total", direction="minimize"),
                    Dict(metric="latency_ms", direction="minimize")]))
            def objective(trial):
                """Fail the first proposal, then return valid NSGA objective values."""
                if trial.number == 0:
                    trial.set_user_attr("measurement_failure_class", "infrastructure")
                    raise MeasurementInfrastructureError("recovery exhausted")
                return 1.0, 2.0
            client.objective = objective
            study = client.run_nas("infra", storage="sqlite:///" + str(Path(directory) / "study.db"))
            self.assertIsInstance(study.sampler, optuna.samplers.NSGAIISampler)
            self.assertEqual([t.state for t in study.trials], [TrialState.FAIL, TrialState.COMPLETE])
            self.assertIsNone(study.trials[0].values)
            self.assertEqual(len(study.trials), 2)  # No replacement proposal beyond attempt cap.

    def test_real_run_unrelated_exception_still_aborts(self):
        """Unrelated objective exceptions must still abort NAS execution."""
        with tempfile.TemporaryDirectory() as directory:
            client = _build_test_client(Path(directory))
            client.config.training.nas_trials = 2
            client.config.training.max_total_trials = 2
            client.objective = MagicMock(side_effect=ValueError("unrelated programming error"))
            with self.assertRaisesRegex(ValueError, "unrelated programming error"):
                client.run_nas("unrelated", storage="sqlite:///" + str(Path(directory) / "study.db"))
            study = optuna.load_study(study_name="unrelated", storage="sqlite:///" + str(Path(directory) / "study.db"))
            self.assertEqual(len(study.trials), 1)
            self.assertEqual(study.trials[0].state, TrialState.FAIL)

    """Run_nas should continue until completed trials meet the target."""

    class DummyStudy:
        """Dummy Optuna study used by NAS client tests."""

        def __init__(self, states):
            """Initialize the instance.

            Parameters
            ----------
            states : object
                Trial states requested from the fake study.
            """
            self.states_queue = list(states)
            self.trials = []
            self.optimize_calls = []
            self.best_trial = SimpleNamespace(value=None, params={})
            self.best_value = None
            self.enqueue_calls = []
            self.metric_names_calls = []
            self.user_attrs = {}

        def set_metric_names(self, metric_names):
            """Set metric names.

            Parameters
            ----------
            metric_names : object
                Metric names reported by the fake study.
            """
            self.metric_names_calls.append(list(metric_names))

        def set_user_attr(self, key, value):
            """Set user attr.

            Parameters
            ----------
            key : object
                Dictionary key recorded by the test double.
            value : object
                Value recorded by the test double.
            """
            self.user_attrs[key] = value

        def optimize(self, func, n_trials, callbacks=None, catch=()):
            """Run optimize.

            Parameters
            ----------
            func : object
                Objective callback invoked by the fake study.
            n_trials : object
                Number of fake trials to execute.
            """
            self.optimize_calls.append(n_trials)
            for _ in range(n_trials):
                state = self.states_queue.pop(0) if self.states_queue else TrialState.FAIL
                trial = SimpleNamespace(state=state)
                self.trials.append(trial)
            complete = [t for t in self.trials if t.state == TrialState.COMPLETE]
            if complete:
                self.best_trial = SimpleNamespace(value=1.0, params={})
                self.best_value = self.best_trial.value

        def enqueue_trial(self, params):
            """Run enqueue trial.

            Parameters
            ----------
            params : object
                Parameter dictionary recorded by the fake trial.
            """
            self.enqueue_calls.append(params)

    def test_run_nas_retries_until_completed_target(self) -> None:
        """Continue trials until the configured completion target is reached.

        The orchestration loop should not enqueue the model family's default
        seed trial; fresh NAS runs should begin with sampler-generated
        candidates.
        """
        client = _build_test_client()
        client.config.training.nas_trials = 2
        client.config.training.max_total_trials = 5
        states = [TrialState.PRUNED, TrialState.COMPLETE, TrialState.COMPLETE]
        dummy = self.DummyStudy(states)
        client.objective = MagicMock()

        with patch("nas_model_client.optuna.create_study", return_value=dummy):
            study = client.run_nas(study_name="demo", storage="sqlite:///dummy.db")

        self.assertIs(study, dummy)
        self.assertEqual(sum(t.state == TrialState.COMPLETE for t in dummy.trials), 2)
        self.assertEqual(len(dummy.trials), 3)
        self.assertEqual(dummy.optimize_calls, [2, 1])
        self.assertEqual(dummy.enqueue_calls, [])

    def test_run_nas_sets_multiobjective_metric_names(self) -> None:
        """Multi-objective studies should expose configured objective names."""
        client = _build_test_client()
        client.config.training.nas_trials = 1
        client.config.training.max_total_trials = 1
        client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(
                objectives=[
                    Dict(metric="rmse_total", direction="minimize"),
                    Dict(metric="latency_ms", direction="minimize"),
                ]
            ),
        )
        dummy = self.DummyStudy([TrialState.COMPLETE])
        client.objective = MagicMock()

        with patch("nas_model_client.optuna.create_study", return_value=dummy) as mock_create:
            study = client.run_nas(study_name="demo", storage="sqlite:///dummy.db")

        self.assertIs(study, dummy)
        self.assertEqual(mock_create.call_args.kwargs["directions"], ["minimize", "minimize"])
        self.assertEqual(dummy.metric_names_calls, [["rmse_total", "latency_ms"]])
        self.assertEqual(dummy.optimize_calls, [1])

    def test_run_nas_wires_constraints_sampler_when_feasibility_enabled(self) -> None:
        """Both sampler families should receive the persisted constraints hook."""
        client = _build_test_client()
        client.config.training.nas_trials = 1
        client.config.training.max_total_trials = 1
        client.config.nas.feasibility = Dict(
            train_if_infeasible=False,
            rules=[
                Dict(
                    rule="latency_budget",
                    metric="latency_ms",
                    condition="gt",
                    reference=Dict(type="metric", metric="latency_budget_ms"),
                    reason="Latency exceeds cadence budget",
                )
            ],
        )
        dummy = self.DummyStudy([TrialState.COMPLETE])
        client.objective = MagicMock()
        sentinel_sampler = object()

        with patch("nas_model_client.optuna.samplers.TPESampler", return_value=sentinel_sampler) as mock_tpe:
            with patch("nas_model_client.optuna.create_study", return_value=dummy):
                client.run_nas(study_name="demo", storage="sqlite:///dummy.db")

        constraints_func = mock_tpe.call_args.kwargs["constraints_func"]
        self.assertIs(constraints_func.__self__, client)
        self.assertIs(constraints_func.__func__, client._constraints_func.__func__)
        self.assertEqual(
            dummy.user_attrs["crest_feasibility_policy_signature"]["rules"][0]["rule"],
            "latency_budget",
        )

    def test_run_nas_does_not_count_not_evaluated_complete_trials_as_infeasible(self) -> None:
        """Hard-pruned COMPLETE penalty trials should log separately.

        Returns
        -------
        None
            The test passes when not-evaluated complete trials remain out of
            the infeasible count and the summary is emitted through logging.
        """
        client = _build_test_client()
        client.config.training.nas_trials = 1
        client.config.training.max_total_trials = 1
        client.config.nas.feasibility = Dict(
            train_if_infeasible=False,
            rules=[
                Dict(
                    rule="latency_budget",
                    metric="latency_ms",
                    condition="gt",
                    reference=Dict(type="metric", metric="latency_budget_ms"),
                    reason="Latency exceeds cadence budget",
                )
            ],
        )
        trial = SimpleNamespace(
            state=TrialState.COMPLETE,
            user_attrs={"feasible": False, "feasibility_status": "not_evaluated"},
        )

        class DummyStudy:
            """Study double with one hard-pruned complete penalty trial."""

            def __init__(self) -> None:
                """Initialize the instance."""
                self.trials = [trial]
                self.user_attrs = {
                    "crest_feasibility_policy_signature": client._feasibility_policy_signature()
                }
                self.metric_names_calls = []

            def set_metric_names(self, metric_names):
                """Record metric names like an Optuna study.

                Parameters
                ----------
                metric_names : object
                    Metric names reported by the fake study.
                """
                self.metric_names_calls.append(list(metric_names))

        dummy = DummyStudy()

        with patch("nas_model_client.optuna.create_study", return_value=dummy):
            with self.assertLogs("nas_model_client", level="INFO") as captured:
                client.run_nas(study_name="demo", storage="sqlite:///dummy.db")

        final_line = "\n".join(captured.output)
        self.assertIn("0 feasible, 0 infeasible, 1 completed", final_line)

    def test_run_nas_rejects_mismatched_feasibility_signature_on_resume(self) -> None:
        """Resume should fail when stored feasibility policy differs."""
        client = _build_test_client()
        client.config.nas.feasibility = Dict(
            train_if_infeasible=False,
            rules=[
                Dict(
                    rule="latency_budget",
                    metric="latency_ms",
                    condition="gt",
                    reference=Dict(type="metric", metric="latency_budget_ms"),
                    reason="Latency exceeds cadence budget",
                )
            ],
        )
        dummy = self.DummyStudy([TrialState.COMPLETE])
        dummy.user_attrs["crest_feasibility_policy_signature"] = {
            "train_if_infeasible": False,
            "rules": [
                {
                    "rule": "other",
                    "metric": "latency_ms",
                    "condition": "gt",
                    "reference": {"type": "metric", "metric": "latency_budget_ms"},
                }
            ],
        }

        with patch("nas_model_client.optuna.create_study", return_value=dummy):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                client.run_nas(study_name="demo", storage="sqlite:///dummy.db")

    def test_run_nas_honors_max_total_trials_cap(self) -> None:
        """Stop retrying when the global trial-attempt cap is reached.

        Failed and pruned attempts should consume the cap without adding a
        model-family seed trial to the study.
        """
        client = _build_test_client()
        client.config.training.nas_trials = 2
        client.config.training.max_total_trials = 3
        states = [TrialState.PRUNED, TrialState.FAIL, TrialState.PRUNED]
        dummy = self.DummyStudy(states)
        client.objective = MagicMock()

        with patch("nas_model_client.optuna.create_study", return_value=dummy):
            study = client.run_nas(study_name="demo", storage="sqlite:///dummy.db")

        self.assertIs(study, dummy)
        self.assertEqual(len(dummy.trials), 3)
        self.assertEqual(dummy.optimize_calls, [2, 1])
        self.assertEqual(sum(t.state == TrialState.COMPLETE for t in dummy.trials), 0)
        self.assertEqual(dummy.enqueue_calls, [])

class TrainBestTrialTests(unittest.TestCase):
    """Best-trial retraining should honor the task abstraction."""

    def test_train_best_trial_uses_task_fit_plan_with_override_task_settings(self) -> None:
        """Best-trial retraining should ignore runtime-only trial params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            built_model = client.model_family.build_model.return_value
            built_model.fit.return_value = SimpleNamespace(history={"loss": [1.0], "val_loss": [0.5]})
            fit_task = MagicMock()
            fit_task.build_fit_plan.return_value = FitPlan(
                fit_kwargs={
                    "x": client.dataset_bundle.train.inputs,
                    "y": [client.dataset_bundle.train.targets["velx"], client.dataset_bundle.train.targets["vely"]],
                    "validation_data": (
                        client.dataset_bundle.val.inputs,
                        [client.dataset_bundle.val.targets["velx"], client.dataset_bundle.val.targets["vely"]],
                    ),
                    "shuffle": True,
                },
                callbacks=["task-callback"],
                monitor_metric="val_loss",
            )
            best_trial = SimpleNamespace(
                params={
                    "nb_filters": 2,
                    "kernel_size": 2,
                    "dropout_rate": 0.1,
                    "use_skip_connections": True,
                    "norm_flag": True,
                    "dilations_index": 0,
                    "cpu_clock_mhz_index": 2,
                    "quantization_mode": "int8_ptq",
                }
            )

            with patch.object(client, "_instantiate_task", return_value=fit_task) as instantiate_mock, patch(
                "nas_model_client.optuna.load_study",
                return_value=SimpleNamespace(best_trial=best_trial),
            ):
                history = client.train_best_trial(
                    study_storage="sqlite:///optuna.db",
                    study_name="demo",
                    patience=7,
                    checkpoint_path=base / "best.keras",
                    history_path=base / "history.json",
                )

            instantiate_mock.assert_called_once_with(
                client.task_name,
                client.config,
                client.task_config,
                checkpoint_path=base / "best.keras",
                early_stopping_patience=7,
            )
            fit_task.build_fit_plan.assert_called_once_with(
                client.dataset_bundle,
                client.task_config,
                client.target_spec,
                mode="final",
                combine_train_val=False,
            )
            client.model_family.decode_trial_hparams.assert_called_once_with(
                {
                    "nb_filters": 2,
                    "kernel_size": 2,
                    "dropout_rate": 0.1,
                    "use_skip_connections": True,
                    "norm_flag": True,
                    "dilations_index": 0,
                },
                client.model_build_context,
                client.model_config,
            )
            built_model.fit.assert_called_once()
            fit_call = built_model.fit.call_args.kwargs
            self.assertEqual(fit_call["callbacks"], ["task-callback"])
            self.assertEqual(fit_call["epochs"], client.config.training.model_epochs)
            self.assertEqual(fit_call["batch_size"], 256)
            self.assertEqual(history["loss"], [1.0])


class PlotTrainingHistoryTests(unittest.TestCase):
    """Plotting helpers should emit PNGs without requiring a display."""

    def test_plot_training_history_writes_pngs(self) -> None:
        # Training-history plots should materialize PNG artifacts so the run bundle is self-contained.
        """Validate plot training history writes pngs."""
        import matplotlib.pyplot as plt

        plt.switch_backend("Agg")
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _build_test_client(base_dir=Path(tmpdir))
            history = {
                "loss": [1.0, 0.5],
                "val_loss": [1.1, 0.6],
                "velx_loss": [0.9, 0.4],
                "val_velx_loss": [1.0, 0.5],
            }

            result = client.plot_training_history(history=history, output_dir=Path(tmpdir), study_name="demo")

            loss_path = Path(result["loss_plot"])
            self.assertTrue(loss_path.is_file())
            self.assertGreater(loss_path.stat().st_size, 0)
            self.assertIn("loss_components_plot", result)
            components_path = Path(result["loss_components_plot"])
            self.assertTrue(components_path.is_file())
            self.assertGreater(components_path.stat().st_size, 0)


class EvaluateCheckpointTests(unittest.TestCase):
    """Checkpoint evaluation should write metrics and (optionally) export TFLite."""

    def test_evaluate_checkpoint_writes_metrics(self) -> None:
        # Checkpoint evaluation should persist its metrics file so post-train analysis can run offline.
        """Validate evaluate checkpoint writes metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            length = 4
            gt_vx = np.ones((length, 1), dtype=np.float32)
            gt_vy = np.ones((length, 1), dtype=np.float32)
            client.dataset_bundle.test = DataSplit(
                inputs=np.zeros((length, 1, 1), dtype=np.float32),
                targets={"velx": gt_vx, "vely": gt_vy},
                metadata={},
            )
            client.task.evaluate.return_value = EvaluationResult(
                metrics={"rmse_vel_x": 0.0, "rmse_vel_y": 0.0, "rmse_total": 0.0},
                predictions=[gt_vx, gt_vy],
            )

            metrics_path = base / "metrics.json"
            metrics = client.evaluate_checkpoint(
                checkpoint_path=base / "ckpt.keras",
                metrics_path=metrics_path,
                export_tflite=False,
            )

            self.assertTrue(metrics_path.is_file())
            self.assertTrue(metrics_path.with_suffix(".csv").is_file())
            self.assertAlmostEqual(metrics["rmse_vel_x"], 0.0)
            self.assertAlmostEqual(metrics["rmse_vel_y"], 0.0)
            self.assertNotIn("keras_rmse_vel_x", metrics)
            self.assertNotIn("keras_rmse_total", metrics)
            self.assertEqual(metrics["checkpoint_path"], str(base / "ckpt.keras"))

    def test_evaluate_checkpoint_preserves_task_defined_metric_names(self) -> None:
        # Checkpoint evaluation should preserve task-defined metric names instead of flattening them into generic labels.
        """Validate evaluate checkpoint preserves task defined metric names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            client.task.evaluate.return_value = EvaluationResult(
                metrics={"f1_macro": np.float32(0.8), "loss": np.float64(0.25)},
                predictions=None,
            )

            metrics_path = base / "metrics.json"
            metrics = client.evaluate_checkpoint(
                checkpoint_path=base / "ckpt.keras",
                metrics_path=metrics_path,
                export_tflite=False,
            )

            self.assertAlmostEqual(metrics["f1_macro"], 0.8)
            self.assertAlmostEqual(metrics["loss"], 0.25)
            self.assertNotIn("rmse_vel_x", metrics)
            with metrics_path.open() as handle:
                persisted = json.load(handle)
            self.assertAlmostEqual(persisted["f1_macro"], 0.8)
            self.assertAlmostEqual(persisted["loss"], 0.25)

    def test_evaluate_checkpoint_keeps_quantization_out_of_hparams(self) -> None:
        """Checkpoint metrics should report quantization separately from hparams."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            best_trial = SimpleNamespace(
                params={"nb_filters": 8, "quantization_mode": "int8_ptq", "cpu_clock_mhz_index": 1}
            )

            with patch(
                "nas_model_client.optuna.load_study",
                return_value=SimpleNamespace(best_trial=best_trial),
            ):
                metrics = client.evaluate_checkpoint(
                    checkpoint_path=base / "ckpt.keras",
                    metrics_path=base / "metrics.json",
                    study_storage="sqlite:///optuna.db",
                    study_name="demo",
                    export_tflite=False,
                )

            self.assertEqual(metrics["hyperparameters"], {"nb_filters": 8})
            self.assertEqual(metrics["quantization_mode"], "int8_ptq")

    def test_evaluate_checkpoint_tflite_logs_keras_accuracy(self) -> None:
        """Checkpoint TFLite evaluation should persist paired Keras accuracy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            client.task.evaluate_predictions.return_value = EvaluationResult(
                metrics={"accuracy": 0.7, "loss": 0.4},
                predictions=None,
            )
            client.task.evaluate.return_value = EvaluationResult(
                metrics={"accuracy": 0.9, "loss": 0.2},
                predictions=None,
            )
            client.config.training.quantization.mode = "int8_ptq"
            client.config.training.quantization.choices = ["int8_ptq"]
            metrics_path = base / "metrics.json"

            with patch("nas_model_client.convert_to_tflite_model"), patch(
                "nas_model_client.predict_tflite_model_subprocess",
                return_value=np.zeros((2, 1), dtype=np.float32),
            ):
                metrics = client.evaluate_checkpoint(
                    checkpoint_path=base / "ckpt.keras",
                    metrics_path=metrics_path,
                    export_tflite=False,
                    evaluation_backend="tflite",
                )

            self.assertEqual(metrics["accuracy"], 0.7)
            self.assertEqual(metrics["keras_accuracy"], 0.9)
            with metrics_path.open() as handle:
                persisted = json.load(handle)
            self.assertEqual(persisted["keras_accuracy"], 0.9)
            csv_text = metrics_path.with_suffix(".csv").read_text(encoding="utf-8")
            self.assertIn("keras_accuracy", csv_text)

    def test_evaluate_checkpoint_tflite_failure_propagates_without_metrics(self) -> None:
        """Checkpoint TFLite worker failures should not write success metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            metrics_path = base / "metrics.json"
            failure = TFLiteSubprocessError(
                model_path=base / "model.tflite",
                return_code=-6,
                timeout=False,
                stderr_tail="abort",
                command=["python", "-m", "crest.tflite_predict_worker"],
            )

            with patch("nas_model_client.convert_to_tflite_model"), patch(
                "nas_model_client.predict_tflite_model_subprocess",
                side_effect=failure,
            ):
                with self.assertRaises(TFLiteSubprocessError):
                    client.evaluate_checkpoint(
                        checkpoint_path=base / "ckpt.keras",
                        metrics_path=metrics_path,
                        export_tflite=False,
                        evaluation_backend="tflite",
                    )

            self.assertFalse(metrics_path.exists())
            self.assertFalse(metrics_path.with_suffix(".csv").exists())
            client.task.evaluate.assert_not_called()

    def test_evaluate_checkpoint_exports_tflite_when_requested(self) -> None:
        # Checkpoint evaluation should export TFLite when requested so downstream deployment steps do not need a second conversion pass.
        """Validate evaluate checkpoint exports tflite when requested."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            calibration_inputs = np.full((3, 16, 3), 7.0, dtype=np.float32)
            client.dataset_bundle.calibration = DataSplit(
                inputs=calibration_inputs,
                targets={"velx": np.zeros((3, 1), dtype=np.float32), "vely": np.zeros((3, 1), dtype=np.float32)},
                metadata={},
            )
            gt_vx = np.zeros((2, 1), dtype=np.float32)
            gt_vy = np.zeros((2, 1), dtype=np.float32)
            client.dataset_bundle.test = DataSplit(
                inputs=np.zeros((2, 1, 1), dtype=np.float32),
                targets={"velx": gt_vx, "vely": gt_vy},
                metadata={},
            )
            client.task.evaluate.return_value = EvaluationResult(
                metrics={"rmse_vel_x": 0.0, "rmse_vel_y": 0.0, "rmse_total": 0.0},
                predictions=[gt_vx, gt_vy],
            )

            tflite_path = base / "model.tflite"
            with patch("nas_model_client.convert_to_tflite_model") as mock_convert:
                client.evaluate_checkpoint(
                    checkpoint_path=base / "ckpt.keras",
                    metrics_path=base / "metrics.json",
                    export_tflite=True,
                    tflite_path=tflite_path,
                )

            self.assertTrue(mock_convert.called)
            self.assertEqual(mock_convert.call_args.kwargs["output_name"], tflite_path)
            self.assertIsNone(mock_convert.call_args.kwargs["training_data"])
            self.assertEqual(mock_convert.call_args.kwargs["quantization_mode"], "float")

    def test_evaluate_checkpoint_rejects_tflite_when_family_does_not_support_it(self) -> None:
        # TFLite export should fail fast when the active model family opts out of export support.
        """Validate evaluate checkpoint rejects tflite when family does not support it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            client.model_family.supports_tflite.return_value = False

            with self.assertRaisesRegex(ValueError, "does not support TFLite export"):
                client.evaluate_checkpoint(
                    checkpoint_path=base / "ckpt.keras",
                    metrics_path=base / "metrics.json",
                    export_tflite=True,
                    tflite_path=base / "model.tflite",
                )


class FoldRotationReportingTests(unittest.TestCase):
    """Fold-rotation reporting should reuse the selected NAS hparams safely."""

    def test_run_scoring_nas_runs_fixed_final_before_fold_rotation(self) -> None:
        """Scoring orchestration should preserve fixed export before reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            client.task_config.evaluation = Dict(protocol="fold_rotation", fold_rotation=Dict(test_folds=[1]))
            order: list[str] = []
            trials_df = MagicMock()
            study = SimpleNamespace(
                trials=[object()],
                best_value=0.75,
                best_trial=SimpleNamespace(value=0.75, params={}),
                trials_dataframe=MagicMock(return_value=trials_df),
            )

            with patch.object(client, "run_nas", return_value=study), patch.object(
                client,
                "train_best_trial",
                side_effect=lambda **_kwargs: order.append("fixed_train") or {"loss": [1.0]},
            ), patch.object(
                client,
                "plot_training_history",
                side_effect=lambda **_kwargs: order.append("plots") or {"loss_plot": "loss.png"},
            ), patch.object(
                client,
                "evaluate_checkpoint",
                side_effect=lambda **_kwargs: order.append("fixed_eval_export")
                or {"checkpoint_path": "fixed.keras", "tflite_path": "fixed.tflite", "accuracy": 0.8},
            ), patch.object(
                client,
                "run_fold_rotation_final_evaluation",
                side_effect=lambda **_kwargs: order.append("fold_rotation")
                or {"summary_path": "fold_rotation/fold_rotation_summary.json"},
            ), patch.object(
                client, "write_summary_bundle"
            ) as write_summary:
                client.run_scoring_nas(study_name="demo")

            self.assertEqual(order, ["fixed_train", "plots", "fixed_eval_export", "fold_rotation"])
            write_summary.assert_called_once()
            self.assertEqual(
                write_summary.call_args.kwargs["fold_rotation_artifacts"]["summary_path"],
                "fold_rotation/fold_rotation_summary.json",
            )

    def test_run_scoring_nas_requires_feasible_trial_before_final_training(self) -> None:
        """Single-objective closeout should not retrain infeasible penalty trials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            client.config.nas.feasibility = Dict(
                train_if_infeasible=False,
                rules=[
                    Dict(
                        rule="latency_budget",
                        metric="latency_ms",
                        condition="gt",
                        reference=Dict(type="metric", metric="latency_budget_ms"),
                        reason="Latency exceeds cadence budget",
                    )
                ],
            )
            trials_df = MagicMock()
            infeasible_trial = SimpleNamespace(
                state=TrialState.COMPLETE,
                value=-100.0,
                params={"nb_filters": 2},
                user_attrs={"feasible": False, "feasibility_status": "infeasible"},
            )
            study = SimpleNamespace(
                trials=[infeasible_trial],
                trials_dataframe=MagicMock(return_value=trials_df),
                get_trials=MagicMock(return_value=[infeasible_trial]),
            )

            with patch.object(client, "run_nas", return_value=study), patch.object(
                client, "train_best_trial"
            ) as train_best:
                with self.assertRaisesRegex(RuntimeError, "without any feasible completed trials"):
                    client.run_scoring_nas(study_name="demo")

            train_best.assert_not_called()
            trials_df.to_csv.assert_called_once()

    def test_run_fold_rotation_uses_per_fold_context_without_export(self) -> None:
        """Fold reporting should write success artifacts for requested folds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            client.config.dataset.params.fold_rotation_cache_dir = str(base / "fold_cache")
            client.task_config.evaluation = Dict(
                protocol="fold_rotation",
                fold_rotation=Dict(test_folds=[1, 2]),
            )
            fold_tasks = {}

            def _pipeline_for_fold(fold_cache_dir: Path) -> SimpleNamespace:
                """Build one fake fold pipeline from the requested cache path.

                Parameters
                ----------
                fold_cache_dir : pathlib.Path
                    Per-fold cache directory requested by the runner.

                Returns
                -------
                types.SimpleNamespace
                    Fake bootstrapped pipeline for the fold.
                """
                fold = int(fold_cache_dir.name.split("_")[1])
                task = MagicMock()
                task.generate_closeout_artifacts.return_value = {"fold": fold}
                fold_tasks[fold] = task
                return SimpleNamespace(
                    bundle=client.dataset_bundle,
                    target_spec=client.target_spec,
                    model_build_context=client.model_build_context,
                    selection={
                        "model_config": client.model_config,
                        "task_config": client.task_config,
                    },
                    task=task,
                )

            with patch.object(client, "_best_trial_params", return_value={"nb_filters": 2}), patch.object(
                client, "_bootstrap_fold_pipeline", side_effect=_pipeline_for_fold
            ), patch.object(
                client, "_train_with_decoded_hparams", autospec=True, return_value={"loss": [1.0]}
            ) as train_mock, patch.object(
                client,
                "_evaluate_checkpoint_with_context",
                autospec=True,
                side_effect=[
                    {"accuracy": 0.8, "macro_f1": 0.7, "loss": 0.5, "quantization_mode": "int8_ptq"},
                    {"accuracy": 0.9, "macro_f1": 0.8, "loss": 0.4, "quantization_mode": "int8_ptq"},
                ],
            ) as eval_mock:
                result = client.run_fold_rotation_final_evaluation(
                    study_storage="sqlite:///optuna.db",
                    study_name="demo",
                    output_dir=base / "fold_rotation",
                )

            self.assertEqual(result["requested_test_folds"], [1, 2])
            self.assertEqual(result["completed_test_folds"], [1, 2])
            self.assertEqual(train_mock.call_count, 2)
            self.assertEqual(eval_mock.call_count, 2)
            self.assertNotIn("task", train_mock.call_args_list[0].kwargs)
            self.assertIs(eval_mock.call_args_list[0].kwargs["task"], fold_tasks[1])
            self.assertIs(eval_mock.call_args_list[1].kwargs["task"], fold_tasks[2])
            self.assertFalse(eval_mock.call_args_list[0].kwargs["export_tflite"])
            self.assertNotIn("quantization_mode", eval_mock.call_args_list[0].kwargs)
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            self.assertTrue(summary["partial"])
            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["folds"][0]["quantization_mode"], "int8_ptq")
            self.assertAlmostEqual(summary["aggregates"]["accuracy"]["mean"], 0.85)
            self.assertIsNotNone(summary["aggregates"]["accuracy"]["std"])

    def test_run_fold_rotation_writes_partial_manifest_on_failure(self) -> None:
        """Fold reporting should fail fast and persist completed folds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            client.config.dataset.params.fold_rotation_cache_dir = str(base / "fold_cache")
            client.task_config.evaluation = Dict(
                protocol="fold_rotation",
                fold_rotation=Dict(test_folds=[1, 2]),
            )
            pipeline = SimpleNamespace(
                bundle=client.dataset_bundle,
                target_spec=client.target_spec,
                model_build_context=client.model_build_context,
                selection={"model_config": client.model_config, "task_config": client.task_config},
                task=MagicMock(generate_closeout_artifacts=MagicMock(return_value={})),
            )

            with patch.object(client, "_best_trial_params", return_value={"nb_filters": 2}), patch.object(
                client, "_bootstrap_fold_pipeline", return_value=pipeline
            ), patch.object(
                client, "_train_with_decoded_hparams", autospec=True, return_value={"loss": [1.0]}
            ), patch.object(
                client,
                "_evaluate_checkpoint_with_context",
                autospec=True,
                side_effect=[
                    {"accuracy": 0.8, "macro_f1": 0.7, "loss": 0.5},
                    {"accuracy": float("nan"), "macro_f1": 0.8, "loss": 0.4},
                ],
            ):
                with self.assertRaisesRegex(ValueError, "not finite"):
                    client.run_fold_rotation_final_evaluation(
                        study_storage="sqlite:///optuna.db",
                        study_name="demo",
                        output_dir=base / "fold_rotation",
                    )

            partial = json.loads(
                (base / "fold_rotation" / "fold_rotation_summary.partial.json").read_text(encoding="utf-8")
            )
            self.assertEqual(partial["status"], "failed")
            self.assertTrue(partial["partial"])
            self.assertEqual(partial["completed_test_folds"], [1])
            self.assertIsNone(partial["aggregates"])


class TrajectoryMetricsTests(unittest.TestCase):
    """Trajectory metrics/plots should be generated with stubbed models."""

    def test_trajectory_metrics_and_plots_zero_error(self) -> None:
        # Perfect trajectories should evaluate to zero error and still emit the expected plots.
        """Validate trajectory metrics and plots zero error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            length = 4
            vx = np.full((length, 1), 0.5, dtype=np.float32)
            vy = np.full((length, 1), 0.5, dtype=np.float32)
            client.dataset_bundle.metadata.update(
                {"sampling_rate_hz": 100, "window_size": 2, "stride": 1}
            )
            client.dataset_bundle.test = DataSplit(
                inputs=np.zeros((length, 1, 1), dtype=np.float32),
                targets={"velx": vx, "vely": vy},
                metadata={"size_of_each": [length], "x0": [0.0], "y0": [0.0]},
            )

            class FakeModel:
                """Fake model used by artifact and prediction tests."""

                def predict(self, _inputs):
                    """Run predict.

                    Parameters
                    ----------
                    _inputs : object
                        Input tensor batch passed to the fake model.

                    Returns
                    -------
                    object
                        Predictions emitted by the fake model.
                    """
                    return [vx, vy]

            client.model_family.load_model.return_value = FakeModel()
            with patch.object(client.model_family, "load_model", return_value=FakeModel()):
                metrics = client.trajectory_metrics_and_plots(
                    checkpoint_path=base / "ckpt.keras",
                    plot_dir=base,
                    stride=1,
                    window_size=2,
                    study_name="demo",
                )

            metrics_path = base / "demo_trajectory_metrics.json"
            self.assertTrue(metrics_path.is_file())
            self.assertAlmostEqual(metrics["ate_mean"], 0.0)
            self.assertEqual(len(metrics["plots"]), 1)
            self.assertTrue(Path(metrics["plots"][0]).is_file())
            # RTE uses a 60s window; with tiny synthetic data it should be NaN.
            self.assertTrue(np.isnan(metrics["rte_median"]))

    def test_trajectory_metrics_and_plots_requires_odometry_metadata(self) -> None:
        # Trajectory analysis should fail fast when the evaluation metadata is missing the odometry fields it depends on.
        """Validate trajectory metrics and plots requires odometry metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            vx = np.full((2, 1), 0.5, dtype=np.float32)
            vy = np.full((2, 1), 0.5, dtype=np.float32)
            client.dataset_bundle.metadata.update(
                {"sampling_rate_hz": 100, "window_size": 2, "stride": 1}
            )
            client.dataset_bundle.test = DataSplit(
                inputs=np.zeros((2, 1, 1), dtype=np.float32),
                targets={"velx": vx, "vely": vy},
                metadata={},
            )

            with self.assertRaisesRegex(ValueError, "odometry-specific"):
                client.trajectory_metrics_and_plots(
                    checkpoint_path=base / "ckpt.keras",
                    plot_dir=base,
                    study_name="demo",
                )

    def test_require_trajectory_split_requires_velocity_targets(self) -> None:
        # Trajectory split views should require velocity targets so the plotted channels remain meaningful.
        """Validate require trajectory split requires velocity targets."""
        cases = (
            {"class_id": np.array([0, 1], dtype=np.int32)},
            {"velx": None, "vely": np.zeros((2, 1), dtype=np.float32)},
            {"velx": np.zeros((2, 1), dtype=np.float32), "vely": None},
            ["not", "a", "mapping"],
        )
        for targets in cases:
            with self.subTest(targets=targets):
                client = _build_test_client()
                client.dataset_bundle.test = DataSplit(
                    inputs=np.zeros((2, 1, 1), dtype=np.float32),
                    targets=targets,
                    metadata={"size_of_each": [2], "x0": [0.0], "y0": [0.0]},
                )
                with self.assertRaisesRegex(ValueError, "velocity targets named 'velx' and 'vely'"):
                    client._require_trajectory_split()


class SummaryBundleTests(unittest.TestCase):
    """Summary bundle aggregation should fuse existing artifacts."""

    def test_write_summary_bundle_persists_expected_fields(self) -> None:
        # Summary bundles should persist the expected metadata fields so later reporting code can read one stable artifact format.
        """Validate write summary bundle persists expected fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            history_path = base / "history.json"
            history_path.write_text("{}")
            loss_plots = {"loss_plot": str(base / "loss.png")}
            test_metrics = {"checkpoint_path": str(base / "ckpt.keras"), "tflite_path": None}
            closeout_artifacts = {"ate_mean": 0.1}

            class DummyStudy:
                """Dummy Optuna study used by NAS client tests."""

                def __init__(self):
                    """Initialize the instance."""
                    self.best_trial = SimpleNamespace(
                        params={"nb_filters": 8, "quantization_mode": "int8_ptq", "cpu_clock_mhz_index": 1}
                    )

            with patch("nas_model_client.optuna.load_study", return_value=DummyStudy()):
                summary_path = client.write_summary_bundle(
                    study_storage="sqlite:///dummy.db",
                    study_name="demo",
                    history_path=history_path,
                    loss_plots=loss_plots,
                    test_metrics=test_metrics,
                    closeout_artifacts=closeout_artifacts,
                    summary_path=base / "summary.json",
                )

            self.assertTrue(summary_path.is_file())
            content = json.loads(summary_path.read_text())
            self.assertEqual(content["best_params"], {"nb_filters": 8})
            self.assertEqual(content["quantization_mode"], "int8_ptq")
            self.assertEqual(content["loss_plots"], loss_plots)
            self.assertEqual(content["test_metrics"], test_metrics)
            self.assertEqual(content["task_closeout_artifacts"], closeout_artifacts)


class CloseTests(unittest.TestCase):
    """Verify resources are released when NASModelClient.close is called."""

    def test_close_shuts_socket_and_context(self) -> None:
        """Closing the client should close the socket and terminate the context.

        This protects long test runs from leaking file descriptors, so the unit
        test just confirms we call the underlying pyzmq cleanup hooks.
        """
        # Closing the NAS client should shut down both the socket and the ZeroMQ context cleanly.
        client = _build_test_client()
        socket = client.socket
        context = client.context
        client.close()
        socket.close.assert_called_once_with(linger=0)
        context.term.assert_called_once()
        self.assertIsNone(client.socket)
        self.assertIsNone(client.context)

    def test_close_tolerates_unopened_hil_socket(self) -> None:
        """Closing before any HIL request should be a no-op."""
        client = _build_test_client()
        client.socket = None
        client.context = None

        client.close()

        self.assertIsNone(client.socket)
        self.assertIsNone(client.context)


if __name__ == "__main__":
    unittest.main()
