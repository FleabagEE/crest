"""Exercise actual pure policy functions without importing the TensorFlow stack."""
import ast
import json
from unittest.mock import Mock
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import unittest
from addict import Dict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "NONNEGATIVE_METRICS", "INFRASTRUCTURE_SCORE_METRICS", "DEFAULT_TASK_SCORE_METRICS",
    "FEASIBILITY_EQUALITY_EPSILON", "FeasibilityEvaluation", "_resolve_nonnegative_metric_names",
    "_metric_unavailable", "_resolve_metric_value", "_typed_reference_value",
    "_signed_feasibility_constraint", "evaluate_feasibility_rules",
    "_normalize_json_safe", "_record_measurement_evidence",
}
tree = ast.parse((ROOT / "src/crest/model.py").read_text())
selected = [node for node in tree.body if getattr(node, "name", None) in NAMES or (
    isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in NAMES for t in node.targets))]
namespace = dict(Dict=Dict, Any=Any, dataclass=dataclass, np=np, Path=Path, json=json, TrialLike=Any)
exec(compile(ast.Module(body=selected, type_ignores=[]), "model_policy_subset", "exec"), namespace)


class MeasurementFeasibilityTests(unittest.TestCase):
    def test_explicit_policy_rejects_invalid_and_missing_measurements(self):
        """Explicit measurement policy must reject missing or invalid acceptance evidence."""
        score = Dict(type="multi-objective", params=dict(objectives=[
            dict(metric="energy_mj_per_inference", direction="minimize")]))
        feasibility = Dict(train_if_infeasible=False, rules=[dict(
            rule="measured_deployment_eligibility",
            metric="measurement_deployment_eligible",
            condition="lt", reference=dict(type="literal", value=1.0),
            reason="Measured deployment is ineligible")])
        for value, expected in [(True, True), (False, False), (None, False),
                                (-1, False), (float("nan"), False)]:
            with self.subTest(value=value):
                metrics = {"latency_ms": 5, "latency_budget_ms": 500}
                if value is not None:
                    metrics["measurement_deployment_eligible"] = value
                result = namespace["evaluate_feasibility_rules"](
                    metrics, Dict(flops=10), score, feasibility)
                self.assertEqual(result.feasible, expected)
                if not expected:
                    self.assertEqual(result.status, "infeasible")
                    self.assertEqual(result.first_violation["rule"],
                                     "measured_deployment_eligibility")

    def test_evidence_persisted_in_trial_attributes_and_csv_columns(self):
        """Measurement evidence must survive in both trial attributes and CSV columns."""
        trial = Mock()
        metrics = {
            "measurement_deployment_eligible": False,
            "measurement_record_path": Path("/server/attempt.json"),
            "measurement_record_paths": ["/server/base.json", "/server/cadence.json"],
            "measurement_request_record": "/server/request.json",
            "measurement_instrumentation_reasons": [],
            "measurement_infeasibility_reasons": ["deadline_miss"],
            "other": 5,
        }
        columns = namespace["_record_measurement_evidence"](metrics, trial)
        attrs = {args[0]: args[1] for args, kwargs in trial.set_user_attr.call_args_list}
        self.assertEqual(attrs["measurement_record_path"], "/server/attempt.json")
        self.assertEqual(attrs["measurement_infeasibility_reasons"], ["deadline_miss"])
        self.assertFalse(attrs["measurement_deployment_eligible"])
        self.assertEqual(json.loads(columns["metric__measurement_record_paths"]),
                         ["/server/base.json", "/server/cadence.json"])
        self.assertNotIn("other", attrs)

    def test_metric_registered_for_policy(self):
        """The measurement eligibility metric must be available to feasibility policies."""
        for name in ["NONNEGATIVE_METRICS", "INFRASTRUCTURE_SCORE_METRICS"]:
            self.assertIn("measurement_deployment_eligible", namespace[name])

if __name__ == "__main__":
    unittest.main()
