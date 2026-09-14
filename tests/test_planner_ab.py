"""Offline checks of request budgeting and fixed-context A/B isolation."""
import copy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from agent.planner import FinancialPlanner
from agent.planning import create_planner
from evaluations.run_planner_ab import RequestBudget, context_snapshots, run_case, smoke_cases, summarize
from tests.test_function_calling_planner import FakeClient, completion, tool


class PlannerABTests(unittest.TestCase):
    def setUp(self):
        p = patch.object(socket.socket, "connect", side_effect=AssertionError("offline only"))
        p.start(); self.addCleanup(p.stop)

    def test_budget_is_persistent_and_does_not_allow_overrun(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            RequestBudget(path, 2, 1).reserve()
            second = RequestBudget(path, 2, 2)
            second.reserve()
            with self.assertRaises(RuntimeError): second.reserve()
            self.assertEqual(2, json.loads(path.read_text())["attempted_requests"])

    def test_budget_local_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = RequestBudget(Path(directory)/"b.json", 160, 1)
            budget.reserve()
            with self.assertRaises(RuntimeError): budget.reserve()

    def test_context_does_not_read_gold_or_predictions(self):
        cases = smoke_cases()[:2]
        cases[1]["session_id"] = cases[0]["session_id"]
        original = context_snapshots(cases)
        changed = copy.deepcopy(cases)
        changed[0]["expected_company_codes"] = ["999999"]
        changed[0]["expected_report_periods"] = ["wrong"]
        changed[0]["expected_tools"] = ["invented"]
        self.assertEqual(original, context_snapshots(changed))
        self.assertEqual("600519", original[1]["companies"][0]["ticker"])

    def test_offline_smoke_runner_rule_and_metrics(self):
        cases = smoke_cases()
        rows = [run_case(case, snapshot, FinancialPlanner()) for case, snapshot in zip(cases, context_snapshots(cases))]
        summary = summarize(rows)
        self.assertEqual(0, summary["planner_diagnostics"]["request_count"])
        self.assertIsNone(summary["planner_diagnostics"]["argument_accuracy"]["value"])
        self.assertEqual(8, summary["legacy_v1_metrics"]["completed"])

    def test_fc_runner_exact_delivery_and_serialization(self):
        client = FakeClient([completion([tool("market_mcp", {"ticker": "002594"})])])
        planner = create_planner(mode="function_calling", model="qwen3.8-max", client=client, fallback_to_rule=False)
        row = run_case(smoke_cases()[1], {}, planner)
        self.assertTrue(row["argument_delivery_exact"])
        self.assertTrue(row["delivered_target_exact"])
        self.assertEqual(1, summarize([row])["planner_diagnostics"]["request_count"])
        json.dumps(row, allow_nan=False)

    def test_constructor_explicit_options_override_config(self):
        from base.config import config
        with patch.object(config, "AGENT_PLANNER_MODE", "function_calling"):
            self.assertIsInstance(create_planner(mode="rule"), FinancialPlanner)
        with patch.object(config, "AGENT_PLANNER_TIMEOUT", 1):
            planner = create_planner(mode="function_calling", timeout=12, max_calls=3,
                                     model="qwen3.8-max", client=FakeClient([]))
            self.assertEqual(12, planner.settings.timeout)
            self.assertEqual(3, planner.settings.max_calls)


if __name__ == "__main__": unittest.main()
