"""Replay the failed public Composite; all execution is offline."""
import copy
import json
from decimal import Decimal
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

from agent.langgraph_agent import LangGraphFinancialAgent
from agent.numeric_guardrail import _evidence, _resolve_amount_type, amount_tokens
from agent.streaming import AgentStreamingAdapter


class SavedGuardrailAgent:
    def __init__(self, state):
        self.state = state
        self.final_state = None

    def stream(self, query, thread_id, user_id):
        agent = object.__new__(LangGraphFinancialAgent)
        events = []
        agent._emit = lambda kind, **data: events.append({"kind": kind, "data": data})
        self.final_state = {**self.state, **agent._guardrail(self.state)}
        yield from events
        yield {"kind": "complete", "state": self.final_state}


class ReleaseBlockerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = json.loads((Path(__file__).parent / "fixtures/release_blocker_public_e2e.json").read_text())

    def setUp(self):
        for method in ("connect", "connect_ex"):
            blocker = patch.object(socket.socket, method, side_effect=AssertionError("offline only"))
            blocker.start()
            self.addCleanup(blocker.stop)
        self.agent = object.__new__(LangGraphFinancialAgent)
        self.agent._emit = lambda *a, **kw: None

    def check_number(self, answer, supported, state=None):
        flags = self.agent._guardrail_violations(state if state is not None else self.saved, answer)
        self.assertEqual(not supported, "unsupported_numeric_claim" in flags, (answer, flags))

    def test_postposed_financial_metric_classification(self):
        token = amount_tokens("实现了约907.03亿元的营业收入。")[0]
        self.assertEqual("financial_amount", token.semantic_type)
        self.assertEqual(Decimal("90703000000"), token.value)
        self.check_number("实现了约907.03亿元的营业收入。", True)

    def test_unit_only_approximation_uses_matching_financial_evidence(self):
        evidence, _ = _evidence(self.saved)
        claim = amount_tokens("约907.03亿元")[0]
        resolved = _resolve_amount_type(claim, evidence)
        self.assertEqual("financial_amount", resolved.semantic_type)
        self.check_number("约907.03亿元", True)

    def test_approximation_outside_tolerance_fails(self):
        self.check_number("约950亿元", False)
        self.check_number("实现了约950亿元的营业收入。", False)

    def test_approximation_without_financial_evidence_fails(self):
        state = copy.deepcopy(self.saved)
        state["tool_results"] = [{"tool_name": "financial_rag", "success": True, "answer": "未提供验证金额。"}]
        state["news_results"] = []
        self.check_number("约907.03亿元", False, state)

    def test_supported_amount_units_share_existing_tolerance(self):
        for answer in ("约90703260964元", "约9070326万元", "约0.090703万亿元"):
            with self.subTest(answer=answer):
                self.check_number(answer, True)

    def test_no_approximation_for_exact_or_unitless_claims(self):
        self.check_number("907.03亿元", False)
        self.check_number("约907.03", False)
        self.check_number("约907.03万元", False)

    def test_threshold_and_percent_remain_strict(self):
        financial_only = copy.deepcopy(self.saved)
        financial_only["news_results"] = []
        financial_only["market_results"] = []
        self.check_number("超907亿元", True, financial_only)
        self.check_number("超1000亿元", False, financial_only)
        self.check_number("涨跌幅为0.22%", True)
        self.check_number("涨跌幅为22%", False)

    def test_price_cannot_borrow_financial_approximation(self):
        self.check_number("股价约1280元", False)
        self.check_number("约1280元的股价", False)
        self.check_number("约1280元", False)
        self.check_number("股价1277.96元", True)

    def test_saved_public_composite_retains_synthesis(self):
        self.assertEqual("unsupported_numeric_claim", self.saved["guardrail_status"])
        output = self.agent._guardrail(copy.deepcopy(self.saved))
        self.assertEqual("passed", output["guardrail_status"])
        self.assertEqual(self.saved["synthesis_answer"], output["final_answer"])

    def test_real_guardrail_pass_matches_websocket(self):
        agent = SavedGuardrailAgent(copy.deepcopy(self.saved))
        events = list(AgentStreamingAdapter(agent).iter_events({"query": self.saved["query"]}))
        self.assertEqual("passed", agent.final_state["guardrail_status"])
        self.assertEqual(["passed"], [e["data"]["status"] for e in events if e["type"] == "guardrail"])
        self.assertEqual(self.saved["synthesis_answer"], "".join(e["data"]["content"] for e in events if e["type"] == "token"))

    def test_real_guardrail_replacement_is_public_error(self):
        state = copy.deepcopy(self.saved)
        state["synthesis_answer"] += "营业收入约950亿元。"
        agent = SavedGuardrailAgent(state)
        events = list(AgentStreamingAdapter(agent).iter_events({"query": state["query"]}))
        self.assertIn("unsupported_numeric_claim", agent.final_state["guardrail_status"])
        self.assertNotEqual(state["synthesis_answer"], agent.final_state["final_answer"])
        self.assertEqual(["error"], [e["data"]["status"] for e in events if e["type"] == "guardrail"])

    def test_final_state_overrides_stale_guardrail_notification(self):
        for status, expected in (("passed", "passed"), ("news_unavailable", "degraded"),
                                 ("unsupported_numeric_claim", "error"),
                                 ("news_unavailable,unsupported_numeric_claim", "error")):
            with self.subTest(status=status):
                class Events:
                    def stream(self, *a, **kw):
                        yield {"kind": "guardrail", "data": {"status": "passed"}}
                        yield {"kind": "complete", "state": {"guardrail_status": status, "final_answer": "安全答复"}}
                events = list(AgentStreamingAdapter(Events()).iter_events({"query": "测试"}))
                self.assertEqual([expected], [e["data"]["status"] for e in events if e["type"] == "guardrail"])
                self.assertTrue(all(set(e)=={"type", "data", "timestamp"} for e in events))

    def test_missing_final_status_cannot_default_to_pass(self):
        class Events:
            def stream(self, *a, **kw):
                yield {"kind": "guardrail", "data": {"status": "passed"}}
                yield {"kind": "complete", "state": {"final_answer": "must not leak"}}
        events = list(AgentStreamingAdapter(Events()).iter_events({"query": "测试"}))
        self.assertEqual(["start", "error"], [e["type"] for e in events])


if __name__ == "__main__":
    unittest.main()
