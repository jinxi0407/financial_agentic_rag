"""Offline FC integration tests. No model, provider, Redis or RAG services."""
import json
import math
import socket
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from agent.function_calling_planner import FunctionCallingPlanner
from agent.langgraph_agent import LangGraphFinancialAgent
from agent.planner import FinancialPlanner
from agent.planning import PlannerSettings, planning_context, tool_schemas
from agent.schemas import MCPToolResult, PlannerDecision
from agent.streaming import AgentStreamingAdapter
from evaluations.run_real_user_paraphrase_regression import RegressionPreferenceStore, RegressionSynthesizer


def tool(name, args, identifier="call_1"):
    return NS(id=identifier, type="function", function=NS(name=name, arguments=json.dumps(args, ensure_ascii=False)))


def completion(calls=None, content=None, finish=None):
    return NS(choices=[NS(message=NS(tool_calls=calls, content=content), finish_reason=finish or ("tool_calls" if calls else "stop"))],
              usage=NS(prompt_tokens=11, completion_tokens=7, total_tokens=18))


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.options = []
        self.chat = NS(completions=self)

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception): raise response
        return response


class SpyQA:
    def __init__(self): self.calls = []

    def query(self, query):
        self.calls.append(query)
        yield "财报证据已提供。", True


class SpyMCP:
    def __init__(self, fail_ticker=None):
        self.calls = []
        self.fail_ticker = fail_ticker

    def call_tool(self, server, name, args):
        self.calls.append((server, name, dict(args)))
        success = args.get("ticker") != self.fail_ticker
        data = {"success": success, "company_name": args.get("company_name"), "ticker": args.get("ticker"),
                "price": 20, "source": "fixture", "currency": "CNY"}
        if server == "news": data["results"] = ([{"title": "固定新闻", "source": "fixture", "published_at": "2026-09-01", "url": "https://example.test/news"}] if success else [])
        return MCPToolResult(name, server, args, data, success, None if success else "fixture_timeout", 0.01)


class FunctionCallingTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def planner(self, responses, fallback=False, max_calls=8):
        client = FakeClient(responses)
        return FunctionCallingPlanner(PlannerSettings(mode="function_calling", model="qwen3.8-max", fallback_to_rule=fallback, max_calls=max_calls), client=client), client

    def agent(self, responses, fallback=False, fail_ticker=None):
        planner, client = self.planner(responses, fallback)
        qa, mcp = SpyQA(), SpyMCP(fail_ticker)
        agent = LangGraphFinancialAgent(planner=planner, qa_system=qa, mcp_client=mcp,
                                        preference_store=RegressionPreferenceStore(), synthesizer=RegressionSynthesizer())
        return agent, client, qa, mcp

    def test_four_single_tools_receive_exact_arguments(self):
        fixtures = [
            ("贵州茅台2026H1营业收入", "financial_rag", {"query": "贵州茅台 2026H1 营业收入是多少？"}),
            ("茅台股价", "market_mcp", {"ticker": "600519"}),
            ("茅台新闻", "news_mcp", {"query": "贵州茅台近期新闻", "ticker": "600519", "max_results": 3}),
            ("1+9*3是多少", "calculator", {"operation": "expression", "expression": "1 + 9 * 3"}),
        ]
        for query, name, args in fixtures:
            with self.subTest(name=name):
                agent, client, qa, mcp = self.agent([completion([tool(name, args)])])
                with patch.object(agent.calculator, "run", wraps=agent.calculator.run) as calc:
                    state = agent.run(query, "one")
                self.assertEqual([name], state["executed_tools"])
                self.assertEqual(args, state["call_results"][0]["validated_arguments"])
                if name == "financial_rag": self.assertEqual([args["query"]], qa.calls)
                elif name == "calculator": self.assertEqual(args, calc.call_args.kwargs)
                else: self.assertEqual(args, mcp.calls[0][2])
                self.assertEqual(1, len(client.requests))

    def test_native_request_options_and_usage(self):
        planner, client = self.planner([completion([tool("calculator", {"operation": "expression", "expression": "3+4"})])])
        decision = planner.plan("3+4")
        request = client.requests[0]
        self.assertEqual("auto", request["tool_choice"])
        self.assertTrue(request["parallel_tool_calls"])
        self.assertFalse(request["stream"])
        self.assertFalse(request["extra_body"]["enable_thinking"])
        self.assertEqual(0, client.options[0]["max_retries"])
        self.assertEqual(18, decision.planner_metadata["token_usage"]["total_tokens"])
        self.assertEqual(1, decision.planner_metadata["request_count"])

    def test_three_tools_composite_fixed_execution_order(self):
        calls = [tool("news_mcp", {"query": "比亚迪新闻", "ticker": "002594"}, "n"),
                 tool("market_mcp", {"ticker": "002594"}, "m"),
                 tool("financial_rag", {"query": "比亚迪2026H1营业收入"}, "r")]
        agent, _, qa, mcp = self.agent([completion(calls)])
        state = agent.run("比亚迪2026H1营业收入、股价和新闻", "a")
        self.assertEqual("composite_query", state["intent"])
        self.assertEqual(["r", "m", "n"], [r["call_id"] for r in state["call_results"]])
        self.assertEqual("passed", state["guardrail_status"])

    def test_calculator_market_composite_uses_model_expression(self):
        calls = [tool("calculator", {"operation": "expression", "expression": "3 + 4"}, "c"), tool("market_mcp", {"ticker": "600519"}, "m")]
        agent, _, _, mcp = self.agent([completion(calls)])
        with patch.object(FinancialPlanner, "simple_calculation_request", side_effect=AssertionError("must not reparse")):
            state = agent.run("3+4=多少，还有茅台股票", "a")
        self.assertEqual(7, state["calculation_result"]["result"])
        self.assertEqual(["market_mcp", "calculator"], state["executed_tools"])

    def test_same_tool_multiple_market_targets(self):
        args = [{"ticker": "600519"}, {"company_name": "五粮液", "ticker": "000858"}]
        agent, _, _, mcp = self.agent([completion([tool("market_mcp", a, f"m{i}") for i, a in enumerate(args)])])
        state = agent.run("茅台和五粮液股价", "a")
        self.assertEqual(args, [c[2] for c in mcp.calls])
        self.assertEqual(2, len(state["call_results"]))
        self.assertEqual("market_query", state["intent"])

    def test_news_multiple_targets_one_failure(self):
        args = [{"ticker": "600519", "query": "茅台公告", "max_results": 2}, {"ticker": "000858", "query": "五粮液公告", "max_results": 4}]
        agent, _, _, mcp = self.agent([completion([tool("news_mcp", a, f"n{i}") for i, a in enumerate(args)])], fail_ticker="000858")
        state = agent.run("茅台和五粮液近期公告", "a")
        self.assertEqual(args, [c[2] for c in mcp.calls])
        self.assertEqual(["success", "failed"], [c["status"] for c in state["call_results"]])
        self.assertIn("news_unavailable", state["guardrail_status"])

    def test_repeated_singleton_rejected_atomically(self):
        for name, query, args in [("financial_rag", "茅台2026H1营业收入", {"query": "茅台2026H1营业收入"}),
                                  ("calculator", "3+4", {"operation": "expression", "expression": "3+4"})]:
            agent, _, qa, mcp = self.agent([completion([tool(name, args, "a"), tool(name, args, "b")])])
            state = agent.run(query, "a")
            self.assertEqual("unsupported_plan", state["planning_status"])
            self.assertFalse(state["executed_tools"])
            self.assertFalse(qa.calls)

    def test_invalid_security_conflict_and_unknown(self):
        for args in ({"ticker": "000858", "company_name": "贵州茅台"}, {"ticker": "999999"}, {"ticker": "002594"}):
            agent, _, _, mcp = self.agent([completion([tool("market_mcp", args)])])
            state = agent.run("茅台股价", "a")
            self.assertEqual("invalid_plan", state["planning_status"])
            self.assertFalse(mcp.calls)

    def test_new_thread_pronoun_is_prehandled_without_request(self):
        agent, client, _, mcp = self.agent([])
        state = agent.run("它的股票", "fresh")
        self.assertFalse(client.requests)
        self.assertFalse(mcp.calls)
        self.assertEqual("shared_safety", state["planner_metadata"]["effective_mode"])

    def test_context_is_readonly_and_same_thread_recovers(self):
        responses = [completion([tool("market_mcp", {"ticker": "600519"})]), completion([tool("news_mcp", {"ticker": "600519", "query": "茅台新闻"})])]
        agent, client, _, mcp = self.agent(responses)
        agent.run("茅台股票", "same")
        state = agent.run("那它最近新闻呢", "same")
        context = json.loads(client.requests[1]["messages"][1]["content"])
        self.assertEqual([], context["explicit_companies"])
        self.assertEqual("600519", context["session_companies"][0]["ticker"])
        self.assertEqual("600519", mcp.calls[-1][2]["ticker"])
        self.assertEqual(1, len(state["planned_calls"]))
        self.assertEqual(1, len(state["call_results"]))

    def test_explicit_company_overrides_session(self):
        agent, client, _, mcp = self.agent([completion([tool("market_mcp", {"ticker": "600519"})]), completion([tool("market_mcp", {"ticker": "002594"})])])
        agent.run("茅台股票", "a")
        state = agent.run("现在看比亚迪股价", "a")
        self.assertEqual(["002594"], state["tickers"])
        self.assertEqual("002594", mcp.calls[-1][2]["ticker"])

    def test_multiperiod_rag_preserved_and_mutation_rejected(self):
        query = "比较茅台2025H1和2026H1营业收入"
        for new_query, expected in [(query, "ready"), ("茅台2025FY营业收入", "invalid_plan"), ("茅台2025H1营业收入", "invalid_plan")]:
            agent, _, _, _ = self.agent([completion([tool("financial_rag", {"query": new_query})])])
            state = agent.run(query, "a")
            self.assertEqual(expected, state["planning_status"])

    def test_definition_without_company_period(self):
        agent, _, qa, _ = self.agent([completion([tool("financial_rag", {"query": "什么是流动比率？"})])])
        state = agent.run("什么是流动比率？", "a")
        self.assertEqual(["什么是流动比率？"], qa.calls)
        self.assertEqual("ready", state["planning_status"])

    def test_financial_metric_scope_preserved(self):
        for query, changed in [("茅台2026H1研发投入", "茅台2026H1研发费用"), ("茅台2026H1营业收入", "茅台2026H1营业总收入"), ("茅台2026H1行业毛利率", "茅台2026H1毛利率")]:
            agent, _, qa, _ = self.agent([completion([tool("financial_rag", {"query": changed})])])
            self.assertEqual("invalid_plan", agent.run(query, "a")["planning_status"])
            self.assertFalse(qa.calls)

    def test_no_tool_is_valid_but_model_financial_text_is_not_exposed(self):
        for content in ("您好", '{"tools":["market_mcp"]}', "茅台股价为999999"):
            planner, _ = self.planner([completion(content=content)])
            decision = planner.plan("茅台股票")
            self.assertEqual("no_tool", decision.planning_status)
            self.assertTrue(decision.planner_metadata["api_success"])
            self.assertNotIn("999999", decision.no_tool_response)
            self.assertEqual((), decision.tools)

    def test_greeting_no_tool(self):
        agent, _, _, _ = self.agent([completion(content="你好")])
        self.assertEqual("greeting", agent.run("你好", "a")["intent"])

    def test_empty_truncated_and_api_error(self):
        for response, status in [(completion(), "empty_response"), (completion(content="x", finish="length"), "truncated")]:
            planner, _ = self.planner([response])
            self.assertEqual(status, planner.plan("茅台股票").planning_status)

    def test_api_exception_redaction_and_missing_usage(self):
        planner, client = self.planner([TimeoutError("secret must not be recorded")])
        decision = planner.plan("茅台股票")
        self.assertEqual("api_error", decision.planning_status)
        self.assertNotIn("secret", str(decision.to_dict()))
        self.assertEqual(1, len(client.requests))
        response = completion(content="no tool"); response.usage = None
        planner, _ = self.planner([response])
        self.assertIsNone(planner.plan("你好").planner_metadata["token_usage"])

    def test_unknown_extra_invalid_json_duplicate_ids(self):
        bad_json = tool("market_mcp", {}); bad_json.function.arguments = "{broken"
        cases = [[tool("unknown", {})], [tool("market_mcp", {"ticker": "600519", "endpoint": "x"})],
                 [bad_json], [tool("market_mcp", {"ticker": "600519"}, "a"), tool("market_mcp", {"ticker": "600519"}, "a")]]
        for calls in cases:
            agent, _, _, mcp = self.agent([completion(calls)])
            self.assertEqual("invalid_plan", agent.run("茅台股票", "a")["planning_status"])
            self.assertFalse(mcp.calls)

    def test_total_call_limit_no_truncation(self):
        calls = [tool("market_mcp", {"ticker": "600519"}, f"c{i}") for i in range(9)]
        agent, _, _, mcp = self.agent([completion(calls)])
        self.assertEqual("unsupported_plan", agent.run("茅台股票", "a")["planning_status"])
        self.assertFalse(mcp.calls)

    def test_partial_invalid_plan_executes_nothing(self):
        agent, _, _, mcp = self.agent([completion([tool("market_mcp", {"ticker": "600519"}, "a"), tool("news_mcp", {"ticker": "002594", "query": "新闻"}, "b")])])
        state = agent.run("茅台股价和新闻", "a")
        self.assertFalse(mcp.calls)
        self.assertEqual([True, False], [x["valid"] for x in state["planner_metadata"]["validation_results"]])

    def test_calculator_unsafe_nonfinite_ungrounded_and_zero(self):
        for args in [{"operation": "expression", "expression": "open('x')"}, {"operation": "expression", "expression": "2**100"},
                     {"operation": "addition", "left": math.inf, "right": 1}, {"operation": "addition", "left": 999, "right": 1},
                     {"operation": "ratio", "numerator": 1, "denominator": 0}]:
            agent, _, _, _ = self.agent([completion([tool("calculator", args)])])
            self.assertEqual("invalid_plan", agent.run("1+0+2是多少", "a")["planning_status"])

    def test_existing_financial_growth_stays_in_rag(self):
        query = "茅台2025H1与2026H1营业收入同比"
        planner, _ = self.planner([completion([tool("financial_rag", {"query": query})])])
        self.assertEqual("ready", planner.plan(query).planning_status)

    def test_fallback_single_rule_plan_and_original_error(self):
        agent, client, _, mcp = self.agent([TimeoutError("hidden")], fallback=True)
        state = agent.run("茅台股票", "a")
        self.assertEqual(1, len(mcp.calls))
        self.assertTrue(state["planner_metadata"]["fallback_used"])
        self.assertEqual("rule", state["planner_metadata"]["effective_mode"])
        self.assertEqual("api_error", state["planner_metadata"]["fc_status"])
        self.assertEqual([], state["planned_calls"])

    def test_error_does_not_carry_to_next_turn(self):
        agent, _, _, _ = self.agent([TimeoutError("x"), completion([tool("market_mcp", {"ticker": "600519"})])])
        agent.run("茅台股票", "a")
        state = agent.run("它现在股价", "a")
        self.assertEqual("ready", state["planning_status"])
        self.assertNotIn("error_type", state["planner_metadata"])
        self.assertFalse(state["errors"])

    def test_period_clarification_shared_reset_rule_and_fc(self):
        for mode in ("rule", "function_calling"):
            agent, _, qa, _ = self.agent([completion([tool("financial_rag", {"query": "比较茅台和五粮液经营表现"})]),
                                         completion([tool("financial_rag", {"query": "茅台和五粮液2026H1财报"})])])
            if mode == "rule": agent.planner = FinancialPlanner()
            first = agent.run("比较茅台和五粮液经营表现", "a")
            self.assertTrue(first["period_clarification"])
            second = agent.run("2026H1", "a")
            self.assertEqual("", second["period_clarification"])
            self.assertEqual(["financial_rag"], second["executed_tools"])

    def test_legacy_schema_and_stream_events_remain_compatible(self):
        self.assertEqual({"intent", "tools", "reason", "status", "skill"}, set(PlannerDecision("unsupported", (), "x").to_dict()))
        agent, _, _, _ = self.agent([completion([tool("calculator", {"operation": "expression", "expression": "3+4"})])])
        events = list(AgentStreamingAdapter(agent).iter_events({"query": "3+4", "thread_id": "a"}))
        self.assertEqual("end", events[-1]["type"])
        self.assertTrue(all(set(event) == {"type", "data", "timestamp"} for event in events))

    def test_schema_registry_and_settings(self):
        self.assertEqual({"financial_rag", "market_mcp", "news_mcp", "calculator"}, {x["function"]["name"] for x in tool_schemas()})
        with self.assertRaises(ValueError): PlannerSettings(mode="unknown")
        with self.assertRaises(ValueError): PlannerSettings(max_calls=0)

    def test_ambiguous_multicompany_pronoun_cannot_pick_one(self):
        planner, _ = self.planner([completion([tool("market_mcp", {"ticker": "600519"})])])
        context = planning_context("它的股价", {"companies": [{"company_name":"贵州茅台", "ticker":"600519"}, {"company_name":"五粮液", "ticker":"000858"}]})
        decision = planner.plan("它的股价", context=context)
        self.assertEqual("invalid_plan", decision.planning_status)
        self.assertEqual((), decision.planned_calls)

    def test_rag_cannot_invent_numeric_condition(self):
        agent, _, qa, _ = self.agent([completion([tool("financial_rag", {"query":"茅台2026H1营业收入100亿元"})])])
        self.assertEqual("invalid_plan", agent.run("茅台2026H1营业收入", "a")["planning_status"])
        self.assertFalse(qa.calls)


if __name__ == "__main__": unittest.main()
