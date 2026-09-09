"""Focused safety tests for composite synthesis, preferences, and Agent traces."""
import unittest

from agent.langgraph_agent import LangGraphFinancialAgent
from agent.planner import FinancialPlanner
from agent.schemas import MCPToolResult
from agent.skills import get_skill
from agent.streaming import AgentStreamingAdapter


class FakeQA:
    def query(self, query):
        yield "贵州茅台营业收入为 100 元，五粮液营业收入为 80 元。", False
        yield "", True


class FakeSynthesizer:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def synthesize(self, payload):
        self.calls.append(payload)
        return self.answer, 0.02


class FakeMCP:
    def __init__(self, market_success=True, news_results=None, news_success=True):
        self.market_success = market_success
        self.news_success = news_success
        self.news_results = news_results if news_results is not None else [
            {
                "title": "测试新闻", "source": "测试来源",
                "published_at": "2026-01-01T00:00:00+00:00",
                "url": "https://example.test/news", "provider": "test",
            }
        ]

    def call_tool(self, server_name, tool_name, arguments):
        if server_name == "market":
            payload = {
                "company_name": arguments["company_name"], "ticker": arguments["ticker"],
                "price": 20.0 if self.market_success else None, "change_percent": 1.0 if self.market_success else None,
                "currency": "CNY", "market_time": "2026-01-01", "source": "测试行情",
                "success": self.market_success, "error": None if self.market_success else "market unavailable",
            }
        else:
            payload = {
                "results": self.news_results,
                "success": self.news_success,
                "error": None if self.news_success else "news unavailable",
            }
        return MCPToolResult(tool_name, server_name, arguments, payload, payload["success"], payload["error"], 0.01)


class FakePreferenceStore:
    def __init__(self):
        self.data = {}
        self.write_count = 0

    def get(self, user_id):
        return self.data.get(user_id, {"preferred_companies": [], "preferred_metrics": []})

    def save_explicit(self, user_id, query, companies):
        if "以后" not in query or "关注" not in query:
            return None
        metrics = []
        if "营业收入" in query:
            metrics.append("revenue")
        if "归母净利润" in query:
            metrics.append("net_profit")
        payload = {"preferred_companies": [item["ticker"] for item in companies], "preferred_metrics": metrics}
        self.data[user_id] = payload
        self.write_count += 1
        return payload


class AgentFinalTests(unittest.TestCase):
    def make_agent(self, answer="财务表现：营业收入为 100 元。市场表现：价格为 20 CNY。近期事件：测试新闻（测试来源，2026-01-01T00:00:00+00:00）。", **kwargs):
        return LangGraphFinancialAgent(
            qa_system=kwargs.pop("qa_system", FakeQA()),
            mcp_client=kwargs.pop("mcp_client", FakeMCP()),
            synthesizer=kwargs.pop("synthesizer", FakeSynthesizer(answer)),
            preference_store=kwargs.pop("preference_store", FakePreferenceStore()),
            **kwargs,
        )

    def test_composite_uses_synthesis_and_trace(self):
        synthesizer = FakeSynthesizer("财务表现：营业收入为 100 元。市场表现：价格为 20 CNY。近期事件：测试新闻（测试来源，2026-01-01T00:00:00+00:00）。")
        state = self.make_agent(synthesizer=synthesizer).run("比较贵州茅台和五粮液2026H1财务表现、最近股价和新闻", "a")
        self.assertEqual(1, len(synthesizer.calls))
        self.assertEqual("passed", state["guardrail_status"])
        self.assertIn("synthesis", state["trace"]["graph_path"])
        self.assertIn("synthesis", state["trace"]["per_tool_latency"])
        self.assertEqual("company_comparison", state["skill"])

    def test_successful_external_tools_override_financial_local_disclaimer(self):
        synthesizer = FakeSynthesizer(
            "财务表现：营业收入为 100 元。"
            "上下文中未提供当前行情和近期新闻相关信息，因此无法展开分析。"
            "市场表现：价格为 20 CNY。"
            "近期事件：测试新闻（测试来源，2026-01-01T00:00:00+00:00）。"
        )
        state = self.make_agent(synthesizer=synthesizer).run(
            "结合比亚迪2026H1财报、当前行情和近期新闻分析其经营表现", "all-success"
        )
        self.assertEqual("passed", state["guardrail_status"])
        self.assertNotIn("未提供当前行情", state["final_answer"])
        self.assertNotIn("近期新闻相关信息", state["final_answer"])
        self.assertIn("市场表现", state["final_answer"])
        self.assertIn("近期事件", state["final_answer"])
        self.assertTrue(synthesizer.calls[0]["tool_availability"]["market_mcp"])
        self.assertTrue(synthesizer.calls[0]["tool_availability"]["news_mcp"])

    def test_financial_only_skips_synthesis(self):
        synthesizer = FakeSynthesizer("不应调用")
        state = self.make_agent(synthesizer=synthesizer).run("贵州茅台2026H1营业收入是多少？", "a")
        self.assertEqual([], synthesizer.calls)
        self.assertEqual("贵州茅台营业收入为 100 元，五粮液营业收入为 80 元。", state["final_answer"])

    def test_failed_market_cannot_create_price(self):
        state = self.make_agent(
            answer="市场表现：价格为 999 CNY。",
            mcp_client=FakeMCP(market_success=False),
        ).run("比较贵州茅台和五粮液2026H1财务表现、最近股价", "a")
        self.assertIn("market_unavailable", state["guardrail_status"])
        self.assertNotIn("999", state["final_answer"])

    def test_empty_news_cannot_create_event(self):
        state = self.make_agent(
            answer="近期事件：凭空新闻（虚构来源，2026-01-01）。",
            mcp_client=FakeMCP(news_results=[]),
        ).run("比较贵州茅台和五粮液2026H1财务表现和最近新闻", "a")
        self.assertIn("news_unavailable", state["guardrail_status"])
        self.assertNotIn("凭空新闻", state["final_answer"])

    def test_actual_news_failure_keeps_market_and_safe_degradation(self):
        state = self.make_agent(
            answer="市场表现：价格为 20 CNY。近期事件：凭空新闻（虚构来源，2026-01-01）。",
            mcp_client=FakeMCP(news_results=[]),
        ).run("结合贵州茅台2026H1财报、当前行情和近期新闻分析", "news-failure")
        self.assertIn("news_unavailable", state["guardrail_status"])
        self.assertIn("市场表现", state["final_answer"])
        self.assertIn("News MCP 未返回可用新闻", state["final_answer"])
        self.assertNotIn("凭空新闻", state["final_answer"])

    def test_real_shape_failed_news_payload_still_emits_degradation(self):
        agent = self.make_agent(
            mcp_client=FakeMCP(news_results=[], news_success=False),
            answer="市场表现：价格为 20 CNY。近期事件：凭空新闻。",
        )
        events = list(AgentStreamingAdapter(agent).iter_events({
            "query": "结合贵州茅台2026H1财报、当前行情和近期新闻分析",
            "thread_id": "real-shape-news-failure", "user_id": "test-user",
        }))

        self.assertEqual(
            "degraded",
            next(event["data"]["status"] for event in events if event["type"] == "guardrail"),
        )
        answer = "".join(event["data"]["content"] for event in events if event["type"] == "token")
        self.assertIn("News MCP 未返回可用新闻", answer)
        self.assertNotIn("凭空新闻", answer)

    def test_real_shape_successful_tools_do_not_emit_external_degradation(self):
        class LocalDisclaimerQA:
            def query(self, _query):
                yield "财报上下文没有当前行情和近期新闻。营业收入为 100 元。", False
                yield "", True

        # This mirrors the real GPU failure mode: synthesis adds an unsupported
        # number, so the guardrail falls back even though all tool payloads are
        # real-shape, successful, and provenance-complete.
        agent = self.make_agent(
            qa_system=LocalDisclaimerQA(),
            answer="财务表现：营业收入为 999 元。市场表现：价格为 20 CNY。近期事件：测试新闻（测试来源，2026-01-01T00:00:00+00:00）。",
        )
        events = list(AgentStreamingAdapter(agent).iter_events({
            "query": "结合比亚迪2026H1财报、当前行情和近期新闻分析其经营表现",
            "thread_id": "real-shape-success", "user_id": "test-user",
        }))
        state = agent.run(
            "结合比亚迪2026H1财报、当前行情和近期新闻分析其经营表现",
            "real-shape-state", "test-user",
        )

        self.assertEqual("unsupported_numeric_claim", state["guardrail_status"])
        self.assertIn("市场表现", state["final_answer"])
        self.assertIn("近期事件", state["final_answer"])
        self.assertNotIn("没有当前行情", state["final_answer"])
        self.assertEqual(
            "passed",
            next(event["data"]["status"] for event in events if event["type"] == "guardrail"),
        )

    def test_unverified_calculation_falls_back(self):
        state = self.make_agent(
            answer="两家公司同比增长率为 50%。",
        ).run("比较贵州茅台和五粮液2026H1财务表现、最近新闻和同比增长率", "a")
        self.assertIn("unverified_calculation", state["guardrail_status"])
        self.assertIn("不自行推导", state["final_answer"])

    def test_skills_and_explicit_preferences(self):
        self.assertEqual("company_comparison", FinancialPlanner().plan("比较贵州茅台和五粮液营收").skill)
        self.assertIn("market_mcp", get_skill("market_intelligence").required_capabilities)
        preferences = FakePreferenceStore()
        agent = self.make_agent(preference_store=preferences)
        agent.run("以后比较公司时，我主要关注营业收入和归母净利润。", "first", user_id="user-a")
        following = agent.run("比较贵州茅台和五粮液2026H1营业收入", "second", user_id="user-a")
        self.assertEqual(["revenue", "net_profit"], following["long_term_preferences"]["preferred_metrics"])
        self.assertEqual(1, preferences.write_count)
        agent.run("贵州茅台2026H1营业收入是多少？", "third", user_id="user-b")
        self.assertEqual(1, preferences.write_count)
