"""Focused LangGraph orchestration and RedisSaver tests."""
import unittest

from agent.langgraph_agent import LangGraphFinancialAgent
from agent.schemas import MCPToolResult


class FakeQA:
    def query(self, query):
        yield f"RAG:{query}", False
        yield "", True


class FakeMCP:
    def call_tool(self, server_name, tool_name, arguments):
        payload = {"success": True, "error": None, "company_name": arguments["company_name"], "ticker": arguments["ticker"]}
        if server_name == "news": payload["results"] = [{"title": f"{arguments['company_name']}新闻"}]
        else: payload.update({"price": 100.0, "change_percent": 1.0})
        return MCPToolResult(tool_name, server_name, arguments, payload, True, None, 0.01)


class LangGraphAgentTests(unittest.TestCase):
    def make_agent(self, saver=None):
        return LangGraphFinancialAgent(qa_system=FakeQA(), mcp_client=FakeMCP(), checkpointer=saver)

    def test_financial_and_composite_paths(self):
        agent = self.make_agent()
        financial = agent.run("比较贵州茅台和五粮液2026H1营业收入", "financial")
        self.assertEqual(["financial_rag"], financial["executed_tools"])
        composite = agent.run("比较贵州茅台和五粮液2026H1财务表现，再看看最近股价和新闻", "composite")
        self.assertEqual(["financial_rag", "market_mcp", "news_mcp"], composite["executed_tools"])

    def test_same_thread_follow_up_and_thread_isolation(self):
        agent = self.make_agent()
        agent.run("比较贵州茅台和五粮液2026H1营业收入", "a")
        follow_up = agent.run("那他们最近有什么新闻？", "a")
        self.assertEqual(["600519", "000858"], follow_up["tickers"])
        no_context = agent.run("那他们最近有什么新闻？", "b")
        self.assertEqual([], no_context["companies"])
        self.assertTrue(any("缺少公司上下文" in error for error in no_context["errors"]))

    def test_redis_saver_cross_instance_recovery(self):
        saver = LangGraphFinancialAgent.redis_checkpointer()
        self.make_agent(saver).run("比较贵州茅台和五粮液2026H1营业收入", "redis-test")
        recovered = self.make_agent(LangGraphFinancialAgent.redis_checkpointer()).run("那他们最近有什么新闻？", "redis-test")
        self.assertEqual(["600519", "000858"], recovered["tickers"])
