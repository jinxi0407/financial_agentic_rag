"""Focused Planner and Runner tests for executable MCP intents."""

import unittest

from agent.planner import FinancialPlanner
from agent.runner import FinancialAgentRunner
from agent.schemas import MCPToolResult


class FakeMCPClient:
    def __init__(self, success=True):
        self.success = success
        self.calls = []

    def call_tool(self, server_name, tool_name, arguments):
        self.calls.append((server_name, tool_name, arguments))
        payload = {"success": self.success, "error": None if self.success else "provider unavailable"}
        if server_name == "market":
            payload.update({"company_name": "贵州茅台", "ticker": "600519", "price": 100.0, "change_percent": 1.2, "currency": "CNY", "market_time": "2026-01-01T00:00:00+00:00"})
        else:
            payload.update({"results": [{"title": "比亚迪新闻", "source": "测试来源"}]})
        return MCPToolResult(tool_name, server_name, arguments, payload, self.success, payload["error"], 0.01)


class AgentMCPTests(unittest.TestCase):
    def test_planner_preserves_market_news_rag_and_calculation_routes(self):
        planner = FinancialPlanner()
        self.assertEqual("market_query", planner.plan("今天茅台股价是多少？").intent)
        self.assertEqual("news_query", planner.plan("最近比亚迪有什么新闻？").intent)
        self.assertEqual("financial_report_query", planner.plan("贵州茅台2026H1营业收入是多少？").intent)
        self.assertEqual("calculation_query", planner.plan("从100增长到120，增长率是多少？").intent)

    def test_runner_executes_market_and_news_through_mcp_client(self):
        client = FakeMCPClient()
        runner = FinancialAgentRunner(mcp_client=client)
        market = runner.run("今天贵州茅台股价怎么样？")
        news = runner.run("最近比亚迪有什么新闻？")
        self.assertTrue(market.success)
        self.assertTrue(news.success)
        self.assertEqual(("market_mcp",), market.executed_tools)
        self.assertEqual(("news_mcp",), news.executed_tools)
        self.assertEqual(["market", "news"], [call[0] for call in client.calls])

    def test_mcp_failure_does_not_crash_runner(self):
        response = FinancialAgentRunner(mcp_client=FakeMCPClient(success=False)).run("今天茅台股价是多少？")
        self.assertFalse(response.success)
        self.assertEqual("provider unavailable", response.error)
