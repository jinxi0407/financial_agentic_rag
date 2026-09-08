"""Integration tests that use real stdio MCP handshakes, not direct imports."""

import unittest

from agent.mcp_client import FinancialMCPClient


class MCPProtocolTests(unittest.TestCase):
    def setUp(self):
        self.client = FinancialMCPClient()

    def test_client_discovers_and_calls_market_tool_over_stdio(self):
        self.assertIn("get_market_snapshot", [tool["name"] for tool in self.client.list_tools("market")])
        result = self.client.call_tool("market", "get_market_snapshot", {"ticker": "999999"})
        self.assertFalse(result.success)
        self.assertIsNotNone(result.result)
        self.assertIn("仅支持", result.error)

    def test_client_discovers_and_calls_news_tool_over_stdio(self):
        self.assertIn("search_financial_news", [tool["name"] for tool in self.client.list_tools("news")])
        result = self.client.call_tool("news", "search_financial_news", {"query": ""})
        self.assertFalse(result.success)
        self.assertIsNotNone(result.result)
        self.assertIn("不能为空", result.error)
