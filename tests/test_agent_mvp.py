"""Focused tests for the first tool-based Financial Agent MVP."""

import unittest

from agent.planner import FinancialPlanner
from agent.runner import FinancialAgentRunner
from agent.schemas import MCPToolResult
from agent.tools.financial_rag_tool import FinancialRAGTool


class FakeQASystem:
    def query(self, query):
        yield f"财报答案：{query}", False
        yield "", True


class FailingQASystem:
    def query(self, query):
        raise RuntimeError("service unavailable")
        yield query, True


class FailingMCPClient:
    def call_tool(self, server_name, tool_name, arguments):
        return MCPToolResult(
            tool_name=tool_name,
            server_name=server_name,
            inputs=arguments,
            result=None,
            success=False,
            error="MCP 调用失败：RuntimeError",
            latency=0.0,
        )


class FinancialAgentMVPTests(unittest.TestCase):
    def setUp(self):
        self.planner = FinancialPlanner()

    def test_financial_report_question_routes_to_tool(self):
        decision = self.planner.plan("贵州茅台2026H1营业收入是多少？")
        self.assertEqual("financial_report_query", decision.intent)
        self.assertEqual(("financial_rag",), decision.tools)

    def test_realtime_market_question_is_planned_but_unavailable(self):
        decision = self.planner.plan("今天茅台股价是多少？")
        self.assertEqual("market_query", decision.intent)
        self.assertEqual(("market_mcp",), decision.tools)
        self.assertEqual("ready", decision.status)

    def test_news_question_is_planned_but_unavailable(self):
        decision = self.planner.plan("帮我查今天AI新闻")
        self.assertEqual("news_query", decision.intent)
        self.assertEqual(("news_mcp",), decision.tools)
        self.assertEqual("ready", decision.status)

    def test_tool_success_returns_stable_schema(self):
        result = FinancialRAGTool(qa_system=FakeQASystem()).run("营业收入是多少？")
        self.assertTrue(result.success)
        self.assertEqual("financial_rag", result.tool_name)
        self.assertEqual("营业收入是多少？", result.query)
        self.assertIn("财报答案", result.answer)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.latency, 0)
        self.assertEqual({"tool_name", "query", "answer", "success", "error", "latency"}, set(result.to_dict()))

    def test_tool_exception_does_not_crash_agent(self):
        runner = FinancialAgentRunner(
            planner=self.planner,
            financial_rag_tool=FinancialRAGTool(qa_system=FailingQASystem()),
        )
        response = runner.run("贵州茅台2026H1营业收入是多少？")
        self.assertFalse(response.success)
        self.assertEqual("financial_report_query", response.planner.intent)
        self.assertEqual(1, len(response.tool_results))
        self.assertIn("RuntimeError", response.error)

    def test_market_query_does_not_call_rag_when_mcp_is_unavailable(self):
        runner = FinancialAgentRunner(
            planner=self.planner,
            financial_rag_tool=FinancialRAGTool(qa_system=FailingQASystem()),
            mcp_client=FailingMCPClient(),
        )
        response = runner.run("今天茅台股价是多少？")
        self.assertFalse(response.success)
        self.assertIn("MCP 调用失败", response.error)
        self.assertEqual(1, len(response.tool_results))


if __name__ == "__main__":
    unittest.main()
