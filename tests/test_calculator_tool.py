"""Focused tests for the deterministic Agent Calculator Tool."""

import unittest

from agent.runner import FinancialAgentRunner
from agent.schemas import MCPToolResult
from agent.tools.calculator_tool import CalculatorTool
from agent.tools.financial_rag_tool import FinancialRAGTool


class FailingCalculator:
    def run(self, **kwargs):
        return CalculatorTool().run("growth_rate", current=1, previous=0)


class FailingRAG:
    def run(self, query):
        raise AssertionError("Market/news must not call FinancialRAGTool")


class UnavailableMCPClient:
    def call_tool(self, server_name, tool_name, arguments):
        return MCPToolResult(tool_name, server_name, arguments, None, False, "provider unavailable", 0.0)


class CalculatorToolTests(unittest.TestCase):
    def setUp(self):
        self.tool = CalculatorTool()

    def test_growth_rate(self):
        result = self.tool.run("growth_rate", current=120, previous=100)
        self.assertTrue(result.success)
        self.assertEqual(20.0, result.result)

    def test_absolute_change(self):
        result = self.tool.run("absolute_change", current=120, previous=100)
        self.assertTrue(result.success)
        self.assertEqual(20.0, result.result)

    def test_percentage_point_change(self):
        result = self.tool.run("percentage_point_change", current=1.05, previous=0.94)
        self.assertTrue(result.success)
        self.assertAlmostEqual(0.11, result.result)

    def test_basic_arithmetic_operations(self):
        for operation, left, right, expected in (
            ("addition", 100, 30, 130),
            ("subtraction", 100, 30, 70),
            ("multiplication", 100, 1.2, 120),
        ):
            result = self.tool.run(operation, left=left, right=right)
            self.assertTrue(result.success)
            self.assertEqual(expected, result.result)

    def test_safe_compound_expressions_respect_precedence(self):
        cases = {
            "1+9*3是多少": 28,
            "(1+9)*3是多少": 30,
            "100/4+5": 30,
            "100-20*2": 60,
            "10+20/5": 14,
            "1+9×3": 28,
            "100÷4+5": 30,
        }
        for expression, expected in cases.items():
            result = self.tool.run("expression", expression=expression)
            self.assertTrue(result.success, expression)
            self.assertEqual(expected, result.result, expression)

    def test_unsafe_expressions_are_rejected(self):
        for expression in ("__import__('os').system('x')", "open('x')", "2**100"):
            result = self.tool.run("expression", expression=expression)
            self.assertFalse(result.success, expression)
            self.assertIn("unsafe arithmetic expression", result.error)

    def test_division_by_zero_is_safe(self):
        result = self.tool.run("growth_rate", current=120, previous=0)
        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        self.assertIn("ZeroDivisionError", result.error)

    def test_market_and_news_do_not_call_rag(self):
        runner = FinancialAgentRunner(financial_rag_tool=FailingRAG(), mcp_client=UnavailableMCPClient())
        for query in ("今天茅台股价是多少？", "帮我查今天AI新闻"):
            response = runner.run(query)
            self.assertFalse(response.success)
            self.assertEqual("provider unavailable", response.error)
            self.assertEqual(1, len(response.tool_results))

    def test_calculator_failure_does_not_crash_runner(self):
        runner = FinancialAgentRunner(calculator_tool=FailingCalculator())
        response = runner.run("从100增长到120，增长率是多少？")
        self.assertFalse(response.success)
        self.assertEqual("calculation_query", response.planner.intent)
        self.assertIn("ZeroDivisionError", response.error)

    def test_runner_executes_simple_growth_query(self):
        response = FinancialAgentRunner().run("从100增长到120，增长率是多少？")
        self.assertTrue(response.success)
        self.assertEqual("calculation_query", response.planner.intent)
        self.assertEqual("20.0", response.answer)


if __name__ == "__main__":
    unittest.main()
