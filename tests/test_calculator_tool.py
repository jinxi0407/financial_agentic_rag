"""Focused tests for the deterministic Agent Calculator Tool."""

import unittest

from agent.runner import FinancialAgentRunner
from agent.tools.calculator_tool import CalculatorTool
from agent.tools.financial_rag_tool import FinancialRAGTool


class FailingCalculator:
    def run(self, **kwargs):
        return CalculatorTool().run("growth_rate", current=1, previous=0)


class FailingRAG:
    def run(self, query):
        raise AssertionError("Market/news must not call FinancialRAGTool")


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

    def test_division_by_zero_is_safe(self):
        result = self.tool.run("growth_rate", current=120, previous=0)
        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        self.assertIn("ZeroDivisionError", result.error)

    def test_market_and_news_do_not_call_rag(self):
        runner = FinancialAgentRunner(financial_rag_tool=FailingRAG())
        for query in ("今天茅台股价是多少？", "帮我查今天AI新闻"):
            response = runner.run(query)
            self.assertFalse(response.success)
            self.assertEqual("planned_but_tool_unavailable", response.error)
            self.assertEqual((), response.tool_results)

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
