"""Focused checks for the development-only real-user paraphrase fixes."""
import unittest

from agent.planner import FinancialPlanner
from evaluations.run_real_user_paraphrase_regression import audit_dataset, load_dataset, make_agent


class RealUserAgentRegressionTests(unittest.TestCase):
    def test_dataset_audit_and_size(self):
        audit = audit_dataset(load_dataset())
        self.assertTrue(audit["passed"])
        self.assertEqual(55, audit["case_count"])

    def test_greeting_and_definition_do_not_become_calculations(self):
        planner = FinancialPlanner()
        self.assertEqual("greeting", planner.plan("你好").intent)
        for query in ("什么是流动比率？", "流动比率是啥？", "什么是速动比率？", "市盈率是什么意思？"):
            decision = planner.plan(query)
            self.assertEqual("financial_report_query", decision.intent)
            self.assertEqual(("financial_rag",), decision.tools)

    def test_calculation_paraphrases_are_deterministic(self):
        planner = FinancialPlanner()
        self.assertEqual("growth_rate", planner.simple_calculation_request("100比80增长多少百分比？")["operation"])
        self.assertEqual("absolute_change", planner.simple_calculation_request("从100减少到80，减少了多少？")["operation"])
        self.assertEqual("ratio", planner.simple_calculation_request("20除以80是多少？")["operation"])
        self.assertEqual("percentage_point_change", planner.simple_calculation_request("不良贷款率从1.2%降到1.0%，下降多少个百分点？")["operation"])

    def test_market_news_paraphrases_and_thread_context(self):
        planner = FinancialPlanner()
        self.assertEqual("market_query", planner.plan("帮我看看比亚迪这个票").intent)
        self.assertEqual("news_query", planner.plan("中芯国际最近发生什么了？").intent)
        agent = make_agent()
        agent.run("查比亚迪2026H1营业收入", "byd")
        follow_up = agent.run("看看它现在的股票行情", "byd")
        self.assertEqual(["002594"], follow_up["tickers"])
        self.assertIn("当前价格：", follow_up["final_answer"])

    def test_friendly_answers_and_report_period_guidance(self):
        agent = make_agent()
        greeting = agent.run("你能做什么？", "greeting")
        self.assertIn("8 家", greeting["final_answer"])
        unsupported = agent.run("给我推荐电影", "unsupported")
        self.assertIn("例如", unsupported["final_answer"])
        report = agent.run("中芯国际的半年度报告能看吗？", "report")
        self.assertIn("2025H1", report["final_answer"])
        self.assertIn("2026H1", report["final_answer"])

    def test_period_only_follow_up_binds_the_remembered_company_to_rag(self):
        agent = make_agent()
        agent.run("茅台的财报看看", "period-slot")
        follow_up = agent.run("2025H1就是上半年啊", "period-slot")
        rag_result = next(item for item in follow_up["tool_results"] if item["tool_name"] == "financial_rag")
        self.assertEqual(["600519"], follow_up["tickers"])
        self.assertEqual(["2025H1"], follow_up["report_periods"])
        self.assertEqual("贵州茅台 2025H1 财报", rag_result["query"])

    def test_explicit_company_replaces_inherited_company(self):
        agent = make_agent()
        agent.run("茅台的财报看看", "company-slot")
        follow_up = agent.run("比亚迪2025H1营业收入", "company-slot")
        rag_result = next(item for item in follow_up["tool_results"] if item["tool_name"] == "financial_rag")
        self.assertEqual(["002594"], follow_up["tickers"])
        self.assertEqual("比亚迪2025H1营业收入", rag_result["query"])


if __name__ == "__main__":
    unittest.main()
