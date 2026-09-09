"""Focused checks for the development-only real-user paraphrase fixes."""
import unittest

from agent.planner import FinancialPlanner
from evaluations.run_real_user_paraphrase_regression import audit_dataset, load_dataset, make_agent


class RealUserAgentRegressionTests(unittest.TestCase):
    def test_dataset_audit_and_size(self):
        audit = audit_dataset(load_dataset())
        self.assertTrue(audit["passed"])
        self.assertEqual(92, audit["case_count"])

    def test_greeting_and_definition_do_not_become_calculations(self):
        planner = FinancialPlanner()
        self.assertEqual("greeting", planner.plan("你好").intent)
        self.assertEqual("greeting", planner.plan("你好，你能做什么？").intent)
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
        for query, operation in {
            "100+30多少": "addition",
            "100-30": "subtraction",
            "100*1.2": "multiplication",
            "100×1.2": "multiplication",
            "100÷4": "ratio",
            "100加30": "addition",
            "100减30": "subtraction",
        }.items():
            self.assertEqual(operation, planner.simple_calculation_request(query)["operation"])
        for query in ("贵州茅台600519现在股价怎么样", "比亚迪2026H1营业收入是多少", "看看2025FY报告"):
            self.assertIsNone(planner.simple_calculation_request(query))
        for query, expected in {
            "1+9*3是多少": 28,
            "(1+9)*3是多少": 30,
            "100/4+5": 30,
            "100-20*2": 60,
            "10+20/5": 14,
            "1+9×3": 28,
            "100÷4+5": 30,
        }.items():
            request = planner.simple_calculation_request(query)
            self.assertEqual("expression", request["operation"])
            result = make_agent().calculator.run(**request)
            self.assertEqual(expected, result.result)
        for query in ("__import__('os').system('x')", "open('x')", "2**100"):
            self.assertIsNone(planner.simple_calculation_request(query))

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

    def test_market_pronoun_follow_up_uses_session_company_only(self):
        agent = make_agent()
        agent.run("五粮液财报看看", "market-slot")
        agent.run("2025H1", "market-slot")
        follow_up = agent.run("帮我看一下它的股票", "market-slot")
        self.assertEqual("market_query", follow_up["intent"])
        self.assertEqual(["market_mcp"], follow_up["required_tools"])
        self.assertEqual(["000858"], follow_up["tickers"])
        self.assertEqual(["market_mcp"], follow_up["executed_tools"])

    def test_market_pronoun_without_company_context_is_safe(self):
        response = make_agent().run("帮我看一下它的股票", "no-market-context")
        self.assertEqual("unsupported", response["intent"])
        self.assertEqual([], response["required_tools"])
        self.assertEqual([], response["tickers"])
        self.assertIn("请先说明要查询的公司", response["final_answer"])

    def test_company_pronouns_without_context_never_call_tools(self):
        agent = make_agent()
        for query in ("帮我看一下它的股票", "帮我看一下它的新闻", "看看它的财报", "它现在行情怎么样？"):
            response = agent.run(query, f"no-context-{query}")
            self.assertEqual("unsupported", response["intent"])
            self.assertEqual([], response["required_tools"])
            self.assertEqual([], response["executed_tools"])
            self.assertEqual([], response["errors"])
            self.assertEqual("passed", response["guardrail_status"])

    def test_explicit_company_overrides_session_for_news(self):
        agent = make_agent()
        agent.run("先看看五粮液的财报", "override-news")
        response = agent.run("那比亚迪最近有什么新闻", "override-news")
        self.assertEqual("news_query", response["intent"])
        self.assertEqual(["002594"], response["tickers"])
        self.assertEqual(["news_mcp"], response["executed_tools"])

    def test_explicit_company_resolves_pronoun_without_session(self):
        response = make_agent().run("贵州茅台，它最近有什么新闻？", "explicit-pronoun")
        self.assertEqual("news_query", response["intent"])
        self.assertEqual(["600519"], response["tickers"])
        self.assertEqual(["news_mcp"], response["executed_tools"])


if __name__ == "__main__":
    unittest.main()
