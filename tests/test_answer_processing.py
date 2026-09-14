"""Offline regressions from the saved 20260914 GPU E2E tool payloads."""
import copy
import json
from pathlib import Path
import unittest

from agent.langgraph_agent import LangGraphFinancialAgent


class AnswerProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = json.loads(
            (Path(__file__).parent / "fixtures" / "answer_processing_e2e.json").read_text()
        )

    def setUp(self):
        # The checks need neither graph construction nor any external service.
        self.agent = object.__new__(LangGraphFinancialAgent)
        self.agent._emit = lambda *args, **kwargs: None

    def test_saved_news_title_with_negation_survives(self):
        saved = self.saved["5"]
        answer = self.agent._remove_false_tool_unavailability(saved["state"], saved["synthesis_answer"])
        self.assertEqual(saved["synthesis_answer"], answer)
        self.assertIn("燃油车已没有未来", answer)
        self.assertNotEqual(saved["final_answer"], answer)

    def test_media_name_is_not_an_availability_statement(self):
        state = self.saved["5"]["state"]
        text = "每日经济新闻报道：公司表示没有暂停研发。"
        self.assertEqual(text, self.agent._remove_false_tool_unavailability(state, text))

    def test_ten_real_titles_and_provenance_are_preserved(self):
        state = copy.deepcopy(self.saved["5"]["state"])
        state["news_results"] += self.saved["7"]["state"]["news_results"]
        articles = [n for r in state["news_results"] for n in r["result"]["results"]]
        self.assertEqual(10, len(articles))
        text = "\n".join(" | ".join(n[k] for k in ("title", "source", "url", "published_at")) for n in articles)
        state["synthesis_answer"] = text
        result = self.agent._guardrail(state)
        self.assertEqual("passed", result["guardrail_status"])
        self.assertEqual(text, result["final_answer"])

    def test_empty_or_failed_news_still_unavailable(self):
        for success in (True, False):
            with self.subTest(success=success):
                state = {"required_tools": ["news_mcp"], "news_results": [{"success": success, "result": {"results": []}}]}
                self.assertIn("news_unavailable", self.agent._guardrail_violations(state, "当前未检索到相关新闻。"))

    def test_false_placeholder_removed_but_financial_gap_kept(self):
        state = self.saved["7"]["state"]
        text = "财报上下文没有当前行情和近期新闻。当前财务证据不足。"
        self.assertEqual("当前财务证据不足。", self.agent._remove_false_tool_unavailability(state, text))

    def market_state(self, value=0.22):
        return {
            "query": "结合财报和行情分析", "required_tools": ["financial_rag", "market_mcp"],
            "tool_results": [{"tool_name": "financial_rag", "success": True, "answer": "营业收入为 100 元。"}],
            "market_results": [{"success": True, "result": {"price": 1500, "change_percent": value, "source": "test"}}],
        }

    def test_percent_value_matches_its_percent_rendering(self):
        state = self.market_state()
        self.assertEqual({"0.22"}, self.agent._numbers("0.22"))
        self.assertEqual({"0.22%"}, self.agent._numbers("0.22%"))
        self.assertEqual({"0.22%"}, self.agent._market_percent_evidence(state["market_results"]))
        self.assertNotIn("unsupported_numeric_claim", self.agent._guardrail_violations(state, "涨跌幅为 +0.22%。"))

    def test_twenty_two_percent_is_not_supported(self):
        self.assertIn("unsupported_numeric_claim", self.agent._guardrail_violations(self.market_state(), "涨跌幅为 22%。"))

    def test_price_units_and_unsupported_price_keep_original_behavior(self):
        state = self.market_state()
        self.assertNotIn("unsupported_numeric_claim", self.agent._guardrail_violations(state, "价格为 1500元。"))
        self.assertIn("unsupported_numeric_claim", self.agent._guardrail_violations(state, "价格为 9999元。"))

    def test_unseen_percentage_remains_blocked(self):
        self.assertIn("unsupported_numeric_claim", self.agent._guardrail_violations(self.market_state(), "涨跌幅为 3.14%。"))

    def test_plain_price_or_ratio_does_not_authorize_percent(self):
        state = self.market_state(None)
        state["market_results"][0]["result"].update(price=0.22, ratio=0.22)
        self.assertIn("unsupported_numeric_claim", self.agent._guardrail_violations(state, "涨跌幅为 0.22%。"))

    def test_failed_or_non_numeric_market_values_do_not_authorize_percent(self):
        for value in (None, True, "0.22", float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertEqual(set(), self.agent._market_percent_evidence(self.market_state(value)["market_results"]))
        state = self.market_state()
        state["market_results"][0]["success"] = False
        self.assertEqual(set(), self.agent._market_percent_evidence(state["market_results"]))

    def test_negative_percent_preserves_sign(self):
        state = self.market_state(-0.22)
        self.assertNotIn("unsupported_numeric_claim", self.agent._guardrail_violations(state, "涨跌幅为 -0.22%。"))
        self.assertIn("unsupported_numeric_claim", self.agent._guardrail_violations(state, "涨跌幅为 0.22%。"))

    def test_financial_growth_and_percentage_points_unchanged(self):
        state = self.market_state()
        state["tool_results"][0]["answer"] = "收入增长率为 5%，毛利率上升 2个百分点。"
        self.assertNotIn("unsupported_numeric_claim", self.agent._guardrail_violations(state, "收入增长率为 5%，毛利率上升 2个百分点。"))
        self.assertIn("unsupported_numeric_claim", self.agent._guardrail_violations(state, "毛利率上升 2%。"))
        state["query"] = "计算增长率"
        self.assertIn("unverified_calculation", self.agent._guardrail_violations(state, "增长率为5%。"))

    def test_cjk_number_boundary(self):
        self.assertEqual({"22%"}, self.agent._numbers("涨跌幅为22%"))
        self.assertEqual({"22%"}, self.agent._numbers("涨跌幅为 22%"))
        self.assertEqual({"0.22%"}, self.agent._numbers("涨跌幅为0.22%"))
        self.assertEqual({"1.5E+3"}, self.agent._numbers("股价为1500元"))
        self.assertEqual({"123.45"}, self.agent._numbers("营业收入为123.45亿元"))

    def test_cjk_percent_guardrail_accepts_only_supported_value(self):
        state = self.market_state()
        self.assertNotIn("unsupported_numeric_claim", self.agent._guardrail_violations(state, "涨跌幅为0.22%"))
        self.assertIn("unsupported_numeric_claim", self.agent._guardrail_violations(state, "涨跌幅为22%"))
        self.assertIn("unsupported_numeric_claim", self.agent._guardrail_violations(self.market_state(None), "涨跌幅为22%"))

    def test_cjk_price_guardrail(self):
        self.assertNotIn("unsupported_numeric_claim", self.agent._guardrail_violations(self.market_state(), "股价为1500元"))
        self.assertIn("unsupported_numeric_claim", self.agent._guardrail_violations(self.market_state(), "股价为2500元"))

    def test_cjk_signed_decimals(self):
        self.assertEqual({"-0.22%", "0.3%"}, self.agent._numbers("涨跌幅为-0.22%，另一值为+0.3%"))
        self.assertNotIn("unsupported_numeric_claim", self.agent._guardrail_violations(self.market_state(-0.22), "涨跌幅为-0.22%"))

    def test_identifier_date_and_report_period_boundaries(self):
        self.assertEqual(set(), self.agent._numbers("field123 ABC0.22 _123"))
        self.assertEqual({"2026", "9", "14"}, self.agent._numbers("2026-09-14"))
        self.assertEqual({"2026", "600519"}, self.agent._numbers("贵州茅台600519的2026H1"))
        state = self.market_state()
        state["query"] = "贵州茅台600519的2026H1财报和行情"
        state["market_results"][0]["result"]["market_time"] = "2026-09-14"
        self.assertNotIn("unsupported_numeric_claim", self.agent._guardrail_violations(state, "贵州茅台600519，2026H1；数据日期2026-09-14。股价为1500元。"))

    def test_saved_composite_verified_amount_representation_preserves_synthesis(self):
        saved = self.saved["7"]
        state = copy.deepcopy(saved["state"])
        state["synthesis_answer"] = saved["synthesis_answer"]
        self.assertEqual("unsupported_numeric_claim", saved["guardrail_status"])
        result = self.agent._guardrail(state)
        # Now covered by the bounded, typed amount-representation contract.
        self.assertEqual("passed", result["guardrail_status"])
        self.assertIn("907.03", self.agent._numbers(saved["synthesis_answer"]))
        self.assertIn("0.22%", self.agent._market_percent_evidence(state["market_results"]))

    def test_saved_composite_with_original_yuan_value_preserves_synthesis(self):
        saved = self.saved["7"]
        state = copy.deepcopy(saved["state"])
        state["synthesis_answer"] = saved["synthesis_answer"].replace("约907.03亿元", "90,703,260,964.48元")
        result = self.agent._guardrail(state)
        self.assertEqual("passed", result["guardrail_status"])
        self.assertEqual(state["synthesis_answer"], result["final_answer"])


if __name__ == "__main__":
    unittest.main()
