"""Unit, approximation and temporal checks without model or data services."""
import copy
import json
from decimal import Decimal
from pathlib import Path
import unittest

from agent.langgraph_agent import LangGraphFinancialAgent
from agent.numeric_guardrail import AMOUNT_APPROX_REL_TOLERANCE, amount_tokens, temporal_tokens


class NumericGuardrailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = json.loads((Path(__file__).parent / "fixtures/numeric_representation_e2e.json").read_text())

    def setUp(self):
        self.agent = object.__new__(LangGraphFinancialAgent)
        self.agent._emit = lambda *a, **kw: None

    def check(self, text, supported, state=None):
        violations = self.agent._guardrail_violations(state or self.saved, text)
        self.assertEqual(not supported, "unsupported_numeric_claim" in violations, (text, violations))

    def test_exact_yuan_to_yi_conversion(self):
        self.check("营业收入为907.0326096448亿元。", True)
        token = amount_tokens("营业收入为907.0326096448亿元。")[0]
        self.assertEqual(Decimal("90703260964.48"), token.value)
        self.assertEqual("financial_amount", token.semantic_type)

    def test_wrong_unit_cannot_pass_on_same_mantissa(self):
        self.check("营业收入为90,703,260,964.48万元。", False)

    def test_strict_above_supported_threshold(self):
        self.check("营业收入超907亿元。", True)

    def test_strict_above_wrong_threshold(self):
        self.check("营业收入超1000亿元。", False)

    def test_strict_above_is_not_greater_equal(self):
        self.check("营业收入超907.0326096448亿元。", False)

    def test_approximate_market_capitalization(self):
        for prefix in ("约", "近"):
            with self.subTest(prefix=prefix):
                self.check(f"总市值{prefix}1.6万亿元。", True)
        self.assertEqual(Decimal("0.02"), AMOUNT_APPROX_REL_TOLERANCE)

    def test_wrong_approximation_rejected(self):
        self.check("总市值约2.5万亿元。", False)
        state = copy.deepcopy(self.saved)
        state["news_results"] = [{"success": True, "result": {"results": [{"title": "测试", "source": "test", "url": "https://example.test", "published_at": "2026-09-14", "snippet": "总市值为1.2万亿元。"}]}}]
        self.check("总市值约1.6万亿元。", False, state)

    def test_bare_amount_does_not_get_approximation(self):
        self.check("总市值为1.6万亿元。", False)
        self.check("营业收入为907.03亿元。", False)

    def test_explicit_approximate_revenue(self):
        self.check("营业收入约907.03亿元。", True)

    def test_real_range_uses_one_value_inside_both_bounds(self):
        self.check("总市值重回1.6万亿至1.7万亿元区间。", True)
        self.check("总市值为1.8万亿至1.9万亿元区间。", False)
        self.check("总市值为1.7万亿至1.6万亿元区间。", False)

    def test_timestamp_representation_not_financial_hour(self):
        text = "数据时间为2026年9月14日16:14:50。"
        self.check(text, True)
        self.assertEqual(["date", "time"], [t.unit for t in temporal_tokens(text)])
        self.check("数据时间为2026-09-14 16:14。", True)

    def test_unsupported_time_is_not_blanket_exempt(self):
        self.check("数据时间为2026-09-14 16:32。", False)

    def test_sixteen_percent_still_rejected(self):
        self.assertEqual({"16%"}, self.agent._numbers("增长16%"))
        self.check("增长16%。", False)

    def test_sixteen_yuan_still_rejected(self):
        self.assertEqual({"16"}, self.agent._numbers("股价16元"))
        self.check("股价16元。", False)
        self.check("营收16亿元。", False)

    def test_unknown_amount_rejected(self):
        self.check("营业收入为321亿元。", False)

    def test_market_percent_semantics_unchanged(self):
        self.check("涨跌幅0.22%。", True)
        self.check("涨跌幅22%。", False)

    def test_price_no_approximation(self):
        self.check("当前股价为1277.96元。", True)
        self.check("当前股价为1277.96万元。", False)
        self.check("当前股价约1280元。", False)

    def test_market_amount_does_not_support_financial_amount(self):
        self.check("营业收入约1.6万亿元。", False)

    def test_ticker_period_and_date_from_evidence(self):
        self.check("贵州茅台600519的2026H1报告，行情日期2026-09-14。", True)

    def test_saved_composite_roundtrip(self):
        self.assertEqual("unsupported_numeric_claim", self.saved["guardrail_status"])
        result = self.agent._guardrail(self.saved)
        self.assertEqual("passed", result["guardrail_status"])
        self.assertEqual(self.saved["synthesis_answer"], result["final_answer"])


if __name__ == "__main__":
    unittest.main()
