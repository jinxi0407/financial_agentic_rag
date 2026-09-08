import unittest
from decimal import Decimal

from langchain_core.documents import Document

from rag_qa.core.financial_evidence import (
    extract_document_evidence,
    extract_verified_evidence,
    format_verified_evidence_block,
)


def make_document(parent_id, company_code, report_period, content, company_name="测试公司"):
    return Document(
        page_content=content,
        metadata={
            "parent_id": parent_id,
            "company_name": company_name,
            "company_code": company_code,
            "report_period": report_period,
        },
    )


class StructuredFinancialEvidenceTests(unittest.TestCase):
    def test_amount_units_normalize_to_yuan(self):
        cases = (
            ("元", "1,234,567", Decimal("1234567")),
            ("千元", "1,234", Decimal("1234000")),
            ("万元", "123.4", Decimal("1234000")),
            ("百万元", "12.34", Decimal("12340000")),
            ("亿元", "1.234", Decimal("123400000")),
        )
        for unit, value, expected in cases:
            with self.subTest(unit=unit):
                document = make_document(
                    f"parent-{unit}", "600519", "2026H1",
                    f"单位：{unit}\n本报告期 上年同期\n营业收入\n{value}\n1",
                    company_name="贵州茅台",
                )
                evidence = extract_document_evidence(document, ("revenue",))
                self.assertEqual(1, len(evidence))
                self.assertEqual(expected, evidence[0].normalized_value)
                self.assertEqual("元", evidence[0].normalized_unit)

    def test_rate_is_normalized_as_percentage_points(self):
        document = make_document(
            "pab-nim", "000001", "2026H1",
            "2026年1-6月 2025年1-6月\n净息差\n1.80%\n1.80%",
            company_name="平安银行",
        )
        evidence = extract_document_evidence(document, ("net_interest_margin",))
        self.assertEqual(Decimal("1.80"), evidence[0].normalized_value)
        self.assertEqual("percentage_point", evidence[0].normalized_unit)

    def test_explicit_narrative_binds_period_metric_value_and_unit(self):
        document = make_document(
            "byd-2026", "002594", "2026H1",
            "2026年上半年，本集团实现营业收入约人民币344,815百万元，同比下降7.13%。",
            company_name="比亚迪",
        )
        evidence = extract_document_evidence(document, ("revenue",))
        self.assertEqual("344,815", evidence[0].raw_value)
        self.assertEqual(Decimal("344815000000"), evidence[0].normalized_value)
        self.assertTrue(evidence[0].exact_match)
        self.assertEqual("high", evidence[0].confidence)

    def test_table_header_unit_and_wrapped_metric_label_are_bound(self):
        document = make_document(
            "north-2026", "002371", "2026H1",
            "（人民币百万元，特别注明除外）\n本报告期 上年同期\n"
            "归属于上市公司股东的净利\n润（元）\n3,370,020,676.75\n3,207,978,523.58",
            company_name="北方华创",
        )
        evidence = extract_document_evidence(document, ("net_profit",))
        self.assertEqual(1, len(evidence))
        self.assertEqual("3,370,020,676.75", evidence[0].raw_value)
        self.assertEqual("元", evidence[0].unit)

    def test_parenthetical_rmb_million_table_unit_is_bound(self):
        document = make_document(
            "cmb-2026", "600036", "2026H1",
            "（人民币百万元，特别注明除外）\n2026年1-6月 2025年1-6月\n"
            "营业收入\n178,181\n169,969",
            company_name="招商银行",
        )
        evidence = extract_document_evidence(document, ("revenue",))
        self.assertEqual(1, len(evidence))
        self.assertEqual(Decimal("178181000000"), evidence[0].normalized_value)

    def test_rejects_missing_unit_or_company_or_period_binding(self):
        no_unit = make_document(
            "missing-unit", "600519", "2026H1", "本报告期\n营业收入\n90,703,260,964.48",
            company_name="贵州茅台",
        )
        missing_company = Document(
            page_content="单位：元\n本报告期\n营业收入\n90,703,260,964.48",
            metadata={"parent_id": "missing-company", "company_code": "600519", "report_period": "2026H1"},
        )
        missing_binding = make_document(
            "missing-binding", "600519", "2026H1", "单位：元\n营业收入\n90,703,260,964.48",
            company_name="贵州茅台",
        )
        self.assertEqual((), extract_document_evidence(no_unit, ("revenue",)))
        self.assertEqual((), extract_document_evidence(missing_company, ("revenue",)))
        self.assertEqual((), extract_document_evidence(missing_binding, ("revenue",)))

    def test_conflicting_values_for_one_cell_are_rejected(self):
        documents = (
            make_document(
                "first", "600519", "2026H1", "单位：元\n本报告期\n营业收入\n90,703,260,964.48",
                company_name="贵州茅台",
            ),
            make_document(
                "second", "600519", "2026H1", "单位：元\n本报告期\n营业收入\n90,000,000,000.00",
                company_name="贵州茅台",
            ),
        )
        self.assertEqual((), extract_verified_evidence(documents, ("revenue",)))

    def test_metric_boundaries_and_verified_block(self):
        investment = make_document(
            "investment", "002594", "2025FY", "单位：元\n本报告期\n研发投入\n63,441,379,000.00",
            company_name="比亚迪",
        )
        expense = make_document(
            "expense", "002594", "2025FY", "单位：元\n本报告期\n研发费用\n57,978,105,000.00",
            company_name="比亚迪",
        )
        evidence = extract_verified_evidence((investment, expense), ("research_investment",))
        self.assertEqual(1, len(evidence))
        self.assertEqual("research_investment", evidence[0].metric)
        block = format_verified_evidence_block(evidence)
        self.assertIn("系统已验证财务数值", block)
        self.assertIn("63,441,379,000元", block)


if __name__ == "__main__":
    unittest.main()
