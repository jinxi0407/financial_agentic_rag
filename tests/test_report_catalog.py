import tempfile
import unittest
from pathlib import Path

from new_main import IntegratedQASystem
from rag_qa.core.query_metadata import (
    build_report_lookup_response,
    extract_query_metadata,
)
from rag_qa.core.report_catalog import ReportCatalog


REPORT_FILENAMES = (
    "贵州茅台_600519_2025H1.pdf",
    "贵州茅台_600519_2025FY.pdf",
    "贵州茅台_600519_2026H1.pdf",
)


def build_catalog():
    with tempfile.TemporaryDirectory() as temporary_directory:
        reports_directory = Path(temporary_directory) / "annual_reports"
        reports_directory.mkdir()
        for filename in REPORT_FILENAMES:
            (reports_directory / filename).touch()
        return ReportCatalog.from_directory(reports_directory)


class QueryMetadataTests(unittest.TestCase):
    def test_company_aliases_and_codes_resolve_to_canonical_metadata(self):
        for query in ("茅台营业收入", "贵州茅台营业收入", "600519营业收入"):
            metadata = extract_query_metadata(query)
            self.assertEqual("贵州茅台", metadata.company_name)
            self.assertEqual("600519", metadata.company_code)

        self.assertEqual("600036", extract_query_metadata("招行利润").company_code)
        self.assertEqual("000001", extract_query_metadata("000001年报").company_code)

    def test_all_supported_company_codes_resolve_without_losing_leading_zeroes(self):
        expected_codes = {
            "五粮液": "000858",
            "比亚迪": "002594",
            "宁德时代": "300750",
            "招商银行": "600036",
            "平安银行": "000001",
            "中芯国际": "688981",
            "北方华创": "002371",
        }
        for company_name, company_code in expected_codes.items():
            self.assertEqual(company_code, extract_query_metadata(company_name).company_code)

    def test_h1_expressions_and_fy_expressions_are_explicit_only(self):
        for query in ("2025年上半年", "2025半年", "2025半年度", "2025H1"):
            self.assertEqual("2025H1", extract_query_metadata(query).report_period)

        for query in ("2025年度", "2025年报", "2025全年", "2025FY"):
            self.assertEqual("2025FY", extract_query_metadata(query).report_period)

        metadata = extract_query_metadata("贵州茅台2025年营业收入")
        self.assertIsNone(metadata.report_period)
        self.assertIsNone(metadata.report_year)

    def test_single_report_and_company_only_filters(self):
        single_report = extract_query_metadata("贵州茅台2025年上半年营业收入是多少？")
        self.assertEqual(
            {"company_code": "600519", "report_period": "2025H1"},
            single_report.to_metadata_filter(),
        )

        company_only = extract_query_metadata("茅台的营业收入怎么样？")
        self.assertEqual({"company_code": "600519"}, company_only.to_metadata_filter())

    def test_cross_period_query_is_not_compressed_to_one_report_filter(self):
        metadata = extract_query_metadata("比较贵州茅台2025H1和2026H1营业收入")
        self.assertEqual(("2025H1", "2026H1"), metadata.report_periods)
        self.assertIsNone(metadata.report_period)
        self.assertEqual({"company_code": "600519"}, metadata.to_metadata_filter())

    def test_only_report_seeking_requests_are_report_lookup(self):
        self.assertEqual(
            "REPORT_LOOKUP",
            extract_query_metadata("我想看贵州茅台的半年度报告").intent,
        )
        self.assertEqual(
            "RAG",
            extract_query_metadata("贵州茅台报告期营业收入是多少？").intent,
        )


class ReportCatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = build_catalog()

    def test_catalog_uses_existing_filename_parser_and_sorts_reports(self):
        reports = self.catalog.find_reports("600519")
        self.assertEqual(["2025H1", "2025FY", "2026H1"], [r.report_period for r in reports])

    def test_catalog_fails_fast_for_an_invalid_annual_report_filename(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            reports_directory = Path(temporary_directory) / "annual_reports"
            reports_directory.mkdir()
            (reports_directory / "贵州茅台_2025年报.pdf").touch()
            with self.assertRaisesRegex(ValueError, "annual_reports PDF filename"):
                ReportCatalog.from_directory(reports_directory)

    def test_h1_without_year_returns_disambiguation(self):
        metadata = extract_query_metadata("我想看贵州茅台的半年度报告")
        response = build_report_lookup_response(metadata, self.catalog)
        self.assertIn("2025年半年度报告", response)
        self.assertIn("2026年半年度报告", response)
        self.assertIn("哪一年", response)

    def test_unique_fy_without_year_returns_the_only_report(self):
        metadata = extract_query_metadata("我想看贵州茅台的年度报告")
        response = build_report_lookup_response(metadata, self.catalog)
        self.assertIn("贵州茅台2025年年度报告", response)
        self.assertIn("贵州茅台_600519_2025FY.pdf", response)

    def test_specific_year_and_period_returns_the_unique_report(self):
        metadata = extract_query_metadata("我想看贵州茅台2026年半年度报告")
        response = build_report_lookup_response(metadata, self.catalog)
        self.assertIn("贵州茅台2026年半年度报告", response)
        self.assertIn("贵州茅台_600519_2026H1.pdf", response)

    def test_generic_financial_report_request_returns_available_list(self):
        metadata = extract_query_metadata("我想看贵州茅台的财报")
        response = build_report_lookup_response(metadata, self.catalog)
        self.assertIn("2025年半年度报告", response)
        self.assertIn("2025年年度报告", response)
        self.assertIn("2026年半年度报告", response)


class IntegratedQueryMetadataTests(unittest.TestCase):
    def _system(self):
        system = IntegratedQASystem.__new__(IntegratedQASystem)
        system.report_catalog = build_catalog()
        system._fetch_recent_history = lambda session_id: []
        system.update_session_history = lambda **_: None
        return system

    def test_report_lookup_short_circuits_rag_and_keeps_stream_shape(self):
        system = self._system()

        class Faq:
            @staticmethod
            def query(query, threshold):
                return "", True

        class RAG:
            no_context_response = "no context"

            @staticmethod
            def generate_answer(*args, **kwargs):
                raise AssertionError("report lookup must not enter RAG")

        system.faq = Faq()
        system.rag = RAG()
        result = list(system.query("我想看贵州茅台的财报"))

        self.assertEqual(2, len(result))
        self.assertFalse(result[0][1])
        self.assertEqual(("", True), result[1])
        self.assertIn("当前可用报告", result[0][0])

    def test_faq_fast_path_still_precedes_report_lookup(self):
        system = self._system()

        class Faq:
            @staticmethod
            def query(query, threshold):
                return "faq answer", False

        class RAG:
            no_context_response = "no context"

            @staticmethod
            def generate_answer(*args, **kwargs):
                raise AssertionError("FAQ response must not enter RAG")

        system.faq = Faq()
        system.rag = RAG()
        self.assertEqual([("faq answer", True)], list(system.query("我想看贵州茅台的财报")))

    def test_regular_rag_query_passes_derived_filter_without_changing_streaming(self):
        system = self._system()

        class Faq:
            @staticmethod
            def query(query, threshold):
                return "", True

        class RAG:
            no_context_response = "no context"
            received_kwargs = None

            @classmethod
            def generate_answer(cls, *args, **kwargs):
                cls.received_kwargs = kwargs
                return iter(("answer",))

        system.faq = Faq()
        system.rag = RAG()
        result = list(system.query("贵州茅台2025年上半年营业收入是多少？"))

        self.assertEqual(
            {"company_code": "600519", "report_period": "2025H1"},
            RAG.received_kwargs["metadata_filter"],
        )
        self.assertEqual([("answer", False), ("", True)], result)

    def test_out_of_scope_query_still_reaches_the_existing_rag_router(self):
        system = self._system()

        class Faq:
            @staticmethod
            def query(query, threshold):
                return "", True

        class RAG:
            no_context_response = "no context"
            received_kwargs = None

            @classmethod
            def generate_answer(cls, *args, **kwargs):
                cls.received_kwargs = kwargs
                return "该问题超出金融知识库范围。"

        system.faq = Faq()
        system.rag = RAG()
        result = list(system.query("法国的首都是哪里？"))

        self.assertIsNone(RAG.received_kwargs["metadata_filter"])
        self.assertEqual(
            [("该问题超出金融知识库范围。", False), ("", True)],
            result,
        )


if __name__ == "__main__":
    unittest.main()
