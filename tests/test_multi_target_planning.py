import unittest

from langchain_core.documents import Document

from new_main import IntegratedQASystem
from rag_qa.core.new_rag_system import RAGSystem
from rag_qa.core.query_metadata import extract_query_metadata


PLANNING_CASES = {
    "yoy_001": (
        "贵州茅台2026年上半年营业收入相比2025年上半年变化多少？",
        [("600519", "2026H1"), ("600519", "2025H1")],
    ),
    "yoy_002": (
        "五粮液2026年上半年归母净利润相比2025年上半年变化多少？",
        [("000858", "2026H1"), ("000858", "2025H1")],
    ),
    "yoy_003": (
        "比亚迪2026年上半年营业收入相比2025年上半年变化多少？",
        [("002594", "2026H1"), ("002594", "2025H1")],
    ),
    "period_001": (
        "贵州茅台2025年度营业收入与2026年上半年营业收入相比变化多少？请说明两个期间是否可直接视为同比。",
        [("600519", "2025FY"), ("600519", "2026H1")],
    ),
    "period_003": (
        "比亚迪2025年度归母净利润与2026年上半年归母净利润可以直接比较同比增速吗？分别是多少？",
        [("002594", "2025FY"), ("002594", "2026H1")],
    ),
    "cross_001": (
        "比较贵州茅台和五粮液2026年上半年的营业收入。",
        [("600519", "2026H1"), ("000858", "2026H1")],
    ),
    "cross_003": (
        "比较比亚迪和宁德时代2026年上半年的营业收入。",
        [("002594", "2026H1"), ("300750", "2026H1")],
    ),
    "cross_005": (
        "比较招商银行和平安银行2026年上半年的净息差。",
        [("600036", "2026H1"), ("000001", "2026H1")],
    ),
    "multi_006": (
        "比较招商银行和平安银行2025年度的不良贷款率和拨备覆盖率，并概括资产质量差异。",
        [("600036", "2025FY"), ("000001", "2025FY")],
    ),
}


class TargetPlanningTests(unittest.TestCase):
    def test_regression_cases_produce_bound_company_period_targets(self):
        for case_id, (query, expected_bindings) in PLANNING_CASES.items():
            with self.subTest(case_id=case_id):
                metadata = extract_query_metadata(query)
                plan = metadata.subquery_plan()

                self.assertTrue(metadata.requires_deterministic_subqueries())
                self.assertIsNone(metadata.to_metadata_filter())
                self.assertEqual(
                    expected_bindings,
                    [
                        (
                            target["metadata_filter"]["company_code"],
                            target["metadata_filter"]["report_period"],
                        )
                        for target in plan
                    ],
                )

    def test_single_company_single_period_keeps_strategy_selection_behavior(self):
        metadata = extract_query_metadata("贵州茅台2026年上半年营业收入是多少？")
        self.assertFalse(metadata.requires_deterministic_subqueries())
        self.assertEqual(
            {"company_code": "600519", "report_period": "2026H1"},
            metadata.to_metadata_filter(),
        )

    def test_broad_multi_company_comparison_expands_core_metric_targets(self):
        metadata = extract_query_metadata("比较贵州茅台和五粮液2026H1的经营表现")

        self.assertEqual(
            ("revenue", "net_profit", "operating_cash_flow"),
            metadata.requested_metrics,
        )
        self.assertTrue(metadata.requires_deterministic_subqueries())
        self.assertEqual(6, len(metadata.subquery_targets))
        self.assertEqual(
            ["600519", "600519", "600519", "000858", "000858", "000858"],
            [target.company_code for target in metadata.subquery_targets],
        )
        self.assertTrue(all(target.report_period == "2026H1" for target in metadata.subquery_targets))
        self.assertEqual(
            ["营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额"],
            [target.query.rsplit(" ", 1)[-1] for target in metadata.subquery_targets[:3]],
        )

    def test_exact_metric_and_missing_period_do_not_trigger_broad_expansion(self):
        exact = extract_query_metadata("比较贵州茅台和五粮液2026H1营业收入")
        missing_period = extract_query_metadata("比较贵州茅台和五粮液的经营表现")

        self.assertEqual(("revenue",), exact.requested_metrics)
        self.assertEqual(2, len(exact.subquery_targets))
        self.assertEqual(2, len(missing_period.subquery_targets))


class _RecordingStore:
    def __init__(self):
        self.call = None

    def hybrid_search_subqueries_with_batched_embedding(self, subqueries, **kwargs):
        self.call = {"subqueries": subqueries, **kwargs}
        results = [
            [Document(page_content=query, metadata={"parent_id": f"parent-{index}"})]
            for index, query in enumerate(subqueries)
        ]
        diagnostics = {
            "timing": {
                "embedding_seconds": 0.0,
                "hybrid_search_seconds": 0.0,
                "parent_dedup_seconds": 0.0,
                "reranker_seconds": 0.0,
            },
            "reranker_predict_calls": 0,
            "per_subquery": [
                {
                    "query": query,
                    "hybrid_search_seconds": 0.0,
                    "parent_dedup_seconds": 0.0,
                    "parent_count": 1,
                }
                for query in subqueries
            ],
        }
        return results, diagnostics


class RAGTargetBindingTests(unittest.TestCase):
    def test_rag_uses_paired_targets_without_calling_subquery_llm(self):
        query, expected_bindings = PLANNING_CASES["cross_003"]
        plan = extract_query_metadata(query).subquery_plan()
        store = _RecordingStore()
        rag = RAGSystem.__new__(RAGSystem)
        rag.vector_store = store
        rag.llm = lambda _: self.fail("deterministic target plan must not call the SubQuery LLM")

        rag.retrieve_and_merge(
            query,
            strategy="子查询检索",
            strategy_selection_seconds=0.0,
            subquery_targets=plan,
        )

        self.assertEqual(
            [
                {"company_code": company_code, "report_period": report_period}
                for company_code, report_period in expected_bindings
            ],
            store.call["metadata_filters"],
        )


class IntegratedTargetPlanningTests(unittest.TestCase):
    def test_integrated_runtime_forces_subquery_with_the_same_bound_plan(self):
        query, expected_bindings = PLANNING_CASES["period_001"]
        system = IntegratedQASystem.__new__(IntegratedQASystem)
        system._fetch_recent_history = lambda session_id: []
        system.update_session_history = lambda **_: None

        class Faq:
            @staticmethod
            def query(*args, **kwargs):
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
        self.assertEqual([("answer", False), ("", True)], list(system.query(query)))
        self.assertEqual("子查询检索", RAG.received_kwargs["strategy"])
        self.assertIsNone(RAG.received_kwargs["metadata_filter"])
        self.assertEqual(
            [
                {"company_code": company_code, "report_period": report_period}
                for company_code, report_period in expected_bindings
            ],
            [target["metadata_filter"] for target in RAG.received_kwargs["subquery_targets"]],
        )


if __name__ == "__main__":
    unittest.main()
