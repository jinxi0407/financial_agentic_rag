import unittest

from langchain_core.documents import Document

from base.config import config
from rag_qa.core.financial_calculator import (
    build_calculation_guardrail,
    build_calculation_note,
    document_mentions_metric,
)
from rag_qa.core.financial_evidence import extract_verified_evidence
from rag_qa.core.new_rag_system import RAGSystem
from rag_qa.core.query_metadata import extract_query_metadata


def make_document(parent_id, company_code, report_period, text, rerank_score=None):
    company_names = {
        "600519": "贵州茅台",
        "600036": "招商银行",
        "000001": "平安银行",
        "002594": "比亚迪",
    }
    return Document(
        page_content=text,
        metadata={
            "parent_id": parent_id,
            "company_name": company_names.get(company_code, company_code),
            "company_code": company_code,
            "report_period": report_period,
            "rerank_score": rerank_score,
        },
    )


class DynamicContextSelectionTests(unittest.TestCase):
    def test_simple_single_metric_request_keeps_existing_top_three_order(self):
        metadata = extract_query_metadata("贵州茅台2026年上半年营业收入是多少？")
        docs = [
            make_document(f"parent-{index}", "600519", "2026H1", f"doc-{index}")
            for index in range(5)
        ]

        self.assertEqual(config.CANDIDATE_M, RAGSystem._context_limit(metadata))
        self.assertEqual(
            ["parent-0", "parent-1", "parent-2"],
            [
                doc.metadata["parent_id"]
                for doc in RAGSystem._select_context_docs(
                    docs, config.CANDIDATE_M, metadata
                )
            ],
        )

    def test_multi_target_multi_metric_context_covers_targets_and_metrics(self):
        metadata = extract_query_metadata(
            "比较招商银行和平安银行2025年度的不良贷款率和拨备覆盖率。"
        )
        docs = [
            make_document("cmb-npl", "600036", "2025FY", "不良贷款率\n0.94%"),
            make_document("pab-npl", "000001", "2025FY", "不良贷款率\n1.05%"),
            make_document("cmb-coverage", "600036", "2025FY", "拨备覆盖率\n391.79%"),
            make_document("pab-coverage", "000001", "2025FY", "拨备覆盖率\n230.00%"),
            make_document("duplicate", "600036", "2025FY", "不良贷款率\n0.94%"),
        ]

        context_limit = RAGSystem._context_limit(metadata)
        selected = RAGSystem._select_context_docs(docs, context_limit, metadata)
        parent_ids = [doc.metadata["parent_id"] for doc in selected]

        self.assertEqual(4, context_limit)
        self.assertEqual(
            {"cmb-npl", "pab-npl", "cmb-coverage", "pab-coverage"}, set(parent_ids)
        )
        self.assertEqual(len(parent_ids), len(set(parent_ids)))

    def test_dynamic_pool_is_bounded_and_keeps_milvus_k(self):
        metadata = extract_query_metadata(
            "比较招商银行和平安银行2025年度的不良贷款率和拨备覆盖率。"
        )

        class Store:
            call = None

            def hybrid_search_subqueries_with_batched_embedding(self, subqueries, **kwargs):
                self.call = {"subqueries": subqueries, **kwargs}
                docs = [
                    [
                        make_document(
                            f"{index}-{rank}",
                            "600036" if index == 0 else "000001",
                            "2025FY",
                            "不良贷款率\n1.05%\n拨备覆盖率\n230.00%",
                        )
                        for rank in range(8)
                    ]
                    for index, _ in enumerate(subqueries)
                ]
                diagnostics = {
                    "timing": {
                        "embedding_seconds": 0.0,
                        "hybrid_search_seconds": 0.0,
                        "parent_dedup_seconds": 0.0,
                        "reranker_seconds": 0.0,
                    },
                    "reranker_predict_calls": 1,
                    "per_subquery": [
                        {
                            "query": query,
                            "hybrid_search_seconds": 0.0,
                            "parent_dedup_seconds": 0.0,
                            "parent_count": 8,
                        }
                        for query in subqueries
                    ],
                }
                return docs, diagnostics

        store = Store()
        rag = RAGSystem.__new__(RAGSystem)
        rag.vector_store = store
        contexts = rag.retrieve_and_merge(
            metadata.query,
            strategy="子查询检索",
            strategy_selection_seconds=0.0,
            subquery_targets=metadata.subquery_plan(),
            query_metadata=metadata,
        )

        self.assertEqual(config.RETRIEVAL_K, store.call["k"])
        self.assertEqual(8, store.call["result_limit"])
        self.assertEqual(4, len(contexts))

    def test_two_period_calculation_reserves_evidence_slots_for_both_periods(self):
        metadata = extract_query_metadata(
            "贵州茅台2026年上半年营业收入相比2025年上半年变化多少？"
        )

        self.assertEqual(4, RAGSystem._context_limit(metadata))

    def test_evidence_cells_choose_the_highest_rerank_score_per_target_metric(self):
        metadata = extract_query_metadata(
            "比较招商银行和平安银行2025年度的不良贷款率和拨备覆盖率。"
        )
        docs = [
            make_document("cmb-npl-low", "600036", "2025FY", "不良贷款率 0.94%", 0.1),
            make_document("cmb-npl-high", "600036", "2025FY", "不良贷款率 0.94%", 0.9),
            make_document("pab-npl", "000001", "2025FY", "不良贷款率 1.05%", 0.8),
            make_document("cmb-coverage", "600036", "2025FY", "拨备覆盖率 391.79%", 0.7),
            make_document("pab-coverage", "000001", "2025FY", "拨备覆盖率 220.88%", 0.6),
        ]

        selected = RAGSystem._select_context_docs(
            docs, RAGSystem._context_limit(metadata), metadata
        )
        cells = RAGSystem._evidence_cell_status(selected, metadata)

        self.assertEqual(4, len(cells))
        self.assertTrue(all(cell["parent_id"] for cell in cells))
        self.assertIn("cmb-npl-high", [doc.metadata["parent_id"] for doc in selected])
        self.assertNotIn("cmb-npl-low", [doc.metadata["parent_id"] for doc in selected])

    def test_more_than_six_required_cells_can_use_the_bounded_eight_context_budget(self):
        metadata = extract_query_metadata(
            "比较招商银行和平安银行2025年度的不良贷款率、拨备覆盖率、净息差和毛利率。"
        )

        self.assertEqual(8, len(RAGSystem._required_evidence_cells(metadata)))
        self.assertEqual(8, RAGSystem._context_limit(metadata))

    def test_metric_aliases_keep_research_investment_and_expense_separate(self):
        investment_doc = make_document("investment", "002594", "2025FY", "研发投入 63,441,379千元")
        expense_doc = make_document("expense", "002594", "2025FY", "研发费用 57,978,105千元")
        industry_margin_doc = make_document("industry", "002594", "2025FY", "按行业毛利率为21.9%")

        self.assertTrue(document_mentions_metric(investment_doc, "research_investment"))
        self.assertFalse(document_mentions_metric(investment_doc, "research_expense"))
        self.assertTrue(document_mentions_metric(expense_doc, "research_expense"))
        self.assertFalse(document_mentions_metric(expense_doc, "research_investment"))
        self.assertFalse(document_mentions_metric(industry_margin_doc, "gross_margin"))


class DeterministicCalculationTests(unittest.TestCase):
    def test_amount_delta_and_growth_are_added_only_from_two_evidence_values(self):
        metadata = extract_query_metadata(
            "贵州茅台2026年上半年营业收入相比2025年上半年变化多少？"
        )
        docs = [
            make_document(
                "moutai-2026", "600519", "2026H1",
                "单位：元\n本报告期 上年同期\n营业收入\n90,703,260,964.48\n89,389,354,416.84",
            ),
            make_document(
                "moutai-2025", "600519", "2025H1",
                "单位：元\n本报告期 上年同期\n营业收入\n89,389,354,416.84\n80,000,000,000.00",
            ),
        ]

        note = build_calculation_note(
            metadata, extract_verified_evidence(docs, metadata.requested_metrics)
        )

        self.assertIn("1,313,906,547.64元", note)
        self.assertIn("1.47%", note)

    def test_rate_change_uses_percentage_points_not_growth_rate(self):
        metadata = extract_query_metadata(
            "平安银行2026年上半年不良贷款率相比2025年上半年有何变化？"
        )
        docs = [
            make_document("pab-2026", "000001", "2026H1", "本报告期 上年同期\n不良贷款率\n1.05%"),
            make_document("pab-2025", "000001", "2025H1", "本报告期 上年同期\n不良贷款率\n1.00%"),
        ]

        note = build_calculation_note(
            metadata, extract_verified_evidence(docs, metadata.requested_metrics)
        )

        self.assertIn("0.05个百分点", note)
        self.assertNotIn("增长率", note)

    def test_annual_and_half_year_values_do_not_produce_a_derived_comparison(self):
        metadata = extract_query_metadata(
            "贵州茅台2025年度营业收入与2026年上半年营业收入相比变化多少。"
        )
        docs = [
            make_document("moutai-fy", "600519", "2025FY", "单位：元\n本报告期\n营业收入\n168,838,102,514.79"),
            make_document("moutai-h1", "600519", "2026H1", "单位：元\n本报告期\n营业收入\n90,703,260,964.48"),
        ]

        self.assertEqual(
            "", build_calculation_note(
                metadata, extract_verified_evidence(docs, metadata.requested_metrics)
            )
        )

    def test_missing_verified_period_value_adds_calculation_guardrail(self):
        metadata = extract_query_metadata(
            "比亚迪2026年上半年营业收入相比2025年上半年变化多少？"
        )
        docs = [
            make_document(
                "byd-2025", "002594", "2025H1",
                "单位：元\n本报告期\n营业收入\n371,280,948,000.00",
            ),
        ]
        evidence = extract_verified_evidence(docs, metadata.requested_metrics)

        self.assertEqual("", build_calculation_note(metadata, evidence))
        guardrail = build_calculation_guardrail(metadata, evidence)
        self.assertIn("营业收入", guardrail)
        self.assertIn("不得根据原始上下文中的数字自行计算", guardrail)

    def test_partial_verified_calculation_guards_only_missing_metric(self):
        metadata = extract_query_metadata(
            "贵州茅台2026年上半年营业收入和归母净利润相比2025年上半年变化多少？"
        )
        docs = [
            make_document(
                "moutai-2026", "600519", "2026H1",
                "单位：元\n本报告期\n营业收入\n90,703,260,964.48",
            ),
            make_document(
                "moutai-2025", "600519", "2025H1",
                "单位：元\n本报告期\n营业收入\n89,389,354,416.84",
            ),
        ]
        evidence = extract_verified_evidence(docs, metadata.requested_metrics)

        self.assertIn("revenue:", build_calculation_note(metadata, evidence))
        guardrail = build_calculation_guardrail(metadata, evidence)
        self.assertIn("归母净利润", guardrail)
        self.assertNotIn("营业收入、归母净利润", guardrail)

    def test_percentage_point_request_requires_verified_calculation(self):
        metadata = extract_query_metadata(
            "平安银行2026年上半年不良贷款率相比2025年上半年百分点变化多少？"
        )
        self.assertTrue(metadata.requires_deterministic_calculation())


if __name__ == "__main__":
    unittest.main()
