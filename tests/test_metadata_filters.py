import unittest

from langchain_core.documents import Document

from rag_qa.core.new_rag_system import RAGSystem
from rag_qa.core.vector_store import VectorStore


def make_document(parent_id):
    return Document(page_content=parent_id, metadata={"parent_id": parent_id})


class RecordingVectorStore:
    def __init__(self):
        self.single_calls = []
        self.subquery_calls = []

    def hybrid_search_with_rerank(self, query, **kwargs):
        self.single_calls.append({"query": query, **kwargs})
        return [make_document("single-parent")]

    def hybrid_search_subqueries_with_batched_embedding(self, subqueries, **kwargs):
        self.subquery_calls.append({"subqueries": subqueries, **kwargs})
        results = [[make_document(f"parent-{index}")] for index, _ in enumerate(subqueries)]
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


def make_rag_system(vector_store, llm_output="generated retrieval text"):
    rag = RAGSystem.__new__(RAGSystem)
    rag.vector_store = vector_store
    rag.llm = lambda prompt: llm_output
    return rag


class MetadataFilterExpressionTests(unittest.TestCase):
    def test_company_and_single_period_expression(self):
        expression = VectorStore.build_metadata_filter_expression(
            {"company_code": "600519", "report_period": "2025H1"}
        )

        self.assertEqual('company_code == "600519" AND report_period == "2025H1"', expression)

    def test_company_and_period_type_expression(self):
        expression = VectorStore.build_metadata_filter_expression(
            {"company_code": "600519", "period_type": "H1"}
        )

        self.assertEqual('company_code == "600519" AND period_type == "H1"', expression)

    def test_source_company_period_expression_and_leading_zero(self):
        expression = VectorStore.build_metadata_filter_expression(
            {
                "source": "annual_reports",
                "company_code": "000001",
                "report_period": "2025FY",
            }
        )

        self.assertEqual(
            'source == "annual_reports" AND company_code == "000001" AND report_period == "2025FY"',
            expression,
        )

    def test_report_year_requires_integer(self):
        self.assertEqual(
            "report_year == 2025",
            VectorStore.build_metadata_filter_expression({"report_year": 2025}),
        )
        with self.assertRaisesRegex(TypeError, "report_year must be an integer"):
            VectorStore.build_metadata_filter_expression({"report_year": "2025"})

    def test_none_is_ignored(self):
        expression = VectorStore.build_metadata_filter_expression(
            {"source": None, "company_code": "600519", "report_period": None}
        )

        self.assertEqual('company_code == "600519"', expression)

    def test_none_filter_preserves_an_unfiltered_expression(self):
        self.assertEqual("", VectorStore.build_metadata_filter_expression(None))

    def test_unknown_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported metadata filter fields"):
            VectorStore.build_metadata_filter_expression({"raw_expr": "1 == 1"})

    def test_quotes_and_special_characters_are_escaped(self):
        expression = VectorStore.build_metadata_filter_expression(
            {"company_name": '甲" OR report_year == 2025 OR "', "source": "annual\\reports"}
        )

        self.assertEqual(
            'source == "annual\\\\reports" AND company_name == "甲\\" OR report_year == 2025 OR \\""',
            expression,
        )

    def test_legacy_source_filter_uses_the_same_builder(self):
        expression = VectorStore.build_metadata_filter_expression(
            {"company_code": "600519"}, source_filter="annual_reports"
        )

        self.assertEqual(
            'source == "annual_reports" AND company_code == "600519"',
            expression,
        )


class MetadataFilterPropagationTests(unittest.TestCase):
    METADATA_FILTER = {"company_code": "600519", "report_period": "2025H1"}

    def test_direct_filter_is_forwarded(self):
        store = RecordingVectorStore()
        rag = make_rag_system(store)

        rag.retrieve_and_merge(
            "query",
            metadata_filter=self.METADATA_FILTER,
            strategy="直接检索",
            strategy_selection_seconds=0.0,
        )

        self.assertEqual(self.METADATA_FILTER, store.single_calls[0]["metadata_filter"])

    def test_hyde_filter_is_forwarded(self):
        store = RecordingVectorStore()
        rag = make_rag_system(store)

        rag.retrieve_and_merge(
            "query",
            metadata_filter=self.METADATA_FILTER,
            strategy="假设问题检索",
            strategy_selection_seconds=0.0,
        )

        self.assertEqual("generated retrieval text", store.single_calls[0]["query"])
        self.assertEqual(self.METADATA_FILTER, store.single_calls[0]["metadata_filter"])

    def test_backtracking_filter_is_forwarded(self):
        store = RecordingVectorStore()
        rag = make_rag_system(store)

        rag.retrieve_and_merge(
            "query",
            metadata_filter=self.METADATA_FILTER,
            strategy="回溯问题检索",
            strategy_selection_seconds=0.0,
        )

        self.assertEqual("generated retrieval text", store.single_calls[0]["query"])
        self.assertEqual(self.METADATA_FILTER, store.single_calls[0]["metadata_filter"])

    def test_subquery_filters_are_forwarded_in_matching_order(self):
        store = RecordingVectorStore()
        rag = make_rag_system(
            store,
            '{"subqueries": ["贵州茅台2025年上半年营业收入", "贵州茅台2026年上半年营业收入"]}',
        )
        filters = [
            {"company_code": "600519", "report_period": "2025H1"},
            {"company_code": "600519", "report_period": "2026H1"},
        ]

        rag.retrieve_and_merge(
            "comparison query",
            subquery_filters=filters,
            strategy="子查询检索",
            strategy_selection_seconds=0.0,
        )

        call = store.subquery_calls[0]
        self.assertEqual(
            ["贵州茅台2025年上半年营业收入", "贵州茅台2026年上半年营业收入"],
            call["subqueries"],
        )
        self.assertEqual(filters, call["metadata_filters"])

    def test_mismatched_subquery_filters_do_not_trigger_retrieval(self):
        store = RecordingVectorStore()
        rag = make_rag_system(store, '{"subqueries": ["q1", "q2"]}')

        contexts = rag.retrieve_and_merge(
            "comparison query",
            subquery_filters=[{"report_period": "2025H1"}],
            strategy="子查询检索",
            strategy_selection_seconds=0.0,
        )

        self.assertEqual([], contexts)
        self.assertEqual([], store.subquery_calls)

    def test_none_filter_preserves_unfiltered_direct_call(self):
        store = RecordingVectorStore()
        rag = make_rag_system(store)

        rag.retrieve_and_merge(
            "query",
            metadata_filter=None,
            strategy="直接检索",
            strategy_selection_seconds=0.0,
        )

        self.assertIsNone(store.single_calls[0]["metadata_filter"])
        self.assertIsNone(store.single_calls[0]["source_filter"])

    def test_vector_store_binds_each_subquery_filter_by_index(self):
        class SparseRow:
            col = [0]
            data = [1.0]

        store = VectorStore.__new__(VectorStore)
        store.embedding_function = lambda queries: {
            "dense": [[index] for index, _ in enumerate(queries)],
            "sparse": [SparseRow() for _ in queries],
        }
        captured_filters = []

        def fake_search(dense, sparse, k, source_filter=None, metadata_filter=None):
            captured_filters.append(metadata_filter)
            return [make_document(f"parent-{len(captured_filters)}")], {
                "hybrid_search_seconds": 0.0,
                "parent_dedup_seconds": 0.0,
            }

        store._search_parent_docs = fake_search
        filters = [
            {"company_code": "600519", "report_period": "2025H1"},
            {"company_code": "600519", "report_period": "2026H1"},
        ]

        store.hybrid_search_subqueries_with_batched_embedding(
            ["query-2025", "query-2026"], metadata_filters=filters
        )

        self.assertEqual(filters, captured_filters)

    def test_vector_store_rejects_mismatched_subquery_filter_count(self):
        with self.assertRaisesRegex(ValueError, "same length as subqueries"):
            VectorStore._validated_subquery_metadata_filters(
                ["query-2025", "query-2026"], [{"report_period": "2025H1"}]
            )


if __name__ == "__main__":
    unittest.main()
