import unittest

from langchain_core.documents import Document

from rag_qa.core.new_rag_system import RAGSystem
from rag_qa.core.vector_store import VectorStore


def make_document(parent_id):
    return Document(page_content=parent_id, metadata={"parent_id": parent_id})


class SparseRow:
    def __init__(self, value):
        self.col = [0]
        self.data = [value]


class FakeEmbeddingFunction:
    def __call__(self, queries):
        return {
            "dense": [[len(query), index] for index, query in enumerate(queries)],
            "sparse": [SparseRow(len(query)) for query in queries],
        }


class RetrievalAggregationTests(unittest.TestCase):
    def test_batch_embedding_preserves_query_order(self):
        store = VectorStore.__new__(VectorStore)
        store.embedding_function = FakeEmbeddingFunction()

        batch_embeddings = store._embed_queries(["first", "second"])
        first_single = store._embed_queries(["first"])
        second_single = store._embed_queries(["second"])

        batch_first = store._query_vectors_from_embeddings(batch_embeddings, 0)
        batch_second = store._query_vectors_from_embeddings(batch_embeddings, 1)
        self.assertEqual(first_single["dense"][0][0], batch_first[0][0])
        self.assertEqual(second_single["dense"][0][0], batch_second[0][0])
        self.assertEqual({0: 5}, batch_first[1])
        self.assertEqual({0: 6}, batch_second[1])

    def test_batch_reranker_scores_restore_to_the_correct_subquery(self):
        parent_docs_by_subquery = [
            [make_document("A1"), make_document("A2")],
            [make_document("B1"), make_document("B2")],
        ]

        restored = VectorStore._restore_subquery_rankings(
            parent_docs_by_subquery,
            [0.1, 0.9, 0.8, 0.2],
        )

        self.assertEqual(["A2", "A1"], [doc.metadata["parent_id"] for _, doc in restored[0]])
        self.assertEqual(["B1", "B2"], [doc.metadata["parent_id"] for _, doc in restored[1]])

    def test_batch_ranked_results_keep_coverage_aggregation_unchanged(self):
        parent_docs_by_subquery = [
            [make_document("A1"), make_document("A2")],
            [make_document("B1"), make_document("B2")],
            [make_document("C1"), make_document("C2")],
        ]
        restored = VectorStore._restore_subquery_rankings(
            parent_docs_by_subquery,
            [0.4, 0.9, 0.8, 0.1, 0.2, 0.7],
        )
        reranked_docs = [[doc for _, doc in pairs] for pairs in restored]

        merged = RAGSystem._merge_subquery_results_coverage_first(reranked_docs, limit=3)

        self.assertEqual(["A2", "B1", "C2"], [doc.metadata["parent_id"] for doc in merged])

    def test_reranker_tie_sort_is_stable(self):
        first = make_document("first")
        second = make_document("second")

        ranked_pairs = VectorStore._sort_scored_documents([0.5, 0.5], [first, second])

        self.assertEqual([first, second], [doc for _, doc in ranked_pairs])

    def test_subquery_merge_prioritizes_each_subquery_top_parent(self):
        results = [
            [make_document("A1"), make_document("A2"), make_document("A3")],
            [make_document("B1"), make_document("B2"), make_document("B3")],
            [make_document("C1"), make_document("C2"), make_document("C3")],
        ]

        merged = RAGSystem._merge_subquery_results_coverage_first(results, limit=3)

        self.assertEqual(["A1", "B1", "C1"], [doc.metadata["parent_id"] for doc in merged])

    def test_subquery_merge_skips_duplicate_parent_and_backfills(self):
        results = [
            [make_document("A1"), make_document("A2"), make_document("A3")],
            [make_document("A1"), make_document("B2"), make_document("B3")],
            [make_document("C1"), make_document("C2"), make_document("C3")],
        ]

        merged = RAGSystem._merge_subquery_results_coverage_first(results, limit=3)

        self.assertEqual(["A1", "B2", "C1"], [doc.metadata["parent_id"] for doc in merged])
        self.assertEqual(3, len({doc.metadata["parent_id"] for doc in merged}))


if __name__ == "__main__":
    unittest.main()
