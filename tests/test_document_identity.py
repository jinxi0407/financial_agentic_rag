import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from rag_qa.core.document_processor import (
    build_child_id,
    build_milvus_primary_key,
    build_parent_id,
    calculate_file_sha256,
    load_documents_from_directory,
    parse_annual_report_filename,
    process_documents,
)
from rag_qa.text_splitters import ChineseRecursiveTextSplitter
from rag_qa.core.vector_store import VectorStore


class DocumentIdentityTests(unittest.TestCase):
    def test_annual_report_filename_metadata_parses_h1(self):
        metadata = parse_annual_report_filename("贵州茅台_600519_2025H1.pdf")

        self.assertEqual("贵州茅台", metadata["company_name"])
        self.assertEqual("600519", metadata["company_code"])
        self.assertEqual(2025, metadata["report_year"])
        self.assertEqual("H1", metadata["period_type"])
        self.assertEqual("2025H1", metadata["report_period"])

    def test_annual_report_filename_metadata_preserves_leading_zero_and_fy(self):
        metadata = parse_annual_report_filename("平安银行_000001_2025FY.pdf")

        self.assertEqual("000001", metadata["company_code"])
        self.assertEqual("FY", metadata["period_type"])
        self.assertEqual("2025FY", metadata["report_period"])

    def test_annual_report_filename_metadata_parses_other_company(self):
        metadata = parse_annual_report_filename("中芯国际_688981_2026H1.pdf")

        self.assertEqual("中芯国际", metadata["company_name"])
        self.assertEqual("688981", metadata["company_code"])
        self.assertEqual(2026, metadata["report_year"])

    def test_malformed_annual_report_filename_fails(self):
        with self.assertRaisesRegex(ValueError, "annual_reports PDF filename"):
            parse_annual_report_filename("贵州茅台_600519_2025年报.pdf")

    def test_non_annual_report_document_has_no_report_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            knowledge_directory = Path(temporary_directory) / "financial_knowledge"
            knowledge_directory.mkdir()
            (knowledge_directory / "overview.txt").write_text("Financial knowledge", encoding="utf-8")

            document = load_documents_from_directory(knowledge_directory)[0]

            self.assertNotIn("company_name", document.metadata)
            self.assertNotIn("report_period", document.metadata)

    def test_report_metadata_is_inherited_by_parent_and_child_chunks(self):
        metadata = parse_annual_report_filename("贵州茅台_600519_2025H1.pdf")
        document = Document(page_content="甲。乙。丙。", metadata=metadata)
        splitter = ChineseRecursiveTextSplitter(chunk_size=4, chunk_overlap=0)

        parent = splitter.split_documents([document])[0]
        parent.metadata["parent_id"] = "parent-1"
        child = splitter.split_documents([parent])[0]

        for field, value in metadata.items():
            self.assertEqual(value, parent.metadata[field])
            self.assertEqual(value, child.metadata[field])
        self.assertEqual("parent-1", child.metadata["parent_id"])

    def test_milvus_insert_payload_contains_report_metadata(self):
        class DenseVector:
            def tolist(self):
                return [0.0]

        class SparseVector:
            col = [1]
            data = [0.5]

        class FakeMilvusClient:
            def __init__(self):
                self.upserted_data = None

            def upsert(self, collection_name, data):
                self.upserted_data = data

            def flush(self, collection_name):
                return None

        report_metadata = parse_annual_report_filename("平安银行_000001_2025FY.pdf")
        child_metadata = {
            "document_id": "a" * 64,
            "file_sha256": "a" * 64,
            "source_filename": "平安银行_000001_2025FY.pdf",
            "parent_id": "parent-1",
            "parent_content": "parent content",
            "milvus_id": "b" * 64,
            "source": "annual_reports",
            "timestamp": "2026-01-01T00:00:00",
            **report_metadata,
        }
        store = VectorStore.__new__(VectorStore)
        store.collection_name = "unit_test_collection"
        store.collection_fields = set(VectorStore.INGESTION_SCHEMA_FIELDS)
        store.identity_schema_ready = True
        store.embedding_function = lambda texts: {
            "dense": [DenseVector() for _ in texts],
            "sparse": [SparseVector() for _ in texts],
        }
        store.client = FakeMilvusClient()

        store.add_documents([Document(page_content="child content", metadata=child_metadata)])

        inserted = store.client.upserted_data[0]
        for field in VectorStore.REPORT_METADATA_FIELDS:
            self.assertIn(field, inserted)
            self.assertEqual(report_metadata[field], inserted[field])

    def test_same_file_has_stable_identity_and_source_filename(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_file = Path(temporary_directory) / "report.txt"
            source_file.write_text("Financial report content", encoding="utf-8")

            document_id = calculate_file_sha256(source_file)
            documents = load_documents_from_directory(temporary_directory)

            self.assertEqual(1, len(documents))
            self.assertEqual(document_id, documents[0].metadata["document_id"])
            self.assertEqual(document_id, documents[0].metadata["file_sha256"])
            self.assertEqual("report.txt", documents[0].metadata["source_filename"])
            self.assertEqual(
                build_parent_id(document_id, 0),
                build_parent_id(document_id, 0),
            )
            self.assertEqual(
                build_child_id(document_id, 0, 0),
                build_child_id(document_id, 0, 0),
            )

    def test_existing_document_callback_skips_the_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_file = Path(temporary_directory) / "already_ingested.txt"
            source_file.write_text("Already present", encoding="utf-8")
            document_id = calculate_file_sha256(source_file)

            documents = load_documents_from_directory(
                temporary_directory,
                is_document_ingested=lambda identity: identity == document_id,
            )

            self.assertEqual([], documents)

    def test_same_text_from_different_files_has_distinct_primary_keys(self):
        shared_text = "Identical boilerplate text"
        document_a = "a" * 64
        document_b = "b" * 64

        self.assertNotEqual(
            build_milvus_primary_key(document_a, 0, 0, shared_text),
            build_milvus_primary_key(document_b, 0, 0, shared_text),
        )

    def test_same_file_has_stable_parent_child_and_milvus_ids(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_file = Path(temporary_directory) / "stable.txt"
            source_file.write_text(
                "financial reporting identity must remain stable across ingestion runs. " * 4,
                encoding="utf-8",
            )

            first_run = process_documents(
                temporary_directory,
                parent_chunk_size=80,
                child_chunk_size=30,
                chunk_overlap=0,
            )
            second_run = process_documents(
                temporary_directory,
                parent_chunk_size=80,
                child_chunk_size=30,
                chunk_overlap=0,
            )

            first_ids = [
                (chunk.metadata["parent_id"], chunk.metadata["id"], chunk.metadata["milvus_id"])
                for chunk in first_run
            ]
            second_ids = [
                (chunk.metadata["parent_id"], chunk.metadata["id"], chunk.metadata["milvus_id"])
                for chunk in second_run
            ]
            self.assertTrue(first_ids)
            self.assertEqual(first_ids, second_ids)

    def test_legacy_collection_is_blocked_before_ingestion(self):
        store = VectorStore.__new__(VectorStore)
        store.collection_name = "financial_rag_v1"
        store.collection_fields = {"id", "text", "parent_id", "source"}
        store.identity_schema_ready = False

        with self.assertRaisesRegex(RuntimeError, "legacy schema"):
            store.ensure_identity_schema_for_ingestion()


if __name__ == "__main__":
    unittest.main()
