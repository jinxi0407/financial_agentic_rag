import tempfile
import unittest
from pathlib import Path

from rag_qa.core.document_processor import (
    build_child_id,
    build_milvus_primary_key,
    build_parent_id,
    calculate_file_sha256,
    load_documents_from_directory,
    process_documents,
)
from rag_qa.core.vector_store import VectorStore


class DocumentIdentityTests(unittest.TestCase):
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
