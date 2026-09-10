import unittest
import inspect
from decimal import Decimal

from evaluations.financial_rag_only import (
    ENTRYPOINT_NAME,
    build_prompt_evidence_capture,
    build_capture_environment,
    build_financial_rag_system,
    generate_financial_rag_only,
    serialize_verified_evidence,
)
from evaluations.run_ragas_final_evaluation import (
    PILOT_CATEGORY_COUNTS,
    _manifest_dataset_provenance,
    build_pilot_manifest,
    fairness_dry_run,
    _capture_cases,
)


class _Evidence:
    company_name = "贵州茅台"
    company_code = "600519"
    report_period = "2026H1"
    metric = "revenue"
    raw_value = "10"
    normalized_value = Decimal("100000")
    unit = "万元"
    normalized_unit = "元"
    source_parent = "parent-1"
    exact_match = True
    confidence = "high"


class RagasEvaluationTests(unittest.TestCase):
    def test_capture_path_does_not_reference_integrated_qa(self):
        source = inspect.getsource(_capture_cases)
        self.assertNotIn("IntegratedQASystem()", source)
        self.assertNotIn("from new_main", source)
        self.assertNotIn("MysqlClient", source)
        self.assertNotIn("BM25Search", source)
        self.assertNotIn("DASHSCOPE_API_KEY", source)

    def test_rag_factory_builds_only_direct_dependencies(self):
        class Runtime:
            DASHSCOPE_API_KEY = "test-key"
            DASHSCOPE_BASE_URL = "https://example.invalid/v1"
            LLM_MODEL = "qwen3.8-max"
            MILVUS_COLLECTION_NAME = "financial_rag_v1"
            MILVUS_HOST = "127.0.0.1"
            MILVUS_PORT = "19530"
            MILVUS_DATABASE_NAME = "financial"

        calls = []

        class OpenAI:
            def __init__(self, **kwargs):
                calls.append(("openai", kwargs))

        class VectorStore:
            def __init__(self, **kwargs):
                calls.append(("vector_store", kwargs))

        class Router:
            def __init__(self, **kwargs):
                calls.append(("router", kwargs))

        class RAG:
            def __init__(self, **kwargs):
                calls.append(("rag", kwargs))

        result = build_financial_rag_system(
            runtime_config=Runtime(), openai_class=OpenAI,
            vector_store_class=VectorStore, query_router_class=Router,
            rag_system_class=RAG,
        )
        self.assertIsInstance(result, RAG)
        self.assertEqual([name for name, _ in calls], ["openai", "vector_store", "router", "rag"])
        self.assertEqual(calls[1][1]["database"], "financial")

    def test_subprocess_environment_keeps_rag_values_without_faq_values(self):
        class Runtime:
            DASHSCOPE_API_KEY = "test-key"
            DASHSCOPE_BASE_URL = "https://example.invalid/v1"
            LLM_MODEL = "qwen3.8-max"
            MILVUS_HOST = "127.0.0.1"
            MILVUS_PORT = "19530"
            MILVUS_DATABASE_NAME = "financial"
            MILVUS_COLLECTION_NAME = "financial_rag_v1"
            BGE_M3_MODEL_PATH = "/models/bge"
            RERANKER_MODEL_PATH = "/models/reranker"
            RETRIEVAL_K = 30
            CANDIDATE_M = 3

        environment = build_capture_environment(
            {"MYSQL_HOST": "mysql", "REDIS_HOST": "redis", "PATH": "/bin"}, Runtime()
        )
        self.assertNotIn("MYSQL_HOST", environment)
        self.assertNotIn("REDIS_HOST", environment)
        self.assertEqual(environment["MILVUS_HOST"], "127.0.0.1")
        self.assertEqual(environment["RETRIEVAL_K"], "30")
        self.assertEqual(environment["DASHSCOPE_API_KEY"], "test-key")

    def test_historical_adapter_calls_rag_without_an_faq_object(self):
        class HistoricalRag:
            def __init__(self):
                self.calls = []

            def generate_answer(self, query, history=None, source_filter=None, metadata_filter=None, subquery_filters=None):
                self.calls.append({"query": query, "history": history})
                return "direct RAG answer"

        rag = HistoricalRag()
        answer, trace = generate_financial_rag_only(rag, "什么是营业收入？")
        self.assertEqual(answer, "direct RAG answer")
        self.assertEqual(rag.calls, [{"query": "什么是营业收入？", "history": None}])
        self.assertEqual(trace["entrypoint"], ENTRYPOINT_NAME)
        self.assertTrue(trace["faq_fast_path_bypassed"])

    def test_prompt_capture_keeps_actual_order_and_provenance(self):
        verified = serialize_verified_evidence([_Evidence()])
        capture = build_prompt_evidence_capture(
            [{"text": "Parent evidence", "metadata": {"parent_id": "parent-1"}}],
            verified,
            "【系统已验证财务数值】\n- 来源 Parent：parent-1。",
            [{"kind": "deterministic_calculation", "text": "【系统基于已验证财务数值的确定性计算】"}],
        )
        self.assertEqual(capture["verified_evidence"][0]["source_parent"], "parent-1")
        self.assertEqual(capture["verified_evidence"][0]["normalized_value"], "100000")
        self.assertEqual(
            capture["ragas_contexts"],
            [
                "Parent evidence",
                "【系统已验证财务数值】\n- 来源 Parent：parent-1。",
                "【系统基于已验证财务数值的确定性计算】",
            ],
        )

    def test_pilot_is_deterministic_subset_with_all_categories(self):
        first = build_pilot_manifest(write=False)
        second = build_pilot_manifest(write=False)
        self.assertEqual(first, second)
        self.assertEqual(len(first["samples"]), 20)
        self.assertEqual(first["sample_category_counts"], PILOT_CATEGORY_COUNTS)
        self.assertEqual(first["source_dataset_schema_version"], "financial_holdout_300_v1_1")

    def test_legacy_manifest_without_optional_source_schema_is_safe(self):
        provenance = _manifest_dataset_provenance({
            "schema_version": "legacy_pilot",
            "source_dataset": "frozen.json",
            "source_dataset_sha256": "fingerprint",
        })
        self.assertIsNone(provenance["dataset_version"])
        self.assertFalse(provenance["source_dataset_schema_version_recorded"])
        self.assertEqual(provenance["source_dataset_sha256"], "fingerprint")

    def test_dry_run_declares_faq_bypass_for_shared_smoke_ids(self):
        manifest = build_pilot_manifest(write=False)
        audit = fairness_dry_run(manifest, [sample["sample_id"] for sample in manifest["samples"][:5]])
        self.assertTrue(audit["passed"])
        for version in ("baseline", "final"):
            for row in audit["versions"][version]:
                self.assertEqual(row["entrypoint"], ENTRYPOINT_NAME)
                self.assertTrue(row["faq_fast_path_bypassed"])


if __name__ == "__main__":
    unittest.main()
