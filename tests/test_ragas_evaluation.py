import unittest
from decimal import Decimal

from evaluations.financial_rag_only import (
    ENTRYPOINT_NAME,
    build_prompt_evidence_capture,
    generate_financial_rag_only,
    serialize_verified_evidence,
)
from evaluations.run_ragas_final_evaluation import (
    PILOT_CATEGORY_COUNTS,
    build_pilot_manifest,
    fairness_dry_run,
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
