"""Evaluation-only helpers for direct Financial RAG capture.

This module deliberately never imports or invokes ``IntegratedQASystem.query``.
It adapts the two frozen ``RAGSystem.generate_answer`` signatures so a
Baseline-versus-Final RAGAS comparison measures Financial RAG on both sides,
without allowing the application FAQ/BM25 fast path to decide an answer.
"""
from __future__ import annotations

import inspect
from decimal import Decimal
from typing import Any


ENTRYPOINT_NAME = "financial_rag_only"
EVIDENCE_CAPTURE_SCHEMA_VERSION = "financial_rag_prompt_evidence_v1"


def _json_value(value: Any) -> Any:
    """Preserve provenance while converting dataclass values to JSON-safe data."""
    if isinstance(value, Decimal):
        return str(value)
    return value


def serialize_verified_evidence(values: Any) -> list[dict[str, Any]]:
    """Serialize exactly the values returned by the live extractor.

    No answer text, gold data, or inferred provenance is added here.  The
    fields mirror ``StructuredFinancialEvidence`` so a later audit can trace a
    verified value back to the Parent used by the Answer Layer.
    """
    fields = (
        "company_name",
        "company_code",
        "report_period",
        "metric",
        "raw_value",
        "normalized_value",
        "unit",
        "normalized_unit",
        "source_parent",
        "exact_match",
        "confidence",
    )
    return [
        {field: _json_value(getattr(value, field, None)) for field in fields}
        for value in values or ()
    ]


def build_prompt_evidence_capture(
    retrieved_contexts: list[dict[str, Any]],
    verified_evidence: list[dict[str, Any]],
    verified_evidence_block: str | None,
    deterministic_evidence: list[dict[str, str]],
) -> dict[str, Any]:
    """Build the evidence contract from text actually concatenated into Prompt.

    ``ragas_contexts`` preserves the same order as ``RAGSystem.generate_answer``:
    selected Parents, then the verified-value block, followed by any
    deterministic calculation or calculation-guardrail blocks.  This is a
    capture of live prompt inputs, not a relevance-filtered replacement.
    """
    ragas_contexts = [
        context["text"]
        for context in retrieved_contexts
        if isinstance(context.get("text"), str) and context["text"]
    ]
    if verified_evidence_block:
        ragas_contexts.append(verified_evidence_block)
    for item in deterministic_evidence:
        text = item.get("text")
        if isinstance(text, str) and text:
            ragas_contexts.append(text)
    return {
        "retrieved_contexts": retrieved_contexts,
        "verified_evidence": verified_evidence,
        "deterministic_evidence": deterministic_evidence,
        "ragas_contexts": ragas_contexts,
    }


def generate_financial_rag_only(rag: Any, query: str) -> tuple[Any, dict[str, Any]]:
    """Call a version's RAG entrypoint directly, bypassing application FAQ.

    The historical Baseline accepts only the original RAG arguments.  Final
    accepts deterministic query metadata and multi-target bindings.  Signature
    inspection keeps both frozen versions on their native orchestration path
    without importing ``new_main`` or calling its user-facing ``query`` method.
    """
    parameters = inspect.signature(rag.generate_answer).parameters
    kwargs: dict[str, Any] = {}
    if "history" in parameters:
        kwargs["history"] = None

    if "query_metadata" in parameters:
        # Imported only after the capture worker activates its requested code
        # root, so Baseline and Final use their own frozen metadata module.
        from rag_qa.core.query_metadata import extract_query_metadata

        metadata = extract_query_metadata(query)
        kwargs["query_metadata"] = metadata
        if "metadata_filter" in parameters:
            kwargs["metadata_filter"] = metadata.to_metadata_filter()
        if "subquery_targets" in parameters and metadata.requires_deterministic_subqueries():
            kwargs["subquery_targets"] = metadata.subquery_plan()
        if "strategy" in parameters and metadata.requires_deterministic_subqueries():
            kwargs["strategy"] = "子查询检索"

    result = rag.generate_answer(query, **kwargs)
    return result, {
        "entrypoint": ENTRYPOINT_NAME,
        "faq_fast_path_bypassed": True,
        "rag_generate_kwargs": sorted(kwargs),
    }
