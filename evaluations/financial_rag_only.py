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
RUNTIME_ENVIRONMENT_KEYS = (
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_BASE_URL",
    "LLM_MODEL",
    "MILVUS_HOST",
    "MILVUS_PORT",
    "MILVUS_DATABASE",
    "MILVUS_COLLECTION",
    "BGE_M3_MODEL_PATH",
    "RERANKER_MODEL_PATH",
    "RETRIEVAL_K",
    "CANDIDATE_M",
)


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


def build_capture_environment(parent_environment: dict[str, str], runtime_config: Any) -> dict[str, str]:
    """Return the minimum child environment needed by a detached RAG worktree.

    The parent process has already loaded the current project's ``.env`` via
    ``base.config``.  Only RAG/LLM/Milvus settings are copied into the child;
    FAQ MySQL and Redis variables are deliberately removed.  Callers must not
    serialize this mapping because it contains the API key required by Qwen.
    """
    environment = dict(parent_environment)
    for key in tuple(environment):
        if key.startswith(("MYSQL_", "REDIS_")):
            environment.pop(key, None)
    values = {
        "DASHSCOPE_API_KEY": runtime_config.DASHSCOPE_API_KEY,
        "DASHSCOPE_BASE_URL": runtime_config.DASHSCOPE_BASE_URL,
        "LLM_MODEL": runtime_config.LLM_MODEL,
        "MILVUS_HOST": runtime_config.MILVUS_HOST,
        "MILVUS_PORT": str(runtime_config.MILVUS_PORT),
        "MILVUS_DATABASE": runtime_config.MILVUS_DATABASE_NAME,
        "MILVUS_COLLECTION": runtime_config.MILVUS_COLLECTION_NAME,
        "BGE_M3_MODEL_PATH": runtime_config.BGE_M3_MODEL_PATH,
        "RERANKER_MODEL_PATH": runtime_config.RERANKER_MODEL_PATH,
        "RETRIEVAL_K": str(runtime_config.RETRIEVAL_K),
        "CANDIDATE_M": str(runtime_config.CANDIDATE_M),
    }
    environment.update({key: value for key, value in values.items() if value})
    return environment


def _streaming_llm(client: Any, model: str):
    def call(prompt: str):
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是严谨的金融分析助手，必须遵循用户消息中的资料边界。"},
                {"role": "user", "content": prompt},
            ],
            timeout=30,
            stream=True,
            extra_body={"enable_thinking": False},
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return call


def build_financial_rag_system(
    *,
    runtime_config: Any,
    openai_class: Any,
    vector_store_class: Any,
    query_router_class: Any,
    rag_system_class: Any,
) -> Any:
    """Construct only the dependency graph needed by ``RAGSystem``.

    This factory intentionally has no dependency on ``new_main``,
    ``IntegratedQASystem``, ``MysqlClient``, FAQ, or Redis.  Dependency
    injection keeps this invariant unit-testable without a live model or DB.
    """
    if not runtime_config.DASHSCOPE_API_KEY:
        raise RuntimeError("financial_rag_only capture requires an LLM API key in the subprocess environment")
    if not runtime_config.DASHSCOPE_BASE_URL or not runtime_config.LLM_MODEL:
        raise RuntimeError("financial_rag_only capture requires LLM base URL and model configuration")
    client = openai_class(
        api_key=runtime_config.DASHSCOPE_API_KEY,
        base_url=runtime_config.DASHSCOPE_BASE_URL,
    )
    vector_store = vector_store_class(
        collection_name=runtime_config.MILVUS_COLLECTION_NAME,
        host=runtime_config.MILVUS_HOST,
        port=runtime_config.MILVUS_PORT,
        database=runtime_config.MILVUS_DATABASE_NAME,
    )
    query_router = query_router_class(client=client, model=runtime_config.LLM_MODEL)
    return rag_system_class(
        vector_store=vector_store,
        llm=_streaming_llm(client, runtime_config.LLM_MODEL),
        query_router=query_router,
    )


def build_financial_rag_system_from_active_code() -> Any:
    """Import RAG-only dependencies after a worker activates its code root."""
    from base.config import config
    from openai import OpenAI
    from rag_qa.core.new_rag_system import RAGSystem
    from rag_qa.core.query_router import FinancialQueryRouter
    from rag_qa.core.vector_store import VectorStore

    return build_financial_rag_system(
        runtime_config=config,
        openai_class=OpenAI,
        vector_store_class=VectorStore,
        query_router_class=FinancialQueryRouter,
        rag_system_class=RAGSystem,
    )
