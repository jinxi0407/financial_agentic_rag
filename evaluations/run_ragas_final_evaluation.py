"""Prepare and run the frozen 300Q RAGAS final evaluation.

The runner deliberately separates three concerns:

* deterministic manifest creation from the frozen 300Q v1.1 dataset;
* answer/context capture from a versioned RAG code root; and
* RAGAS scoring of the captured, real answer-generation contexts.

It never substitutes gold documents for retrieved contexts and does not modify
the production RAG.  ``run`` requires ``--confirm-run``; use ``smoke`` for the
five-case validation only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # Supports both ``python -m evaluations...`` and capture-worker scripts.
    from evaluations.financial_rag_only import (
        ENTRYPOINT_NAME,
        EVIDENCE_CAPTURE_SCHEMA_VERSION,
        build_capture_environment,
        build_financial_rag_system_from_active_code,
        build_prompt_evidence_capture,
        generate_financial_rag_only,
        serialize_verified_evidence,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by the worker process.
    from financial_rag_only import (  # type: ignore[no-redef]
        ENTRYPOINT_NAME,
        EVIDENCE_CAPTURE_SCHEMA_VERSION,
        build_capture_environment,
        build_financial_rag_system_from_active_code,
        build_prompt_evidence_capture,
        generate_financial_rag_only,
        serialize_verified_evidence,
    )


ROOT = Path(__file__).resolve().parents[1]
EVALUATIONS = ROOT / "evaluations"
DATASET_PATH = EVALUATIONS / "financial_holdout_300_v1_1.json"
MANIFEST_PATH = EVALUATIONS / "ragas_100_stratified_v1.json"
PILOT_MANIFEST_PATH = EVALUATIONS / "ragas_20_pilot_v1.json"
BASELINE_COMMIT = "bcf70ae"
BASELINE_WORKTREE = Path(tempfile.gettempdir()) / "financial_rag_ragas_baseline_bcf70ae"
FINAL_RESULTS_PATH = EVALUATIONS / "ragas_final_100_v1_results.json"
BASELINE_RESULTS_PATH = EVALUATIONS / "ragas_baseline_100_v1_results.json"
SUMMARY_PATH = EVALUATIONS / "ragas_baseline_vs_final_100_v1_summary.json"
SEED = 42
SAMPLE_SIZE = 100
RUNTIME_ENV = {"RETRIEVAL_K": "30", "CANDIDATE_M": "3"}
PILOT_SIZE = 20
PILOT_CATEGORY_COUNTS = {
    "complex_multi_target": 1,
    "explanation_risk": 1,
    "metric_alias_ambiguity": 2,
    "multi_company_multi_period": 3,
    "multi_company_single_period": 4,
    "negative_oos_invalid": 2,
    "single_company_multi_period": 3,
    "single_company_single_period": 4,
}
SMOKE_SAMPLE_IDS = ("holdout_201", "holdout_251", "holdout_221", "holdout_161", "holdout_103")
EVALUATION_COLLECTION = "financial_rag_v1"
EVALUATION_DATABASE = "financial"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(code_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=code_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _category_quotas(counts: Counter[str], sample_size: int) -> dict[str, int]:
    total = sum(counts.values())
    if total < sample_size:
        raise ValueError(f"dataset has only {total} cases, cannot sample {sample_size}")
    raw = {category: count * sample_size / total for category, count in counts.items()}
    quotas = {category: math.floor(value) for category, value in raw.items()}
    remaining = sample_size - sum(quotas.values())
    for category in sorted(
        counts,
        key=lambda item: (raw[item] - quotas[item], item),
        reverse=True,
    )[:remaining]:
        quotas[category] += 1
    return quotas


def build_manifest(dataset_path: Path = DATASET_PATH, output_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    dataset = _load(dataset_path)
    cases = dataset.get("cases", [])
    if len(cases) != 300:
        raise ValueError(f"expected frozen 300Q dataset, got {len(cases)} cases")
    categories = Counter(case["category"] for case in cases)
    quotas = _category_quotas(categories, SAMPLE_SIZE)
    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    for category in sorted(categories):
        group = sorted(
            (case for case in cases if case["category"] == category),
            key=lambda case: case["id"],
        )
        chosen = rng.sample(group, quotas[category])
        selected.extend(sorted(chosen, key=lambda case: case["id"]))
    selected.sort(key=lambda case: case["id"])
    manifest = {
        "schema_version": "ragas_100_stratified_v1",
        "source_dataset": dataset_path.name,
        "source_dataset_schema_version": dataset.get("schema_version"),
        "source_dataset_sha256": _sha256(dataset_path),
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "sampling_strategy": "category-stratified largest-remainder quota; random.Random(seed=42) within category",
        "source_category_counts": dict(sorted(categories.items())),
        "sample_category_counts": dict(sorted(quotas.items())),
        "samples": [
            {
                "sample_id": case["id"],
                "source_dataset": dataset_path.name,
                "original_category": case["category"],
                "seed": SEED,
                "sampling_strategy": "category-stratified",
                "case": case,
            }
            for case in selected
        ],
    }
    audit_manifest(manifest, dataset_path)
    _dump(output_path, manifest)
    return manifest


def audit_manifest(manifest: dict[str, Any], dataset_path: Path = DATASET_PATH) -> dict[str, Any]:
    source = _load(dataset_path)
    source_by_id = {case["id"]: case for case in source.get("cases", [])}
    samples = manifest.get("samples", [])
    errors: list[str] = []
    if manifest.get("sample_size") != SAMPLE_SIZE or len(samples) != SAMPLE_SIZE:
        errors.append("sample size must be exactly 100")
    ids = [sample.get("sample_id") for sample in samples]
    if len(set(ids)) != len(ids):
        errors.append("sample IDs are not unique")
    if manifest.get("source_dataset_sha256") != _sha256(dataset_path):
        errors.append("source dataset fingerprint mismatch")
    for sample in samples:
        source_case = source_by_id.get(sample.get("sample_id"))
        if source_case is None or sample.get("case") != source_case:
            errors.append(f"sample does not exactly match source case: {sample.get('sample_id')}")
    actual_counts = Counter(sample["original_category"] for sample in samples)
    if dict(sorted(actual_counts.items())) != manifest.get("sample_category_counts"):
        errors.append("manifest category counts do not match samples")
    if set(actual_counts) != set(Counter(case["category"] for case in source["cases"])):
        errors.append("manifest omitted a frozen dataset category")
    return {
        "passed": not errors,
        "errors": errors,
        "sample_size": len(samples),
        "category_counts": dict(sorted(actual_counts.items())),
    }


def build_pilot_manifest(
    manifest_path: Path = MANIFEST_PATH,
    output_path: Path = PILOT_MANIFEST_PATH,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Create a fixed 20-case, eight-category pilot from the frozen 100Q set."""
    source_manifest = _load(manifest_path)
    audit = audit_manifest(source_manifest)
    if not audit["passed"]:
        raise ValueError("cannot build pilot from an invalid 100Q manifest: " + "; ".join(audit["errors"]))
    samples = source_manifest["samples"]
    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    for category, quota in sorted(PILOT_CATEGORY_COUNTS.items()):
        group = sorted(
            (sample for sample in samples if sample["original_category"] == category),
            key=lambda sample: sample["sample_id"],
        )
        if len(group) < quota:
            raise ValueError(f"pilot category {category} has {len(group)} samples, needs {quota}")
        selected.extend(sorted(rng.sample(group, quota), key=lambda sample: sample["sample_id"]))
    selected.sort(key=lambda sample: sample["sample_id"])
    pilot = {
        "schema_version": "ragas_20_pilot_v1",
        "source_manifest": manifest_path.name,
        "source_manifest_sha256": _sha256(manifest_path),
        "source_dataset": source_manifest["source_dataset"],
        "source_dataset_sha256": source_manifest["source_dataset_sha256"],
        "sample_size": PILOT_SIZE,
        "seed": SEED,
        "sampling_strategy": "fixed category-stratified subset of ragas_100_stratified_v1; random.Random(seed=42)",
        "sample_category_counts": dict(PILOT_CATEGORY_COUNTS),
        "samples": selected,
    }
    pilot_audit = audit_pilot_manifest(pilot, manifest_path)
    if not pilot_audit["passed"]:
        raise ValueError("pilot audit failed: " + "; ".join(pilot_audit["errors"]))
    if write:
        _dump(output_path, pilot)
    return pilot


def audit_pilot_manifest(pilot: dict[str, Any], manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    source_manifest = _load(manifest_path)
    source_by_id = {sample["sample_id"]: sample for sample in source_manifest.get("samples", [])}
    samples = pilot.get("samples", [])
    errors: list[str] = []
    if pilot.get("sample_size") != PILOT_SIZE or len(samples) != PILOT_SIZE:
        errors.append("pilot size must be exactly 20")
    ids = [sample.get("sample_id") for sample in samples]
    if len(ids) != len(set(ids)):
        errors.append("pilot IDs are not unique")
    for sample in samples:
        if source_by_id.get(sample.get("sample_id")) != sample:
            errors.append(f"pilot sample does not exactly match 100Q manifest: {sample.get('sample_id')}")
    counts = dict(sorted(Counter(sample["original_category"] for sample in samples).items()))
    if counts != dict(sorted(PILOT_CATEGORY_COUNTS.items())):
        errors.append("pilot category counts do not match fixed quotas")
    if pilot.get("source_manifest_sha256") != _sha256(manifest_path):
        errors.append("pilot source manifest fingerprint mismatch")
    return {"passed": not errors, "errors": errors, "sample_size": len(samples), "category_counts": counts}


def fairness_dry_run(
    manifest: dict[str, Any], sample_ids: list[str] | tuple[str, ...] = SMOKE_SAMPLE_IDS,
) -> dict[str, Any]:
    """Validate the entrypoint contract without initializing RAG, FAQ, or Qwen."""
    known_ids = {sample["sample_id"] for sample in manifest.get("samples", [])}
    missing_ids = [sample_id for sample_id in sample_ids if sample_id not in known_ids]
    versions = {
        version: [
            {
                "sample_id": sample_id,
                "entrypoint": ENTRYPOINT_NAME,
                "faq_fast_path_bypassed": True,
                "retrieval_k": int(RUNTIME_ENV["RETRIEVAL_K"]),
                "candidate_m": int(RUNTIME_ENV["CANDIDATE_M"]),
                "milvus_database": EVALUATION_DATABASE,
                "milvus_collection": EVALUATION_COLLECTION,
            }
            for sample_id in sample_ids
            if sample_id in known_ids
        ]
        for version in ("baseline", "final")
    }
    return {
        "passed": not missing_ids and versions["baseline"] == versions["final"],
        "sample_ids": list(sample_ids),
        "missing_sample_ids": missing_ids,
        "versions": versions,
        "note": "Dry-run only: it proves the evaluation adapter contract and makes no RAG, FAQ, Qwen, or Milvus call.",
    }


def _activate_code_root(code_root: Path) -> None:
    """Ensure a capture worker imports the requested version, not this runner."""
    code_root = code_root.resolve()
    runner_root = ROOT.resolve()
    sys.path[:] = [
        entry for entry in sys.path
        if Path(entry or os.getcwd()).resolve() not in {runner_root, runner_root / "evaluations"}
    ]
    sys.path.insert(0, str(code_root))
    os.chdir(code_root)
    for module_name in list(sys.modules):
        if module_name == "new_main" or module_name.startswith(("base.", "rag_qa.", "mysql_qa.")):
            sys.modules.pop(module_name, None)


def _document_context(document: Any) -> dict[str, Any]:
    metadata = getattr(document, "metadata", {}) or {}
    return {
        "text": getattr(document, "page_content", ""),
        "metadata": {
            "parent_id": metadata.get("parent_id"),
            "document_id": metadata.get("document_id"),
            "source_filename": metadata.get("source_filename"),
            "company_code": metadata.get("company_code"),
            "report_period": metadata.get("report_period"),
            "rerank_score": metadata.get("rerank_score"),
        },
    }


def _instrument_prompt_evidence(active_trace: dict[str, Any]):
    """Observe Final Answer-Layer evidence without changing its behavior.

    The hooks wrap the exact module-level helpers called by the active
    ``RAGSystem.generate_answer``.  Historical Baseline does not contain these
    helpers, so its unified schema correctly remains Parent-only.
    """
    try:
        import rag_qa.core.new_rag_system as rag_module
    except ImportError:
        return lambda: None

    required = (
        "extract_verified_evidence",
        "format_verified_evidence_block",
        "build_calculation_note",
        "build_calculation_guardrail",
    )
    if not all(hasattr(rag_module, name) for name in required):
        return lambda: None

    originals = {name: getattr(rag_module, name) for name in required}

    def extract_verified(*args: Any, **kwargs: Any):
        values = originals["extract_verified_evidence"](*args, **kwargs)
        active_trace["verified_evidence"] = serialize_verified_evidence(values)
        return values

    def format_verified(values: Any):
        block = originals["format_verified_evidence_block"](values)
        active_trace["verified_evidence_block"] = block
        return block

    def calculation_note(*args: Any, **kwargs: Any):
        text = originals["build_calculation_note"](*args, **kwargs)
        if text:
            active_trace.setdefault("deterministic_evidence", []).append({
                "kind": "deterministic_calculation",
                "text": text,
            })
        return text

    def calculation_guardrail(*args: Any, **kwargs: Any):
        text = originals["build_calculation_guardrail"](*args, **kwargs)
        if text:
            active_trace.setdefault("deterministic_evidence", []).append({
                "kind": "calculation_guardrail",
                "text": text,
            })
        return text

    rag_module.extract_verified_evidence = extract_verified
    rag_module.format_verified_evidence_block = format_verified
    rag_module.build_calculation_note = calculation_note
    rag_module.build_calculation_guardrail = calculation_guardrail

    def restore() -> None:
        for name, original in originals.items():
            setattr(rag_module, name, original)

    return restore


def _capture_cases(code_root: Path, manifest: dict[str, Any], limit: int | None) -> dict[str, Any]:
    _activate_code_root(code_root)
    from base.config import config

    # bcf70ae predates environment-based retrieval overrides. Mutating this
    # process-local Config instance preserves the historical orchestration
    # while enforcing the frozen comparison contract (K=30, M=3).
    config.RETRIEVAL_K = int(RUNTIME_ENV["RETRIEVAL_K"])
    config.CANDIDATE_M = int(RUNTIME_ENV["CANDIDATE_M"])

    samples = list(manifest["samples"])
    if limit is not None:
        samples = _smoke_samples(samples, limit)
    # Do not instantiate IntegratedQASystem: its constructor opens FAQ MySQL,
    # FAQ Redis and conversation storage.  This direct factory constructs only
    # OpenAI, FinancialQueryRouter, VectorStore and the versioned RAGSystem.
    rag_system = build_financial_rag_system_from_active_code()
    active_trace: dict[str, Any] = {}
    original_retrieve = rag_system.retrieve_and_merge

    def retrieve_and_capture(*args: Any, **kwargs: Any):
        documents = original_retrieve(*args, **kwargs)
        active_trace["retrieved_contexts"] = [
            _document_context(document) for document in documents
        ]
        return documents

    rag_system.retrieve_and_merge = retrieve_and_capture
    results = []
    restore_evidence_hooks = _instrument_prompt_evidence(active_trace)
    try:
        for index, sample in enumerate(samples, start=1):
            case = sample["case"]
            active_trace.clear()
            started = time.perf_counter()
            parts: list[str] = []
            error = None
            invocation_trace: dict[str, Any] = {
                "entrypoint": ENTRYPOINT_NAME,
                "faq_fast_path_bypassed": True,
            }
            try:
                rag_result, invocation_trace = generate_financial_rag_only(
                    rag_system, case["question"]
                )
                token_stream = (rag_result,) if isinstance(rag_result, str) else rag_result
                for token in token_stream:
                    if token:
                        parts.append(token)
            except Exception as exc:  # Preserve the raw failure for scoring policy.
                error = f"{type(exc).__name__}: {exc}"
            evidence_capture = build_prompt_evidence_capture(
                active_trace.get("retrieved_contexts", []),
                active_trace.get("verified_evidence", []),
                active_trace.get("verified_evidence_block"),
                active_trace.get("deterministic_evidence", []),
            )
            results.append({
                "sample_id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "expected_rag": case["expected_rag"],
                "answer": "".join(parts),
                "entrypoint_trace": invocation_trace,
                **evidence_capture,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": error,
            })
            print(f"[{index}/{len(samples)}] captured {case['id']}", flush=True)
    finally:
        restore_evidence_hooks()
    return {
        "capture_metadata": {
            "git_commit": _git_commit(code_root),
            "evaluation_entrypoint": ENTRYPOINT_NAME,
            "faq_fast_path_bypassed": True,
            "evidence_capture_schema": EVIDENCE_CAPTURE_SCHEMA_VERSION,
            "retrieval_k": int(config.RETRIEVAL_K),
            "candidate_m": int(config.CANDIDATE_M),
            "milvus_database": config.MILVUS_DATABASE_NAME,
            "milvus_collection": config.MILVUS_COLLECTION_NAME,
            "llm_model": config.LLM_MODEL,
            "bge_m3_model": config.BGE_M3_MODEL_PATH,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }


def _smoke_samples(samples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Take deterministic RAG-answer cases so all three smoke metrics are evaluable."""
    selected: list[dict[str, Any]] = []
    remaining = sorted(
        (sample for sample in samples if sample["case"].get("expected_rag") is True),
        key=lambda sample: (sample["original_category"], sample["sample_id"]),
    )
    seen_categories: set[str] = set()
    for sample in remaining:
        if sample["original_category"] not in seen_categories:
            selected.append(sample)
            seen_categories.add(sample["original_category"])
        if len(selected) == limit:
            return selected
    return selected + [sample for sample in remaining if sample not in selected][:limit - len(selected)]


def _ensure_baseline_worktree(path: Path) -> Path:
    if path.exists():
        actual = _git_commit(path)
        if not actual.startswith(BASELINE_COMMIT):
            raise ValueError(f"baseline worktree has unexpected commit: {actual}")
        return path
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(path), BASELINE_COMMIT],
        cwd=ROOT,
        check=True,
    )
    return path


def _capture_subprocess(version: str, manifest_path: Path, output_path: Path, limit: int | None, code_root: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_capture-worker",
        "--manifest", str(manifest_path),
        "--output", str(output_path),
        "--code-root", str(code_root),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    # ``config`` loads the primary checkout's .env in the parent only.  The
    # detached Baseline worktree receives the required values as process env;
    # no .env is copied and this mapping is never stored in an artifact.
    from base.config import config

    environment = build_capture_environment(os.environ, config)
    environment.update(RUNTIME_ENV)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    # Historical worktrees intentionally contain only source. Reuse the frozen
    # checkout's local model assets and runtime LLM settings without copying
    # either models or .env files into the baseline worktree. These values stay
    # in the child process environment and are never written to evaluation data.
    # The capture worker must use the target RAG's dependency set, not the
    # isolated RAGAS judge dependencies supplied through PYTHONPATH.
    environment.pop("PYTHONPATH", None)
    subprocess.run(command, cwd=code_root, env=environment, check=True)
    captured = _load(output_path)
    captured["version"] = version
    return captured


def _ragas_dependencies_path(path: str | None) -> None:
    if path:
        dependency_path = Path(path).resolve()
        if not dependency_path.is_dir():
            raise ValueError(f"RAGAS dependency path does not exist: {dependency_path}")
        sys.path.insert(0, str(dependency_path))


def _ragas_components():
    try:
        import ragas
        from langchain_core.embeddings import Embeddings
        from langchain_openai import ChatOpenAI
        from ragas.dataset_schema import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import LLMContextPrecisionWithoutReference, answer_relevancy, faithfulness
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS 0.2.6 is unavailable. Install it in an isolated path and pass "
            "--ragas-deps <path>; production dependencies are intentionally untouched."
        ) from exc
    return {
        "ragas": ragas,
        "Embeddings": Embeddings,
        "ChatOpenAI": ChatOpenAI,
        "SingleTurnSample": SingleTurnSample,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "LLMContextPrecisionWithoutReference": LLMContextPrecisionWithoutReference,
        "answer_relevancy": answer_relevancy,
        "faithfulness": faithfulness,
    }


def _score_capture(
    captured: dict[str, Any],
    manifest: dict[str, Any],
    ragas_deps: str | None,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    _ragas_dependencies_path(ragas_deps)
    components = _ragas_components()
    from base.config import config
    from milvus_model.hybrid import BGEM3EmbeddingFunction

    Embeddings = components["Embeddings"]

    class BGEM3Embeddings(Embeddings):
        def __init__(self, model_path: str):
            self.model = BGEM3EmbeddingFunction(model_name_or_path=model_path, use_f16=False, device="cpu")

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            dense_vectors = self.model(texts)["dense"]
            return dense_vectors.tolist() if hasattr(dense_vectors, "tolist") else dense_vectors

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

    judge = components["ChatOpenAI"](
        model=config.LLM_MODEL,
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        temperature=0,
        timeout=60,
        max_retries=0,
        extra_body={"enable_thinking": False},
    )
    llm = components["LangchainLLMWrapper"](judge)
    embeddings = components["LangchainEmbeddingsWrapper"](BGEM3Embeddings(config.BGE_M3_MODEL_PATH))
    metrics = {
        "faithfulness": components["faithfulness"].__class__(llm=llm),
        "answer_relevancy": components["answer_relevancy"].__class__(llm=llm, embeddings=embeddings),
        "context_precision": components["LLMContextPrecisionWithoutReference"](llm=llm),
    }
    scores = []
    for item in captured["results"]:
        row = {**item, "ragas": {}, "unevaluable_reason": None}
        # Faithfulness and Context Precision see only the exact Parent and
        # Answer-Layer evidence blocks that were concatenated into the prompt.
        # Never substitute a gold parent, answer, or relevance-filtered context.
        contexts = [
            context for context in item.get("ragas_contexts", [])
            if isinstance(context, str) and context
        ]
        if item.get("error") or not item.get("answer") or not contexts:
            row["unevaluable_reason"] = item.get("error") or "missing answer or final prompt evidence"
            scores.append(row)
            continue
        sample = components["SingleTurnSample"](
            user_input=item["question"], response=item["answer"], retrieved_contexts=contexts
        )
        for name, metric in metrics.items():
            try:
                value = metric.single_turn_score(sample)
                row["ragas"][name] = float(value) if value is not None and not math.isnan(value) else None
            except Exception as exc:
                row["ragas"][name] = None
                row.setdefault("metric_errors", {})[name] = f"{type(exc).__name__}: {exc}"
        scores.append(row)
    metric_summary = {}
    for name in metrics:
        values = [row["ragas"].get(name) for row in scores if row.get("ragas", {}).get(name) is not None]
        metric_summary[name] = {
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "valid_count": len(values),
            "failed_or_unevaluable_count": len(scores) - len(values),
        }
    return {
        "evaluation_metadata": {
            "git_commit": captured["capture_metadata"]["git_commit"],
            "ragas_version": components["ragas"].__version__,
            "judge_model": config.LLM_MODEL,
            "judge_temperature": 0,
            "embedding_model": config.BGE_M3_MODEL_PATH,
            "dataset_version": manifest["source_dataset_schema_version"],
            "sample_manifest": manifest_path.name,
            "sample_manifest_sha256": _sha256(manifest_path),
            "context_precision_metric": "LLMContextPrecisionWithoutReference",
            "context_precision_note": (
                "Measures relevance of returned prompt evidence without a reference; it is not "
                "Document Recall@3 or multi-target coverage. Multi-target Final may legitimately "
                "supply more prompt evidence than Baseline. No gold answer or gold context was used."
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **captured["capture_metadata"],
        },
        "metrics": metric_summary,
        "results": scores,
    }


def _compare(baseline: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    for name in ("faithfulness", "answer_relevancy", "context_precision"):
        baseline_mean = baseline["metrics"][name]["mean"]
        final_mean = final["metrics"][name]["mean"]
        metrics[name] = {
            "baseline": baseline["metrics"][name],
            "final": final["metrics"][name],
            "mean_delta": None if baseline_mean is None or final_mean is None else final_mean - baseline_mean,
        }
    return {
        "baseline_metadata": baseline["evaluation_metadata"],
        "final_metadata": final["evaluation_metadata"],
        "metrics": metrics,
    }


def _audit_evaluation_manifest(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    if manifest.get("schema_version") == "ragas_20_pilot_v1":
        return audit_pilot_manifest(manifest, MANIFEST_PATH)
    return audit_manifest(manifest)


def _run_version(args: argparse.Namespace, version: str, limit: int | None, output_path: Path) -> None:
    manifest = _load(Path(args.manifest))
    audit = _audit_evaluation_manifest(manifest, Path(args.manifest))
    if not audit["passed"]:
        raise SystemExit("manifest audit failed: " + "; ".join(audit["errors"]))
    if version == "baseline":
        code_root = _ensure_baseline_worktree(Path(args.baseline_worktree))
    else:
        code_root = ROOT
    # Keep the raw system answer and exact final contexts even if a later RAGAS
    # metric fails. This is an evaluation artifact, not a gold-context fallback.
    capture_path = output_path.with_name(output_path.stem + "_capture.json")
    _capture_subprocess(version, Path(args.manifest), capture_path, limit, code_root)
    # Keep RAGAS in a clean interpreter. The capture worker imports the RAG's
    # tokenizer and embedding stack; scoring in that same process can inherit
    # fork/parallelism state and fail before emitting a result artifact.
    score_command = [
        sys.executable,
        "-m",
        "evaluations.run_ragas_final_evaluation",
        "score-capture",
        "--manifest", str(args.manifest),
        "--capture", str(capture_path),
        "--output", str(output_path),
    ]
    if args.ragas_deps:
        score_command.extend(["--ragas-deps", args.ragas_deps])
    score_environment = os.environ.copy()
    score_environment["TOKENIZERS_PARALLELISM"] = "false"
    subprocess.run(score_command, cwd=ROOT, env=score_environment, check=True)


def _worker(args: argparse.Namespace) -> None:
    manifest = _load(Path(args.manifest))
    captured = _capture_cases(Path(args.code_root), manifest, args.limit)
    _dump(Path(args.output), captured)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen 300Q RAGAS final evaluation")
    parser.add_argument(
        "command",
        choices=(
            "audit", "build-pilot", "fairness-audit", "smoke", "pilot", "run",
            "compare", "score-capture", "_capture-worker",
        ),
    )
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--version", choices=("baseline", "final"))
    parser.add_argument("--output")
    parser.add_argument("--capture")
    parser.add_argument("--baseline-worktree", default=str(BASELINE_WORKTREE))
    parser.add_argument("--ragas-deps", help="isolated directory containing RAGAS 0.2.6 dependencies")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--code-root")
    args = parser.parse_args()

    if args.command == "_capture-worker":
        if not args.output or not args.code_root:
            raise SystemExit("worker requires --output and --code-root")
        _worker(args)
        return
    if args.command == "score-capture":
        if not args.capture or not args.output:
            raise SystemExit("score-capture requires --capture and --output")
        manifest_path = Path(args.manifest)
        scored = _score_capture(
            _load(Path(args.capture)), _load(manifest_path), args.ragas_deps, manifest_path
        )
        _dump(Path(args.output), scored)
        print(json.dumps(scored["metrics"], ensure_ascii=False, indent=2))
        return
    if args.command == "build-pilot":
        pilot = build_pilot_manifest(Path(args.manifest), PILOT_MANIFEST_PATH)
        print(json.dumps(audit_pilot_manifest(pilot, Path(args.manifest)), ensure_ascii=False, indent=2))
        return
    if args.command == "fairness-audit":
        manifest = _load(Path(args.manifest))
        print(json.dumps(fairness_dry_run(manifest), ensure_ascii=False, indent=2))
        return
    if args.command == "audit":
        manifest = build_manifest(DATASET_PATH, Path(args.manifest))
        print(json.dumps(audit_manifest(manifest), ensure_ascii=False, indent=2))
        return
    if args.command == "compare":
        summary = _compare(_load(BASELINE_RESULTS_PATH), _load(FINAL_RESULTS_PATH))
        _dump(SUMMARY_PATH, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command in {"smoke", "pilot", "run"} and not args.confirm_run:
        raise SystemExit("refusing to execute RAGAS evaluation without --confirm-run")
    if not args.version:
        raise SystemExit("smoke/pilot/run requires --version baseline or final")
    if args.command == "pilot" and args.manifest == str(MANIFEST_PATH):
        args.manifest = str(PILOT_MANIFEST_PATH)
    limit = 5 if args.command == "smoke" else args.limit
    default = (
        EVALUATIONS / f"ragas_{args.version}_{'5_smoke' if limit else ('20_pilot' if args.command == 'pilot' else '100')}_v1_results.json"
    )
    output_path = Path(args.output).expanduser() if args.output else default
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()
    _run_version(args, args.version, limit, output_path)


if __name__ == "__main__":
    main()
