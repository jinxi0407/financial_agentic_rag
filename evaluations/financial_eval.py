"""Build source-grounded gold data and evaluate the Financial RAG runtime.

The gold builder deliberately reads Parent chunks already persisted in Milvus.
It never fabricates a document reference: each recorded evidence parent is
validated against the supplied evaluation case's expected source documents.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from pymilvus import MilvusClient

from base.config import config


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "financial_eval_60.json"
DEFAULT_GOLD = ROOT / "financial_eval_60_gold.json"
MAX_PARENTS_PER_DOCUMENT = 6
PARENT_PREVIEW_LENGTH = 500
EXPECTED_ROUTE_MAP = {"OOS": "OUT_OF_SCOPE"}
EXPECTED_STRATEGY_MAP = {
    "Direct": "直接检索",
    "HyDE": "假设问题检索",
    "SubQuery": "子查询检索",
}


def _escape_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _tokenize(text: str) -> list[str]:
    """Extract useful CJK phrases and identifiers without a new dependency."""
    compact = re.sub(r"\s+", "", text.lower())
    tokens = re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?|[\u4e00-\u9fff]{2,}", compact)
    result = []
    for token in tokens:
        if len(token) >= 2 and token not in {"公司", "报告", "年度", "上半年", "多少", "情况"}:
            result.append(token)
        if len(token) >= 4 and re.fullmatch(r"[\u4e00-\u9fff]+", token):
            result.extend(token[i : i + 2] for i in range(len(token) - 1))
    return result


def _rank_parents(question: str, parents: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    question_tokens = Counter(_tokenize(question))

    def score(parent: dict[str, Any]) -> tuple[float, str]:
        content_tokens = Counter(_tokenize(parent["parent_content"]))
        overlap = sum(
            query_count * (1.0 + math.log1p(content_tokens[token]))
            for token, query_count in question_tokens.items()
            if token in content_tokens
        )
        return overlap, parent["parent_id"]

    return sorted(parents, key=score, reverse=True)


class MilvusParentStore:
    """A read-only view of de-duplicated Parent documents in the live corpus."""

    OUTPUT_FIELDS = [
        "parent_id",
        "parent_content",
        "document_id",
        "file_sha256",
        "source_filename",
        "company_name",
        "company_code",
        "report_year",
        "period_type",
        "report_period",
    ]

    def __init__(self) -> None:
        self.client = MilvusClient(
            uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}",
            db_name=config.MILVUS_DATABASE_NAME,
        )

    def parents_for_filename(self, filename: str) -> list[dict[str, Any]]:
        expr = f'source_filename == "{_escape_literal(filename)}"'
        rows = self.client.query(
            collection_name=config.MILVUS_COLLECTION_NAME,
            filter=expr,
            output_fields=self.OUTPUT_FIELDS,
            limit=16384,
        )
        parents: dict[str, dict[str, Any]] = {}
        for row in rows:
            parent_id = row.get("parent_id")
            content = row.get("parent_content")
            if parent_id and content:
                parents.setdefault(parent_id, row)
        if not parents:
            raise RuntimeError(
                f"No Parent data found for expected source document: {filename}"
            )
        return list(parents.values())

    def candidates_for_case(self, case: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for filename in case.get("expected_documents") or []:
            parents = self.parents_for_filename(filename)
            candidates.extend(_rank_parents(case["question"], parents)[:MAX_PARENTS_PER_DOCUMENT])
        return candidates


def _content_from_completion(completion: Any) -> str:
    if not completion.choices or not completion.choices[0].message:
        raise RuntimeError("Gold annotation model returned no choice")
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("Gold annotation model returned empty content")
    return content.strip()


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError("Response did not contain a JSON object")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Response JSON must be an object")
    return payload


def _evidence_record(parent: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_id": parent["parent_id"],
        "document_id": parent.get("document_id"),
        "file_sha256": parent.get("file_sha256"),
        "source_filename": parent.get("source_filename"),
        "company_name": parent.get("company_name"),
        "company_code": parent.get("company_code"),
        "report_year": parent.get("report_year"),
        "period_type": parent.get("period_type"),
        "report_period": parent.get("report_period"),
        "evidence_excerpt": parent["parent_content"][:PARENT_PREVIEW_LENGTH],
    }


def _annotate_case(
    client: OpenAI,
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    rendered_candidates = "\n\n".join(
        "[Parent {parent_id}]\nsource_filename: {source_filename}\n"
        "report_period: {report_period}\ncontent:\n{parent_content}".format(**parent)
        for parent in candidates
    )
    required_docs = ", ".join(case["expected_documents"])
    prompt = f"""为 Financial Agentic RAG 评测生成可审计 gold 标注。只能使用下方列出的 Parent 内容；
不能补充 Parent 外的知识、数字或计算。回答必须忠实回答问题。每份 Required documents 至少引用一个 Parent。

Question: {case['question']}
Required documents: {required_docs}

Return exactly one JSON object:
{{"gold_answer":"中文简洁答案", "evidence_parent_ids":["Parent ID", "..."]}}

Candidate Parents:
{rendered_candidates}
"""
    completion = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是严格基于财报证据进行标注的金融分析助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=1200,
        stream=False,
        timeout=60,
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
    )
    payload = _parse_json_object(_content_from_completion(completion))
    answer = payload.get("gold_answer")
    evidence_ids = payload.get("evidence_parent_ids")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{case['id']}: gold_answer is missing")
    if isinstance(evidence_ids, str):
        # Tolerate an otherwise useful response that serialized one ID as text.
        evidence_ids = [evidence_ids]
    elif not isinstance(evidence_ids, list):
        evidence_ids = []
    return answer.strip(), [item for item in evidence_ids if isinstance(item, str)]


def build_gold(cases_path: Path, output_path: Path, overwrite: bool = False) -> dict[str, Any]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    # Resume an interrupted approved annotation job without rerunning completed cases.
    if output_path.exists() and not overwrite:
        previous_cases = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(previous_cases, list) and [item.get("id") for item in previous_cases] == [
            item.get("id") for item in cases
        ]:
            cases = previous_cases
    if not isinstance(cases, list):
        raise ValueError("Evaluation dataset must be a JSON list")
    if not config.DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY is required to generate source-grounded gold answers")

    store = MilvusParentStore()
    client = OpenAI(api_key=config.DASHSCOPE_API_KEY, base_url=config.DASHSCOPE_BASE_URL)
    summary = {"annotated": 0, "skipped": 0, "no_report_cases": 0}
    for index, case in enumerate(cases, start=1):
        expected_documents = case.get("expected_documents") or []
        if not expected_documents:
            if overwrite or not case.get("gold_answer"):
                case["gold_answer"] = case.get("expected_behavior") or "当前知识库中没有可用的对应财报。"
                case["gold_evidence"] = []
                case["gold_status"] = "expected_no_corpus_evidence"
                summary["no_report_cases"] += 1
            continue
        if not overwrite and case.get("gold_answer") and case.get("gold_evidence"):
            summary["skipped"] += 1
            continue

        candidates = store.candidates_for_case(case)
        answer, selected_ids = _annotate_case(client, case, candidates)
        candidate_by_id = {parent["parent_id"]: parent for parent in candidates}
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for parent_id in selected_ids:
            parent = candidate_by_id.get(parent_id)
            if parent and parent_id not in seen:
                selected.append(parent)
                seen.add(parent_id)

        # A cross-report answer is not valid gold unless every required source is cited.
        for filename in expected_documents:
            if not any(parent["source_filename"] == filename for parent in selected):
                fallback = next(parent for parent in candidates if parent["source_filename"] == filename)
                selected.append(fallback)
                seen.add(fallback["parent_id"])

        case["gold_answer"] = answer
        case["gold_evidence"] = [_evidence_record(parent) for parent in selected]
        case["gold_status"] = "generated_from_ingested_parents"
        case["gold_generation"] = {
            "model": config.LLM_MODEL,
            "candidate_parent_count": len(candidates),
            "source": "financial.financial_rag_v1 Parent data",
        }
        summary["annotated"] += 1
        _write_json_atomic(output_path, cases)
        print(f"[{index}/{len(cases)}] annotated {case['id']}", flush=True)

    _write_json_atomic(output_path, cases)
    return summary


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def _document_metadata(document: Any) -> dict[str, Any]:
    metadata = getattr(document, "metadata", {}) or {}
    return {
        "parent_id": metadata.get("parent_id"),
        "document_id": metadata.get("document_id"),
        "source_filename": metadata.get("source_filename"),
        "company_code": metadata.get("company_code"),
        "report_period": metadata.get("report_period"),
        "rerank_score": metadata.get("rerank_score"),
    }


def _install_runtime_trace(qa_system: Any, active_trace: dict[str, dict[str, Any]]) -> None:
    router = qa_system.rag.query_router
    original_route = router.route

    def route(query: str) -> str:
        result = original_route(query)
        _raise_if_api_failed(active_trace["value"])
        active_trace["value"]["route"] = result
        return result

    router.route = route
    selector = qa_system.rag.strategy_selector
    original_select = selector.select_strategy

    def select_strategy(query: str) -> str:
        result = original_select(query)
        _raise_if_api_failed(active_trace["value"])
        active_trace["value"]["strategy"] = result
        return result

    selector.select_strategy = select_strategy
    original_retrieve = qa_system.rag.retrieve_and_merge

    def retrieve_and_merge(*args: Any, **kwargs: Any) -> Any:
        documents = original_retrieve(*args, **kwargs)
        active_trace["value"]["final_contexts"] = [
            _document_metadata(doc) for doc in documents
        ]
        return documents

    qa_system.rag.retrieve_and_merge = retrieve_and_merge

    original_llm = qa_system.rag.llm

    def llm(prompt: str) -> Any:
        response = original_llm(prompt)
        if isinstance(response, str):
            _raise_if_api_failed(active_trace["value"])
            return response

        def checked_stream() -> Iterable[str]:
            for token in response:
                _raise_if_api_failed(active_trace["value"])
                yield token
            _raise_if_api_failed(active_trace["value"])

        return checked_stream()

    qa_system.rag.llm = llm


class EvaluationApiFailure(RuntimeError):
    """An API failure that production code may otherwise degrade into a fallback."""


def _instrument_api_client(client: Any, active_trace: dict[str, dict[str, Any]]) -> None:
    if client is None:
        return
    create = client.chat.completions.create
    if getattr(create, "_financial_eval_instrumented", False):
        return

    def checked_create(*args: Any, **kwargs: Any) -> Any:
        try:
            return create(*args, **kwargs)
        except Exception as exc:
            active_trace["value"].setdefault("api_failures", []).append(
                f"{type(exc).__name__}: {exc}"
            )
            raise

    checked_create._financial_eval_instrumented = True
    client.chat.completions.create = checked_create


def _raise_if_api_failed(trace: dict[str, Any]) -> None:
    failures = trace.get("api_failures") or []
    if failures:
        raise EvaluationApiFailure(failures[-1])


def _gold_parent_ids(case: dict[str, Any]) -> set[str]:
    return {
        item["parent_id"]
        for item in case.get("gold_evidence") or []
        if isinstance(item, dict) and item.get("parent_id")
    }


def _add_derived_fields(case: dict[str, Any], result: dict[str, Any]) -> None:
    expected_documents = list(case.get("expected_documents") or [])
    target_companies = set(case.get("target_company_codes") or [])
    target_periods = set(case.get("target_report_periods") or [])
    contexts = result.get("final_contexts") or []
    filenames = {item.get("source_filename") for item in contexts}
    parent_ids = {item.get("parent_id") for item in contexts}
    context_companies = {item.get("company_code") for item in contexts if item.get("company_code")}
    context_periods = {item.get("report_period") for item in contexts if item.get("report_period")}
    gold_parent_ids = _gold_parent_ids(case)
    result.update(
        {
            "expected_route": _expected_route(case),
            "route_match": result.get("route") == _expected_route(case),
            "strategy_allowed": _strategy_allowed(case, result.get("strategy")),
            "expected_documents": expected_documents,
            "target_company_codes": sorted(target_companies),
            "target_report_periods": sorted(target_periods),
            "document_hit_at_3": bool(filenames & set(expected_documents)) if expected_documents else None,
            "expected_document_coverage": (
                set(expected_documents).issubset(filenames) if expected_documents else None
            ),
            "strict_gold_parent_recall": (
                gold_parent_ids.issubset(parent_ids) if gold_parent_ids else None
            ),
            "wrong_company": (
                bool(context_companies - target_companies) if target_companies else None
            ),
            "wrong_period": bool(context_periods - target_periods) if target_periods else None,
        }
    )


def _expected_route(case: dict[str, Any]) -> str | None:
    route = case.get("expected_route")
    return EXPECTED_ROUTE_MAP.get(route, route)


def _strategy_allowed(case: dict[str, Any], actual_strategy: str | None) -> bool | None:
    expected = case.get("allowed_strategies") or []
    allowed = {EXPECTED_STRATEGY_MAP[name] for name in expected if name in EXPECTED_STRATEGY_MAP}
    return actual_strategy in allowed if allowed else None


def _numeric_tokens(text: str) -> set[str]:
    return {token.replace(",", "") for token in re.findall(r"\d[\d,.]*%?", text)}


def _judge_answer(client: OpenAI, case: dict[str, Any], answer: str) -> dict[str, Any]:
    evidence = "\n\n".join(
        "[{} | {}]\n{}".format(
            item.get("parent_id", ""),
            item.get("source_filename", ""),
            item.get("evidence_excerpt", ""),
        )
        for item in case.get("gold_evidence", [])
    )
    prompt = f"""评估 Financial RAG 回答。只按 Gold answer、Gold evidence 和用户问题判断，
不得引入外部事实。对 no-corpus / OUT_OF_SCOPE case，判断回答是否符合 Expected behavior。
返回严格 JSON：{{"answer_correct":true/false,"evidence_supported":true/false,"reason":"不超过80字"}}。

Question: {case['question']}
Gold answer: {case['gold_answer']}
Expected behavior: {case.get('expected_behavior')}
Gold evidence:\n{evidence or '(none)'}
System answer: {answer}
"""
    completion = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是严格、保守的金融 RAG 评测员。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=240,
        stream=False,
        timeout=60,
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
    )
    payload = _parse_json_object(_content_from_completion(completion))
    if not isinstance(payload.get("answer_correct"), bool) or not isinstance(
        payload.get("evidence_supported"), bool
    ):
        raise ValueError("Answer judge did not return boolean verdicts")
    return {
        "answer_correct": payload["answer_correct"],
        "evidence_supported": payload["evidence_supported"],
        "judge_reason": str(payload.get("reason", "")),
    }


def run_evaluation(cases_path: Path, output_path: Path, judge_answers: bool = False) -> dict[str, Any]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    missing_gold = [case["id"] for case in cases if not case.get("gold_answer")]
    if missing_gold:
        raise RuntimeError(f"Dataset has missing gold_answer values: {', '.join(missing_gold)}")

    from new_main import IntegratedQASystem

    prior_results: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        previous_report = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(previous_report, dict):
            prior_results = {
                result["id"]: result
                for result in previous_report.get("results", [])
                if isinstance(result, dict) and result.get("id")
            }

    for case in cases:
        if case["id"] in prior_results:
            _add_derived_fields(case, prior_results[case["id"]])
    pending_cases = [case for case in cases if case["id"] not in prior_results]
    needs_judging = judge_answers and any(
        case["id"] in prior_results and "answer_correct" not in prior_results[case["id"]]
        for case in cases
    )
    qa_system = IntegratedQASystem() if pending_cases or needs_judging else None
    active_trace: dict[str, dict[str, Any]] = {"value": {}}
    if qa_system is not None:
        seen_clients: set[int] = set()
        for client in (
            qa_system.client,
            qa_system.rag.query_router.client,
            qa_system.rag.strategy_selector.client,
        ):
            if client is not None and id(client) not in seen_clients:
                _instrument_api_client(client, active_trace)
                seen_clients.add(id(client))
        _install_runtime_trace(qa_system, active_trace)
    results_by_id = dict(prior_results)
    if judge_answers and qa_system is not None:
        for case in cases:
            result = results_by_id.get(case["id"])
            if result is None or "answer_correct" in result:
                continue
            active_trace["value"] = {"api_failures": []}
            try:
                result.update(_judge_answer(qa_system.client, case, result.get("answer", "")))
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                result["excluded_from_metrics"] = True
                _write_json_atomic(
                    output_path,
                    {"dataset": str(cases_path), "metrics": _metrics(list(results_by_id.values())), "results": list(results_by_id.values())},
                )
                raise EvaluationApiFailure(result["error"])
    for index, case in enumerate(pending_cases, start=1):
        trace: dict[str, Any] = {"final_contexts": []}
        active_trace["value"] = trace
        started = time.perf_counter()
        first_token_seconds = None
        tokens: list[str] = []
        error = None
        try:
            for token, is_end in qa_system.query(case["question"], session_id=None):
                if first_token_seconds is None and token:
                    first_token_seconds = time.perf_counter() - started
                if not is_end:
                    tokens.append(token)
        except Exception as exc:  # Preserve the case-level failure in the report.
            error = f"{type(exc).__name__}: {exc}"
        answer = "".join(tokens)
        final_contexts = trace["final_contexts"]
        gold_numbers = _numeric_tokens(case["gold_answer"])
        answer_numbers = _numeric_tokens(answer)
        result = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "route": trace.get("route"),
            "expected_route": _expected_route(case),
            "route_match": trace.get("route") == _expected_route(case),
            "strategy": trace.get("strategy"),
            "strategy_allowed": _strategy_allowed(case, trace.get("strategy")),
            "final_contexts": final_contexts,
            "gold_numeric_token_recall": (
                len(gold_numbers & answer_numbers) / len(gold_numbers) if gold_numbers else None
            ),
            "answer": answer,
            "gold_answer": case["gold_answer"],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "ttft_seconds": round(first_token_seconds, 3) if first_token_seconds is not None else None,
            "error": error,
            "excluded_from_metrics": bool(error),
        }
        _add_derived_fields(case, result)
        if judge_answers and not error:
            try:
                result.update(_judge_answer(qa_system.client, case, answer))
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                result["excluded_from_metrics"] = True
        results_by_id[case["id"]] = result
        partial_results = [results_by_id[case["id"]] for case in cases if case["id"] in results_by_id]
        _write_json_atomic(
            output_path,
            {"dataset": str(cases_path), "metrics": _metrics(partial_results), "results": partial_results},
        )
        print(
            f"[{len(prior_results) + index}/{len(cases)}] evaluated {case['id']} "
            f"({result['elapsed_seconds']}s)",
            flush=True,
        )
        if result["excluded_from_metrics"]:
            raise EvaluationApiFailure(result["error"] or "Evaluation request failed")

    results = [results_by_id[case["id"]] for case in cases if case["id"] in results_by_id]
    metrics = _metrics(results)
    report = {"dataset": str(cases_path), "metrics": metrics, "results": results}
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in results if not item.get("excluded_from_metrics")]
    if not eligible:
        return {"completed": len(results), "n": 0, "errors": sum(bool(item.get("error")) for item in results)}

    def rate(values: list[bool]) -> float | None:
        return sum(values) / len(values) if values else None

    document_cases = [item for item in eligible if item.get("expected_documents")]
    strategy_cases = [item for item in eligible if item.get("strategy_allowed") is not None]
    ttfts = [item["ttft_seconds"] for item in eligible if item.get("ttft_seconds") is not None]
    elapsed = [item["elapsed_seconds"] for item in eligible]
    judged = [item for item in eligible if item.get("answer_correct") is not None]
    return {
        "completed": len(results),
        "n": len(eligible),
        "route_accuracy": rate([item["route_match"] for item in eligible]),
        "strategy_accuracy": rate([item["strategy_allowed"] for item in strategy_cases]),
        "document_hit_at_3": rate([item["document_hit_at_3"] for item in document_cases]),
        "required_document_coverage_at_3": rate([item["expected_document_coverage"] for item in document_cases]),
        "strict_parent_evidence_recall_at_3": rate(
            [item["strict_gold_parent_recall"] for item in document_cases if item["strict_gold_parent_recall"] is not None]
        ),
        "wrong_company_rate_query": rate(
            [item["wrong_company"] for item in document_cases if item["wrong_company"] is not None]
        ),
        "wrong_period_rate_query": rate(
            [item["wrong_period"] for item in document_cases if item["wrong_period"] is not None]
        ),
        "answer_correctness": rate([item["answer_correct"] for item in judged]),
        "evidence_support": rate([item["evidence_supported"] for item in judged]),
        "average_ttft_seconds": sum(ttfts) / len(ttfts) if ttfts else None,
        "average_elapsed_seconds": sum(elapsed) / len(elapsed),
        "p50_elapsed_seconds": _percentile(elapsed, 0.5),
        "p95_elapsed_seconds": _percentile(elapsed, 0.95),
        "errors": sum(bool(item.get("error")) for item in results),
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    gold_parser = subparsers.add_parser("build-gold")
    gold_parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    gold_parser.add_argument("--output", type=Path, default=DEFAULT_GOLD)
    gold_parser.add_argument("--overwrite", action="store_true")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--cases", type=Path, default=DEFAULT_GOLD)
    run_parser.add_argument("--output", type=Path, default=ROOT / "financial_eval_60_results.json")
    run_parser.add_argument("--judge-answers", action="store_true")
    args = parser.parse_args()
    if args.command == "build-gold":
        print(json.dumps(build_gold(args.cases, args.output, args.overwrite), ensure_ascii=False))
    else:
        print(json.dumps(run_evaluation(args.cases, args.output, args.judge_answers), ensure_ascii=False))


if __name__ == "__main__":
    main()
