"""Offline-only v1.1 scoring correction for the saved Agent 150-turn run.

This module never imports the Agent, Redis, MCP, or Qwen clients.  It reads the
immutable v1 result artifact, applies documented label overlays in memory, and
writes separate scoring and changelog artifacts.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "evaluations" / "financial_agent_holdout_150_v1_results.json"
DEFAULT_SCORING = ROOT / "evaluations" / "financial_agent_holdout_150_v1_1_scoring.json"
DEFAULT_CHANGELOG = ROOT / "evaluations" / "financial_agent_holdout_150_v1_1_changelog.json"

# Both questions explicitly contain 2024H1.  This is a gold label correction,
# not an Agent-result correction.
LABEL_CORRECTIONS = {
    "agent_holdout_077": {"expected_report_periods": ["2024H1"]},
    "agent_holdout_090": {"expected_report_periods": ["2024H1"]},
}
EXTERNAL_TOOL_NAMES = {"get_market_snapshot", "search_financial_news"}
SCORING_PROTOCOL = {
    "intent_accuracy": "全部 completed turns；actual_intent 与 expected_intent 完全相等。",
    "tool_selection_accuracy": "全部 completed turns；planned_tools 与 expected_tools 使用集合完全相等比较。",
    "required_tool_coverage": "expected_tools 非空的 completed turns；每个 expected tool 均实际 executed。",
    "tool_execution_success_rate": "所有保存的 raw tool_results；保留 provider timeout 对该指标的影响。",
    "overall_agent_state_period_accuracy": "所有 expected_report_periods 非空的 turns；保留 v1 严格 state 口径，仅修正已确认 gold label。",
    "explicit_period_resolution_accuracy": "问题文本显式写出期间、且有 expected_report_periods 的 turns。",
    "period_memory_carryover_accuracy": "仅需要将省略期间带入后续 financial_report_query 的 recovery turns；News/Market follow-up 不进入分母。",
    "full_context_memory_recovery_accuracy": "所有 memory recovery turns；公司必须精确恢复，只有 period-carryover financial turn 还要求期间精确恢复。",
    "thread_isolation_accuracy": "仅 thread_isolation_recover turns；actual company 集合必须精确等于当前线程 expected company，不要求 News/Market 保存历史 report period。",
    "guardrail_correctness": "仅 existing guardrail_status/flags 能映射到 expected behavior 的 turns；无专用 flag 的 passed 结果为 unevaluable，不猜测 final answer。",
    "safe_degradation_rate": "allow_external_tool_failure=true 且保存的 Market/News tool_result 实际失败的 turns。",
    "composite_completion_rate": "20 个 composite turns；required tools 均 planned+executed、answer 非空、无 unsupported fabrication flag；provider failure 时须 recorded safe degradation。",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _case_with_overlay(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    case = copy.deepcopy(row["case"])
    correction = LABEL_CORRECTIONS.get(case["id"])
    if correction:
        case.update(correction)
    return case, correction


def _external_failures(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [tool for tool in row.get("tool_results", []) if tool.get("tool_name") in EXTERNAL_TOOL_NAMES and not tool.get("success")]


def _period_is_explicit(case: dict[str, Any]) -> bool:
    question = case["question"].lower()
    return bool(re.search(r"20(?:24|25|26)(?:h1|fy|年(?:上半年|半年报|半年度|报|年度|全年)|半年报|半年度)", question))


def _memory_period_required(case: dict[str, Any]) -> bool:
    """Only financial follow-up turns require retained report period state.

    News/market follow-ups may inherit company context, but their current tool
    request does not require an old report period.  This prevents treating a
    missing period on a news turn as cross-thread contamination.
    """
    return (
        case.get("expected_memory_behavior") in {"recover_company_and_period", "recover_companies_and_period", "thread_isolation_recover"}
        and case.get("expected_intent") == "financial_report_query"
        and bool(case.get("expected_report_periods"))
        and not _period_is_explicit(case)
    )


def _guardrail_verdict(case: dict[str, Any], row: dict[str, Any]) -> tuple[bool | None, str]:
    """Use only existing status/flags, never answer-text keyword heuristics."""
    behavior = case.get("expected_guardrail_behavior", "none")
    status = str(row.get("guardrail_status") or "")
    if behavior == "none":
        return None, "no_guardrail_label"
    if behavior == "no_unverified_calculation":
        return (True, "unverified_calculation_flag") if "unverified_calculation" in status else (None, "flag_absent")
    if behavior in {"no_fabricated_news", "news_provenance_required"}:
        if "news_provenance_missing" in status:
            return True, "news_provenance_missing_flag"
        if "news_unavailable" in status:
            return True, "news_unavailable_safe_degradation_flag"
        return None, "news_guardrail_flag_absent"
    if behavior == "no_fabricated_market":
        if "market_unavailable" in status:
            return True, "market_unavailable_safe_degradation_flag"
        return None, "market_guardrail_not_triggered"
    if behavior == "no_unsupported_financial_number":
        if "unsupported_numeric_claim" in status:
            return True, "unsupported_numeric_claim_flag"
        return None, "financial_claim_guardrail_flag_absent"
    if behavior in {"no_deterministic_investment_advice", "safe_response_required"}:
        return None, "no_dedicated_deterministic_guardrail_flag"
    return None, "unknown_guardrail_label"


def _safe_degradation(case: dict[str, Any], row: dict[str, Any]) -> tuple[bool | None, str]:
    failures = _external_failures(row)
    if not case.get("allow_external_tool_failure") or not failures:
        return None, "not_in_denominator"
    status = str(row.get("guardrail_status") or "")
    return ("unavailable" in status, "external_failure_with_unavailable_flag" if "unavailable" in status else "external_failure_without_safe_flag")


def _corrected_row(raw: dict[str, Any]) -> dict[str, Any]:
    case, correction = _case_with_overlay(raw)
    expected_tools = set(case.get("expected_tools", []))
    optional_tools = set(case.get("optional_tools", []))
    planned = set(raw.get("planned_tools", []))
    executed = set(raw.get("executed_tools", []))
    actual_codes = set(raw.get("resolved_company_codes", []))
    actual_periods = set(raw.get("resolved_periods", []))
    expected_codes = set(case.get("expected_company_codes", []))
    expected_periods = set(case.get("expected_report_periods", []))
    external_failures = _external_failures(raw)
    safe, safe_reason = _safe_degradation(case, raw)
    guardrail, guardrail_reason = _guardrail_verdict(case, raw)
    exact_company = actual_codes == expected_codes if expected_codes else None
    exact_period = actual_periods == expected_periods if expected_periods else None
    isolation = case.get("expected_memory_behavior") == "thread_isolation_recover"
    # Thread isolation is company/context isolation.  A report period is only
    # a separate requirement for an explicit financial carry-over scenario.
    isolation_correct = exact_company if isolation else None
    full_memory = None
    if case.get("expected_memory_behavior") in {"recover_company_and_period", "recover_companies_and_period", "thread_isolation_recover"}:
        full_memory = (exact_company is not False) and (exact_period is not False if _memory_period_required(case) else True)
    forbidden = set(case.get("forbidden_tools", []))
    unnecessary = (executed - expected_tools - optional_tools) | (executed & forbidden)
    return {
        "id": case["id"],
        "case": case,
        "label_correction": correction,
        "actual_intent": raw.get("actual_intent"),
        "planned_tools": sorted(planned),
        "executed_tools": sorted(executed),
        "tool_results": raw.get("tool_results", []),
        "resolved_company_codes": sorted(actual_codes),
        "resolved_periods": sorted(actual_periods),
        "final_answer_nonempty": bool(raw.get("final_answer")),
        "guardrail_status": raw.get("guardrail_status"),
        "error": raw.get("error"),
        "latency_seconds": raw.get("latency_seconds"),
        "intent_correct": raw.get("actual_intent") == case.get("expected_intent"),
        "tool_selection_correct": planned == expected_tools,
        "planned_tool_coverage": expected_tools <= planned,
        "required_tool_coverage": expected_tools <= executed,
        "unnecessary_tools": sorted(unnecessary),
        "company_context_correct": exact_company,
        "period_context_strict_correct": exact_period,
        "explicit_period_case": bool(expected_periods) and _period_is_explicit(case),
        "period_memory_case": _memory_period_required(case),
        "period_memory_correct": exact_period if _memory_period_required(case) else None,
        "full_context_memory_recovery_correct": full_memory,
        "thread_isolation_correct": isolation_correct,
        "safe_degradation": safe,
        "safe_degradation_reason": safe_reason,
        "external_provider_failures": [{"tool_name": item.get("tool_name"), "error": item.get("error")} for item in external_failures],
        "guardrail_evaluable": guardrail is not None,
        "guardrail_correct": guardrail,
        "guardrail_reason": guardrail_reason,
    }


def _composite_detail(row: dict[str, Any]) -> dict[str, Any]:
    case = row["case"]
    provider_failed = bool(row["external_provider_failures"])
    unsafe_flag = any(flag in str(row.get("guardrail_status") or "") for flag in ("fabrication", "unsafe"))
    passed = (
        row["planned_tool_coverage"]
        and row["required_tool_coverage"]
        and row["final_answer_nonempty"]
        and not row["unnecessary_tools"]
        and not unsafe_flag
        and (row["safe_degradation"] is True if provider_failed else True)
    )
    reasons = []
    if not row["planned_tool_coverage"]:
        reasons.append("required_tools_not_planned")
    if not row["required_tool_coverage"]:
        reasons.append("required_tools_not_executed")
    if not row["final_answer_nonempty"]:
        reasons.append("empty_final_answer")
    if row["unnecessary_tools"]:
        reasons.append("unnecessary_or_forbidden_tool")
    if unsafe_flag:
        reasons.append("unsupported_fabrication_flag")
    if provider_failed and row["safe_degradation"] is not True:
        reasons.append("provider_failure_without_safe_degradation")
    return {
        "id": row["id"],
        "expected_tools": case["expected_tools"],
        "planned_tools": row["planned_tools"],
        "executed_tools": row["executed_tools"],
        "provider_failures": row["external_provider_failures"],
        "final_answer_nonempty": row["final_answer_nonempty"],
        "guardrail_status": row["guardrail_status"],
        "safe_degradation": row["safe_degradation"],
        "pass": passed,
        "failure_reasons": reasons,
    }


def _denominator(values: list[Any]) -> dict[str, int]:
    return {"numerator": sum(value is True for value in values), "denominator": len(values), "unevaluable": sum(value is None for value in values)}


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = rows
    tool_results = [tool for row in completed for tool in row["tool_results"]]
    labeled_period = [row for row in completed if row["period_context_strict_correct"] is not None]
    explicit_period = [row for row in completed if row["explicit_period_case"]]
    memory_period = [row for row in completed if row["period_memory_case"]]
    memory_rows = [row for row in completed if row["full_context_memory_recovery_correct"] is not None]
    isolation_rows = [row for row in completed if row["thread_isolation_correct"] is not None]
    guardrail_rows = [row for row in completed if row["guardrail_evaluable"]]
    safe_rows = [row for row in completed if row["safe_degradation"] is not None]
    composite = [_composite_detail(row) for row in completed if row["case"]["category"] == "composite"]
    expected_tool_rows = [row for row in completed if row["case"].get("expected_tools")]
    return {
        "completed": len(completed),
        "total": len(completed),
        "intent_accuracy": _rate([row["intent_correct"] for row in completed]),
        "tool_selection_accuracy": _rate([row["tool_selection_correct"] for row in completed]),
        "planned_tool_coverage": _rate([row["planned_tool_coverage"] for row in expected_tool_rows]),
        "required_tool_coverage": _rate([row["required_tool_coverage"] for row in expected_tool_rows]),
        "unnecessary_tool_call_rate": sum(len(row["unnecessary_tools"]) for row in completed) / sum(len(row["executed_tools"]) for row in completed),
        "tool_execution_success_rate": _rate([bool(tool.get("success")) for tool in tool_results]),
        "company_context_accuracy": _rate([row["company_context_correct"] for row in completed if row["company_context_correct"] is not None]),
        "overall_agent_state_period_accuracy": _rate([row["period_context_strict_correct"] for row in labeled_period]),
        "explicit_period_resolution_accuracy": _rate([row["period_context_strict_correct"] for row in explicit_period]),
        "period_memory_carryover_accuracy": _rate([row["period_memory_correct"] for row in memory_period]),
        "full_context_memory_recovery_accuracy": _rate([row["full_context_memory_recovery_correct"] for row in memory_rows]),
        "thread_isolation_accuracy": _rate([row["thread_isolation_correct"] for row in isolation_rows]),
        "guardrail_evaluable_count": len(guardrail_rows),
        "guardrail_correct_count": sum(row["guardrail_correct"] is True for row in guardrail_rows),
        "guardrail_unevaluable_count": sum(row["case"].get("expected_guardrail_behavior") != "none" and not row["guardrail_evaluable"] for row in completed),
        "guardrail_correctness": _rate([row["guardrail_correct"] for row in guardrail_rows]),
        "safe_degradation_rate": _rate([row["safe_degradation"] for row in safe_rows]),
        "composite_completion_rate": _rate([row["pass"] for row in composite]),
        "mean_latency_seconds": statistics.mean([float(row["latency_seconds"]) for row in completed]),
        "p50_latency_seconds": _percentile([float(row["latency_seconds"]) for row in completed], 0.5),
        "p95_latency_seconds": _percentile([float(row["latency_seconds"]) for row in completed], 0.95),
        "denominators": {
            "overall_agent_state_period_accuracy": _denominator([row["period_context_strict_correct"] for row in labeled_period]),
            "explicit_period_resolution_accuracy": _denominator([row["period_context_strict_correct"] for row in explicit_period]),
            "period_memory_carryover_accuracy": _denominator([row["period_memory_correct"] for row in memory_period]),
            "full_context_memory_recovery_accuracy": _denominator([row["full_context_memory_recovery_correct"] for row in memory_rows]),
            "thread_isolation_accuracy": _denominator([row["thread_isolation_correct"] for row in isolation_rows]),
            "guardrail_correctness": _denominator([row["guardrail_correct"] for row in guardrail_rows]),
            "safe_degradation_rate": _denominator([row["safe_degradation"] for row in safe_rows]),
            "composite_completion_rate": _denominator([row["pass"] for row in composite]),
        },
    }


def _tool_failure_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [tool for row in rows for tool in row["tool_results"] if not tool.get("success")]
    by_tool = Counter(str(tool.get("tool_name")) for tool in failures)
    external = [tool for tool in failures if tool.get("tool_name") in EXTERNAL_TOOL_NAMES]
    non_external = [tool for tool in failures if tool.get("tool_name") not in EXTERNAL_TOOL_NAMES]
    return {
        "failure_count_by_tool": dict(sorted(by_tool.items())),
        "external_provider_failure_count": len(external),
        "non_external_tool_failure_count": len(non_external),
        "all_news_failures_external_provider": all(tool.get("tool_name") == "search_financial_news" for tool in external) if external else None,
        "external_failure_errors": sorted({str(tool.get("error")) for tool in external}),
    }


def _planner_failure_taxonomy(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        case = row["case"]
        failure = not row["intent_correct"] or not row["tool_selection_correct"] or not row["required_tool_coverage"]
        if not failure:
            continue
        expected = case.get("expected_intent")
        if expected == "market_query":
            bucket = "market_query"
        elif expected == "composite_query":
            bucket = "composite_query"
        elif expected == "calculation_query":
            bucket = "calculator_query"
        else:
            bucket = "financial_or_other_query"
        groups[bucket].append({"id": row["id"], "expected_intent": expected, "actual_intent": row["actual_intent"], "expected_tools": case["expected_tools"], "planned_tools": row["planned_tools"], "executed_tools": row["executed_tools"]})
    return dict(groups)


def _changelog() -> dict[str, Any]:
    return {
        "version": "financial_agent_holdout_150_v1_1_scoring",
        "scope": "offline scoring-only; no questions, production code, or raw v1 artifacts were changed",
        "changes": [
            {"type": "gold_label_correction", "case_id": "agent_holdout_077", "old": {"expected_report_periods": ["2024FY"]}, "new": {"expected_report_periods": ["2024H1"]}, "reason": "question explicitly says 2024H1"},
            {"type": "gold_label_correction", "case_id": "agent_holdout_090", "old": {"expected_report_periods": ["2024FY"]}, "new": {"expected_report_periods": ["2024H1"]}, "reason": "question explicitly says 2024H1"},
            {"type": "scoring_correction", "area": "guardrail", "reason": "only deterministic guardrail_status/flags enter denominator; status=passed without a behavior-specific flag is unevaluable, not false"},
            {"type": "scoring_correction", "area": "thread_isolation", "reason": "isolation checks exact company-context recovery and absence of foreign companies; it does not require historical report_period on News/Market turns"},
            {"type": "scoring_correction", "area": "composite_completion", "reason": "external timeout is not a planner failure when tools were planned and attempted, final answer exists, and guardrail recorded safe degradation"},
        ],
    }


def _preference_diagnostics(raw: dict[str, Any]) -> dict[str, Any]:
    """Preserve the frozen evaluator's saved preference verdicts as diagnostics."""
    completed = [row for row in raw["results"] if row.get("status") == "completed"]
    explicit_write = [row.get("preference", {}).get("correct") for row in completed if row["case"].get("expected_preference_behavior") in {"write_explicit_metrics", "write_explicit_companies"}]
    recovery = [row.get("preference", {}).get("correct") for row in completed if row["case"].get("expected_preference_behavior") == "recover_explicit"]
    implicit_nonwrite = [row.get("preference", {}).get("correct") for row in completed if row["case"].get("expected_preference_behavior") == "must_not_write_implicit"]
    return {
        "explicit_preference_write_accuracy": _rate(explicit_write),
        "preference_recovery_accuracy": _rate(recovery),
        "implicit_preference_nonwrite_accuracy": _rate(implicit_nonwrite),
        "denominators": {
            "explicit_preference_write_accuracy": _denominator(explicit_write),
            "preference_recovery_accuracy": _denominator(recovery),
            "implicit_preference_nonwrite_accuracy": _denominator(implicit_nonwrite),
        },
    }


def _tool_execution_detail(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_tool: dict[str, dict[str, int]] = {}
    for tool in (tool for row in rows for tool in row["tool_results"]):
        name = str(tool.get("tool_name"))
        stats = by_tool.setdefault(name, {"attempts": 0, "success": 0, "failure": 0})
        stats["attempts"] += 1
        stats["success" if tool.get("success") else "failure"] += 1
    return {"by_tool": dict(sorted(by_tool.items())), **_tool_failure_diagnostics(rows)}


def rescore(raw: dict[str, Any], source_path: Path = DEFAULT_RESULTS) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [_corrected_row(row) for row in raw["results"] if row.get("status") == "completed"]
    composites = [_composite_detail(row) for row in rows if row["case"]["category"] == "composite"]
    guardrails = [{"id": row["id"], "behavior": row["case"].get("expected_guardrail_behavior"), "guardrail_status": row["guardrail_status"], "evaluable": row["guardrail_evaluable"], "correct": row["guardrail_correct"], "reason": row["guardrail_reason"]} for row in rows if row["case"]["category"] == "guardrail"]
    scoring = {
        "schema_version": "financial_agent_holdout_150_v1_1_scoring",
        "mode": "offline_rescoring_only",
        "source_results": {"path": str(source_path.relative_to(ROOT)), "runtime_config": raw.get("runtime_config", {})},
        "raw_v1_metrics": raw.get("metrics", {}),
        "v1_1_scoring_protocol": SCORING_PROTOCOL,
        "v1_1_corrected_metrics": _metrics(rows),
        "gold_label_corrections": [row for row in rows if row["label_correction"]],
        "guardrail_dedicated_cases": guardrails,
        "composite_cases": composites,
        "tool_execution_diagnostics": _tool_failure_diagnostics(rows),
        "tool_execution_detail": _tool_execution_detail(rows),
        "preference_diagnostics": _preference_diagnostics(raw),
        "planner_failure_taxonomy": _planner_failure_taxonomy(rows),
    }
    return scoring, _changelog()


def _comparison_value(value: Any, denominator: dict[str, int] | None = None) -> dict[str, Any]:
    return {"value": value, "display": "N/A" if value is None else value, "denominator": denominator}


def _latency_delta(baseline: float | None, candidate: float | None) -> dict[str, float | None]:
    absolute = candidate - baseline if baseline is not None and candidate is not None else None
    return {
        "baseline": baseline,
        "candidate": candidate,
        "absolute_change_seconds": absolute,
        "percent_change": absolute / baseline if absolute is not None and baseline else None,
    }


def comparison(baseline_raw: dict[str, Any], candidate_raw: dict[str, Any], baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    """Compare two saved raw runs by recomputing both with the same v1.1 protocol."""
    baseline, _ = rescore(baseline_raw, baseline_path)
    candidate, _ = rescore(candidate_raw, candidate_path)
    base_metrics = baseline["v1_1_corrected_metrics"]
    candidate_metrics = candidate["v1_1_corrected_metrics"]
    base_preferences = baseline["preference_diagnostics"]
    candidate_preferences = candidate["preference_diagnostics"]
    metric_names = (
        "intent_accuracy", "tool_selection_accuracy", "planned_tool_coverage", "required_tool_coverage",
        "unnecessary_tool_call_rate", "tool_execution_success_rate", "company_context_accuracy",
        "overall_agent_state_period_accuracy", "explicit_period_resolution_accuracy",
        "period_memory_carryover_accuracy", "full_context_memory_recovery_accuracy",
        "thread_isolation_accuracy", "guardrail_correctness", "safe_degradation_rate",
        "composite_completion_rate",
    )
    metrics = {
        name: {
            "baseline": _comparison_value(base_metrics[name], base_metrics.get("denominators", {}).get(name)),
            "candidate": _comparison_value(candidate_metrics[name], candidate_metrics.get("denominators", {}).get(name)),
        }
        for name in metric_names
    }
    preference_metrics = {
        name: {
            "baseline": _comparison_value(base_preferences[name], base_preferences["denominators"][name]),
            "candidate": _comparison_value(candidate_preferences[name], candidate_preferences["denominators"][name]),
        }
        for name in ("explicit_preference_write_accuracy", "preference_recovery_accuracy", "implicit_preference_nonwrite_accuracy")
    }
    return {
        "schema_version": "financial_agent_holdout_150_v1_1_news_comparison",
        "mode": "offline_rescoring_only",
        "comparison_protocol": {
            "baseline": "Agent v1 (Google News), reconstructed from its saved raw result with v1.1 scoring.",
            "candidate": "Agent News v1.1 (Domestic Provider), reconstructed from its saved raw result with the identical v1.1 scoring.",
            "gold_label_corrections": LABEL_CORRECTIONS,
            "safe_degradation": "null / N/A when no eligible external provider failure exists, rather than 0%.",
        },
        "baseline_source": {"path": str(baseline_path.relative_to(ROOT)), "runtime_config": baseline_raw.get("runtime_config", {})},
        "candidate_source": {"path": str(candidate_path.relative_to(ROOT)), "runtime_config": candidate_raw.get("runtime_config", {})},
        "metrics": metrics,
        "tool_execution": {
            "baseline": baseline["tool_execution_detail"],
            "candidate": candidate["tool_execution_detail"],
        },
        "preference": preference_metrics,
        "guardrail": {
            "baseline": {name: base_metrics[name] for name in ("guardrail_evaluable_count", "guardrail_correct_count", "guardrail_unevaluable_count", "guardrail_correctness")},
            "candidate": {name: candidate_metrics[name] for name in ("guardrail_evaluable_count", "guardrail_correct_count", "guardrail_unevaluable_count", "guardrail_correctness")},
            "baseline_dedicated_cases": baseline["guardrail_dedicated_cases"],
            "candidate_dedicated_cases": candidate["guardrail_dedicated_cases"],
        },
        "composite": {
            "baseline": baseline["composite_cases"],
            "candidate": candidate["composite_cases"],
            "tracked_prior_planner_failures": {
                case_id: next(item for item in candidate["composite_cases"] if item["id"] == case_id)
                for case_id in ("agent_holdout_053", "agent_holdout_064", "agent_holdout_065", "agent_holdout_066", "agent_holdout_067")
            },
        },
        "planner_failure_taxonomy": {
            "baseline": baseline["planner_failure_taxonomy"],
            "candidate": candidate["planner_failure_taxonomy"],
        },
        "latency_seconds": {
            name: _latency_delta(base_metrics[name], candidate_metrics[name])
            for name in ("mean_latency_seconds", "p50_latency_seconds", "p95_latency_seconds")
        },
        "scoring_changelog_applied": _changelog(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline rescore for Financial Agent holdout v1.1")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--scoring", type=Path, default=DEFAULT_SCORING)
    parser.add_argument("--changelog", type=Path, default=DEFAULT_CHANGELOG)
    parser.add_argument("--baseline-results", type=Path, help="Optional raw v1 result for identical-protocol comparison.")
    parser.add_argument("--comparison", type=Path, help="Optional comparison artifact path; requires --baseline-results.")
    args = parser.parse_args()
    raw = _load(args.results)
    scoring, changelog = rescore(raw, args.results)
    _dump(args.scoring, scoring)
    _dump(args.changelog, changelog)
    if args.comparison:
        if not args.baseline_results:
            raise SystemExit("--comparison requires --baseline-results")
        baseline_raw = _load(args.baseline_results)
        _dump(args.comparison, comparison(baseline_raw, raw, args.baseline_results, args.results))
    print(json.dumps({"completed": scoring["v1_1_corrected_metrics"]["completed"], "scoring": str(args.scoring), "changelog": str(args.changelog)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
