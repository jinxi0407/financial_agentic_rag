"""Development-only regression runner for real-user Agent paraphrases.

The runner uses the real Planner and LangGraph nodes, with deterministic fake
tool payloads.  It intentionally does not call Qwen, Milvus, public MCP
providers, Redis, or the frozen benchmark suites.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agent.langgraph_agent import LangGraphFinancialAgent
from agent.schemas import MCPToolResult


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "financial_real_user_paraphrase_regression_v1.json"


class RegressionQA:
    """Stable FinancialRAGTool stand-in for routing and presentation checks."""

    def query(self, query: str):
        if "财报" in query and not any(period in query.lower() for period in ("h1", "fy", "上半年", "年度", "全年", "年报")):
            answer = "当前可用报告期间：2025H1、2025FY、2026H1。请指定希望查看的期间。"
        elif any(term in query.lower() for term in ("流动比率", "速动比率", "市盈率", "roe", "归母净利润", "营业收入和净利润")):
            answer = "FAQ/BM25 fast path：这是金融指标定义说明。"
        else:
            answer = f"Financial RAG 已处理：{query}"
        yield answer, False
        yield "", True


class RegressionMCP:
    """Deterministic tool-shaped payloads, not provider simulations."""

    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        company = arguments.get("company_name") or "未知公司"
        ticker = arguments.get("ticker") or "------"
        if server_name == "market":
            payload = {
                "company_name": company,
                "ticker": ticker,
                "price": 100.25,
                "change_percent": 1.5,
                "high": 101.0,
                "low": 98.8,
                "currency": "CNY",
                "market_time": "2026-09-09T09:30:00+08:00",
                "source": "Regression market payload",
            }
        else:
            payload = {
                "results": [{
                    "title": f"{company} Regression 新闻",
                    "source": "Regression news payload",
                    "published_at": "2026-09-09T09:30:00+08:00",
                    "url": "https://example.com/regression-news",
                }]
            }
        return MCPToolResult(tool_name, server_name, arguments, payload, True, None, 0.001)


class RegressionSynthesizer:
    def synthesize(self, payload: dict[str, Any]) -> tuple[str, float]:
        sections = [f"财务表现：{payload.get('financial_result', '')}"]
        for market in payload.get("market_results", []):
            result = market.get("result") or {}
            if market.get("success"):
                sections.append(f"市场表现：{result.get('company_name')}当前价格 {result.get('price')}，来源 {result.get('source')}。")
        for news in payload.get("news_results", []):
            for item in (news.get("result") or {}).get("results", []):
                sections.append(f"近期事件：{item.get('title')}（{item.get('source')}，{item.get('published_at')}）。")
        return "\n".join(sections), 0.001


class RegressionPreferenceStore:
    def get(self, _user_id: str) -> dict[str, list[str]]:
        return {"preferred_companies": [], "preferred_metrics": []}

    def save_explicit(self, _user_id: str, _query: str, _companies: list[dict[str, str]]):
        return None


def make_agent() -> LangGraphFinancialAgent:
    return LangGraphFinancialAgent(
        qa_system=RegressionQA(),
        mcp_client=RegressionMCP(),
        synthesizer=RegressionSynthesizer(),
        preference_store=RegressionPreferenceStore(),
    )


def load_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def audit_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    cases = dataset["cases"]
    identifiers = [case["id"] for case in cases]
    questions = [case["question"] for case in cases]
    categories = Counter(case["category"] for case in cases)
    failures = []
    if not 40 <= len(cases) <= 120:
        failures.append("case_count_out_of_range")
    if len(identifiers) != len(set(identifiers)):
        failures.append("duplicate_ids")
    if len(questions) != len(set(questions)):
        failures.append("duplicate_questions")
    for session_id, turns in _group_turns(cases).items():
        indexes = [case["turn_index"] for case in turns]
        if indexes != list(range(1, len(indexes) + 1)):
            failures.append(f"non_sequential_turns:{session_id}")
    return {
        "passed": not failures,
        "case_count": len(cases),
        "categories": dict(sorted(categories.items())),
        "memory_sessions": len({case["session_id"] for case in cases if case["category"] in {"memory_pronoun", "memory_period_carryover", "memory_market_carryover", "memory_explicit_override"}}),
        "failures": failures,
    }


def _group_turns(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    sessions: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        sessions.setdefault(case["session_id"], []).append(case)
    for turns in sessions.values():
        turns.sort(key=lambda item: item["turn_index"])
    return sessions


def _check_answer(case: dict[str, Any], state: dict[str, Any]) -> dict[str, bool]:
    answer = state.get("final_answer", "")
    checks: dict[str, bool] = {}
    for check in case.get("checks", []):
        if check == "greeting":
            checks[check] = "你好" in answer and "Financial Agent" in answer
        elif check == "identity":
            checks[check] = "Financial Agent" in answer and "8 家" in answer
        elif check == "faq_route":
            checks[check] = "financial_rag" in state.get("executed_tools", []) and "FAQ/BM25" in answer
        elif check == "calculator_percent":
            checks[check] = "25.0%" in answer
        elif check == "calculator_absolute":
            expected_direction = "减少" if "减少" in case["question"] else "增加"
            checks[check] = "20" in answer and expected_direction in answer
        elif check == "calculator_ratio":
            checks[check] = "0.25" in answer
        elif check == "calculator_percentage_point":
            checks[check] = "0.2" in answer and "百分点" in answer
        elif check == "market_format":
            checks[check] = "当前价格：" in answer and "数据源：" in answer and "{'" not in answer
        elif check == "memory_company":
            expected = set(case.get("expected_company_codes", []))
            actual = set(state.get("tickers", []))
            checks[check] = expected == actual
        elif check == "period_company_carryover":
            expected_companies = {company["company_name"] for company in state.get("companies", [])}
            expected_periods = set(case.get("expected_report_periods", []))
            rag_query = next(
                (result.get("query", "") for result in state.get("tool_results", []) if result.get("tool_name") == "financial_rag"),
                "",
            )
            checks[check] = (
                expected_companies
                and expected_periods == set(state.get("report_periods", []))
                and all(company in rag_query for company in expected_companies)
                and all(period in rag_query for period in expected_periods)
            )
        elif check == "market_pronoun_carryover":
            checks[check] = (
                state.get("intent") == "market_query"
                and state.get("required_tools") == ["market_mcp"]
                and state.get("executed_tools") == ["market_mcp"]
            )
        elif check == "arithmetic_exact":
            result = state.get("calculation_result", {})
            checks[check] = result.get("success") and result.get("result") == case.get("expected_arithmetic_result")
        elif check == "no_company_pronoun":
            checks[check] = (
                not state.get("tickers")
                and not state.get("required_tools")
                and not state.get("executed_tools")
                and not state.get("errors")
                and state.get("guardrail_status") == "passed"
                and "请先说明要查询的公司" in answer
            )
        elif check == "composite_all_success":
            checks[check] = (
                state.get("guardrail_status") == "passed"
                and {"financial_rag", "market_mcp", "news_mcp"}.issubset(state.get("executed_tools", []))
                and "行情不可用" not in answer
                and "新闻不可用" not in answer
                and all(section in answer for section in ("财务表现", "市场表现", "近期事件"))
            )
        elif check == "period_clarity":
            checks[check] = all(period in answer for period in case.get("expected_period_options", []))
        elif check == "friendly_unsupported":
            checks[check] = "支持" in answer and "例如" in answer and "缺少足够上下文" not in answer
    return checks


def run(dataset: dict[str, Any]) -> dict[str, Any]:
    agent = make_agent()
    rows = []
    for case in dataset["cases"]:
        state = agent.run(case["question"], thread_id=f"regression:{case['session_id']}", user_id="regression-user")
        expected_tools = set(case.get("expected_tools", []))
        actual_tools = set(state.get("required_tools", []))
        expected_companies = set(case.get("expected_company_codes", []))
        answer_checks = _check_answer(case, state)
        effective_financial_query = next(
            (result.get("query") for result in state.get("tool_results", []) if result.get("tool_name") == "financial_rag"),
            None,
        )
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "expected_intent": case["expected_intent"],
            "actual_intent": state.get("intent"),
            "expected_tools": sorted(expected_tools),
            "actual_tools": sorted(actual_tools),
            "executed_tools": state.get("executed_tools", []),
            "expected_company_codes": sorted(expected_companies),
            "actual_company_codes": state.get("tickers", []),
            "expected_report_periods": case.get("expected_report_periods", []),
            "actual_report_periods": state.get("report_periods", []),
            "effective_financial_query": effective_financial_query,
            "answer": state.get("final_answer", ""),
            "intent_pass": state.get("intent") == case["expected_intent"],
            "tools_pass": actual_tools == expected_tools,
            "company_pass": not expected_companies or expected_companies == set(state.get("tickers", [])),
            "answer_checks": answer_checks,
            "format_pass": all(answer_checks.values()) if answer_checks else True,
        })
    return {"dataset_version": dataset["version"], "audit": audit_dataset(dataset), "rows": rows, "metrics": _metrics(rows)}


def _rate(rows: list[dict[str, Any]], predicate) -> dict[str, Any]:
    total = len(rows)
    passed = sum(bool(predicate(row)) for row in rows)
    return {"passed": passed, "total": total, "rate": round(passed / total, 4) if total else None}


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    memory = [row for row in rows if row["category"] == "memory_pronoun" and "memory_company" in row["answer_checks"]]
    period_carryover = [row for row in rows if row["category"] == "memory_period_carryover" and "period_company_carryover" in row["answer_checks"]]
    market_carryover = [row for row in rows if row["category"] == "memory_market_carryover" and "market_pronoun_carryover" in row["answer_checks"]]
    arithmetic = [row for row in rows if "arithmetic_exact" in row["answer_checks"]]
    no_context_pronoun = [row for row in rows if "no_company_pronoun" in row["answer_checks"]]
    definitions = [row for row in rows if row["category"] == "definition_faq"]
    unsupported = [row for row in rows if row["category"] == "unsupported"]
    format_rows = [row for row in rows if row["answer_checks"]]
    return {
        "intent_routing_accuracy": _rate(rows, lambda row: row["intent_pass"]),
        "required_tool_accuracy": _rate(rows, lambda row: row["tools_pass"]),
        "company_context_accuracy": _rate([row for row in rows if row["expected_company_codes"]], lambda row: row["company_pass"]),
        "memory_pronoun_recovery": _rate(memory, lambda row: row["answer_checks"].get("memory_company")),
        "period_company_carryover": _rate(period_carryover, lambda row: row["answer_checks"].get("period_company_carryover")),
        "market_pronoun_carryover": _rate(market_carryover, lambda row: row["answer_checks"].get("market_pronoun_carryover")),
        "simple_arithmetic": _rate(arithmetic, lambda row: row["answer_checks"].get("arithmetic_exact")),
        "no_context_pronoun_safety": _rate(no_context_pronoun, lambda row: row["answer_checks"].get("no_company_pronoun")),
        "definition_faq_routing": _rate(definitions, lambda row: row["intent_pass"] and row["tools_pass"] and row["format_pass"]),
        "unsupported_handling": _rate(unsupported, lambda row: row["intent_pass"] and row["format_pass"]),
        "final_formatting_checks": _rate(format_rows, lambda row: row["format_pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "run"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset = load_dataset()
    result = audit_dataset(dataset) if args.command == "audit" else run(dataset)
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"] if args.command == "run" else result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
