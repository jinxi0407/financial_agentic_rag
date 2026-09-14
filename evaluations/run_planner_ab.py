"""Known-set planner comparison. Only the FC planner may access a paid service.

Every turn starts from the same query-derived context snapshot in both arms.
Downstream services and synthesis are fixtures. This is not a memory/E2E holdout.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from time import perf_counter
from types import SimpleNamespace
import uuid

from agent.function_calling_planner import FunctionCallingPlanner
from agent.langgraph_agent import LangGraphFinancialAgent
from agent.planner import FinancialPlanner
from agent.planning import PlannerSettings
from agent.preferences import RedisPreferenceStore, _METRICS
from agent.schemas import MCPToolResult
from mcp_servers.providers.market_provider import extract_securities_from_query, normalize_security
from rag_qa.core.query_metadata import extract_query_metadata
from evaluations.run_financial_agent_holdout import _record, _metrics
from evaluations.rescore_financial_agent_holdout import _corrected_row, _metrics as corrected_metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evaluations/financial_agent_holdout_150_v1.json"


def dump(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class RequestBudget:
    """Reserve before network access; interrupted attempts still consume budget."""
    def __init__(self, path, limit, local_limit):
        self.path = Path(path)
        self.limit, self.local_limit, self.used = limit, local_limit, 0
        if not 1 <= limit <= 160 or not 0 <= local_limit <= 150:
            raise ValueError("invalid request budget")

    def reserve(self):
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            text = handle.read()
            ledger = json.loads(text) if text else {"limit": self.limit, "attempted_requests": 0}
            if ledger["limit"] != self.limit or ledger["attempted_requests"] >= self.limit or self.used >= self.local_limit:
                raise RuntimeError("RequestBudgetExceeded")
            ledger["attempted_requests"] += 1
            handle.seek(0)
            json.dump(ledger, handle)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
            self.used += 1


class FixturePreferences:
    def __init__(self, initial=None): self.data = copy.deepcopy(initial or {})

    def get(self, _user):
        return copy.deepcopy({"preferred_companies": [], "preferred_metrics": [], **self.data})

    def save_explicit(self, _user, query, companies):
        if not RedisPreferenceStore._is_explicit_preference(query): return None
        metrics = list(dict.fromkeys(value for key, value in _METRICS.items() if key in query))
        codes = [c["ticker"] for c in companies]
        if not metrics and not codes: return None
        current = self.get(_user)
        self.data = {"preferred_companies": list(dict.fromkeys(current["preferred_companies"] + codes)),
                     "preferred_metrics": list(dict.fromkeys(current["preferred_metrics"] + metrics))}
        return self.get(_user)


class FixtureQA:
    def __init__(self, observed): self.observed = observed

    def query(self, query):
        self.observed.append({"tool_name": "financial_rag", "arguments": {"query": query}})
        yield "固定财报测试返回；本次仅验证规划和参数传递。", True


class FixtureMCP:
    def __init__(self, observed): self.observed = observed

    def call_tool(self, server, name, arguments):
        label = "market_mcp" if server == "market" else "news_mcp"
        self.observed.append({"tool_name": label, "arguments": copy.deepcopy(arguments)})
        security = normalize_security(company_name=arguments.get("company_name"), ticker=arguments.get("ticker"))
        success = security is not None
        payload = {**(security or {}), "success": success, "error": None if success else "fixture_invalid_target",
                   "source": "fixed_planner_fixture", "price": 20.0, "currency": "CNY", "market_time": "fixture"}
        if server == "news":
            payload["query"] = arguments["query"]
            payload["results"] = [{"title": "固定测试新闻", "source": "fixed_planner_fixture", "published_at": "2026-01-01", "url": "https://example.test/fixture"}] if success else []
        return MCPToolResult(name, server, arguments, payload, success, payload["error"], 0.0)


class FixtureSynthesis:
    def synthesize(self, payload):
        return "财务表现：固定测试返回。市场表现：固定行情。近期事件：固定新闻。", 0.0


def context_snapshots(cases):
    """Question-only replay; expected labels and model predictions never seed context."""
    sessions, preferences, snapshots = {}, {}, []
    for case in cases:
        session = copy.deepcopy(sessions.get(case["session_id"], {}))
        user = case["user_id"]
        session["long_term_preferences"] = copy.deepcopy(preferences.get(user, {}))
        snapshots.append(session)
        companies = extract_securities_from_query(case["question"]) or session.get("companies", [])
        period = LangGraphFinancialAgent._period(case["question"])
        store = FixturePreferences(preferences.get(user))
        store.save_explicit(user, case["question"], companies)
        preferences[user] = store.get(user)
        sessions[case["session_id"]] = {"companies": companies, "report_periods": [period] if period else session.get("report_periods", []),
                                       "last_query": case["question"]}
    return snapshots


def run_case(case, snapshot, planner):
    observed = []
    agent = LangGraphFinancialAgent(planner=planner, qa_system=FixtureQA(observed),
        mcp_client=FixtureMCP(observed), synthesizer=FixtureSynthesis(),
        preference_store=FixturePreferences(snapshot.get("long_term_preferences")))
    thread = "planner-fixture:" + uuid.uuid4().hex
    if snapshot:
        agent.graph.update_state({"configurable": {"thread_id": thread}}, copy.deepcopy(snapshot), as_node="guardrail")
    calculate = agent.calculator.run
    def spy_calculate(**arguments):
        observed.append({"tool_name": "calculator", "arguments": copy.deepcopy(arguments)})
        return calculate(**arguments)
    agent.calculator.run = spy_calculate
    started = perf_counter()
    state = agent.run(case["question"], thread, "fixture-user")
    row = _record(case, state, perf_counter() - started)
    expected_delivery = [{"tool_name": c["tool_name"], "arguments": c["validated_arguments"]} for c in state.get("planned_calls", [])]
    order = {name: i for i, name in enumerate(("financial_rag", "market_mcp", "news_mcp", "calculator"))}
    expected_delivery.sort(key=lambda x: order[x["tool_name"]])
    delivered_codes = set()
    for call in observed:
        args = call["arguments"]
        if call["tool_name"] == "financial_rag": delivered_codes.update(extract_query_metadata(args["query"]).company_codes)
        elif call["tool_name"] in {"market_mcp", "news_mcp"}:
            security = normalize_security(company_name=args.get("company_name"), ticker=args.get("ticker"))
            if security: delivered_codes.add(security["ticker"])
    expected_codes = set(case["expected_company_codes"])
    fc = state["planner_metadata"].get("effective_mode") == "function_calling"
    row.update({"context_snapshot": snapshot, "planning_status": state["planning_status"],
                "planner_metadata": state["planner_metadata"], "planned_calls": state["planned_calls"],
                "call_results": state["call_results"], "observed_tool_arguments": observed,
                "argument_delivery_exact": expected_delivery == observed if fc and expected_delivery else None,
                "argument_accuracy": None,
                "delivered_company_codes": sorted(delivered_codes),
                "delivered_target_coverage": len(expected_codes & delivered_codes) / len(expected_codes) if expected_codes else None,
                "delivered_target_exact": delivered_codes == expected_codes if expected_codes else None})
    return row


def average(values):
    values = [v for v in values if v is not None]
    return {"value": statistics.mean(values) if values else None, "denominator": len(values)}


def summarize(rows):
    metadata = [r["planner_metadata"] for r in rows]
    api = [m for m in metadata if m.get("request_count")]
    validations = [v for m in metadata for v in m.get("validation_results", [])]
    usage = [m["token_usage"] for m in api if m.get("token_usage") is not None]
    totals = {k: sum(u[k] for u in usage if u.get(k) is not None) for k in ("prompt_tokens", "completion_tokens", "total_tokens")} if usage else None
    composite = [r for r in rows if r["case"]["category"] == "composite"]
    same_tool_multi = [r for r in rows if len(r["case"]["expected_company_codes"]) > 1 and set(r["case"]["expected_tools"]) in ({"market_mcp"}, {"news_mcp"})]
    return {"legacy_v1_metrics": _metrics(rows),
            "legacy_v1_1_metrics": corrected_metrics([_corrected_row(r) for r in rows]),
            "planner_diagnostics": {
                "request_count": sum(m.get("request_count", 0) for m in metadata),
                "api_success_rate": average([m.get("api_success") for m in api]),
                "api_failure_rate": average([not m.get("api_success") for m in api]),
                "argument_validity": average([v["valid"] for v in validations]),
                "argument_accuracy": {"value": None, "denominator": 0, "reason": "No independently audited full argument labels; validity is not correctness."},
                "argument_delivery_exact": average([r["argument_delivery_exact"] for r in rows]),
                "delivered_company_coverage": average([r["delivered_target_coverage"] for r in rows]),
                "delivered_company_exact": average([r["delivered_target_exact"] for r in rows]),
                "same_tool_multiple_target_coverage": average([r["delivered_target_coverage"] for r in same_tool_multi]),
                "composite_planning_exact": average([r["tool_selection_correct"] for r in composite]),
                "plan_rejection_rate": average([r["planning_status"] in {"invalid_plan", "unsupported_plan"} for r in rows]),
                "rule_fallback_rate": average([m.get("fallback_used", False) for m in metadata]),
                "prehandled_count": sum(bool(m.get("prehandled")) for m in metadata),
                "no_tool_correct": sum(not r["planned_tools"] and not r["case"]["expected_tools"] for r in rows),
                "no_tool_missed": sum(not r["planned_tools"] and bool(r["case"]["expected_tools"]) for r in rows),
                "model_no_tool_correct": sum(r["planning_status"] == "no_tool" and not r["case"]["expected_tools"] for r in rows),
                "model_no_tool_missed": sum(r["planning_status"] == "no_tool" and bool(r["case"]["expected_tools"]) for r in rows),
                "planner_latency_seconds": average([m.get("planner_latency") for m in metadata]),
                "token_usage": totals, "usage_available_requests": len(usage), "usage_missing_requests": len(api) - len(usage),
            }}


def smoke_cases():
    specs = [
        ("贵州茅台2026H1营业收入是多少？", ["financial_rag"]),
        ("看看比亚迪现在股价", ["market_mcp"]),
        ("招商银行最近新闻", ["news_mcp"]),
        ("35+7*2是多少？", ["calculator"]),
        ("五粮液2025FY营业收入、当前股价和近期新闻", ["financial_rag", "market_mcp", "news_mcp"]),
        ("3+4=多少，还有茅台的股票看看", ["calculator", "market_mcp"]),
        ("贵州茅台和五粮液现在的股价分别是多少？", ["market_mcp"]),
        ("比亚迪和宁德时代最近有什么新闻？", ["news_mcp"]),
    ]
    cases = []
    for index, (query, names) in enumerate(specs):
        companies = extract_securities_from_query(query)
        cases.append({"id": f"fc_smoke_{index+1}", "category": "smoke", "question": query,
                      "session_id": str(index), "user_id": str(index),
                      "expected_intent": "composite_query" if len(names)>1 else {"financial_rag":"financial_report_query", "market_mcp":"market_query", "news_mcp":"news_query", "calculator":"calculation_query"}[names[0]],
                      "expected_tools": names, "forbidden_tools": [], "expected_company_codes": [c["ticker"] for c in companies],
                      "expected_report_periods": list(extract_query_metadata(query).report_periods),
                      "expected_memory_behavior": "none", "expected_preference_behavior": "none", "expected_guardrail_behavior": "none", "allow_external_tool_failure": False})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run"))
    parser.add_argument("--planner-mode", choices=("rule", "function_calling"), required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-file", type=Path)
    parser.add_argument("--request-limit", type=int, default=150)
    parser.add_argument("--total-request-limit", type=int, default=160)
    parser.add_argument("--confirm-model-calls", action="store_true")
    args = parser.parse_args()
    if args.output.exists(): raise SystemExit("Output already exists; refusing overwrite or implicit replay")
    if args.planner_mode == "function_calling" and (not args.confirm_model_calls or not args.budget_file):
        raise SystemExit("FC requires --confirm-model-calls and a shared --budget-file")
    cases = smoke_cases() if args.command == "smoke" else json.loads(args.dataset.read_text())["cases"]
    if args.command == "run" and len(cases) != 150: raise SystemExit("Expected frozen 150-turn dataset")
    snapshots = context_snapshots(cases)
    from base.config import config
    settings = PlannerSettings(mode=args.planner_mode, model=config.LLM_MODEL, fallback_to_rule=False)
    cap = min(args.request_limit, 10 if args.command == "smoke" else 150)
    budget = RequestBudget(args.budget_file, args.total_request_limit, cap) if args.planner_mode == "function_calling" else None
    planner = FunctionCallingPlanner(settings, before_request=budget.reserve) if budget else FinancialPlanner()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    protocol = {"name": "known_150_planner_ab_v1", "planner_mode": args.planner_mode,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "shared_base_commit": "f2cd9856c88e0c4d0270e4a5bd935f9a41166492", "model": settings.model if budget else None,
                "dataset_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "context_sha256": hashlib.sha256(json.dumps(snapshots, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "context_source": "query_only_fixed_snapshots_not_gold_not_model_output",
                "execution": "fixed_fake_tools_and_synthesis_in_memory_checkpoint_no_business_stores",
                "rule_fallback": False, "sdk_max_retries": 0, "shared_fix": "reset period_clarification every turn",
                "limitations": ["known regression set, not unseen", "not full memory E2E", "no independently audited full argument accuracy", "FC receives structured context; rule retains original bool interface"]}
    source_paths = [*sorted((ROOT / "agent").rglob("*.py")), ROOT / "base/config.py", Path(__file__).resolve()]
    protocol["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    rows = []
    dump(args.output, {"protocol": protocol, "results": rows})
    for case, snapshot in zip(cases, snapshots):
        row = run_case(case, snapshot, planner)
        rows.append(row)
        dump(args.output, {"protocol": protocol, "results": rows, "summary": summarize(rows)})
        print(json.dumps({"completed": len(rows), "total": len(cases), "id": case["id"], "status": row["planning_status"], "requests": budget.used if budget else 0}), flush=True)
        if args.command == "smoke" and (row["planning_status"] != "ready" or not row["tool_selection_correct"] or row["argument_delivery_exact"] is not True):
            raise SystemExit("Smoke stopped on unexpected result; no automatic retry")
        if row["planner_metadata"].get("error_type") == "RuntimeError" and budget and budget.used >= cap:
            raise SystemExit("Request budget exhausted")
    print(json.dumps(summarize(rows)["planner_diagnostics"], ensure_ascii=False))


if __name__ == "__main__": main()
