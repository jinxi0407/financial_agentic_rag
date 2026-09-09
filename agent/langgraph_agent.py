"""Minimal LangGraph orchestration over existing Agent tools."""
from __future__ import annotations

import os
import re
from contextvars import ContextVar
from decimal import Decimal, InvalidOperation
from queue import Queue
from threading import Thread
from time import perf_counter
from typing import Any, Callable, Iterator, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.redis import RedisSaver

from mcp_servers.providers.market_provider import extract_securities_from_query

from .mcp_client import FinancialMCPClient
from .planner import FinancialPlanner
from .preferences import RedisPreferenceStore
from .synthesis import QwenSynthesizer
from .tools.calculator_tool import CalculatorTool
from .tools.financial_rag_tool import FinancialRAGTool


_STREAM_OBSERVER: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar(
    "agent_stream_observer", default=None
)


class AgentState(TypedDict, total=False):
    query: str
    thread_id: str
    user_id: str
    intent: str
    skill: str | None
    companies: list[dict[str, str]]
    tickers: list[str]
    report_periods: list[str]
    required_tools: list[str]
    executed_tools: list[str]
    tool_results: list[dict[str, Any]]
    financial_result: str
    market_results: list[dict[str, Any]]
    news_results: list[dict[str, Any]]
    calculation_result: dict[str, Any]
    errors: list[str]
    previous_query: str
    last_query: str
    last_intent: str
    long_term_preferences: dict[str, list[str]]
    draft_answer: str
    synthesis_answer: str
    synthesis_latency: float
    guardrail_status: str
    final_answer: str
    graph_path: list[str]
    trace: dict[str, Any]


class LangGraphFinancialAgent:
    """StateGraph with short-term thread memory; no retrieval logic is duplicated."""

    def __init__(self, qa_system=None, checkpointer=None, planner=None, mcp_client=None, synthesizer=None, preference_store=None):
        self.planner = planner or FinancialPlanner()
        self.rag_tool = FinancialRAGTool(qa_system=qa_system) if qa_system else FinancialRAGTool()
        self.calculator = CalculatorTool()
        self.mcp_client = mcp_client or FinancialMCPClient()
        self.synthesizer = synthesizer
        self.preference_store = preference_store or RedisPreferenceStore()
        self.graph = self._build_graph().compile(checkpointer=checkpointer or InMemorySaver())

    @staticmethod
    def redis_checkpointer(redis_url: str | None = None) -> RedisSaver:
        if redis_url is None:
            host = os.getenv("AGENT_REDIS_HOST", "127.0.0.1")
            port = os.getenv("AGENT_REDIS_PORT", "6380")
            database = os.getenv("AGENT_REDIS_DB", "0")
            password = os.getenv("AGENT_REDIS_PASSWORD")
            auth = f":{password}@" if password else ""
            redis_url = f"redis://{auth}{host}:{port}/{database}"
        prefix = os.getenv("AGENT_REDIS_PREFIX", "financial:agent:")
        saver = RedisSaver(
            redis_url=redis_url,
            checkpoint_prefix=f"{prefix}memory",
            checkpoint_write_prefix=f"{prefix}memory:write",
        )
        saver.setup()
        return saver

    def run(self, query: str, thread_id: str, user_id: str | None = None) -> dict[str, Any]:
        started = perf_counter()
        state = self.graph.invoke(
            {"query": query, "thread_id": thread_id, "user_id": user_id or thread_id},
            {"configurable": {"thread_id": thread_id}},
        )
        latencies = {}
        for result in state.get("tool_results", []):
            tool_name = result.get("tool_name")
            if tool_name:
                latencies[tool_name] = latencies.get(tool_name, 0.0) + result.get("latency", 0.0)
        if state.get("synthesis_latency") is not None:
            latencies["synthesis"] = state["synthesis_latency"]
        state["trace"] = {
            **state.get("trace", {}),
            "thread_id": thread_id,
            "intent": state.get("intent"),
            "skill": state.get("skill"),
            "graph_path": ["plan", "financial_rag", "market", "news", "calculator", "aggregation", "synthesis", "guardrail"],
            "executed_tools": state.get("executed_tools", []),
            "per_tool_latency": latencies,
            "total_latency": perf_counter() - started,
            "success": not state.get("errors"),
        }
        return state

    def stream(self, query: str, thread_id: str, user_id: str | None = None) -> Iterator[dict[str, Any]]:
        """Yield lifecycle notifications while preserving ``run`` as the stable API.

        Nodes remain synchronous because the frozen tools are synchronous.  A
        request-local ContextVar carries an observer only for this invocation;
        no callback is written into checkpointed LangGraph state.
        """
        events: Queue[object] = Queue()
        finished = object()

        def publish(event: dict[str, Any]) -> None:
            events.put(event)

        def invoke() -> None:
            observer_token = _STREAM_OBSERVER.set(publish)
            try:
                state = self.run(query, thread_id, user_id)
                publish({"kind": "complete", "state": state})
            except Exception as exc:  # The public adapter deliberately redacts this.
                publish({"kind": "error", "error_type": type(exc).__name__})
            finally:
                _STREAM_OBSERVER.reset(observer_token)
                events.put(finished)

        Thread(target=invoke, name="financial-agent-stream", daemon=True).start()
        while True:
            event = events.get()
            if event is finished:
                return
            yield event  # type: ignore[misc]

    @staticmethod
    def _emit(kind: str, **data: Any) -> None:
        observer = _STREAM_OBSERVER.get()
        if observer is not None:
            observer({"kind": kind, "data": data})

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("financial_rag", self._financial_rag)
        graph.add_node("market", self._market)
        graph.add_node("news", self._news)
        graph.add_node("calculator", self._calculator)
        graph.add_node("aggregation", self._aggregation)
        graph.add_node("synthesis", self._synthesis)
        graph.add_node("guardrail", self._guardrail)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "financial_rag")
        graph.add_edge("financial_rag", "market")
        graph.add_edge("market", "news")
        graph.add_edge("news", "calculator")
        graph.add_edge("calculator", "aggregation")
        graph.add_edge("aggregation", "synthesis")
        graph.add_edge("synthesis", "guardrail")
        graph.add_edge("guardrail", END)
        return graph

    def _plan(self, state: AgentState):
        decision = self.planner.plan(state["query"])
        companies = extract_securities_from_query(state["query"]) or state.get("companies", [])
        period = self._period(state["query"])
        preferences, preference_error = self._preferences(state.get("user_id", state["thread_id"]))
        preference_written = False
        try:
            saved = self.preference_store.save_explicit(
                state.get("user_id", state["thread_id"]), state["query"], companies
            )
            if saved is not None:
                preferences = saved
                preference_written = True
        except Exception as exc:
            preference_error = f"preference_store: {type(exc).__name__}"
        response = {"intent": decision.intent, "skill": decision.skill, "required_tools": list(decision.tools), "companies": companies, "tickers": [item["ticker"] for item in companies], "report_periods": [period] if period else state.get("report_periods", []), "previous_query": state.get("last_query", ""), "last_query": state["query"], "last_intent": decision.intent, "long_term_preferences": preferences, "executed_tools": [], "tool_results": [], "financial_result": "", "market_results": [], "news_results": [], "calculation_result": {}, "errors": [], "draft_answer": "", "synthesis_answer": "", "trace": {"plan": decision.to_dict(), "preference_written": preference_written, "preference_error": preference_error}}
        self._emit("plan", intent=decision.intent, required_tools=list(decision.tools), skill=decision.skill)
        return response

    @staticmethod
    def _period(query: str) -> str | None:
        import re
        match = re.search(r"(20\d{2})(?:年)?(?:H1|上半年|半年度)", query, re.I)
        if match:
            return f"{match.group(1)}H1"
        match = re.search(r"(20\d{2})(?:年)?(?:FY|年度|全年|年报)", query, re.I)
        return f"{match.group(1)}FY" if match else None

    def _financial_rag(self, state: AgentState):
        if "financial_rag" not in state.get("required_tools", []): return {}
        self._emit("tool_start", tool="financial_rag")
        result = self.rag_tool.run(self._effective_financial_query(state))
        response = {"financial_result": result.answer, "tool_results": [*state.get("tool_results", []), result.to_dict()], "executed_tools": [*state.get("executed_tools", []), "financial_rag"], "errors": [*state.get("errors", []), *([result.error] if result.error else [])]}
        self._emit("tool_end", tool="financial_rag", results=[result.to_dict()])
        return response

    def _market(self, state: AgentState):
        if "market_mcp" not in state.get("required_tools", []): return {}
        self._emit("tool_start", tool="market_mcp")
        response = self._mcp_many(state, "market", "get_market_snapshot", "market_mcp", "market_results")
        self._emit("tool_end", tool="market_mcp", results=response.get("market_results", []))
        return response

    def _news(self, state: AgentState):
        if "news_mcp" not in state.get("required_tools", []): return {}
        self._emit("tool_start", tool="news_mcp")
        response = self._mcp_many(state, "news", "search_financial_news", "news_mcp", "news_results")
        self._emit("tool_end", tool="news_mcp", results=response.get("news_results", []))
        return response

    def _mcp_many(self, state, server, tool, label, target):
        companies = state.get("companies") or []
        if not companies:
            return {"errors": ["缺少公司上下文，无法调用实时工具。"]}
        results = []
        errors = []
        for company in companies:
            args = {"company_name": company["company_name"], "ticker": company["ticker"]}
            if server == "news": args.update({"query": state["query"], "max_results": 5})
            result = self.mcp_client.call_tool(server, tool, args)
            results.append(result.to_dict())
            if result.error: errors.append(result.error)
        return {target: results, "tool_results": [*state.get("tool_results", []), *results], "executed_tools": [*state.get("executed_tools", []), label], "errors": [*state.get("errors", []), *errors]}

    def _calculator(self, state):
        if "calculator" not in state.get("required_tools", []): return {}
        self._emit("tool_start", tool="calculator")
        request = self.planner.simple_calculation_request(state["query"])
        if not request:
            self._emit("tool_end", tool="calculator", results=[])
            return {"errors": ["缺少可验证的计算输入。"]}
        result = self.calculator.run(**request)
        response = {"calculation_result": result.to_dict(), "tool_results": [*state.get("tool_results", []), result.to_dict()], "executed_tools": [*state.get("executed_tools", []), "calculator"], "errors": [*state.get("errors", []), *([result.error] if result.error else [])]}
        self._emit("tool_end", tool="calculator", results=[result.to_dict()])
        return response

    def _aggregation(self, state):
        if state.get("intent") == "greeting":
            return {"draft_answer": self._greeting_answer(state.get("query", ""))}
        if state.get("intent") == "unsupported":
            return {"draft_answer": self._unsupported_answer()}
        if state.get("intent") != "composite_query":
            financial_result = self._tool_result(state, "financial_rag")
            if financial_result and financial_result.get("success"):
                guidance = self._report_period_guidance(state.get("query", ""))
                if guidance:
                    return {"draft_answer": guidance}
                return {"draft_answer": financial_result["answer"]}
            if state.get("market_results"):
                return {"draft_answer": self._market_answer(state["market_results"])}
            if state.get("news_results"):
                return {"draft_answer": self._news_answer(state["news_results"])}
            if state.get("calculation_result"):
                return {"draft_answer": self._calculator_answer(state["calculation_result"], state.get("query", ""))}
        return {"draft_answer": self._safe_summary(state)}

    def _synthesis(self, state):
        if state.get("intent") != "composite_query":
            return {"synthesis_answer": state.get("draft_answer", ""), "synthesis_latency": 0.0}
        self._emit("synthesis_start")
        try:
            answer, latency = self._get_synthesizer().synthesize(self._synthesis_payload(state))
            return {"synthesis_answer": answer, "synthesis_latency": latency}
        except Exception as exc:
            return {
                "synthesis_answer": state.get("draft_answer", ""),
                "synthesis_latency": 0.0,
                "errors": [*state.get("errors", []), f"synthesis: {type(exc).__name__}"],
            }

    def _guardrail(self, state):
        candidate = state.get("synthesis_answer") or state.get("draft_answer")
        violations = self._guardrail_violations(state, candidate)
        if violations:
            response = {
                "final_answer": self._safe_summary(state, guardrail_notice=True),
                "guardrail_status": ",".join(violations),
            }
            self._emit("guardrail", status="degraded")
            return response
        self._emit("guardrail", status="passed")
        return {"final_answer": candidate, "guardrail_status": "passed"}

    def _get_synthesizer(self):
        if self.synthesizer is None:
            system = self.rag_tool._system()
            if system.client is None:
                raise RuntimeError("Qwen client is not configured")
            self.synthesizer = QwenSynthesizer(system.client, system.config.LLM_MODEL)
        return self.synthesizer

    def _preferences(self, user_id):
        try:
            return self.preference_store.get(user_id), None
        except Exception as exc:
            return {"preferred_companies": [], "preferred_metrics": []}, f"preference_store: {type(exc).__name__}"

    @staticmethod
    def _synthesis_payload(state):
        return {
            "user_query": state["query"],
            "session_context": {
                "companies": state.get("companies", []),
                "report_periods": state.get("report_periods", []),
                "previous_query": state.get("previous_query", ""),
                "preferences": state.get("long_term_preferences", {}),
            },
            "financial_result": state.get("financial_result", ""),
            "market_results": state.get("market_results", []),
            "news_results": state.get("news_results", []),
            "verified_calculation": state.get("calculation_result", {}),
            "errors": state.get("errors", []),
        }

    def _guardrail_violations(self, state, candidate):
        violations = []
        if "market_mcp" in state.get("required_tools", []) and self._failed_tool_results(state.get("market_results", [])):
            violations.append("market_unavailable")
        if "news_mcp" in state.get("required_tools", []) and self._empty_or_failed_news(state.get("news_results", [])):
            violations.append("news_unavailable")
        if self._missing_news_provenance(state.get("news_results", []), candidate):
            violations.append("news_provenance_missing")
        calculation_requested = any(term in state.get("query", "") for term in ("同比", "环比", "增长率", "增加", "减少", "差值", "百分点"))
        if calculation_requested and not state.get("calculation_result", {}).get("success"):
            if any(term in candidate for term in ("同比", "环比", "增长率", "增加", "减少", "差值", "百分点")):
                violations.append("unverified_calculation")
        financial_result = self._tool_result(state, "financial_rag")
        if financial_result and financial_result.get("success"):
            allowed = self._numbers(financial_result.get("answer", ""))
            guidance = self._report_period_guidance(state.get("query", ""))
            if guidance:
                allowed.update(self._numbers(guidance))
            allowed.update(self._numbers(str(state.get("market_results", []))))
            allowed.update(self._numbers(str(state.get("news_results", []))))
            allowed.update(self._numbers(str(state.get("calculation_result", {}))))
            allowed.update(self._numbers(state.get("query", "")))
            if self._numbers(candidate) - allowed:
                violations.append("unsupported_numeric_claim")
        return violations

    @staticmethod
    def _tool_result(state, tool_name):
        return next((item for item in state.get("tool_results", []) if item.get("tool_name") == tool_name), None)

    def _effective_financial_query(self, state: AgentState) -> str:
        """Bind an explicit follow-up period to remembered companies for RAG."""
        query = state["query"]
        if extract_securities_from_query(query) or not self._period(query):
            return query
        companies = state.get("companies") or []
        if not companies:
            return query
        company_names = "、".join(company["company_name"] for company in companies)
        return f"{company_names} {self._period(query)} 财报"

    @staticmethod
    def _greeting_answer(query: str) -> str:
        if "什么" in query or "能做" in query:
            return "我是 Financial Agent，目前支持 8 家 A 股公司的财报、实时行情、财经新闻和财务计算。"
        return "你好，我是 Financial Agent，目前支持 8 家 A 股公司的财报、实时行情、财经新闻和财务计算。"

    @staticmethod
    def _unsupported_answer() -> str:
        return "当前 Financial Agent 目前支持 8 家 A 股公司的财报、实时行情、财经新闻和确定性财务计算。例如：‘贵州茅台 2026H1 营业收入是多少？’"

    @staticmethod
    def _report_period_guidance(query: str) -> str | None:
        if not any(term in query for term in ("财报", "报告")):
            return None
        if re.search(r"20\d{2}", query):
            return None
        if any(term in query for term in ("半年报", "半年度", "上半年")):
            return "当前可用半年度报告：2025H1、2026H1。请指定希望查看的年份。"
        if any(term in query for term in ("年度报告", "年报", "全年")):
            return "当前可用年度报告：2025FY。该期间为全年口径。"
        return "当前可用报告期间：2025H1、2025FY、2026H1。请指定希望查看的期间。"

    @staticmethod
    def _calculator_answer(result: dict[str, Any], query: str) -> str:
        if not result.get("success"):
            return "当前 Calculator 工具无法完成该计算，请提供明确的数值输入。"
        value = result.get("result")
        if value is None:
            return "当前 Calculator 工具未返回可展示结果。"
        operation = result.get("operation")
        if operation == "growth_rate":
            return f"增长率为 {value:.1f}%。"
        if operation == "percentage_point_change":
            direction = "下降" if value < 0 else "上升"
            return f"百分点变化为 {abs(value):.1f} 个百分点（{direction}）。"
        if operation == "absolute_change":
            direction = "减少" if "减少" in query or "下降" in query else "增加"
            return f"{direction}了 {abs(value):.1f}。"
        if operation == "ratio":
            return f"计算结果为 {value:.2f}。"
        return f"计算结果为 {value}。"

    @staticmethod
    def _market_answer(results: list[dict[str, Any]]) -> str:
        rows = []
        for item in results:
            payload = item.get("result") or {}
            if not item.get("success"):
                rows.append("当前 Market MCP 未返回可用行情，不提供价格结论。")
                continue
            lines = [f"{payload.get('company_name')}（{payload.get('ticker')}）"]
            if payload.get("price") is not None:
                lines.append(f"当前价格：¥{payload['price']}")
            if payload.get("change_percent") is not None:
                lines.append(f"涨跌幅：{payload['change_percent']}%")
            if payload.get("high") is not None:
                lines.append(f"今日最高：¥{payload['high']}")
            if payload.get("low") is not None:
                lines.append(f"今日最低：¥{payload['low']}")
            if payload.get("market_time"):
                lines.append(f"数据时间：{payload['market_time']}")
            if payload.get("source"):
                lines.append(f"数据源：{payload['source']}")
            rows.append("\n".join(lines))
        return "\n\n".join(rows) or "当前未返回行情数据。"

    @staticmethod
    def _news_answer(results: list[dict[str, Any]]) -> str:
        rows = []
        for tool_result in results:
            for item in (tool_result.get("result") or {}).get("results", []):
                rows.append(f"- {item.get('title')}（来源：{item.get('source')}；发布时间：{item.get('published_at')}）")
        return "\n".join(rows) or "当前未检索到相关新闻。"

    @staticmethod
    def _failed_tool_results(results):
        return not results or any(not item.get("success") for item in results)

    @staticmethod
    def _empty_or_failed_news(results):
        return not results or any(
            not item.get("success") or not (item.get("result") or {}).get("results") for item in results
        )

    @staticmethod
    def _missing_news_provenance(results, candidate):
        for tool_result in results:
            for item in (tool_result.get("result") or {}).get("results", []):
                title = item.get("title")
                if title and title in candidate:
                    if not item.get("source") or not item.get("published_at"):
                        return True
                    if item["source"] not in candidate or item["published_at"] not in candidate:
                        return True
        return False

    @staticmethod
    def _numbers(text):
        normalized = set()
        for match in re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?", text):
            suffix = "%" if match.endswith("%") else ""
            try:
                normalized.add(f"{Decimal(match.rstrip('%').replace(',', '')).normalize()}{suffix}")
            except InvalidOperation:
                normalized.add(match.replace(",", ""))
        return normalized

    def _safe_summary(self, state, guardrail_notice=False):
        sections = []
        financial_result = self._tool_result(state, "financial_rag")
        if financial_result and financial_result.get("success"):
            sections.append("财务表现\n" + financial_result["answer"])
        elif "financial_rag" in state.get("required_tools", []):
            sections.append("财务表现\nFinancial RAG 未能返回可验证结果。")
        if state.get("market_results"):
            if self._failed_tool_results(state["market_results"]):
                sections.append("市场表现\nMarket MCP 未返回完整可用行情，因此不提供价格结论。")
            else:
                rows = []
                for item in state["market_results"]:
                    payload = item.get("result") or {}
                    rows.append(f"{payload.get('company_name')}（{payload.get('ticker')}）：{payload.get('price')} {payload.get('currency', '')}，涨跌幅 {payload.get('change_percent')}%，数据时间 {payload.get('market_time')}，来源 {payload.get('source')}。")
                sections.append("市场表现\n" + "\n".join(rows))
        if state.get("news_results"):
            if self._empty_or_failed_news(state["news_results"]):
                sections.append("近期事件\nNews MCP 未返回可用新闻，不据此作出事件判断。")
            else:
                rows = []
                for tool_result in state["news_results"]:
                    for item in (tool_result.get("result") or {}).get("results", []):
                        rows.append(f"- {item.get('title')}（来源：{item.get('source')}；发布时间：{item.get('published_at')}）")
                sections.append("近期事件\n" + "\n".join(rows))
        if state.get("calculation_result", {}).get("success"):
            result = state["calculation_result"]
            sections.append(f"确定性计算\n{result.get('operation')} = {result.get('result')}")
        elif any(term in state.get("query", "") for term in ("同比", "环比", "增长率", "增加", "减少", "差值", "百分点")):
            sections.append("计算说明\n当前证据未产生经 Calculator 验证的计算结果，因此不自行推导同比、增长率或差值。")
        if guardrail_notice:
            sections.append("数据来源\n为避免引入未经工具验证的信息，已仅保留可追溯的工具结果。")
        return "\n\n".join(sections) or "当前缺少足够上下文，无法可靠回答。"
