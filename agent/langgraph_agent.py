"""Minimal LangGraph orchestration over existing Agent tools."""
from __future__ import annotations

import os
from time import perf_counter
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.redis import RedisSaver

from mcp_servers.providers.market_provider import extract_securities_from_query

from .mcp_client import FinancialMCPClient
from .planner import FinancialPlanner
from .tools.calculator_tool import CalculatorTool
from .tools.financial_rag_tool import FinancialRAGTool


class AgentState(TypedDict, total=False):
    query: str
    thread_id: str
    intent: str
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
    last_intent: str
    final_answer: str
    trace: dict[str, Any]


class LangGraphFinancialAgent:
    """StateGraph with short-term thread memory; no retrieval logic is duplicated."""

    def __init__(self, qa_system=None, checkpointer=None, planner=None, mcp_client=None):
        self.planner = planner or FinancialPlanner()
        self.rag_tool = FinancialRAGTool(qa_system=qa_system) if qa_system else FinancialRAGTool()
        self.calculator = CalculatorTool()
        self.mcp_client = mcp_client or FinancialMCPClient()
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

    def run(self, query: str, thread_id: str) -> dict[str, Any]:
        started = perf_counter()
        state = self.graph.invoke({"query": query, "thread_id": thread_id}, {"configurable": {"thread_id": thread_id}})
        state["trace"] = {**state.get("trace", {}), "thread_id": thread_id, "total_latency": perf_counter() - started}
        return state

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("financial_rag", self._financial_rag)
        graph.add_node("market", self._market)
        graph.add_node("news", self._news)
        graph.add_node("calculator", self._calculator)
        graph.add_node("aggregation", self._aggregation)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "financial_rag")
        graph.add_edge("financial_rag", "market")
        graph.add_edge("market", "news")
        graph.add_edge("news", "calculator")
        graph.add_edge("calculator", "aggregation")
        graph.add_edge("aggregation", END)
        return graph

    def _plan(self, state: AgentState):
        decision = self.planner.plan(state["query"])
        companies = extract_securities_from_query(state["query"]) or state.get("companies", [])
        period = self._period(state["query"])
        return {"intent": decision.intent, "required_tools": list(decision.tools), "companies": companies, "tickers": [item["ticker"] for item in companies], "report_periods": [period] if period else state.get("report_periods", []), "previous_query": state.get("query", ""), "last_intent": decision.intent, "executed_tools": [], "tool_results": [], "financial_result": "", "market_results": [], "news_results": [], "calculation_result": {}, "errors": [], "trace": {"plan": decision.to_dict()}}

    @staticmethod
    def _period(query: str) -> str | None:
        import re
        match = re.search(r"(20\d{2})(?:年)?(?:H1|上半年|半年度)", query, re.I)
        return f"{match.group(1)}H1" if match else None

    def _financial_rag(self, state: AgentState):
        if "financial_rag" not in state.get("required_tools", []): return {}
        result = self.rag_tool.run(state["query"])
        return {"financial_result": result.answer, "tool_results": [*state.get("tool_results", []), result.to_dict()], "executed_tools": [*state.get("executed_tools", []), "financial_rag"], "errors": [*state.get("errors", []), *([result.error] if result.error else [])]}

    def _market(self, state: AgentState):
        if "market_mcp" not in state.get("required_tools", []): return {}
        return self._mcp_many(state, "market", "get_market_snapshot", "market_mcp", "market_results")

    def _news(self, state: AgentState):
        if "news_mcp" not in state.get("required_tools", []): return {}
        return self._mcp_many(state, "news", "search_financial_news", "news_mcp", "news_results")

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
        request = self.planner.simple_calculation_request(state["query"])
        if not request: return {"errors": ["缺少可验证的计算输入。"]}
        result = self.calculator.run(**request)
        return {"calculation_result": result.to_dict(), "tool_results": [*state.get("tool_results", []), result.to_dict()], "executed_tools": [*state.get("executed_tools", []), "calculator"], "errors": [*state.get("errors", []), *([result.error] if result.error else [])]}

    def _aggregation(self, state):
        # Existing verified RAG/tool outputs are preserved; no second retrieval occurs.
        if state.get("financial_result"): answer = state["financial_result"]
        elif state.get("market_results"): answer = "\n".join(str(item.get("result")) for item in state["market_results"])
        elif state.get("news_results"): answer = "\n".join(str(item.get("result")) for item in state["news_results"])
        elif state.get("calculation_result"): answer = str(state["calculation_result"].get("result"))
        else: answer = "当前缺少足够上下文，无法可靠回答。"
        return {"final_answer": answer}
