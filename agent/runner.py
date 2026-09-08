"""One-step Agent runner: plan, execute a tool, return a stable contract."""

from time import perf_counter

from mcp_servers.providers.market_provider import extract_security_from_query

from .mcp_client import FinancialMCPClient
from .planner import FinancialPlanner
from .schemas import AgentResponse
from .tools.calculator_tool import CalculatorTool
from .tools.financial_rag_tool import FinancialRAGTool


class FinancialAgentRunner:
    def __init__(self, planner=None, financial_rag_tool=None, calculator_tool=None, mcp_client=None):
        self.planner = planner or FinancialPlanner()
        self.financial_rag_tool = financial_rag_tool or FinancialRAGTool()
        self.calculator_tool = calculator_tool or CalculatorTool()
        self.mcp_client = mcp_client or FinancialMCPClient()

    def run(self, query: str) -> AgentResponse:
        started_at = perf_counter()
        decision = self.planner.plan(query)
        if decision.status == "planned_but_tool_unavailable":
            return AgentResponse(
                query=query,
                answer="该需求已被识别，但当前 Agent MVP 尚未提供对应的实时行情或新闻工具。",
                success=False,
                planner=decision,
                total_latency=perf_counter() - started_at,
                error="planned_but_tool_unavailable",
            )
        if decision.intent == "unsupported":
            return AgentResponse(
                query=query,
                answer="当前 Agent MVP 仅支持已入库财报和金融知识库问题，不支持实时行情或新闻。",
                success=False,
                planner=decision,
                total_latency=perf_counter() - started_at,
                error="unsupported",
            )

        if decision.intent == "calculation_query":
            request = self.planner.simple_calculation_request(query)
            if request is None:
                return AgentResponse(
                    query=query,
                    answer="当前 Calculator MVP 仅支持明确的结构化计算或“从 X 增长到 Y，增长率是多少”形式。",
                    success=False,
                    planner=decision,
                    total_latency=perf_counter() - started_at,
                    error="calculation_input_required",
                )
            result = self.calculator_tool.run(**request)
            return AgentResponse(
                query=query,
                answer=str(result.result) if result.success else "当前 Calculator 工具无法完成计算。",
                success=result.success,
                planner=decision,
                tool_results=(result,),
                executed_tools=(result.tool_name,),
                total_latency=perf_counter() - started_at,
                error=result.error,
            )

        if decision.intent == "market_query":
            security = extract_security_from_query(query)
            arguments = {
                "company_name": security["company_name"] if security else None,
                "ticker": security["ticker"] if security else None,
            }
            result = self.mcp_client.call_tool("market", "get_market_snapshot", arguments)
            return AgentResponse(
                query=query,
                answer=self._format_market_answer(result.result) if result.success else "当前 Market MCP 工具无法获取行情。",
                success=result.success,
                planner=decision,
                tool_results=(result,),
                executed_tools=("market_mcp",),
                total_latency=perf_counter() - started_at,
                error=result.error,
            )

        if decision.intent == "news_query":
            security = extract_security_from_query(query)
            arguments = {
                "query": query,
                "company_name": security["company_name"] if security else None,
                "ticker": security["ticker"] if security else None,
                "max_results": 5,
            }
            result = self.mcp_client.call_tool("news", "search_financial_news", arguments)
            return AgentResponse(
                query=query,
                answer=self._format_news_answer(result.result) if result.success else "当前 News MCP 工具无法获取新闻。",
                success=result.success,
                planner=decision,
                tool_results=(result,),
                executed_tools=("news_mcp",),
                total_latency=perf_counter() - started_at,
                error=result.error,
            )

        result = self.financial_rag_tool.run(query)
        return AgentResponse(
            query=query,
            answer=result.answer,
            success=result.success,
            planner=decision,
            tool_results=(result,),
            executed_tools=(result.tool_name,),
            total_latency=perf_counter() - started_at,
            error=result.error,
        )

    @staticmethod
    def _format_market_answer(payload: dict | None) -> str:
        if not payload:
            return "当前未返回行情数据。"
        price = payload.get("price")
        change_percent = payload.get("change_percent")
        time_value = payload.get("market_time") or "数据源未提供时间"
        return f"{payload.get('company_name')}（{payload.get('ticker')}）最新价为 {price} {payload.get('currency', '')}，涨跌幅 {change_percent}%，数据时间：{time_value}。"

    @staticmethod
    def _format_news_answer(payload: dict | None) -> str:
        if not payload:
            return "当前未返回新闻数据。"
        results = payload.get("results") or []
        if not results:
            return "当前未检索到相关新闻。"
        return "\n".join(
            f"{index}. {item.get('title')}（{item.get('source') or '未知来源'}）"
            for index, item in enumerate(results, start=1)
        )
