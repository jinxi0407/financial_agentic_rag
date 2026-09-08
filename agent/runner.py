"""One-step Agent runner: plan, call the frozen RAG tool, return a contract."""

from .planner import FinancialPlanner
from .schemas import AgentResponse
from .tools.calculator_tool import CalculatorTool
from .tools.financial_rag_tool import FinancialRAGTool


class FinancialAgentRunner:
    def __init__(self, planner=None, financial_rag_tool=None, calculator_tool=None):
        self.planner = planner or FinancialPlanner()
        self.financial_rag_tool = financial_rag_tool or FinancialRAGTool()
        self.calculator_tool = calculator_tool or CalculatorTool()

    def run(self, query: str) -> AgentResponse:
        decision = self.planner.plan(query)
        if decision.status == "planned_but_tool_unavailable":
            return AgentResponse(
                query=query,
                answer="该需求已被识别，但当前 Agent MVP 尚未提供对应的实时行情或新闻工具。",
                success=False,
                planner=decision,
                error="planned_but_tool_unavailable",
            )
        if decision.intent == "unsupported":
            return AgentResponse(
                query=query,
                answer="当前 Agent MVP 仅支持已入库财报和金融知识库问题，不支持实时行情或新闻。",
                success=False,
                planner=decision,
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
                    error="calculation_input_required",
                )
            result = self.calculator_tool.run(**request)
            return AgentResponse(
                query=query,
                answer=str(result.result) if result.success else "当前 Calculator 工具无法完成计算。",
                success=result.success,
                planner=decision,
                tool_results=(result,),
                error=result.error,
            )

        result = self.financial_rag_tool.run(query)
        return AgentResponse(
            query=query,
            answer=result.answer,
            success=result.success,
            planner=decision,
            tool_results=(result,),
            error=result.error,
        )
