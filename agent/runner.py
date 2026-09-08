"""One-step Agent runner: plan, call the frozen RAG tool, return a contract."""

from .planner import FinancialPlanner
from .schemas import AgentResponse
from .tools.financial_rag_tool import FinancialRAGTool


class FinancialAgentRunner:
    def __init__(self, planner=None, financial_rag_tool=None):
        self.planner = planner or FinancialPlanner()
        self.financial_rag_tool = financial_rag_tool or FinancialRAGTool()

    def run(self, query: str) -> AgentResponse:
        decision = self.planner.plan(query)
        if decision.intent == "unsupported":
            return AgentResponse(
                query=query,
                answer="当前 Agent MVP 仅支持已入库财报和金融知识库问题，不支持实时行情或新闻。",
                success=False,
                planner=decision,
                error="unsupported",
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
