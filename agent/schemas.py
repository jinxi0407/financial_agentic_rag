"""Small, serializable contracts for the Agent MVP."""

from dataclasses import asdict, dataclass, field
from typing import Literal


PlannerIntent = Literal["financial_report_query", "unsupported"]


@dataclass(frozen=True)
class PlannerDecision:
    intent: PlannerIntent
    tools: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tools"] = list(self.tools)
        return payload


@dataclass(frozen=True)
class FinancialRAGToolResult:
    tool_name: str
    query: str
    answer: str
    success: bool
    error: str | None
    latency: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AgentResponse:
    query: str
    answer: str
    success: bool
    planner: PlannerDecision
    tool_results: tuple[FinancialRAGToolResult, ...] = field(default_factory=tuple)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "success": self.success,
            "planner": self.planner.to_dict(),
            "tool_results": [result.to_dict() for result in self.tool_results],
            "error": self.error,
        }
