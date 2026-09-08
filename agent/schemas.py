"""Small, serializable contracts for the Agent MVP."""

from dataclasses import asdict, dataclass, field
from typing import Literal


PlannerIntent = Literal[
    "financial_report_query",
    "calculation_query",
    "market_query",
    "news_query",
    "composite_query",
    "unsupported",
]
PlannerStatus = Literal["ready", "planned_but_tool_unavailable"]


@dataclass(frozen=True)
class PlannerDecision:
    intent: PlannerIntent
    tools: tuple[str, ...]
    reason: str
    status: PlannerStatus = "ready"

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
class CalculatorToolResult:
    tool_name: str
    operation: str
    inputs: dict
    result: float | None
    success: bool
    error: str | None
    latency: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MCPToolResult:
    tool_name: str
    server_name: str
    inputs: dict
    result: dict | None
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
    tool_results: tuple[FinancialRAGToolResult | CalculatorToolResult | MCPToolResult, ...] = field(default_factory=tuple)
    executed_tools: tuple[str, ...] = field(default_factory=tuple)
    total_latency: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "success": self.success,
            "planner": self.planner.to_dict(),
            "tool_results": [result.to_dict() for result in self.tool_results],
            "executed_tools": list(self.executed_tools),
            "total_latency": self.total_latency,
            "error": self.error,
        }
