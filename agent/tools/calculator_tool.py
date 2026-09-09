"""Small deterministic calculator for Agent requests; no LLM or RAG access."""

from __future__ import annotations

import time

from agent.schemas import CalculatorToolResult


class CalculatorTool:
    tool_name = "calculator"
    _OPERATIONS = {
        "absolute_change": ("current", "previous"),
        "growth_rate": ("current", "previous"),
        "percentage_point_change": ("current", "previous"),
        "ratio": ("numerator", "denominator"),
        "addition": ("left", "right"),
        "subtraction": ("left", "right"),
        "multiplication": ("left", "right"),
    }

    @staticmethod
    def _number(value, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        return float(value)

    def run(self, operation: str, **inputs) -> CalculatorToolResult:
        started_at = time.perf_counter()
        try:
            if operation not in self._OPERATIONS:
                raise ValueError(f"unsupported operation: {operation}")
            required = self._OPERATIONS[operation]
            missing = [name for name in required if name not in inputs or inputs[name] is None]
            if missing:
                raise ValueError(f"missing required inputs: {', '.join(missing)}")
            values = {name: self._number(inputs[name], name) for name in required}
            if operation == "absolute_change":
                result = values["current"] - values["previous"]
            elif operation == "growth_rate":
                if values["previous"] == 0:
                    raise ZeroDivisionError("previous must not be zero for growth_rate")
                result = (values["current"] - values["previous"]) / values["previous"] * 100
            elif operation == "percentage_point_change":
                result = values["current"] - values["previous"]
            elif operation == "ratio":
                if values["denominator"] == 0:
                    raise ZeroDivisionError("denominator must not be zero for ratio")
                result = values["numerator"] / values["denominator"]
            elif operation == "addition":
                result = values["left"] + values["right"]
            elif operation == "subtraction":
                result = values["left"] - values["right"]
            else:
                result = values["left"] * values["right"]
            return CalculatorToolResult(
                tool_name=self.tool_name,
                operation=operation,
                inputs=values,
                result=result,
                success=True,
                error=None,
                latency=time.perf_counter() - started_at,
            )
        except Exception as exc:
            return CalculatorToolResult(
                tool_name=self.tool_name,
                operation=operation,
                inputs=dict(inputs),
                result=None,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                latency=time.perf_counter() - started_at,
            )
