"""Small deterministic calculator for Agent requests; no LLM or RAG access."""

from __future__ import annotations

import ast
import math
import re
import time

from agent.schemas import CalculatorToolResult


class CalculatorTool:
    tool_name = "calculator"
    _MAX_EXPRESSION_LENGTH = 96
    _MAX_EXPRESSION_OPERATORS = 16
    _EXPRESSION_SUFFIX_PATTERN = re.compile(
        r"\s*(?:(?:=+\s*)?(?:(?:是|等于)\s*)?(?:多少|算一下)|=+\s*[?？]|[?？])\s*[！!。]*\s*$"
    )
    _OPERATIONS = {
        "absolute_change": ("current", "previous"),
        "growth_rate": ("current", "previous"),
        "percentage_point_change": ("current", "previous"),
        "ratio": ("numerator", "denominator"),
        "addition": ("left", "right"),
        "subtraction": ("left", "right"),
        "multiplication": ("left", "right"),
        "expression": ("expression",),
    }

    @classmethod
    def parse_safe_expression(cls, query: str) -> str | None:
        """Return a normalized, whitelisted arithmetic expression or ``None``."""
        expression = query.strip()
        # Strip only a question suffix.  An equals sign elsewhere remains in
        # the expression and is rejected by the character and AST allowlists.
        expression = cls._EXPRESSION_SUFFIX_PATTERN.sub("", expression)
        expression = expression.replace("×", "*").replace("÷", "/")
        expression = expression.replace("除以", "/").replace("乘以", "*")
        expression = re.sub(r"(?<=[\d)])\s*加\s*(?=[\d(+\-])", "+", expression)
        expression = re.sub(r"(?<=[\d)])\s*减\s*(?=[\d(+\-])", "-", expression)
        if not expression or len(expression) > cls._MAX_EXPRESSION_LENGTH:
            return None
        if not re.fullmatch(r"[\d\s+\-*/().]+", expression):
            return None
        if sum(expression.count(operator) for operator in "+-*/") > cls._MAX_EXPRESSION_OPERATORS:
            return None
        try:
            tree = ast.parse(expression, mode="eval")
            cls._validate_expression_node(tree)
        except (SyntaxError, ValueError):
            return None
        return expression

    @classmethod
    def is_compound_expression(cls, expression: str) -> bool:
        return "(" in expression or ")" in expression or sum(expression.count(operator) for operator in "+-*/") > 1

    @classmethod
    def _validate_expression_node(cls, node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            cls._validate_expression_node(node.body)
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            cls._validate_expression_node(node.left)
            cls._validate_expression_node(node.right)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            cls._validate_expression_node(node.operand)
            return
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if not math.isfinite(float(node.value)):
                raise ValueError("non-finite number")
            return
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    @classmethod
    def _evaluate_expression(cls, expression: str) -> float:
        tree = ast.parse(expression, mode="eval")
        cls._validate_expression_node(tree)

        def evaluate(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant):
                return float(node.value)
            if isinstance(node, ast.UnaryOp):
                value = evaluate(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right

        result = evaluate(tree)
        if not math.isfinite(result):
            raise ValueError("non-finite result")
        return result

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
            if operation == "expression":
                expression = self.parse_safe_expression(str(inputs.get("expression", "")))
                if expression is None:
                    raise ValueError("unsafe arithmetic expression")
                result = self._evaluate_expression(expression)
                return CalculatorToolResult(
                    tool_name=self.tool_name,
                    operation=operation,
                    inputs={"expression": expression},
                    result=result,
                    success=True,
                    error=None,
                    latency=time.perf_counter() - started_at,
                )
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
