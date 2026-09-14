"""Planning contracts, local validation and mode selection; no service startup."""
from __future__ import annotations

import ast
import copy
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal

from mcp_servers.providers.market_provider import extract_securities_from_query, normalize_security
from rag_qa.core.query_metadata import extract_query_metadata, METRIC_DEFINITIONS

from .planner import FinancialPlanner
from .schemas import PlannerDecision
from .tools.calculator_tool import CalculatorTool

TOOL_INTENTS = {"financial_rag": "financial_report_query", "market_mcp": "market_query",
                "news_mcp": "news_query", "calculator": "calculation_query"}
MAX_TEXT = 2000


@dataclass(frozen=True)
class PlannerSettings:
    mode: str = "rule"
    model: str = ""
    timeout: float = 30.0
    fallback_to_rule: bool = False
    max_calls: int = 8

    def __post_init__(self):
        if self.mode not in {"rule", "function_calling"}:
            raise ValueError("unknown planner mode")
        if not math.isfinite(self.timeout) or not 0 < self.timeout <= 120:
            raise ValueError("planner timeout must be in (0,120]")
        if not 1 <= self.max_calls <= 32:
            raise ValueError("planner max_calls must be in [1,32]")


def create_planner(*, mode=None, model=None, timeout=None, fallback_to_rule=None,
                   max_calls=None, client=None):
    from base.config import config
    settings = PlannerSettings(
        mode=config.AGENT_PLANNER_MODE if mode is None else mode,
        model=config.AGENT_PLANNER_MODEL if model is None else model,
        timeout=config.AGENT_PLANNER_TIMEOUT if timeout is None else timeout,
        fallback_to_rule=config.AGENT_PLANNER_FALLBACK_TO_RULE if fallback_to_rule is None else fallback_to_rule,
        max_calls=config.AGENT_PLANNER_MAX_CALLS if max_calls is None else max_calls,
    )
    if settings.mode == "rule":
        return FinancialPlanner()
    from .function_calling_planner import FunctionCallingPlanner
    return FunctionCallingPlanner(settings, client=client)


def planning_context(query, state):
    explicit = extract_securities_from_query(query)
    periods = list(extract_query_metadata(query).report_periods)
    return copy.deepcopy({
        "query": query, "explicit_companies": explicit, "explicit_periods": periods,
        "session_companies": state.get("companies", []),
        "session_periods": state.get("report_periods", []),
        "previous_query": state.get("last_query", "")[:MAX_TEXT],
        "preferences": state.get("long_term_preferences", {}),
        "preference_source": "explicit_user_preferences_only; not permission to add targets",
    })


def allowed_targets(context):
    return context["explicit_companies"] or context["session_companies"]


def tool_schemas():
    security = {"company_name": {"type": "string"}, "ticker": {"type": "string", "pattern": "^[0-9]{6}$"}}
    calculator = {"operation": {"type": "string", "enum": list(CalculatorTool._OPERATIONS)},
                  "expression": {"type": "string", "maxLength": 96}}
    calculator.update({key: {"type": "number"} for key in ("current", "previous", "numerator", "denominator", "left", "right")})
    definitions = [
        ("financial_rag", "查询已入库财报或金融定义。一次query可包含多个公司和期间；财报同比可由其内部验证计算。", {"query": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT}}, ["query"]),
        ("market_mcp", "查询一家公司的当前股票行情。多个公司分别调用。company_name或ticker至少一个。", security, []),
        ("news_mcp", "查询一家公司的财经新闻。多个目标分别调用；必须提供company_name或ticker。", {**security, "query": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT}, "max_results": {"type": "integer", "minimum": 1, "maximum": 10}}, ["query"]),
        ("calculator", "仅计算用户已经给出的数字。按operation传其所需参数；不可猜财报数据。growth_rate/absolute_change/percentage_point_change用current,previous；ratio用numerator,denominator；addition/subtraction/multiplication用left,right；expression用expression。", calculator, ["operation"]),
    ]
    tools = []
    for name, description, properties, required in definitions:
        params = {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
        if name in {"market_mcp", "news_mcp"}:
            params["anyOf"] = [{"required": ["company_name"]}, {"required": ["ticker"]}]
        tools.append({"type": "function", "function": {"name": name, "description": description, "parameters": params}})
    return tools


class PlanValidationError(ValueError):
    pass


def _text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT:
        raise PlanValidationError("invalid_text")
    if re.search(r"https?://|www\.", value, re.I):
        raise PlanValidationError("url_not_allowed")
    return value


def _metric_constraints(query):
    # Longest aliases first keep total revenue, R&D expense and investment distinct.
    aliases = dict(METRIC_DEFINITIONS)
    aliases.update({"total_revenue": ("营业总收入",), "profit": ("净利润",),
                    "industry_margin": ("行业毛利率",), "eps": ("EPS", "每股收益"),
                    "roe": ("ROE", "净资产收益率"), "operating_cash_flow": (*aliases["operating_cash_flow"], "经营活动现金流量净额")})
    remaining = query.lower()
    found = set()
    for alias, metric in sorted(((a.lower(), k) for k, values in aliases.items() for a in values), key=lambda x: -len(x[0])):
        if alias in remaining:
            found.add(metric)
            remaining = remaining.replace(alias, " ")
    return found


def _numbers(text):
    return {Decimal(value) for value in re.findall(r"(?<![A-Za-z0-9.])-?\d+(?:\.\d+)?(?![A-Za-z0-9.])", text)}


def validate_arguments(name, arguments, context):
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tool_schemas()}
    if name not in schemas:
        raise PlanValidationError("unknown_tool")
    if not isinstance(arguments, dict):
        raise PlanValidationError("arguments_not_object")
    schema = schemas[name]
    if set(arguments) - set(schema["properties"]):
        raise PlanValidationError("unknown_fields")
    if set(schema["required"]) - set(arguments):
        raise PlanValidationError("missing_fields")
    args = copy.deepcopy(arguments)
    targets = allowed_targets(context)
    allowed_codes = {item["ticker"] for item in targets}
    if name in {"market_mcp", "news_mcp"}:
        for key in ("company_name", "ticker"):
            if key in args: _text(args[key])
        by_name = normalize_security(company_name=args.get("company_name")) if "company_name" in args else None
        by_code = normalize_security(ticker=args.get("ticker")) if "ticker" in args else None
        if ("company_name" in args and not by_name) or ("ticker" in args and not by_code):
            raise PlanValidationError("unsupported_security")
        if by_name and by_code and by_name["ticker"] != by_code["ticker"]:
            raise PlanValidationError("security_conflict")
        security = by_code or by_name
        if not security or security["ticker"] not in allowed_codes:
            raise PlanValidationError("ungrounded_security")
        if name == "news_mcp":
            _text(args["query"])
            mentioned = {x["ticker"] for x in extract_securities_from_query(args["query"])}
            if mentioned - {security["ticker"]}:
                raise PlanValidationError("news_target_conflict")
            periods = set(extract_query_metadata(args["query"]).report_periods)
            if periods - set(context["explicit_periods"] or context["session_periods"]):
                raise PlanValidationError("ungrounded_period")
            limit = args.get("max_results", 5)
            if type(limit) is not int or not 1 <= limit <= 10:
                raise PlanValidationError("invalid_max_results")
    elif name == "financial_rag":
        text = _text(args["query"])
        actual = extract_query_metadata(text)
        if set(actual.company_codes) != allowed_codes:
            # Definitions without a company must stay company-free even in a session.
            definition = any(word in context["query"] for word in ("什么", "是啥", "区别", "定义")) and not context["explicit_companies"]
            if not (definition and not actual.company_codes):
                raise PlanValidationError("rag_targets_changed")
        required_periods = context["explicit_periods"] or context["session_periods"]
        if set(actual.report_periods) != set(required_periods):
            definition = any(word in context["query"] for word in ("什么", "是啥", "区别", "定义")) and not context["explicit_periods"]
            if not (definition and not actual.report_periods):
                raise PlanValidationError("rag_periods_changed")
        if not _metric_constraints(context["query"]).issubset(_metric_constraints(text)):
            raise PlanValidationError("rag_metrics_dropped")
        if _metric_constraints(text) - _metric_constraints(context["query"]):
            raise PlanValidationError("rag_metrics_added")
        number_sources = context["query"] + " " + context["previous_query"] + " " + " ".join(allowed_codes)
        if _numbers(text) - _numbers(number_sources):
            raise PlanValidationError("rag_ungrounded_numbers")
        for term in ("同比", "环比", "行业", "扣非"):
            if term in context["query"] and term not in text:
                raise PlanValidationError("rag_constraint_dropped")
    else:
        op = args["operation"]
        if not isinstance(op, str) or op not in CalculatorTool._OPERATIONS:
            raise PlanValidationError("unknown_operation")
        fields = set(CalculatorTool._OPERATIONS[op])
        if set(args) != fields | {"operation"}:
            raise PlanValidationError("operation_fields_mismatch")
        # Strip report periods/tickers so identifiers cannot become invented amounts.
        source = re.sub(r"20\d{2}(?:H1|FY|年)", " ", context["query"], flags=re.I)
        source = re.sub(r"(?<!\d)\d{6}(?!\d)", " ", source)
        available_numbers = _numbers(source)
        if op == "expression":
            if not isinstance(args["expression"], str):
                raise PlanValidationError("invalid_expression")
            expression = CalculatorTool.parse_safe_expression(args["expression"])
            if expression is None:
                raise PlanValidationError("unsafe_expression")
            tree = ast.parse(expression, mode="eval")
            operands = {Decimal(str(n.value)) for n in ast.walk(tree) if isinstance(n, ast.Constant)}
            if not operands <= {abs(n) for n in available_numbers}:
                raise PlanValidationError("ungrounded_operands")
        else:
            for key in fields:
                if type(args[key]) not in (float, int) or not math.isfinite(args[key]):
                    raise PlanValidationError("nonfinite_or_nonnumeric")
                if Decimal(str(args[key])) not in available_numbers:
                    raise PlanValidationError("ungrounded_operands")
        # Calculator is pure: preflight also rejects zero division before any plan executes.
        if not CalculatorTool().run(**args).success:
            raise PlanValidationError("invalid_calculation")
    return args


def decision_from_calls(calls, metadata):
    names = tuple(dict.fromkeys(call["tool_name"] for call in calls))
    intent = "composite_query" if len(names) > 1 else TOOL_INTENTS[names[0]]
    decision = FinancialPlanner._attach_skill(PlannerDecision(intent, names, "原生 Function Calling 计划已通过本地校验。"), metadata.pop("_query", ""))
    return PlannerDecision(decision.intent, decision.tools, decision.reason, skill=decision.skill,
                           planned_calls=tuple(calls), planning_status="ready", planner_metadata=metadata)


def rule_metadata():
    return {"requested_mode": "rule", "effective_mode": "rule", "fallback_used": False,
            "fallback_reason": None, "request_count": 0, "token_usage": None, "api_success": None}
