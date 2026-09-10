"""Deterministic planner for the small multi-tool Financial Agent MVP."""

import re

from .schemas import PlannerDecision
from .tools.calculator_tool import CalculatorTool


_MARKET_TERMS = ("股价", "行情", "涨停", "跌停", "成交量", "市值", "实时行情", "最新股价", "涨跌", "市场表现", "股票表现", "这个票")
_NEWS_TERMS = ("新闻", "资讯", "消息", "news", "舆情", "公告", "事件", "发生什么")
_FINANCIAL_REPORT_TERMS = (
    "财报", "年报", "年度报告", "半年报", "半年度", "季度报告", "营业收入", "营业总收入",
    "净利润", "归母", "毛利率", "研发", "现金流", "净息差", "不良贷款率",
    "拨备覆盖率", "资产负债率", "同比", "环比", "roe", "财务指标",
    "财务表现", "经营表现", "经营情况", "经营状况", "整体表现", "业绩表现",
    "营收", "h1", "fy", "报告",
)
_DEFINITION_TERMS = ("流动比率", "速动比率", "市盈率", "roe", "归母净利润", "营业收入和净利润")
_GREETING_PATTERN = re.compile(r"^(?:hi|hello|你好|您好|嗨)[!！。,.\s]*$", re.I)
_IDENTITY_PATTERN = re.compile(r"^(?:(?:你好|您好)[，,！!。\s]*)?(?:你是什么|你能做什么)[？?！!。\s]*$")
_CONTEXTUAL_MARKET_FOLLOW_UP_PATTERN = re.compile(
    r"(?:他|它|这家公司|这个票|那家公司).*?(?:股票|股价|行情|市场表现|涨跌)"
)
_COMPANY_PRONOUN_PATTERN = re.compile(r"(?:[他它](?:的|现在|最近)?|这家公司|该公司|这个票|那家公司)")
_STOCK_REQUEST_PATTERN = re.compile(
    r"(?:股票\s*(?:怎么样|如何|看看|行情)?|今天的股票|当前股票)"
)
_AMBIGUOUS_STOCK_FINANCIAL_PATTERN = re.compile(r"股票[^。！？!?，,；;]*财务|财务[^。！？!?，,；;]*股票")
_CALCULATION_CLAUSE_SPLIT_PATTERN = re.compile(r"[，,；;。]|(?:然后|还有|并且)")
_REPORT_SUBTASK_PATTERN = re.compile(r"(?:财报|报告|年报|半年报|20\d{2}(?:H1|FY|年))", re.I)
_SIMPLE_GROWTH_PATTERN = re.compile(
    r"从\s*(?P<previous>-?\d+(?:\.\d+)?)\s*(?:增长(?:到)?|增加(?:到)?|变为|到)\s*"
    r"(?P<current>-?\d+(?:\.\d+)?)(?=[，,。！？?\s]|$)"
)
_COMPARE_GROWTH_PATTERN = re.compile(
    r"(?P<current>-?\d+(?:\.\d+)?)\s*比\s*(?P<previous>-?\d+(?:\.\d+)?)\s*增长(?:了|多少)?(?:百分比|%)?"
)
_ABSOLUTE_CHANGE_PATTERN = re.compile(
    r"(?:从\s*)?(?P<previous>-?\d+(?:\.\d+)?)\s*(?:(?:增加|减少|上升|下降)(?:到|至)|(?:到|至))\s*(?P<current>-?\d+(?:\.\d+)?)(?:\s*(?:增加|减少|上升|下降))?"
)
_RATIO_PATTERN = re.compile(r"(?P<numerator>-?\d+(?:\.\d+)?)\s*(?:除以|/)\s*(?P<denominator>-?\d+(?:\.\d+)?)")
_PERCENTAGE_POINT_PATTERN = re.compile(
    r"(?:从\s*)?(?P<previous>-?\d+(?:\.\d+)?)%\s*(?:降到|下降至|到|至|升到|上升至)\s*(?P<current>-?\d+(?:\.\d+)?)%.*百分点"
)
_SIMPLE_ARITHMETIC_PATTERN = re.compile(
    r"^\s*(?P<left>-?\d+(?:\.\d+)?)\s*(?P<operator>[-+*/×÷]|加|减)\s*"
    r"(?P<right>-?\d+(?:\.\d+)?)\s*(?:(?:是|等于)\s*)?(?:多少)?\s*[？?！!。.\s]*$"
)


class FinancialPlanner:
    """Route only frozen-corpus financial questions in the MVP."""

    def plan(self, query: str, *, has_company_context: bool = False) -> PlannerDecision:
        return self._attach_skill(self._plan(query, has_company_context=has_company_context), query)

    def _plan(self, query: str, *, has_company_context: bool = False) -> PlannerDecision:
        normalized = query.strip()
        lowered = normalized.lower()
        if not normalized:
            return PlannerDecision(
                intent="unsupported",
                tools=(),
                reason="查询为空，无法选择金融财报工具。",
            )
        if _GREETING_PATTERN.match(normalized) or _IDENTITY_PATTERN.match(normalized):
            return PlannerDecision(
                intent="greeting",
                tools=(),
                reason="用户正在问候或询问 Agent 能力范围。",
            )
        has_market = self._is_market_request(normalized, lowered, has_company_context)
        has_news = any(term in lowered for term in _NEWS_TERMS)
        has_report = any(term in lowered for term in _FINANCIAL_REPORT_TERMS)
        calculation_request = self.simple_calculation_request(normalized)
        if calculation_request is not None and has_report:
            has_report = self._has_independent_report_subtask(normalized)
        if any(term in lowered for term in _DEFINITION_TERMS):
            return PlannerDecision(
                intent="financial_report_query",
                tools=("financial_rag",),
                reason="用户询问金融定义，交由 Financial RAG 的 FAQ fast path 处理。",
            )
        requested_tools = []
        if has_report:
            requested_tools.append("financial_rag")
        if has_market:
            requested_tools.append("market_mcp")
        if has_news:
            requested_tools.append("news_mcp")
        if calculation_request is not None:
            requested_tools.append("calculator")
        if len(requested_tools) > 1:
            return PlannerDecision(
                intent="composite_query",
                tools=tuple(requested_tools),
                reason="查询包含多个可独立执行的财报、实时信息或确定性计算子任务。",
            )
        if has_market:
            return PlannerDecision(
                intent="market_query",
                tools=("market_mcp",),
                reason="用户请求实时市场行情，将调用 Market MCP 工具。",
            )
        if has_news:
            return PlannerDecision(
                intent="news_query",
                tools=("news_mcp",),
                reason="用户请求近期新闻，将调用 News MCP 工具。",
            )
        if calculation_request is not None:
            return PlannerDecision(
                intent="calculation_query",
                tools=("calculator",),
                reason="用户请求确定性数值计算。",
            )
        if has_report:
            return PlannerDecision(
                intent="financial_report_query",
                tools=("financial_rag",),
                reason="用户询问冻结财报语料中的金融指标或报告信息。",
            )
        return PlannerDecision(
            intent="unsupported",
            tools=(),
            reason="当前 Agent MVP 仅支持已入库财报和金融知识库问题。",
        )

    @staticmethod
    def is_market_pronoun_follow_up(query: str) -> bool:
        return bool(_CONTEXTUAL_MARKET_FOLLOW_UP_PATTERN.search(query.strip()))

    @staticmethod
    def _is_market_request(query: str, lowered: str, has_company_context: bool) -> bool:
        """Recognize a stock request only when a company is available for bare 股票."""
        if any(term in lowered for term in _MARKET_TERMS):
            return True
        if not has_company_context or not _STOCK_REQUEST_PATTERN.search(query):
            return False
        # "股票相关财务资产" is not a quote request.  A bare 股票 token must not
        # override an otherwise ambiguous financial phrase.
        return not bool(_AMBIGUOUS_STOCK_FINANCIAL_PATTERN.search(query))

    @staticmethod
    def _has_independent_report_subtask(query: str) -> bool:
        """Require an explicit report clause before pairing Calculator with RAG."""
        clauses = [clause.strip() for clause in _CALCULATION_CLAUSE_SPLIT_PATTERN.split(query) if clause.strip()]
        return len(clauses) > 1 and any(_REPORT_SUBTASK_PATTERN.search(clause) for clause in clauses)

    @staticmethod
    def is_company_pronoun_query(query: str) -> bool:
        return bool(_COMPANY_PRONOUN_PATTERN.search(query.strip()))

    @staticmethod
    def missing_company_context_decision() -> PlannerDecision:
        return PlannerDecision(
            intent="unsupported",
            tools=(),
            reason="该请求依赖公司上下文，但当前会话未提供可解析的公司。",
        )

    @staticmethod
    def _attach_skill(decision: PlannerDecision, query: str) -> PlannerDecision:
        lowered = query.lower()
        if decision.intent == "unsupported":
            return decision
        if "比较" in query or "相比" in query:
            skill = "company_comparison"
        elif decision.intent in {"market_query", "news_query"} or any(
            term in lowered for term in (*_MARKET_TERMS, *_NEWS_TERMS)
        ):
            skill = "market_intelligence"
        else:
            skill = "financial_report_analysis"
        return PlannerDecision(
            intent=decision.intent,
            tools=decision.tools,
            reason=decision.reason,
            status=decision.status,
            skill=skill,
        )

    @staticmethod
    def simple_calculation_request(query: str) -> dict | None:
        """Parse only unambiguous, self-contained calculator requests."""
        expression = CalculatorTool.parse_safe_expression(query)
        expression_from_clause = False
        if expression is None:
            # Composite requests may contain one complete arithmetic clause plus
            # another tool request.  Each candidate is still parsed by the
            # existing AST whitelist; we never evaluate arbitrary substrings.
            for clause in _CALCULATION_CLAUSE_SPLIT_PATTERN.split(query):
                expression = CalculatorTool.parse_safe_expression(clause)
                if expression is not None:
                    expression_from_clause = True
                    break
        if expression and (
            CalculatorTool.is_compound_expression(expression)
            or expression_from_clause
            or "=" in query
        ):
            return {"operation": "expression", "expression": expression}
        match = _PERCENTAGE_POINT_PATTERN.search(query)
        if match:
            return {
                "operation": "percentage_point_change",
                "current": float(match.group("current")),
                "previous": float(match.group("previous")),
            }
        match = _SIMPLE_GROWTH_PATTERN.search(query)
        if not match:
            match = _COMPARE_GROWTH_PATTERN.search(query)
        if match:
            return {
                "operation": "growth_rate",
                "current": float(match.group("current")),
                "previous": float(match.group("previous")),
            }
        match = _ABSOLUTE_CHANGE_PATTERN.search(query)
        if match:
            return {
                "operation": "absolute_change",
                "current": float(match.group("current")),
                "previous": float(match.group("previous")),
            }
        match = _RATIO_PATTERN.search(query)
        if match:
            return {
                "operation": "ratio",
                "numerator": float(match.group("numerator")),
                "denominator": float(match.group("denominator")),
            }
        match = _SIMPLE_ARITHMETIC_PATTERN.fullmatch(query)
        if match:
            operation = {
                "+": "addition", "加": "addition",
                "-": "subtraction", "减": "subtraction",
                "*": "multiplication", "×": "multiplication",
                "/": "ratio", "÷": "ratio",
            }[match.group("operator")]
            left, right = float(match.group("left")), float(match.group("right"))
            if operation == "ratio":
                return {"operation": operation, "numerator": left, "denominator": right}
            return {"operation": operation, "left": left, "right": right}
        return None
