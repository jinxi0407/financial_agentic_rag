"""Deterministic planner for the small multi-tool Financial Agent MVP."""

import re

from .schemas import PlannerDecision


_MARKET_TERMS = ("股价", "行情", "涨停", "跌停", "成交量", "市值", "实时行情", "最新股价", "涨跌", "市场表现", "股票表现", "这个票")
_NEWS_TERMS = ("新闻", "资讯", "消息", "news", "舆情", "公告", "事件", "发生什么")
_FINANCIAL_REPORT_TERMS = (
    "财报", "年报", "年度报告", "半年报", "半年度", "季度报告", "营业收入", "营业总收入",
    "净利润", "归母", "毛利率", "研发", "现金流", "净息差", "不良贷款率",
    "拨备覆盖率", "资产负债率", "同比", "环比", "roe", "财务指标",
    "财务表现", "营收", "h1", "fy", "报告",
)
_DEFINITION_TERMS = ("流动比率", "速动比率", "市盈率", "roe", "归母净利润", "营业收入和净利润")
_GREETING_PATTERN = re.compile(r"^(?:hi|hello|你好|您好|嗨)[!！。,.\s]*$", re.I)
_IDENTITY_PATTERN = re.compile(r"^(?:你是什么|你能做什么)[？?！!。\s]*$")
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


class FinancialPlanner:
    """Route only frozen-corpus financial questions in the MVP."""

    def plan(self, query: str) -> PlannerDecision:
        return self._attach_skill(self._plan(query), query)

    def _plan(self, query: str) -> PlannerDecision:
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
        has_market = any(term in lowered for term in _MARKET_TERMS)
        has_news = any(term in lowered for term in _NEWS_TERMS)
        has_report = any(term in lowered for term in _FINANCIAL_REPORT_TERMS)
        if any(term in lowered for term in _DEFINITION_TERMS):
            return PlannerDecision(
                intent="financial_report_query",
                tools=("financial_rag",),
                reason="用户询问金融定义，交由 Financial RAG 的 FAQ fast path 处理。",
            )
        if (has_market or has_news) and has_report:
            tools = ["financial_rag"]
            if has_market:
                tools.append("market_mcp")
            if has_news:
                tools.append("news_mcp")
            return PlannerDecision(
                intent="composite_query",
                tools=tuple(tools),
                reason="查询同时包含历史财报与实时市场或新闻需求。",
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
        if self.simple_calculation_request(normalized) is not None:
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
        return None
