"""Deterministic planner for the small multi-tool Financial Agent MVP."""

import re

from .schemas import PlannerDecision


_MARKET_TERMS = ("股价", "行情", "涨停", "跌停", "成交量", "市值", "实时行情")
_NEWS_TERMS = ("新闻", "资讯", "消息", "news")
_FINANCIAL_REPORT_TERMS = (
    "财报", "年报", "半年报", "半年度", "季度报告", "营业收入", "营业总收入",
    "净利润", "归母", "毛利率", "研发", "现金流", "净息差", "不良贷款率",
    "拨备覆盖率", "资产负债率", "同比", "环比", "roe", "财务指标",
)
_CALCULATION_TERMS = ("增长率", "增长", "增加", "减少", "差值", "比率", "百分点")
_SIMPLE_GROWTH_PATTERN = re.compile(
    r"从\s*(?P<previous>-?\d+(?:\.\d+)?)\s*(?:增长(?:到)?|增加(?:到)?|变为|到)\s*"
    r"(?P<current>-?\d+(?:\.\d+)?)(?=[，,。！？?\s]|$)"
)


class FinancialPlanner:
    """Route only frozen-corpus financial questions in the MVP."""

    def plan(self, query: str) -> PlannerDecision:
        normalized = query.strip()
        lowered = normalized.lower()
        if not normalized:
            return PlannerDecision(
                intent="unsupported",
                tools=(),
                reason="查询为空，无法选择金融财报工具。",
            )
        has_market = any(term in lowered for term in _MARKET_TERMS)
        has_news = any(term in lowered for term in _NEWS_TERMS)
        has_report = any(term in lowered for term in _FINANCIAL_REPORT_TERMS)
        if (has_market or has_news) and has_report:
            return PlannerDecision(
                intent="composite_query",
                tools=("financial_rag", "market_data" if has_market else "news_search"),
                reason="查询同时包含财报信息和未实现的实时市场或新闻需求。",
                status="planned_but_tool_unavailable",
            )
        if has_market:
            return PlannerDecision(
                intent="market_query",
                tools=("market_data",),
                reason="当前 Agent MVP 尚未接入实时行情工具。",
                status="planned_but_tool_unavailable",
            )
        if has_news:
            return PlannerDecision(
                intent="news_query",
                tools=("news_search",),
                reason="当前 Agent MVP 尚未接入新闻检索工具。",
                status="planned_but_tool_unavailable",
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
        if any(term in lowered for term in _CALCULATION_TERMS):
            return PlannerDecision(
                intent="calculation_query",
                tools=("calculator",),
                reason="用户请求确定性数值计算。",
            )
        return PlannerDecision(
            intent="unsupported",
            tools=(),
            reason="当前 Agent MVP 仅支持已入库财报和金融知识库问题。",
        )

    @staticmethod
    def simple_calculation_request(query: str) -> dict | None:
        """Parse only the unambiguous MVP growth-rate sentence form."""
        if "增长率" not in query:
            return None
        match = _SIMPLE_GROWTH_PATTERN.search(query)
        if not match:
            return None
        return {
            "operation": "growth_rate",
            "current": float(match.group("current")),
            "previous": float(match.group("previous")),
        }
