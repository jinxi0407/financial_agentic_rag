"""Deterministic planner for the first Financial Agent MVP."""

from .schemas import PlannerDecision


_REALTIME_OR_NEWS_TERMS = (
    "今天", "今日", "当前", "实时", "最新", "股价", "行情", "涨停", "跌停",
    "新闻", "资讯", "消息", "news",
)
_FINANCIAL_REPORT_TERMS = (
    "财报", "年报", "半年报", "半年度", "季度报告", "营业收入", "营业总收入",
    "净利润", "归母", "毛利率", "研发", "现金流", "净息差", "不良贷款率",
    "拨备覆盖率", "资产负债率", "同比", "环比", "roe", "财务指标",
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
        if any(term in lowered for term in _REALTIME_OR_NEWS_TERMS):
            return PlannerDecision(
                intent="unsupported",
                tools=(),
                reason="当前 Agent MVP 不提供实时行情或新闻查询。",
            )
        if any(term in lowered for term in _FINANCIAL_REPORT_TERMS):
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
