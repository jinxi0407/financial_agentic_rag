"""Conservative, evidence-backed arithmetic for financial comparison questions."""

import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal

from rag_qa.core.financial_evidence import StructuredFinancialEvidence
from rag_qa.core.query_metadata import METRIC_DEFINITIONS


_NUMBER_PATTERN = re.compile(
    r"(?P<value>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?P<percent>\s*%)?"
)
_UNIT_PATTERN = re.compile(r"单位\s*[：:]\s*(?P<unit>人民币元|元|万元|亿元|千元|百万元)")
_AMOUNT_MULTIPLIERS = {
    "元": 1,
    "人民币元": 1,
    "千元": 1_000,
    "万元": 10_000,
    "百万元": 1_000_000,
    "亿元": 100_000_000,
}
_RATE_METRICS = {"gross_margin", "net_interest_margin", "npl_ratio", "provision_coverage"}
_METRIC_LABELS = {
    "revenue": "营业收入",
    "net_profit": "归母净利润",
    "operating_cash_flow": "经营活动产生的现金流量净额",
    "research_investment": "研发投入",
    "research_expense": "研发费用",
    "gross_margin": "毛利率",
    "net_interest_margin": "净息差",
    "npl_ratio": "不良贷款率",
    "provision_coverage": "拨备覆盖率",
}


@dataclass(frozen=True)
class EvidenceValue:
    metric: str
    report_period: str
    value: float
    unit: str | None
    is_rate: bool


def metric_aliases(metric):
    return METRIC_DEFINITIONS.get(metric, ())


def document_mentions_metric(document, metric):
    text = document.page_content
    for alias in metric_aliases(metric):
        start = 0
        while True:
            index = text.find(alias, start)
            if index < 0:
                break
            # A sector/industry margin is not a company-wide gross margin.
            if metric == "gross_margin" and "行业" in text[max(0, index - 12):index + len(alias) + 12]:
                start = index + len(alias)
                continue
            return True
    return False


def _unit_near(text, index):
    window = text[max(0, index - 300):index]
    matches = list(_UNIT_PATTERN.finditer(window))
    return matches[-1].group("unit") if matches else None


def _candidate_numbers(text, index, is_rate):
    # A parent chunk often includes an "附注 43" marker between the metric
    # label and its values.  Amounts below 1,000 without separators are not
    # reliable financial values here, so skip those marker-like tokens.
    values = []
    for match in _NUMBER_PATTERN.finditer(text[index:index + 420]):
        raw = match.group("value")
        value = float(raw.replace(",", ""))
        has_percent = bool(match.group("percent"))
        if is_rate:
            if has_percent:
                values.append(value)
        elif "," in raw or abs(value) >= 1_000:
            values.append(value)
        if len(values) == 2:
            break
    return values


def extract_document_value(document, metric):
    """Extract the current-period value for one explicitly named metric.

    This intentionally returns ``None`` instead of guessing from a table when
    a unit, label, or amount cannot be established confidently.
    """
    report_period = document.metadata.get("report_period")
    if not report_period:
        return None
    text = document.page_content
    is_rate = metric in _RATE_METRICS
    aliases = sorted(metric_aliases(metric), key=len, reverse=True)
    for alias in aliases:
        index = text.find(alias)
        if index < 0:
            continue
        values = _candidate_numbers(text, index + len(alias), is_rate)
        if not values:
            continue
        unit = None if is_rate else _unit_near(text, index)
        if not is_rate and unit is None:
            continue
        return EvidenceValue(
            metric=metric,
            report_period=report_period,
            value=values[0],
            unit=unit,
            is_rate=is_rate,
        )
    return None


def _format_number(value):
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _verified_calculation_lines(query_metadata, evidence_values):
    """Return derived lines and their metrics from verified structured evidence only."""
    if not query_metadata or not query_metadata.requires_deterministic_calculation():
        return ()
    if len(query_metadata.company_codes) != 1 or len(query_metadata.report_periods) != 2:
        return ()

    base_period, current_period = sorted(query_metadata.report_periods)
    if base_period[4:] != current_period[4:]:
        # Annual and interim reports are deliberately never used for a derived
        # growth rate or delta, even if both raw numbers happen to be present.
        return ()

    values_by_cell = defaultdict(list)
    for value in evidence_values:
        if not isinstance(value, StructuredFinancialEvidence):
            continue
        values_by_cell[(value.company_code, value.report_period, value.metric)].append(value)

    company_code = query_metadata.company_codes[0]
    lines = []
    for metric in query_metadata.requested_metrics:
        current_values = values_by_cell[(company_code, current_period, metric)]
        base_values = values_by_cell[(company_code, base_period, metric)]
        if len(current_values) != 1 or len(base_values) != 1:
            continue
        current, base = current_values[0], base_values[0]
        if current.normalized_unit != base.normalized_unit:
            continue
        change = current.normalized_value - base.normalized_value
        if current.normalized_unit == "percentage_point":
            lines.append((
                metric,
                f"{metric}: {current_period}为{_format_number(current.normalized_value)}%，"
                f"{base_period}为{_format_number(base.normalized_value)}%，"
                f"差值为{_format_number(change)}个百分点。",
            ))
            continue
        if base.normalized_value == 0:
            continue
        growth_rate = change / base.normalized_value * Decimal("100")
        lines.append((
            metric,
            f"{metric}: {current_period}为{current.raw_value}{current.unit}，"
            f"{base_period}为{base.raw_value}{base.unit}；"
            f"按统一金额单位计算，绝对变化为{_format_number(change)}元，"
            f"增长率为{_format_number(growth_rate)}%。",
        ))
    return tuple(lines)


def build_calculation_note(query_metadata, evidence_values):
    """Calculate only from verified structured evidence, never raw Parent text."""
    lines = _verified_calculation_lines(query_metadata, evidence_values)
    if not lines:
        return ""
    return "【系统基于已验证财务数值的确定性计算】\n" + "\n".join(
        line for _, line in lines
    )


def build_calculation_guardrail(query_metadata, evidence_values):
    """Block LLM arithmetic for each requested metric lacking verified inputs."""
    if not query_metadata or not query_metadata.requires_deterministic_calculation():
        return ""
    verified_metrics = {metric for metric, _ in _verified_calculation_lines(
        query_metadata, evidence_values
    )}
    missing_metrics = [
        metric for metric in query_metadata.requested_metrics
        if metric not in verified_metrics
    ]
    if not missing_metrics:
        return ""
    labels = "、".join(_METRIC_LABELS.get(metric, metric) for metric in missing_metrics)
    return (
        "【系统计算约束】\n"
        f"本问题明确要求派生计算，但以下指标未获得完成计算所需的已验证证据：{labels}。\n"
        "不得根据原始上下文中的数字自行计算或推导同比、增长率、增加/减少金额、差值或百分点变化。"
        "可以逐项引用系统已验证财务数值；并且必须明确说明："
        "当前证据不足以完成可靠的确定性计算。"
    )
