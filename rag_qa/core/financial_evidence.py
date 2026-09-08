"""High-confidence extraction of financial values from already-selected Parent documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable

from rag_qa.core.query_metadata import METRIC_DEFINITIONS


_NUMBER_PATTERN = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_DIRECT_UNIT_PATTERN = re.compile(r"\s*(?P<unit>百万元|千元|万元|亿元|人民币元|元)")
_TABLE_UNIT_PATTERNS = (
    re.compile(r"单位\s*[：:]\s*(?:人民币)?(?P<unit>百万元|千元|万元|亿元|人民币元|元)"),
    re.compile(r"(?:货币|金额)单位\s*(?:均为)?\s*[：:]?\s*(?:人民币)?(?P<unit>百万元|千元|万元|亿元|人民币元|元)"),
    re.compile(r"[（(]\s*(?:人民币)?(?P<unit>百万元|千元|万元|亿元|人民币元|元)(?:[，,）)])"),
)
_AMOUNT_MULTIPLIERS = {
    "元": Decimal("1"),
    "人民币元": Decimal("1"),
    "千元": Decimal("1000"),
    "万元": Decimal("10000"),
    "百万元": Decimal("1000000"),
    "亿元": Decimal("100000000"),
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
class StructuredFinancialEvidence:
    company_name: str
    company_code: str
    report_period: str
    metric: str
    raw_value: str
    normalized_value: Decimal
    unit: str
    normalized_unit: str
    source_parent: str
    exact_match: bool
    confidence: str


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _period_labels(report_period: str) -> tuple[str, ...]:
    year, period_type = report_period[:4], report_period[4:]
    if period_type == "H1":
        return (f"{year}年上半年", f"{year}年半年度", f"{year}年1-6月", f"{year}H1")
    if period_type == "FY":
        return (f"{year}年度", f"{year}年报", f"{year}全年", f"{year}FY")
    return ()


def _table_unit_before(text: str, index: int) -> str | None:
    window = text[max(0, index - 700):index]
    matches = []
    for pattern in _TABLE_UNIT_PATTERNS:
        matches.extend(pattern.finditer(window))
    if not matches:
        return None
    return max(matches, key=lambda match: match.start()).group("unit")


def _metric_inline_unit(text: str, index: int) -> str | None:
    """Read an explicit unit printed immediately after a table-row label."""
    match = re.match(
        r"\s*[（(]\s*(?P<unit>百万元|千元|万元|亿元|人民币元|元)\s*[）)]",
        text[index:index + 32],
    )
    return match.group("unit") if match else None


def _current_period_binding(text: str, index: int, report_period: str) -> bool:
    window = _compact(text[max(0, index - 700):index])
    if "本报告期" in window or "本期" in window:
        return True
    return any(label in window for label in _period_labels(report_period))


def _is_industry_metric(text: str, index: int, metric: str) -> bool:
    return metric == "gross_margin" and "行业" in text[max(0, index - 16):index + 16]


def _alias_matches(text: str, alias: str):
    """Find an exact label while tolerating PDF line wrapping inside it."""
    pattern = re.compile(r"\s*".join(re.escape(character) for character in alias))
    return pattern.finditer(text)


def _amount_after_metric(text: str, index: int, report_period: str):
    if not _current_period_binding(text, index, report_period):
        return None
    unit_from_table = _table_unit_before(text, index)
    unit_from_label = _metric_inline_unit(text, index)
    for match in _NUMBER_PATTERN.finditer(text[index:index + 320]):
        raw_value = match.group(0)
        try:
            numeric_value = Decimal(raw_value.replace(",", ""))
        except InvalidOperation:
            continue
        value_end = index + match.end()
        direct_unit_match = _DIRECT_UNIT_PATTERN.match(text[value_end:value_end + 16])
        unit = (
            direct_unit_match.group("unit")
            if direct_unit_match
            else unit_from_label or unit_from_table
        )
        if unit is None:
            continue
        return raw_value, numeric_value, unit
    return None


def _rate_after_metric(text: str, index: int, report_period: str):
    if not _current_period_binding(text, index, report_period):
        return None
    for match in _NUMBER_PATTERN.finditer(text[index:index + 240]):
        raw_value = match.group(0)
        value_end = index + match.end()
        if not re.match(r"\s*%", text[value_end:value_end + 8]):
            continue
        try:
            return raw_value, Decimal(raw_value.replace(",", ""))
        except InvalidOperation:
            continue
    return None


def extract_document_evidence(document, metrics: Iterable[str] | None = None):
    """Return only values with exact company, period, metric and unit bindings."""
    metadata = getattr(document, "metadata", {}) or {}
    company_name = metadata.get("company_name")
    company_code = metadata.get("company_code")
    report_period = metadata.get("report_period")
    source_parent = metadata.get("parent_id")
    if not all(isinstance(value, str) and value for value in (
        company_name, company_code, report_period, source_parent,
    )):
        return ()

    text = getattr(document, "page_content", "") or ""
    selected_metrics = tuple(metrics or METRIC_DEFINITIONS)
    extracted = []
    for metric in selected_metrics:
        aliases = sorted(METRIC_DEFINITIONS.get(metric, ()), key=len, reverse=True)
        metric_extracted = False
        for alias in aliases:
            for match in _alias_matches(text, alias):
                if _is_industry_metric(text, match.start(), metric):
                    continue
                if metric in _RATE_METRICS:
                    candidate = _rate_after_metric(text, match.end(), report_period)
                    if candidate is None:
                        continue
                    raw_value, normalized_value = candidate
                    extracted.append(StructuredFinancialEvidence(
                        company_name=company_name,
                        company_code=company_code,
                        report_period=report_period,
                        metric=metric,
                        raw_value=raw_value,
                        normalized_value=normalized_value,
                        unit="%",
                        normalized_unit="percentage_point",
                        source_parent=source_parent,
                        exact_match=True,
                        confidence="high",
                    ))
                    metric_extracted = True
                    break

                candidate = _amount_after_metric(text, match.end(), report_period)
                if candidate is None:
                    continue
                raw_value, numeric_value, unit = candidate
                extracted.append(StructuredFinancialEvidence(
                    company_name=company_name,
                    company_code=company_code,
                    report_period=report_period,
                    metric=metric,
                    raw_value=raw_value,
                    normalized_value=numeric_value * _AMOUNT_MULTIPLIERS[unit],
                    unit=unit,
                    normalized_unit="元",
                    source_parent=source_parent,
                    exact_match=True,
                    confidence="high",
                ))
                metric_extracted = True
                break
            if metric_extracted:
                break
    return tuple(extracted)


def extract_verified_evidence(documents, metrics: Iterable[str] | None = None):
    """Return one value per cell, rejecting cells with conflicting exact values."""
    candidates = {}
    for document in documents:
        for value in extract_document_evidence(document, metrics):
            key = (value.company_code, value.report_period, value.metric)
            candidates.setdefault(key, []).append(value)

    verified = []
    for values in candidates.values():
        if len({value.normalized_value for value in values}) != 1:
            continue
        verified.append(values[0])
    return tuple(verified)


def _format_decimal(value: Decimal) -> str:
    rendered = f"{value:,.2f}"
    return rendered.rstrip("0").rstrip(".")


def format_verified_evidence_block(evidence_values) -> str:
    if not evidence_values:
        return ""
    lines = ["【系统已验证财务数值】"]
    for value in evidence_values:
        label = _METRIC_LABELS.get(value.metric, value.metric)
        normalized = f"{_format_decimal(value.normalized_value)}{value.normalized_unit}"
        lines.append(
            f"- {value.company_name}（{value.company_code}）{value.report_period} {label}："
            f"{value.raw_value}{value.unit}；标准化值：{normalized}；"
            f"来源 Parent：{value.source_parent}。"
        )
    lines.append("以上数值已由系统绑定公司、报告期间、指标和单位；必须原样使用，不得改写或重新计算。")
    return "\n".join(lines)
