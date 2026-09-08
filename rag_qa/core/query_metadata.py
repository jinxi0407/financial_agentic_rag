"""Deterministic financial-query metadata extraction and report lookup replies."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CompanyDefinition:
    company_name: str
    company_code: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class SubQueryTarget:
    """A deterministic retrieval target with its filter bound to the query."""

    query: str
    company_name: str | None = None
    company_code: str | None = None
    report_period: str | None = None

    def to_dict(self):
        metadata_filter = {}
        if self.company_code:
            metadata_filter["company_code"] = self.company_code
        if self.report_period:
            metadata_filter["report_period"] = self.report_period
        return {"query": self.query, "metadata_filter": metadata_filter or None}


COMPANIES = (
    CompanyDefinition("贵州茅台", "600519", ("贵州茅台", "茅台", "600519")),
    CompanyDefinition("五粮液", "000858", ("五粮液", "000858")),
    CompanyDefinition("比亚迪", "002594", ("比亚迪", "002594")),
    CompanyDefinition("宁德时代", "300750", ("宁德时代", "300750")),
    CompanyDefinition("招商银行", "600036", ("招商银行", "招行", "600036")),
    CompanyDefinition("平安银行", "000001", ("平安银行", "000001")),
    CompanyDefinition("中芯国际", "688981", ("中芯国际", "中芯", "688981")),
    CompanyDefinition("北方华创", "002371", ("北方华创", "002371")),
)

_H1_PATTERN = re.compile(
    r"(?P<year>20\d{2})\s*(?:年\s*)?(?:上半年|半年度|半年|H1)",
    re.IGNORECASE,
)
_FY_PATTERN = re.compile(
    r"(?P<year>20\d{2})\s*(?:年\s*)?(?:年度|年报|全年|FY)",
    re.IGNORECASE,
)
_EXPLICIT_YEAR_PATTERN = re.compile(r"(?<!\d)20\d{2}(?!\d)")
_REPORT_ACTION_TERMS = ("想看", "查看", "看一下", "给我看", "下载", "获取")
_REPORT_NOUN_TERMS = ("财报", "报告", "年报", "半年报", "半年度报告", "年度报告")
_H1_LOOKUP_TERMS = ("上半年", "半年度", "半年报")
_FY_LOOKUP_TERMS = ("年度报告", "年报", "年度", "全年")
_COMPARISON_TERMS = ("比较", "相比", "分别", "同比", "变化多少")


@dataclass(frozen=True)
class QueryMetadata:
    query: str = ""
    company_name: str | None = None
    company_code: str | None = None
    report_year: int | None = None
    period_type: str | None = None
    report_period: str | None = None
    report_periods: tuple[str, ...] = ()
    company_names: tuple[str, ...] = ()
    company_codes: tuple[str, ...] = ()
    subquery_targets: tuple[SubQueryTarget, ...] = ()
    intent: str = "RAG"

    def to_metadata_filter(self):
        # Multi-target requests must be bound separately, not compressed globally.
        if len(self.company_codes) > 1 or len(self.report_periods) > 1:
            return None
        if not self.company_code:
            return None
        metadata_filter = {"company_code": self.company_code}
        if self.report_period:
            metadata_filter["report_period"] = self.report_period
        return metadata_filter

    def requires_deterministic_subqueries(self):
        multiple_explicit_targets = (
            len(self.company_codes) >= 2 or len(self.report_periods) >= 2
        )
        comparison_expression = any(term in self.query for term in _COMPARISON_TERMS)
        return bool(
            self.intent == "RAG"
            and self.subquery_targets
            and (multiple_explicit_targets or (
                comparison_expression and len(self.subquery_targets) >= 2
            ))
        )

    def subquery_plan(self):
        return tuple(target.to_dict() for target in self.subquery_targets)


def _contains_alias(query, alias):
    if alias.isdigit():
        return re.search(rf"(?<!\d){re.escape(alias)}(?!\d)", query) is not None
    return alias in query


def _extract_companies(query):
    matches = []
    for company in COMPANIES:
        positions = [
            query.find(alias)
            for alias in company.aliases
            if _contains_alias(query, alias)
        ]
        if positions:
            matches.append((min(position for position in positions if position >= 0), company))
    return tuple(company for _, company in sorted(matches, key=lambda item: item[0]))


def _extract_report_periods(query):
    matches = []
    for match in _H1_PATTERN.finditer(query):
        matches.append((match.start(), f"{match.group('year')}H1"))
    for match in _FY_PATTERN.finditer(query):
        matches.append((match.start(), f"{match.group('year')}FY"))

    periods = []
    for _, period in sorted(matches):
        if period not in periods:
            periods.append(period)
    return tuple(periods)


def _target_query(query, company, report_period):
    labels = [company.company_name] if company else []
    if report_period:
        labels.append(report_period)
    return f"{query}（检索目标：{'，'.join(labels)}）" if labels else query


def _build_subquery_targets(query, companies, periods):
    if not companies and not periods:
        return ()

    return tuple(
        SubQueryTarget(
            query=_target_query(query, company, report_period),
            company_name=company.company_name if company else None,
            company_code=company.company_code if company else None,
            report_period=report_period,
        )
        for company in (companies or (None,))
        for report_period in (periods or (None,))
    )


def is_report_lookup_query(query):
    return (
        any(term in query for term in _REPORT_ACTION_TERMS)
        and any(term in query for term in _REPORT_NOUN_TERMS)
    )


def _extract_lookup_period_type(query):
    if any(term in query for term in _H1_LOOKUP_TERMS):
        return "H1"
    if any(term in query for term in _FY_LOOKUP_TERMS):
        return "FY"
    return None


def extract_query_metadata(query):
    """Extract only explicit metadata; bare years deliberately remain unclassified."""
    companies = _extract_companies(query)
    company = companies[0] if len(companies) == 1 else None
    periods = _extract_report_periods(query)
    if len(periods) == 1:
        report_period = periods[0]
        report_year = int(report_period[:4])
        period_type = report_period[4:]
    else:
        report_period = None
        report_year = None
        period_type = None

    intent = "REPORT_LOOKUP" if is_report_lookup_query(query) else "RAG"
    if intent == "REPORT_LOOKUP" and period_type is None:
        period_type = _extract_lookup_period_type(query)

    return QueryMetadata(
        query=query,
        company_name=company.company_name if company else (companies[0].company_name if companies else None),
        company_code=company.company_code if company else (companies[0].company_code if companies else None),
        report_year=report_year,
        period_type=period_type,
        report_period=report_period,
        report_periods=periods,
        company_names=tuple(company.company_name for company in companies),
        company_codes=tuple(company.company_code for company in companies),
        subquery_targets=_build_subquery_targets(query, companies, periods),
        intent=intent,
    )


def should_bypass_faq(query, metadata=None):
    """Keep company-, period-, and report-specific requests out of generic FAQ."""
    metadata = metadata or extract_query_metadata(query)
    return bool(
        metadata.company_name
        or metadata.company_code
        or metadata.report_period
        or metadata.intent == "REPORT_LOOKUP"
        or _EXPLICIT_YEAR_PATTERN.search(query)
    )


def build_report_lookup_response(metadata, catalog):
    """Return a deterministic catalog response without entering vector retrieval."""
    if not metadata.company_code:
        return "请说明你想查看哪家公司的财报。"

    reports = catalog.find_reports(
        metadata.company_code,
        report_year=metadata.report_year,
        period_type=metadata.period_type,
    )
    company_name = metadata.company_name or metadata.company_code
    if not reports:
        return f"当前报告目录中未找到{company_name}符合条件的报告。"
    if len(reports) == 1:
        report = reports[0]
        return f"当前可用：{report.display_name}（文件名：{report.source_filename}）。"

    report_names = "、".join(report.display_name for report in reports)
    if metadata.period_type:
        return f"当前有{report_names}，请问你想查看哪一年？"
    return "当前可用报告：\n" + "\n".join(
        f"- {report.display_name}" for report in reports
    )
