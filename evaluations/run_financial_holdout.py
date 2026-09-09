"""Generate, audit, and (only when explicitly confirmed) run the 300Q holdout.

The default commands are intentionally offline: they only inspect the frozen
PDF filename catalog and JSON labels.  ``run`` requires ``--confirm-run`` and
uses retrieval only; it never calls the final-answer LLM, Answer Judge, or any
Agent/MCP tool.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

try:
    # Prefer the production parser when the full ingestion dependency graph is
    # available.  This utility must also remain runnable in a lightweight
    # audit environment where optional document splitters are absent.
    from rag_qa.core.document_processor import parse_annual_report_filename as _production_filename_parser
except ImportError:  # pragma: no cover - depends on optional ingestion extras
    _production_filename_parser = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = ROOT / "financial_data" / "annual_reports"
DEFAULT_GOLD_60 = ROOT / "evaluations" / "financial_eval_60_gold_v2.json"
DEFAULT_DATASET = ROOT / "evaluations" / "financial_holdout_300_v1.json"
DEFAULT_AUDIT = ROOT / "evaluations" / "financial_holdout_300_v1_audit.json"

PERIODS = ("2025H1", "2025FY", "2026H1")
CATEGORY_COUNTS = {
    "single_company_single_period": 60,
    "single_company_multi_period": 40,
    "multi_company_single_period": 60,
    "multi_company_multi_period": 40,
    "complex_multi_target": 20,
    "metric_alias_ambiguity": 30,
    "explanation_risk": 20,
    "negative_oos_invalid": 30,
}
VALID_ROUTES = {"RAG", "OUT_OF_SCOPE"}
ANNUAL_REPORT_FILENAME_RE = re.compile(
    r"(?P<company_name>[^_]+)_(?P<company_code>\d{6})_"
    r"(?P<report_year>\d{4})(?P<period_type>H1|FY)\.pdf"
)


@dataclass(frozen=True)
class Company:
    name: str
    code: str
    kind: str  # bank or industrial
    aliases: tuple[str, ...]


COMPANIES = (
    Company("贵州茅台", "600519", "industrial", ("贵州茅台", "茅台", "600519")),
    Company("五粮液", "000858", "industrial", ("五粮液", "000858")),
    Company("比亚迪", "002594", "industrial", ("比亚迪", "002594")),
    Company("宁德时代", "300750", "industrial", ("宁德时代", "300750")),
    Company("招商银行", "600036", "bank", ("招商银行", "招行", "600036")),
    Company("平安银行", "000001", "bank", ("平安银行", "000001")),
    Company("中芯国际", "688981", "industrial", ("中芯国际", "中芯", "688981")),
    Company("北方华创", "002371", "industrial", ("北方华创", "002371")),
)
COMPANY_BY_CODE = {company.code: company for company in COMPANIES}

# The IDs are deliberately conservative.  They do not make an accounting
# equivalence claim simply because two labels sound similar.
METRICS: dict[str, dict[str, Any]] = {
    "revenue": {"label": "营业收入", "phrases": ("营业收入", "营收"), "kinds": {"bank", "industrial"}},
    "net_profit": {"label": "归母净利润", "phrases": ("归母净利润", "归属于上市公司股东的净利润"), "kinds": {"bank", "industrial"}},
    "operating_cash_flow": {"label": "经营现金流", "phrases": ("经营现金流", "经营活动产生的现金流量净额"), "kinds": {"bank", "industrial"}},
    "total_assets": {"label": "资产总额", "phrases": ("资产总额", "总资产"), "kinds": {"bank", "industrial"}},
    "total_liabilities": {"label": "负债总额", "phrases": ("负债总额", "总负债"), "kinds": {"bank", "industrial"}},
    "rd_expense": {"label": "研发费用", "phrases": ("研发费用",), "kinds": {"industrial"}},
    "rd_investment": {"label": "研发投入", "phrases": ("研发投入",), "kinds": {"industrial"}},
    "gross_margin": {"label": "毛利率", "phrases": ("毛利率",), "kinds": {"industrial"}},
    "net_margin": {"label": "净利率", "phrases": ("净利率",), "kinds": {"industrial"}},
    "net_interest_margin": {"label": "净息差", "phrases": ("净息差", "净利息收益率"), "kinds": {"bank"}},
    "npl_ratio": {"label": "不良贷款率", "phrases": ("不良贷款率", "不良率"), "kinds": {"bank"}},
    "provision_coverage": {"label": "拨备覆盖率", "phrases": ("拨备覆盖率",), "kinds": {"bank"}},
    "business_risk": {"label": "经营风险", "phrases": ("经营风险", "资产质量风险"), "kinds": {"bank", "industrial"}},
    "business_change": {"label": "业务变化", "phrases": ("业务变化", "经营变化"), "kinds": {"bank", "industrial"}},
}


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_question(question: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", question.lower())


def period_label(period: str) -> str:
    year, suffix = period[:4], period[4:]
    return f"{year}年上半年" if suffix == "H1" else f"{year}年度"


def metric_label(metric_id: str) -> str:
    return METRICS[metric_id]["label"]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_annual_report_filename(filename: str) -> dict[str, Any]:
    """Use the production parser, with an exact local mirror for offline audit.

    The mirror is intentionally byte-for-byte equivalent in contract to
    ``document_processor.parse_annual_report_filename``; it only avoids
    importing OCR/splitter dependencies when this script merely audits names.
    """
    if _production_filename_parser is not None:
        return _production_filename_parser(filename)
    match = ANNUAL_REPORT_FILENAME_RE.fullmatch(filename)
    if not match:
        raise ValueError(
            "annual_reports PDF filename must match "
            "<company_name>_<6-digit company_code>_<YYYY><H1|FY>.pdf: "
            f"{filename}"
        )
    metadata: dict[str, Any] = match.groupdict()
    metadata["report_year"] = int(metadata["report_year"])
    metadata["report_period"] = f"{metadata['report_year']}{metadata['period_type']}"
    return metadata


def build_catalog(reports_dir: Path = DEFAULT_REPORTS_DIR) -> dict[tuple[str, str], str]:
    """Build labels solely from actual standard PDF names using the production parser."""
    catalog: dict[tuple[str, str], str] = {}
    errors: list[str] = []
    for path in sorted(reports_dir.glob("*.pdf")):
        try:
            metadata = parse_annual_report_filename(path.name)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        code = str(metadata["company_code"])
        period = str(metadata["report_period"])
        if code not in COMPANY_BY_CODE:
            errors.append(f"非 holdout 合法公司: {path.name}")
            continue
        expected_name = COMPANY_BY_CODE[code].name
        if metadata["company_name"] != expected_name:
            errors.append(f"公司名称/代码不一致: {path.name}")
            continue
        key = (code, period)
        if key in catalog:
            errors.append(f"重复报告期间: {path.name} / {catalog[key]}")
        catalog[key] = path.name

    expected = {(company.code, period) for company in COMPANIES for period in PERIODS}
    if errors or set(catalog) != expected:
        missing = sorted(expected - set(catalog))
        extra = sorted(set(catalog) - expected)
        detail = errors + ([f"缺少: {missing}"] if missing else []) + ([f"额外: {extra}"] if extra else [])
        raise ValueError("PDF catalog 不满足冻结 24 份财报契约: " + "; ".join(detail))
    return catalog


def _available_metrics(companies: Iterable[Company]) -> list[str]:
    kinds = {company.kind for company in companies}
    return [metric_id for metric_id, spec in METRICS.items() if kinds <= spec["kinds"]]


def _case(
    *,
    case_id: str,
    category: str,
    question: str,
    targets: list[dict[str, Any]],
    metric_ids: list[str],
    catalog: dict[tuple[str, str], str],
    expected_rag: bool = True,
    expected_route: str = "RAG",
    negative_type: str | None = None,
) -> dict[str, Any]:
    codes = sorted({target["company_code"] for target in targets})
    periods = sorted({target["report_period"] for target in targets})
    documents = sorted({catalog[(target["company_code"], target["report_period"])] for target in targets})
    return {
        "id": case_id,
        "category": category,
        "question": question,
        "expected_rag": expected_rag,
        "expected_route": expected_route,
        "expected_company_codes": codes,
        "expected_company_names": [COMPANY_BY_CODE[code].name for code in codes],
        "expected_report_periods": periods,
        "required_documents": documents,
        "metric": " + ".join(metric_label(metric_id) for metric_id in metric_ids),
        "metrics": metric_ids,
        "targets": targets,
        "negative_type": negative_type,
    }


def _valid_case(
    category: str,
    question: str,
    target_pairs: Iterable[tuple[Company, str]],
    metric_ids: list[str],
    catalog: dict[tuple[str, str], str],
) -> dict[str, Any]:
    targets = [
        {
            "company_name": company.name,
            "company_code": company.code,
            "report_period": period,
            "metrics": metric_ids,
        }
        for company, period in target_pairs
    ]
    return _case(
        case_id="",
        category=category,
        question=question,
        targets=targets,
        metric_ids=metric_ids,
        catalog=catalog,
    )


def _negative_case(
    category: str,
    question: str,
    negative_type: str,
    *,
    expected_route: str,
) -> dict[str, Any]:
    return {
        "id": "",
        "category": category,
        "question": question,
        "expected_rag": expected_route == "RAG",
        "expected_route": expected_route,
        "expected_company_codes": [],
        "expected_company_names": [],
        "expected_report_periods": [],
        "required_documents": [],
        "metric": None,
        "metrics": [],
        "targets": [],
        "negative_type": negative_type,
    }


def _single_pool(catalog: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    templates = (
        "帮我查一下{company}{period}的{metric}。",
        "{company}{period}{metric}是多少？",
        "请给出{company}在{period}披露的{metric}。",
        "我想了解{company}{period}的{metric}表现。",
        "从财报里找{company}{period}{metric}。",
        "{period}，{company}的{metric}数据如何？",
    )
    pool: list[dict[str, Any]] = []
    for index, company in enumerate(COMPANIES):
        for period_index, period in enumerate(PERIODS):
            for metric_index, metric_id in enumerate(_available_metrics([company])):
                template = templates[(index + period_index + metric_index) % len(templates)]
                pool.append(_valid_case(
                    "single_company_single_period",
                    template.format(company=company.name, period=period_label(period), metric=metric_label(metric_id)),
                    [(company, period)], [metric_id], catalog,
                ))
    return pool


def _single_multi_period_pool(catalog: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    templates = (
        "比较{company}{left}和{right}的{metric}，分别是多少？",
        "{company}从{left}到{right}的{metric}有什么变化？",
        "请对照{company}{left}、{right}披露的{metric}。",
        "想看{company}在{left}与{right}的{metric}差异。",
    )
    pairs = (("2025H1", "2026H1"), ("2025H1", "2025FY"), ("2025FY", "2026H1"))
    pool: list[dict[str, Any]] = []
    for company_index, company in enumerate(COMPANIES):
        metrics = _available_metrics([company])
        for pair_index, (left, right) in enumerate(pairs):
            for metric_index, metric_id in enumerate(metrics):
                template = templates[(company_index + pair_index + metric_index) % len(templates)]
                pool.append(_valid_case(
                    "single_company_multi_period",
                    template.format(company=company.name, left=period_label(left), right=period_label(right), metric=metric_label(metric_id)),
                    [(company, left), (company, right)], [metric_id], catalog,
                ))
    return pool


def _company_pairs() -> list[tuple[Company, Company]]:
    return [(COMPANIES[i], COMPANIES[j]) for i in range(len(COMPANIES)) for j in range(i + 1, len(COMPANIES))]


def _multi_single_period_pool(catalog: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    templates = (
        "比较{left}和{right}{period}的{metric}。",
        "{left}、{right}在{period}谁的{metric}更高？",
        "请分别列出{left}与{right}{period}的{metric}。",
        "对比一下{left}和{right}的{period}{metric}表现。",
    )
    pool: list[dict[str, Any]] = []
    for pair_index, (left, right) in enumerate(_company_pairs()):
        metrics = _available_metrics([left, right])
        for period_index, period in enumerate(PERIODS):
            for metric_index, metric_id in enumerate(metrics):
                template = templates[(pair_index + period_index + metric_index) % len(templates)]
                pool.append(_valid_case(
                    "multi_company_single_period",
                    template.format(left=left.name, right=right.name, period=period_label(period), metric=metric_label(metric_id)),
                    [(left, period), (right, period)], [metric_id], catalog,
                ))
    return pool


def _multi_multi_period_pool(catalog: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    templates = (
        "比较{left}和{right}从{first}到{second}的{metric}变化。",
        "请对照{left}、{right}在{first}及{second}披露的{metric}。",
        "{left}与{right}的{metric}，从{first}到{second}分别如何？",
        "帮我分析{left}和{right}{first}、{second}的{metric}差异。",
    )
    period_pairs = (("2025H1", "2026H1"), ("2025H1", "2025FY"), ("2025FY", "2026H1"))
    pool: list[dict[str, Any]] = []
    for pair_index, (left, right) in enumerate(_company_pairs()):
        metrics = _available_metrics([left, right])
        for period_index, (first, second) in enumerate(period_pairs):
            for metric_index, metric_id in enumerate(metrics):
                template = templates[(pair_index + period_index + metric_index) % len(templates)]
                pool.append(_valid_case(
                    "multi_company_multi_period",
                    template.format(left=left.name, right=right.name, first=period_label(first), second=period_label(second), metric=metric_label(metric_id)),
                    [(left, first), (left, second), (right, first), (right, second)], [metric_id], catalog,
                ))
    return pool


def _complex_pool(catalog: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    pairs = _company_pairs()
    for index, (left, right) in enumerate(pairs):
        period = PERIODS[index % len(PERIODS)]
        metrics = ["revenue", "net_profit"]
        pool.append(_valid_case(
            "complex_multi_target",
            f"比较{left.name}和{right.name}{period_label(period)}的营业收入与归母净利润，并说明差异。",
            [(left, period), (right, period)], metrics, catalog,
        ))
    for index in range(len(COMPANIES)):
        first, second, third = (COMPANIES[index], COMPANIES[(index + 1) % 8], COMPANIES[(index + 2) % 8])
        period = PERIODS[index % len(PERIODS)]
        pool.append(_valid_case(
            "complex_multi_target",
            f"{first.name}、{second.name}和{third.name}在{period_label(period)}的营业收入分别是多少？",
            [(first, period), (second, period), (third, period)], ["revenue"], catalog,
        ))
    for index, (left, right) in enumerate(pairs):
        if index % 2 == 0:
            pool.append(_valid_case(
                "complex_multi_target",
                f"对比{left.name}与{right.name}在2025年上半年和2026年上半年的营业收入、归母净利润。",
                [(left, "2025H1"), (left, "2026H1"), (right, "2025H1"), (right, "2026H1")],
                ["revenue", "net_profit"], catalog,
            ))
    return pool


def _ambiguity_pool(catalog: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for index, company in enumerate(COMPANIES):
        if company.kind == "bank":
            variants = (
                ("net_interest_margin", "请只按净息差口径回答，不要把净利息收入当成净息差。"),
                ("npl_ratio", "只说明不良贷款率，不要改用不良贷款余额。"),
                ("provision_coverage", "请按拨备覆盖率的百分比口径给出数据。"),
            )
        else:
            variants = (
                ("revenue", "只回答营业收入，不要将营业总收入混为同一指标。"),
                ("rd_expense", "仅查研发费用，不要把研发投入替代为研发费用。"),
                ("rd_investment", "仅查研发投入，不要用研发费用替代。"),
                ("gross_margin", "请给公司毛利率，不要引用行业毛利率。"),
            )
        for period in PERIODS:
            for metric_id, suffix in variants:
                pool.append(_valid_case(
                    "metric_alias_ambiguity",
                    f"{company.name}{period_label(period)}的{metric_label(metric_id)}是多少？{suffix}",
                    [(company, period)], [metric_id], catalog,
                ))
    return pool


def _explanation_pool(catalog: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    templates = (
        "结合{company}{period}财报，分析其{topic}，只依据披露内容。",
        "{company}{period}有哪些值得关注的{topic}？",
        "请基于{company}{period}报告解释{topic}，不要延伸到未披露事实。",
    )
    pool: list[dict[str, Any]] = []
    for company_index, company in enumerate(COMPANIES):
        topics = (
            ("资产质量与经营风险", "business_risk"),
            ("业务变化与经营表现", "business_change"),
        ) if company.kind == "bank" else (
            ("经营风险", "business_risk"),
            ("业务变化", "business_change"),
        )
        for period_index, period in enumerate(PERIODS):
            for topic_index, (topic, metric_id) in enumerate(topics):
                template = templates[(company_index + period_index + topic_index) % len(templates)]
                pool.append(_valid_case(
                    "explanation_risk",
                    template.format(company=company.name, period=period_label(period), topic=topic),
                    [(company, period)], [metric_id], catalog,
                ))
    return pool


def _negative_pool() -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    unsupported = (("工商银行", "601398"), ("中国平安", "601318"), ("隆基绿能", "601012"), ("腾讯控股", "00700"),
                   ("阿里巴巴", "BABA"), ("美的集团", "000333"), ("海康威视", "002415"), ("中国石油", "601857"),
                   ("格力电器", "000651"), ("药明康德", "603259"), ("京东集团", "JD"), ("中国移动", "600941"))
    for name, code in unsupported:
        pool.append(_negative_case(
            "negative_oos_invalid", f"{name}{code}2026年上半年的营业收入是多少？", "unsupported_company", expected_route="RAG"))
    for index, company in enumerate(COMPANIES):
        unavailable = "2024FY" if index % 2 == 0 else "2026FY"
        pool.append(_negative_case(
            "negative_oos_invalid", f"{company.name}{period_label(unavailable)}的营业收入是多少？", "unavailable_period", expected_route="RAG"))
    oos_questions = (
        "法国的首都是哪里？", "帮我解释量子纠缠。", "写一首关于秋天的诗。", "如何做番茄炒蛋？",
        "Python 的装饰器怎么用？", "推荐一部科幻电影。", "今天天气怎么样？", "如何学习西班牙语？",
        "给我一道微积分练习题。", "东京有哪些旅游景点？",
    )
    for question in oos_questions:
        pool.append(_negative_case("negative_oos_invalid", question, "out_of_scope", expected_route="OUT_OF_SCOPE"))
    realtime_questions = (
        "贵州茅台今天的股价是多少？", "五粮液现在的成交量如何？", "比亚迪最新实时市值是多少？",
        "宁德时代今天涨跌了多少？", "招商银行当前股价是多少？", "平安银行实时行情怎么样？",
        "中芯国际今天有什么即时公告？", "北方华创当前股价是多少？", "贵州茅台今日实时估值如何？",
    )
    for question in realtime_questions:
        pool.append(_negative_case("negative_oos_invalid", question, "realtime_not_report_query", expected_route="OUT_OF_SCOPE"))
    return pool


def _metric_ids_from_text(question: str) -> tuple[str, ...]:
    matched: list[str] = []
    for metric_id, spec in METRICS.items():
        if any(phrase in question for phrase in spec["phrases"]):
            matched.append(metric_id)
    return tuple(sorted(matched))


def _old_signature(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(case.get("category", "")),
        tuple(sorted(str(value) for value in case.get("target_company_codes", []))),
        tuple(sorted(str(value) for value in case.get("target_report_periods", []))),
        _metric_ids_from_text(str(case.get("question", ""))),
    )


def _new_signature(case: dict[str, Any]) -> tuple[Any, ...]:
    if case.get("negative_type") is not None:
        # Negative cases intentionally have no company/period/metric target;
        # their contract is instead the negative class plus the normalized
        # request.  Treating all of them as one empty target signature would
        # make a 30-case negative holdout impossible to audit.
        return ("negative", case["negative_type"], normalize_question(case["question"]))
    return (
        str(case["category"]),
        tuple(sorted(case["expected_company_codes"])),
        tuple(sorted(case["expected_report_periods"])),
        tuple(sorted(case.get("metrics", []))),
    )


def _semantic_signature(case: dict[str, Any]) -> tuple[Any, ...]:
    if case.get("negative_type") is not None:
        return ()
    return (
        tuple(sorted(case.get("expected_company_codes") or case.get("target_company_codes", []))),
        tuple(sorted(case.get("expected_report_periods") or case.get("target_report_periods", []))),
        tuple(sorted(case.get("metrics") or _metric_ids_from_text(str(case.get("question", ""))))),
    )


def _near_duplicate_score(question: str, old_question: str) -> float:
    return SequenceMatcher(None, normalize_question(question), normalize_question(old_question)).ratio()


def _select_cases(
    pool: Iterable[dict[str, Any]],
    count: int,
    old_cases: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    old_normalized = {normalize_question(str(case["question"])) for case in old_cases}
    old_signatures = {_old_signature(case) for case in old_cases}
    old_semantics = {_semantic_signature(case) for case in old_cases}
    selected_normalized = {normalize_question(case["question"]) for case in selected}
    selected_signatures = {_new_signature(case) for case in selected}
    result: list[dict[str, Any]] = []
    # The 60Q overlap checks are invariant during selection.  Do them once;
    # repeatedly calculating character similarity inside each balancing round
    # would make offline generation needlessly slow.
    remaining: list[dict[str, Any]] = []
    for candidate in pool:
        normalized = normalize_question(candidate["question"])
        signature = _new_signature(candidate)
        semantic = _semantic_signature(candidate)
        if normalized in old_normalized or signature in old_signatures:
            continue
        if candidate.get("negative_type") is None and semantic and semantic in old_semantics:
            continue
        if any(_near_duplicate_score(candidate["question"], old["question"]) >= 0.80 for old in old_cases):
            continue
        remaining.append(candidate)
    while len(result) < count:
        eligible: list[dict[str, Any]] = []
        for candidate in remaining:
            normalized = normalize_question(candidate["question"])
            signature = _new_signature(candidate)
            if normalized in selected_normalized:
                continue
            if signature in selected_signatures:
                continue
            eligible.append(candidate)
        if not eligible:
            raise ValueError(f"无法为 holdout 类别筛出 {count} 条未见候选，仅得到 {len(result)} 条")

        # Do not let the fixed company order of template generation become a
        # sampling bias.  Select the candidate whose projected company/period
        # distribution is least concentrated, with question text only as a
        # deterministic final tie-breaker.
        current_company_counts = Counter(
            target["company_code"] for case in [*selected, *result]
            for target in case.get("targets", [])
        )
        current_period_counts = Counter(
            target["report_period"] for case in [*selected, *result]
            for target in case.get("targets", [])
        )

        def balance_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
            candidate_codes = [target["company_code"] for target in candidate.get("targets", [])]
            candidate_periods = [target["report_period"] for target in candidate.get("targets", [])]
            projected_codes = current_company_counts.copy()
            projected_periods = current_period_counts.copy()
            projected_codes.update(candidate_codes)
            projected_periods.update(candidate_periods)
            code_values = [projected_codes[company.code] for company in COMPANIES]
            period_values = [projected_periods[period] for period in PERIODS]
            return (
                max(code_values) - min(code_values),
                max(code_values),
                max(period_values) - min(period_values),
                max(period_values),
                candidate["question"],
            )

        candidate = min(eligible, key=balance_key)
        result.append(candidate)
        selected_normalized.add(normalize_question(candidate["question"]))
        selected_signatures.add(_new_signature(candidate))
        remaining.remove(candidate)
    return result


def generate_dataset(
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    gold_60_path: Path = DEFAULT_GOLD_60,
) -> list[dict[str, Any]]:
    catalog = build_catalog(reports_dir)
    old_cases = _json_load(gold_60_path)
    pools = {
        "single_company_single_period": _single_pool(catalog),
        "single_company_multi_period": _single_multi_period_pool(catalog),
        "multi_company_single_period": _multi_single_period_pool(catalog),
        "multi_company_multi_period": _multi_multi_period_pool(catalog),
        "complex_multi_target": _complex_pool(catalog),
        "metric_alias_ambiguity": _ambiguity_pool(catalog),
        "explanation_risk": _explanation_pool(catalog),
        "negative_oos_invalid": _negative_pool(),
    }
    selected: list[dict[str, Any]] = []
    for category, count in CATEGORY_COUNTS.items():
        selected.extend(_select_cases(pools[category], count, old_cases, selected))

    if len(selected) != 300:
        raise AssertionError(f"holdout 条数错误: {len(selected)}")
    for index, case in enumerate(selected, 1):
        case["id"] = f"holdout_{index:03d}"
    return selected


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def audit_dataset(
    cases: list[dict[str, Any]],
    *,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    gold_60_path: Path = DEFAULT_GOLD_60,
) -> dict[str, Any]:
    catalog = build_catalog(reports_dir)
    old_cases = _json_load(gold_60_path)
    errors: list[str] = []
    warnings: list[str] = []
    category_counts = Counter(case.get("category") for case in cases)
    if len(cases) != 300:
        errors.append(f"总数应为 300，实际为 {len(cases)}")
    if dict(category_counts) != CATEGORY_COUNTS:
        errors.append(f"类别分布不符: {dict(category_counts)}")
    ids = [case.get("id") for case in cases]
    if len(set(ids)) != len(ids) or any(not case_id for case_id in ids):
        errors.append("case id 重复或为空")
    normalized = [normalize_question(str(case.get("question", ""))) for case in cases]
    if len(set(normalized)) != len(normalized):
        errors.append("存在重复 normalized question")
    signatures = [_new_signature(case) for case in cases]
    if len(set(signatures)) != len(signatures):
        errors.append("存在重复 canonical signature")

    old_normalized = {normalize_question(str(case["question"])) for case in old_cases}
    old_signatures = {_old_signature(case) for case in old_cases}
    old_semantics = {_semantic_signature(case) for case in old_cases}
    exact_question_overlap = [case["id"] for case in cases if normalize_question(case["question"]) in old_normalized]
    signature_overlap = [case["id"] for case in cases if _new_signature(case) in old_signatures]
    semantic_overlap = [case["id"] for case in cases if case.get("negative_type") is None and _semantic_signature(case) and _semantic_signature(case) in old_semantics]
    if exact_question_overlap:
        errors.append(f"与 60Q normalized question 重复: {exact_question_overlap}")
    if signature_overlap:
        errors.append(f"与 60Q canonical signature 重复: {signature_overlap}")
    if semantic_overlap:
        errors.append(f"与 60Q company/period/metric 语义签名重复: {semantic_overlap}")

    near_duplicates: list[dict[str, Any]] = []
    for case in cases:
        best = max(
            ((_near_duplicate_score(case["question"], old["question"]), old["id"], old["question"]) for old in old_cases),
            default=(0.0, "", ""),
        )
        if best[0] >= 0.80:
            near_duplicates.append({"holdout_id": case["id"], "similarity": round(best[0], 4), "gold60_id": best[1], "gold60_question": best[2]})
    if near_duplicates:
        errors.append(f"存在疑似近重复题: {[item['holdout_id'] for item in near_duplicates]}")

    company_counts: Counter[str] = Counter()
    period_counts: Counter[str] = Counter()
    document_counts: Counter[str] = Counter()
    target_count_distribution: Counter[str] = Counter()
    evidence_cell_distribution: Counter[str] = Counter()
    negative_counts: Counter[str] = Counter()
    for case in cases:
        route = case.get("expected_route")
        if route not in VALID_ROUTES:
            errors.append(f"{case.get('id')}: 非法 expected_route={route}")
        is_negative = case.get("negative_type") is not None
        if is_negative:
            negative_counts[str(case["negative_type"])] += 1
            if case.get("required_documents") or case.get("targets"):
                errors.append(f"{case['id']}: negative case 不应有 required documents/targets")
            continue
        if not case.get("expected_rag") or route != "RAG":
            errors.append(f"{case['id']}: valid case 必须 expected_rag=true, expected_route=RAG")
        expected_docs = []
        for target in case.get("targets", []):
            code, period = str(target.get("company_code")), str(target.get("report_period"))
            if code not in COMPANY_BY_CODE or period not in PERIODS:
                errors.append(f"{case['id']}: 非法 target {code}/{period}")
                continue
            if target.get("company_name") != COMPANY_BY_CODE[code].name:
                errors.append(f"{case['id']}: target company_name 与 code 不一致")
            expected_docs.append(catalog[(code, period)])
            company_counts[code] += 1
            period_counts[period] += 1
        if sorted(set(expected_docs)) != sorted(case.get("required_documents", [])):
            errors.append(f"{case['id']}: required_documents 未严格由 targets/catalog 推导")
        document_counts.update(case.get("required_documents", []))
        doc_count = len(case.get("required_documents", []))
        target_count_distribution["4+" if doc_count >= 4 else str(doc_count)] += 1
        cell_count = len(case.get("targets", [])) * len(case.get("metrics", []))
        evidence_cell_distribution["4+" if cell_count >= 4 else str(cell_count)] += 1
        for metric_id in case.get("metrics", []):
            if metric_id not in METRICS:
                errors.append(f"{case['id']}: 未知 metric={metric_id}")

    valid_company_values = list(company_counts.values())
    valid_period_values = list(period_counts.values())
    if valid_company_values and max(valid_company_values) - min(valid_company_values) > 20:
        warnings.append(f"公司出现次数分布偏差大于 20: {dict(company_counts)}")
    if valid_period_values and max(valid_period_values) - min(valid_period_values) > 20:
        warnings.append(f"期间出现次数分布偏差大于 20: {dict(period_counts)}")
    expected_negative = CATEGORY_COUNTS["negative_oos_invalid"]
    if sum(negative_counts.values()) != expected_negative:
        errors.append(f"negative 数量应为 {expected_negative}")

    return {
        "schema_version": "financial_holdout_300_v1",
        "generated_from": {
            "reports_dir": str(reports_dir.relative_to(ROOT)),
            "actual_report_count": len(catalog),
            "gold_60": str(gold_60_path.relative_to(ROOT)),
            "gold_60_case_count": len(old_cases),
            "generator_commit": git_commit(),
        },
        "total_cases": len(cases),
        "category_counts": dict(sorted(category_counts.items())),
        "company_target_counts": {company.code: company_counts[company.code] for company in COMPANIES},
        "period_target_counts": {period: period_counts[period] for period in PERIODS},
        "document_reference_counts": dict(sorted(document_counts.items())),
        "required_document_count_distribution": dict(sorted(target_count_distribution.items())),
        "required_evidence_cell_count_distribution": dict(sorted(evidence_cell_distribution.items())),
        "negative_type_counts": dict(sorted(negative_counts.items())),
        "overlap_audit": {
            "normalized_question_overlap_with_gold60": exact_question_overlap,
            "canonical_signature_overlap_with_gold60": signature_overlap,
            "company_period_metric_overlap_with_gold60": semantic_overlap,
            "near_duplicate_threshold": 0.80,
            "near_duplicate_matches": near_duplicates,
        },
        "baseline_protocol": {
            "final": {"commit": "b50dbcd", "tag": "agent-freeze-v1"},
            "baseline_candidate": {
                "commit": "bcf70ae",
                "before_commit": "002efbf",
                "rationale": "002efbf 引入 deterministic company×period multi-target planning；bcf70ae 为其直接父提交。",
                "execution_status": "prepared_only_not_run",
                "fairness_constraints": [
                    "same frozen 24-document Milvus collection",
                    "same BGE-M3 and reranker",
                    "RETRIEVAL_K=30",
                    "CANDIDATE_M=3",
                    "same retrieval-only evaluator",
                ],
            },
        },
        "validation": {"passed": not errors, "errors": errors, "warnings": warnings},
    }


def _source_documents(docs: Iterable[Any]) -> list[dict[str, str | None]]:
    output: list[dict[str, str | None]] = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        output.append({
            "source_filename": metadata.get("source_filename"),
            "document_id": metadata.get("document_id"),
            "parent_id": metadata.get("parent_id"),
            "company_code": metadata.get("company_code"),
            "report_period": metadata.get("report_period"),
        })
    return output


def _result_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if item.get("status") == "completed"]
    valid = [item for item in completed if item["case"].get("negative_type") is None]
    negative = [item for item in completed if item["case"].get("negative_type") is not None]

    def rate(values: list[bool]) -> float | None:
        return sum(values) / len(values) if values else None

    latencies = [float(item["latency_seconds"]) for item in completed if item.get("latency_seconds") is not None]
    output = {
        "completed": len(completed),
        "total": len(results),
        "company_target_accuracy": rate([item["company_target_accuracy"] for item in valid]),
        "period_target_accuracy": rate([item["period_target_accuracy"] for item in valid]),
        "document_hit_at_3": rate([item["document_hit_at_3"] for item in valid]),
        "required_document_coverage_at_3": statistics.mean([item["required_document_coverage_at_3"] for item in valid]) if valid else None,
        "multi_target_full_coverage_rate_at_3": rate([item["full_document_coverage_at_3"] for item in valid if len(item["case"]["required_documents"]) > 1]),
        "wrong_company_rate": statistics.mean([item["wrong_company_rate"] for item in valid]) if valid else None,
        "wrong_period_rate": statistics.mean([item["wrong_period_rate"] for item in valid]) if valid else None,
        "negative_handling_accuracy": rate([item["negative_handling_correct"] for item in negative]),
        "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
        "p50_latency_seconds": _percentile(latencies, 0.50),
        "p95_latency_seconds": _percentile(latencies, 0.95),
    }
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({item["case"]["category"] for item in completed}):
        rows = [item for item in completed if item["case"]["category"] == category]
        by_category[category] = {
            "n": len(rows),
            "document_hit_at_3": rate([item["document_hit_at_3"] for item in rows if item["case"].get("negative_type") is None]),
            "full_document_coverage_at_3": rate([item["full_document_coverage_at_3"] for item in rows if item["case"].get("negative_type") is None]),
            "negative_handling_accuracy": rate([item["negative_handling_correct"] for item in rows if item["case"].get("negative_type") is not None]),
        }
    output["category_metrics"] = by_category
    return output


def run_retrieval_holdout(cases: list[dict[str, Any]], checkpoint_path: Path) -> dict[str, Any]:
    """Run only after explicit CLI confirmation; never requests a final answer."""
    from new_main import IntegratedQASystem
    from rag_qa.core.query_metadata import extract_query_metadata

    prior: dict[str, dict[str, Any]] = {}
    if checkpoint_path.exists():
        payload = _json_load(checkpoint_path)
        prior = {item["case"]["id"]: item for item in payload.get("results", []) if item.get("status") == "completed"}
    system = IntegratedQASystem()
    results: list[dict[str, Any]] = []
    for case in cases:
        if case["id"] in prior:
            results.append(prior[case["id"]])
            continue
        started = time.perf_counter()
        row: dict[str, Any] = {"case": case, "status": "completed"}
        try:
            route = system.rag.query_router.route(case["question"])
            route_value = getattr(route, "value", str(route))
            docs: list[Any] = []
            strategy = None
            if route_value == "RAG" and case["expected_rag"]:
                query_metadata = extract_query_metadata(case["question"])
                deterministic = query_metadata.requires_deterministic_subqueries()
                strategy = "子查询检索" if deterministic else system.rag.strategy_selector.select_strategy(case["question"])
                docs = system.rag.retrieve_and_merge(
                    case["question"],
                    metadata_filter=query_metadata.to_metadata_filter(),
                    subquery_targets=query_metadata.subquery_plan() if deterministic else None,
                    strategy=strategy,
                    query_metadata=query_metadata,
                )
            top3 = _source_documents(docs[:3])
            expected_documents = set(case["required_documents"])
            returned_documents = {item["source_filename"] for item in top3 if item["source_filename"]}
            expected_codes = set(case["expected_company_codes"])
            expected_periods = set(case["expected_report_periods"])
            actual_codes = {item["company_code"] for item in top3 if item["company_code"]}
            actual_periods = {item["report_period"] for item in top3 if item["report_period"]}
            row.update({
                "router_result": route_value,
                "strategy": strategy,
                "top3": top3,
                "company_target_accuracy": expected_codes <= actual_codes if expected_codes else None,
                "period_target_accuracy": expected_periods <= actual_periods if expected_periods else None,
                "document_hit_at_3": bool(expected_documents & returned_documents) if expected_documents else None,
                "required_document_coverage_at_3": len(expected_documents & returned_documents) / len(expected_documents) if expected_documents else None,
                "full_document_coverage_at_3": expected_documents <= returned_documents if expected_documents else None,
                "wrong_company_rate": len(actual_codes - expected_codes) / len(actual_codes) if expected_codes and actual_codes else 0.0,
                "wrong_period_rate": len(actual_periods - expected_periods) / len(actual_periods) if expected_periods and actual_periods else 0.0,
                "negative_handling_correct": route_value == case["expected_route"] if case["negative_type"] else None,
            })
        except Exception as exc:  # checkpoint preserves a transparent non-result
            row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        row["latency_seconds"] = round(time.perf_counter() - started, 6)
        results.append(row)
        _json_dump(checkpoint_path, {"runtime_config": _runtime_config(), "results": results, "metrics": _result_metrics(results)})
    return {"runtime_config": _runtime_config(), "results": results, "metrics": _result_metrics(results)}


def _runtime_config() -> dict[str, Any]:
    try:
        from base.config import config
        return {
            "git_commit": git_commit(),
            "retrieval_k": config.RETRIEVAL_K,
            "candidate_m": config.CANDIDATE_M,
            "milvus_database": config.MILVUS_DATABASE_NAME,
            "milvus_collection": config.MILVUS_COLLECTION_NAME,
            "mode": "retrieval_only_no_final_llm_no_mcp",
        }
    except Exception as exc:
        return {"git_commit": git_commit(), "mode": "runtime_config_unavailable", "error": type(exc).__name__}


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial 300Q unseen-query holdout utility")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "audit"):
        child = subparsers.add_parser(command)
        child.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
        child.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
        child.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
        child.add_argument("--gold-60", type=Path, default=DEFAULT_GOLD_60)
    run = subparsers.add_parser("run", help="Retrieval-only execution; requires explicit confirmation")
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--checkpoint", type=Path, default=ROOT / "evaluations" / "financial_holdout_300_v1_results.json")
    run.add_argument("--confirm-run", action="store_true", help="Required: permits actual RAG retrieval execution")
    args = parser.parse_args()

    if args.command == "generate":
        cases = generate_dataset(args.reports_dir, args.gold_60)
        audit = audit_dataset(cases, reports_dir=args.reports_dir, gold_60_path=args.gold_60)
        if not audit["validation"]["passed"]:
            raise SystemExit("生成后的 holdout audit 未通过: " + "; ".join(audit["validation"]["errors"]))
        _json_dump(args.dataset, {"schema_version": "financial_holdout_300_v1", "cases": cases})
        _json_dump(args.audit, audit)
        print(f"generated={len(cases)} dataset={args.dataset} audit={args.audit}")
        return
    if args.command == "audit":
        payload = _json_load(args.dataset)
        cases = payload["cases"] if isinstance(payload, dict) else payload
        audit = audit_dataset(cases, reports_dir=args.reports_dir, gold_60_path=args.gold_60)
        _json_dump(args.audit, audit)
        print(json.dumps(audit["validation"], ensure_ascii=False))
        return
    if not args.confirm_run:
        raise SystemExit("安全停止：run 必须显式传入 --confirm-run；本轮只允许生成和审计。")
    payload = _json_load(args.dataset)
    cases = payload["cases"] if isinstance(payload, dict) else payload
    _json_dump(args.checkpoint, run_retrieval_holdout(cases, args.checkpoint))


if __name__ == "__main__":
    main()
