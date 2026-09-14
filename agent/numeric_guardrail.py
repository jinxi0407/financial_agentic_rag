"""Bounded answer-representation checks; no retrieval or financial calculation."""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import re


# Explicit approximate amounts only: 1.62 -> 1.6 is 1.23%; prices and
# percentages never use this tolerance, nor do unqualified exact amounts.
AMOUNT_APPROX_REL_TOLERANCE = Decimal("0.02")
_SCALE = {"元": Decimal(1), "CNY": Decimal(1), "万元": Decimal("1e4"),
          "亿元": Decimal("1e8"), "万亿": Decimal("1e12"), "万亿元": Decimal("1e12")}
_NUMBER = r"[+-]?\d[\d,]*(?:\.\d+)?"
_AMOUNT = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<op>超过|大于|小于|低于|不到|至少|至多|约|近|超)?"
    rf"\s*(?:\*\*)?(?P<value>{_NUMBER})\s*(?:\*\*)?\s*"
    r"(?P<unit>万亿元|万亿|亿元|万元|元|CNY)"
)
_OPS = {None: "eq", "约": "approx", "近": "approx", "超过": "gt", "超": "gt",
        "大于": "gt", "小于": "lt", "低于": "lt", "不到": "lt", "至少": "ge", "至多": "le"}
_DATE = re.compile(r"(?<!\d)(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})日?(?!\d)")
_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?(?!\d)")
_COMPACT_TIME = re.compile(r"(?<!\d)20\d{12}(?!\d)")
_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_])(?:20\d{2}(?:H[12]|FY)|\d{6}(?:\.(?:SH|SZ))?)(?!\d)")


@dataclass(frozen=True)
class NumericEvidence:
    value: Decimal | str
    unit: str
    semantic_type: str
    operator: str = "eq"
    upper: Decimal | None = None
    span: tuple[int, int] = (0, 0)
    source: str = ""


def _amount_type(text, start, default):
    context = re.split(r"[。！？\n；;]", text[:start])[-1][-100:]
    if re.search(r"市值|成交额|成交金额", context):
        return "market_amount"
    if re.search(r"营业收入|营收|净利润|现金流|资产总额|总资产|负债", context):
        return "financial_amount"
    if re.search(r"价格|股价|现价|零售价|最高|最低|涨跌额|今开|昨收|开盘价", context):
        return "price"
    return default


def amount_tokens(text, default_type="other_numeric", source=""):
    tokens = []
    for match in _AMOUNT.finditer(text):
        value = Decimal(match["value"].replace(",", "")) * _SCALE[match["unit"]]
        tokens.append(NumericEvidence(value, "元", _amount_type(text, match.start(), default_type),
                                      _OPS[match["op"]], span=match.span(), source=source))
    return tokens


def temporal_tokens(text):
    tokens = []
    for match in _COMPACT_TIME.finditer(text):
        try:
            value = datetime.strptime(match[0], "%Y%m%d%H%M%S").isoformat()
        except ValueError:
            continue
        tokens.append(NumericEvidence(value, "datetime", "time/date", span=match.span()))
    for match in _DATE.finditer(text):
        try:
            value = datetime(*map(int, match.groups())).date().isoformat()
        except ValueError:
            continue
        tokens.append(NumericEvidence(value, "date", "time/date", span=match.span()))
    for match in _TIME.finditer(text):
        value = f"{int(match[1]):02d}:{match[2]}" + (f":{match[3]}" if match[3] else "")
        tokens.append(NumericEvidence(value, "time", "time/date", span=match.span()))
    return tokens


def _temporal_values(texts):
    values = set()
    for text in texts:
        for token in temporal_tokens(text):
            values.add((token.unit, token.value))
            if token.unit == "datetime":
                date, time = token.value.split("T")
                values.update({("date", date), ("time", time), ("time", time[:5])})
            elif token.unit == "time":
                values.add(("time", token.value[:5]))
    return values


def _evidence(state):
    amounts, texts = [], []
    for result in state.get("tool_results", []):
        if result.get("tool_name") == "financial_rag" and result.get("success"):
            text = result.get("answer", "")
            texts.append(text)
            amounts.extend(amount_tokens(text, "financial_amount", "financial_rag.answer"))
    for result in state.get("news_results", []):
        if not result.get("success"):
            continue
        for item in (result.get("result") or {}).get("results", []):
            for key in ("title", "snippet", "summary", "published_at"):
                text = item.get(key) or ""
                texts.append(text)
                amounts.extend(amount_tokens(text, source=f"news.{key}"))
    for result in state.get("market_results", []):
        if not result.get("success"):
            continue
        payload = result.get("result") or {}
        texts.append(str(payload.get("market_time") or ""))
        for key in ("price", "change", "open", "high", "low", "previous_close", "turnover"):
            value = payload.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
                continue
            value = Decimal(str(value))
            if value.is_finite() and payload.get("currency", "CNY") == "CNY":
                amounts.append(NumericEvidence(value, "元", "market_amount" if key == "turnover" else "price",
                                               source=f"market.{key}"))
    texts.append(state.get("query", ""))
    return amounts, _temporal_values(texts)


def _amount_supported(claim, evidence):
    for item in evidence:
        if claim.semantic_type != "other_numeric" and item.semantic_type != claim.semantic_type:
            continue
        # A lower/upper-bound source is not an exact value for new comparisons.
        if item.operator not in {"eq", "approx"}:
            continue
        actual = item.value
        if claim.operator == "eq" and actual == claim.value:
            return True
        if claim.operator == "approx" and claim.semantic_type in {"financial_amount", "market_amount"}:
            if actual != 0 and abs(actual - claim.value) / abs(actual) <= AMOUNT_APPROX_REL_TOLERANCE:
                return True
        if claim.operator == "gt" and actual > claim.value:
            return True
        if claim.operator == "lt" and actual < claim.value:
            return True
        if claim.operator == "ge" and actual >= claim.value:
            return True
        if claim.operator == "le" and actual <= claim.value:
            return True
        if claim.operator == "between" and claim.value <= actual <= claim.upper:
            return True
    return False


def unsupported_numeric_claims(state, candidate, allowed_numbers, numbers):
    """Return unsupported claims; retain legacy exact checks for other numbers.

    Explicit money is always checked with units, even if its raw mantissa is in
    the legacy whitelist. Percentages keep their exact '%' canonical semantics.
    Temporal spans must match evidence; their components are never financial
    evidence. Identifiers retain the existing exact/query-label policy.
    """
    amounts, temporal_values = _evidence(state)
    claims = amount_tokens(candidate)
    combined = []
    index = 0
    while index < len(claims):
        claim = claims[index]
        if index + 1 < len(claims):
            other = claims[index + 1]
            gap = candidate[claim.span[1]:other.span[0]].strip()
            if gap in {"至", "到", "~", "～"} and claim.operator == other.operator == "eq":
                claim = NumericEvidence(claim.value, "元", claim.semantic_type, "between", other.value,
                                        (claim.span[0], other.span[1]))
                index += 1
        combined.append(claim)
        index += 1
    spans, unsupported = [], []
    for claim in combined:
        spans.append(claim.span)
        if not _amount_supported(claim, amounts):
            unsupported.append(claim)
    for claim in temporal_tokens(candidate):
        if any(start < claim.span[1] and end > claim.span[0] for start, end in spans):
            continue
        spans.append(claim.span)
        if (claim.unit, claim.value) not in temporal_values:
            unsupported.append(claim)
    for match in _IDENTIFIER.finditer(candidate):
        if numbers(match[0]) <= allowed_numbers:
            spans.append(match.span())
    remaining = list(candidate)
    for start, end in spans:
        remaining[start:end] = " " * (end - start)
    for value in numbers("".join(remaining)) - allowed_numbers:
        percent = value.endswith("%")
        unsupported.append(NumericEvidence(value, "%" if percent else "", "percentage" if percent else "other_numeric"))
    return unsupported
