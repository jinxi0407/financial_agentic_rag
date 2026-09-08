"""Public A-share market snapshot provider backed by Eastmoney JSON."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import requests


_SECURITIES = {
    "600519": {"company_name": "贵州茅台", "exchange": "SH", "aliases": ("贵州茅台", "茅台")},
    "000858": {"company_name": "五粮液", "exchange": "SZ", "aliases": ("五粮液",)},
    "002594": {"company_name": "比亚迪", "exchange": "SZ", "aliases": ("比亚迪",)},
    "300750": {"company_name": "宁德时代", "exchange": "SZ", "aliases": ("宁德时代",)},
    "600036": {"company_name": "招商银行", "exchange": "SH", "aliases": ("招商银行", "招行")},
    "000001": {"company_name": "平安银行", "exchange": "SZ", "aliases": ("平安银行",)},
    "688981": {"company_name": "中芯国际", "exchange": "SH", "aliases": ("中芯国际", "中芯")},
    "002371": {"company_name": "北方华创", "exchange": "SZ", "aliases": ("北方华创",)},
}
_TICKER_PATTERN = re.compile(r"(?<!\d)(\d{6})(?!\d)")
_EASTMONEY_URL = "https://push2.eastmoney.com/api/qt/stock/get"
_EASTMONEY_FIELDS = "f43,f44,f45,f46,f47,f48,f57,f58,f60,f169,f170,f124"
_TENCENT_URL = "https://qt.gtimg.cn/q="


def normalize_security(company_name: str | None = None, ticker: str | None = None) -> dict[str, str] | None:
    """Resolve only the eight supported A-share securities without coercing ticker to int."""
    candidate = ticker.strip() if isinstance(ticker, str) else None
    if candidate and candidate in _SECURITIES:
        return {"company_name": _SECURITIES[candidate]["company_name"], "ticker": candidate, "exchange": _SECURITIES[candidate]["exchange"]}
    if company_name:
        normalized_name = company_name.strip()
        for code, details in _SECURITIES.items():
            if normalized_name in details["aliases"]:
                return {"company_name": details["company_name"], "ticker": code, "exchange": details["exchange"]}
    return None


def extract_security_from_query(query: str) -> dict[str, str] | None:
    """Find one supported security in a user query deterministically."""
    ticker_match = _TICKER_PATTERN.search(query)
    if ticker_match:
        resolved = normalize_security(ticker=ticker_match.group(1))
        if resolved:
            return resolved
    for code, details in _SECURITIES.items():
        if any(alias in query for alias in details["aliases"]):
            return normalize_security(ticker=code)
    return None


def extract_securities_from_query(query: str) -> list[dict[str, str]]:
    """Return every supported security mentioned in query, preserving mention order."""
    matches = []
    for code, details in _SECURITIES.items():
        positions = [query.find(code), *(query.find(alias) for alias in details["aliases"])]
        position = min((item for item in positions if item >= 0), default=-1)
        if position >= 0:
            matches.append((position, normalize_security(ticker=code)))
    return [security for _, security in sorted(matches, key=lambda item: item[0])]


class MarketProvider:
    """Fetch a best-effort delayed market snapshot without API credentials."""

    source = "Eastmoney public quote API"

    def __init__(self, session: requests.Session | None = None, timeout_seconds: float = 10.0):
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def get_market_snapshot(self, company_name: str | None = None, ticker: str | None = None) -> dict[str, Any]:
        security = normalize_security(company_name=company_name, ticker=ticker)
        if security is None:
            return self._failure("仅支持当前配置的 8 家 A 股公司，请提供有效 company_name 或 ticker。")
        try:
            market_prefix = "1" if security["exchange"] == "SH" else "0"
            response = self.session.get(
                _EASTMONEY_URL,
                params={"secid": f"{market_prefix}.{security['ticker']}", "fields": _EASTMONEY_FIELDS},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json().get("data")
            if not data or data.get("f43") is None:
                return self._failure("数据源未返回可用行情。", security)
            market_time = self._format_epoch(data.get("f124"))
            return {
                **security,
                "price": self._scaled(data.get("f43")),
                "change": self._scaled(data.get("f169")),
                "change_percent": self._scaled(data.get("f170")),
                "open": self._scaled(data.get("f46")),
                "high": self._scaled(data.get("f44")),
                "low": self._scaled(data.get("f45")),
                "previous_close": self._scaled(data.get("f60")),
                "volume": data.get("f47"),
                "volume_unit": "hand",
                "turnover": data.get("f48"),
                "currency": "CNY",
                "market_time": market_time,
                "source": self.source,
                "success": True,
                "error": None,
            }
        except requests.RequestException as exc:
            return self._get_tencent_snapshot(security, f"Eastmoney 请求失败：{type(exc).__name__}")
        except (TypeError, ValueError) as exc:
            return self._failure(f"Eastmoney 返回解析失败：{type(exc).__name__}", security)

    @staticmethod
    def _scaled(value: Any) -> float | None:
        return None if value is None else float(value) / 100

    @staticmethod
    def _format_epoch(value: Any) -> str | None:
        if not value:
            return None
        return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()

    def _get_tencent_snapshot(self, security: dict[str, str], upstream_error: str) -> dict[str, Any]:
        """Use one public fallback only when the primary provider is unreachable."""
        prefix = "sh" if security["exchange"] == "SH" else "sz"
        try:
            response = self.session.get(
                f"{_TENCENT_URL}{prefix}{security['ticker']}",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            raw_value = response.content.decode("gbk", errors="replace")
            values = raw_value.split('="', 1)[-1].rstrip('";\n').split("~")
            if len(values) < 35 or not values[3]:
                return self._failure(f"{upstream_error}；腾讯财经未返回可用行情。", security)
            return {
                **security,
                "price": self._to_float(values[3]),
                "change": self._to_float(values[31]),
                "change_percent": self._to_float(values[32]),
                "open": self._to_float(values[5]),
                "high": self._to_float(values[33]),
                "low": self._to_float(values[34]),
                "previous_close": self._to_float(values[4]),
                "volume": self._to_float(values[6]),
                "volume_unit": "hand",
                "turnover": None,
                "currency": "CNY",
                "market_time": values[30] or None,
                "source": "Tencent Finance public quote API (fallback)",
                "success": True,
                "error": None,
            }
        except requests.RequestException as exc:
            return self._failure(f"{upstream_error}；腾讯财经请求失败：{type(exc).__name__}", security)
        except (IndexError, TypeError, ValueError) as exc:
            return self._failure(f"{upstream_error}；腾讯财经返回解析失败：{type(exc).__name__}", security)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        return None if value in (None, "") else float(value)

    def _failure(self, error: str, security: dict[str, str] | None = None) -> dict[str, Any]:
        return {
            **(security or {}),
            "price": None,
            "change": None,
            "change_percent": None,
            "open": None,
            "high": None,
            "low": None,
            "previous_close": None,
            "volume": None,
            "turnover": None,
            "currency": "CNY",
            "market_time": None,
            "source": self.source,
            "success": False,
            "error": error,
        }
