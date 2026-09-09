"""Credential-free financial news providers with domestic-first fallback."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import quote_plus
from xml.etree import ElementTree

import requests


logger = logging.getLogger(__name__)
_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://so.eastmoney.com/"}


class NewsSourceProvider(Protocol):
    name: str

    def search(self, query: str, company_name: str | None, ticker: str | None, max_results: int) -> dict[str, Any]: ...


class EastmoneyNewsProvider:
    """Search the public Eastmoney company-news JSONP endpoint."""

    name = "东方财富搜索"
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    _GENERIC_MARKET_TITLE_TERMS = (
        "主力资金",
        "资金净流",
        "资金流",
        "概念",
        "板块",
        "大宗交易",
        "杠杆资金",
        "净买入",
        "净流出",
        "排行榜",
    )
    _COMPANY_EVENT_TITLE_TERMS = (
        "公告", "披露", "发布", "中报", "年报", "财报", "业绩", "净利润", "营业收入", "营收",
        "项目", "产品", "技术", "工艺", "投资", "融资", "回购", "股东", "董事", "经营", "合作",
        "签约", "订单", "销量", "诉讼", "处罚",
    )
    _MIN_RELEVANCE_SCORE = 4

    def __init__(self, session: requests.Session | None = None, timeout_seconds: float = 3.0):
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, company_name: str | None, ticker: str | None, max_results: int) -> dict[str, Any]:
        keyword = ticker or company_name or query
        candidate_limit = min(max(max_results * 5, 20), 50)
        payload = {
            "uid": "", "keyword": keyword, "type": ["cmsArticleWebOld"],
            "client": "web", "clientType": "web", "clientVersion": "curr",
            "param": {"cmsArticleWebOld": {"searchScope": "default", "sort": "default", "pageIndex": 1, "pageSize": candidate_limit, "preTag": "", "postTag": ""}},
        }
        try:
            response = self.session.get(self.url, params={"cb": "financialNews", "param": json.dumps(payload, separators=(",", ":"))}, headers=_HEADERS, timeout=self.timeout_seconds)
            response.raise_for_status()
            match = re.fullmatch(r"\s*financialNews\((.*)\)\s*", response.text, re.DOTALL)
            if not match:
                return self._failure("invalid_jsonp")
            data = json.loads(match.group(1))
            articles = (data.get("result") or {}).get("cmsArticleWebOld") or []
            if isinstance(articles, dict):
                articles = articles.get("list") or []
            scored = []
            for index, article in enumerate(articles):
                item = self._item(article)
                if not item:
                    continue
                score = self._relevance_score(item, company_name, ticker)
                if score >= self._MIN_RELEVANCE_SCORE:
                    scored.append((score, index, item))
            scored.sort(key=lambda entry: (-entry[0], entry[1]))
            results = [item for _, _, item in scored[:max_results]]
            return {"success": True, "results": results, "error": None}
        except (requests.RequestException, ValueError, TypeError) as exc:
            return self._failure(type(exc).__name__)

    def _item(self, article: dict[str, Any]) -> dict[str, str | None] | None:
        title = _clean(article.get("title"))
        url = str(article.get("url") or "").strip()
        if not title or not url:
            return None
        return {"title": title, "snippet": _clean(article.get("content")) or None, "source": _clean(article.get("mediaName")) or self.name, "published_at": str(article.get("date") or "").strip() or None, "url": url, "provider": self.name}

    @classmethod
    def _relevance_score(cls, item: dict[str, str | None], company_name: str | None, ticker: str | None) -> int:
        """Favor company-subject coverage over ticker-only market lists."""
        title = item["title"] or ""
        snippet = item["snippet"] or ""
        if cls._is_generic_market_headline(title) and not any(term in title for term in cls._COMPANY_EVENT_TITLE_TERMS):
            return -100
        score = 0
        company_in_title = bool(company_name and company_name in title)
        if company_in_title:
            score += 12
        elif company_name and company_name in snippet:
            score += 6
            if cls._snippet_has_company_subject(snippet, company_name):
                score += 3
        if ticker and ticker in title:
            score += 4
        elif ticker and ticker in snippet:
            score += 1
        if not company_in_title and cls._is_generic_market_headline(title):
            score -= 5
        return score

    @classmethod
    def _is_generic_market_headline(cls, title: str) -> bool:
        return any(term in title for term in cls._GENERIC_MARKET_TITLE_TERMS)

    @staticmethod
    def _snippet_has_company_subject(snippet: str, company_name: str) -> bool:
        leading_text = snippet[:100].lstrip()
        return leading_text.startswith(company_name) or bool(re.search(rf"(?:^|[。；])[^。；]{{0,20}}{re.escape(company_name)}(?:[（(：,，]|$)", leading_text))

    @staticmethod
    def _failure(reason: str) -> dict[str, Any]:
        return {"success": False, "results": [], "error": reason}


class SinaFinanceNewsProvider:
    """Fallback over Sina Finance's public 7x24 feed, filtered locally."""

    name = "新浪财经7x24"
    url = "https://feed.mix.sina.com.cn/api/roll/get"

    def __init__(self, session: requests.Session | None = None, timeout_seconds: float = 3.0):
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, company_name: str | None, ticker: str | None, max_results: int) -> dict[str, Any]:
        needles = [item for item in (company_name, ticker) if item]
        if not needles:
            return {"success": True, "results": [], "error": None}
        try:
            response = self.session.get(self.url, params={"pageid": "153", "lid": "2516", "k": "", "num": "50", "page": "1"}, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}, timeout=self.timeout_seconds)
            response.raise_for_status()
            items = ((response.json().get("result") or {}).get("data") or [])
            results = []
            for item in items:
                text = " ".join(str(item.get(key) or "") for key in ("title", "summary", "intro"))
                if any(needle in text for needle in needles) and (normalized := self._item(item)):
                    results.append(normalized)
                if len(results) >= max_results:
                    break
            return {"success": True, "results": results, "error": None}
        except (requests.RequestException, ValueError, TypeError) as exc:
            return {"success": False, "results": [], "error": type(exc).__name__}

    def _item(self, item: dict[str, Any]) -> dict[str, str | None] | None:
        title = _clean(item.get("title"))
        url = str(item.get("url") or "").strip()
        if not title or not url:
            return None
        return {"title": title, "snippet": _clean(item.get("summary") or item.get("intro")) or None, "source": _clean(item.get("media_name")) or self.name, "published_at": self._published_at(item.get("ctime") or item.get("intime")), "url": url, "provider": self.name}

    @staticmethod
    def _published_at(value: Any) -> str | None:
        try:
            return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            return None


class GoogleNewsProvider:
    """Optional Google RSS fallback; never the default in domestic mode."""

    name = "Google News RSS"

    def __init__(self, session: requests.Session | None = None, timeout_seconds: float = 3.0):
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, company_name: str | None, ticker: str | None, max_results: int) -> dict[str, Any]:
        search_query = " ".join(part for part in (company_name, ticker, query.strip()) if part)
        url = f"https://news.google.com/rss/search?q={quote_plus(search_query)}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
        try:
            response = self.session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=self.timeout_seconds)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
            results = []
            for item in root.findall("./channel/item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                if not title or not link:
                    continue
                source = item.find("source")
                results.append({"title": title, "snippet": (item.findtext("description") or "").strip() or None, "source": source.text.strip() if source is not None and source.text else self.name, "published_at": self._published_at(item.findtext("pubDate")), "url": link, "provider": self.name})
                if len(results) >= max_results:
                    break
            return {"success": True, "results": results, "error": None}
        except requests.RequestException as exc:
            return {"success": False, "results": [], "error": type(exc).__name__}
        except (ElementTree.ParseError, TypeError, ValueError) as exc:
            return {"success": False, "results": [], "error": type(exc).__name__}

    @staticmethod
    def _published_at(value: str | None) -> str | None:
        try:
            return parsedate_to_datetime(value).astimezone(timezone.utc).isoformat() if value else None
        except (TypeError, ValueError, IndexError):
            return None


class NewsProvider:
    """Domestic-first orchestration retaining the original MCP result contract."""

    def __init__(self, *, mode: str | None = None, session: requests.Session | None = None, timeout_seconds: float | None = None, providers: list[NewsSourceProvider] | None = None):
        self.mode = (mode or os.getenv("NEWS_PROVIDER_MODE", "domestic")).strip().lower()
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else float(os.getenv("NEWS_PROVIDER_TIMEOUT_SECONDS", "3"))
        self.providers = providers or self._providers(session)

    def _providers(self, session: requests.Session | None) -> list[NewsSourceProvider]:
        domestic: list[NewsSourceProvider] = [EastmoneyNewsProvider(session, self.timeout_seconds), SinaFinanceNewsProvider(session, self.timeout_seconds)]
        if self.mode == "google":
            return [GoogleNewsProvider(session, self.timeout_seconds)]
        if self.mode in {"domestic_google", "auto"}:
            return [*domestic, GoogleNewsProvider(session, self.timeout_seconds)]
        return domestic

    def search_news(self, query: str, company_name: str | None = None, ticker: str | None = None, max_results: int = 5) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            return self._failure(query, "query 不能为空。", [])
        try:
            limit = min(max(int(max_results), 1), 10)
        except (TypeError, ValueError):
            return self._failure(query, "max_results 必须是整数。", [])
        attempts: list[dict[str, Any]] = []
        collected: list[dict[str, Any]] = []
        for source in self.providers:
            started = perf_counter()
            payload = source.search(query, company_name, ticker, limit)
            received = self._deduplicate([*collected, *payload.get("results", [])], limit)
            attempt = {"provider": source.name, "success": bool(payload.get("success")), "result_count": len(payload.get("results", [])), "error": payload.get("error"), "latency_seconds": round(perf_counter() - started, 6)}
            attempts.append(attempt)
            logger.info("news_provider provider=%s success=%s result_count=%s error=%s", source.name, attempt["success"], attempt["result_count"], attempt["error"])
            if payload.get("success") and received:
                return {"query": query, "results": received, "result_count": len(received), "source": source.name, "success": True, "error": None, "providers_attempted": [item["provider"] for item in attempts], "provider_trace": attempts}
            if payload.get("success"):
                attempt["error"] = "no_matching_results"
                logger.info("news_provider fallback_from=%s reason=no_matching_results", source.name)
            else:
                logger.warning("news_provider fallback_from=%s reason=%s", source.name, payload.get("error"))
        error = "; ".join(f"{item['provider']}: {item.get('error') or 'no_results'}" for item in attempts)
        return self._failure(query, f"新闻源未返回可靠结果：{error}", attempts)

    @staticmethod
    def _deduplicate(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        results = []
        seen_urls: set[str] = set()
        seen_titles: list[str] = []
        for item in items:
            title = _clean(item.get("title"))
            url = str(item.get("url") or "").strip()
            if not title or not url or url in seen_urls:
                continue
            normalized = re.sub(r"\W+", "", title.lower())
            if any(normalized == prior or SequenceMatcher(None, normalized, prior).ratio() >= 0.92 for prior in seen_titles):
                continue
            seen_urls.add(url)
            seen_titles.append(normalized)
            results.append(item)
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _failure(query: str, error: str, attempts: list[dict[str, Any]]) -> dict[str, Any]:
        return {"query": query, "results": [], "result_count": 0, "source": "Domestic news fallback", "success": False, "error": error, "providers_attempted": [item["provider"] for item in attempts], "provider_trace": attempts}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", str(value or ""))).strip()
