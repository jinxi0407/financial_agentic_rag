"""Public financial-news search provider backed by Google News RSS."""

from __future__ import annotations

from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus
from xml.etree import ElementTree

import requests


class NewsProvider:
    """Fetch real news metadata without credentials; no publication time is inferred."""

    source = "Google News RSS"

    def __init__(self, session: requests.Session | None = None, timeout_seconds: float = 10.0):
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def search_news(
        self,
        query: str,
        company_name: str | None = None,
        ticker: str | None = None,
        max_results: int = 5,
    ) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            return self._failure(query, "query 不能为空。")
        try:
            limited_results = min(max(int(max_results), 1), 10)
        except (TypeError, ValueError):
            return self._failure(query, "max_results 必须是整数。")
        search_query = " ".join(part for part in (company_name, ticker, query.strip()) if part)
        url = f"https://news.google.com/rss/search?q={quote_plus(search_query)}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
        try:
            response = self.session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=self.timeout_seconds)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
            results = self._deduplicated_items(root, limited_results)
            return {
                "query": query,
                "results": results,
                "result_count": len(results),
                "source": self.source,
                "success": True,
                "error": None,
            }
        except requests.RequestException as exc:
            return self._failure(query, f"Google News 请求失败：{type(exc).__name__}")
        except (ElementTree.ParseError, TypeError, ValueError) as exc:
            return self._failure(query, f"Google News 返回解析失败：{type(exc).__name__}")

    def _deduplicated_items(self, root: ElementTree.Element, max_results: int) -> list[dict[str, str | None]]:
        results = []
        seen_titles = set()
        for item in root.findall("./channel/item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            if not title or not link:
                continue
            dedupe_key = " ".join(title.lower().split())
            if dedupe_key in seen_titles:
                continue
            seen_titles.add(dedupe_key)
            source = item.find("source")
            results.append(
                {
                    "title": title,
                    "snippet": (item.findtext("description") or "").strip() or None,
                    "source": source.text.strip() if source is not None and source.text else None,
                    "published_at": self._published_at(item.findtext("pubDate")),
                    "url": link,
                }
            )
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def _published_at(value: str | None) -> str | None:
        if not value:
            return None
        try:
            return parsedate_to_datetime(value).astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError, IndexError):
            return None

    def _failure(self, query: str, error: str) -> dict[str, Any]:
        return {"query": query, "results": [], "result_count": 0, "source": self.source, "success": False, "error": error}
