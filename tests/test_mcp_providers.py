"""Focused unit tests for public-provider normalization and safety."""

import unittest

import requests

from mcp_servers.providers.market_provider import MarketProvider, extract_security_from_query, normalize_security
from mcp_servers.providers.news_provider import GoogleNewsProvider, NewsProvider


class FakeResponse:
    def __init__(self, payload=None, content=b""):
        self.payload = payload
        self.content = content
        self.text = content.decode("utf-8")

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def get(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.response


class MarketProviderTests(unittest.TestCase):
    def test_normalizes_supported_alias_and_leading_zero_ticker(self):
        self.assertEqual("000001", normalize_security(ticker="000001")["ticker"])
        self.assertEqual("招商银行", normalize_security(company_name="招行")["company_name"])
        self.assertEqual("600519", extract_security_from_query("今天茅台股价如何？")["ticker"])

    def test_success_schema_uses_provider_values(self):
        payload = {"data": {"f43": 130930, "f44": 132300, "f45": 130905, "f46": 131800, "f47": 17534, "f48": 230.0, "f57": "600519", "f58": "贵州茅台", "f60": 131601, "f124": 0, "f169": -671, "f170": -51}}
        result = MarketProvider(session=FakeSession(FakeResponse(payload))).get_market_snapshot(ticker="600519")
        self.assertTrue(result["success"])
        self.assertEqual(1309.3, result["price"])
        self.assertEqual(-0.51, result["change_percent"])
        self.assertEqual("Eastmoney public quote API", result["source"])

    def test_unknown_ticker_and_provider_exception_are_safe(self):
        self.assertFalse(MarketProvider().get_market_snapshot(ticker="999999")["success"])
        result = MarketProvider(session=FakeSession(error=requests.Timeout())).get_market_snapshot(ticker="600519")
        self.assertFalse(result["success"])
        self.assertIn("Timeout", result["error"])


class NewsProviderTests(unittest.TestCase):
    _RSS = '''<?xml version="1.0"?><rss><channel>
      <item><title>比亚迪新闻</title><link>https://example.com/1</link><description>摘要 1</description><source>示例媒体</source><pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate></item>
      <item><title>比亚迪新闻</title><link>https://example.com/duplicate</link><description>重复</description></item>
      <item><title>第二条新闻</title><link>https://example.com/2</link></item>
    </channel></rss>'''.encode("utf-8")

    def test_result_schema_limit_and_deduplication(self):
        result = NewsProvider(providers=[GoogleNewsProvider(session=FakeSession(FakeResponse(content=self._RSS)))]).search_news("比亚迪", max_results=20)
        self.assertTrue(result["success"])
        self.assertEqual(2, result["result_count"])
        self.assertEqual("比亚迪新闻", result["results"][0]["title"])
        self.assertEqual("示例媒体", result["results"][0]["source"])
        self.assertIsNotNone(result["results"][0]["published_at"])

    def test_provider_failure_is_safe(self):
        result = NewsProvider(providers=[GoogleNewsProvider(session=FakeSession(error=requests.Timeout()))]).search_news("比亚迪")
        self.assertFalse(result["success"])
        self.assertEqual([], result["results"])
        self.assertIn("Timeout", result["error"])

    def test_domestic_primary_failure_uses_fallback(self):
        primary = StubNewsProvider("东方财富搜索", {"success": False, "results": [], "error": "Timeout"})
        fallback = StubNewsProvider("新浪财经7x24", {"success": True, "results": [{"title": "贵州茅台新闻", "source": "新浪财经", "published_at": "2026-01-01T00:00:00+00:00", "url": "https://example.com/news", "provider": "新浪财经7x24"}], "error": None})
        result = NewsProvider(providers=[primary, fallback]).search_news("贵州茅台最近新闻", company_name="贵州茅台")
        self.assertTrue(result["success"])
        self.assertEqual("新浪财经7x24", result["source"])
        self.assertEqual(["东方财富搜索", "新浪财经7x24"], result["providers_attempted"])
        self.assertEqual("Timeout", result["provider_trace"][0]["error"])

    def test_all_provider_failures_return_safe_empty_payload(self):
        providers = [StubNewsProvider("东方财富搜索", {"success": False, "results": [], "error": "Timeout"}), StubNewsProvider("新浪财经7x24", {"success": True, "results": [], "error": None})]
        result = NewsProvider(providers=providers).search_news("中芯国际最新消息", company_name="中芯国际")
        self.assertFalse(result["success"])
        self.assertEqual([], result["results"])
        self.assertEqual(["东方财富搜索", "新浪财经7x24"], result["providers_attempted"])


class StubNewsProvider:
    def __init__(self, name, payload):
        self.name = name
        self.payload = payload

    def search(self, query, company_name, ticker, max_results):
        return self.payload
