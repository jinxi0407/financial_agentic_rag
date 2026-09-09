"""Focused tests for the public-safe Agent streaming adapter."""

from __future__ import annotations

import asyncio
import unittest

from starlette.websockets import WebSocketDisconnect

from agent.langgraph_agent import LangGraphFinancialAgent
from agent.streaming import AgentStreamingAdapter, serve_agent_websocket, validate_stream_request


def _state(answer="安全最终回答", guardrail="passed"):
    return {
        "thread_id": "thread-a",
        "intent": "composite_query",
        "skill": "company_comparison",
        "required_tools": ["financial_rag", "market_mcp", "news_mcp"],
        "executed_tools": ["financial_rag", "market_mcp", "news_mcp"],
        "final_answer": answer,
        "guardrail_status": guardrail,
        "market_results": [{
            "success": True,
            "latency": 0.2,
            "result": {"ticker": "600519", "source": "Eastmoney", "market_time": "2026-01-01"},
        }],
        "news_results": [{
            "success": True,
            "latency": 0.3,
            "result": {"results": [{
                "title": "公司新闻", "source": "Sina Finance", "published_at": "2026-01-01",
                "url": "https://example.test/news", "provider": "sina",
            }]},
        }],
        "trace": {
            "thread_id": "thread-a",
            "per_tool_latency": {"financial_rag": 1.0, "market_mcp": 0.2, "news_mcp": 0.3},
            "total_latency": 1.5,
            "raw_prompt": "must_not_be_exposed",
            "api_key": "must_not_be_exposed",
        },
    }


class FakeAgent:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def stream(self, query, thread_id, user_id):
        self.calls.append((query, thread_id, user_id))
        yield from self.events


class DisconnectingWebSocket:
    def __init__(self):
        self.accepted = False
        self.sent = []

    async def accept(self):
        self.accepted = True

    async def receive_json(self):
        raise WebSocketDisconnect(code=1000)

    async def send_json(self, event):
        self.sent.append(event)


class FakePreferenceStore:
    def get(self, _user_id):
        return {"preferred_companies": [], "preferred_metrics": []}

    def save_explicit(self, _user_id, _query, _companies):
        return None


class AgentStreamingTests(unittest.TestCase):
    def _adapter(self, events):
        return AgentStreamingAdapter(FakeAgent(events), chunk_size=4)

    def _complete_events(self, state=None):
        return [
            {"kind": "plan", "data": {"intent": "composite_query", "required_tools": ["financial_rag", "market_mcp"], "skill": "company_comparison"}},
            {"kind": "tool_start", "data": {"tool": "financial_rag"}},
            {"kind": "tool_end", "data": {"tool": "financial_rag", "results": [{"success": True, "latency": 0.4}]}},
            {"kind": "synthesis_start", "data": {}},
            {"kind": "guardrail", "data": {"status": "passed"}},
            {"kind": "complete", "state": state or _state()},
        ]

    def test_request_validation(self):
        self.assertEqual("问题", validate_stream_request({"query": " 问题 "}).query)
        with self.assertRaises(ValueError):
            validate_stream_request({"query": " "})
        with self.assertRaises(ValueError):
            validate_stream_request({"query": "问题", "thread_id": 1})

    def test_plan_tool_order_guardrail_and_safe_chunks(self):
        events = list(self._adapter(self._complete_events()).iter_events({"query": "测试", "thread_id": "t", "user_id": "u"}))
        kinds = [event["type"] for event in events]
        self.assertEqual("start", kinds[0])
        self.assertLess(kinds.index("plan"), kinds.index("tool_start"))
        self.assertLess(kinds.index("tool_start"), kinds.index("tool_end"))
        self.assertLess(kinds.index("guardrail"), kinds.index("token"))
        self.assertEqual("end", kinds[-1])
        self.assertEqual(["financial_rag", "market_mcp"], events[kinds.index("plan")]["data"]["required_tools"])
        self.assertTrue(all(set(event) == {"type", "data", "timestamp"} for event in events))

    def test_failed_tool_is_redacted_and_classified(self):
        events = [
            {"kind": "plan", "data": {"intent": "news_query", "required_tools": ["news_mcp"], "skill": "market_intelligence"}},
            {"kind": "tool_start", "data": {"tool": "news_mcp"}},
            {"kind": "tool_end", "data": {"tool": "news_mcp", "results": [{"success": False, "latency": 1.2, "error": "ConnectTimeout: internal detail"}]}},
            {"kind": "guardrail", "data": {"status": "degraded"}},
            {"kind": "complete", "state": _state("新闻暂不可用", "news_unavailable")},
        ]
        output = list(self._adapter(events).iter_events({"query": "新闻"}))
        tool_end = next(event for event in output if event["type"] == "tool_end")
        self.assertFalse(tool_end["data"]["success"])
        self.assertEqual("provider_timeout", tool_end["data"]["error_type"])
        self.assertNotIn("ConnectTimeout", str(tool_end))

    def test_sources_and_trace_are_public_safe(self):
        output = list(self._adapter(self._complete_events()).iter_events({"query": "测试"}))
        sources = next(event["data"] for event in output if event["type"] == "sources")
        trace = next(event["data"] for event in output if event["type"] == "trace")
        self.assertEqual("Sina Finance", sources[1]["source"])
        self.assertEqual("https://example.test/news", sources[1]["url"])
        self.assertNotIn("raw_prompt", trace)
        self.assertNotIn("api_key", trace)
        self.assertNotIn("tool_results", trace)

    def test_missing_guardrail_never_exposes_answer(self):
        events = [{"kind": "complete", "state": _state()}]
        output = list(self._adapter(events).iter_events({"query": "测试"}))
        self.assertEqual(["start", "error"], [event["type"] for event in output])

    def test_async_events_and_disconnect_are_handled(self):
        async def run():
            adapter = self._adapter(self._complete_events())
            observed = [event async for event in adapter.events({"query": "测试"})]
            socket = DisconnectingWebSocket()
            await serve_agent_websocket(socket, lambda: adapter)
            return observed, socket

        observed, socket = asyncio.run(run())
        self.assertTrue(observed)
        self.assertTrue(socket.accepted)
        self.assertEqual([], socket.sent)

    def test_real_agent_stream_observes_sync_calculator_node(self):
        agent = LangGraphFinancialAgent(preference_store=FakePreferenceStore())
        events = list(agent.stream("从80增长到100，增长率是多少？", "stream-calculator"))
        kinds = [event["kind"] for event in events]
        self.assertEqual("plan", kinds[0])
        self.assertEqual(["plan", "tool_start", "tool_end", "guardrail", "complete"], kinds)
        self.assertEqual("calculator", events[1]["data"]["tool"])
