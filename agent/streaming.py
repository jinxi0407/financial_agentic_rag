"""Public-safe streaming adapter for the synchronous LangGraph Agent."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Iterator

from starlette.websockets import WebSocketDisconnect


class AgentStreamRequest:
    """Validated WebSocket request.  This contract intentionally stays small."""

    def __init__(self, query: str, thread_id: str | None = None, user_id: str | None = None):
        self.query = query
        self.thread_id = thread_id
        self.user_id = user_id


def validate_stream_request(payload: Any) -> AgentStreamRequest:
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    thread_id = payload.get("thread_id")
    user_id = payload.get("user_id")
    if thread_id is not None and (not isinstance(thread_id, str) or not thread_id.strip()):
        raise ValueError("thread_id must be a non-empty string when provided")
    if user_id is not None and (not isinstance(user_id, str) or not user_id.strip()):
        raise ValueError("user_id must be a non-empty string when provided")
    return AgentStreamRequest(query.strip(), thread_id, user_id)


class AgentStreamingAdapter:
    """Map internal lifecycle notifications to a stable browser event protocol."""

    def __init__(self, agent: Any, chunk_size: int = 32):
        self.agent = agent
        self.chunk_size = chunk_size

    def iter_events(self, payload: Any) -> Iterator[dict[str, Any]]:
        request = validate_stream_request(payload)
        thread_id = request.thread_id or str(uuid.uuid4())
        user_id = request.user_id or str(uuid.uuid4())
        yield self._event("start", {"thread_id": thread_id, "user_id": user_id, "query": request.query})

        guardrail_seen = False
        for internal in self.agent.stream(request.query, thread_id=thread_id, user_id=user_id):
            kind = internal.get("kind")
            data = internal.get("data", {})
            if kind == "plan":
                yield self._event("plan", {
                    "intent": data.get("intent"),
                    "required_tools": list(data.get("required_tools", [])),
                    "skill": data.get("skill"),
                })
            elif kind == "tool_start":
                yield self._event("tool_start", {"tool": data.get("tool")})
            elif kind == "tool_end":
                yield self._tool_end_event(data)
            elif kind == "synthesis_start":
                yield self._event("synthesis_start", {})
            elif kind == "guardrail":
                guardrail_seen = True
            elif kind == "complete":
                state = internal.get("state", {})
                status = state.get("guardrail_status")
                if not guardrail_seen or not isinstance(status, str) or not status.strip():
                    yield self._event("error", {
                        "code": "AGENT_EXECUTION_ERROR",
                        "message": "Agent request failed.",
                    })
                    return
                yield self._event("guardrail", {"status": self._public_guardrail_status(status)})
                yield self._event("sources", self._sources(state))
                yield self._event("trace", self._trace(state))
                for chunk in self._chunks(str(state.get("final_answer", ""))):
                    yield self._event("token", {"content": chunk})
                yield self._event("end", {"success": bool(state.get("final_answer"))})
            elif kind == "error":
                yield self._event("error", {
                    "code": "AGENT_EXECUTION_ERROR",
                    "message": "Agent request failed.",
                })
                return

    @staticmethod
    def _public_guardrail_status(status: str) -> str:
        """Keep the browser enum; only final, explicit PASS means passed."""
        if status == "passed":
            return "passed"
        flags = set(status.split(","))
        if flags <= {"degraded", "market_unavailable", "news_unavailable", "news_provenance_missing"}:
            return "degraded"
        return "error"

    async def events(self, payload: Any) -> AsyncIterator[dict[str, Any]]:
        iterator = self.iter_events(payload)
        while True:
            has_event, event = await asyncio.to_thread(self._next_or_done, iterator)
            if not has_event:
                return
            yield event

    @staticmethod
    def _next_or_done(iterator: Iterator[dict[str, Any]]) -> tuple[bool, dict[str, Any] | None]:
        try:
            return True, next(iterator)
        except StopIteration:
            return False, None

    def _tool_end_event(self, data: dict[str, Any]) -> dict[str, Any]:
        tool = data.get("tool")
        results = list(data.get("results", []))
        success = bool(results) and all(item.get("success") for item in results)
        payload: dict[str, Any] = {
            "tool": tool,
            "success": success,
            "latency": round(sum(float(item.get("latency", 0.0)) for item in results), 4),
            "summary": self._tool_summary(tool, results, success),
        }
        if not success:
            payload["error_type"] = self._error_type(results)
        return self._event("tool_end", payload)

    @staticmethod
    def _tool_summary(tool: str | None, results: list[dict[str, Any]], success: bool) -> str:
        if not success:
            return "工具未返回可展示结果。"
        if tool == "financial_rag":
            return "Financial RAG 已完成。"
        if tool == "market_mcp":
            return f"已获取 {len(results)} 个标的的行情结果。"
        if tool == "news_mcp":
            count = sum(len((item.get("result") or {}).get("results", [])) for item in results)
            return f"已获取 {count} 条可展示新闻。"
        if tool == "calculator":
            return "确定性计算已完成。"
        return "工具已完成。"

    @staticmethod
    def _error_type(results: list[dict[str, Any]]) -> str:
        errors = " ".join(str(item.get("error", "")) for item in results).lower()
        if "timeout" in errors or "超时" in errors:
            return "provider_timeout"
        return "tool_execution_error"

    @staticmethod
    def _sources(state: dict[str, Any]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for result in state.get("market_results", []):
            payload = result.get("result") or {}
            if payload.get("source"):
                sources.append({
                    "tool": "market_mcp",
                    "provider": payload.get("provider") or payload.get("source"),
                    "symbol": payload.get("ticker"),
                    "timestamp": payload.get("market_time"),
                    "source": payload.get("source"),
                })
        for result in state.get("news_results", []):
            for item in (result.get("result") or {}).get("results", []):
                if item.get("title") and item.get("source") and item.get("url"):
                    sources.append({
                        "tool": "news_mcp",
                        "title": item.get("title"),
                        "source": item.get("source"),
                        "published_at": item.get("published_at"),
                        "url": item.get("url"),
                        "provider": item.get("provider"),
                    })
        # FinancialRAGTool currently exposes an answer only.  Do not invent
        # document/page citations until its frozen result contract supplies them.
        return sources

    @staticmethod
    def _trace(state: dict[str, Any]) -> dict[str, Any]:
        trace = state.get("trace", {})
        return {
            "thread_id": trace.get("thread_id") or state.get("thread_id"),
            "intent": state.get("intent"),
            "skill": state.get("skill"),
            "planned_tools": list(state.get("required_tools", [])),
            "executed_tools": list(state.get("executed_tools", [])),
            "tool_latencies": dict(trace.get("per_tool_latency", {})),
            "total_latency": trace.get("total_latency"),
        }

    def _chunks(self, answer: str) -> Iterator[str]:
        for offset in range(0, len(answer), self.chunk_size):
            yield answer[offset:offset + self.chunk_size]

    @staticmethod
    def _event(event_type: str, data: Any) -> dict[str, Any]:
        return {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


async def serve_agent_websocket(
    websocket: Any,
    adapter_factory: Callable[[], AgentStreamingAdapter],
    log: Callable[[str], None] | None = None,
) -> None:
    """Serve multiple requests per socket without exposing internal errors."""
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            try:
                adapter = adapter_factory()
                async for event in adapter.events(payload):
                    await websocket.send_json(event)
            except ValueError:
                await websocket.send_json(AgentStreamingAdapter._event("error", {
                    "code": "INVALID_REQUEST",
                    "message": "query must be a non-empty string.",
                }))
            except WebSocketDisconnect:
                return
            except Exception:
                if log:
                    log("agent_stream_request_failed")
                await websocket.send_json(AgentStreamingAdapter._event("error", {
                    "code": "AGENT_EXECUTION_ERROR",
                    "message": "Agent request failed.",
                }))
    except WebSocketDisconnect:
        if log:
            log("agent_stream_disconnected")
