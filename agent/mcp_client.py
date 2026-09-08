"""Synchronous wrapper around official MCP v2 stdio client sessions."""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .schemas import MCPToolResult


_SERVER_MODULES = {
    "market": "mcp_servers.market_server",
    "news": "mcp_servers.news_server",
}
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FinancialMCPClient:
    """Start local MCP server processes and communicate only through stdio MCP."""

    def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        return asyncio.run(self._list_tools(server_name))

    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        started_at = perf_counter()
        try:
            payload = asyncio.run(self._call_tool(server_name, tool_name, arguments))
            success = bool(payload.get("success", True))
            return MCPToolResult(
                tool_name=tool_name,
                server_name=server_name,
                inputs=arguments,
                result=payload,
                success=success,
                error=payload.get("error"),
                latency=perf_counter() - started_at,
            )
        except Exception as exc:
            return MCPToolResult(
                tool_name=tool_name,
                server_name=server_name,
                inputs=arguments,
                result=None,
                success=False,
                error=f"MCP 调用失败：{type(exc).__name__}: {exc}",
                latency=perf_counter() - started_at,
            )

    async def _list_tools(self, server_name: str) -> list[dict[str, Any]]:
        async with self._session(server_name) as session:
            result = await session.list_tools()
            return [tool.model_dump(mode="json") for tool in result.tools]

    async def _call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with self._session(server_name) as session:
            result = await session.call_tool(tool_name, arguments)
            structured = getattr(result, "structured_content", None)
            if isinstance(structured, dict):
                return structured
            if getattr(result, "is_error", False):
                raise RuntimeError(str(getattr(result, "content", "MCP tool error")))
            raise RuntimeError("MCP tool 未返回结构化结果。")

    def _server_parameters(self, server_name: str) -> StdioServerParameters:
        try:
            module = _SERVER_MODULES[server_name]
        except KeyError as exc:
            raise ValueError(f"未知 MCP server：{server_name}") from exc
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", module],
            cwd=_PROJECT_ROOT,
            env=dict(os.environ),
        )

    @asynccontextmanager
    async def _session(self, server_name: str):
        """Use the SDK's documented nested context-manager lifecycle."""
        async with stdio_client(self._server_parameters(server_name)) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session
