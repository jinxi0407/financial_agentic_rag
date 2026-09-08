"""Adapter around the frozen public IntegratedQASystem query contract."""

from __future__ import annotations

import time

from agent.schemas import FinancialRAGToolResult


class FinancialRAGTool:
    tool_name = "financial_rag"

    def __init__(self, qa_system=None, qa_system_factory=None):
        self._qa_system = qa_system
        self._qa_system_factory = qa_system_factory

    def _system(self):
        if self._qa_system is None:
            if self._qa_system_factory is None:
                # Import lazily so planner-only tests never initialize models,
                # database clients, Redis, or the frozen RAG runtime.
                from new_main import IntegratedQASystem

                self._qa_system_factory = IntegratedQASystem
            self._qa_system = self._qa_system_factory()
        return self._qa_system

    def run(self, query: str) -> FinancialRAGToolResult:
        started_at = time.perf_counter()
        try:
            answer_parts = []
            for token, _is_complete in self._system().query(query):
                if token:
                    answer_parts.append(token)
            answer = "".join(answer_parts)
            if not answer:
                raise RuntimeError("Financial RAG returned an empty answer")
            return FinancialRAGToolResult(
                tool_name=self.tool_name,
                query=query,
                answer=answer,
                success=True,
                error=None,
                latency=time.perf_counter() - started_at,
            )
        except Exception as exc:
            return FinancialRAGToolResult(
                tool_name=self.tool_name,
                query=query,
                answer="当前 Financial RAG 工具暂时无法完成查询。",
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                latency=time.perf_counter() - started_at,
            )
