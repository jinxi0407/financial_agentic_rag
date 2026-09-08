"""Minimal tool-oriented entry point for the frozen Financial RAG runtime."""

from .runner import FinancialAgentRunner


_default_runner = None


def run(query: str):
    """Run one Agent MVP request with a lazily initialized Financial RAG tool."""
    global _default_runner
    if _default_runner is None:
        _default_runner = FinancialAgentRunner()
    return _default_runner.run(query)


__all__ = ["FinancialAgentRunner", "run"]
