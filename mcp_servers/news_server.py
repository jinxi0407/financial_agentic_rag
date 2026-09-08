"""Expose the public news provider through the official MCP stdio protocol."""

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

from .providers.news_provider import NewsProvider


server = MCPServer(
    name="financial-news-mcp",
    description="Search public financial news metadata without API credentials.",
)
provider = NewsProvider()


class NewsItem(BaseModel):
    title: str
    snippet: str | None = None
    source: str | None = None
    published_at: str | None = None
    url: str


class NewsSearchResult(BaseModel):
    query: str
    results: list[NewsItem]
    result_count: int
    source: str
    success: bool
    error: str | None = None


@server.tool(name="search_financial_news", structured_output=True)
def search_financial_news(
    query: str,
    company_name: str | None = None,
    ticker: str | None = None,
    max_results: int = 5,
) -> NewsSearchResult:
    """Return at most ten deduplicated public news records."""
    return NewsSearchResult(
        **provider.search_news(query=query, company_name=company_name, ticker=ticker, max_results=max_results)
    )


if __name__ == "__main__":
    server.run(transport="stdio")
