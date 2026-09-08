"""Expose the public market provider through the official MCP stdio protocol."""

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

from .providers.market_provider import MarketProvider


server = MCPServer(
    name="financial-market-mcp",
    description="Get a public A-share market snapshot for one supported company.",
)
provider = MarketProvider()


class MarketSnapshot(BaseModel):
    company_name: str | None = None
    ticker: str | None = None
    exchange: str | None = None
    price: float | None = None
    change: float | None = None
    change_percent: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    previous_close: float | None = None
    volume: float | None = None
    volume_unit: str | None = None
    turnover: float | None = None
    currency: str | None = None
    market_time: str | None = None
    source: str
    success: bool
    error: str | None = None


@server.tool(name="get_market_snapshot", structured_output=True)
def get_market_snapshot(company_name: str | None = None, ticker: str | None = None) -> MarketSnapshot:
    """Return a best-effort public market snapshot for company_name or ticker."""
    if not company_name and not ticker:
        return MarketSnapshot(**provider._failure("至少必须提供 company_name 或 ticker。"))
    return MarketSnapshot(**provider.get_market_snapshot(company_name=company_name, ticker=ticker))


if __name__ == "__main__":
    server.run(transport="stdio")
