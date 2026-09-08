"""Small declarative registry for the Agent's currently supported skills."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AgentSkill:
    name: str
    description: str
    required_capabilities: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["required_capabilities"] = list(self.required_capabilities)
        return payload


SKILLS = {
    "financial_report_analysis": AgentSkill(
        name="financial_report_analysis",
        description="分析已入库上市公司财报与金融知识库内容。",
        required_capabilities=("financial_rag",),
    ),
    "company_comparison": AgentSkill(
        name="company_comparison",
        description="对多家公司或多个报告期间的财务、行情和新闻信息进行综合比较。",
        required_capabilities=("financial_rag", "market_mcp", "news_mcp", "calculator"),
    ),
    "market_intelligence": AgentSkill(
        name="market_intelligence",
        description="查询已支持证券的公开行情与近期新闻。",
        required_capabilities=("market_mcp", "news_mcp"),
    ),
}


def get_skill(name: str | None) -> AgentSkill | None:
    return SKILLS.get(name or "")
