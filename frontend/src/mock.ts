import type { AgentEvent } from "./types/agent";

const now = () => new Date().toISOString();

export function mockEvents(query: string, threadId: string, userId: string): AgentEvent[] {
  return [
    { type: "start", data: { query, thread_id: threadId, user_id: userId }, timestamp: now() },
    { type: "plan", data: { intent: "composite_query", required_tools: ["financial_rag", "market_mcp", "news_mcp"], skill: "company_comparison" }, timestamp: now() },
    { type: "tool_start", data: { tool: "financial_rag" }, timestamp: now() },
    { type: "tool_end", data: { tool: "financial_rag", success: true, latency: 1.62, summary: "Financial RAG 已完成。" }, timestamp: now() },
    { type: "tool_start", data: { tool: "market_mcp" }, timestamp: now() },
    { type: "tool_end", data: { tool: "market_mcp", success: true, latency: 0.31, summary: "已获取 2 个标的的行情结果。" }, timestamp: now() },
    { type: "tool_start", data: { tool: "news_mcp" }, timestamp: now() },
    { type: "tool_end", data: { tool: "news_mcp", success: true, latency: 0.48, summary: "已获取 2 条可展示新闻。" }, timestamp: now() },
    { type: "synthesis_start", data: {}, timestamp: now() },
    { type: "guardrail", data: { status: "passed" }, timestamp: now() },
    { type: "sources", data: [{ tool: "market_mcp", provider: "Mock market", symbol: "600519", timestamp: "mock" }, { tool: "news_mcp", title: "Mock 新闻条目", source: "Mock provider", published_at: "mock", url: "https://example.com", provider: "mock" }], timestamp: now() },
    { type: "trace", data: { thread_id: threadId, intent: "composite_query", skill: "company_comparison", planned_tools: ["financial_rag", "market_mcp", "news_mcp"], executed_tools: ["financial_rag", "market_mcp", "news_mcp"], tool_latencies: { financial_rag: 1.62, market_mcp: 0.31, news_mcp: 0.48 }, total_latency: 2.49 }, timestamp: now() },
    { type: "token", data: { content: "这是 **Mock mode** 的界面预览，不代表真实 Agent、行情或新闻结果。\n\n" }, timestamp: now() },
    { type: "token", data: { content: "请配置真实后端后查看可验证的工具执行与回答。" }, timestamp: now() },
    { type: "end", data: { success: true }, timestamp: now() },
  ];
}
