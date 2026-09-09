import type { AgentEvent, AgentTrace, Source, ToolName } from "./types/agent";

const now = () => new Date().toISOString();

interface MockPlan {
  intent: string;
  skill: string;
  tools: ToolName[];
  failureTool?: ToolName;
}

function mockPlanFor(query: string): MockPlan {
  const text = query.toLowerCase();
  const wantsCalculator = /计算|增长率|增幅|差值|百分点|从\s*\d/.test(text);
  const wantsMarket = /股价|行情|涨跌|成交|市值/.test(text);
  const wantsNews = /新闻|消息|公告|事件|动态/.test(text);
  const wantsFinancial = /财报|年报|半年报|h1|fy|营业收入|营收|净利润|研发|毛利率|财务|经营/.test(text);
  const failureTool = /模拟失败|mock fail/.test(text) ? "news_mcp" : undefined;

  if (wantsCalculator && !wantsFinancial && !wantsMarket && !wantsNews) {
    return { intent: "calculation_query", skill: "financial_calculation", tools: ["calculator"] };
  }
  if (wantsFinancial && (wantsMarket || wantsNews)) {
    const tools: ToolName[] = ["financial_rag"];
    if (wantsMarket) tools.push("market_mcp");
    if (wantsNews) tools.push("news_mcp");
    if (wantsCalculator) tools.push("calculator");
    return { intent: "composite_query", skill: "company_comparison", tools, failureTool };
  }
  if (wantsMarket) return { intent: "market_query", skill: "market_intelligence", tools: ["market_mcp"] };
  if (wantsNews) return { intent: "news_query", skill: "market_intelligence", tools: ["news_mcp"], failureTool };
  if (wantsCalculator) return { intent: "calculation_query", skill: "financial_calculation", tools: ["calculator"] };
  return { intent: "financial_report_query", skill: "financial_report_analysis", tools: ["financial_rag"] };
}

function sourceEvents(tools: ToolName[]): Source[] {
  const sources: Source[] = [];
  if (tools.includes("market_mcp")) {
    sources.push({ tool: "market_mcp", provider: "Mock market", symbol: "600519", timestamp: "mock" });
  }
  if (tools.includes("news_mcp")) {
    sources.push({
      tool: "news_mcp",
      title: "Mock 新闻条目",
      source: "Mock provider",
      published_at: "mock",
      url: "https://example.com",
      provider: "mock",
    });
  }
  return sources;
}

function toolSummary(tool: ToolName): string {
  if (tool === "financial_rag") return "Mock Financial RAG UI 状态已完成。";
  if (tool === "market_mcp") return "Mock Market MCP UI 状态已完成。";
  if (tool === "news_mcp") return "Mock News MCP UI 状态已完成。";
  return "Mock Calculator UI 状态已完成。";
}

export function mockEvents(query: string, threadId: string, userId: string): AgentEvent[] {
  const plan = mockPlanFor(query);
  const events: AgentEvent[] = [
    { type: "start", data: { query, thread_id: threadId, user_id: userId }, timestamp: now() },
    { type: "plan", data: { intent: plan.intent, required_tools: plan.tools, skill: plan.skill }, timestamp: now() },
  ];

  for (const tool of plan.tools) {
    const failed = tool === plan.failureTool;
    events.push({ type: "tool_start", data: { tool }, timestamp: now() });
    events.push({
      type: "tool_end",
      data: {
        tool,
        success: !failed,
        latency: tool === "financial_rag" ? 1.62 : tool === "calculator" ? 0.02 : 0.31,
        summary: failed ? "Mock 数据源暂不可用。" : toolSummary(tool),
        ...(failed ? { error_type: "provider_timeout" } : {}),
      },
      timestamp: now(),
    });
  }

  if (plan.tools.length > 1) events.push({ type: "synthesis_start", data: {}, timestamp: now() });
  events.push({ type: "guardrail", data: { status: plan.failureTool ? "degraded" : "passed" }, timestamp: now() });
  events.push({ type: "sources", data: sourceEvents(plan.tools.filter((tool) => tool !== plan.failureTool)), timestamp: now() });
  const toolLatencies = Object.fromEntries(plan.tools.map((tool) => [tool, tool === "financial_rag" ? 1.62 : tool === "calculator" ? 0.02 : 0.31]));
  const trace: AgentTrace = {
    thread_id: threadId,
    intent: plan.intent,
    skill: plan.skill,
    planned_tools: plan.tools,
    executed_tools: plan.tools,
    tool_latencies: toolLatencies,
    total_latency: Object.values(toolLatencies).reduce((total, latency) => total + latency, 0),
  };
  events.push({ type: "trace", data: trace, timestamp: now() });
  events.push({
    type: "token",
    data: { content: `这是 **Mock mode** 的 UI 预览。当前仅按本地关键词模拟 ${plan.tools.join(" + ")} 路径，未调用真实 Agent、财报、行情或新闻服务。\n\n` },
    timestamp: now(),
  });
  events.push({ type: "token", data: { content: "请配置真实后端后查看可验证的 Planner、工具执行与最终回答。" }, timestamp: now() });
  events.push({ type: "end", data: { success: true }, timestamp: now() });
  return events;
}
