import type { AgentActivityState, ToolName, ToolStatus } from "../types/agent";

const TOOLS: Array<{ name: ToolName; label: string }> = [
  { name: "financial_rag", label: "Financial RAG" },
  { name: "market_mcp", label: "Market MCP" },
  { name: "news_mcp", label: "News MCP" },
  { name: "calculator", label: "Calculator" },
];

const STATUS_LABEL: Record<ToolStatus, string> = {
  pending: "等待计划",
  planned: "已计划",
  running: "正在执行",
  success: "已完成",
  failed: "数据源暂不可用",
  skipped: "未调用",
};

function StatusMark({ status }: { status: ToolStatus }) {
  if (status === "running") return <span className="status-mark spinner" aria-label="正在执行" />;
  if (status === "success") return <span className="status-mark success" aria-label="已完成">✓</span>;
  if (status === "failed") return <span className="status-mark failed" aria-label="执行失败">!</span>;
  if (status === "skipped") return <span className="status-mark skipped" aria-label="未调用">○</span>;
  return <span className="status-mark pending" aria-label={STATUS_LABEL[status]}>·</span>;
}

export function AgentActivity({ activity }: { activity: AgentActivityState }) {
  return (
    <section className="agent-activity" aria-label="Agent Activity">
      <div className="activity-heading">
        <span>Agent Activity</span>
        {activity.intent && <span className="activity-intent">{activity.intent}</span>}
      </div>
      <div className="tool-list">
        {TOOLS.map(({ name, label }) => {
          const tool = activity.tools[name];
          return (
            <div className="tool-row" key={name}>
              <StatusMark status={tool.status} />
              <span className="tool-label">{label}</span>
              <span className="tool-detail">
                {tool.latency !== undefined ? `${tool.latency.toFixed(2)}s` : STATUS_LABEL[tool.status]}
              </span>
            </div>
          );
        })}
      </div>
      {activity.synthesisStarted && <p className="activity-note">正在综合多个数据源…</p>}
      {activity.guardrail === "passed" && <p className="guardrail-pass">✓ Evidence checked</p>}
      {activity.guardrail === "degraded" && (
        <p className="guardrail-degraded">部分外部数据不可用，已安全降级</p>
      )}
    </section>
  );
}
