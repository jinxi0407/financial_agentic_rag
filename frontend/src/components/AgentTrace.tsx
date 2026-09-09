import { useState } from "react";
import type { AgentTrace as AgentTraceValue } from "../types/agent";

const seconds = (value?: number) => value === undefined ? "—" : `${value.toFixed(2)}s`;

export function AgentTrace({ trace }: { trace: AgentTraceValue }) {
  const [open, setOpen] = useState(false);
  const shortThread = trace.thread_id ? `${trace.thread_id.slice(0, 8)}…` : "—";
  return (
    <div className="disclosure agent-trace">
      <button type="button" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        Agent Trace <span aria-hidden="true">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <dl className="trace-grid">
          <dt>Intent</dt><dd>{trace.intent ?? "—"}</dd>
          <dt>Skill</dt><dd>{trace.skill ?? "—"}</dd>
          <dt>Planned Tools</dt><dd>{trace.planned_tools?.join(", ") || "—"}</dd>
          <dt>Executed Tools</dt><dd>{trace.executed_tools?.join(", ") || "—"}</dd>
          <dt>Tool Latency</dt><dd>{Object.entries(trace.tool_latencies ?? {}).map(([tool, value]) => `${tool} ${seconds(value)}`).join(" · ") || "—"}</dd>
          <dt>Total</dt><dd>{seconds(trace.total_latency)}</dd>
          <dt>Thread</dt><dd>{shortThread}</dd>
        </dl>
      )}
    </div>
  );
}
