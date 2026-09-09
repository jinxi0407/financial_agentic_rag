export type ToolName = "financial_rag" | "market_mcp" | "news_mcp" | "calculator";
export type ToolStatus = "pending" | "planned" | "running" | "success" | "failed" | "skipped";
export type GuardrailStatus = "passed" | "degraded" | "error" | null;

export interface EventEnvelope<T> {
  type: string;
  data: T;
  timestamp: string;
}

export interface StartData {
  thread_id: string;
  user_id: string;
  query: string;
}

export interface PlanData {
  intent: string;
  required_tools: ToolName[];
  skill?: string | null;
}

export interface ToolStartData {
  tool: ToolName;
}

export interface ToolEndData {
  tool: ToolName;
  success: boolean;
  latency: number;
  summary: string;
  error_type?: "provider_timeout" | "tool_execution_error";
}

export interface GuardrailData {
  status: Exclude<GuardrailStatus, null>;
}

export interface Source {
  tool: "market_mcp" | "news_mcp" | "financial_rag";
  provider?: string;
  symbol?: string;
  timestamp?: string;
  source?: string;
  title?: string;
  published_at?: string;
  url?: string;
}

export interface AgentTrace {
  thread_id?: string;
  intent?: string;
  skill?: string | null;
  planned_tools?: ToolName[];
  executed_tools?: ToolName[];
  tool_latencies?: Partial<Record<ToolName | "synthesis", number>>;
  total_latency?: number;
}

export interface TokenData {
  content: string;
}

export interface ErrorData {
  code: "INVALID_REQUEST" | "AGENT_EXECUTION_ERROR";
  message: string;
}

export type AgentEvent =
  | (EventEnvelope<StartData> & { type: "start" })
  | (EventEnvelope<PlanData> & { type: "plan" })
  | (EventEnvelope<ToolStartData> & { type: "tool_start" })
  | (EventEnvelope<ToolEndData> & { type: "tool_end" })
  | (EventEnvelope<Record<string, never>> & { type: "synthesis_start" })
  | (EventEnvelope<GuardrailData> & { type: "guardrail" })
  | (EventEnvelope<Source[]> & { type: "sources" })
  | (EventEnvelope<AgentTrace> & { type: "trace" })
  | (EventEnvelope<TokenData> & { type: "token" })
  | (EventEnvelope<{ success: boolean }> & { type: "end" })
  | (EventEnvelope<ErrorData> & { type: "error" });

export interface ToolActivity {
  status: ToolStatus;
  latency?: number;
  summary?: string;
}

export interface AgentActivityState {
  intent?: string;
  skill?: string | null;
  tools: Record<ToolName, ToolActivity>;
  synthesisStarted: boolean;
  guardrail: GuardrailStatus;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  activity?: AgentActivityState;
  sources?: Source[];
  trace?: AgentTrace;
  isStreaming?: boolean;
  error?: boolean;
}
