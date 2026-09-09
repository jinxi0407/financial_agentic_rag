import { useCallback, useEffect, useRef, useState } from "react";
import { mockEvents } from "../mock";
import type {
  AgentActivityState,
  AgentEvent,
  ChatMessage,
  ToolActivity,
  ToolName,
} from "../types/agent";

const TOOLS: ToolName[] = ["financial_rag", "market_mcp", "news_mcp", "calculator"];
const isMockMode = import.meta.env.VITE_USE_MOCK === "true";

function newId(): string {
  return crypto.randomUUID?.() ?? `agent-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function initialTools(): Record<ToolName, ToolActivity> {
  return Object.fromEntries(TOOLS.map((tool) => [tool, { status: "pending" }])) as Record<ToolName, ToolActivity>;
}

function initialActivity(): AgentActivityState {
  return { tools: initialTools(), synthesisStarted: false, guardrail: null };
}

function websocketUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim();
  const endpoint = new URL(configured || window.location.origin, window.location.href);
  endpoint.protocol = endpoint.protocol === "https:" ? "wss:" : "ws:";
  endpoint.pathname = `${endpoint.pathname.replace(/\/$/, "")}/api/agent/stream`.replace(/\/+/g, "/");
  endpoint.search = "";
  endpoint.hash = "";
  return endpoint.toString();
}

export function useAgentStream() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [backendStatus, setBackendStatus] = useState<"ready" | "online" | "offline">("ready");
  const socketRef = useRef<WebSocket | null>(null);
  const completedRef = useRef(false);
  const [userId, setUserId] = useState("");
  const [threadId, setThreadId] = useState("");

  useEffect(() => {
    const stored = localStorage.getItem("financial-agent-user-id");
    const currentUser = stored || newId();
    if (!stored) localStorage.setItem("financial-agent-user-id", currentUser);
    setUserId(currentUser);
    setThreadId(newId());
    return () => socketRef.current?.close();
  }, []);

  const updateAssistant = useCallback((id: string, change: (message: ChatMessage) => ChatMessage) => {
    setMessages((items) => items.map((item) => item.id === id ? change(item) : item));
  }, []);

  const updateTool = useCallback((assistantId: string, tool: ToolName, value: ToolActivity) => {
    updateAssistant(assistantId, (message) => ({
      ...message,
      activity: {
        ...(message.activity ?? initialActivity()),
        tools: { ...(message.activity?.tools ?? initialTools()), [tool]: value },
      },
    }));
  }, [updateAssistant]);

  const applyEvent = useCallback((assistantId: string, event: AgentEvent) => {
    switch (event.type) {
      case "start":
        setThreadId(event.data.thread_id);
        setUserId(event.data.user_id);
        return;
      case "plan":
        updateAssistant(assistantId, (message) => ({
          ...message,
          activity: {
            ...(message.activity ?? initialActivity()),
            intent: event.data.intent,
            skill: event.data.skill,
            tools: Object.fromEntries(TOOLS.map((tool) => [tool, {
              status: event.data.required_tools.includes(tool) ? "planned" : "skipped",
            }])) as Record<ToolName, ToolActivity>,
          },
        }));
        return;
      case "tool_start":
        updateTool(assistantId, event.data.tool, { status: "running" });
        return;
      case "tool_end":
        updateTool(assistantId, event.data.tool, {
          status: event.data.success ? "success" : "failed",
          latency: event.data.latency,
          summary: event.data.summary,
        });
        return;
      case "synthesis_start":
        updateAssistant(assistantId, (message) => ({
          ...message,
          activity: { ...(message.activity ?? initialActivity()), synthesisStarted: true },
        }));
        return;
      case "guardrail":
        updateAssistant(assistantId, (message) => ({
          ...message,
          activity: { ...(message.activity ?? initialActivity()), guardrail: event.data.status },
        }));
        return;
      case "sources":
        updateAssistant(assistantId, (message) => ({ ...message, sources: event.data }));
        return;
      case "trace":
        updateAssistant(assistantId, (message) => ({ ...message, trace: event.data }));
        return;
      case "token":
        updateAssistant(assistantId, (message) => ({ ...message, content: message.content + event.data.content }));
        return;
      case "end":
        completedRef.current = true;
        setIsStreaming(false);
        updateAssistant(assistantId, (message) => ({ ...message, isStreaming: false }));
        socketRef.current?.close();
        socketRef.current = null;
        return;
      case "error":
        completedRef.current = true;
        setIsStreaming(false);
        updateAssistant(assistantId, (message) => ({
          ...message,
          content: "请求失败，请稍后重试。",
          isStreaming: false,
          error: true,
        }));
    }
  }, [updateAssistant, updateTool]);

  const send = useCallback((query: string) => {
    if (!query.trim() || isStreaming || !threadId || !userId) return;
    socketRef.current?.close();
    completedRef.current = false;
    setIsStreaming(true);
    const assistantId = newId();
    setMessages((items) => [
      ...items,
      { id: newId(), role: "user", content: query.trim() },
      { id: assistantId, role: "assistant", content: "", activity: initialActivity(), isStreaming: true },
    ]);

    if (isMockMode) {
      mockEvents(query.trim(), threadId, userId).forEach((event, index) => {
        window.setTimeout(() => applyEvent(assistantId, event), index * 180);
      });
      return;
    }

    let socket: WebSocket;
    try {
      socket = new WebSocket(websocketUrl());
    } catch {
      setBackendStatus("offline");
      applyEvent(assistantId, { type: "error", data: { code: "AGENT_EXECUTION_ERROR", message: "Agent request failed." }, timestamp: new Date().toISOString() });
      return;
    }
    socketRef.current = socket;
    socket.onopen = () => {
      setBackendStatus("online");
      socket.send(JSON.stringify({ query: query.trim(), thread_id: threadId, user_id: userId }));
    };
    socket.onmessage = (message) => {
      try {
        applyEvent(assistantId, JSON.parse(message.data) as AgentEvent);
      } catch {
        applyEvent(assistantId, { type: "error", data: { code: "AGENT_EXECUTION_ERROR", message: "Agent request failed." }, timestamp: new Date().toISOString() });
      }
    };
    socket.onerror = () => setBackendStatus("offline");
    socket.onclose = () => {
      if (!completedRef.current) {
        setBackendStatus("offline");
        applyEvent(assistantId, { type: "error", data: { code: "AGENT_EXECUTION_ERROR", message: "Agent request failed." }, timestamp: new Date().toISOString() });
      }
    };
  }, [applyEvent, isStreaming, threadId, userId]);

  const newChat = useCallback(() => {
    socketRef.current?.close();
    completedRef.current = true;
    setMessages([]);
    setThreadId(newId());
    setIsStreaming(false);
  }, []);

  return { messages, isStreaming, backendStatus, userId, threadId, send, newChat, isMockMode };
}
