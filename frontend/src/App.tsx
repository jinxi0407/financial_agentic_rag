import { useEffect, useRef, useState } from "react";
import { AgentTrace } from "./components/AgentTrace";
import { ChatMessage } from "./components/ChatMessage";
import { CompanySidebar } from "./components/CompanySidebar";
import { Composer } from "./components/Composer";
import type { SupportedCompany } from "./data/companies";
import { useAgentStream } from "./hooks/useAgentStream";

const EXAMPLES = [
  "贵州茅台 2026H1 营业收入是多少？",
  "比较贵州茅台和五粮液 2026H1 的经营表现。",
  "中芯国际最近有哪些重要新闻？",
  "结合比亚迪 2026H1 财报、实时行情和近期新闻分析其经营表现。",
];

const githubUrl = import.meta.env.VITE_GITHUB_URL?.trim() || "https://github.com/jinxi0407/financial_agentic_rag";

export default function App() {
  const [draft, setDraft] = useState("");
  const [selectedCompany, setSelectedCompany] = useState<string | null>(null);
  const [focusRequest, setFocusRequest] = useState(0);
  const conversationRef = useRef<HTMLDivElement>(null);
  const followBottomRef = useRef(true);
  const { messages, isStreaming, backendStatus, isReady, send, newChat, isMockMode } = useAgentStream();

  useEffect(() => {
    const container = conversationRef.current;
    if (container && followBottomRef.current) {
      container.scrollTop = container.scrollHeight;
    }
  }, [messages]);

  const onScroll = () => {
    const container = conversationRef.current;
    if (!container) return;
    followBottomRef.current = container.scrollHeight - container.scrollTop - container.clientHeight < 72;
  };

  const submit = () => {
    if (!draft.trim() || !isReady) return;
    send(draft);
    setDraft("");
  };

  const chooseExample = (prompt: string) => setDraft(prompt);
  const selectCompany = (company: SupportedCompany) => {
    setSelectedCompany(company.name);
    setDraft((current) => {
      if (current.includes(company.name)) return current;
      return current.trim() ? `${current.trimEnd()} ${company.name} ` : `${company.name} `;
    });
    setFocusRequest((value) => value + 1);
  };

  const startNewChat = () => {
    newChat();
    setSelectedCompany(null);
  };

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">F</span>
          <div>
            <h1>Financial Agent</h1>
            <p>Financial Research Assistant</p>
          </div>
        </div>
        <nav aria-label="主要操作">
          <a href={githubUrl} target="_blank" rel="noreferrer">GitHub</a>
          <button type="button" onClick={startNewChat} aria-label="开始新对话">New Chat</button>
        </nav>
      </header>

      <div className="app-body">
        <CompanySidebar selectedCompany={selectedCompany} onSelect={selectCompany} />
        <div className="chat-column">
          {isMockMode && <div className="mode-banner">Mock mode · 仅用于 UI 预览，不代表真实 Agent 输出</div>}
          {backendStatus === "offline" && (
            <div className="offline-banner">
              演示服务当前离线。项目代码、架构与评测结果仍可在 <a href={githubUrl} target="_blank" rel="noreferrer">GitHub</a> 查看。
            </div>
          )}

          <section className={`conversation ${messages.length ? "has-messages" : "empty"}`} ref={conversationRef} onScroll={onScroll}>
            {!messages.length ? (
              <div className="empty-state">
                <p className="eyebrow">MULTI-TOOL FINANCIAL RESEARCH</p>
                <h2>Financial Agent</h2>
                <p>基于财报、实时行情与财经新闻的多工具金融研究助手</p>
                <div className="example-grid">
                  {EXAMPLES.map((prompt) => (
                    <button key={prompt} type="button" onClick={() => chooseExample(prompt)}>{prompt}</button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="message-list">
                {messages.map((message) => <ChatMessage key={message.id} message={message} />)}
              </div>
            )}
          </section>

          <div className="composer-wrap">
            <Composer value={draft} onChange={setDraft} onSend={submit} disabled={isStreaming || !isReady} focusRequest={focusRequest} />
            <p className="composer-hint">Enter 发送 · Shift + Enter 换行</p>
          </div>
          <footer>Research demo · Not investment advice · 仅用于金融信息研究与技术演示</footer>
        </div>
      </div>
    </main>
  );
}
