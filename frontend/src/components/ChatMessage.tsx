import type { ReactNode } from "react";
import type { ChatMessage as ChatMessageValue } from "../types/agent";
import { AgentActivity } from "./AgentActivity";
import { AgentTrace } from "./AgentTrace";
import { Sources } from "./Sources";

function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={index}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`")) return <code key={index}>{part.slice(1, -1)}</code>;
    return part;
  });
}

function MarkdownBody({ content }: { content: string }) {
  const lines = content.split("\n");
  return (
    <div className="markdown-body">
      {lines.map((line, index) => {
        if (/^###\s+/.test(line)) return <h3 key={index}>{inline(line.slice(4))}</h3>;
        if (/^##\s+/.test(line)) return <h2 key={index}>{inline(line.slice(3))}</h2>;
        if (/^#\s+/.test(line)) return <h1 key={index}>{inline(line.slice(2))}</h1>;
        if (/^[-*]\s+/.test(line)) return <div className="markdown-bullet" key={index}>• {inline(line.slice(2))}</div>;
        if (/^\|.*\|$/.test(line)) return <div className="markdown-table-row" key={index}>{line.split("|").filter(Boolean).map((cell, cellIndex) => <span key={cellIndex}>{inline(cell.trim())}</span>)}</div>;
        if (!line.trim()) return <div className="markdown-gap" key={index} />;
        return <p key={index}>{inline(line)}</p>;
      })}
    </div>
  );
}

export function ChatMessage({ message }: { message: ChatMessageValue }) {
  if (message.role === "user") {
    return <article className="message message-user"><p>{message.content}</p></article>;
  }
  return (
    <article className="message message-assistant" aria-live={message.isStreaming ? "polite" : undefined}>
      {message.activity && <AgentActivity activity={message.activity} />}
      {message.content && <MarkdownBody content={message.content} />}
      {message.isStreaming && <span className="typing-cursor" aria-label="正在生成">▍</span>}
      {message.sources && <Sources sources={message.sources} />}
      {message.trace && <AgentTrace trace={message.trace} />}
    </article>
  );
}
