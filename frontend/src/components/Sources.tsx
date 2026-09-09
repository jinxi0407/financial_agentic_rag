import { useState } from "react";
import type { Source } from "../types/agent";

function SourceGroup({ title, sources }: { title: string; sources: Source[] }) {
  if (!sources.length) return null;
  return (
    <section className="source-group">
      <h4>{title}</h4>
      {sources.map((source, index) => (
        <div className="source-row" key={`${source.title ?? source.symbol ?? source.source}-${index}`}>
          {source.title && source.url ? (
            <a href={source.url} target="_blank" rel="noreferrer">{source.title}</a>
          ) : (
            <span>{source.symbol ?? source.source}</span>
          )}
          <small>{source.source ?? source.provider}{source.published_at ? ` · ${source.published_at}` : source.timestamp ? ` · ${source.timestamp}` : ""}</small>
        </div>
      ))}
    </section>
  );
}

export function Sources({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  if (!sources.length) return null;
  const news = sources.filter((source) => source.tool === "news_mcp");
  const market = sources.filter((source) => source.tool === "market_mcp");
  const financial = sources.filter((source) => source.tool === "financial_rag");
  return (
    <div className="disclosure">
      <button type="button" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        Sources {sources.length} <span aria-hidden="true">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="disclosure-content">
          <SourceGroup title="Financial Reports" sources={financial} />
          <SourceGroup title="Market" sources={market} />
          <SourceGroup title="News" sources={news} />
        </div>
      )}
    </div>
  );
}
