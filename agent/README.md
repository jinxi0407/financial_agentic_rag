# Financial Agent MVP

The Agent MVP keeps the frozen Financial RAG runtime behind one tool boundary:

```text
Planner
|- FinancialRAGTool
|- CalculatorTool
|- Market MCP Server
`- News MCP Server
```

`FinancialRAGTool` calls `IntegratedQASystem.query()` and does not duplicate
retrieval, Milvus, embedding, or reranker logic. `CalculatorTool` performs
only deterministic arithmetic locally.

Market and news calls use local stdio MCP servers. The servers are the only
layer that accesses public providers; Agent Runner communicates with them via
the official MCP client protocol, never by importing provider functions.

## Next Stage TODO

- LangGraph orchestration
- Session memory
- Composite query execution
- Skills and Agent trace
