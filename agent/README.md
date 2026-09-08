# Financial Agent MVP

The Agent MVP keeps the frozen Financial RAG runtime behind one tool boundary:

```text
Planner
|- FinancialRAGTool
|- CalculatorTool
|- MarketDataTool   # TODO: not implemented
`- NewsSearchTool   # TODO: not implemented
```

`FinancialRAGTool` calls `IntegratedQASystem.query()` and does not duplicate
retrieval, Milvus, embedding, or reranker logic. `CalculatorTool` performs
only deterministic arithmetic locally.

Market and news intents are recognized as planned but unavailable. They never
fall back to Financial RAG and make no external API calls in this MVP.
