# Evaluation

This document records the frozen evaluation protocol, denominators and interpretation limits for Financial Agentic RAG. Values are reported from saved artifacts; updating this document does not re-run RAG, Agent, MCP or a Judge.

## Benchmark Map

| Suite | Purpose | Scope |
|---|---|---|
| 60Q Gold v2 | Development regression for routing, retrieval and numeric answers | Not the headline holdout |
| 300Q frozen holdout | Unseen-query retrieval/orchestration evaluation | Same frozen report corpus; no MCP |
| 100Q RAGAS | Answer faithfulness and returned-context quality | Financial RAG-only capture + Judge |
| 150-turn Agent | Planner, tools, memory and failure handling | Real Agent E2E |

The 300Q set is an unseen-query holdout, not an unseen-document benchmark. It uses the same 24 frozen reports as the system corpus.

## 300Q Retrieval Holdout

Baseline and Final keep BGE-M3, Milvus, reranker, the frozen corpus, `RETRIEVAL_K=30` and `CANDIDATE_M=3` constant. The Final variant adds deterministic company × period multi-target orchestration; it does not change the retrieval model or globally increase K.

| Metric | Baseline | Final |
|---|---:|---:|
| Document Recall@3 | 83.4% | 94.6% |
| Period Target Accuracy | 89.6% | 100% |
| Wrong Period Rate | 25.4% | 0% |

`Document Recall@3` is required-document recall. It must not be interpreted as a generic semantic relevance score.

## 100Q RAGAS

The 100 samples are stratified from the frozen 300Q manifest. Baseline and Final use the same sample IDs and a Financial RAG-only evaluation entrypoint that bypasses FAQ/MySQL fast paths. This keeps the comparison Financial RAG vs Financial RAG rather than FAQ vs RAG.

Evidence capture records only material actually available to the answer layer: retrieved Parent contexts, verified financial evidence, deterministic calculations and provenance. It never injects gold contexts.

| Metric | Paired Baseline | Final |
|---|---:|---:|
| Faithfulness | 0.864 | 0.892 |
| Context Precision | - | 0.851 |

RAGAS uses Faithfulness, Answer Relevancy and `LLMContextPrecisionWithoutReference`. Answer Relevancy is retained in raw diagnostics, not used as the README headline. Context Precision measures returned-context relevance; it is not a substitute for Document Recall@3 or multi-target coverage.

## 150-turn Agent Benchmark

The Agent holdout separates Planner correctness from tool availability. A correct News tool selection remains a planner success even if a provider later times out; tool execution success is reported independently.

| Metric | Result | Definition |
|---|---:|---|
| Tool Selection Accuracy | 91.33% | Planned tool set equals expected tool set |
| Context Recovery | 97.44% | Full labeled memory-recovery scenarios |
| Tool Execution Success Rate | 100% | Success among invoked tools |

The final run uses the domestic-news provider chain. Results do not mean the Agent is 100% accurate: execution success is specifically a provider/tool-runtime metric. Session memory, thread isolation, explicit preference recovery and safe degradation use separate deterministic checks.

## Reproducibility and Artifact Policy

- Dataset manifests, audit scripts and runners are under `evaluations/`.
- Formal runners require explicit confirmation before a full run and checkpoint completed cases.
- Agent benchmarks namespace effective thread/user IDs by run ID to avoid Redis session or preference contamination.
- Raw captures, smoke outputs and local result JSON files are ignored by default. Keep reproducible scripts and frozen manifests in Git; publish aggregate summaries rather than transient captures or secrets.
- Public Market/News providers are nondeterministic external dependencies. Their failures must be reported as execution outcomes, not silently converted into planner failures.
