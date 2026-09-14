# Agent FC v1 Freeze Record

This release packages the accepted one-shot Qwen Function Calling implementation.
The immutable release reference is `agent-fc-v1`; resolve its commit with
`git rev-parse agent-fc-v1^{commit}`. Deployment must use that exact commit, not
an uncommitted overlay or an unspecified latest revision.

## Architecture

```text
User Query
  -> Session Memory / Context
  -> Planner Mode
       Rule Planner
       Qwen Function Calling Planner (one request, native tools / tool_calls)
         -> arguments validation
  -> AgentState
  -> fixed LangGraph nodes
       Financial RAG / Market MCP / News MCP / Calculator
  -> Aggregation
  -> Composite Qwen Synthesis
  -> deterministic Guardrail
  -> Final Answer
  -> WebSocket
```

This is not a ReAct loop. Tool results do not trigger additional planning.
The WebSocket exposes graph activity and chunks of the completed, checked answer;
it does not expose unchecked native Qwen answer tokens. The legacy runner contract
and WebSocket event schema remain unchanged.

The intended production configuration matches strict acceptance:
`AGENT_PLANNER_MODE=function_calling`, `AGENT_PLANNER_MODEL=qwen3.8-max`,
`AGENT_PLANNER_FALLBACK_TO_RULE=false`. Repository defaults remain rule mode.
Secrets and deployment-specific paths belong only in the GPU-side environment.

## Separate Evaluation Results

These are previously completed experiments, not reruns for this release.

| Planner regression, same 150 inputs | Rule | Function Calling |
|---|---:|---:|
| Tool selection | 137/150 (91.33%) | 140/150 (93.33%) |
| Composite exact tool set | 15/20 | 20/20 |
| Required tool coverage | 91.10% | 93.15% |
| Same-tool multi-target coverage | 21/24 | 24/24 |

FC adds approximately 1.5 seconds of planning latency: a flexibility/latency
trade-off. This is a known development regression set, not a new unseen holdout.
Argument schema validity and exact delivery do not establish full semantic
Argument Accuracy without independent argument labels.

Historic retrieval 300Q results remain separate: Document Recall@3
83.4% -> 94.6%, Period Target Accuracy 89.6% -> 100%, Wrong Period Rate
25.4% -> 0%. Historic final RAGAS 100Q scores are approximately 0.892
Faithfulness and 0.851 Context Precision. They are not FC planning scores or
evidence of a newly run end-to-end benchmark.

## Accepted Source and Limitations

The accepted isolated GPU run `20260914T145628Z` used real RAG, Market, News,
aggregation, Qwen synthesis, numeric guardrail and WebSocket. All three validated
calls reached execution with identical arguments; guardrail passed and the
synthesis answer equaled the final answer. It made five Qwen HTTP requests,
used 8,602 tokens and completed the WebSocket exchange in approximately 42.7s.
Earlier isolated runs also verified Calculator and Redis context recovery.

Offline acceptance: 129 focused tests passed, including 40 answer-processing
and numeric tests. Expanded offline regression: 252 passed out of 253;
one live Redis test was excluded from this offline suite. The sole failure is
`tests.test_langgraph_agent.LangGraphAgentTests.test_same_thread_follow_up_and_thread_isolation`.
It also fails on the pre-fix source archive and remains `PRE_EXISTING_FAILURE`;
neither Memory behavior nor the old assertion was changed to hide it.

News provenance fields are protected from availability text cleanup. Numeric
checks retain typed units and exact percentages; explicit approximate financial
or market amounts have a centralized 2% relative tolerance. Prices and percentages
do not get that tolerance. This bounded check is not a general factual/causal
claim judge and does not guarantee investment-analysis correctness.

## Deployment Gate and Rollback

Production freeze is conditional on the exact tagged source passing a fresh
8002 canary before replacing 8001, followed by public smoke checks. The accepted
isolated run above is not a substitute for those deployment checks.

Keep the old deployment directory, commit, environment and start command intact.
Create the new deployment independently, using a GPU-local configuration copy;
do not copy a Mac `.env`. Preserve existing data services, SSH tunnels, Cloudflare
routing and historical tags. If public smoke fails seriously, stop the new
process and restart the recorded old version. Never patch production in place.

Record the final commit, source fingerprints, canary/public results and rollback
details in the external deployment audit. Raw runtime logs, credentials and
benchmark result dumps are not part of this source release.
