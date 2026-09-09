# Evaluation

本文记录冻结 Financial Agentic RAG 的评测目的、协议、artifact 与已知边界。所有指标来自仓库 `evaluations/` 中保存的 JSON；本文不代表重新执行 RAG、Agent、MCP 或 LLM Judge。

## 三套评测的分工

| 套件 | 目的 | 是否调用最终回答 LLM |
|---|---|---|
| 60Q Gold v2 | 开发期 routing、retrieval 与 numeric-answer regression | 依 runner 配置；不是主 holdout |
| 300Q v1.1 | 同一冻结财报语料上的 unseen-query retrieval/routing holdout | 否，retrieval-only，且不调用 MCP |
| 150-turn Agent v1 | Agent planning、Tool、MCP、Memory 与 failure handling | 是，真实 Agent E2E |

300Q 是 **frozen unseen-query holdout over the same frozen report corpus**，不是 unseen-document benchmark。150-turn 数据集和 raw outputs 均在执行前冻结；v1.1 scoring 只对保存的 raw outputs 离线重计分。

## 300Q RAG Holdout

### 数据与控制变量

数据集由 300 条 query 组成，覆盖单公司、跨期间、多公司、多目标、指标口径歧义、解释/风险与 negative cases。所有 valid query 的 required documents 均来自冻结 24 份财报。Baseline 与 Final 固定：

- BGE-M3、Milvus collection、frozen corpus、reranker
- `RETRIEVAL_K=30`、`CANDIDATE_M=3`
- retrieval-only evaluator，不调用 final-answer LLM 或 MCP

Final 相比 baseline 增加 deterministic company × period multi-target planning；不是通过降低 TopK 或更换模型取得结果。

### Baseline vs Final

| 指标 | Baseline | Final |
|---|---:|---:|
| Company Target Accuracy | 98.89% | 97.78% |
| Period Target Accuracy | 89.63% | 100.00% |
| Document Hit@3 | 95.56% | 100.00% |
| Required Document Coverage@3 | 83.36% | 94.63% |
| Multi-target Full Coverage@3 | 49.38% | 66.25% |
| Wrong Company Rate | 0% | 0% |
| Wrong Period Rate | 25.37% | 0% |
| Negative Handling Accuracy | 73.33% | 73.33% |
| Mean latency | 6.38s | 16.59s |
| P50 latency | 4.85s | 12.68s |
| P95 latency | 17.28s | 48.76s |

Final 运行 artifact 的 runtime config 为 commit `4dbb304`、`financial.financial_rag_v1`、`K=30/M=3`；baseline artifact 为 commit `bcf70ae`，其余控制变量相同。

### Full Coverage@3 的容量边界

`Required Document Coverage@3` 按所需文档逐份计数；`Multi-target Full Coverage@3` 要求一个 case 的全部 required documents 都出现于 Top3。300Q 中有 54 个 case 的 required documents 超过 3：40 个 `multi_company_multi_period` 和 14 个 `complex_multi_target`。这些 case 在 Top3 下数学上不能做到 strict full coverage。

因此 Final 的 strict Full Coverage@3 为 66.25%，而在 `required_docs ≤ 3` 的 216 个可行 case 上为 `216/216 = 100%`。这不是放宽指标，而是对固定 Top3 容量的单独说明。

## 60Q Gold v2 Regression

60Q 用于开发过程中的 routing、retrieval 和 numeric-answer regression，保留 gold answer/evidence 与 Gold v2 changelog。当前跟踪的 `financial_eval_60_results.json` artifact 显示：60 条已处理，其中 59 条进入可评分样本、1 条错误；其保存指标为 Router Accuracy 98.31%、Strategy Accuracy 82.14%、Document Hit@3 100%、Required Document Coverage@3 73.58%、Strict Parent Evidence Recall@3 22.64%、Wrong Company 0%、Wrong Period 13.21%。

这与历史工作记录中“60/60 final frozen”数值并不一致。为避免将不可由当前 artifact 证明的结果写成事实，README 只将 60Q 描述为独立 regression suite，不将其作为主项目成绩展示。

## 150-turn Frozen Agent E2E

### 最终 Domestic News Provider Run

最终 raw artifact 为 `financial_agent_holdout_150_v1_news_v1_1_results.json`：`150/150` 完成，运行 commit `7204eca`，dataset fingerprint 与首次 run 一致。指标使用 `financial_agent_holdout_150_v1_1_scoring.json` 定义的 v1.1 离线协议重新计算。

| 指标 | 结果 | 分母/说明 |
|---|---:|---|
| Intent Accuracy | 91.33% | 150 completed turns |
| Tool Selection Accuracy | 91.33% | planned tools 与 expected tools 的集合完全相等 |
| Required Tool Coverage | 91.10% | 133/146，有 expected tools 的 turn |
| Unnecessary Tool Call Rate | 0.63% | 1/158 executed tool labels |
| Tool Execution Success | 100.00% | Financial RAG 70/70、Market 47/47、News 61/61、Calculator 9/9 |
| Company Context Accuracy | 100.00% | 有 company label 的 turn，集合严格相等 |
| Strict Agent State Period | 61.00% | 61/100 |
| Explicit Period Resolution | 53.45% | 31/58 |
| Period Memory Carryover | 66.67% | 2/3，只统计需要期间恢复的 financial follow-up |
| Full Context Memory Recovery | 97.44% | 38/39 |
| Thread Isolation | 100.00% | 8/8，只检查公司上下文隔离 |
| Composite Completion | 75.00% | 15/20 |
| Mean / P50 / P95 latency | 11.01s / 6.58s / 32.34s | 150 turns |

Preference diagnostics 也来自保存的 runtime state：显式 preference write `2/2`，cross-thread preference recovery `3/3`，ordinary query 不隐式写 preference `3/3`。

### Planner、执行与 Provider 的区分

Tool Selection 判断的是 Planner 是否计划了正确工具；Tool Execution Success 判断的是已执行工具是否实际成功。外部 provider 超时不能被当成 Planner 错误。

Google News-only 首次 run 中：News 为 `0/61`，所有 61 次失败都是 `search_financial_news` 的 Google ConnectTimeout，non-external tool failures 为 0。因此 raw Tool Execution Success 是 `126/187 = 67.38%`，而 Intent/Tool Selection 已是 91.33%。

国内 News Provider 回归后：东方财富主源、Sina Finance fallback、Google optional fallback；News 为 `61/61`，external failures `61 → 0`，Tool Execution `67.38% → 100%`。Planner 指标与 Composite Completion 均不变，说明该修复解决的是 provider connectivity，不是 planner 质量。Mean/P50/P95 latency 从 `12.58/7.94/36.92s` 降至 `11.01/6.58/32.34s`。

### Scoring-only v1.1 Correction

原始 evaluator 复核后发现三类评分问题：两个期间 gold label、部分 guardrail 的可评估性，以及 thread isolation/composite 对外部 provider failure 的归因。v1.1 没有修改 Agent、benchmark question 或 raw outputs，只在保存的结果上做离线 rescoring：

- `agent_holdout_077`：`2024FY → 2024H1`
- `agent_holdout_090`：`2024FY → 2024H1`

两题的 question 均明确写为 `2024H1`。Guardrail 只在现有 `guardrail_status` 或 deterministic flags 能直接映射时进入分母，不通过 final answer 关键词猜测安全性。Google run 的可评估 guardrail 为 `3/3`，Domestic News run 为 `1/1`；其余分别为 20 和 22 条 unevaluable。两者都不应宣传为泛化的“Guardrail Accuracy 100%”。

Safe Degradation 仅统计 `allow_external_tool_failure=true` 且实际 Market/News tool 失败的 turn。Google run 为 `45/45`；Domestic News run 外部失败分母为 0，因此记为 `N/A`，不是 0%。

Composite completion 要求 required tools 均 planned 且 executed、final answer 非空、无 unsupported fabrication flag；若 provider 失败，还必须有记录的 safe degradation。当前剩余 5 个真实 Planner failure 均为 required tool 遗漏，未因 News provider 修复而改变。

## 复现与边界

评测 runner、dataset、audit 和 raw artifacts 位于 `evaluations/`。300Q runner 的 `run` 子命令需要显式 `--confirm-run`，并逐 case checkpoint；Agent holdout 同样隔离 run-specific `thread_id` / `user_id` namespace，避免 Redis session 与 preference 污染历史或其他 run。

公开 Market/News provider 的接口、可用性和响应格式会随时间变化。当前 benchmark 不使用 LLM Judge；数值、路由、工具选择、context/memory、工具执行和 runtime flags 尽量以 deterministic checks 评分。
