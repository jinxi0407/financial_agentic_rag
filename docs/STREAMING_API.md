# Agent Streaming API

`WS /api/agent/stream` 是完整 LangGraph Agent 的浏览器事件接口。已有的
`WS /api/stream` 保持不变，仍只服务 Financial RAG 的流式回答。

## 请求

客户端发送 JSON：

```json
{
  "query": "比较贵州茅台和五粮液2026H1营业收入，并看看最近新闻。",
  "thread_id": "optional-thread-id",
  "user_id": "optional-user-id"
}
```

`query` 必填且不能是空字符串。未提供的 `thread_id`、`user_id` 会在服务端各自生成 UUID；不要在浏览器端复用固定值。`thread_id` 驱动 RedisSaver session memory，`user_id` 用于显式长期 preference。

## 事件

每条消息均为：

```json
{"type": "event_type", "data": {}, "timestamp": "2026-01-01T00:00:00+00:00"}
```

| Event | 说明 |
|---|---|
| `start` | 回显本次安全生成的 thread/user ID 与 query。 |
| `plan` | 可展示的 `intent`、`required_tools`、`skill`；不包含 Planner reason 或隐藏推理。 |
| `tool_start` | 某个已计划 Tool 即将执行。 |
| `tool_end` | Tool 成功状态、聚合 latency、简短 summary；失败只给 `error_type`。 |
| `synthesis_start` | 仅 Composite Query 的 Qwen synthesis 开始。 |
| `guardrail` | `passed` 或 `degraded`。 |
| `sources` | 来自已执行 Tool 的公开 provenance；当前 FinancialRAGTool 不提供 page/source contract 时不伪造财报页码。 |
| `trace` | intent、工具计划/执行、latency 等精简可公开 trace。 |
| `token` | Guardrail 后的安全 answer chunk。 |
| `end` | 请求正常完成。 |
| `error` | 请求格式或 Agent 执行失败的稳定错误码，不返回 traceback。 |

`required_tools` 未包含的 Tool 由前端显示为未调用即可，服务端不会为每个 skipped Tool 发送冗余事件。

## 完整事件示例

```json
{"type":"start","data":{"thread_id":"...","user_id":"...","query":"从80增长到100，增长率是多少？"},"timestamp":"..."}
{"type":"plan","data":{"intent":"calculation_query","required_tools":["calculator"],"skill":"financial_report_analysis"},"timestamp":"..."}
{"type":"tool_start","data":{"tool":"calculator"},"timestamp":"..."}
{"type":"tool_end","data":{"tool":"calculator","success":true,"latency":0.0001,"summary":"确定性计算已完成。"},"timestamp":"..."}
{"type":"guardrail","data":{"status":"passed"},"timestamp":"..."}
{"type":"sources","data":[],"timestamp":"..."}
{"type":"trace","data":{"intent":"calculation_query","planned_tools":["calculator"],"executed_tools":["calculator"],"tool_latencies":{"calculator":0.0001},"total_latency":0.001},"timestamp":"..."}
{"type":"token","data":{"content":"增长率为 25%。"},"timestamp":"..."}
{"type":"end","data":{"success":true},"timestamp":"..."}
```

## Streaming 语义与安全边界

当前 Financial RAG Tool 会消费完整的 RAG answer，Composite synthesis 也使用非流式 Qwen 调用；而 deterministic Guardrail 可能替换 synthesis 文本。因此本接口提供的是 **Guardrail 后 safe chunk streaming**，不是 token-level LLM streaming。客户端不会先看到随后可能被 Guardrail 拒绝的 synthesis 内容。

News source 仅在 Tool 返回 title、source、URL 等 provenance 时序列化。Market source 使用 Tool 返回的 provider、symbol 与时间。Trace 不包含原始 prompt、Chain of Thought、完整 Parent 文本、连接信息或凭据。

WebSocket 可以连续发送多条请求。浏览器断开时服务端停止后续发送；同步 Tool 可能完成当前后台调用，但不会写入已断开的 socket。
