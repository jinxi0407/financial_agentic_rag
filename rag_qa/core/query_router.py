import json

from base.config import config
from base.logger import logger


class FinancialQueryRouter:
    """Use the configured LLM to decide whether a query enters Financial RAG."""

    RAG = "RAG"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    ALLOWED_ROUTES = {RAG, OUT_OF_SCOPE}

    SYSTEM_PROMPT = """你是 Financial RAG 的查询路由器。
你的唯一任务是判断用户 Query 是否应该进入金融知识库检索，不要回答用户的问题。

判为 RAG：
- 公司财报、财务指标、金融概念、财务分析
- 公司经营、风险、治理、业务模式
- 当前或未来 financial_knowledge 应覆盖的金融知识
- 即使当前知识库可能没有答案，只要属于 Financial RAG 的合理业务范围，也判为 RAG

判为 OUT_OF_SCOPE：
- 天气、旅行、编程、普通常识、写作、非金融闲聊
- 其他与 Financial RAG 无关的问题

请严格按照给定 JSON Schema 返回 route，不要生成答案或额外内容。
"""

    RESPONSE_FORMAT = {
        "type": "json_schema",
        "json_schema": {
            "name": "financial_query_route",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "route": {
                        "type": "string",
                        "enum": [RAG, OUT_OF_SCOPE],
                    }
                },
                "required": ["route"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(self, client, model=None):
        self.client = client
        self.model = model or config.LLM_MODEL

    def route(self, query):
        if self.client is None:
            return self._fallback("LLM client is not configured")
        if not self.model:
            return self._fallback("LLM router model is not configured")

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": str(query)},
                ],
                response_format=self.RESPONSE_FORMAT,
                temperature=0,
                max_tokens=32,
                stream=False,
                timeout=30,
                extra_body={"enable_thinking": False},
            )
            if not completion.choices or not completion.choices[0].message:
                raise ValueError("Router API returned no message")

            content = completion.choices[0].message.content
            if not content:
                raise ValueError("Router API returned empty content")

            payload = json.loads(content)
            if not isinstance(payload, dict) or set(payload) != {"route"}:
                raise ValueError("Router response must contain only the route field")

            route = payload["route"]
            if route not in self.ALLOWED_ROUTES:
                raise ValueError(f"Invalid router value: {route!r}")

            logger.info(f"Financial Query Router 结果: {route} (查询: '{query}')")
            return route
        except Exception as exc:
            return self._fallback(f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _fallback(reason):
        logger.warning(f"Financial Query Router 调用或解析失败，保守回退到 RAG: {reason}")
        return FinancialQueryRouter.RAG
