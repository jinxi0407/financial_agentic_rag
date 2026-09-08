"""Qwen synthesis adapter for composite Agent responses only."""
from __future__ import annotations

import json
from time import perf_counter
from typing import Any


class QwenSynthesizer:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def synthesize(self, payload: dict[str, Any]) -> tuple[str, float]:
        started = perf_counter()
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是谨慎的金融信息整合助手，只能依据提供的工具结果回答。"},
                {"role": "user", "content": self._prompt(payload)},
            ],
            timeout=30,
            stream=False,
            extra_body={"enable_thinking": False},
        )
        content = (completion.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("Qwen synthesis returned empty content")
        return content, perf_counter() - started

    @staticmethod
    def _prompt(payload: dict[str, Any]) -> str:
        return """请基于以下受控工具结果生成中文综合回答。

硬性规则：
1. 严格区分财报数据、实时行情和近期新闻。
2. 不得修改 Financial RAG 的已验证数值，不得补造任何财务、行情或新闻事实。
3. 新闻条目必须保留 source 和 published_at；不得把新闻写成财报披露。
4. 某工具失败或无结果时明确说明，不要用其他信息替代。
5. 仅可原样使用 verified_calculation；没有该结果时，不得自行计算同比、增长率、差值或百分点变化。
6. 不提供确定性的投资建议。
7. 使用以下结构（没有可用内容的章节可说明缺失）：财务表现、市场表现、近期事件、综合分析、数据来源。

工具结果：
""" + json.dumps(payload, ensure_ascii=False, indent=2)
