"""Explicit, small Redis-backed user preferences isolated from session checkpoints."""
from __future__ import annotations

import json
import os
from hashlib import sha256

from redis import Redis


_EXPLICIT_PREFERENCE_TERMS = ("以后", "今后", "往后", "未来")
_PREFERENCE_ACTION_TERMS = ("关注", "主要看", "偏好")
_METRICS = {
    "营业收入": "revenue",
    "营收": "revenue",
    "归母净利润": "net_profit",
    "归属于母公司股东的净利润": "net_profit",
}


class RedisPreferenceStore:
    """Persist only user-declared company/metric preferences, never tool payloads."""

    def __init__(self, client: Redis | None = None):
        self.prefix = os.getenv("AGENT_REDIS_PREFIX", "financial:agent:") + "preferences:"
        self.client = client or Redis(
            host=os.getenv("AGENT_REDIS_HOST", "127.0.0.1"),
            port=int(os.getenv("AGENT_REDIS_PORT", "6380")),
            db=int(os.getenv("AGENT_REDIS_DB", "0")),
            password=os.getenv("AGENT_REDIS_PASSWORD") or None,
            decode_responses=True,
        )

    def get(self, user_id: str) -> dict:
        raw = self.client.get(self._key(user_id))
        if not raw:
            return {"preferred_companies": [], "preferred_metrics": []}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {"preferred_companies": [], "preferred_metrics": []}
        return {
            "preferred_companies": list(payload.get("preferred_companies", [])),
            "preferred_metrics": list(payload.get("preferred_metrics", [])),
        }

    def save_explicit(self, user_id: str, query: str, companies: list[dict[str, str]]) -> dict | None:
        if not self._is_explicit_preference(query):
            return None
        metrics = [value for phrase, value in _METRICS.items() if phrase in query]
        preferred_companies = [company["ticker"] for company in companies]
        if not metrics and not preferred_companies:
            return None
        current = self.get(user_id)
        payload = {
            "preferred_companies": list(dict.fromkeys([*current["preferred_companies"], *preferred_companies])),
            "preferred_metrics": list(dict.fromkeys([*current["preferred_metrics"], *metrics])),
        }
        self.client.set(self._key(user_id), json.dumps(payload, ensure_ascii=False))
        return payload

    def _key(self, user_id: str) -> str:
        return f"{self.prefix}{sha256(user_id.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _is_explicit_preference(query: str) -> bool:
        return any(term in query for term in _EXPLICIT_PREFERENCE_TERMS) and any(
            term in query for term in _PREFERENCE_ACTION_TERMS
        )
