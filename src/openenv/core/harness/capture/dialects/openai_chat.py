"""OpenAI Chat Completions transformer with SGLang training enhancements."""

from __future__ import annotations

from typing import Any

from .base import BaseTransformer


class OpenAIChatTransformer(BaseTransformer):
    """Transform OpenAI Chat requests (passthrough + training params)."""

    def transform_request(self, body: dict[str, Any]) -> dict[str, Any]:
        result = body.copy()
        if "max_tokens" not in result and "max_completion_tokens" in result:
            result["max_tokens"] = result["max_completion_tokens"]
        return self._normalize_request(
            result,
            body.get("_served_model"),
        )

    def transform_response(
        self,
        response: dict[str, Any],
        original_request: dict[str, Any],
    ) -> dict[str, Any]:
        result = response.copy()
        if "model" in original_request:
            result["model"] = original_request["model"]
        return result

    def transform_stream_chunk(
        self,
        chunk: dict[str, Any],
        original_request: dict[str, Any],
        is_first: bool = False,
    ) -> dict[str, Any]:
        result = chunk.copy()
        if "model" in original_request:
            result["model"] = original_request["model"]
        return result
