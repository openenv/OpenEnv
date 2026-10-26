"""A small OpenAI-compatible chat helper that tolerates reasoning models.

Reasoning models (for example the GPT-5 family) reject non-default ``temperature``
and require ``max_completion_tokens``. ``chat_complete`` sends ``temperature`` only when explicitly requested and
transparently retries without it if the server rejects the value, so the same call
site works for both reasoning and non-reasoning models. It also refuses to treat a
length-truncated, empty response as a final answer.
"""

from __future__ import annotations

from typing import Any


def _is_temperature_unsupported(exc: Exception) -> bool:
    text = str(exc).lower()
    return "temperature" in text and (
        "does not support" in text or "unsupported value" in text
    )


async def chat_complete(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float | None = None,
) -> str:
    """Return the assistant text for a chat completion.

    Args:
        client:
            An `AsyncOpenAI` / `AsyncAzureOpenAI` instance.
        model (`str`):
            Deployment or model name.
        messages (`list[dict]`):
            Chat messages in OpenAI format.
        max_tokens (`int`):
            Maximum number of completion tokens.
        temperature (`float`, *optional*):
            Sampling temperature. Omitted from the request when `None`; if the model
            rejects the value (reasoning models), the call is retried without it.

    Returns:
        `str`: The assistant message content (empty string if none).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        response = await client.chat.completions.create(**kwargs)
    except Exception as exc:
        if temperature is not None and _is_temperature_unsupported(exc):
            kwargs.pop("temperature", None)
            response = await client.chat.completions.create(**kwargs)
        else:
            raise
    choice = response.choices[0]
    content = choice.message.content or ""
    if not content and getattr(choice, "finish_reason", None) == "length":
        raise RuntimeError(
            f"Model {model!r} exhausted max_completion_tokens={max_tokens} before "
            "returning visible content."
        )
    return content
