"""Reasoning round-trip helpers shared across API transformers.

SGLang's `--reasoning-parser` (split mode: `qwen3`, `minimax`, `deepseek-r1`,
etc.) splits the model's chain-of-thought into the assistant message's
`reasoning_content` field. This module helps each transform convert that
field to / from the API-specific reasoning shape:

- Anthropic:  `thinking` content block with `thinking` + `signature`
- Gemini:     part with `thought: true`, `text`, `thoughtSignature`
- Responses:  `reasoning` output item with `summary`, `content`, `encrypted_content`
- OAI Chat:   `reasoning_content` field on the assistant message (passthrough)

Signatures and `encrypted_content` only need to round-trip opaquely through
the harness (the gateway is the API server on both ends), so we use
deterministic synthetic tokens — no real cryptography is necessary.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any


def make_signature(reasoning_text: str) -> str:
    """Deterministic synthetic signature for an Anthropic/Gemini thought block."""
    if not reasoning_text:
        return ""
    digest = hashlib.sha256(reasoning_text.encode("utf-8")).digest()
    return "sg_oe_" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def make_google_signature(reasoning_text: str) -> str:
    """Encode a proxy-local thought marker as Google's JSON bytes field.

    This is only for responses synthesized by our OpenAI-to-Google bridge;
    it is not a native Gemini signature and must not replace native signatures.
    Standard padded base64 is required by clients decoding the field as bytes.
    """
    marker = make_signature(reasoning_text)
    return base64.b64encode(marker.encode("ascii")).decode("ascii")


def encrypt_reasoning(reasoning_text: str) -> str:
    """Pack reasoning into Responses-style `encrypted_content`.

    Base64-encoded so it survives transport. Decoded by `decrypt_reasoning`
    when the harness replays it on the next turn.
    """
    if not reasoning_text:
        return ""
    return "oe:" + base64.urlsafe_b64encode(reasoning_text.encode("utf-8")).decode(
        "ascii"
    )


def decrypt_reasoning(encrypted: str | None) -> str:
    """Reverse of `encrypt_reasoning`. Returns empty string on any failure."""
    if not isinstance(encrypted, str) or not encrypted.startswith("oe:"):
        return ""
    try:
        return base64.urlsafe_b64decode(encrypted[len("oe:") :].encode("ascii")).decode(
            "utf-8"
        )
    except Exception:
        return ""


def extract_reasoning_from_anthropic_content(content: Any) -> str:
    """Extract reasoning_content from Anthropic assistant content blocks."""
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "thinking":
            text = block.get("thinking", "")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts)


def extract_reasoning_from_gemini_parts(parts: Any) -> str:
    """Extract reasoning_content from Gemini content parts (thought:true)."""
    if not isinstance(parts, list):
        return ""
    pieces: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("thought") is True:
            text = part.get("text", "")
            if isinstance(text, str) and text:
                pieces.append(text)
    return "\n".join(pieces)


def extract_reasoning_from_responses_item(item: dict[str, Any]) -> str:
    """Extract reasoning_content text from a Responses `reasoning` input item.

    Prefers `content[*].text` (full chain), falls back to `summary[*].text`,
    finally tries `encrypted_content` (decoded by `decrypt_reasoning`).
    """
    content = item.get("content")
    if isinstance(content, list):
        chunks = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and isinstance(b.get("text"), str)
        ]
        joined = "\n".join(c for c in chunks if c)
        if joined:
            return joined
    summary = item.get("summary")
    if isinstance(summary, list):
        chunks = [
            b.get("text", "")
            for b in summary
            if isinstance(b, dict) and isinstance(b.get("text"), str)
        ]
        joined = "\n".join(c for c in chunks if c)
        if joined:
            return joined
    return decrypt_reasoning(item.get("encrypted_content"))
