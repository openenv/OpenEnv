"""Unit tests for the chat environment server implementation."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from envs.chat_env.models import ChatAction
from envs.chat_env.server.chat_environment import ChatEnvironment


class SimpleTokenizer:
    """Minimal tokenizer for reset/step unit tests."""

    def apply_chat_template(
        self,
        conversation,
        tokenize: bool = True,
        return_tensors: str | None = None,
        **kwargs,
    ):
        del tokenize, kwargs
        text = "".join(message["content"] for message in conversation)
        tokens = [ord(ch) % 256 for ch in text]
        if return_tensors == "pt":
            return [tokens]
        return tokens

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        del skip_special_tokens, kwargs
        if hasattr(token_ids, "tolist") and callable(token_ids.tolist):
            token_ids = token_ids.tolist()
        return "".join(chr(token) for token in token_ids)


def test_chat_reset_with_system_prompt_keeps_tokens_serializable() -> None:
    """Reset should work when a system prompt seeds token history."""
    env = ChatEnvironment(
        tokenizer=SimpleTokenizer(),
        system_prompt="You are a helpful AI assistant.",
    )

    observation = env.reset()

    assert observation.messages == [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]
    assert observation.tokens


def test_chat_reset_and_step_track_episode_id_and_step_count() -> None:
    env = ChatEnvironment(tokenizer=SimpleTokenizer())

    env.reset(episode_id="ep-123")
    assert env.state.episode_id == "ep-123"
    assert env.state.step_count == 0

    env.step(ChatAction(tokens=[72, 105]))
    env.step(ChatAction(tokens=[33]))
    assert env.state.step_count == 2

    env.reset()
    assert env.state.step_count == 0
    assert env.state.episode_id not in (None, "ep-123")

    episode_id = env.state.episode_id
    env.step(ChatAction(tokens=[33]))
    asyncio.run(env.reset_async(seed=1))
    assert env.state.step_count == 0
    assert env.state.episode_id != episode_id
