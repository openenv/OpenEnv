# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Chat Environment Client.

This module provides the client for connecting to a Chat Environment server
via WebSocket for persistent sessions.
"""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.interfaces import Message

from .models import ChatAction, ChatObservation, ChatState


class ChatEnv(EnvClient[ChatAction, ChatObservation, ChatState]):
    """
    Client for the Chat Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.

    The client transports token ids as plain JSON lists.

    Example:
        >>> # Connect to a running server
        >>> with ChatEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.messages)
        ...
        ...     # Send an action with tokens
        ...     tokens = [1, 2, 3, 4, 5]
        ...     result = client.step(ChatAction(tokens=tokens))
        ...     print(result.observation.messages)
        ...     print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ChatEnv.from_docker_image("chat-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(ChatAction(tokens=[1, 2, 3]))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ChatAction) -> Dict:
        """
        Convert ChatAction to JSON payload for step request.

        Args:
            action: ChatAction instance with tokens

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "tokens": action.tokens,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ChatObservation]:
        """
        Parse server response into StepResult[ChatObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with ChatObservation
        """
        obs_data = payload.get("observation", {})

        tokens_data = obs_data.get("tokens", [])
        tokens = tokens_data if isinstance(tokens_data, list) else []

        # Parse messages
        messages = obs_data.get("messages", [])

        observation = ChatObservation(
            messages=messages,
            tokens=tokens,
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict) -> ChatState:
        """
        Parse server response into ChatState object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            ChatState object with conversation history
        """
        # Parse history messages
        history_messages = payload.get("history_messages", [])

        # Parse history tokens
        history_tokens_data = payload.get("history_tokens", [])
        history_tokens = []
        for token_list in history_tokens_data:
            history_tokens.append(token_list if isinstance(token_list, list) else [])

        return ChatState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            history_messages=history_messages,
            history_tokens=history_tokens,
        )

    def message_to_action(self, message: Message, tokenizer: Any) -> ChatAction:
        """
        Helper method to convert a message to a ChatAction using a tokenizer.

        This is a client-side convenience method for users who have a tokenizer
        and want to create actions from messages.

        Args:
            message: Message dict with 'role' and 'content'
            tokenizer: Tokenizer with apply_chat_template method

        Returns:
            ChatAction with tokenized message

        Example:
            >>> from transformers import AutoTokenizer
            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> client = ChatEnv(base_url="http://localhost:8000")
            >>> message = {"role": "user", "content": "Hello!"}
            >>> action = client.message_to_action(message, tokenizer)
            >>> result = client.step(action)
        """
        if "role" not in message:
            raise ValueError("Message must contain a 'role' key")
        if "content" not in message:
            raise ValueError("Message must contain a 'content' key")
        if message["content"] is None:
            raise ValueError("Message content cannot be None")

        # Tokenize the message
        tokens = tokenizer.apply_chat_template(conversation=[message], tokenize=True)

        return ChatAction(tokens=tokens)
