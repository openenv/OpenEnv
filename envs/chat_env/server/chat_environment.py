# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Chat Environment Implementation.

A chat-based environment for LLMs, designed as a blank canvas for conversation and RL.
"""

from openenv.core.env_server.interfaces import (
    Environment,
    Message,
    ModelTokenizer,
    Transform,
)

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import ChatAction, ChatObservation, ChatState
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import ChatAction, ChatObservation, ChatState


class ChatEnvironment(Environment):
    """A chat-based environment for LLMs, designed as a blank canvas for conversation and RL.

    This environment is designed to work with language models. It provides the fundamental structure
    for managing conversation state but is intentionally minimal to allow maximum flexibility.

    The environment owns the tokenizer and is responsible for managing both message history and tokens.
    Actions contain only tokens that interface directly with models.

    Args:
        tokenizer: A tokenizer that will be used to tokenize the conversation
        system_prompt: An optional system prompt string to use during reset calls (optional)
        system_role: The role of the system (at reset time). Defaults to "system"
        transform: Optional transform to apply to observations
    """

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        system_prompt: str | None = None,
        system_role: str = "system",
        transform: Transform | None = None,
    ):
        super().__init__(transform=transform)

        if not hasattr(tokenizer, "apply_chat_template") and not hasattr(
            tokenizer, "encode"
        ):
            raise ValueError(
                "Tokenizer must have 'apply_chat_template' or 'encode' method"
            )
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.system_role = system_role

        self._state = ChatState()

        if system_prompt:
            system_message: Message = {"role": system_role, "content": system_prompt}
            self._state.history_messages.append(system_message)
            system_tokens = self._tokenize_conversation([system_message])
            self._state.history_tokens.append(system_tokens)

    def _coerce_tokens(self, tokens) -> list[int]:
        """Normalize tokenizer outputs into a flat list of ints."""
        if hasattr(tokens, "tolist") and callable(tokens.tolist):
            tokens = tokens.tolist()

        if isinstance(tokens, tuple):
            tokens = list(tokens)

        if isinstance(tokens, list):
            flattened: list[int] = []
            for token in tokens:
                flattened.extend(self._coerce_tokens(token))
            return flattened

        return [int(tokens)]

    def _tokenize_conversation(self, conversation: list[Message]) -> list[int]:
        """Tokenize a conversation with a chat-template fallback for base tokenizers."""
        try:
            tokens = self.tokenizer.apply_chat_template(conversation=conversation, tokenize=True)
        except Exception:
            # Some tokenizers (e.g. gpt2) do not define `chat_template`.
            fallback_text = "".join(
                f"{m['role']}: {m['content']}\n" for m in conversation
            )
            if hasattr(self.tokenizer, "encode"):
                tokens = self.tokenizer.encode(fallback_text)  # type: ignore[attr-defined]
            else:
                raise ValueError("Tokenizer must support apply_chat_template or encode")

        return self._coerce_tokens(tokens)

    def reset(self) -> ChatObservation:
        """Reset the environment to initial state.

        Returns:
            ChatObservation: Initial observation with system prompt (if any)
        """
        self._state.history_messages = []
        self._state.history_tokens = []
        if self.system_prompt:
            system_message: Message = {
                "role": self.system_role,
                "content": self.system_prompt,
            }
            self._state.history_messages = [system_message]
            system_tokens = self._tokenize_conversation([system_message])
            self._state.history_tokens = [system_tokens]

        return self._create_observation()

    def step(self, action: ChatAction) -> ChatObservation:  # type: ignore[override]
        """Take a step in the environment by adding tokens to the chat history.

        Args:
            action: A ChatAction object containing tokens.

        Returns:
            ChatObservation: The updated observation with the new tokens added.
        """
        action_tokens = [int(token) for token in action.tokens]

        # Store the tokens directly from the action
        self._state.history_tokens.append(action_tokens)

        # Decode tokens to text and add as a message to history
        decoded_text = self.tokenizer.decode(action_tokens, skip_special_tokens=True)
        assistant_message: Message = {"role": "assistant", "content": decoded_text}
        self._state.history_messages.append(assistant_message)

        return self._create_observation()

    def _create_observation(self) -> ChatObservation:
        """Create a ChatObservation from the current state.

        Returns both the message history and the tokens flattened as a single tensor
        ready to be used by models.

        Returns:
            ChatObservation: Observation with messages and flattened tokens
        """
        if self._state.history_tokens:
            flattened_tokens = [
                token
                for token_list in self._state.history_tokens
                for token in token_list
            ]
        else:
            flattened_tokens = []

        observation = ChatObservation(
            messages=self._state.history_messages.copy(),  # Copy to prevent external mutation
            tokens=flattened_tokens,
        )

        transformed = self._apply_transform(observation)
        if isinstance(transformed, ChatObservation):
            return transformed
        else:
            # If transform returns base Observation, convert back to ChatObservation
            return ChatObservation(
                messages=getattr(transformed, "messages", []),
                tokens=self._coerce_tokens(getattr(transformed, "tokens", [])),
                done=transformed.done,
                reward=transformed.reward,
            )

    @property
    def state(self) -> ChatState:
        """Get the current state of the environment.

        Returns:
            ChatState: The current state.
        """
        return self._state

    def message_to_action(self, message: Message) -> ChatAction:
        """Convert a message dictionary to a ChatAction with tokens.

        Args:
            message: Dictionary with 'role' and 'content' keys

        Returns:
            ChatAction: A new ChatAction instance with tokenized content

        Raises:
            ValueError: If required keys are missing
        """
        if "role" not in message:
            raise ValueError("Message must contain a 'role' key")
        if "content" not in message:
            raise ValueError("Message must contain a 'content' key")
        if message["content"] is None:
            raise ValueError("Message content cannot be None")

        tokens = self._tokenize_conversation([message])

        return ChatAction(tokens=tokens)
