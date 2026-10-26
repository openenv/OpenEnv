# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test suite for ChatEnvironment.

Proper unit tests with assertions to verify correct behavior.
"""

from openenv.core.env_server.interfaces import Message

from ..models import ChatAction, ChatObservation, ChatState
from .chat_environment import ChatEnvironment


class MockTokenizer:
    """Mock tokenizer for testing without requiring transformers library."""

    def apply_chat_template(
        self,
        conversation: list[Message],
        tokenize: bool = True,
        return_tensors: str | None = None,
        **kwargs,
    ):
        """Mock implementation that creates deterministic tokens from text."""
        # Concatenate all message content
        del tokenize, return_tensors, kwargs
        text = " ".join([msg["content"] for msg in conversation])

        # Create deterministic tokens based on text content
        # Use character codes modulo 256 to get valid token IDs
        tokens = [ord(c) % 256 for c in text]

        return tokens

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        """Mock decode that reverses the encoding process."""
        del skip_special_tokens, kwargs
        if hasattr(token_ids, "tolist") and callable(token_ids.tolist):
            token_ids = token_ids.tolist()

        # Reverse the encoding: convert tokens back to characters
        chars = [chr(t) for t in token_ids]
        return "".join(chars)


def test_tokenization_consistency():
    """Test that tokenizing the same string produces the same tokens."""
    tokenizer = MockTokenizer()
    env = ChatEnvironment(tokenizer=tokenizer)

    # Create the same message twice
    message1: Message = {"role": "user", "content": "Hello, world!"}
    message2: Message = {"role": "user", "content": "Hello, world!"}

    # Convert to actions
    action1 = env.message_to_action(message1)
    action2 = env.message_to_action(message2)

    # Verify tokens are identical
    assert action1.tokens == action2.tokens, (
        "Same message should produce identical tokens"
    )

    # Verify tokens are not empty
    assert len(action1.tokens) > 0, "Tokens should not be empty"

    print("✓ test_tokenization_consistency passed")


def test_message_content_preservation():
    """Test that message content is preserved in the observation."""
    tokenizer = MockTokenizer()
    env = ChatEnvironment(tokenizer=tokenizer)

    env.reset()

    # Test with user message
    user_content = "What is the capital of France?"
    user_message: Message = {"role": "user", "content": user_content}
    action = env.message_to_action(user_message)
    obs = env.step(action)

    # The last message should have the decoded content
    assert len(obs.messages) > 0, "Observation should have at least one message"
    last_message = obs.messages[-1]

    # Verify the decoded content matches what we sent
    # Note: The environment decodes the tokens, so we verify the round-trip
    decoded_content = last_message["content"]
    assert decoded_content == user_content, (
        f"Message content should be preserved. "
        f"Expected: {user_content}, Got: {decoded_content}"
    )

    # Test with assistant message
    assistant_content = "The capital of France is Paris."
    assistant_message: Message = {"role": "assistant", "content": assistant_content}
    action = env.message_to_action(assistant_message)
    obs = env.step(action)

    # Verify the last message has the assistant content
    assert len(obs.messages) >= 2, "Should have at least 2 messages now"
    last_message = obs.messages[-1]
    decoded_content = last_message["content"]
    assert decoded_content == assistant_content, (
        f"Assistant message content should be preserved. "
        f"Expected: {assistant_content}, Got: {decoded_content}"
    )

    print("✓ test_message_content_preservation passed")


def test_system_prompt_preserved():
    """Test that system prompt is preserved after reset."""
    tokenizer = MockTokenizer()
    system_prompt = "You are a helpful assistant."

    env = ChatEnvironment(tokenizer=tokenizer, system_prompt=system_prompt)

    # Check after initialization
    obs = env.reset()
    assert len(obs.messages) == 1, "Should have exactly one message (system prompt)"
    assert obs.messages[0]["role"] == "system", "First message should have system role"
    assert obs.messages[0]["content"] == system_prompt, (
        "System prompt content should match"
    )

    # Add some messages
    action = env.message_to_action({"role": "user", "content": "Hello"})
    env.step(action)

    # Reset and verify system prompt is still there
    obs = env.reset()
    assert len(obs.messages) == 1, "After reset, should only have system prompt"
    assert obs.messages[0]["content"] == system_prompt, (
        "System prompt should be preserved after reset"
    )

    print("✓ test_system_prompt_preserved passed")


def test_token_history_accumulation():
    """Test that tokens accumulate correctly in the observation."""
    tokenizer = MockTokenizer()
    env = ChatEnvironment(tokenizer=tokenizer)

    obs = env.reset()
    initial_token_count = len(obs.tokens)

    # Step with first message
    message1 = {"role": "user", "content": "Hi"}
    action1 = env.message_to_action(message1)
    obs1 = env.step(action1)
    token_count_1 = len(obs1.tokens)

    # Tokens should increase
    assert token_count_1 > initial_token_count, "Token count should increase after step"

    # Step with second message
    message2 = {"role": "assistant", "content": "Hello there"}
    action2 = env.message_to_action(message2)
    obs2 = env.step(action2)
    token_count_2 = len(obs2.tokens)

    # Tokens should continue to accumulate
    assert token_count_2 > token_count_1, (
        "Token count should keep increasing with more messages"
    )

    # Verify tokens are the concatenation of both messages
    expected_tokens = action1.tokens + action2.tokens
    assert obs2.tokens == expected_tokens, (
        "Tokens should be concatenation of all actions"
    )

    print("✓ test_token_history_accumulation passed")


def test_direct_token_action():
    """Test creating actions directly from tokens."""
    tokenizer = MockTokenizer()
    env = ChatEnvironment(tokenizer=tokenizer)

    env.reset()

    # Create raw tokens
    raw_tokens = [[72, 101, 108, 108, 111]]  # ASCII for "Hello"
    action = ChatAction(tokens=raw_tokens)

    # Step with raw tokens
    obs = env.step(action)

    # Verify message was added
    assert len(obs.messages) == 1, "Should have one message"
    assert obs.messages[0]["role"] == "assistant", "Should default to assistant role"

    # Verify tokens match what we sent (flattened)
    assert obs.tokens == [72, 101, 108, 108, 111], (
        "Observation tokens should match input tokens"
    )

    print("✓ test_direct_token_action passed")


def test_chat_models_use_message_shape():
    """Test that chat models accept the shared Message structure."""
    conversation: list[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]

    state = ChatState(history_messages=conversation)
    observation = ChatObservation(messages=conversation)

    assert state.history_messages == conversation
    assert observation.messages == conversation

    print("✓ test_chat_models_use_message_shape passed")


def test_empty_tokens_validation():
    """Test that empty tokens raise a ValueError."""
    try:
        action = ChatAction(tokens=[])
        assert False, "Should have raised ValueError for empty tokens"
    except ValueError as e:
        assert "at least 1 item" in str(e).lower(), (
            "Error message should mention the minimum token count"
        )

    print("✓ test_empty_tokens_validation passed")


def test_message_validation():
    """Test that invalid messages raise appropriate errors."""
    tokenizer = MockTokenizer()
    env = ChatEnvironment(tokenizer=tokenizer)

    # Test missing 'role' key
    try:
        env.message_to_action({"content": "test"})  # type: ignore
        assert False, "Should have raised error for missing 'role' key"
    except (ValueError, KeyError):
        pass

    # Test missing 'content' key
    try:
        env.message_to_action({"role": "user"})  # type: ignore
        assert False, "Should have raised error for missing 'content' key"
    except (ValueError, KeyError):
        pass

    # Test None content
    try:
        env.message_to_action({"role": "user", "content": None})  # type: ignore
        assert False, "Should have raised error for None content"
    except ValueError:
        pass

    print("✓ test_message_validation passed")


def test_reset_clears_history():
    """Test that reset properly clears all message and token history."""
    tokenizer = MockTokenizer()
    env = ChatEnvironment(tokenizer=tokenizer, system_prompt="System message")

    # Add some messages
    obs1 = env.reset()
    initial_messages = len(obs1.messages)

    action = env.message_to_action({"role": "user", "content": "Test message"})
    obs2 = env.step(action)

    # Verify message was added
    assert len(obs2.messages) > initial_messages, "Message should be added after step"

    # Reset
    obs3 = env.reset()

    # Verify we're back to just the system prompt
    assert len(obs3.messages) == initial_messages, (
        "Reset should clear history back to initial state"
    )
    assert obs3.messages[0]["content"] == "System message", (
        "System prompt should be preserved"
    )

    print("✓ test_reset_clears_history passed")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ChatEnvironment Test Suite")
    print("=" * 60 + "\n")

    tests = [
        test_tokenization_consistency,
        test_message_content_preservation,
        test_system_prompt_preserved,
        test_token_history_accumulation,
        test_direct_token_action,
        test_empty_tokens_validation,
        test_message_validation,
        test_reset_clears_history,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} errored: {e}")
            import traceback

            traceback.print_exc()
            failed.append(test.__name__)

    print("\n" + "=" * 60)
    if not failed:
        print(f"✓ All {len(tests)} tests passed!")
        print("=" * 60)
        return 0
    else:
        print(f"✗ {len(failed)}/{len(tests)} tests failed:")
        for name in failed:
            print(f"  - {name}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())
