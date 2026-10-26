<!-- openenv-source: chat_env -->
# Chat Environment

A chat-based environment for LLMs with built-in tokenization and message history management. This environment is designed to work directly with language models and provides a minimal, flexible foundation for conversation-based RL training.

## Overview

ChatEnvironment is a lightweight environment that:
- Manages conversation history in Huggingface chat format
- Handles tokenization internally using any compatible tokenizer
- Stores both messages and tokens for efficient model interaction
- Provides a clean interface for building chat-based RL agents

ChatEnvironment can be used in **two ways**:
1. **Direct usage**: Import and use ChatEnvironment directly in your Python code (best for local development)
2. **HTTP client**: Use ChatEnv client to connect to a ChatEnvironment server (best for distributed/containerized deployments)

## Quick Start

### Option 1: Direct Usage (Local)

```python
from transformers import AutoTokenizer
from envs.chat_env import ChatAction, ChatObservation
from envs.chat_env.server import ChatEnvironment
from openenv.core.env_server import Message

# Initialize with a tokenizer and optional system prompt
tokenizer = AutoTokenizer.from_pretrained("gpt2")
env = ChatEnvironment(
    tokenizer=tokenizer,
    system_prompt="You are a helpful assistant.",
    system_role="system"
)

# Reset the environment
obs = env.reset()
print(f"Messages: {obs.messages}")
print(f"Tokens shape: {obs.tokens.shape}")

# Create an action from a message
user_message: Message = {"role": "user", "content": "Hello!"}
action = env.message_to_action(user_message)

# Step the environment
obs = env.step(action)
print(f"Updated messages: {obs.messages}")
print(f"Updated tokens shape: {obs.tokens.shape}")
```

### Option 2: HTTP Client (Distributed)

```python
from transformers import AutoTokenizer
from envs.chat_env import ChatEnv, ChatAction
import torch

# Create environment from Docker image
client = ChatEnv.from_docker_image("chat-env:latest")

# Or connect to existing server
# client = ChatEnv(base_url="http://localhost:8000")

# Reset
result = client.reset()
print(f"Initial messages: {result.observation.messages}")

# Send an action with tokens
tokenizer = AutoTokenizer.from_pretrained("gpt2")
message = {"role": "user", "content": "Hello!"}
action = client.message_to_action(message, tokenizer)

result = client.step(action)
print(f"Messages: {result.observation.messages}")
print(f"Reward: {result.reward}")

# Cleanup
client.close()
```

### Building the Docker Image

Before using the HTTP client, build the Docker image:

```bash
# From project root
docker build -t chat-env:latest -f envs/chat_env/server/Dockerfile .

# Optionally specify a different tokenizer
docker build -t chat-env:latest \
  --build-arg TOKENIZER_NAME=meta-llama/Llama-2-7b-chat-hf \
  -f envs/chat_env/server/Dockerfile .
```

## Architecture

### Data Models

#### ChatAction
Actions contain only tokens (PyTorch tensors) that interface directly with models:
```python
@dataclass
class ChatAction(Action):
    tokens: torch.Tensor  # Required, cannot be empty
```

#### ChatObservation
Observations contain both the message history and flattened tokens:
```python
@dataclass
class ChatObservation(Observation):
    messages: list[Message]  # List of {"role": str, "content": str}
    tokens: torch.Tensor     # Flattened tensor of all conversation tokens
    # Inherited: done, reward, metadata
```

#### ChatState
Internal state tracking message and token history:
```python
@dataclass
class ChatState(State):
    history_messages: list[Message]
    history_tokens: list[torch.Tensor]
    # Inherited: episode_id, step_count
```

### Key Methods

#### `reset() -> ChatObservation`
Resets the environment to initial state with optional system prompt.

#### `step(action: ChatAction) -> ChatObservation`
Takes an action (tokens), decodes to text, adds to history, returns updated observation.

#### `message_to_action(message: Message) -> ChatAction`
Convenience method to convert a message dict to a tokenized ChatAction.

## Usage Patterns

### Basic Conversation

```python
from transformers import AutoTokenizer
from envs.chat_env.server import ChatEnvironment
from openenv.core.env_server import Message

tokenizer = AutoTokenizer.from_pretrained("gpt2")
env = ChatEnvironment(tokenizer=tokenizer)

# Reset
obs = env.reset()

# User turn
user_msg: Message = {"role": "user", "content": "What is 2+2?"}
action = env.message_to_action(user_msg)
obs = env.step(action)

# Assistant turn
assistant_msg: Message = {"role": "assistant", "content": "2+2 equals 4."}
action = env.message_to_action(assistant_msg)
obs = env.step(action)

# Access conversation history
print(f"Full conversation: {obs.messages}")
print(f"All tokens: {obs.tokens}")
```

### With Transforms

You can add transforms to compute rewards or modify observations:

```python
from openenv.core.env_server import Transform, Observation

class LengthRewardTransform(Transform):
    """Reward based on response length."""

    def __call__(self, observation: Observation) -> Observation:
        if hasattr(observation, 'messages') and observation.messages:
            last_message = observation.messages[-1]
            observation.reward = len(last_message['content']) * 0.1
        return observation

env = ChatEnvironment(
    tokenizer=tokenizer,
    transform=LengthRewardTransform()
)
```

### Direct Token Usage

If you're generating tokens from a model, you can create actions directly:

```python
import torch
from envs.chat_env import ChatAction

# Assume you have tokens from your model
generated_tokens = torch.tensor([[1, 2, 3, 4, 5]])

# Create action directly
action = ChatAction(tokens=generated_tokens)

# Step environment
obs = env.step(action)
```

## Design Philosophy

ChatEnvironment is intentionally minimal and flexible:

1. **No HTTP overhead**: Works directly with Python objects and tensors
2. **Tokenizer ownership**: Environment handles tokenization consistently
3. **Dual representation**: Maintains both human-readable messages and model-ready tokens
4. **Transform support**: Extensible reward computation and observation modification
5. **Type-safe**: Uses typed Messages compatible with Huggingface format

## Integration with Models

ChatEnvironment pairs naturally with language models:

```python
# Pseudo-code for RL training loop
model = YourLanguageModel()
env = ChatEnvironment(tokenizer=model.tokenizer)

for episode in range(num_episodes):
    obs = env.reset()

    while not obs.done:
        # Model generates response tokens
        action_tokens = model.generate(obs.tokens)
        action = ChatAction(tokens=action_tokens)

        # Step environment
        obs = env.step(action)

        # Use obs.reward for RL updates
        model.update(obs.reward)
```

## Project Structure

```
chat_env/
├── __init__.py              # Module exports (ChatEnv, ChatAction, etc.)
├── README.md                # This file
├── client.py                # ChatEnv HTTP client
├── models.py                # ChatAction, ChatObservation, ChatState
└── server/
    ├── __init__.py          # Server module exports
    ├── chat_environment.py  # Core ChatEnvironment implementation
    ├── app.py               # FastAPI server application
    ├── test_chat_env.py     # Unit tests
    └── Dockerfile           # Container image for HTTP server
```

## Requirements

- Python 3.10+
- PyTorch
- A tokenizer with `apply_chat_template` method (e.g., Huggingface transformers)

## Notes

- ChatEnvironment does **not** generate responses - it only manages conversation state
- You need to provide tokens from your model or other source
- The environment is thread-safe for single-threaded use only
- For multi-turn conversations, alternate between user and assistant messages
