# RFC: OpenEnv Basic Abstractions

**Status**: Implemented
**Created**: 10/20/2025
**Amended**: November 12, 2025
**Authors**: @Darktex, @pankit-eng, @jspisak, @zkwentz
**RFC ID:** 001

## Amendment History

**November 12, 2025**: Added two-interface model (MCP for agents, HTTP for operations), simulation layer clarity, event queues, state management, and "The Time Problem" section.

## Summary
This document defines what we call an "Environment", what its responsibilities are, and how we expect our customers to use our environments in their systems.

We will both explain *our* abstractions as well as what abstractions we expect *you* to have when working with us.

## Terminology

### The term Environment
The term "Environment" is an overloaded term in a field currently lacking established abstractions and terminology.

At a minimum, there are two meanings for the word "Environment":
1. RL environments used by the RL community (e.g. Gymnasium)
2. Execution environments for sandboxing and distribution (e.g. Docker)

In this project, we define the **environment** as "the system that the agent interacts with". This ends up including both meanings above.

### Ingredients
In general, the ingredients are at this point well established. What has not yet consolidated is how we group things.

Let's then look at the ingredients that need to belong to an abstraction and then we will introduce how we propose to group them.

1. **Tokenizer**. The model understands token IDs, not text. At some point you need to tokenize/detokenize. You _could_ have the inference server own this and simply communicate with text in/text out (e.g. like OpenAI API).
2. **Data/task loading**. Environments contain the necessary machinery to compute responses to the policy, but the policy makes its move when given a _task_ (e.g. a question). This comes from somewhere else: when training/testing, it comes from a dataset. When in production, it comes from a user while the model waits behind an endpoint.
3. **Reward pipelines/rubrics**. When training, you need this component to compute rewards. We will assume that these are data-independent, and are a property of the environment. For example, no matter the coding question, the agent always gets a reward of +1 when its code compiles. Please provide counterexamples to this if you feel that they shouldn't be.
4. **Evals**. They are similar to rewards in that they compute some score based on what the policy did, but they differ in two key ways:
    a. They are **data-dependent**. Evals are always connected to their dataset, and they can assume a specific format for it.
    b. They are **aggregated**. Unlike rewards where you get a score per-sample, here the score that matters is after aggregation.
5. **Tools**. External functions that the agent may or may not call while solving its task. They may be local or remote. These are often standardized using MCP. There are two schools of thought on whether a tool call should be a _whole_ action (traditional tool calling), or _part_ of an action (CodeAct paradigm). We will support both, *and* we will support converting from one to the other without requiring that users write their env twice.
6. **Sandbox**. Solves two issues: distribution of binaries and deps, and security. We propose a Docker-based solution (see RFC 002 for the spec).
7. **Code Execution**. We propose to make this a first-class citizen since it runs in the container and it's the single most foundational tool (especially for CodeAct). We can consider optionally disabling it based on feedback.

### Environments vs Agents
As mentioned before, an area of confusion is how to draw abstraction boundaries between Agents and Environments.

<claude: draw me an ASCII with two boxes, one being the Agent and the other being the Environment. One arrow goes from Agent to Environment and it's labeled Action, and the other goes from the Environment to the Agent and it's labeled Observation>

There are essentially two camps in OSS at the moment:

1. The "traditional" RL camp represents Agents as a thin layer on top of the model itself: they just run inference (basically this term is interchangeable with "policy"). Environments (usually following the Gym API) own almost everything: state, tools, sandboxing, evals, data loading etc.
2. Newer Agentic libraries often have a larger Agent abstraction that just owns tools and state (HF's Smolagents and Google's AgentKit do this), which makes the Agent abstraction subsume Environments.

Our proposal takes elements from both and can easily convert into either.

```
┌─────────────────┐                          ┌─────────────────┐
│                 │                          │                 │
│     Agent       │        Action            │  Environment    │
│   (Policy)      │─────────────────────────>│                 │
│                 │                          │                 │
│                 │<─────────────────────────│                 │
│                 │      Observation         │                 │
└─────────────────┘                          └─────────────────┘
```

#### Proposed Abstractions
This is the contract that we are proposing. We feel it strikes a good balance between supporting single-turn environments for LLM post-training (such as the GSM8K) while also extending to the more complex agentic tasks, such as [Tau-Bench](https://arxiv.org/abs/2406.12045). We are aiming for flexibility, so we know we may not get this right the first time. We encourage strong feedback to this RFC so that we can improve on it!

These are the key abstractions that we expect. Note that in this project we only implement the "Environment" abstraction under our meaning. You can map to other "agents" or "environment" abstractions by writing adapters to and from OpenEnvs.

Key assumptions:
1. The Environment bundles everything needed for agent interaction: tools (MCP servers), sandboxing, code execution, reward computation, tasks/datasets, and evals. This packaging makes environments self-contained and reusable.
2. We hold the state of everything **external** to the agent in the Environment. For example, if your agent defines `a = 4` with an action and wants to read `a` some time in the future, the environment will persist the interpreter state and remember variable assignments.
3. We expect a _thin_ Agent abstraction around your model that holds the state of everything pertaining to your model, such as conversation history, tokenizer etc.

```mermaid
flowchart TB
    subgraph outer["OUTER SYSTEM (RL Training Infrastructure)"]
        agent["Agent (Thin Wrapper)

        - Model/Policy
        - Tokenizer
        - Conversation History"]

        env["Environment (Docker Container)

        - MCP Servers
        - Sandbox
        - Code Execution
        - Reward Pipeline
        - External State
        - Task/Dataset Loader
        - Evals (aggregated)"]

        orchestration["RL Orchestration (Training Loop)

        - reset, step, get_state
        - Simulation control
        - Metrics and monitoring"]

        agent <-->|"MCP
        (Tool Calls)"| env
        orchestration -->|"HTTP
        (Orchestration)"| env
    end

    classDef agentBox fill:#e1f5ff,stroke:#333,stroke-width:2px
    classDef envBox fill:#fff4e1,stroke:#333,stroke-width:2px
    classDef orchBox fill:#f0f0f0,stroke:#333,stroke-width:2px

    class agent agentBox
    class env envBox
    class orchestration orchBox
```

**Key Interfaces:**
- **MCP (Agent ↔ Environment)**: Agent-environment tool interaction (training AND production)
- **HTTP (Orchestration ↔ Environment)**: Simulation control + operations (training AND production)


**Critical insight**: The Agent uses **MCP exclusively** to interact with the Environment. The HTTP interface is for orchestration (simulation control in training, operations in production), never for agent actions.

## Two Interfaces, Two Purposes

A critical insight shapes OpenEnv's architecture: **environments expose two distinct interfaces** serving fundamentally different purposes.

**1. MCP (Agent Interface)**
- Agent ↔ Environment tool interaction
- Present in training AND production
- Operations: Tool calls (`search()`, `execute_sql()`, etc.)
- **This is the ONLY interface agents use** (see RFC 005)

**2. HTTP (Service/Operations Interface)**
- RL Orchestration ↔ Environment control
- Present in training AND production (different purposes)
- Operations:
  - Training: `reset()`, `step()`, `get_state()` (simulation control)
  - Production: Health checks, metrics, logs (operations)
- **Agents NEVER access this directly**

**Key principle**: MCP for agent actions, HTTP for orchestration. See RFC 002 for detailed specification of how these interfaces work in practice, including graceful degradation from training to production.

**Special note**: Simulation control methods (`.reset()`, `.step()`) are **never** exposed as MCP tools. This ensures agents never learn they can reset reality—critical for safe production deployment.

## The Time Problem: Simulation vs Production

A critical insight that shapes our entire design:

**Simulation Time (Training/Eval)**:
- Time only advances when we say so (via `.step()`)
- Agent can "think" for arbitrary real-world time - simulation is paused
- Environment state is frozen until agent acts
- Can reset to initial state infinitely
- Code execution blocks execute atomically from environment's perspective

**Real Time (Production)**:
- Time flows continuously
- Events arrive on their own schedule (people get hired *now*, not when agent is ready)
- Agent must react with bounded latency
- Cannot reset (it's the real world). Deleting records is a one-way door.
- No "turns" in the traditional sense - continuous stream of events

**Key insight**: You can simulate production (via event queues), but you can't "productionize" simulation (can't pause reality).

This temporal duality drives the need for two distinct interfaces:
- **Simulation control** (HTTP): Reset, step, reward computation (training/eval only)
- **Agent-environment interaction** (MCP): Tool calls (training AND production)

**See RFC 006** for how we simulate production performance characteristics (latency, reliability) during training to minimize the training-production delta.

## Event Queues: First-Class Abstraction

Environments fall into two categories:

1. **Static environments**: State only changes when agent acts (chess, coding puzzles)
2. **Dynamic environments**: State changes independently (database with external events, customer service)

We make the event queue a **first-class abstraction**:
- **Empty queue** = static environment
- **Populated queue** = dynamic environment with external events

```python
class Environment:
    def __init__(
        self,
        mode: str,  # "sim" or "prod"
        mcp_servers: List[MCPServerConfig],
        event_queue: EventQueue,  # Empty for static, populated for dynamic
        ..
.
    ):
        self.event_queue = event_queue
        self.mode = mode
```

The event queue delivers external events (e.g., "new employee hired", "API request received") that change the environment state independently of agent actions. This enables realistic simulation of production scenarios where the world doesn't wait for the agent.

## State Management: Why It's Separate

**State** is a distinct concept from both **tools** and **data**:

1. **Not part of the dataset**: While datasets contain tasks, the initial state snapshot (e.g., database contents) is separate. You can have many different tasks operate on the same state snapshot!

2. **Not part of MCP tools**: Tools query and mutate state, but state itself isn't defined by MCP. MCP only deals with the interface to state.

3. **Simulation-specific reset capability**: During training, we need the ability to reset state to its original snapshot. **Crucially**, the agent absolutely cannot trigger this reset—it's exclusively for the training loop via `.reset()` (HTTP). If the agent could reset state, it would learn that every error is recoverable, creating a huge training-production delta.

**Example**: Database maintenance environment
- Initial state: SQLite database with employee records
- Agent calls `execute_sql("DELETE FROM employees")` → receives penalty in reward
- Training loop calls `env.reset()` → database restored to initial snapshot
- Agent learns not to delete records (because it can't undo the damage)

In production, there is no reset. The agent must live with consequences of its actions.

## Python Interfaces

Below are the core Python interfaces that define the contract between agents and environments.

### Core Types

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

# Type aliases
Scalar = Union[int, float, bool]


@dataclass(kw_only=True)
class Action:
    """Base class for all environment actions.

    Actions represent what the agent wants to do in the environment.
    Subclasses should add specific fields for their action type.
    """
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Observation:
    """Base class for all environment observations.

    Observations represent what the agent perceives from the environment
    after taking an action.
    """
    done: bool = False  # Whether the episode has terminated
    reward: Union[bool, int, float, None] = None  # Immediate reward signal
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class State:
    """Represents the internal state of the environment.

    This is separate from observations - the state is what the environment
    tracks internally, while observations are what gets sent to the agent.
    """
    episode_id: Optional[str] = None
    step_count: int = 0
```

### Environment Interface

```python
from abc import ABC, abstractmethod


class Environment(ABC):
    """Base class for all environments (Gym/Gymnasium compatible).

    The environment owns:
    - Tools (exposed via MCP or similar)
    - Sandbox/execution context (e.g., Docker container)
    - Code execution capabilities
    - Reward computation pipeline (mechanism TBD in Phase 2 RFC)
    - External state (e.g., interpreter variables, file system)
    """

    @abstractmethod
    def reset(self) -> Observation:
        """Reset the environment to an initial state.

        This is called at the start of each episode. It should:
        - Clear any persistent state
        - Reset the sandbox/execution context
        - Return the initial observation

        Returns:
            Initial observation for the new episode
        """
        pass

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Execute an action in the environment.

        This is where the core environment logic happens:
        - Parse the action
        - Execute any tool calls or code
        - Update internal state
        - Compute the observation
        - Compute rewards (mechanism TBD in separate RFC)

        Args:
            action: The action to execute

        Returns:
            Observation resulting from the action
        """
        pass

    @property
    @abstractmethod
    def state(self) -> State:
        """Get the current internal state of the environment.

        Returns:
            Current environment state
        """
        pass
```

### Agent Interface (Thin Wrapper)

```python
from typing import Protocol, TypedDict


class Message(TypedDict):
    """A message in a conversation.

    Compatible with Huggingface chat template format.
    """
    role: str  # e.g., "user", "assistant", "system"
    content: str


class ModelTokenizer(Protocol):
    """Protocol for tokenizers that support chat templates.

    The agent owns the tokenizer. This protocol ensures compatibility
    with Huggingface transformers and similar libraries.
    """

    def apply_chat_template(
        self,
        conversation: list[Message],
        tokenize: bool = True,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Apply a chat template to format and optionally tokenize a conversation.

        Args:
            conversation: List of message dictionaries with 'role' and 'content'
            tokenize: Whether to tokenize the output
            return_tensors: Format for returned tensors ('pt' for PyTorch)
            **kwargs: Additional arguments

        Returns:
            Formatted and optionally tokenized conversation
        """
        ...

    def decode(
        self, token_ids: Any, skip_special_tokens: bool = False, **kwargs: Any
    ) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
            **kwargs: Additional arguments

        Returns:
            Decoded text string
        """
        ...


class Agent(ABC):
    """Thin wrapper around the model/policy.

    The agent owns:
    - Model/Policy for inference
    - Tokenizer for text<->tokens conversion
    - Conversation history management

    Note: This is intentionally minimal. Complex agent logic (like
    ReAct loops, planning, etc.) should be built on top of this interface,
    not baked into it.
    """

    def __init__(self, tokenizer: ModelTokenizer):
        self.tokenizer = tokenizer
        self.conversation_history: list[Message] = []

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Generate an action based on the observation.

        This method should:
        - Convert observation to text/tokens
        - Run model inference
        - Parse model output into an action
        - Update conversation history

        Args:
            observation: Current observation from environment

        Returns:
            Action to take in the environment
        """
        pass

    def reset(self) -> None:
        """Reset agent state (e.g., clear conversation history)."""
        self.conversation_history = []
```

### Task/Dataset Interface

```python
from typing import Iterator, Generic, TypeVar
from dataclasses import dataclass
from torch.utils.data import IterableDataset

T = TypeVar('T')


@dataclass
class Task(Generic[T]):
    """Represents a single task instance.

    Tasks are provided to the environment at reset time. They contain
    the initial prompt/question and any task-specific metadata.

    This is a simple dataclass to work seamlessly with PyTorch DataLoaders.
    Subclass and add fields as needed for your specific task type.
    """
    task_id: str
    prompt: str
    ground_truth: T
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskDataset(IterableDataset, Generic[T]):
    """Iterable-style dataset abstraction for loading tasks.

    This follows PyTorch's IterableDataset interface for seamless integration
    with DataLoaders. This is separate from the environment to enable
    reuse of environments across different datasets/tasks.

    Iterable-style datasets are ideal for:
    - Sequential data access patterns
    - Streaming large datasets
    - Data that comes from external sources (databases, APIs)
    - Situations where random access is not needed or inefficient

    Example usage:
        dataset = MyTaskDataset()

        # Direct iteration
        for task in dataset:
            env.reset()
            # provide task.prompt to agent
            # ...

        # With PyTorch DataLoader
        dataloader = DataLoader(dataset, batch_size=4, num_workers=4)
        for batch in dataloader:
            # batch contains Task instances
            for task in batch:
                # process task
                pass
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Task[T]]:
        """Return an iterator over tasks in the dataset.

        Required by PyTorch IterableDataset. This method should yield
        Task instances one at a time.

        When using with DataLoader and multiple workers, each worker will
        get a separate copy of the dataset and call __iter__ independently.
        You may want to implement worker-aware splitting to avoid duplicate
        data processing.

        Returns:
            Iterator yielding Task instances
        """
        pass
```

### Evaluation Interface

> **Note**: The evaluation interface design is still under discussion. We need to
> determine how to best group evaluation logic (per-task vs aggregated, online vs
> offline, etc.). This section will be expanded in a future RFC once we've settled
> on the abstractions.

For now, we assume evaluators will:
1. Take task instances and agent trajectories as input
2. Compute per-task metrics (data-dependent)
3. Aggregate metrics across tasks
4. Support common patterns like pass@k, accuracy, etc.

Specific interfaces TBD.

### Usage Example

```python
from torch.utils.data import DataLoader

# Setup
dataset = MyTaskDataset()  # PyTorch-compatible dataset
env = MyEnvironment()  # Environment
agent = MyAgent(tokenizer=my_tokenizer)  # Thin agent wrapper

# Option 1: Iterate directly over dataset
for task in dataset:
    obs = env.reset()
    agent.reset()

    # Provide task prompt to agent (implementation-specific)
    # e.g., via initial observation or special method

    trajectory = []
    done = False

    while not done:
        action = agent.act(obs)
        obs = env.step(action)
        trajectory.append((action, obs))
        done = obs.done

    # Evaluation logic (TBD in separate RFC)
    # evaluate(task.ground_truth, trajectory)

# Option 2: Use PyTorch DataLoader for batching/shuffling
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for batch_of_tasks in dataloader:
    # Process multiple tasks in parallel
    # (requires parallel environment execution - see RFC 002)
    for task in batch_of_tasks:
        # ... same as above
        pass
```

## Design Notes

1. **Separation of concerns**: Dataset, Environment, Agent, and Evaluator are all separate abstractions that can be mixed and matched.

2. **Reward vs Eval**: Rewards are computed per-step by the environment (data-independent). Evals are computed per-episode by evaluation logic that has access to the dataset (data-dependent, aggregated). The specific abstractions for rewards and evals will be defined in separate RFCs.

3. **PyTorch DataLoader compatibility**: `TaskDataset` follows the PyTorch `IterableDataset` interface (implements `__iter__`), making it seamlessly compatible with PyTorch's `DataLoader` for streaming data, multiprocess loading, etc. This is ideal for sequential data access and large datasets.

4. **Flexibility**: Environments can support both traditional tool calling (where each tool call is a separate action) and CodeAct (where an action contains code that may call multiple tools). See RFC 005 for details on unified action interface, RFC 003 for traditional MCP integration, and RFC 004 for CodeAct.

5. **State ownership**: The Environment owns all external state (file system, interpreter state, tool outputs). The Agent owns internal state (conversation history, model hidden states, etc.).

6. **Compatibility**: These interfaces are designed to be compatible with both:
   - Traditional RL frameworks (Gym/Gymnasium API)
   - Modern agentic frameworks (Smolagents, AgentKit, LangGraph)
   - PyTorch training workflows (via DataLoader integration)
