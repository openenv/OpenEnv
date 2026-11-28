# Core API

The `openenv.core` package provides the core abstractions for building and running environments. For a walkthrough of building an environment, see [Your First Environment](../guides/first-environment).

[Simulation vs Production](../guides/simulation-vs-production) explains when OpenEnv exposes the training loop or direct MCP access, and how MCP tool calls move through `step()`, `step_async()` and `call_tool()`.

For dataset-backed environments that publish enumerable tasks and splits, see the [Task API](../guides/task-api) guide.

## Server

### Environment server primitives[[openenv.core.Message]]

#### openenv.core.Message[[openenv.core.Message]]

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L29)

A message in a conversation.

Compatible with Huggingface chat template format.

#### openenv.core.ModelTokenizer[[openenv.core.ModelTokenizer]]

```python
openenv.core.ModelTokenizer(*args, **kwargs)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L39)

Protocol for tokenizers that support chat templates.

This protocol defines the interface that tokenizers must implement
to work with chat-based environments. It's compatible with
Huggingface transformers tokenizers.

#### apply_chat_template[[openenv.core.ModelTokenizer.apply_chat_template]]

```python
apply_chat_template(conversation: list[openenv.core.env_server.interfaces.Message], tokenize: bool = True, return_tensors: str | None = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L47)

**Parameters:**

conversation (`list[Message]`) : List of message dictionaries with 'role' and 'content'.

tokenize (`bool`, *optional*, defaults to `True`) : Whether to tokenize the output.

return_tensors (`str`, *optional*) : Format for returned tensors ('pt' for PyTorch).

- ****kwargs** : Additional arguments.

**Returns:**

Formatted and optionally tokenized conversation.

Apply a chat template to format and optionally tokenize a conversation.

#### decode[[openenv.core.ModelTokenizer.decode]]

```python
decode(token_ids: Any, skip_special_tokens: bool = False, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L71)

**Parameters:**

token_ids (`Any`) : Token IDs to decode.

skip_special_tokens (`bool`, *optional*, defaults to `False`) : Whether to skip special tokens in output.

- ****kwargs** : Additional arguments.

**Returns:** `str`

Decoded text string.

Decode token IDs back to text.

#### openenv.core.Transform[[openenv.core.Transform]]

```python
openenv.core.Transform()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L201)

Transform observations to add rewards, metrics, or other modifications.

Transforms follow the TorchRL pattern where they take an observation
and return a (potentially modified) observation. This allows for
flexible reward computation and observation augmentation.

#### openenv.core.TaskProvider[[openenv.core.TaskProvider]]

```python
openenv.core.TaskProvider(*args, **kwargs)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L90)

Optional task discovery API for dataset-backed environments.

An environment implements this protocol structurally — declare the methods on
an [Environment](https://openenv.github.io/OpenEnv/main/reference/core#openenv.core.Environment) subclass, without
inheriting from `TaskProvider`. When the methods are present,
[HTTPEnvServer](https://openenv.github.io/OpenEnv/main/reference/core#openenv.core.HTTPEnvServer) exposes them as HTTP
routes under `/{env_name}/…`; when they are absent, those routes return
`501 Not Implemented`. Each method may be sync or async.

Task provider methods are for metadata/discovery only and should be
side-effect-free. They must be callable on a freshly constructed
environment instance because HTTP compatibility routes may create a
short-lived instance solely for task discovery.

Selecting a task is not part of this protocol — pass the chosen split and
index to `reset()` instead. See the
[Task API guide](https://openenv.github.io/OpenEnv/main/guides/task-api).

Examples:

```python
env.list_splits()          # ["train", "test"]
env.num_tasks("test")      # 7595
env.get_task("test", 12)   # {"id": "test-12", "index": 12}
env.reset(split="test", index=12)
```

#### get_task[[openenv.core.TaskProvider.get_task]]

```python
get_task(split: str, index: int)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L159)

**Parameters:**

split (`str`) : Task split name.

index (`int`) : Task index within the split.

**Returns:** `Any`

The task spec at that position.

**Raises:** ``IndexError``

- ``IndexError`` -- If `index` is out of range for the split. The HTTP
  route converts this to a `400 Bad Request`.

Return one task spec by split and index.

#### get_task_range[[openenv.core.TaskProvider.get_task_range]]

```python
get_task_range(split: str, start: typing.Optional[int] = None, stop: typing.Optional[int] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L178)

**Parameters:**

split (`str`) : Task split name.

start (`int`, *optional*) : Inclusive start index. Defaults to the beginning of the split.

stop (`int`, *optional*) : Exclusive stop index. Defaults to the end of the split.

**Returns:** `list[Any]`

Task specs in `[start, stop)`.

Return task specs for Python slice-style range bounds.

#### list_splits[[openenv.core.TaskProvider.list_splits]]

```python
list_splits()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L120)

**Returns:** `list[Any]`

Split descriptors. Plain strings, dicts, and Pydantic
models are all accepted; the server normalizes each entry to
`{"name": ..., "type": ...}`.

Return task split descriptors supported by this environment.

#### list_tasks[[openenv.core.TaskProvider.list_tasks]]

```python
list_tasks(split: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L131)

**Parameters:**

split (`str`) : Task split name.

**Returns:** `list[Any]`

Task specs for the split. Environments backed by very
large or streamed splits may return a bounded preview, but
`num_tasks` should still report the true total.

Return all task specs for a split.

#### num_tasks[[openenv.core.TaskProvider.num_tasks]]

```python
num_tasks(split: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L146)

**Parameters:**

split (`str`) : Task split name.

**Returns:** `int`

Number of task specs available in the split.

Return the number of task specs in a split.

#### openenv.core.Environment[[openenv.core.Environment]]

```python
openenv.core.Environment(transform: typing.Optional[openenv.core.env_server.interfaces.Transform[~ObsT]] = None, rubric: typing.Optional[ForwardRef('Rubric')] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L223)

**Parameters:**

transform (*Transform*, *optional*) : Optional transform to apply to observations.

rubric (*Rubric*, *optional*) : Optional rubric for reward computation. When provided, the rubric's output can be used to set the observation's reward in step().

SUPPORTS_CONCURRENT_SESSIONS (*bool*) : Whether this environment supports concurrent sessions. When `True`, multiple WebSocket connections can each have their own environment instance (up to `max_concurrent_envs`). When `False` (default), the environment should only be used with a single session at a time.  Set this to `True` in your subclass if the environment uses proper session isolation (unique working dirs, no shared mutable state, and external resources that can handle concurrent access).

rubric (*Rubric*, *optional*) : Optional rubric for computing rewards. Set in `__init__` and use in `step()` to compute observation rewards. Training infrastructure can access it for introspection:  ```python for name, r in env.rubric.named_rubrics(): print(f"&amp;lcub;name}: &amp;lcub;r.last_score}") ```

Base class for all environment servers following Gym/Gymnasium API.

See [rfcs/004-rubrics.md](https://github.com/openenv/OpenEnv/blob/main/rfcs/004-rubrics.md) for rubric design details.

#### close[[openenv.core.Environment.close]]

```python
close()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L465)

Clean up resources used by the environment.

Override this method to implement custom cleanup logic.
Called when the environment is being destroyed or reset.

#### get_metadata[[openenv.core.Environment.get_metadata]]

```python
get_metadata()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L322)

**Returns:**

`EnvironmentMetadata` with environment information.

Get metadata about this environment.

Override this method to provide custom metadata for the environment.
Default implementation returns basic metadata derived from class name.

#### render_web[[openenv.core.Environment.render_web]]

```python
render_web(observation: typing.Dict[str, typing.Any])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L434)

**Parameters:**

observation (`dict`) : The serialized observation, as returned by `reset()` or `step()`, plus top-level `reward` and `done`.

HTML that draws an observation in the web interface at `/web`.

Override it to show the environment's state (a board, a page, a plot).
The default draws nothing and the interface lists the observation's fields.

#### reset[[openenv.core.Environment.reset]]

```python
reset(seed: typing.Optional[int] = None, episode_id: typing.Optional[str] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L272)

Reset the environment and return initial observation.

#### reset_async[[openenv.core.Environment.reset_async]]

```python
reset_async(seed: typing.Optional[int] = None, episode_id: typing.Optional[str] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L282)

Async version of reset. Default implementation calls sync reset.

Override to provide true async implementation.

#### step[[openenv.core.Environment.step]]

```python
step(action: ~ActT, timeout_s: typing.Optional[float] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L294)

Take a step in the environment.

#### step_async[[openenv.core.Environment.step_async]]

```python
step_async(action: ~ActT, timeout_s: typing.Optional[float] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L304)

Async version of step. Default implementation calls sync step.

Override to provide true async implementation.

#### web_actions[[openenv.core.Environment.web_actions]]

```python
web_actions(observation: typing.Dict[str, typing.Any])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py#L448)

**Parameters:**

observation (`dict`) : The serialized observation, as returned by `reset()` or `step()`, plus top-level `reward` and `done`.

One-click actions for the web interface at `/web`, as `(label, action)` pairs.

Override it when the valid actions are known from the observation (the legal
moves of a game). Each action is the dict `step()` receives. The default
offers none and the interface shows the action's form only.

### Types[[openenv.core.ServerMode]]

#### openenv.core.ServerMode[[openenv.core.ServerMode]]

```python
openenv.core.ServerMode(value, names = None, module = None, qualname = None, type = None, start = 1, boundary = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L18)

Server operation mode.

#### openenv.core.HealthStatus[[openenv.core.HealthStatus]]

```python
openenv.core.HealthStatus(value, names = None, module = None, qualname = None, type = None, start = 1, boundary = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L25)

Server health status values.

#### openenv.core.WSErrorCode[[openenv.core.WSErrorCode]]

```python
openenv.core.WSErrorCode(value, names = None, module = None, qualname = None, type = None, start = 1, boundary = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L33)

WebSocket error codes for structured error handling.

#### openenv.core.Action[[openenv.core.Action]]

```python
openenv.core.Action(metadata: typing.Dict[str, typing.Any] = <factory>)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L50)

Base class for all environment actions.

All action subclasses should inherit from this base class.
Uses Pydantic for automatic validation and serialization.

#### openenv.core.Observation[[openenv.core.Observation]]

```python
openenv.core.Observation(done: bool = False, reward: bool | int | float | None = None, metadata: typing.Dict[str, typing.Any] = <factory>)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L68)

Base class for all environment observations.

All observation subclasses should inherit from this base class.
Uses Pydantic for automatic validation and serialization.

#### openenv.core.env_server.types.ResetRequest[[openenv.core.env_server.types.ResetRequest]]

```python
openenv.core.env_server.types.ResetRequest(seed: typing.Annotated[typing.Optional[int], Ge(ge=0)] = None, episode_id: typing.Annotated[typing.Optional[str], MaxLen(max_length=255)] = None, **extra_data: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L90)

Request model for environment reset.

#### openenv.core.env_server.types.ResetResponse[[openenv.core.env_server.types.ResetResponse]]

```python
openenv.core.env_server.types.ResetResponse(observation: typing.Dict[str, typing.Any], reward: typing.Optional[float] = None, done: bool = False, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L106)

Response model for environment reset.

#### openenv.core.env_server.types.StepRequest[[openenv.core.env_server.types.StepRequest]]

```python
openenv.core.env_server.types.StepRequest(action: typing.Dict[str, typing.Any], timeout_s: typing.Annotated[typing.Optional[float], Gt(gt=0)] = None, request_id: typing.Annotated[typing.Optional[str], MaxLen(max_length=255)] = None, **extra_data: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L125)

Request model for environment step.

#### openenv.core.env_server.types.StepResponse[[openenv.core.env_server.types.StepResponse]]

```python
openenv.core.env_server.types.StepResponse(observation: typing.Dict[str, typing.Any], reward: typing.Optional[float] = None, done: bool = False, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L154)

Response model for environment step.

#### openenv.core.BaseMessage[[openenv.core.BaseMessage]]

```python
openenv.core.BaseMessage()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L171)

Base class for WebSocket messages with shared configuration.

#### openenv.core.State[[openenv.core.State]]

```python
openenv.core.State(episode_id: typing.Optional[str] = None, step_count: Annotated = 0, **extra_data: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L180)

Base class for environment state.

Represents internal environment state, separate from observations.

#### openenv.core.env_server.types.CodeExecResult[[openenv.core.env_server.types.CodeExecResult]]

```python
openenv.core.env_server.types.CodeExecResult(stdout: str, stderr: str, exit_code: int)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L202)

Result of code execution containing stdout, stderr, and exit code.

#### openenv.core.env_server.types.EnvironmentMetadata[[openenv.core.env_server.types.EnvironmentMetadata]]

```python
openenv.core.env_server.types.EnvironmentMetadata(name: str, description: str, readme_content: typing.Optional[str] = None, version: typing.Optional[str] = None, author: typing.Optional[str] = None, documentation_url: typing.Optional[str] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L210)

Metadata about an environment for documentation and UI purposes.

#### openenv.core.SchemaResponse[[openenv.core.SchemaResponse]]

```python
openenv.core.SchemaResponse(action: typing.Dict[str, typing.Any], observation: typing.Dict[str, typing.Any], state: typing.Dict[str, typing.Any], reset: typing.Optional[typing.Dict[str, typing.Any]] = None, reset_observation: typing.Optional[typing.Dict[str, typing.Any]] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L227)

Response model for the combined schema endpoint.

#### openenv.core.HealthResponse[[openenv.core.HealthResponse]]

```python
openenv.core.HealthResponse(status: HealthStatus = <HealthStatus.HEALTHY: 'healthy'>)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L257)

Response model for health check endpoint.

#### openenv.core.ListTasksRequest[[openenv.core.ListTasksRequest]]

```python
openenv.core.ListTasksRequest(split: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L266)

Request model for ORS-compatible task listing.

#### openenv.core.NumTasksRequest[[openenv.core.NumTasksRequest]]

```python
openenv.core.NumTasksRequest(split: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L272)

Request model for ORS-compatible task counts.

#### openenv.core.GetTaskRequest[[openenv.core.GetTaskRequest]]

```python
openenv.core.GetTaskRequest(split: str, index: int)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L278)

Request model for ORS-compatible task lookup.

#### openenv.core.GetTaskRangeRequest[[openenv.core.GetTaskRangeRequest]]

```python
openenv.core.GetTaskRangeRequest(split: str, start: typing.Optional[int] = None, stop: typing.Optional[int] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L285)

Request model for ORS-compatible task range lookup.

#### openenv.core.WSResetMessage[[openenv.core.WSResetMessage]]

```python
openenv.core.WSResetMessage(type: typing.Literal['reset'] = 'reset', data: typing.Dict[str, typing.Any] = <factory>)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L293)

WebSocket message to reset the environment.

#### openenv.core.WSStepMessage[[openenv.core.WSStepMessage]]

```python
openenv.core.WSStepMessage(type: typing.Literal['step'] = 'step', data: typing.Dict[str, typing.Any])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L303)

WebSocket message to execute a step.

#### openenv.core.WSStateMessage[[openenv.core.WSStateMessage]]

```python
openenv.core.WSStateMessage(type: typing.Literal['state'] = 'state')
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L312)

WebSocket message to request current state.

#### openenv.core.WSCloseMessage[[openenv.core.WSCloseMessage]]

```python
openenv.core.WSCloseMessage(type: typing.Literal['close'] = 'close')
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L318)

WebSocket message to close the session.

#### openenv.core.WSObservationResponse[[openenv.core.WSObservationResponse]]

```python
openenv.core.WSObservationResponse(type: typing.Literal['observation'] = 'observation', data: typing.Dict[str, typing.Any])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L333)

WebSocket response containing an observation.

#### openenv.core.WSStateResponse[[openenv.core.WSStateResponse]]

```python
openenv.core.WSStateResponse(type: typing.Literal['state'] = 'state', data: typing.Dict[str, typing.Any])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L344)

WebSocket response containing environment state.

#### openenv.core.WSErrorResponse[[openenv.core.WSErrorResponse]]

```python
openenv.core.WSErrorResponse(type: typing.Literal['error'] = 'error', data: typing.Dict[str, typing.Any])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L353)

WebSocket response for errors.

#### openenv.core.ConcurrencyConfig[[openenv.core.ConcurrencyConfig]]

```python
openenv.core.ConcurrencyConfig(max_concurrent_envs: Annotated = 1, session_timeout: typing.Annotated[typing.Optional[float], Gt(gt=0)] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L362)

Configuration for concurrent environment sessions.

#### openenv.core.ServerCapacityStatus[[openenv.core.ServerCapacityStatus]]

```python
openenv.core.ServerCapacityStatus(active_sessions: Annotated, max_sessions: Annotated)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L377)

Status of server capacity for concurrent sessions.

#### from_counts[[openenv.core.ServerCapacityStatus.from_counts]]

```python
from_counts(active: int, max_sessions: int)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L408)

Create status from active and max session counts.

#### openenv.core.SessionInfo[[openenv.core.SessionInfo]]

```python
openenv.core.SessionInfo(session_id: str, created_at: float, last_activity_at: float, step_count: Annotated = 0, environment_type: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/types.py#L417)

Information about an active session.

### Exceptions[[openenv.core.OpenEnvError]]

#### openenv.core.OpenEnvError[[openenv.core.OpenEnvError]]

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/exceptions.py#L8)

Base exception for all OpenEnv errors.

#### openenv.core.ConcurrencyConfigurationError[[openenv.core.ConcurrencyConfigurationError]]

```python
openenv.core.ConcurrencyConfigurationError(environment_name: str, max_concurrent_envs: int, message: typing.Optional[str] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/exceptions.py#L14)

Raised when an environment is misconfigured for concurrent sessions.

This error is raised during server startup when max_concurrent_envs > 1
is specified for an environment that is not marked as SUPPORTS_CONCURRENT_SESSIONS.

#### openenv.core.SessionCapacityError[[openenv.core.SessionCapacityError]]

```python
openenv.core.SessionCapacityError(active_sessions: int, max_sessions: int, message: typing.Optional[str] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/exceptions.py#L42)

Raised when the server cannot accept new sessions due to capacity limits.

This error is raised when a new WebSocket connection is attempted but
the server has already reached max_concurrent_envs active sessions. The
default message tells the caller which server setting limits the number of
sessions, since the client (e.g. a trainer opening one session per
concurrent episode) usually cannot see the server configuration.

#### openenv.core.SessionNotFoundError[[openenv.core.SessionNotFoundError]]

```python
openenv.core.SessionNotFoundError(session_id: str, message: typing.Optional[str] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/exceptions.py#L77)

Raised when attempting to access a session that does not exist.

#### openenv.core.SessionCreationError[[openenv.core.SessionCreationError]]

```python
openenv.core.SessionCreationError(reason: str, message: typing.Optional[str] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/exceptions.py#L89)

Raised when a session cannot be created.

#### openenv.core.EnvironmentFactoryError[[openenv.core.EnvironmentFactoryError]]

```python
openenv.core.EnvironmentFactoryError(factory_name: str, message: typing.Optional[str] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/exceptions.py#L101)

Raised when the environment factory fails to create an instance.

### HTTP server utilities[[openenv.core.HTTPEnvServer]]

#### openenv.core.HTTPEnvServer[[openenv.core.HTTPEnvServer]]

```python
openenv.core.HTTPEnvServer(env: Callable[[], Environment], action_cls: Type[Action], observation_cls: Type[Observation], max_concurrent_envs: Optional[int] = None, concurrency_config: Optional[ConcurrencyConfig] = None, env_name: Optional[str] = None, state_cls: Type[State] = <class 'openenv.core.env_server.types.State'>, reset_observation_cls: Optional[Type[Observation]] = None, reset_cls: Type[ResetRequest] = <class 'openenv.core.env_server.types.ResetRequest'>)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/http_server.py#L153)

HTTP server wrapper for Environment instances.

This class wraps an Environment and exposes its reset(), step(), and state
methods as HTTP and WebSocket endpoints compatible with EnvClient.

The server expects:
- Action deserialization: Converts JSON dict to Action subclass
- Observation serialization: Converts Observation subclass to JSON dict

Examples:

```python
from core.env_server import HTTPEnvServer
from envs.coding_env.server import CodeExecutionEnvironment
from envs.coding_env.models import CodeAction, CodeObservation

# Pass environment class (factory pattern)
server = HTTPEnvServer(
    env=CodeExecutionEnvironment,
    action_cls=CodeAction,
    observation_cls=CodeObservation,
    max_concurrent_envs=4,
)

# Register routes with FastAPI
from fastapi import FastAPI
app = FastAPI()
server.register_routes(app)
```

#### get_capacity_status[[openenv.core.HTTPEnvServer.get_capacity_status]]

```python
get_capacity_status()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/http_server.py#L357)

**Returns:**

`ServerCapacityStatus` with current session counts and availability.

Get the current capacity status of the server.

#### get_session_info[[openenv.core.HTTPEnvServer.get_session_info]]

```python
get_session_info(session_id: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/http_server.py#L783)

**Parameters:**

session_id (`str`) : The session ID to query.

**Returns:**

`SessionInfo` if the session exists, `None` otherwise.

Get information about a specific session.

#### register_routes[[openenv.core.HTTPEnvServer.register_routes]]

```python
register_routes(app: FastAPI, mode: ServerMode | str = <ServerMode.SIMULATION: 'simulation'>)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/http_server.py#L1058)

**Parameters:**

app (`FastAPI`) : FastAPI application instance.

mode (`ServerMode` or `str`, *optional*, defaults to `ServerMode.SIMULATION`) : Server mode. In production mode, simulation control endpoints (/reset, /step, /state) are NOT registered. Only safe endpoints (/health, /schema, /metadata, /ws) are available.

**Raises:** ``ValueError``

- ``ValueError`` -- If `mode` is not a valid `ServerMode` or string equivalent.

Register HTTP routes on a FastAPI application.

#### openenv.core.create_app[[openenv.core.create_app]]

```python
openenv.core.create_app(env: Callable[[], Environment], action_cls: Type[Action], observation_cls: Type[Observation], env_name: Optional[str] = None, max_concurrent_envs: Optional[int] = None, concurrency_config: Optional[ConcurrencyConfig] = None, gradio_builder: Optional[Callable[..., Any]] = None, custom_tab_name: str = 'Custom', custom_tab_primary: bool = False, show_default_tab: bool = True, title_override: Optional[str] = None, state_cls: Type[State] = <class 'openenv.core.env_server.types.State'>, mode: Optional[ServerMode | str] = None, reset_observation_cls: Optional[Type[Observation]] = None, reset_cls: Type[ResetRequest] = <class 'openenv.core.env_server.types.ResetRequest'>)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/http_server.py#L2409)

**Parameters:**

env (`Callable[[], Environment]`) : Environment factory (callable) that creates new instances.

action_cls (`Type[Action]`) : The Action subclass this environment expects.

observation_cls (`Type[Observation]`) : The Observation subclass returned by step.

reset_observation_cls (`Type[Observation]`, *optional*) : The reset observation model published in `/schema`. Defaults to `observation_cls`; declare a distinct model when reset differs.

env_name (`str`, *optional*) : Environment name for README loading.

max_concurrent_envs (`int`, *optional*) : Maximum concurrent WebSocket sessions. Mutually exclusive with `concurrency_config`.

concurrency_config (`ConcurrencyConfig`, *optional*) : Advanced concurrency settings. Mutually exclusive with `max_concurrent_envs`.

gradio_builder (`Callable`, *optional*) : Callable to build a custom Gradio UI at /web. Signature: `(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md) -> gr.Blocks`. When `None`, the default Gradio app is used.

custom_tab_name (`str`, *optional*, defaults to `"Custom"`) : Label for the env-specific tab when `gradio_builder` is provided.

custom_tab_primary (`bool`, *optional*, defaults to `False`) : When `True`, the env-specific tab is active first; the auto-generated Playground becomes secondary.

show_default_tab (`bool`, *optional*, defaults to `True`) : When `False`, mount the env's `gradio_builder` output alone (no auto-generated Playground, no tab chrome). Only meaningful when `gradio_builder` is provided.

title_override (`str`, *optional*) : If set, used as the Gradio app title instead of the default `"OpenEnv Agentic Environment: {name}"`.

state_cls (`Type[State]`, *optional*, defaults to `State`) : The `State` subclass this environment reports, used for the `/state` response model and the `state` entry of `/schema`.

reset_cls (`Type[ResetRequest]`, *optional*, defaults to `ResetRequest`) : The `ResetRequest` subclass describing this environment's reset parameters, published as the `reset` entry of `/schema`. Not used for validation.

mode (`ServerMode` or `str`, *optional*) : Server mode. When `None`, resolved from the `OPENENV_MODE` environment variable, defaulting to simulation.

**Returns:**

`FastAPI` application instance with or without web interface and README integration.

Create a FastAPI application with or without web interface.

This function creates a FastAPI app with the web interface enabled by default,
including README integration for better user experience.

#### openenv.core.create_fastapi_app[[openenv.core.create_fastapi_app]]

```python
openenv.core.create_fastapi_app(env: Callable[[], Environment], action_cls: Type[Action], observation_cls: Type[Observation], max_concurrent_envs: Optional[int] = None, concurrency_config: Optional[ConcurrencyConfig] = None, env_name: Optional[str] = None, state_cls: Type[State] = <class 'openenv.core.env_server.types.State'>, mode: Optional[ServerMode | str] = None, reset_observation_cls: Optional[Type[Observation]] = None, reset_cls: Type[ResetRequest] = <class 'openenv.core.env_server.types.ResetRequest'>)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/http_server.py#L2525)

**Parameters:**

env (`Callable[[], Environment]`) : Environment factory (callable) that creates new instances.

action_cls (`Type[Action]`) : The Action subclass this environment expects.

observation_cls (`Type[Observation]`) : The Observation subclass returned by step.

reset_observation_cls (`Type[Observation]`, *optional*) : The reset observation model published in `/schema`. Defaults to `observation_cls`; declare a distinct model when reset differs.

max_concurrent_envs (`int`, *optional*) : Maximum concurrent WebSocket sessions. Mutually exclusive with `concurrency_config`.

concurrency_config (`ConcurrencyConfig`, *optional*) : Advanced concurrency settings. Mutually exclusive with `max_concurrent_envs`.

env_name (`str`, *optional*) : Optional environment name for task/split endpoints.

state_cls (`Type[State]`, *optional*, defaults to `State`) : The `State` subclass this environment reports, used for the `/state` response model and the `state` entry of `/schema`.

reset_cls (`Type[ResetRequest]`, *optional*, defaults to `ResetRequest`) : The `ResetRequest` subclass describing this environment's reset parameters, published as the `reset` entry of `/schema`. Not used for validation.

mode (`ServerMode` or `str`, *optional*) : Server mode. When `None`, resolved from the `OPENENV_MODE` environment variable, defaulting to simulation.

**Returns:**

`FastAPI` application instance.

Create a FastAPI application with comprehensive documentation.

### Web interface helpers[[openenv.core.env_server.web_interface.ActionLog]]

#### openenv.core.env_server.web_interface.ActionLog[[openenv.core.env_server.web_interface.ActionLog]]

```python
openenv.core.env_server.web_interface.ActionLog(timestamp: str, action: typing.Dict[str, typing.Any], observation: typing.Dict[str, typing.Any], reward: typing.Optional[float] = None, done: bool, step_count: int)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/web_interface.py#L255)

Log entry for an action taken.

#### openenv.core.env_server.web_interface.EpisodeState[[openenv.core.env_server.web_interface.EpisodeState]]

```python
openenv.core.env_server.web_interface.EpisodeState(episode_id: typing.Optional[str] = None, step_count: int, current_observation: typing.Optional[typing.Dict[str, typing.Any]] = None, action_logs: typing.List[openenv.core.env_server.web_interface.ActionLog] = <factory>, is_reset: bool = True)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/web_interface.py#L270)

Current episode state for the web interface.

#### openenv.core.WebInterfaceManager[[openenv.core.WebInterfaceManager]]

```python
openenv.core.WebInterfaceManager(env: Environment, action_cls: Type[Action], observation_cls: Type[Observation], metadata: Optional[EnvironmentMetadata] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/web_interface.py#L288)

Manages the web interface for an environment.

#### connect_websocket[[openenv.core.WebInterfaceManager.connect_websocket]]

```python
connect_websocket(websocket: WebSocket)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/web_interface.py#L353)

Connect a new WebSocket client.

#### disconnect_websocket[[openenv.core.WebInterfaceManager.disconnect_websocket]]

```python
disconnect_websocket(websocket: WebSocket)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/web_interface.py#L361)

Disconnect a WebSocket client.

#### get_state[[openenv.core.WebInterfaceManager.get_state]]

```python
get_state()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/web_interface.py#L466)

Get current environment state.

#### reset_environment[[openenv.core.WebInterfaceManager.reset_environment]]

```python
reset_environment(reset_kwargs: Optional[Dict[str, Any]] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/web_interface.py#L388)

Reset the environment and update state.

#### step_environment[[openenv.core.WebInterfaceManager.step_environment]]

```python
step_environment(action_data: Dict[str, Any])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/web_interface.py#L423)

Execute a step in the environment and update state.

#### openenv.core.create_web_interface_app[[openenv.core.create_web_interface_app]]

```python
openenv.core.create_web_interface_app(env: Environment, action_cls: Type[Action], observation_cls: Type[Observation], env_name: Optional[str] = None, max_concurrent_envs: Optional[int] = None, concurrency_config: Optional[Any] = None, gradio_builder: Optional[Callable[..., Any]] = None, custom_tab_name: str = 'Custom', custom_tab_primary: bool = False, show_default_tab: bool = True, title_override: Optional[str] = None, state_cls: Type[State] = <class 'openenv.core.env_server.types.State'>, mode: Optional[Any] = None, reset_observation_cls: Optional[Type[Observation]] = None, reset_cls: Type[ResetRequest] = <class 'openenv.core.env_server.types.ResetRequest'>)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/web_interface.py#L472)

**Parameters:**

env : The Environment instance to serve

action_cls : The Action subclass this environment expects

observation_cls : The Observation subclass returned by step

reset_observation_cls : The reset observation model published in /schema. Defaults to observation_cls.

env_name : Optional environment name for README loading

max_concurrent_envs : Maximum concurrent WebSocket sessions

concurrency_config : Optional ConcurrencyConfig for advanced concurrency settings

gradio_builder : Optional callable (web_manager, action_fields, metadata, is_chat_env, title, quick_start_md) -> gr.Blocks to use instead of the default Gradio UI. Lets envs replace or customize the /web interface.

custom_tab_name : Label shown on the env-specific tab when `gradio_builder` is provided. Defaults to `"Custom"` for backwards compatibility; envs that ship a rich custom UI should pass a descriptive name (e.g. `"REPL"`). Ignored when `show_default_tab=False` (no tab chrome is rendered).

custom_tab_primary : When True, the env-specific tab is rendered first and selected by default; the auto-generated Playground becomes secondary. Use this for envs whose custom tab is the real interaction surface (so visitors don't land on a less informative schema form). Ignored when `show_default_tab=False`.

show_default_tab : When False, the auto-generated Playground tab is not rendered and the env's `gradio_builder` output is mounted directly (single-view UI, no tab chrome). Only meaningful when `gradio_builder` is provided.

title_override : If set, used verbatim as the Gradio app/browser-tab title instead of the default `"OpenEnv Agentic Environment: &amp;lcub;name}"`.

state_cls : The State subclass this environment reports. Used for the /state response model and the state entry of /schema. Defaults to State.

reset_cls : The ResetRequest subclass describing reset parameters. Published as the reset entry of /schema; not used for validation. Defaults to ResetRequest.

mode : Server mode (`ServerMode` or string). When `None`, resolved from the `OPENENV_MODE` environment variable, defaulting to simulation.

**Returns:**

FastAPI application instance with web interface

Create a FastAPI application with web interface for the given environment.

### Serialization[[openenv.core.deserialize_action]]

#### openenv.core.deserialize_action[[openenv.core.deserialize_action]]

```python
openenv.core.deserialize_action(action_data: typing.Dict[str, typing.Any], action_cls: typing.Type[openenv.core.env_server.types.Action])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/serialization.py#L43)

**Parameters:**

action_data (*dict*) : Dictionary containing action data.

action_cls (*type*) : The Action subclass to instantiate.

**Returns:**

*Action* instance.

**Raises:** `ValidationError`

- `ValidationError` -- If *action_data* is invalid for the action class.

Convert JSON dict to Action instance using Pydantic validation.

MCP action types (`list_tools`, `call_tool`) are recognised
automatically via the `"type"` discriminator field, regardless of
the environment's configured `action_cls`.  All other payloads
fall through to `action_cls.model_validate()`.

For special cases (e.g., tensor fields, custom type conversions),
use deserialize_action_with_preprocessing().

#### openenv.core.deserialize_action_with_preprocessing[[openenv.core.deserialize_action_with_preprocessing]]

```python
openenv.core.deserialize_action_with_preprocessing(action_data: typing.Dict[str, typing.Any], action_cls: typing.Type[openenv.core.env_server.types.Action])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/serialization.py#L74)

**Parameters:**

action_data (`dict`) : Dictionary containing action data.

action_cls (`type`) : The Action subclass to instantiate.

**Returns:**

`Action` instance.

**Raises:** ``ValidationError``

- ``ValidationError`` -- If `action_data` is invalid for the action class.

Convert JSON dict to Action instance with preprocessing for special types.

This version handles common type conversions needed for web interfaces:
- Converting JSON string arguments to dict for MCP call_tool actions
- Converting lists/strings to tensors for 'tokens' field
- Converting string action_id to int
- Other custom preprocessing as needed

#### openenv.core.serialize_observation[[openenv.core.serialize_observation]]

```python
openenv.core.serialize_observation(observation: Observation)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/serialization.py#L147)

**Parameters:**

observation (`Observation`) : Observation instance to serialize.

**Returns:** `dict` compatible with `EnvClient._parse_result()`, with keys

- `observation` (`dict`): Observation fields.
- `reward` (`float` or `None`): Reward value.
- `done` (`bool`): Whether the episode is done.
- `metadata` (`dict`, *optional*): Additional observation metadata.

Convert Observation instance to JSON-compatible dict using Pydantic.

### Transforms[[openenv.core.CompositeTransform]]

#### openenv.core.CompositeTransform[[openenv.core.CompositeTransform]]

```python
openenv.core.CompositeTransform(transforms: list[openenv.core.env_server.interfaces.Transform])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/base_transforms.py#L9)

Combines multiple transforms into a single transform.

#### openenv.core.NullTransform[[openenv.core.NullTransform]]

```python
openenv.core.NullTransform()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/base_transforms.py#L21)

Default transform that passes through unchanged.

### Route configuration[[openenv.core.GetEndpointConfig]]

#### openenv.core.GetEndpointConfig[[openenv.core.GetEndpointConfig]]

```python
openenv.core.GetEndpointConfig(path: str, handler: typing.Callable[[], pydantic.main.BaseModel | dict], response_model: typing.Union[typing.Type[pydantic.main.BaseModel], type[dict]], tag: str, summary: str, description: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/route_config.py#L18)

Configuration for a simple GET endpoint.

#### openenv.core.env_server.route_config.register_get_endpoints[[openenv.core.env_server.route_config.register_get_endpoints]]

```python
openenv.core.env_server.route_config.register_get_endpoints(app: FastAPI, configs: typing.List[openenv.core.env_server.route_config.GetEndpointConfig])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/route_config.py#L40)

**Parameters:**

app (`~fastapi.FastAPI`) : FastAPI application instance.

configs (`List[GetEndpointConfig]`) : List of GET endpoint configurations.

Register multiple GET endpoints from configuration.

## Clients

### Base client[[openenv.core.EnvClient]]

#### openenv.core.EnvClient[[openenv.core.EnvClient]]

```python
openenv.core.EnvClient(base_url: Optional[str] = None, connect_timeout_s: float = 10.0, message_timeout_s: float = 60.0, max_message_size_mb: float = 100.0, websocket_ping_interval_s: Optional[float] = 20.0, websocket_ping_timeout_s: Optional[float] = 20.0, provider: Optional['ContainerProvider | RuntimeProvider'] = None, mode: Optional[str] = None, headers: Optional[Dict[str, str]] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_client.py#L287)

Async environment client for persistent sessions.

This client maintains a persistent WebSocket connection to an environment
server, enabling efficient multi-step interactions. Each client instance
corresponds to a dedicated environment session on the server.

The client is async by default. For synchronous usage, use the `.sync()`
method to get a `SyncEnvClient` wrapper.

Features:
- Lower latency for sequential interactions
- Session state is maintained server-side
- Better suited for long-running episodes
- Async by default for modern Python async/await patterns

Examples:

Async usage:

```python
from envs.coding_env.client import CodingEnv

async with CodingEnv(base_url="ws://localhost:8000") as env:
    result = await env.reset(seed=42)
    while not result.done:
        action = agent.predict(result.observation)
        result = await env.step(action)
```

Sync usage via `.sync()` wrapper:

```python
env = CodingEnv(base_url="ws://localhost:8000").sync()
with env:
    result = env.reset(seed=42)
    result = env.step(action)
```

#### from_docker_image[[openenv.core.EnvClient.from_docker_image]]

```python
from_docker_image(image: str, provider: Optional['ContainerProvider'] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_client.py#L803)

**Parameters:**

image (`str`) : Docker image name to run (e.g., `"coding-env:latest"`).

provider (`ContainerProvider`, *optional*) : Container provider to use. Defaults to `LocalDockerProvider`.

- ****kwargs** : Additional arguments to pass to `provider.start_container()`.

**Returns:** `_BootstrapResult`

`await` for a connected async client, or call
`.sync()` for a connected `SyncEnvClient`.

Create an environment client by spinning up a Docker container.

Returns a bootstrap handle so the same call works from both async and
synchronous code: `await` it for the connected async client, or chain
`.sync()` for a connected `SyncEnvClient` (e.g. a TRL GRPO rollout loop
that cannot `await`). Bootstrap is lazy — the container starts when the
handle is resolved.

Examples:

```python
# Async
env = await MyEnv.from_docker_image("coding-env:latest")

# Sync
env = MyEnv.from_docker_image("coding-env:latest").sync()
result = env.reset()
```

#### from_env[[openenv.core.EnvClient.from_env]]

```python
from_env(repo_id: str, use_docker: bool = True, provider: Optional['ContainerProvider | RuntimeProvider'] = None, **provider_kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_client.py#L923)

**Parameters:**

repo_id (`str`) : Hugging Face space identifier `{org}/{space}`.

use_docker (`bool`, *optional*, defaults to `True`) : When `True`, pull from the HF registry and launch via `LocalDockerProvider`. When `False`, run the space locally with `UVProvider`.

provider (`ContainerProvider` or `RuntimeProvider`, *optional*) : Provider instance to reuse. Must be a `ContainerProvider` when `use_docker=True` and a `RuntimeProvider` otherwise.

- ****provider_kwargs** : Additional keyword arguments forwarded to either the container provider's `start_container` (docker) or to the `UVProvider` constructor/start (uv). When `use_docker=False`, the `project_path` argument can be used to override the default git URL (`git+https://huggingface.co/spaces/{repo_id}`). With `use_docker=False` the server doesn't inherit variables named like credentials (such as `HF_TOKEN`): pass the ones it needs in `env_vars`.

**Returns:** `_BootstrapResult`

`await` for a connected async client, or call
`.sync()` for a connected `SyncEnvClient`.

Create a client from a Hugging Face Space.

Returns a bootstrap handle: `await` it for the connected async client, or
chain `.sync()` for a connected `SyncEnvClient`. Bootstrap is lazy — the
Space starts when the handle is resolved.

Examples:

```python
# Async: pull and run from HF Docker registry
env = await MyEnv.from_env("openenv/echo-env")

# Sync: chain .sync()
env = MyEnv.from_env("openenv/echo-env").sync()

# Run locally with UV (clones the space)
env = await MyEnv.from_env("openenv/echo-env", use_docker=False)

# Run from a local checkout
env = await MyEnv.from_env(
    "openenv/echo-env",
    use_docker=False,
    project_path="/path/to/local/checkout"
)
```

#### new_session[[openenv.core.EnvClient.new_session]]

```python
new_session()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_client.py#L493)

**Returns:** `EnvClient`

A connected child client of the same concrete type.

Create and connect a new session against the same environment server.

The child session is tracked by this parent and closed when the parent
is closed. Server-side capacity still applies: when the server is at
`MAX_CONCURRENT_ENVS`, opening the child WebSocket can fail and is
surfaced as a connection error.

#### sync[[openenv.core.EnvClient.sync]]

```python
sync()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_client.py#L1222)

**Returns:**

SyncEnvClient wrapper that provides synchronous methods

Return a synchronous wrapper around this async client.

Use this method when you need synchronous access to the environment
without async/await syntax. This is useful for:
- Integration with synchronous codebases
- Interactive/REPL usage
- Stopping async from "infecting" the call stack

Examples:

```python
async_client = GenericEnvClient(base_url="http://localhost:8000")
sync_client = async_client.sync()

with sync_client:
    result = sync_client.reset()
    result = sync_client.step({"code": "print('hello')"})
```

### Synchronous client[[openenv.SyncEnvClient]]

#### openenv.SyncEnvClient[[openenv.SyncEnvClient]]

```python
openenv.SyncEnvClient(async_client: 'EnvClient[ActT, ObsT, StateT]')
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/sync_client.py#L43)

**Parameters:**

_async : The wrapped async EnvClient instance

Synchronous wrapper around an async EnvClient.

This class provides a synchronous interface to an async EnvClient,
making it easier to use in synchronous code or to stop async from
"infecting" the entire call stack.

The wrapper executes async operations on a dedicated background event loop
so connection state remains bound to a single loop.

For guaranteed resource cleanup, use `with SyncEnvClient(...)` or call
`close()` explicitly. `__del__` is best-effort only and may not run
reliably (for example, during interpreter shutdown).

Examples:

```python
# From an async client
async_client = GenericEnvClient(base_url="http://localhost:8000")
sync_client = async_client.sync()

# Use synchronous context manager
with sync_client:
    result = sync_client.reset()
    result = sync_client.step({"action": "test"})
```

#### close[[openenv.SyncEnvClient.close]]

```python
close()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/sync_client.py#L219)

Close the connection and clean up resources.

#### connect[[openenv.SyncEnvClient.connect]]

```python
connect()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/sync_client.py#L163)

**Returns:**

self for method chaining

Establish connection to the server.

#### disconnect[[openenv.SyncEnvClient.disconnect]]

```python
disconnect()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/sync_client.py#L174)

Close the connection.

#### new_session[[openenv.SyncEnvClient.new_session]]

```python
new_session()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/sync_client.py#L231)

**Returns:** `SyncEnvClient`

A connected child wrapper around a child async
client of the same concrete type.

Create a new synchronous session against the same environment server.

The child session is tracked by this parent and closed when the parent
is closed. Call this after the parent has connected, because the child
reuses the parent's current base URL. Server-side capacity still
applies: when the server is at `MAX_CONCURRENT_ENVS`, opening the child
WebSocket can fail and is surfaced as a connection error.

#### reset[[openenv.SyncEnvClient.reset]]

```python
reset(**kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/sync_client.py#L179)

**Parameters:**

- ****kwargs** : Optional parameters passed to the environment's reset method.

**Returns:**

StepResult containing initial observation

Reset the environment.

#### state[[openenv.SyncEnvClient.state]]

```python
state()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/sync_client.py#L209)

**Returns:**

State object with environment state information

Get the current environment state.

#### step[[openenv.SyncEnvClient.step]]

```python
step(action: ActT, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/sync_client.py#L193)

**Parameters:**

action : The action to execute.

- ****kwargs** : Optional parameters.

**Returns:**

StepResult containing observation, reward, and done status

Execute an action in the environment.

### Generic client[[openenv.GenericEnvClient]]

#### openenv.GenericEnvClient[[openenv.GenericEnvClient]]

```python
openenv.GenericEnvClient(base_url: Optional[str] = None, connect_timeout_s: float = 10.0, message_timeout_s: float = 60.0, max_message_size_mb: float = 100.0, websocket_ping_interval_s: Optional[float] = 20.0, websocket_ping_timeout_s: Optional[float] = 20.0, provider: Optional['ContainerProvider | RuntimeProvider'] = None, mode: Optional[str] = None, headers: Optional[Dict[str, str]] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/generic_client.py#L17)

Environment client that works with raw dictionaries instead of typed classes.

This client doesn't require installing environment-specific packages, making it
ideal for:
- Connecting to remote servers without installing their packages
- Quick prototyping and testing
- Environments where type safety isn't needed
- Security-conscious scenarios where you don't want to run remote code

The trade-off is that you lose type safety and IDE autocomplete for actions
and observations. Instead of typed objects, you work with plain dictionaries.

Examples:

```python
# Direct connection to a running server (no installation needed)
with GenericEnvClient(base_url="http://localhost:8000") as env:
    result = env.reset()
    result = env.step({"code": "print('hello')"})
    print(result.observation)  # Dict[str, Any]
    print(result.observation.get("output"))

# From local Docker image (chain .sync() for synchronous use)
env = GenericEnvClient.from_docker_image("coding-env:latest").sync()
result = env.reset()
result = env.step({"code": "x = 1 + 2"})
env.close()

# From HuggingFace Hub (pulls Docker image, no pip install)
env = GenericEnvClient.from_env("user/my-env", use_docker=True).sync()
result = env.reset()
env.close()
```

`GenericEnvClient` inherits `from_docker_image()` and `from_env()` from
`EnvClient`, so you can use it with Docker containers and HuggingFace
Spaces without any package installation.

#### openenv.GenericAction[[openenv.GenericAction]]

```python
openenv.GenericAction(**kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/generic_client.py#L126)

A dictionary subclass for creating actions when using GenericEnvClient.

This provides a semantic wrapper around dictionaries to make code more
readable when working with GenericEnvClient. It behaves exactly like a
dict but signals intent that this is an action for an environment.

Examples:

```python
# Without GenericAction (works fine)
env.step({"code": "print('hello')"})

# With GenericAction (more explicit)
action = GenericAction(code="print('hello')")
env.step(action)

# With multiple fields
action = GenericAction(code="x = 1", timeout=30, metadata={"tag": "test"})
env.step(action)
```

`GenericAction` is just a dict with a constructor that accepts keyword
arguments. It's provided for symmetry with typed Action classes and
to make code more readable.

### LLM client[[openenv.core.ToolCall]]

#### openenv.core.ToolCall[[openenv.core.ToolCall]]

```python
openenv.core.ToolCall(id: str, name: str, args: dict[str, Any])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/llm_client.py#L106)

A single tool/function call returned by the model.

#### openenv.core.LLMResponse[[openenv.core.LLMResponse]]

```python
openenv.core.LLMResponse(content: str, tool_calls: list[ToolCall] = <factory>)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/llm_client.py#L115)

Normalized response from an LLM, with optional tool calls.

#### to_message_dict[[openenv.core.LLMResponse.to_message_dict]]

```python
to_message_dict()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/llm_client.py#L121)

Convert to an OpenAI-format assistant message dict.

#### openenv.core.LLMClient[[openenv.core.LLMClient]]

```python
openenv.core.LLMClient(endpoint: str, port: int | None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/llm_client.py#L139)

**Parameters:**

endpoint (`str`) : The `http(s)` base URL of the LLM service (e.g. "http://localhost"). May include a port and a path (e.g. "http://localhost:8000/v1"). Credentials, query strings and fragments are rejected.

port (`int` or `None`) : The port the service listens on. Appended to `endpoint` when the URL does not name one; must match the URL's port when both are given.

Abstract base for LLM endpoint clients.

Subclass and implement `complete()` for your protocol.

#### complete[[openenv.core.LLMClient.complete]]

```python
complete(prompt: str, **kwargs)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/llm_client.py#L158)

**Parameters:**

prompt (`str`) : The user prompt to send.

- ****kwargs** : Override default parameters (temperature, max_tokens, etc.).

**Returns:**

The model's text response.

Send a prompt, return the text response.

#### complete_with_tools[[openenv.core.LLMClient.complete_with_tools]]

```python
complete_with_tools(messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/llm_client.py#L173)

**Parameters:**

messages (`list[dict[str, Any]]`) : Conversation history as OpenAI-format message dicts.

tools (`list[dict[str, Any]]`) : MCP tool definitions.

- ****kwargs** : Override default parameters (temperature, max_tokens, etc.).

**Returns:**

An `LLMResponse` with the model's text and any tool calls.

Send messages with tool definitions, return a normalized response.

Messages use OpenAI-format dicts (`{"role": "...", "content": "..."}`).
Tools use MCP tool definitions; they are converted internally.

#### openenv.core.OpenAIClient[[openenv.core.OpenAIClient]]

```python
openenv.core.OpenAIClient(endpoint: str, port: int | None, model: str, api_key: str | None = None, system_prompt: str | None = None, temperature: float = 0.0, max_tokens: int = 256, use_max_completion_tokens: bool = False)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/llm_client.py#L205)

**Parameters:**

endpoint (`str`) : The base URL (e.g. "http://localhost"). May include a port and a path (e.g. "http://localhost:8000/v1"). The `/v1` API prefix is appended when the URL has no path; a URL with a path is used as-is.

port (`int` or `None`) : The port number, appended when `endpoint` does not name one; must match the URL's port when both are given.

model (`str`) : Model name to pass to the API.

api_key (`str`, *optional*) : API key. Defaults to "not-needed" for local endpoints.

system_prompt (`str`, *optional*) : System message prepended to every request.

temperature (`float`, *optional*, defaults to `0.0`) : Default sampling temperature.

max_tokens (`int`, *optional*, defaults to `256`) : Default max tokens in the response.

use_max_completion_tokens (`bool`, *optional*, defaults to `False`) : Use max_completion_tokens instead of max_tokens. Required for newer OpenAI models (gpt-5-mini, o1, o3). Not supported by self-hosted OpenAI-compatible endpoints.

Client for OpenAI-compatible APIs.

Works with: OpenAI, vLLM, TGI, Ollama, HuggingFace Inference API,
or any endpoint that speaks the OpenAI chat completions format.

#### complete[[openenv.core.OpenAIClient.complete]]

```python
complete(prompt: str, **kwargs)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/llm_client.py#L272)

**Parameters:**

prompt (`str`) : The user message.

- ****kwargs** : Overrides for temperature, max_tokens.

**Returns:**

The assistant's response text.

Send a chat completion request.

#### openenv.core.AnthropicClient[[openenv.core.AnthropicClient]]

```python
openenv.core.AnthropicClient(endpoint: str, port: int | None, model: str, api_key: str | None = None, system_prompt: str | None = None, temperature: float = 0.0, max_tokens: int = 256)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/llm_client.py#L321)

**Parameters:**

endpoint (`str`) : The base URL (e.g. `https://api.anthropic.com`). May include a port.

port (`int` or `None`) : The port number, appended when `endpoint` does not name one; must match the URL's port when both are given.

model (`str`) : Model name (e.g. "claude-sonnet-4-20250514").

api_key (`str`, *optional*) : Anthropic API key.

system_prompt (`str`, *optional*) : System message prepended to every request.

temperature (`float`, *optional*, defaults to `0.0`) : Default sampling temperature.

max_tokens (`int`, *optional*, defaults to `256`) : Default max tokens in the response.

Client for Anthropic's Messages API.

Requires the `anthropic` package (lazy-imported at construction time).

#### openenv.core.create_llm_client[[openenv.core.create_llm_client]]

```python
openenv.core.create_llm_client(provider: str, model: str, api_key: str, system_prompt: str | None = None, temperature: float = 0.0, max_tokens: int = 4096)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/llm_client.py#L439)

**Parameters:**

provider (`str`) : Provider name ("openai" or "anthropic").

model (`str`) : Model identifier.

api_key (`str`) : API key for the provider.

system_prompt (`str`, *optional*) : System message prepended to every request.

temperature (`float`, *optional*, defaults to `0.0`) : Sampling temperature.

max_tokens (`int`, *optional*, defaults to `4096`) : Maximum tokens in the response.

**Returns:**

A configured `LLMClient` instance.

Create an LLM client for a hosted provider.

### Shared dataclasses[[openenv.core.client_types.StepResult]]

#### openenv.core.client_types.StepResult[[openenv.core.client_types.StepResult]]

```python
openenv.core.client_types.StepResult(observation: ~ObsT, reward: typing.Optional[float] = None, done: bool = False, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/client_types.py#L11)

**Parameters:**

observation : The environment's observation after the action.

reward (`float`, *optional*) : Scalar reward for this step.

done (`bool`, *optional*, defaults to `False`) : Whether the episode is finished.

metadata (`dict`, *optional*) : Additional metadata returned alongside the observation.

Represents the result of one environment step.

## Auto classes[[openenv.AutoEnv]]

Load an environment client or its action class by name or Hub repo id, without importing the package. See [Auto-discovery](../guides/auto-discovery).

#### openenv.AutoEnv[[openenv.AutoEnv]]

```python
openenv.AutoEnv()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/auto/auto_env.py#L118)

AutoEnv automatically selects and instantiates the correct environment client
based on environment names or HuggingFace Hub repositories.

This class follows the HuggingFace AutoModel pattern, making it easy to work
with different environments without needing to import specific client classes.

The class provides factory methods that:
1. Check if name is a HuggingFace Hub URL/repo ID
2. If Hub: download and install the environment package
3. If local: look up the installed openenv-* package
4. Import and instantiate the client class

Example:
>>> # From installed package
>>> env = AutoEnv.from_env("coding-env")
>>>
>>> # From HuggingFace Hub
>>> env = AutoEnv.from_env("openenv/coding_env")
>>>
>>> # List available environments
>>> AutoEnv.list_environments()

Note:
AutoEnv is not meant to be instantiated directly. Use the class method
from_env() instead.

#### from_env[[openenv.AutoEnv.from_env]]

```python
from_env(name: str, base_url: Optional[str] = None, docker_image: Optional[str] = None, container_provider: Optional[ContainerProvider] = None, wait_timeout: float = 30.0, env_vars: Optional[Dict[str, str]] = None, trust_remote_code: bool = False, skip_install: bool = False, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/auto/auto_env.py#L496)

**Parameters:**

name : Environment name or HuggingFace Hub repo ID Examples: - `"coding"` / `"coding-env"` / `"coding_env"` - `"openenv/coding_env"` (Hub repo ID) - `"https://huggingface.co/openenv/coding_env"` (Hub URL)

base_url : Optional base URL for HTTP connection

docker_image : Optional Docker image name (overrides default)

container_provider : Optional container provider

wait_timeout : Timeout for container startup (seconds)

env_vars : Optional environment variables for the container

trust_remote_code : If True, skip user confirmation when installing from HuggingFace Hub. Can also be set via OPENENV_TRUST_REMOTE_CODE environment variable.

skip_install : If True, skip package installation and return a GenericEnvClient for remote environments. Useful when you only want to connect to a running server without installing any remote code. When True: - If base_url is provided: connects directly using GenericEnvClient - If HF Space is running: connects to Space using GenericEnvClient - If HF Space is not running: uses Docker from HF registry

- ****kwargs** : Additional arguments passed to the client class

**Returns:**

Instance of the environment client class

**Raises:** ``ValueError`` or ``ImportError``

- ``ValueError`` -- If environment not found or cannot be loaded
- ``ImportError`` -- If environment package is not installed

Create an environment client from a name or HuggingFace Hub repository.

This method automatically:
1. Checks if the name is a HuggingFace Hub URL/repo ID
2. If Hub: installs the environment package using git+ URL
3. If local: looks up the installed openenv-* package
4. Imports the client class and instantiates it

Examples:
>>> # From installed package
>>> env = AutoEnv.from_env("coding-env")
>>>
>>> # From HuggingFace Hub
>>> env = AutoEnv.from_env("openenv/coding_env")
>>>
>>> # With custom Docker image
>>> env = AutoEnv.from_env("coding", docker_image="my-coding-env:v2")
>>>
>>> # With environment variables
>>> env = AutoEnv.from_env(
...     "dipg",
...     env_vars={"DIPG_DATASET_PATH": "/data/dipg"}
... )
>>>
>>> # Skip package installation, use GenericEnvClient
>>> env = AutoEnv.from_env(
...     "user/my-env",
...     skip_install=True
... )

#### from_hub[[openenv.AutoEnv.from_hub]]

```python
from_hub(name: str, base_url: Optional[str] = None, docker_image: Optional[str] = None, container_provider: Optional['ContainerProvider'] = None, wait_timeout: float = 30.0, env_vars: Optional[Dict[str, str]] = None, trust_remote_code: bool = False, skip_install: bool = False, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/auto/auto_env.py#L766)

**Parameters:**

name : Environment name or HuggingFace Hub repo ID

base_url : Optional base URL for HTTP connection

docker_image : Optional Docker image name (overrides default)

container_provider : Optional container provider

wait_timeout : Timeout for container startup (seconds)

env_vars : Optional environment variables for the container

trust_remote_code : If True, skip user confirmation when installing from HuggingFace Hub

skip_install : If True, skip package installation and return a GenericEnvClient for remote environments

- ****kwargs** : Additional arguments passed to the client class

**Returns:**

Instance of the environment client class

Create an environment client from a name or HuggingFace Hub repository.

This is an alias for from_env() for backward compatibility.

Examples:
>>> env = AutoEnv.from_hub("coding-env")
>>> env = AutoEnv.from_hub("openenv/coding_env")

#### get_env_class[[openenv.AutoEnv.get_env_class]]

```python
get_env_class(name: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/auto/auto_env.py#L816)

**Parameters:**

name : Environment name

**Returns:**

The environment client class

**Raises:** ``ValueError``

- ``ValueError`` -- If environment not found

Get the environment client class without instantiating it.

Examples:
>>> CodingEnv = AutoEnv.get_env_class("coding")
>>> # Now you can instantiate it yourself
>>> env = CodingEnv(base_url="http://localhost:8000")

#### get_env_info[[openenv.AutoEnv.get_env_info]]

```python
get_env_info(name: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/auto/auto_env.py#L843)

**Parameters:**

name : Environment name

**Returns:**

Dictionary with environment metadata

**Raises:** ``ValueError``

- ``ValueError`` -- If environment not found

Get detailed information about an environment.

Examples:
>>> info = AutoEnv.get_env_info("coding")
>>> print(info['description'])
'Coding environment for OpenEnv'
>>> print(info['default_image'])
'coding-env:latest'

#### list_environments[[openenv.AutoEnv.list_environments]]

```python
list_environments()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/auto/auto_env.py#L884)

Print a formatted list of all available environments.

This discovers all installed openenv-* packages and displays
their metadata in a user-friendly format.

Examples:
>>> AutoEnv.list_environments()
Available OpenEnv Environments:
----------------------------------------------------------------------
echo           : Echo Environment (v0.1.0)
Package: openenv-echo-env
coding         : Coding Environment (v0.1.0)
Package: openenv-coding_env
----------------------------------------------------------------------
Total: 2 environments

#### openenv.AutoAction[[openenv.AutoAction]]

```python
openenv.AutoAction()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/auto/auto_action.py#L40)

AutoAction automatically retrieves the correct Action class based on
environment names or HuggingFace Hub repositories.

This class follows the HuggingFace AutoModel pattern, making it easy to
get the right Action class without needing to know which module to import.

The class provides factory methods that look up the Action class and
return the class (not an instance) for you to instantiate.

Example:
>>> # From installed package
>>> CodeAction = AutoAction.from_env("coding")
>>> action = CodeAction(code="print('test')")
>>>
>>> # From HuggingFace Hub
>>> CodeAction = AutoAction.from_env("openenv/coding_env")
>>> action = CodeAction(code="print('test')")
>>>
>>> # Use with AutoEnv for a complete workflow
>>> env = AutoEnv.from_env("coding-env")
>>> ActionClass = AutoAction.from_env("coding-env")
>>> action = ActionClass(code="print('Hello, AutoAction!')")
>>> result = env.step(action)

Note:
AutoAction is not meant to be instantiated directly. Use the class
method from_env() instead.

#### from_env[[openenv.AutoAction.from_env]]

```python
from_env(name: str, skip_install: bool = False)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/auto/auto_action.py#L78)

**Parameters:**

name : Environment name or HuggingFace Hub repo ID Examples: - `"coding"` / `"coding-env"` / `"coding_env"` - `"openenv/coding_env"` (Hub repo ID) - `"https://huggingface.co/openenv/coding_env"` (Hub URL)

skip_install : If True, skip package installation and return GenericAction class instead. Use this when working with GenericEnvClient to avoid installing remote packages.

**Returns:**

Action class (not an instance!). Returns GenericAction when
skip_install=True.

**Raises:** ``ValueError`` or ``ImportError``

- ``ValueError`` -- If environment not found (only when skip_install=False)
- ``ImportError`` -- If environment package is not installed (only when skip_install=False)

Get the Action class from environment name or HuggingFace Hub repository.

This method automatically:
1. Checks if the name is a HuggingFace Hub URL/repo ID
2. If Hub: downloads and installs the environment package
3. If local: looks up the installed openenv-* package
4. Imports and returns the Action class

Examples:
>>> # From installed package
>>> CodeAction = AutoAction.from_env("coding-env")
>>> action = CodeAction(code="print('Hello!')")
>>>
>>> # From HuggingFace Hub
>>> CodeAction = AutoAction.from_env("openenv/coding_env")
>>> action = CodeAction(code="print('Hello!')")
>>>
>>> # Skip installation, use GenericAction (for GenericEnvClient)
>>> ActionClass = AutoAction.from_env("user/repo", skip_install=True)
>>> action = ActionClass(code="print('Hello!')")  # Returns GenericAction
>>>
>>> # Different name formats
>>> EchoAction = AutoAction.from_env("echo")
>>> EchoAction = AutoAction.from_env("echo-env")
>>> EchoAction = AutoAction.from_env("echo_env")

#### from_hub[[openenv.AutoAction.from_hub]]

```python
from_hub(env_name: str, skip_install: bool = False)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/auto/auto_action.py#L182)

**Parameters:**

env_name : Environment name (e.g., "coding", "echo")

skip_install : If True, skip package installation and return GenericAction class instead.

**Returns:**

Action class (not an instance!)

Get the Action class from environment name.

This is an alias for from_env() for backward compatibility and clarity.

Examples:
>>> CodeAction = AutoAction.from_hub("coding")
>>> action = CodeAction(code="print('Hello!')")

#### get_action_info[[openenv.AutoAction.get_action_info]]

```python
get_action_info(name: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/auto/auto_action.py#L203)

**Parameters:**

name : Environment name

**Returns:**

Dictionary with action class metadata

**Raises:** ``ValueError``

- ``ValueError`` -- If environment not found

Get detailed information about an action class.

Examples:
>>> info = AutoAction.get_action_info("coding")
>>> print(info['action_class'])
'CodingAction'
>>> print(info['module'])
'coding_env.client'

#### list_actions[[openenv.AutoAction.list_actions]]

```python
list_actions()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/auto/auto_action.py#L239)

Print a formatted list of all available action classes.

This discovers all installed openenv-* packages and displays
their action class information in a user-friendly format.

Examples:
>>> AutoAction.list_actions()
Available Action Classes:
----------------------------------------------------------------------
echo           : EchoAction (from openenv-echo-env)
coding         : CodingAction (from openenv-coding_env)
----------------------------------------------------------------------
Total: 2 action classes

## MCP (Model Context Protocol)

### MCP environment[[openenv.core.MCPEnvironment]]

#### openenv.core.MCPEnvironment[[openenv.core.MCPEnvironment]]

```python
openenv.core.MCPEnvironment(mcp_server: Any, transform: typing.Optional[typing.Any] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_environment.py#L133)

**Parameters:**

mcp_server : A FastMCP server instance containing tool definitions. The server's tools will be validated against reserved names.

transform : Optional transform to apply to observations (inherited from Environment).

**Raises:** ``ValueError``

- ``ValueError`` -- If any tool in the MCP server uses a reserved name
  (reset, step, state, close).

Base class for environments that expose tools via MCP (Model Context Protocol).

MCPEnvironment bridges FastMCP servers with OpenEnv's Gym-style API, allowing
agents to discover and invoke MCP tools through the standard step() interface.

The class automatically handles:
- ListToolsAction: Returns available tools from the MCP server
- CallToolAction: Invokes a specific tool with arguments

All other actions are delegated to the abstract _step_impl() method,
which subclasses must implement.

Examples:

```python
from fastmcp import FastMCP

mcp = FastMCP("calculator")

@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

env = MyMCPEnvironment(mcp)
obs = env.step(ListToolsAction())
obs.tools[0].name  # 'add'
```

#### close[[openenv.core.MCPEnvironment.close]]

```python
close()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_environment.py#L643)

Clean up resources used by the environment.

This method cleans up the MCP client and any other resources.
Subclasses should call super().close() if they override this method.

#### execute_code[[openenv.core.MCPEnvironment.execute_code]]

```python
execute_code(code: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_environment.py#L297)

**Parameters:**

code : Python code to execute. Tools are available as functions in the execution namespace. Set a variable named 'result' to capture the return value.

**Returns:**

Observation with result in metadata["result"] or error in
metadata["error"].

Execute Python code with tools available as callables.

This enables the CodeAct pattern where agents write Python code
that calls tools directly as functions, avoiding JSON-RPC overhead.

#### get_callables[[openenv.core.MCPEnvironment.get_callables]]

```python
get_callables()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_environment.py#L266)

**Returns:**

Dictionary mapping tool names to callables.

Get callable functions for code mode.

Returns tool functions as direct Python callables, enabling code mode
where agents write Python code that calls tools directly (no JSON-RPC
overhead). Mode-specific tools are filtered by the current mode.

#### mcp_session[[openenv.core.MCPEnvironment.mcp_session]]

```python
mcp_session()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_environment.py#L218)

Context manager for MCP client sessions.

This wrapper serves two purposes:

1. **Null guard** — raises a clear error if `close()` has already
   been called (`mcp_client` is `None`).

2. **AsyncExitStack adapter** — FastMCP's `Client.__aenter__`
   creates a background `asyncio.Task` for session management.
   When entered directly via `AsyncExitStack` in the HTTP session
   path (`_create_session`), this task can be cancelled by ASGI
   harnesses (e.g. Starlette `TestClient`) between requests,
   corrupting session state.  Wrapping in an `asynccontextmanager`
   generator isolates the task lifecycle: the generator frame keeps
   `async with client:` suspended at `yield`, so cleanup only
   runs when the stack explicitly closes the generator — not when
   the event loop cancels orphaned tasks.

Delegates to FastMCP's `Client` context manager which is
reentrant: the first entry opens the transport and subsequent
(nested) entries simply increment an internal reference counter.
The transport is closed only when the outermost context exits.

No external lock is needed because `Client._connect` /
`Client._disconnect` already serialise connection state changes
through their own `anyio.Lock`.

#### set_mode[[openenv.core.MCPEnvironment.set_mode]]

```python
set_mode(mode: typing.Optional[str])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_environment.py#L202)

Set the active mode ('production' or 'simulation') for mode-aware tools.

#### step[[openenv.core.MCPEnvironment.step]]

```python
step(action: Action, timeout_s: typing.Optional[float] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_environment.py#L425)

**Parameters:**

action (`Action`) : The action to execute. `ListToolsAction` returns available MCP tools, `CallToolAction` invokes a specific MCP tool, and any other action is delegated to _step_impl().

timeout_s (`float`, *optional*) : Timeout in seconds for the action. Defaults to MCP_TOOL_CALL_TIMEOUT (30s) for MCP actions.

- ****kwargs** (`Any`) : Additional arguments passed to handlers.

**Returns:** `Observation`

`ListToolsObservation` for `ListToolsAction`,
`CallToolObservation` for `CallToolAction`, or a subclass-defined
Observation for other actions.

Execute an action in the environment.

This method routes MCP-specific actions (ListToolsAction, CallToolAction)
to the appropriate handlers, while delegating all other actions to
the subclass's _step_impl() method.

#### step_async[[openenv.core.MCPEnvironment.step_async]]

```python
step_async(action: Action, timeout_s: typing.Optional[float] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_environment.py#L596)

Async step that routes MCP actions without going through run_async_safely.

The WebSocket handler calls this directly on the outer event loop, where
the MCP session is already open, avoiding the thread/event-loop deadlock
that occurs when the sync step() path is used via run_in_executor.

#### tool[[openenv.core.MCPEnvironment.tool]]

```python
tool(mode: typing.Optional[str] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_environment.py#L350)

**Parameters:**

mode : Optional mode for the tool ("production" or "simulation"). If None, tool is available in all modes.

**Returns:**

A decorator function for registering tools.

**Raises:** ``ValueError``

- ``ValueError`` -- If mode is not None, "production", or "simulation".

Decorator for registering mode-aware tools.

### MCP types[[openenv.core.JsonRpcErrorCode]]

#### openenv.core.JsonRpcErrorCode[[openenv.core.JsonRpcErrorCode]]

```python
openenv.core.JsonRpcErrorCode(value, names = None, module = None, qualname = None, type = None, start = 1, boundary = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L29)

Standard JSON-RPC 2.0 error codes.

See: https://www.jsonrpc.org/specification#error_object

#### openenv.core.McpMethod[[openenv.core.McpMethod]]

```python
openenv.core.McpMethod(value, names = None, module = None, qualname = None, type = None, start = 1, boundary = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L47)

Supported MCP method names.

#### openenv.core.JsonRpcError[[openenv.core.JsonRpcError]]

```python
openenv.core.JsonRpcError(code: int, message: str, data: typing.Optional[typing.Any] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L54)

JSON-RPC 2.0 error object.

See: https://www.jsonrpc.org/specification#error_object

#### from_code[[openenv.core.JsonRpcError.from_code]]

```python
from_code(code: JsonRpcErrorCode, message: typing.Optional[str] = None, data: Any = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L69)

Create an error from a standard error code.

#### openenv.core.JsonRpcRequest[[openenv.core.JsonRpcRequest]]

```python
openenv.core.JsonRpcRequest(jsonrpc: typing.Literal['2.0'], method: str, params: typing.Dict[str, typing.Any] = <factory>, id: typing.Union[str, int, NoneType] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L89)

JSON-RPC 2.0 request object.

See: https://www.jsonrpc.org/specification#request_object

#### openenv.core.JsonRpcResponse[[openenv.core.JsonRpcResponse]]

```python
openenv.core.JsonRpcResponse(jsonrpc: typing.Literal['2.0'] = '2.0', result: typing.Optional[typing.Any] = None, error: typing.Optional[openenv.core.env_server.mcp_types.JsonRpcError] = None, id: typing.Union[str, int, NoneType] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L108)

JSON-RPC 2.0 response object.

Per JSON-RPC 2.0 spec, a response has either 'result' or 'error', not both.
This model excludes None values during serialization to comply with the spec.

See: https://www.jsonrpc.org/specification#response_object

#### error_response[[openenv.core.JsonRpcResponse.error_response]]

```python
error_response(code: JsonRpcErrorCode, message: typing.Optional[str] = None, data: Any = None, request_id: typing.Union[str, int, NoneType] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L159)

Create an error response from a standard error code.

#### model_dump[[openenv.core.JsonRpcResponse.model_dump]]

```python
model_dump(**kwargs)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L131)

Serialize to dict, excluding result or error when None (JSON-RPC compliance).

#### model_dump_json[[openenv.core.JsonRpcResponse.model_dump_json]]

```python
model_dump_json(**kwargs)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L146)

Serialize to JSON string, excluding result or error when None (JSON-RPC compliance).

#### success[[openenv.core.JsonRpcResponse.success]]

```python
success(result: Any, request_id: typing.Union[str, int, NoneType] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L152)

Create a success response.

#### openenv.core.Tool[[openenv.core.Tool]]

```python
openenv.core.Tool(name: str, description: str, input_schema: typing.Dict[str, typing.Any])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L179)

Strongly typed MCP tool specification.

Follows the MCP ToolSpec format for tool discovery.
See: https://modelcontextprotocol.io/specification/2025-06-18/server/tools

#### openenv.core.ToolErrorType[[openenv.core.ToolErrorType]]

```python
openenv.core.ToolErrorType(value, names = None, module = None, qualname = None, type = None, start = 1, boundary = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L198)

Types of errors that can occur during tool execution.

#### openenv.core.ToolError[[openenv.core.ToolError]]

```python
openenv.core.ToolError(error_type: ToolErrorType, message: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L208)

Structured error for tool execution failures.

This is used for transport/framework errors, NOT for errors returned
by the tool itself (those go in the result field).

#### openenv.core.ListToolsAction[[openenv.core.ListToolsAction]]

```python
openenv.core.ListToolsAction(metadata: typing.Dict[str, typing.Any] = <factory>, type: typing.Literal['list_tools'] = 'list_tools')
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L225)

Request list of available tools from the environment.

This action triggers MCP's tools/list operation and returns
all available tools with their schemas. Does NOT require reset()
to be called first.

#### openenv.core.CallToolAction[[openenv.core.CallToolAction]]

```python
openenv.core.CallToolAction(metadata: typing.Dict[str, typing.Any] = <factory>, type: typing.Literal['call_tool'] = 'call_tool', tool_name: str, arguments: typing.Dict[str, typing.Any] = <factory>)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L239)

Call a specific tool via MCP.

This action triggers MCP's tools/call operation with the
specified tool name and arguments.

#### openenv.core.ListToolsObservation[[openenv.core.ListToolsObservation]]

```python
openenv.core.ListToolsObservation(done: bool = False, reward: bool | int | float | None = None, metadata: typing.Dict[str, typing.Any] = <factory>, tools: typing.List[openenv.core.env_server.mcp_types.Tool])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L259)

Response containing available tools.

Returned when processing a ListToolsAction.

#### openenv.core.CallToolObservation[[openenv.core.CallToolObservation]]

```python
openenv.core.CallToolObservation(done: bool = False, reward: bool | int | float | None = None, metadata: typing.Dict[str, typing.Any] = <factory>, tool_name: str, result: Any = None, error: typing.Optional[openenv.core.env_server.mcp_types.ToolError] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L269)

Response from tool execution.

Contains the tool's result or an error if the call failed.
Tool-specific errors (from the tool itself) are included in the result.
Transport/framework errors use the error field.

#### openenv.core.WSMCPMessage[[openenv.core.WSMCPMessage]]

```python
openenv.core.WSMCPMessage(type: typing.Literal['mcp'] = 'mcp', data: typing.Dict[str, typing.Any])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L290)

WebSocket message for MCP JSON-RPC requests.

Allows direct MCP access via WebSocket for production inference,
bypassing the step() API.

#### openenv.core.WSMCPResponse[[openenv.core.WSMCPResponse]]

```python
openenv.core.WSMCPResponse(type: str = 'mcp', data: typing.Dict[str, typing.Any])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py#L302)

WebSocket response for MCP JSON-RPC.

Contains the JSON-RPC response from the MCP server.

### MCP client[[openenv.core.MCPClientBase]]

#### openenv.core.MCPClientBase[[openenv.core.MCPClientBase]]

```python
openenv.core.MCPClientBase(base_url: str, connect_timeout_s: float = 10.0, message_timeout_s: float = 60.0, websocket_ping_interval_s: typing.Optional[float] = 20.0, websocket_ping_timeout_s: typing.Optional[float] = 20.0, provider: typing.Optional[typing.Any] = None, mode: typing.Optional[str] = None, max_message_size_mb: float = 100.0, headers: typing.Optional[typing.Dict[str, str]] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/mcp_client.py#L95)

**Parameters:**

_tools_cache : Cached list of tools (populated on first `list_tools()` call)

Base class for MCP clients with tool discovery.

This class provides the common `list_tools()` method for discovering
available tools from an MCP-enabled environment. Subclasses implement
specific interaction patterns (tool-calling or CodeAct).

#### list_tools[[openenv.core.MCPClientBase.list_tools]]

```python
list_tools(use_cache: bool = True)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/mcp_client.py#L282)

**Parameters:**

use_cache (`bool`, *optional*, defaults to `True`) : If `True`, return cached tools if available. Set to `False` to force a fresh request.

**Returns:**

List of `Tool` objects with name, description, and input_schema.

Discover available tools from the environment.

Examples:

```python
tools = await env.list_tools()
for tool in tools:
    print(f"{tool.name}: {tool.description}")
```

#### openenv.core.MCPToolClient[[openenv.core.MCPToolClient]]

```python
openenv.core.MCPToolClient(base_url: str, connect_timeout_s: float = 10.0, message_timeout_s: float = 60.0, websocket_ping_interval_s: typing.Optional[float] = 20.0, websocket_ping_timeout_s: typing.Optional[float] = 20.0, provider: typing.Optional[typing.Any] = None, mode: typing.Optional[str] = None, max_message_size_mb: float = 100.0, headers: typing.Optional[typing.Dict[str, str]] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/mcp_client.py#L449)

Async client for tool-calling style MCP interactions.

Each step invokes a single tool. Use this for traditional function-calling
agent patterns where the agent decides which tool to call next.

This client provides convenience methods for tool discovery and invocation:
- `list_tools()`: Get all available tools with their schemas
- `call_tool(name, **kwargs)`: Invoke a tool by name with arguments

Examples:

```python
async with MCPToolClient(base_url="http://localhost:8000") as env:
    # Reset the environment
    await env.reset()

    # Discover available tools
    tools = await env.list_tools()
    print([t.name for t in tools])  # ['echo_message', 'echo_with_length']

    # Call a tool directly
    result = await env.call_tool("echo_message", message="Hello!")
    print(result)  # "Hello!"

    # Or use the full action interface
    from openenv.core.env_server.mcp_types import CallToolAction
    step_result = await env.step(CallToolAction(
        tool_name="echo_with_length",
        arguments={"message": "Test"}
    ))
    print(step_result.observation.result)
```

Sync wrapper:

```python
env = MCPToolClient(base_url="http://localhost:8000").sync()
with env:
    tools = env.list_tools()
    result = env.call_tool("echo_message", message="Hello!")
```

#### call_tool[[openenv.core.MCPToolClient.call_tool]]

```python
call_tool(name: str, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/mcp_client.py#L494)

**Parameters:**

name (`str`) : Name of the tool to invoke (must match a tool from `list_tools()`).

- ****kwargs** : Arguments to pass to the tool. Must match the tool's input_schema.

**Returns:**

The tool's result. The type depends on the tool being called.

**Raises:** ``RuntimeError``

- ``RuntimeError`` -- If the server returns an error response.

Call a tool by name.

This is a convenience method that creates a CallToolAction, executes it,
and returns the result directly. For more control, use `step()` with
a CallToolAction directly.

Examples:

```python
result = await env.call_tool("add", a=5, b=3)
print(result)  # 8

result = await env.call_tool("greet", name="Claude")
print(result)  # "Hello, Claude!"
```

#### get_tool[[openenv.core.MCPToolClient.get_tool]]

```python
get_tool(name: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/mcp_client.py#L570)

**Parameters:**

name (`str`) : Name of the tool to find.

**Returns:**

The `Tool` object if found, `None` otherwise.

Get a specific tool by name.

Examples:

```python
tool = await env.get_tool("echo_message")
if tool:
    print(tool.description)
    print(tool.input_schema)
```

#### has_tool[[openenv.core.MCPToolClient.has_tool]]

```python
has_tool(name: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/mcp_client.py#L596)

**Parameters:**

name (`str`) : Name of the tool to check.

**Returns:**

`True` if the tool exists, `False` otherwise.

Check if a tool exists.

## Harnesses

All classes below are importable from `openenv.core.harness`. See [Harnesses in OpenEnv](../tutorials/harnesses) for when to use each path.

### Agent inside the environment (RFC 005)[[openenv.core.harness.HarnessEnvironment]]

#### openenv.core.harness.HarnessEnvironment[[openenv.core.harness.HarnessEnvironment]]

```python
openenv.core.harness.HarnessEnvironment(adapter: AgenticHarnessAdapter, mcp: Optional[Any] = None, rubric: Optional[Any] = None, transform: Optional[Any] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/environment.py#L46)

**Parameters:**

adapter ([AgenticHarnessAdapter](https://openenv.github.io/OpenEnv/main/reference/core#openenv.core.harness.AgenticHarnessAdapter)) : Adapter owning the harness process lifecycle.

mcp (`FastMCP`, *optional*) : Environment-specific MCP tools to inject into the harness. When `None`, an empty internal server is used and nothing is injected.

rubric ([Rubric](https://openenv.github.io/OpenEnv/main/reference/core#openenv.core.rubrics.Rubric), *optional*) : Reward rubric applied to each turn's observation.

transform (`Transform`, *optional*) : Optional observation transform (inherited from `Environment`).

Environment that wraps an external turn-based agentic harness.

In simulation mode, `reset()` starts a fresh harness process and
conversation, and each `step()` with a [HarnessAction](https://openenv.github.io/OpenEnv/main/reference/core#openenv.core.harness.HarnessAction)
is one conversational turn: the harness runs its internal ReAct loop and
the response is returned as an observation. The harness keeps conversation
context across turns; the training loop controls episode boundaries.
MCP actions (`ListToolsAction`, `CallToolAction`) keep their standard
routing so orchestrators can inspect and invoke domain tools directly.

Environment MCP tools are conflict-resolved against the adapter's built-in
tool names and injected into the harness before it starts. Rubrics run
after each turn completes, outside the harness's control loop.

Factories used with `HTTPEnvServer` must construct a fresh adapter per
environment instance — adapters own a single harness process and must not
be shared. Constructing the environment starts nothing; the harness
process and any tool bridge start on `reset()`.

#### close[[openenv.core.harness.HarnessEnvironment.close]]

```python
close()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/environment.py#L414)

Stop the harness process and release environment resources.

#### reset[[openenv.core.harness.HarnessEnvironment.reset]]

```python
reset(seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/environment.py#L198)

Sync facade over `reset_async` for non-async callers.

#### reset_async[[openenv.core.harness.HarnessEnvironment.reset_async]]

```python
reset_async(seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/environment.py#L113)

**Parameters:**

seed (`int`, *optional*) : Unused; accepted for interface compatibility.

episode_id (`str`, *optional*) : Episode identifier. A UUID is generated when `None`.

**Returns:**

`Observation` with `done=False` and metadata listing the names of
the injected tools.

Start a fresh harness process and conversation.

Stops any running harness, injects the environment's conflict-resolved
MCP tools, and starts the harness in its configured working directory.

#### step_async[[openenv.core.harness.HarnessEnvironment.step_async]]

```python
step_async(action: Action, timeout_s: Optional[float] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/environment.py#L209)

**Parameters:**

action (`Action`) : A [HarnessAction](https://openenv.github.io/OpenEnv/main/reference/core#openenv.core.harness.HarnessAction) runs one conversational turn; `ListToolsAction`/`CallToolAction` keep their standard MCP routing.

timeout_s (`float`, *optional*) : Wall-clock budget for this turn. Defaults to the adapter config's `session_timeout_s`.

**Returns:**

`Observation` with the harness response and turn events in
metadata.

Execute one action: a conversational turn or an MCP action.

#### openenv.core.harness.HarnessAction[[openenv.core.harness.HarnessAction]]

```python
openenv.core.harness.HarnessAction(metadata: typing.Dict[str, typing.Any] = <factory>, message: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/environment.py#L34)

**Parameters:**

message (`str`) : The user message for this turn.

Action carrying one user message for a conversational harness turn.

#### openenv.core.harness.AgenticHarnessAdapter[[openenv.core.harness.AgenticHarnessAdapter]]

```python
openenv.core.harness.AgenticHarnessAdapter(config: HarnessConfig)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/adapter.py#L31)

**Parameters:**

BUILTIN_TOOL_NAMES (`frozenset[str]`) : Names of the harness's built-in tools, used to detect conflicts with injected environment tools. Concrete adapters override this.

Abstract adapter for a turn-based external agentic harness.

Subclass this to integrate a harness such as OpenClaw or Claude Code. The
adapter owns the harness process lifecycle and communication; the harness
itself is a long-lived process that maintains conversation context across
turns. Each `send_message()` call is one conversational turn: the harness
runs its internal ReAct loop and returns when it has a response.

Distinct from the trainer-side rollout
`HarnessAdapter`, which drives an entire
episode in a single call.

#### inject_tools[[openenv.core.harness.AgenticHarnessAdapter.inject_tools]]

```python
inject_tools(tools: list[Tool], bridge_url: Optional[str] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/adapter.py#L81)

**Parameters:**

tools (`list[Tool]`) : Conflict-resolved environment tool definitions to inject.

bridge_url (`str`, *optional*) : URL of the MCP bridge serving these tools, when the environment exposes one.

Inject environment MCP tool definitions into the harness configuration.

Called before `start()` so the harness discovers the tools at startup.
The mechanism is harness-specific (config file, CLI flags, environment
variables).

#### is_alive[[openenv.core.harness.AgenticHarnessAdapter.is_alive]]

```python
is_alive()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/adapter.py#L120)

Check whether the harness process is still running.

#### send_message[[openenv.core.harness.AgenticHarnessAdapter.send_message]]

```python
send_message(message: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/adapter.py#L125)

**Parameters:**

message (`str`) : The user message for this conversational turn.

**Returns:**

`HarnessResponse` with the text
response, all turn events, and the harness's done signal.

**Raises:** `HarnessError`

- `HarnessError` -- 
  If the event stream ends without a `TURN_COMPLETE` event.

Send a message and collect the complete response for one turn.

Drains `send_message_streaming()` and assembles a
`HarnessResponse` from the terminal
`TURN_COMPLETE` event.

#### send_message_streaming[[openenv.core.harness.AgenticHarnessAdapter.send_message_streaming]]

```python
send_message_streaming(message: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/adapter.py#L101)

**Parameters:**

message (`str`) : The user message for this conversational turn.

**Yields:**

`HarnessEvent` instances.

Send a message and stream intermediate events for one turn.

Yields events as the harness processes the turn (tool calls, LLM
chunks, text output). The final event must be a
`HarnessEvent` of type `TURN_COMPLETE`
whose data carries the turn's `response` and optionally `done`.

#### start[[openenv.core.harness.AgenticHarnessAdapter.start]]

```python
start(working_directory: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/adapter.py#L56)

**Parameters:**

working_directory (`str`) : Path where the harness should operate.

**Raises:** `HarnessStartupError`

- `HarnessStartupError` -- 
  If the process fails to start or become ready in time.

Start the harness process.

#### stop[[openenv.core.harness.AgenticHarnessAdapter.stop]]

```python
stop()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/adapter.py#L71)

Stop the harness process and clean up resources.

Implementations must be idempotent: calling `stop()` on an already
stopped (or never started) adapter must succeed silently.

### Rollout sessions[[openenv.core.harness.ResourceSessionFactory]]

#### openenv.core.harness.ResourceSessionFactory[[openenv.core.harness.ResourceSessionFactory]]

```python
openenv.core.harness.ResourceSessionFactory()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/rollout.py#L258)

Factory for producing isolated per-rollout sessions.

Generic over the concrete session type it creates, so a subclass such as
`OpenCodeSessionFactory(ResourceSessionFactory[OpenCodeSession])` types
`create` as returning `OpenCodeSession`. Subclassing without a type
argument stays valid and behaves as before.

#### create[[openenv.core.harness.ResourceSessionFactory.create]]

```python
create(task: Any, seed: int | None = None, episode_id: str | None = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/rollout.py#L267)

Create one isolated resource session for a rollout.

### Training captures[[openenv.core.harness.TrainingTrace]]

#### openenv.core.harness.TrainingTrace[[openenv.core.harness.TrainingTrace]]

```python
openenv.core.harness.TrainingTrace(schema_version: typing.Literal[1] = 1, turns: list[openenv.core.harness.training.TrainingTurn])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/training.py#L39)

Selected agent calls, each with an authoritative mask and unique identity.

Zero-masked turns remain available for context and usage accounting. Task rewards
come from the session's `verify()` method, independently of these masks.

#### from_entries[[openenv.core.harness.TrainingTrace.from_entries]]

```python
from_entries(entries: list[dict[str, Any]])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/training.py#L58)

Adapt producer-selected entries without inferring tokens or masks.

#### to_trace_entries[[openenv.core.harness.TrainingTrace.to_trace_entries]]

```python
to_trace_entries()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/training.py#L77)

Return diagnostic records for reward functions and trace viewers.

#### openenv.core.harness.TrainingTurn[[openenv.core.harness.TrainingTurn]]

```python
openenv.core.harness.TrainingTurn(node_id: Annotated, prompt_token_ids: list[int], completion_token_ids: list[int], per_token_logps: list[float], loss_mask: list[int], request: dict[str, typing.Any] = <factory>, response: dict[str, typing.Any] = <factory>, metadata: dict[str, typing.Any] = <factory>)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/harness/training.py#L12)

One engine call. The required loss mask covers prompt plus completion.

## Evaluation[[openenv.core.evals.InspectAIHarness]]

See [Evaluating with Environments](../tutorials/evaluation-inspect) for a worked example.

#### openenv.core.evals.InspectAIHarness[[openenv.core.evals.InspectAIHarness]]

```python
openenv.core.evals.InspectAIHarness(log_dir: Optional[str] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/evals/inspect_harness.py#L27)

**Parameters:**

log_dir : Directory for evaluation log output. Defaults to None (Inspect AI writes logs to its default location).

Evaluation harness wrapping Inspect AI's `eval()` function.

All `inspect_ai` imports are deferred to `run` so this class is
importable without inspect-ai installed.  An `ImportError` with a clear
message is raised at call time if the dependency is missing.

`eval_parameters` keys accepted by `run`:

+--------------------------+----------+-----------------+-----------------------------------+
| Key                      | Type     | Default         | Purpose                           |
+==========================+==========+=================+===================================+
| `model`                | str      | *required*      | Model string, e.g. "openai/gpt-4o"|
| `model_base_url`       | str|None | None            | Override provider base URL        |
| `task`                 | str|None | `dataset` arg | Task file path or task string     |
| `task_args`            | dict     | `&amp;lcub;}`          | Arguments to pass to the task     |
| `max_samples`          | int|None | None            | Limit samples per task            |
| `max_connections`      | int|None | None            | Max concurrent model connections  |
| `temperature`          | float|None| None           | Model generation temperature      |
| `max_tokens`           | int|None | None            | Max generation tokens             |
| `epochs`               | int|None | None            | Number of evaluation epochs       |
| `solver`               | list|None| None            | Solver pipeline override          |
| `scorer`               | list|None| None            | Scorer override                   |
| `model_args`           | dict     | `&amp;lcub;}`          | Provider-specific model kwargs    |
+--------------------------+----------+-----------------+-----------------------------------+

#### run[[openenv.core.evals.InspectAIHarness.run]]

```python
run(harness_version: str, library_versions: Dict[str, str], dataset: str, eval_parameters: Dict[str, Any])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/evals/inspect_harness.py#L65)

**Parameters:**

harness_version : Version of inspect-ai being used.

library_versions : Versions of supporting libraries.

dataset : Default task string (used when `task` is not specified in *eval_parameters*).

eval_parameters : See class docstring for accepted keys.

**Returns:**

Dictionary mapping metric names to scores.

**Raises:** `ImportError` or `ValueError` or `RuntimeError`

- `ImportError` -- If `inspect-ai` is not installed.
- `ValueError` -- If `model` is missing from *eval_parameters*.
- `RuntimeError` -- If the evaluation fails (log status is not "success").

Run an Inspect AI evaluation.

## Rubrics[[openenv.core.rubrics.Rubric]]

#### openenv.core.rubrics.Rubric[[openenv.core.rubrics.Rubric]]

```python
openenv.core.rubrics.Rubric()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/base.py#L17)

Abstract base class for reward computation.

A Rubric computes a reward signal from an action and observation.
Subclasses implement forward() to define the reward logic.

Examples:

```python
class MyRubric(Rubric):
    def forward(self, action, observation) -> float:
        return 1.0 if action.valid else 0.0

rubric = MyRubric()
reward = rubric(action, observation)
```

Child rubrics are auto-registered when assigned as attributes,
enabling hierarchical composition and introspection.

#### children[[openenv.core.rubrics.Rubric.children]]

```python
children()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/base.py#L197)

Iterate over immediate child rubrics.

#### forward[[openenv.core.rubrics.Rubric.forward]]

```python
forward(action: Any, observation: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/base.py#L162)

**Parameters:**

action : The action taken by the agent.

observation : The resulting observation.

**Returns:** `float`

Reward value (typically 0.0 to 1.0).

Compute the reward. Implement this in subclasses.

#### get_rubric[[openenv.core.rubrics.Rubric.get_rubric]]

```python
get_rubric(path: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/base.py#L218)

**Parameters:**

path (`str`) : Dot-separated path (e.g., "code.syntax").

**Returns:** `Rubric`

The rubric at the specified path.

**Raises:** ``KeyError``

- ``KeyError`` -- If the path does not exist.

Access a nested rubric by dot-separated path.

#### load_state_dict[[openenv.core.rubrics.Rubric.load_state_dict]]

```python
load_state_dict(state: typing.Dict[str, typing.Any])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/base.py#L261)

Load rubric configuration from checkpoint.

#### named_children[[openenv.core.rubrics.Rubric.named_children]]

```python
named_children()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/base.py#L201)

Iterate over immediate child rubrics with names.

#### named_rubrics[[openenv.core.rubrics.Rubric.named_rubrics]]

```python
named_rubrics(prefix: str = '')
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/base.py#L211)

Iterate over all descendant rubrics with dot-separated names.

#### register_forward_hook[[openenv.core.rubrics.Rubric.register_forward_hook]]

```python
register_forward_hook(hook: typing.Callable[[ForwardRef('Rubric'), typing.Any, typing.Any, float], NoneType])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/base.py#L175)

**Parameters:**

hook (`Callable`) : Callable with signature (rubric, action, observation, result).

Register a hook called after forward().

#### register_forward_pre_hook[[openenv.core.rubrics.Rubric.register_forward_pre_hook]]

```python
register_forward_pre_hook(hook: typing.Callable[[ForwardRef('Rubric'), typing.Any, typing.Any], NoneType])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/base.py#L186)

**Parameters:**

hook (`Callable`) : Callable with signature (rubric, action, observation).

Register a hook called before forward().

#### reset[[openenv.core.rubrics.Rubric.reset]]

```python
reset()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/base.py#L239)

Reset internal state, including every child rubric.

Override to clear your own state and call `super().reset()` so child
rubrics are reset too.

#### rubrics[[openenv.core.rubrics.Rubric.rubrics]]

```python
rubrics()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/base.py#L205)

Iterate over all descendant rubrics (depth-first).

#### state_dict[[openenv.core.rubrics.Rubric.state_dict]]

```python
state_dict()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/base.py#L257)

Serialize rubric configuration for checkpointing.

#### validation_config[[openenv.core.rubrics.Rubric.validation_config]]

```python
validation_config()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/base.py#L248)

Return explicitly public configuration for authorized validation.

Override to expose JSON configuration without credentials or runtime state.
`None` means configuration introspection is unavailable. Stock aggregation
containers expose their weights/threshold directly through the server.

#### openenv.core.rubrics.Sequential[[openenv.core.rubrics.Sequential]]

```python
openenv.core.rubrics.Sequential(*rubrics: Rubric)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/containers.py#L27)

Run rubrics in order, fail-fast on zero.

Runs child rubrics in order. If any returns 0, stops immediately
and returns 0. This implements hierarchical gating patterns where
syntax checks run before execution checks.

Examples:

```python
rubric = Sequential(
    Gate(Compiles()),
    Gate(PassesTests(), threshold=0.5),
    WeightedSum([PassesTests(), StyleRubric()], weights=[0.7, 0.3])
)
```

#### forward[[openenv.core.rubrics.Sequential.forward]]

```python
forward(action: Any, observation: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/containers.py#L57)

Run rubrics in order, return 0 if any returns 0. Sync version.

#### openenv.core.rubrics.Gate[[openenv.core.rubrics.Gate]]

```python
openenv.core.rubrics.Gate(rubric: Rubric, threshold: float = 1.0)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/containers.py#L185)

Threshold wrapper - returns 0 if child score is below threshold.

Useful for hard constraints like "must pass 50% of tests".

Examples:

```python
rubric = Gate(PassesTests(), threshold=0.5)
# Returns PassesTests() score if >= 0.5, else 0.0
```

#### forward[[openenv.core.rubrics.Gate.forward]]

```python
forward(action: Any, observation: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/containers.py#L211)

Return child score if >= threshold, else 0. Sync version.

#### openenv.core.rubrics.WeightedSum[[openenv.core.rubrics.WeightedSum]]

```python
openenv.core.rubrics.WeightedSum(rubrics: typing.List[openenv.core.rubrics.base.Rubric], weights: typing.List[float])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/containers.py#L248)

Weighted combination of child rubrics.

Standard aggregation pattern for multi-criteria evaluation.

Examples:

```python
rubric = WeightedSum(
    [PassesTests(), StyleRubric()],
    weights=[0.7, 0.3]
)
```

#### forward[[openenv.core.rubrics.WeightedSum.forward]]

```python
forward(action: Any, observation: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/containers.py#L289)

Return weighted sum of child scores. Sync version.

#### openenv.core.rubrics.RubricList[[openenv.core.rubrics.RubricList]]

```python
openenv.core.rubrics.RubricList(rubrics: typing.List[openenv.core.rubrics.base.Rubric] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/containers.py#L359)

Container for dynamic lists of rubrics.

Analogous to nn.ModuleList. Does not define aggregation - use within
a parent rubric that implements custom logic.

Examples:

```python
class MultiGameRubric(Rubric):
    def __init__(self, games: List[str]):
        super().__init__()
        self.games = RubricList([GameRubric(g) for g in games])

    def forward(self, action, obs) -> float:
        return self.games[obs.game_index](action, obs)
```

#### append[[openenv.core.rubrics.RubricList.append]]

```python
append(rubric: Rubric)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/containers.py#L398)

Add a rubric to the list.

#### extend[[openenv.core.rubrics.RubricList.extend]]

```python
extend(rubrics: typing.List[openenv.core.rubrics.base.Rubric])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/containers.py#L404)

Add multiple rubrics to the list.

#### forward[[openenv.core.rubrics.RubricList.forward]]

```python
forward(action: Any, observation: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/containers.py#L391)

RubricList does not define aggregation - override in parent.

#### openenv.core.rubrics.RubricDict[[openenv.core.rubrics.RubricDict]]

```python
openenv.core.rubrics.RubricDict(rubrics: typing.Dict[str, openenv.core.rubrics.base.Rubric] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/containers.py#L419)

Container for named rubrics with keyed access.

Analogous to nn.ModuleDict. Enables keyed access for multi-task
environments where different tasks require different rubrics.

Examples:

```python
class AtariRubric(Rubric):
    def __init__(self):
        super().__init__()
        self.games = RubricDict({
            "pong": PongRubric(),
            "breakout": BreakoutRubric(),
            "space_invaders": SpaceInvadersRubric(),
        })

    def forward(self, action, obs) -> float:
        return self.games[obs.game_id](action, obs)

# Access: env.rubric.games["pong"]
```

#### forward[[openenv.core.rubrics.RubricDict.forward]]

```python
forward(action: Any, observation: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/containers.py#L457)

RubricDict does not define aggregation - override in parent.

#### items[[openenv.core.rubrics.RubricDict.items]]

```python
items()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/containers.py#L491)

Iterate over (key, rubric) pairs.

#### keys[[openenv.core.rubrics.RubricDict.keys]]

```python
keys()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/containers.py#L483)

Iterate over keys.

#### update[[openenv.core.rubrics.RubricDict.update]]

```python
update(rubrics: typing.Union[typing.Dict[str, openenv.core.rubrics.base.Rubric], typing.Mapping[str, openenv.core.rubrics.base.Rubric]])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/containers.py#L495)

Update with rubrics from a dictionary.

#### values[[openenv.core.rubrics.RubricDict.values]]

```python
values()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/containers.py#L487)

Iterate over rubrics.

#### openenv.core.rubrics.TrajectoryRubric[[openenv.core.rubrics.TrajectoryRubric]]

```python
openenv.core.rubrics.TrajectoryRubric(intermediate_reward: float = 0.0)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/trajectory.py#L22)

Abstract base for rubrics that score based on full trajectories.

Subclasses implement:
- score_trajectory(): Compute final score from trajectory
- compute_step_rewards(): Define credit assignment strategy

The __call__ method accumulates steps and returns rewards according
to the subclass's implementation.

IMPORTANT: Trajectories are stored in CPU memory to avoid GPU pressure.
Environments with GPU tensors in observations must move them to CPU
before returning from step().

Known limitation: Very long episodes (thousands of steps) may consume
significant CPU memory. For such cases, consider streaming rubrics.

Examples:

```python
class WinLossRubric(TrajectoryRubric):
    def score_trajectory(self, trajectory):
        _, final_obs = trajectory[-1]
        return 1.0 if final_obs.metadata.get('won') else 0.0

    def compute_step_rewards(self):
        # Equal credit to all steps
        score = self.score_trajectory(self._trajectory)
        return [score] * len(self._trajectory)

rubric = WinLossRubric()
for action, obs in episode:
    reward = rubric(action, obs)  # 0.0 until done
step_rewards = rubric.compute_step_rewards()  # Credit assignment
```

#### compute_step_rewards[[openenv.core.rubrics.TrajectoryRubric.compute_step_rewards]]

```python
compute_step_rewards()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/trajectory.py#L107)

**Returns:** `list[float]`

Rewards, one per step. Length matches len(trajectory).

Compute per-step rewards from the accumulated trajectory.

Define your credit assignment strategy here (e.g., discounting,
assigning all credit to specific steps, etc.).

#### forward[[openenv.core.rubrics.TrajectoryRubric.forward]]

```python
forward(action: Any, observation: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/trajectory.py#L73)

**Parameters:**

action : The action taken.

observation : The resulting observation. Must have a 'done' attribute.

**Returns:** `float`

intermediate_reward if not done, else score_trajectory() result.

Accumulate step and return reward.

Returns intermediate_reward until done, then computes trajectory score.

#### load_state_dict[[openenv.core.rubrics.TrajectoryRubric.load_state_dict]]

```python
load_state_dict(state: typing.Dict[str, typing.Any])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/trajectory.py#L133)

Load configuration from checkpoint.

#### reset[[openenv.core.rubrics.TrajectoryRubric.reset]]

```python
reset()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/trajectory.py#L119)

Clear accumulated trajectory. Call on env.reset().

#### score_trajectory[[openenv.core.rubrics.TrajectoryRubric.score_trajectory]]

```python
score_trajectory(trajectory: typing.List[typing.Tuple[typing.Any, typing.Any]])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/trajectory.py#L92)

**Parameters:**

trajectory (`list`) : List of (action, observation) tuples.

**Returns:** `float`

Final trajectory score (typically 0.0 to 1.0).

Score the complete trajectory. Return 0.0-1.0.

Called when observation.done=True.

#### state_dict[[openenv.core.rubrics.TrajectoryRubric.state_dict]]

```python
state_dict()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/trajectory.py#L129)

Serialize configuration (not trajectory data).

#### openenv.core.rubrics.ExponentialDiscountingTrajectoryRubric[[openenv.core.rubrics.ExponentialDiscountingTrajectoryRubric]]

```python
openenv.core.rubrics.ExponentialDiscountingTrajectoryRubric(gamma: float = 0.99, intermediate_reward: float = 0.0)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/trajectory.py#L139)

TrajectoryRubric with exponential discounting for credit assignment.

Per-step reward:

```text
r_t = gamma^(T-1-t) * R_final
```

With gamma=0.99, later steps get higher reward (they're "closer" to the outcome).
With gamma=1.0, all steps get equal reward.
With gamma=0.0, only the final step gets reward.

This is the standard temporal discounting used in reinforcement learning,
applied retroactively once the episode outcome is known.

Examples:

```python
class ChessRubric(ExponentialDiscountingTrajectoryRubric):
    def score_trajectory(self, trajectory):
        _, final_obs = trajectory[-1]
        outcome = final_obs.metadata.get('winner')
        if outcome == 'agent': return 1.0
        elif outcome == 'opponent': return 0.0
        else: return 0.5  # Draw

rubric = ChessRubric(gamma=0.99)
reward = rubric(action, obs)  # 0.0 until done, then final score
step_rewards = rubric.compute_step_rewards()  # Discounted per-step rewards
```

#### compute_step_rewards[[openenv.core.rubrics.ExponentialDiscountingTrajectoryRubric.compute_step_rewards]]

```python
compute_step_rewards()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/trajectory.py#L188)

**Returns:** `list[float]`

Discounted rewards where `step_rewards[t] = gamma^(T-1-t) * R_final`,
T is the trajectory length and R_final is score_trajectory().

Apply exponential discounting from final reward.

#### state_dict[[openenv.core.rubrics.ExponentialDiscountingTrajectoryRubric.state_dict]]

```python
state_dict()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/trajectory.py#L202)

Serialize configuration.

#### openenv.core.rubrics.LLMJudge[[openenv.core.rubrics.LLMJudge]]

```python
openenv.core.rubrics.LLMJudge(prompt_template: str, client: LLMClient, score_pattern: str | None = None, default_score: float = 0.0, normalize: bool = True)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/llm_judge.py#L28)

**Parameters:**

prompt_template (*str*) : Template string with &amp;lcub;action} and &amp;lcub;observation} placeholders.

client (*LLMClient*) : An LLMClient instance for making LLM calls.

score_pattern (*str*, *optional*) : Regex to extract the score from the LLM response. Defaults to matching the first decimal number.

default_score (*float*, *optional*, defaults to *0.0*) : Score returned when parsing fails.

normalize (*bool*, *optional*, defaults to *True*) : If True, clamp extracted score to [0, 1].

Rubric that uses an LLM to evaluate agent actions/observations.

The prompt template is formatted with `&amp;lcub;action}` and `&amp;lcub;observation}`
placeholders. The LLM response is parsed for a numeric score.

#### forward[[openenv.core.rubrics.LLMJudge.forward]]

```python
forward(action: Any, observation: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/llm_judge.py#L64)

**Parameters:**

action : The action taken by the agent.

observation : The resulting observation.

**Returns:** `float`

Parsed score from the LLM response.

Evaluate by sending a prompt to the LLM and parsing the score.

#### state_dict[[openenv.core.rubrics.LLMJudge.state_dict]]

```python
state_dict()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/rubrics/llm_judge.py#L104)

Serialize rubric configuration.

## Tools[[openenv.core.tools.RepoInfo]]

#### openenv.core.tools.RepoInfo[[openenv.core.tools.RepoInfo]]

```python
openenv.core.tools.RepoInfo(name: str, url: str, commit: str, clone_url: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/tools/git_server_client.py#L100)

Information about a repository.

#### openenv.core.tools.GitServerClient[[openenv.core.tools.GitServerClient]]

```python
openenv.core.tools.GitServerClient(gitea_url: str, username: str, password: str, workspace_dir: str = '/workspace')
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/tools/git_server_client.py#L109)

**Parameters:**

gitea_url : URL of the Gitea server (e.g., "http://gitea:3000")

username : Gitea username for authentication

password : Gitea password for authentication

workspace_dir : Local workspace directory for cloning repos

Client for connecting to an external Gitea server.

This client is optimized for task-based isolation where:
- Multiple tasks share the same Gitea instance
- Each task has its own isolated workspace
- Fast reset() via git operations (no server restart)
- Repos are pre-migrated to Gitea once

Examples:

```python
import os

# Connect to shared Gitea (credentials from environment)
client = GitServerClient(
    gitea_url=os.getenv("GITEA_URL"),
    username=os.getenv("GITEA_USERNAME"),
    password=os.getenv("GITEA_PASSWORD")
)
client.wait_for_ready()
# Clone repo to workspace
path = client.clone_to_workspace("my-repo", commit="abc123")
# Fast reset to base state
client.reset_workspace("my-repo", commit="abc123")
```

#### clone_to_workspace[[openenv.core.tools.GitServerClient.clone_to_workspace]]

```python
clone_to_workspace(repo_name: str, target_dir: str | None = None, commit: str = 'main')
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/tools/git_server_client.py#L280)

**Parameters:**

repo_name (`str`) : Name of repository to clone.

target_dir (`str`, *optional*) : Target directory name. Defaults to `repo_name`.

commit (`str`, *optional*, defaults to `"main"`) : Commit hash or branch to check out.

**Returns:** `str`

Path to cloned repository.

**Raises:** ``RuntimeError``

- ``RuntimeError`` -- If clone fails.

Clone a repository to the workspace at a specific commit.

This creates a fresh clone optimized for task isolation.

#### execute_git_command[[openenv.core.tools.GitServerClient.execute_git_command]]

```python
execute_git_command(command: str, working_dir: str = '')
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/tools/git_server_client.py#L424)

**Parameters:**

command (`str`) : Git command to execute (without `git` prefix).

working_dir (`str`, *optional*, defaults to `""`) : Working directory relative to workspace.

**Returns:**

`tuple` of (exit_code, stdout, stderr).

Execute a git command in the workspace.

#### get_current_commit[[openenv.core.tools.GitServerClient.get_current_commit]]

```python
get_current_commit(repo_name: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/tools/git_server_client.py#L463)

**Parameters:**

repo_name (`str`) : Name of repository in workspace.

**Returns:** `str`

Commit hash.

Get current commit hash of a workspace repository.

#### list_repositories[[openenv.core.tools.GitServerClient.list_repositories]]

```python
list_repositories()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/tools/git_server_client.py#L241)

**Returns:**

`list` of repository information dictionaries.

List all repositories in Gitea.

#### reset_workspace[[openenv.core.tools.GitServerClient.reset_workspace]]

```python
reset_workspace(repo_name: str, commit: str = 'main')
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/tools/git_server_client.py#L343)

**Parameters:**

repo_name (`str`) : Name of repository (directory in workspace).

commit (`str`, *optional*, defaults to `"main"`) : Commit hash or branch to reset to.

**Returns:** `bool`

True if reset successful.

**Raises:** ``RuntimeError``

- ``RuntimeError`` -- If reset fails.

Fast reset of workspace to base state (optimized for task resets).

This is much faster than re-cloning. It:
1. Checks out the target commit
2. Resets to that commit (hard)
3. Cleans untracked files

#### wait_for_ready[[openenv.core.tools.GitServerClient.wait_for_ready]]

```python
wait_for_ready(timeout: int = 30)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/tools/git_server_client.py#L210)

**Parameters:**

timeout (`int`, *optional*, defaults to `30`) : Maximum seconds to wait.

**Returns:** `bool`

True if server is ready, False otherwise.

Wait for Gitea server to be ready.

#### workspace_exists[[openenv.core.tools.GitServerClient.workspace_exists]]

```python
workspace_exists(repo_name: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/tools/git_server_client.py#L492)

Check if a repository exists in workspace.

## Container providers[[openenv.core.containers.runtime.ContainerProvider]]

#### openenv.core.containers.runtime.ContainerProvider[[openenv.core.containers.runtime.ContainerProvider]]

```python
openenv.core.containers.runtime.ContainerProvider()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L73)

Abstract base class for container providers.

Providers implement this interface to support different container platforms:
- LocalDockerProvider: Runs containers on local Docker daemon
- DockerSwarmProvider: Runs containers as a Docker Swarm service
- KubernetesProvider: optional `openenv[kubernetes]` extra
  (`kubernetes_provider.py`)
- FargateProvider: Runs containers on AWS Fargate
- CloudRunProvider: Runs containers on Google Cloud Run

The provider manages a single container lifecycle and provides the base URL
for connecting to it.

Examples:

```python
provider = LocalDockerProvider()
base_url = provider.start_container("echo-env:latest")
print(base_url)  # http://localhost:8000
# Use the environment via base_url
provider.stop_container()
```

#### close[[openenv.core.containers.runtime.ContainerProvider.close]]

```python
close()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L158)

Release provider-held resources (e.g. SDK clients, connections).

Defaults to a no-op so existing providers are unaffected. Providers that
hold external resources beyond the container itself (such as a cloud SDK
client) override this to release them; it is also invoked on context-
manager exit. Lightweight providers need not override it.

#### start_container[[openenv.core.containers.runtime.ContainerProvider.start_container]]

```python
start_container(image: str, port: Optional[int] = None, env_vars: Optional[Dict[str, str]] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L99)

**Parameters:**

image (`str`) : Provider-specific container *source* identifier. For container-based providers this is a registry image name (e.g. `"echo-env:latest"`); other providers may map it to a provider-specific source (see the provider's documentation).

port (`int`, *optional*) : Port to expose. If `None`, the provider chooses.

env_vars (`dict`, *optional*) : Environment variables to pass to container.

- ****kwargs** : Provider-specific options.

**Returns:** `str`

Base URL to connect to the container (e.g., `"http://localhost:8000"`).

**Raises:** ``RuntimeError``

- ``RuntimeError`` -- If container fails to start.

Start a container from the specified image.

#### stop_container[[openenv.core.containers.runtime.ContainerProvider.stop_container]]

```python
stop_container()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L131)

Stop and remove the running container.

This cleans up the container that was started by start_container().

#### wait_for_ready[[openenv.core.containers.runtime.ContainerProvider.wait_for_ready]]

```python
wait_for_ready(base_url: str, timeout_s: float = 30.0)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L140)

**Parameters:**

base_url (`str`) : Base URL of the container.

timeout_s (`float`, *optional*, defaults to `30.0`) : Maximum time to wait in seconds.

**Raises:** ``TimeoutError``

- ``TimeoutError`` -- If container doesn't become ready in time.

Wait for the container to be ready to accept requests.

This typically polls the /health endpoint until it returns 200.

#### openenv.core.containers.runtime.LocalDockerProvider[[openenv.core.containers.runtime.LocalDockerProvider]]

```python
openenv.core.containers.runtime.LocalDockerProvider()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L177)

Container provider for local Docker daemon.

This provider runs containers on the local machine using Docker.
Useful for development and testing.

Examples:

```python
provider = LocalDockerProvider()
base_url = provider.start_container("echo-env:latest")
# Container running on http://localhost:<random-port>
provider.stop_container()
```

#### start_container[[openenv.core.containers.runtime.LocalDockerProvider.start_container]]

```python
start_container(image: str, port: Optional[int] = None, env_vars: Optional[Dict[str, str]] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L218)

**Parameters:**

image (`str`) : Docker image name.

port (`int`, *optional*) : Port to expose. If `None`, finds an available port.

env_vars (`dict`, *optional*) : Environment variables for the container.

volumes (`dict`, *optional*) : Mapping of host paths or volume names to mount configuration. Each configuration must contain a `bind` container path and may contain a `mode` such as `"ro"` or `"rw"`.

**Returns:** `str`

Base URL to connect to the container.

Start a Docker container locally.

#### stop_container[[openenv.core.containers.runtime.LocalDockerProvider.stop_container]]

```python
stop_container()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L304)

Stop and remove the Docker container.

#### wait_for_ready[[openenv.core.containers.runtime.LocalDockerProvider.wait_for_ready]]

```python
wait_for_ready(base_url: str, timeout_s: float = 30.0)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L336)

**Parameters:**

base_url (`str`) : Base URL of the container.

timeout_s (`float`, *optional*, defaults to `30.0`) : Maximum time to wait in seconds.

**Raises:** ``TimeoutError``

- ``TimeoutError`` -- If container doesn't become ready.

Wait for container to be ready by polling /health endpoint.

#### openenv.core.containers.runtime.DockerSwarmProvider[[openenv.core.containers.runtime.DockerSwarmProvider]]

```python
openenv.core.containers.runtime.DockerSwarmProvider(auto_init_swarm: bool = True, overlay_network: Optional[str] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L406)

Container provider that uses Docker Swarm services for local concurrency.

This provider creates a replicated Swarm service backed by the local Docker
engine. The built-in load-balancer fans requests across the replicas,
allowing multiple container instances to run concurrently on the developer
workstation (mirroring the workflow described in the Docker stack docs).

#### start_container[[openenv.core.containers.runtime.DockerSwarmProvider.start_container]]

```python
start_container(image: str, port: Optional[int] = None, env_vars: Optional[Dict[str, str]] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L443)

**Parameters:**

image (`str`) : Docker image name.

port (`int`, *optional*) : Port to expose. If `None`, finds an available port.

env_vars (`dict`, *optional*) : Environment variables for the container.

replicas (`int`, *optional*, defaults to `2`) : Number of container replicas.

cpu_limit (`float` or `str`, *optional*) : CPU limit passed to `--limit-cpu`.

memory_limit (`str`, *optional*) : Memory limit passed to `--limit-memory`.

constraints (`Sequence[str]`, *optional*) : Placement constraints.

labels (`dict`, *optional*) : Service labels.

command (`Sequence[str]` or `str`, *optional*) : Override container command.

**Returns:** `str`

Base URL to connect to the service.

Start (or scale) a Swarm service for the given image.

#### stop_container[[openenv.core.containers.runtime.DockerSwarmProvider.stop_container]]

```python
stop_container()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L568)

Remove the Swarm service (and keep the Swarm manager running).

#### wait_for_ready[[openenv.core.containers.runtime.DockerSwarmProvider.wait_for_ready]]

```python
wait_for_ready(base_url: str, timeout_s: float = 30.0)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L592)

Wait for at least one replica to become healthy by polling /health.

With Swarm's load balancer, requests round-robin across replicas,
so this only verifies that at least one replica is responding. Some
replicas may still be starting when this returns.

#### openenv.core.containers.runtime.RuntimeProvider[[openenv.core.containers.runtime.RuntimeProvider]]

```python
openenv.core.containers.runtime.RuntimeProvider()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L722)

Abstract base class for runtime providers that are not container providers.
Providers implement this interface to support different runtime platforms:
- UVProvider: Runs environments via `uv run`

The provider manages a single runtime lifecycle and provides the base URL
for connecting to it.

Examples:

```python
provider = UVProvider(project_path="/path/to/env")
base_url = provider.start()
print(base_url)  # http://localhost:8000
provider.stop()
```

#### start[[openenv.core.containers.runtime.RuntimeProvider.start]]

```python
start(port: Optional[int] = None, env_vars: Optional[Dict[str, str]] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L741)

**Parameters:**

port (`int`, *optional*) : Port to expose. If `None`, the provider chooses.

env_vars (`dict`, *optional*) : Environment variables for the runtime.

- ****kwargs** : Additional runtime options.

**Returns:** `str`

Base URL to connect to the runtime.

Start the runtime.

#### stop[[openenv.core.containers.runtime.RuntimeProvider.stop]]

```python
stop()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L763)

Stop the runtime.

#### wait_for_ready[[openenv.core.containers.runtime.RuntimeProvider.wait_for_ready]]

```python
wait_for_ready(timeout_s: float = 30.0)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/providers.py#L770)

Wait for the runtime to be ready to accept requests.

#### openenv.core.containers.runtime.UVProvider[[openenv.core.containers.runtime.UVProvider]]

```python
openenv.core.containers.runtime.UVProvider(project_path: str, app: str = 'server.app:app', host: str = '0.0.0.0', reload: bool = False, env_vars: Optional[Dict[str, str]] = None, context_timeout_s: float = 60.0)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/uv_provider.py#L129)

**Parameters:**

project_path (*str*) : Local path to a uv project (passed to *uv run --project*), or a *git+&amp;lt;url>* spec that is cloned to a temp directory on *start()*.

app (*str*, *optional*, defaults to *"server.app --app"*): ASGI application path for uvicorn.

host (*str*, *optional*, defaults to *"0.0.0.0"*) : Host interface to bind to.

reload (*bool*, *optional*, defaults to *False*) : Whether to enable uvicorn's reload mode.

env_vars (*dict*, *optional*) : Environment variables to pass through to the spawned process. The server inherits this process's environment except variables named like credentials (*TOKEN*, *KEY*, *SECRET* or *PASSWORD* as a part of the name, such as *HF_TOKEN*), so a credential the server needs has to be passed here.

context_timeout_s (*float*, *optional*, defaults to *60.0*) : How long to wait for the environment to become ready.

RuntimeProvider implementation backed by `uv run`.

Examples:

```python
provider = UVProvider(project_path="/path/to/env")
base_url = provider.start()
print(base_url)  # http://localhost:8000
# Use the environment via base_url
provider.stop()
```

#### start[[openenv.core.containers.runtime.UVProvider.start]]

```python
start(port: Optional[int] = None, env_vars: Optional[Dict[str, str]] = None, workers: int = 1, **_: Dict[str, str])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/uv_provider.py#L192)

**Parameters:**

port (`int`, *optional*) : The port to bind the environment to.

env_vars (`dict`, *optional*) : Environment variables to pass to the environment, on top of the constructor's. Credentials reach the server only this way.

workers (`int`, *optional*, defaults to `1`) : The number of workers to use.

**Returns:** `str`

Base URL of the environment.

**Raises:** ``RuntimeError``

- ``RuntimeError`` -- If the environment is already running.

Start the environment via `uv run`.

#### stop[[openenv.core.containers.runtime.UVProvider.stop]]

```python
stop()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/uv_provider.py#L295)

Stop the environment.

#### wait_for_ready[[openenv.core.containers.runtime.UVProvider.wait_for_ready]]

```python
wait_for_ready(timeout_s: float | None = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/uv_provider.py#L273)

**Parameters:**

timeout_s (`float`, *optional*) : Maximum time in seconds to wait for the environment to become ready. Defaults to the provider's `context_timeout_s`.

**Raises:** ``RuntimeError`` or ``TimeoutError``

- ``RuntimeError`` -- If the environment is not running.
- ``TimeoutError`` -- If the environment does not become ready within the timeout.

Wait for the environment to become ready.

#### openenv.core.containers.runtime.daytona_provider.DaytonaProvider[[openenv.core.containers.runtime.daytona_provider.DaytonaProvider]]

```python
openenv.core.containers.runtime.daytona_provider.DaytonaProvider(image: Optional[str] = None, env_vars: Optional[Dict[str, str]] = None, api_key: Optional[str] = None, public: bool = False, resources: Optional[Any] = None, auto_stop_interval: int = 15, target: Optional[str] = None, on_snapshot_create_logs: Optional[Callable[[str], None]] = None, cmd: Optional[str] = None, create_timeout: float = 300)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/daytona_provider.py#L20)

Container provider that runs environments in Daytona cloud sandboxes.

Example:
>>> provider = DaytonaProvider(api_key="your-key")
>>> image = DaytonaProvider.image_from_dockerfile("envs/echo_env/server/Dockerfile")
>>> base_url = provider.start_container(image)
>>> provider.wait_for_ready(base_url)
>>> provider.stop_container()

#### image_from_dockerfile[[openenv.core.containers.runtime.daytona_provider.DaytonaProvider.image_from_dockerfile]]

```python
image_from_dockerfile(dockerfile_path: str, context_dir: str | None = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/daytona_provider.py#L233)

**Parameters:**

dockerfile_path : Path to the Dockerfile on disk.

context_dir : Build context directory.  Defaults to the Dockerfile's grandparent directory, matching the `openenv init` convention where Dockerfiles live in `&amp;lt;env>/server/Dockerfile` and the build context is `&amp;lt;env>/`.  Pass explicitly for non-standard layouts (e.g. `context_dir="."` for repo-root contexts).

**Returns:** A `"dockerfile

&amp;lt;abs_path>"` string to pass to `start_container`.

**Raises:** `FileNotFoundError` or `ValueError`

- `FileNotFoundError` -- If *dockerfile_path* does not exist.
- `ValueError` -- If *context_dir* is given but does not exist,
  or if COPY sources in the Dockerfile cannot be found
  under the resolved context directory.

Validate a Dockerfile and return a `dockerfile:` URI for
`start_container`.

Eagerly validates the Dockerfile (existence, COPY sources,
BuildKit stripping) and stores the processed content in an
internal registry.  The actual `daytona.Image` is created
later inside `start_container`.

#### refresh_preview_url[[openenv.core.containers.runtime.daytona_provider.DaytonaProvider.refresh_preview_url]]

```python
refresh_preview_url()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/daytona_provider.py#L471)

Get a fresh signed preview URL (valid for 24h).

Daytona signed URLs expire after at most 24 hours.  Call this to
get a new one for long-running sessions.  The returned URL points
to the same sandbox — clients will need to reconnect using it.

#### start_container[[openenv.core.containers.runtime.daytona_provider.DaytonaProvider.start_container]]

```python
start_container(image: Optional[str] = None, port: Optional[int] = None, env_vars: Optional[Dict[str, str]] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/daytona_provider.py#L316)

**Parameters:**

image : Docker image name (e.g. `"echo-env:latest"`), `"snapshot:&amp;lt;name>"` to create from a pre-built snapshot, or `"dockerfile:&amp;lt;path>"` returned by `image_from_dockerfile`. May be omitted when supplied to the constructor.

port : Must be `None` or `8000`. Daytona exposes port 8000 via its preview proxy; other ports raise `ValueError`.

env_vars : Environment variables forwarded to the sandbox.

- ****kwargs** : `cmd` (str) to override the server command; remaining kwargs passed through to `Daytona.create()`.

**Returns:**

HTTPS preview URL for the sandbox (base_url).

Create a Daytona sandbox from a Docker image or snapshot.

Daytona does not execute the image's CMD (known bug — ENTRYPOINT
runs, CMD does not).  The server command is resolved in order:

1. Explicit `cmd` passed to the constructor.
2. `cmd` key in `**kwargs` (popped before forwarding).
3. Auto-discovered from `openenv.yaml` inside the sandbox.
4. `CMD` parsed from the Dockerfile (when *image* came from
   `image_from_dockerfile`).

#### stop_container[[openenv.core.containers.runtime.daytona_provider.DaytonaProvider.stop_container]]

```python
stop_container()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/daytona_provider.py#L484)

Delete the Daytona sandbox.

#### strip_buildkit_syntax[[openenv.core.containers.runtime.daytona_provider.DaytonaProvider.strip_buildkit_syntax]]

```python
strip_buildkit_syntax(dockerfile_content: str)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/daytona_provider.py#L167)

Remove BuildKit `--mount=...` flags from `RUN` instructions.

Handles single-line flags, multi-line continuations, and multiple
`--mount` flags spread across continuation lines. Only leading
`--mount` flags are removed (before the actual command starts).

Daytona's `Image.from_dockerfile` does not support BuildKit
`--mount` syntax.  This helper strips the flags so that standard
Dockerfiles (like the ones generated by `openenv build`) can
be used directly.

#### wait_for_ready[[openenv.core.containers.runtime.daytona_provider.DaytonaProvider.wait_for_ready]]

```python
wait_for_ready(base_url: str, timeout_s: float = 120.0)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/daytona_provider.py#L495)

**Parameters:**

base_url : Preview URL returned by `start_container()`.

timeout_s : Maximum seconds to wait.

**Raises:** `TimeoutError` or `RuntimeError`

- `TimeoutError` -- If the sandbox doesn't become ready in time.
- `RuntimeError` -- If the server process died (detected via PID check).

Poll the /health endpoint until the sandbox is ready.

Uses a longer default timeout (120s) than Docker providers because
Daytona sandboxes may have cold-start latency.

#### openenv.core.containers.runtime.aca_provider.ACASandboxProvider[[openenv.core.containers.runtime.aca_provider.ACASandboxProvider]]

```python
openenv.core.containers.runtime.aca_provider.ACASandboxProvider(subscription_id: Optional[str] = None, resource_group: Optional[str] = None, sandbox_group: Optional[str] = None, region: Optional[str] = None, endpoint: Optional[str] = None, credential: Any = None, cpu: str = '1000m', memory: str = '2048Mi', disk_size: Optional[str] = None, auto_suspend_seconds: int = 900, auto_suspend_mode: Literal['Memory', 'Disk'] = 'Memory', labels: Optional[Dict[str, str]] = None, egress_policy: Any = None, anonymous_port: Literal[True] | None = None, image: Optional[str] = None, env_vars: Optional[Dict[str, str]] = None, cmd: Optional[str] = None, working_directory: Optional[str] = None, surface_server_logs: bool = False, create_timeout: int = 300, polling_interval: int = 3, sdk_kwargs: Optional[dict[str, Any]] = None, _adapter: Any = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/aca_provider.py#L256)

Container provider backed by Azure Container Apps Sandboxes.

`start_container`'s `image` is an ACA sandbox source (`disk:<name>` for a
public disk image or `disk-id:<id>` for a private one), not a Docker
registry image. The provider boots an OpenEnv server inside the sandbox,
exposes it on an anonymous ACA port, and returns an `https://` URL that
`EnvClient` connects to over `wss://`.

The environment runs untrusted code, so the provider is secure by default
(proposed RFC 002 security invariants): it requires explicit `anonymous_port=True`
ingress (S2), enforces https/wss transport (S1), offers
`deny_all_egress()` to block the cloud metadata/IMDS endpoint (S3), and
never surfaces raw sandbox output unless `surface_server_logs=True` (S4).

Validated end to end against a live ACA sandbox group; the underlying
`azure-containerapps-sandbox` SDK is preview, so pin it and re-validate
after upgrades (see `tests/test_core/test_aca_provider_integration.py`).

`close()` stops the active sandbox *and* releases the underlying SDK client,
so prefer using the provider as a context manager (or call `close()`
explicitly) for deterministic cleanup. `close()`/`__enter__`/`__exit__` are
defined on `ContainerProvider` (default no-op), so a caller holding a bare
`ContainerProvider` reference can release the client polymorphically.

Only one sandbox is active per provider: calling `start_container` again
before `stop_container()`/`close()` raises `RuntimeError` rather than
orphaning the running sandbox.

```python
with ACASandboxProvider(
    image="disk:my-env",
    anonymous_port=True,
    ...,
) as provider:
    base_url = provider.start_container(cmd=...)
    ...
# sandbox deleted and SDK client closed on exit
```

#### close[[openenv.core.containers.runtime.aca_provider.ACASandboxProvider.close]]

```python
close()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/aca_provider.py#L675)

Stop the active sandbox and close the underlying SDK client.

Overrides the base no-op so a caller can release the preview SDK client
deterministically (also invoked on context-manager exit). The base
`ContainerProvider` defines `close()`/`__enter__`/`__exit__`, so callers
holding a bare `ContainerProvider` can release resources polymorphically.

#### deny_all_egress[[openenv.core.containers.runtime.aca_provider.ACASandboxProvider.deny_all_egress]]

```python
deny_all_egress(allow: Optional[list[str]] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/aca_provider.py#L381)

Build a default-deny ACA `EgressPolicy` (RFC 002 security invariant S3).

Untrusted RL/agent code should not be able to exfiltrate data or reach
the cloud metadata/IMDS endpoint. This returns an `EgressPolicy` whose
default action is `Deny`, with an optional host allowlist (for example a
model endpoint or package registry the environment legitimately needs).

```python
provider = ACASandboxProvider(
    anonymous_port=True,
    egress_policy=ACASandboxProvider.deny_all_egress(
        allow=["my-model.openai.azure.com"]
    ),
)
```

#### start_container[[openenv.core.containers.runtime.aca_provider.ACASandboxProvider.start_container]]

```python
start_container(image: Optional[str] = None, port: Optional[int] = None, env_vars: Optional[Dict[str, str]] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/aca_provider.py#L409)

Start an OpenEnv server in an ACA sandbox and return its base URL.

`image` is an ACA sandbox *source*, not a Docker/OCI registry image: a
bare string or `disk:<name>` for a public disk image, or `disk-id:<id>`
for a private one. It may be omitted when supplied to the constructor.
A value that looks like a container image (a registry path or
`name:tag`) is rejected with guidance. Only port 8000 is supported.
`**kwargs` accepts per-start overrides (`cmd`, `labels`,
`egress_policy`, `anonymous_port`); unknown options raise `ValueError`
so typos cannot silently change sandbox behavior.

#### stop_container[[openenv.core.containers.runtime.aca_provider.ACASandboxProvider.stop_container]]

```python
stop_container()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/aca_provider.py#L647)

Delete the active ACA sandbox.

#### wait_for_ready[[openenv.core.containers.runtime.aca_provider.ACASandboxProvider.wait_for_ready]]

```python
wait_for_ready(base_url: str, timeout_s: float = 120.0)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/aca_provider.py#L603)

Wait for the ACA-hosted OpenEnv server to answer on `/health`.

A `200` on `/health` proves HTTP reachability but not that the exposed
ACA port proxies the `/ws` WebSocket upgrade `EnvClient` needs; the
integration test covers the full `wss://` round-trip. If the captured
server process dies during startup this raises `RuntimeError` (sandbox
output is withheld unless `surface_server_logs=True`; see RFC 002 S4).

#### openenv.core.containers.runtime.modal_provider.ModalProvider[[openenv.core.containers.runtime.modal_provider.ModalProvider]]

```python
openenv.core.containers.runtime.modal_provider.ModalProvider(image: str | None = None, env_vars: dict[str, str] | None = None, app_name: str = 'openenv', use_sandbox_v2: bool | None = None, timeout: int = 300, cmd: str | None = None, cpu: float | None = None, memory: int | None = None, surface_server_logs: bool = False, _adapter: Any = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/modal_provider.py#L112)

Container provider that runs environments in Modal sandboxes.

`start_container`'s `image` is either a registry tag
(`"echo-env:latest"`) or a `"dockerfile:&amp;lt;path>"` reference returned by
`image_from_dockerfile`. The server is exposed on an encrypted Modal
tunnel and the returned `https://` URL is what `EnvClient` connects to
over `wss://`.

The environment runs untrusted code, so the provider is secure by default:
it enforces https/wss transport, treats the tunnel URL as a bearer secret
(never interpolated into errors), and never surfaces raw sandbox output
unless `surface_server_logs=True`.

Only one sandbox is active per provider: calling `start_container` again
before `stop_container()`/``close()` raises `RuntimeError` rather than orphaning the running sandbox. `close()`` (and context-manager exit) stops
the active sandbox.

Example:
```python
with ModalProvider(app_name="openenv", cpu=2.0, memory=4096) as provider:
    image = ModalProvider.image_from_dockerfile(
        "envs/echo_env/server/Dockerfile"
    )
    base_url = provider.start_container(image)
    provider.wait_for_ready(base_url)
# sandbox terminated on exit
```

#### close[[openenv.core.containers.runtime.modal_provider.ModalProvider.close]]

```python
close()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/modal_provider.py#L556)

Stop the active sandbox.

Overrides the base no-op so a caller holding a bare `ContainerProvider`
reference can release the sandbox polymorphically (also invoked on
context-manager exit). `ModalProvider` holds no separate SDK client, so
this is equivalent to `stop_container()`.

#### image_from_dockerfile[[openenv.core.containers.runtime.modal_provider.ModalProvider.image_from_dockerfile]]

```python
image_from_dockerfile(dockerfile_path: str, context_dir: str | None = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/modal_provider.py#L305)

**Parameters:**

dockerfile_path (*str*) : Path to the Dockerfile on disk.

context_dir (*str*, *optional*) : Build context directory.  Defaults to the Dockerfile's grandparent directory, matching the `openenv init` convention where Dockerfiles live in `&amp;lt;env>/server/Dockerfile` and the build context is `&amp;lt;env>/`.  Pass explicitly for non-standard layouts (e.g. `context_dir="."` for repo-root contexts).

**Returns:** `*str*`

A `"dockerfile:&amp;lt;abs_path>"` string to pass to
`start_container`.

**Raises:** `FileNotFoundError` or `ValueError`

- `FileNotFoundError` -- If *dockerfile_path* does not exist.
- `ValueError` -- If *context_dir* is given but does not exist,
  or if COPY sources in the Dockerfile cannot be found
  under the resolved context directory.

Validate a Dockerfile and return a `dockerfile:` URI for
`start_container`.

Eagerly validates the Dockerfile (existence, COPY sources) and stores
its content in an internal registry.  The actual `modal.Image` is
created later inside `start_container`.

#### start_container[[openenv.core.containers.runtime.modal_provider.ModalProvider.start_container]]

```python
start_container(image: str | None = None, port: int | None = None, env_vars: dict[str, str] | None = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/modal_provider.py#L405)

**Parameters:**

image (*str*, *optional*) : Registry image tag (e.g. `"echo-env:latest"`) or `"dockerfile:&amp;lt;path>"` returned by `image_from_dockerfile`. May be omitted when supplied to the constructor.

port (*int*, *optional*) : Must be `None` or `8000`. Modal exposes port 8000 via an encrypted tunnel; other ports raise `ValueError`.

env_vars (*dict*, *optional*) : Environment variables forwarded to the sandbox.

- ****kwargs** : `cmd` (*str*) to override the server command; any remaining keyword arguments are forwarded to `modal.Sandbox.create`.

**Returns:** `*str*`

HTTPS tunnel URL for the sandbox (base_url).

Create a Modal sandbox from a Docker image or Dockerfile.

The sandbox is started with a keep-alive process and the server
command is launched via `exec` afterwards, mirroring the discovery
flow used by other cloud providers. The server command is resolved in
order:

1. Explicit `cmd` passed to the constructor.
2. `cmd` key in `**kwargs` (popped before forwarding).
3. Auto-discovered from `openenv.yaml` inside the sandbox.
4. `CMD` parsed from the Dockerfile (when *image* came from
   `image_from_dockerfile`).

#### stop_container[[openenv.core.containers.runtime.modal_provider.ModalProvider.stop_container]]

```python
stop_container()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/modal_provider.py#L542)

Terminate the Modal sandbox.

#### wait_for_ready[[openenv.core.containers.runtime.modal_provider.ModalProvider.wait_for_ready]]

```python
wait_for_ready(base_url: str, timeout_s: float = 120.0)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/modal_provider.py#L618)

**Parameters:**

base_url (*str*) : Tunnel URL returned by `start_container()`.

timeout_s (*float*, *optional*, defaults to *120.0*) : Maximum seconds to wait.

**Raises:** `TimeoutError` or `RuntimeError`

- `TimeoutError` -- If the sandbox doesn't become ready in time.
- `RuntimeError` -- If the server process died (detected via PID check).

Poll the /health endpoint until the sandbox is ready.

Uses a longer default timeout (120s) than local Docker providers
because Modal sandboxes may have cold-start latency.

#### openenv.core.containers.runtime.novita_provider.NovitaSandboxProvider[[openenv.core.containers.runtime.novita_provider.NovitaSandboxProvider]]

```python
openenv.core.containers.runtime.novita_provider.NovitaSandboxProvider(image: Optional[str] = None, env_vars: Optional[Dict[str, str]] = None, api_key: Optional[str] = None, domain: Optional[str] = None, timeout: int = 3600, metadata: Optional[Dict[str, str]] = None, secure: Optional[bool] = None, cmd: Optional[str] = None, working_directory: Optional[str] = None, surface_server_logs: bool = False, cpu_count: int = 2, memory_mb: int = 1024, template_name: Optional[str] = None, on_build_logs: Optional[Callable[[Any], None]] = None, _adapter: Any = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/novita_provider.py#L518)

Container provider that runs environments in Novita AI sandboxes.

`start_container` accepts either form of source:

- A Docker/OCI registry reference (e.g. `"ghcr.io/org/env:tag"`), which is
  the form the `ContainerProvider` contract specifies. The SDK resolves it
  into a Novita template internally and caches it by image fingerprint.
- A `"template:&amp;lt;id>"` reference returned by
  `image_from_dockerfile`, which builds a template from a local
  Dockerfile.

`image_from_dockerfile` rewrites the Dockerfile for Novita's template
parser, which does not accept multi-stage build definitions. The rewrite is
mechanical and preserves the build's meaning: BuildKit `--mount` flags are
stripped, `ARG`/``--platform` in `FROM` lines are resolved, and a
two-stage build whose stages share one base image is replayed as a single
stage (see *_prepare_dockerfile*). A Dockerfile that does not fit those
rules raises *ValueError* with the registry route as the alternative.

The environment runs untrusted code, so the provider is secure by default: it
enforces https/wss transport (S1) and never surfaces raw sandbox output
unless `surface_server_logs=True` (S4).

Only one sandbox is active per provider: calling `start_container` again
before `stop_container()`/`close()` raises `RuntimeError`` rather than
orphaning the running sandbox.

Examples:

```python
# From a pre-built registry image
with NovitaSandboxProvider(image="ghcr.io/org/echo-env:latest") as provider:
    base_url = provider.start_container()
    provider.wait_for_ready(base_url)

# From a local Dockerfile (builds a template on first use)
image = NovitaSandboxProvider.image_from_dockerfile(
    "envs/echo_env/server/Dockerfile"
)
with NovitaSandboxProvider(image=image) as provider:
    base_url = provider.start_container()
# sandbox killed on exit
```

#### close[[openenv.core.containers.runtime.novita_provider.NovitaSandboxProvider.close]]

```python
close()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/novita_provider.py#L1092)

Stop the active sandbox and release provider-held resources.

Overrides the base no-op so a caller holding a bare `ContainerProvider`
reference can release the sandbox polymorphically (also invoked on
context-manager exit). The Novita client is stateless HTTP per call, so
this is equivalent to `stop_container()`.

#### image_from_dockerfile[[openenv.core.containers.runtime.novita_provider.NovitaSandboxProvider.image_from_dockerfile]]

```python
image_from_dockerfile(dockerfile_path: str, context_dir: Optional[str] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/novita_provider.py#L570)

**Parameters:**

dockerfile_path (*str*) : Path to the Dockerfile on disk.

context_dir (*str*, *optional*) : Build context directory, used to resolve relative `COPY` sources. Defaults to the Dockerfile's grandparent directory, matching the *openenv init* convention where Dockerfiles live in *&amp;lt;env>/server/Dockerfile* and the build context is *&amp;lt;env>/*.

**Returns:** `*str*`

A *"template:&amp;lt;abs_path>"* reference to pass to
*start_container* or the constructor's *image*.

**Raises:** `FileNotFoundError` or `ValueError`

- `FileNotFoundError` -- If *dockerfile_path* does not exist.
- `ValueError` -- If *context_dir* does not exist, if COPY sources cannot
  be found under the resolved context directory, or if the
  Dockerfile cannot be expressed as a Novita template (see
  *_prepare_dockerfile*).

Validate a Dockerfile and return a `template:` reference for
`start_container`.

Eagerly validates the Dockerfile (existence, COPY sources) and stores
the rewritten content in an internal registry. The Novita template is
built later, inside `start_container`, by the adapter's
`build_template` — building needs credentials and network, which this
class-level helper deliberately does not.

The rewrite compensates for Novita's Dockerfile parser, which rejects
multi-stage build definitions (the layout every in-repo OpenEnv
environment uses). BuildKit `--mount` flags are stripped, `ARG` and
`--platform` in `FROM` lines are resolved, and a two-stage build
whose stages share one base image is replayed as a single stage.

Examples:

```python
image = NovitaSandboxProvider.image_from_dockerfile(
    "envs/echo_env/server/Dockerfile"
)
provider = NovitaSandboxProvider(image=image)
base_url = provider.start_container()
```

#### start_container[[openenv.core.containers.runtime.novita_provider.NovitaSandboxProvider.start_container]]

```python
start_container(image: Optional[str] = None, port: Optional[int] = None, env_vars: Optional[Dict[str, str]] = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/novita_provider.py#L873)

**Parameters:**

image (*str*, *optional*) : Either a registry image reference (e.g. *"ghcr.io/org/env:latest"*) or a *"template:&amp;lt;path>"* reference returned by [*~image_from_dockerfile*]. May be omitted when supplied to the constructor.

port (*int*, *optional*) : Must be *None* or *8000*. Novita exposes port 8000 on the sandbox host; other ports raise *ValueError*.

env_vars (*dict*, *optional*) : Environment variables forwarded to the sandbox, overriding the constructor's.

- ****kwargs** : *cmd* (*str*) to override the server command. For a registry OCI image, pass the command explicitly together with any runtime environment the image needs: Novita may not be able to restore the image's `ENV` while converting it to a template. For example, the tbench2 image requires:  `cmd="cd /app/env && PYTHONPATH=/app/env " "/app/.venv/bin/python -m uvicorn server.app:app " "--host 0.0.0.0 --port 8000"`  and any additional values can be supplied through `env_vars`. The provider runs this command as `root` for a registry image; a `template:&amp;lt;id>` keeps the template's configured user. Prefer `NovitaSandboxProvider.image_from_dockerfile(...)` when a local Dockerfile is available; its build preserves the environment and startup configuration in the generated template. Unknown options raise *ValueError* so typos cannot silently change sandbox behavior.

**Returns:** `*str*`

HTTPS sandbox URL for the exposed port (base_url).

**Raises:** `RuntimeError` or `ValueError`

- `RuntimeError` -- If a sandbox is already active on this provider.
- `ValueError` -- If no image is available, the port is unsupported, or an
  unknown option is passed.

Create a Novita sandbox and start an OpenEnv server inside it.

#### stop_container[[openenv.core.containers.runtime.novita_provider.NovitaSandboxProvider.stop_container]]

```python
stop_container()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/novita_provider.py#L1072)

Kill the active Novita sandbox.

`kill()` permanently removes the sandbox: it cannot be resumed or
reconnected, and any snapshot must have been taken beforehand.

#### wait_for_ready[[openenv.core.containers.runtime.novita_provider.NovitaSandboxProvider.wait_for_ready]]

```python
wait_for_ready(base_url: str, timeout_s: float = 120.0)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/novita_provider.py#L1160)

**Parameters:**

base_url (`str`) : Sandbox URL returned by `start_container()`.

timeout_s (`float`, *optional*, defaults to `120.0`) : Maximum seconds to wait.

**Raises:** ``TimeoutError`` or ``RuntimeError``

- ``TimeoutError`` -- If the sandbox doesn't become ready in time.
- ``RuntimeError`` -- If the server process died (detected via PID check).

Poll the /health endpoint until the sandbox is ready.

Uses a longer default timeout (120s) than local Docker providers because
a Novita sandbox is created from an image that may need to be resolved
and built on first use.

A `200` on `/health` proves HTTP reachability but **not** that the
exposed sandbox host proxies the `/ws` WebSocket upgrade `EnvClient`
needs; that requires a real `wss://` round-trip (RFC 002 invariant 2).

#### openenv.core.containers.runtime.hf_sandbox_provider.HFSandboxProvider[[openenv.core.containers.runtime.hf_sandbox_provider.HFSandboxProvider]]

```python
openenv.core.containers.runtime.hf_sandbox_provider.HFSandboxProvider(image: str, env_vars: dict[str, str] | None = None, flavor: str = 'cpu-basic')
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/hf_sandbox_provider.py#L281)

**Parameters:**

image (`str`) : Image to run, for example `hf.co/spaces/openenv/coding_env`. It must provide a `server` command that serves OpenEnv on port 8000.

env_vars (`dict[str, str]`, *optional*) : Environment variables for the server.

flavor (`str`, *optional*, defaults to `"cpu-basic"`) : Hugging Face hardware flavor of the sandbox hosts.

Run an OpenEnv server on Hugging Face infrastructure.

The server runs in a `huggingface_hub` sandbox pool, which bills its hosts as
Hugging Face Jobs. `start_container()` returns a local proxy URL that adds the
Hugging Face auth headers, so any `EnvClient` can connect to it unchanged.

#### openenv.core.containers.runtime.kubernetes_provider.KubernetesProvider[[openenv.core.containers.runtime.kubernetes_provider.KubernetesProvider]]

```python
openenv.core.containers.runtime.kubernetes_provider.KubernetesProvider(image: str | None = None, env_vars: dict[str, str] | None = None, namespace: str = 'default', kubeconfig: str | None = None, context: str | None = None, _adapter: Any = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/kubernetes_provider.py#L403)

Container provider that runs one environment as one Pod and one Service.

`start_container` returns `http://&amp;lcub;name}.&amp;lcub;namespace}:&amp;lcub;port}`. That
address is reachable only from inside the cluster: Pod DNS search expands
it under `svc.&amp;lt;cluster-domain>`. It is plaintext on purpose: in-cluster
HTTP is the connectivity decision for this provider. Cloud sandbox
providers keep their own HTTPS requirement.

The namespace must already exist. This provider does not create namespaces,
Deployments, Ingress objects, or port-forwards.

Only one Pod is active per provider. Calling `start_container` again
before `stop_container()` / `close()` raises `RuntimeError` rather
than orphaning the running Pod. A failed start deletes a resource only
after its OpenEnv ownership labels match. When a create response is lost
and that read fails, the resource stays unconfirmed: `stop_container`
reads the labels again and deletes only on a match. `stop_container` is
safe to call more than once.

The Pod does not mount a service-account token unless
`automount_service_account_token=True` is passed to `start_container`.

Example:
```python
from openenv.core.containers.runtime.kubernetes_provider import (
    KubernetesProvider,
)

provider = KubernetesProvider(namespace="openenv")
base_url = provider.start_container("echo-env:latest")
provider.wait_for_ready(base_url)
provider.stop_container()
```

#### close[[openenv.core.containers.runtime.kubernetes_provider.KubernetesProvider.close]]

```python
close()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/kubernetes_provider.py#L758)

Stop the active Pod and Service.

Overrides the base no-op so a caller holding a bare `ContainerProvider`
reference can release the Pod polymorphically (also invoked on
context-manager exit). Equivalent to `stop_container()`.

#### start_container[[openenv.core.containers.runtime.kubernetes_provider.KubernetesProvider.start_container]]

```python
start_container(image: str | None = None, port: int | None = None, env_vars: dict[str, str] | None = None, **kwargs: Any)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/kubernetes_provider.py#L497)

**Parameters:**

image (*str*, *optional*) : Registry image. May be omitted when supplied to the constructor.

port (*int*, *optional*) : Service port. Defaults to 8000. The container always listens on 8000, matching `LocalDockerProvider`.

env_vars (*dict*, *optional*) : Environment variables for the container. Replaces constructor `env_vars` when passed, including an empty dict.

- ****kwargs** : `resources`, `runtime_class_name`, `image_pull_secrets`, `labels`, `service_account_name`, and `automount_service_account_token`. Unknown keys raise `ValueError`.

**Returns:** `*str*`

`http://&amp;lcub;name}.&amp;lcub;namespace}:&amp;lcub;port}`.

Create one Pod and one Service and return the in-cluster base URL.

#### stop_container[[openenv.core.containers.runtime.kubernetes_provider.KubernetesProvider.stop_container]]

```python
stop_container()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/kubernetes_provider.py#L707)

Delete the Pod and Service this provider created.

Safe when nothing was started and safe to call twice. A 404 from the
API is treated as already deleted. A confirmed resource is deleted by
name. An unconfirmed one is deleted only after a fresh ownership-label
match. If that read fails, the unconfirmed state is kept for another
retry. Objects this instance does not own are left in place, including
a name that was already taken.

#### wait_for_ready[[openenv.core.containers.runtime.kubernetes_provider.KubernetesProvider.wait_for_ready]]

```python
wait_for_ready(base_url: str, timeout_s: float = 120.0)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/core/containers/runtime/kubernetes_provider.py#L774)

**Parameters:**

base_url (*str*) : URL returned by `start_container()`.

timeout_s (*float*, *optional*, defaults to *120.0*) : Maximum seconds to wait. Image pulls are slower than local Docker.

**Raises:** `TimeoutError` or `RuntimeError` or `ValueError`

- `TimeoutError` -- If the Pod does not become ready in time.
- `RuntimeError` -- If the Pod has already ended.
- `ValueError` -- If `base_url` is not the URL this provider returned.

Wait until the Pod is Running and `GET /health` returns 200.

A Failed or Succeeded Pod raises immediately instead of consuming the
timeout. The URL is not included in errors: this method only polls the
Service URL returned by `start_container`.

