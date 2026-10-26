# Core API

The `openenv.core` package provides the core abstractions for building and running environments. For a walkthrough of building an environment, see [Your First Environment](../guides/first-environment).

[Simulation vs Production](../guides/simulation-vs-production) explains when OpenEnv exposes the training loop or direct MCP access, and how MCP tool calls move through `step()`, `step_async()` and `call_tool()`.

For dataset-backed environments that publish enumerable tasks and splits, see the [Task API](../guides/task-api.md) guide.

## Server

### Environment server primitives

[[autodoc]] openenv.core.env_server.interfaces.Message

[[autodoc]] openenv.core.env_server.interfaces.ModelTokenizer

[[autodoc]] openenv.core.env_server.interfaces.Transform

[[autodoc]] openenv.core.env_server.interfaces.TaskProvider

[[autodoc]] openenv.core.env_server.interfaces.Environment

### Types

[[autodoc]] openenv.core.env_server.types.ServerMode

[[autodoc]] openenv.core.env_server.types.HealthStatus

[[autodoc]] openenv.core.env_server.types.WSErrorCode

[[autodoc]] openenv.core.env_server.types.Action

[[autodoc]] openenv.core.env_server.types.Observation

[[autodoc]] openenv.core.env_server.types.ResetRequest

[[autodoc]] openenv.core.env_server.types.ResetResponse

[[autodoc]] openenv.core.env_server.types.StepRequest

[[autodoc]] openenv.core.env_server.types.StepResponse

[[autodoc]] openenv.core.env_server.types.BaseMessage

[[autodoc]] openenv.core.env_server.types.State

[[autodoc]] openenv.core.env_server.types.CodeExecResult

[[autodoc]] openenv.core.env_server.types.EnvironmentMetadata

[[autodoc]] openenv.core.env_server.types.SchemaResponse

[[autodoc]] openenv.core.env_server.types.HealthResponse

[[autodoc]] openenv.core.env_server.types.ListTasksRequest

[[autodoc]] openenv.core.env_server.types.NumTasksRequest

[[autodoc]] openenv.core.env_server.types.GetTaskRequest

[[autodoc]] openenv.core.env_server.types.GetTaskRangeRequest

[[autodoc]] openenv.core.env_server.types.WSResetMessage

[[autodoc]] openenv.core.env_server.types.WSStepMessage

[[autodoc]] openenv.core.env_server.types.WSStateMessage

[[autodoc]] openenv.core.env_server.types.WSCloseMessage

[[autodoc]] openenv.core.env_server.types.WSObservationResponse

[[autodoc]] openenv.core.env_server.types.WSStateResponse

[[autodoc]] openenv.core.env_server.types.WSErrorResponse

[[autodoc]] openenv.core.env_server.types.ConcurrencyConfig

[[autodoc]] openenv.core.env_server.types.ServerCapacityStatus

[[autodoc]] openenv.core.env_server.types.SessionInfo

### Exceptions

[[autodoc]] openenv.core.env_server.exceptions.OpenEnvError

[[autodoc]] openenv.core.env_server.exceptions.ConcurrencyConfigurationError

[[autodoc]] openenv.core.env_server.exceptions.SessionCapacityError

[[autodoc]] openenv.core.env_server.exceptions.SessionNotFoundError

[[autodoc]] openenv.core.env_server.exceptions.SessionCreationError

[[autodoc]] openenv.core.env_server.exceptions.EnvironmentFactoryError

### HTTP server utilities

[[autodoc]] openenv.core.env_server.http_server.HTTPEnvServer

[[autodoc]] openenv.core.env_server.http_server.create_app

[[autodoc]] openenv.core.env_server.http_server.create_fastapi_app

### Web interface helpers

[[autodoc]] openenv.core.env_server.web_interface.ActionLog

[[autodoc]] openenv.core.env_server.web_interface.EpisodeState

[[autodoc]] openenv.core.env_server.web_interface.WebInterfaceManager

[[autodoc]] openenv.core.env_server.web_interface.create_web_interface_app

### Serialization

[[autodoc]] openenv.core.env_server.serialization.deserialize_action

[[autodoc]] openenv.core.env_server.serialization.deserialize_action_with_preprocessing

[[autodoc]] openenv.core.env_server.serialization.serialize_observation

### Transforms

[[autodoc]] openenv.core.env_server.base_transforms.CompositeTransform

[[autodoc]] openenv.core.env_server.base_transforms.NullTransform

### Route configuration

[[autodoc]] openenv.core.env_server.route_config.GetEndpointConfig

[[autodoc]] openenv.core.env_server.route_config.register_get_endpoints

## Clients

### Base client

[[autodoc]] openenv.core.env_client.EnvClient

### Synchronous client

[[autodoc]] openenv.core.sync_client.SyncEnvClient

### Generic client

[[autodoc]] openenv.core.generic_client.GenericEnvClient

[[autodoc]] openenv.core.generic_client.GenericAction

### LLM client

[[autodoc]] openenv.core.llm_client.ToolCall

[[autodoc]] openenv.core.llm_client.LLMResponse

[[autodoc]] openenv.core.llm_client.LLMClient

[[autodoc]] openenv.core.llm_client.OpenAIClient

[[autodoc]] openenv.core.llm_client.AnthropicClient

[[autodoc]] openenv.core.llm_client.create_llm_client

### Shared dataclasses

[[autodoc]] openenv.core.client_types.StepResult

## Auto classes

Load an environment client or its action class by name or Hub repo id, without importing the package. See [Auto-discovery](../guides/auto-discovery).

[[autodoc]] openenv.auto.auto_env.AutoEnv

[[autodoc]] openenv.auto.auto_action.AutoAction

## MCP (Model Context Protocol)

### MCP environment

[[autodoc]] openenv.core.env_server.mcp_environment.MCPEnvironment

### MCP types

[[autodoc]] openenv.core.env_server.mcp_types.JsonRpcErrorCode

[[autodoc]] openenv.core.env_server.mcp_types.McpMethod

[[autodoc]] openenv.core.env_server.mcp_types.JsonRpcError

[[autodoc]] openenv.core.env_server.mcp_types.JsonRpcRequest

[[autodoc]] openenv.core.env_server.mcp_types.JsonRpcResponse

[[autodoc]] openenv.core.env_server.mcp_types.Tool

[[autodoc]] openenv.core.env_server.mcp_types.ToolErrorType

[[autodoc]] openenv.core.env_server.mcp_types.ToolError

[[autodoc]] openenv.core.env_server.mcp_types.ListToolsAction

[[autodoc]] openenv.core.env_server.mcp_types.CallToolAction

[[autodoc]] openenv.core.env_server.mcp_types.ListToolsObservation

[[autodoc]] openenv.core.env_server.mcp_types.CallToolObservation

[[autodoc]] openenv.core.env_server.mcp_types.WSMCPMessage

[[autodoc]] openenv.core.env_server.mcp_types.WSMCPResponse

### MCP client

[[autodoc]] openenv.core.mcp_client.MCPClientBase

[[autodoc]] openenv.core.mcp_client.MCPToolClient

## Harnesses

All classes below are importable from `openenv.core.harness`. See [Harnesses in OpenEnv](../tutorials/harnesses) for when to use each path.

### Agent inside the environment (RFC 005)

[[autodoc]] openenv.core.harness.environment.HarnessEnvironment

[[autodoc]] openenv.core.harness.environment.HarnessAction

[[autodoc]] openenv.core.harness.adapter.AgenticHarnessAdapter

### Rollout sessions

[[autodoc]] openenv.core.harness.rollout.ResourceSessionFactory

### Training captures

[[autodoc]] openenv.core.harness.training.TrainingTrace

[[autodoc]] openenv.core.harness.training.TrainingTurn

## Evaluation

See [Evaluating with Environments](../tutorials/evaluation-inspect) for a worked example.

[[autodoc]] openenv.core.evals.inspect_harness.InspectAIHarness

## Rubrics

[[autodoc]] openenv.core.rubrics.base.Rubric

[[autodoc]] openenv.core.rubrics.containers.Sequential

[[autodoc]] openenv.core.rubrics.containers.Gate

[[autodoc]] openenv.core.rubrics.containers.WeightedSum

[[autodoc]] openenv.core.rubrics.containers.RubricList

[[autodoc]] openenv.core.rubrics.containers.RubricDict

[[autodoc]] openenv.core.rubrics.trajectory.TrajectoryRubric

[[autodoc]] openenv.core.rubrics.trajectory.ExponentialDiscountingTrajectoryRubric

[[autodoc]] openenv.core.rubrics.llm_judge.LLMJudge

## Tools

[[autodoc]] openenv.core.tools.git_server_client.RepoInfo

[[autodoc]] openenv.core.tools.git_server_client.GitServerClient

## Container providers

[[autodoc]] openenv.core.containers.runtime.providers.ContainerProvider

[[autodoc]] openenv.core.containers.runtime.providers.LocalDockerProvider

[[autodoc]] openenv.core.containers.runtime.providers.DockerSwarmProvider

[[autodoc]] openenv.core.containers.runtime.providers.RuntimeProvider

[[autodoc]] openenv.core.containers.runtime.uv_provider.UVProvider

[[autodoc]] openenv.core.containers.runtime.daytona_provider.DaytonaProvider

[[autodoc]] openenv.core.containers.runtime.aca_provider.ACASandboxProvider

[[autodoc]] openenv.core.containers.runtime.modal_provider.ModalProvider

[[autodoc]] openenv.core.containers.runtime.novita_provider.NovitaSandboxProvider

[[autodoc]] openenv.core.containers.runtime.hf_sandbox_provider.HFSandboxProvider
