# RFC: OpenEnv Framework Spec for agent execution environments

**Status**: In Review
**Created**: 10/14/2025
**Amended**: November 12, 2025 _(a "Cloud Sandbox Providers" amendment is proposed below, pending author sign-off — see Amendment History)_
**Authors**: @Darktex, @pankit-eng, @jspisak, @zkwentz
**RFC ID:** 002

## Amendment History

**June 14, 2026** (proposed by @thegovind — pending review/sign-off by the RFC 002 authors): Added the "Cloud Sandbox Providers" subsection — a provider-neutral capability mapping, protocol invariants, and security invariants for adapting hosted sandbox runtimes onto the existing `ContainerProvider` contract without changing the client/server protocol.

**November 12, 2025**: Added tool duality (sim vs prod), Docker Compose patterns, positioning framework (OpenEnv vs systems built on top), and graceful degradation principles.

## Summary

An e2e framework for creating, deploying and using isolated execution environments for agentic RL training, built using Gymnasium style APIs. It provides a clean client-server architecture where environments run as FastAPI servers in Docker containers, and clients interact with them via type-safe HTTP APIs.

## Motivation

### Problem Statement

Building execution environments for AI agents, code execution, or computational tasks typically involves:
- Complex setup and dependency management
- Security concerns with code execution
- Difficulty in scaling and deploying environments
- Lack of standardized interfaces between environments and clients of environments

### Goals

1. **Simplicity**: Simple APIs to interact with the environment from RL training code
2. **Type Safety**: Strongly-typed actions, observations, and state
3. **Isolation**: Each environment runs in its own Docker container
4. **Observability**: Leverage side-car container pattern to observe actions, observation tuples for an RL training episode.


## Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│             RL code(Client Application)                 │
│  ┌────────────────┐              ┌──────────────────┐   │
│  │  Environment   │              │  Environment     │   │
│  │  Client        │              │  Client          │   │
│  │ (HTTPEnvClient)│              │ (HTTPEnvClient)  │   │
│  └────────┬───────┘              └────────┬─────────┘   │
└───────────┼───────────────────────────────┼─────────────┘
            │ HTTP (reset, step, state)     │ HTTP
            │                               │
┌───────────▼───────────────────────────────▼─────────────┐
│              Docker Containers (Isolated)               │
│  ┌──────────────────────┐    ┌──────────────────────┐   │
│  │ FastAPI Server       │    │ FastAPI Server       │   │
│  │   Environment        │    │   Environment        │   │
│  │   Logic              │    │   Logic              │   │
│  └──────────────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Important**: This diagram shows the **HTTP interface** used by RL orchestration for simulation control (`reset()`, `step()`, `get_state()`). The **MCP interface** for agent-environment tool interaction is separate and runs alongside (see "Graceful Degradation to Production" section below and RFC 005).

### Core Abstractions(Already available on the master)

#### 1. Environment (Server-Side)

```python
class Environment(ABC):
    """Base class for all environments."""

    @abstractmethod
    def reset(self) -> Observation:
        """Initialize new episode."""

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Execute action and return observation."""

    @property
    @abstractmethod
    def state(self) -> State:
        """Get current episode state."""
```

**Design Rationale**:
- Familiar interface for RL/environment practitioners
- Clear separation between action execution (step) and state management
- Abstract base class enforces contract across all environments

#### 2. HTTPEnvClient (Client-Side)

```python
class HTTPEnvClient(Generic[ActT, ObsT]):
    """Base class for HTTP environment clients."""

    def reset(self) -> StepResult[ObsT]:
        """Reset environment."""

    def step(self, action: ActT) -> StepResult[ObsT]:
        """Execute action."""

    def state(self) -> State:
        """Get current state."""

    def close(self) -> None:
        """Cleanup resources by signaling to the provider."""
```

**Design Rationale**:

The HTTPEnvClient serves as the primary interface for users to interact with environments, designed with several key principles:

- This base class handles all HTTP communication(resp, req) with the environment
- Generic types (`Generic[ActT, ObsT]`) provide compile-time type safety
- Each environment's concrete client class implements parsing step, observation, and state responses from the server into corresponding data models for the respective response.
- Example: `CodingEnv(HTTPEnvClient[CodeAction, CodeObservation])`
- `state()` method provides visibility into episode metadata
- Explicit `close()` ensures proper resource cleanup

#### 3. Container Providers

```python
class ContainerProvider(ABC):
    """Abstract base for container orchestration."""

    @abstractmethod
    def start_container(self, image: str, ...) -> str:
        """Start container and return base URL."""

    @abstractmethod
    def stop_container(self) -> None:
        """Stop and remove container."""

    @abstractmethod
    def wait_for_ready(self, base_url: str, timeout_s: float) -> None:
        """Wait for container to be ready."""
```

**Design Rationale**:
- Pluggable architecture supports multiple platforms (local Docker, K8s, other orchestration providers)
- Provider abstraction decouples client from deployment details and management with easy integration with existing orchestration solutions
- Consistent interface across all providers
- Higher level RL frameworks can implement their own container providers to integrate with their existing orchestration solutions.

#### Cloud Sandbox Providers

> **Status:** proposed amendment, pending review/sign-off by the RFC 002 authors.
> The normative invariants below (protocol + security) are not finalized until
> the original authors approve.

Cloud sandbox platforms expose the same small set of capabilities under
different names. A cloud sandbox provider is an adapter that maps those
capabilities onto the existing `ContainerProvider` contract **without** changing
the `EnvClient` protocol or blurring the agent/orchestration boundary.

The shared capability surface — and how it maps to OpenEnv:

| Capability (named differently per platform) | Maps to |
|---------------------------------------------|---------|
| Create from a source artifact (image, disk image, snapshot, template) | `start_container(image=...)` |
| Expose a port as a URL (signed URL, tunnel, sandbox port) | `base_url` return value |
| Run a process inside the sandbox | provider-internal start command |
| Inject environment variables | `env_vars` arg |
| Network egress policy | provider-local config |

Cloud sandbox providers generally expose these, so OpenEnv does **not** need a vendor-specific
concept in the core protocol — only a set of invariants that hold for all of
them:

1. **Direct base URL.** `start_container()` returns a `base_url` that an
   `EnvClient` connects to directly, including the WebSocket upgrade on `/ws`.
   Requiring the caller to attach credentials, headers, or SDK objects to
   connect is a core client-auth change and needs a separate RFC. Public or
   self-authenticating ingress is an explicit provider decision, never an
   accidental default.
2. **WebSocket transport is the conformance test.** A `200` on `/health` does
   **not** prove conformance. Providers verify the exposed URL proxies a
   WebSocket upgrade (HTTP/1.1 ingress, not a buffered HTTP/2-only proxy) before
   claiming support, because `EnvClient` speaks WebSocket only.
3. **base_url lifetime and reconnect.** Providers document how long a `base_url`
   stays valid and which operations rotate it; any operation that changes the
   URL or drops the transport requires the orchestrator to reconnect with the
   new `base_url`.
4. **Provider-specific source mapping.** Mapping an OpenEnv environment to a
   runnable sandbox is provider-specific: a registry image that "just runs" on
   one provider may need a build step (disk image, template) on another.
   Providers document their accepted source forms and never imply a Docker
   registry image is universally runnable.
5. **Provider-local control plane, explicit cleanup.** Cloud credentials, signed
   URLs, tunnels, sandbox groups, projects, and regions stay inside the
   provider; core users still see only `ContainerProvider`. Providers clean up
   resources they create — including on failed-start paths — and never delete
   caller-owned disks, snapshots, or shared groups.
6. **Lifecycle is orchestration-only.** Suspend, resume, snapshots, port
   refresh, and egress changes are orchestration operations, never MCP tools
   exposed to the agent. **Automatic** transitions (auto-suspend, scale-to-zero,
   idle timeouts) can drop a live transport mid-episode, so providers choose
   conservative defaults for RL rollouts and document the failure mode. Such
   helpers stay provider-local until multiple providers demonstrate the same
   semantics — OpenEnv does not standardize a cloud-only operation prematurely.
7. **Implementation hygiene.** Network posture (ingress/egress) is documented
   with default-deny egress recommended for untrusted code; provider SDKs are
   optional extras imported lazily, so installing core OpenEnv pulls in no cloud
   SDK; and preview SDKs are wrapped behind a private adapter tested with a fake
   that asserts the real SDK's method names and kwargs, so API churn does not
   leak through OpenEnv.

##### Security invariants

Cloud sandboxes run **untrusted** workloads: the environment server executes
arbitrary RL/agent code, and a compromised environment will try to exfiltrate
data, steal credentials, or reach the orchestrator. Treat the workload as
hostile — no provider feature may rely on in-sandbox code being well-behaved —
and enforce:

S1. **Encrypted transport only.** The returned `base_url` is `https`/`wss`;
   plaintext (`http`/`ws`) is rejected so episode traffic is never sent in the
   clear.
S2. **Ingress URLs are bearer capabilities.** An anonymous or self-authenticating
   URL grants full control of the environment to whoever holds it: never logged,
   scoped and time-boxed, and network-restricted where possible. Authenticated
   ingress is preferred once the client supports it.
S3. **Default-deny egress, block cloud metadata.** For untrusted code the
   recommended posture is default-deny egress with an explicit allowlist, and
   the cloud metadata / IMDS endpoint (`169.254.169.254` and link-local ranges)
   is blocked so the workload cannot steal the sandbox's managed-identity token.
   Providers never silently disable egress filtering.
S4. **Least privilege and secret hygiene.** Control-plane credentials never
   enter the sandbox, and any attached identity has minimum scope. Providers do
   not put secrets, tokens, or ingress URLs in logs or exceptions; captured
   command/server output is withheld by default, and any debug surfacing is an
   explicit opt-in that is bounded and best-effort redacted.
S5. **No command injection.** Commands the provider runs inside the sandbox are
   safely quoted and sourced only from the trusted orchestrator — never
   interpolated from agent- or environment-controlled input.
S6. **Bounded blast radius.** Providers set CPU, memory, disk, and idle-timeout
   limits so a runaway or malicious workload cannot exhaust resources or run up
   unbounded cost; disk images and snapshots are treated as sensitive artifacts
   with scoped access and explicit lifetimes.

All security controls (egress, identity, ports, limits) stay orchestration-only,
behind the same boundary as `reset`/`step`, and are never reachable by the agent
over MCP. These invariants keep cloud sandboxes generalizable without adding
vendor-specific concepts to the core client/server protocol.

**Adding a provider.** Because these are invariants on the *existing*
`ContainerProvider` contract — not new core concepts — a new hosted runtime
becomes an OpenEnv cloud provider by subclassing `ContainerProvider` and
satisfying the invariants above. No change to the `EnvClient`/server protocol is
required, and multiple unrelated provider SDKs can sit behind the same contract
— which is what keeps it genuinely provider-neutral.

### Key Design Decisions

In this RFC, we want to align on four decisions that will shape the overall design of the framework.

#### Decision 1: Baseline API Set

**Chosen Approach**: Define three core APIs as the baseline interface for this framework: `step`, `reset`, and `state`.

**Rationale**:
- **`reset()`**: Initializes a new episode and returns initial observation, providing a clean starting point for agent interactions
- **`step(action)`**: Executes an action and returns an observation, forming the core interaction loop
- **`state()`**: Provides visibility into the current episode state and metadata

These three APIs establish the minimum viable interface for environment interaction and are sufficient for basic RL training workflows. They align with established patterns from Gymnasium and similar frameworks, making them immediately familiar to practitioners.

**Scope**: This RFC focuses exclusively on these baseline APIs. Additional APIs (e.g., `render()`, `seed()`) will be explored in follow-up RFCs. The `actions()` method for action discovery is defined in RFC 005.

#### Decision 2: Environment-Computed Rewards

**Chosen Approach**: Rewards are computed inside the environment and returned as part of the observation.

**Rationale**:
- **Encapsulation**: Reward logic stays with the environment where domain knowledge resides
- **Consistency**: Ensures reward computation is deterministic and reproducible across different client implementations
- **Flexibility**: Environments can use internal state and context not visible to clients for reward computation
- **Standard Pattern**: Aligns with Gymnasium/Gym conventions where rewards are returned from `step()`

The `Observation` base class includes a `reward` field that environments populate:

```python
@dataclass(kw_only=True)
class Observation:
    """Base class for all environment observations."""
    done: bool = False
    reward: Union[bool, int, float, None] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

This design enables environments to compute rewards based on:
- Action outcomes (e.g., exit codes, success/failure)
- Internal state transitions
- Multi-step trajectories
- Domain-specific metrics

Clients receive fully-formed observations with rewards already computed, simplifying the client-side RL loop.

#### Decision 3: HTTP-Based Communication

**Chosen Approach**: Use HTTP/REST for client-server communication

**Rationale**:
- HTTP based RPC is universal and well-understood than other alternatives like grpc or thrift
- Easy to debug with standard tools (curl, Postman)
- Supports language-agnostic clients
- FastAPI provides excellent developer experience

#### Decision 4: Docker-Based runtime isolation and packaging

**Chosen Approach**: Each environment runs in its own Docker container

**Rationale**:
- Strong isolation boundaries compared to process-based isolation
- Reproducible environments with packaged dependencies
- Easy dependency management via Dockerfile
- Industry-standard tooling


### Example Environments

**Purpose**: Test infrastructure, demonstrate patterns, verify deployments

#### Coding Environment

Executes Python code in a sandboxed environment:

```python
from envs.coding_env import CodeAction, CodingEnv

client = CodingEnv.from_docker_image("coding-env:latest")
result = client.step(CodeAction(code="print('Hello, World!')"))
print(result.observation.stdout)     # "Hello, World!\n"
print(result.observation.exit_code)  # 0
client.close()
```

## Tool Duality: Simulation vs Production

Many tools need different implementations in training vs production while maintaining identical interfaces:

**Examples**:
- **Search API**: Production calls actual search; training uses mock
- **Email**: Production sends real emails; training logs to file
- **Database**: Production hits real DB; training uses containerized instance

**Key principle**: The **MCP interface must be identical** to maintain training/production parity (see RFC 005).

### Three-Phase Ecosystem Evolution

**Phase 1 (Current)**: Community provides sim-only tools
- Environment builders create MCP servers for their simulated environments
- Production deployment uses different tooling (acceptable for research)
- Example: SQLite MCP for training, Postgres connector for production

**Phase 2 (6-12 months)**: Tool registry emerges
- Community-maintained mappings: "search_tool (sim) → Algolia (prod)"
- Hugging Face Hub hosts these registries (see future tool registry RFC)
- Still requires manual prod setup, but mapping is documented

**See future tool registry RFC for detailed specification of tool registry format, HF Hub structure, and community contribution workflows.**

**Phase 3 (12+ months)**: Tool providers participate
- Major SaaS companies provide official sim/prod server pairs
- One-line deployment: Specify registry entry, get both modes
- Example: `search: algolia/search-mcp` pulls both sim and prod servers
- Tool providers shipping dual-mode servers becomes standard practice

### Dual-Mode Server Pattern

Tool providers can ship servers that handle both modes:

```python
class SendGridMCPServer:
    def __init__(self):
        self.mode = os.getenv("MODE", "prod")  # "sim" or "prod"

        if self.mode == "sim":
            self.client = MockEmailClient()  # Logs to file
        elif self.mode == "prod":
            self.client = SendGridAPIClient()  # Real API

    @mcp_tool
    def send_email(self, to: str, subject: str, body: str):
        # Same interface, different implementation
        return self.client.send(to, subject, body)
```

**Benefits**:
- Single package to maintain
- Tool provider owns simulation quality
- Realistic test data from source

## Docker Compose: Dual-Mode Deployment

Production and simulation may have different dependency requirements. We use Docker Compose to manage these cleanly:

### Simulation Mode

```yaml
# docker-compose.sim.yml
services:
  env:
    image: openenv/my-env:v1
    environment:
      MODE: sim
    ports:
      - "8080:8080"

  # Lightweight mocks
  mock-db:
    image: postgres:15
    environment:
      POSTGRES_DB: testdb

  mock-email:
    image: openenv/mock-email:v1
```

**Characteristics**:
- Mock services (in-memory database, fake email server)
- Lightweight, fast startup
- No external dependencies
- No API keys required

### Production Mode

```yaml
# docker-compose.prod.yml
services:
  env:
    image: openenv/my-env:v1  # Same image!
    environment:
      MODE: prod
      DB_CONNECTION: ${PROD_DB_URL}
      EMAIL_API_KEY: ${SENDGRID_KEY}
    ports:
      - "8080:8080"
```

**Characteristics**:
- Real services (Postgres, SendGrid)
- API keys, credentials
- Network access, higher latency
- Production-grade reliability

**Key insight**: The environment code is identical—only configuration differs.

## Graceful Degradation to Production

When deploying to production, OpenEnv environments **gracefully degrade** into pure MCP servers:

**Training Mode**:
```
┌─────────────────────────────────────┐
│  HTTP Layer (Simulation + Ops)      │
│  - reset(), step(), get_state()     │
│  - Health checks, metrics           │
├─────────────────────────────────────┤
│  MCP Layer (Agent Tools)            │
│  - search(), execute_sql(), etc.    │
│  - SAME as production               │
└─────────────────────────────────────┘
```

**Production Mode**:
```
┌─────────────────────────────────────┐
│  HTTP Layer (Ops Only)              │
│  - Health checks, metrics, logs     │
│  - NO reset/step (not simulation)   │
├─────────────────────────────────────┤
│  MCP Layer (Agent Tools)            │
│  - search(), execute_sql(), etc.    │
│  - IDENTICAL interface              │
└─────────────────────────────────────┘
```

The agent sees the same MCP interface in both modes. The HTTP layer shifts from simulation control to operational monitoring.

## Dependency Management: Sim vs Prod

**Approach**: Use separate Docker Compose files for different modes

**Training workflow**:
```bash
docker-compose -f docker-compose.sim.yml up
# Fast startup, mock services, no credentials needed
```

**Production workflow**:
```bash
export PROD_DB_URL="postgresql://..."
export SENDGRID_KEY="..."
docker-compose -f docker-compose.prod.yml up
# Real services, production credentials
```

The environment code remains unchanged. Only the orchestration layer differs.

## Positioning: OpenEnv vs Systems Built on OpenEnv

### OpenEnv: The Standard

**Mission**: Source maximum high-quality environment contributions from community

**Characteristics**:
- **Flexible**: Supports both traditional tool calling (RFC 003) and CodeAct (RFC 004) paradigms
- **Open**: Anyone can contribute environments
- **Quality-focused**: High bar for useful, production-relevant environments
- **MCP-native**: Universal interface for all environments (see RFC 005)

**Design philosophy**: Make frontier practices (CodeAct, production-first, progressive disclosure) EASY, not MANDATORY.

**What we optimize for**:
- ✅ Environments that reflect real-world use cases
- ✅ Environments with clear reward signals
- ✅ Environments that work in both training and production
- ❌ Toy environments with no production analog
- ❌ Environments with made-up APIs that don't match real services

This isn't about being exclusive—it's about maintaining a quality bar that makes the ecosystem valuable.

### Systems Built on OpenEnv

**Mission**: Build best-in-class agent training infrastructure for specific use cases

**Characteristics**:
- **Opinionated**: May choose CodeAct-only, specific training algorithms, specific toolsets
- **Customized**: Optimized for particular workloads (e.g., reasoning, coding, customer service)
- **Closed or open**: May be internal systems or community projects
- **Add layers**: Build on OpenEnv foundation with additional infrastructure

**Example**: Internal RL training stack
- 100% CodeAct (no tool-calling mode)
- Custom training infrastructure integration (e.g., TorchForge for async RL)
- Production-first by default (no simulation-only quirks)
- Advanced features (e.g., TimeWalk for tree search, tool-aware checkpointing)
- Uses OpenEnv environments but adds opinionated layers

**The relationship**:
- **OpenEnv provides the foundation**: Environment standard, MCP interface, community contributions
- **Systems add opinions**: Optimizations, integrations, constraints on top
- **Both benefit**: OpenEnv gets community contributions, systems get ecosystem reach

**Analogy**: OpenEnv is like Linux (flexible kernel), systems built on it are like Ubuntu or Red Hat (opinionated distributions).
