# System Invariants

These invariants must NEVER be violated. If a change would violate them, stop and flag for human review.

## API Invariants

1. **Gymnasium API signatures**
   - `reset(seed?, episode_id?) -> Observation`
   - `step(action) -> Observation`
   - `state -> State`
   - These signatures must not change without a major version bump

2. **Generic type safety**
   - All environments must use `Environment[ActT, ObsT, StateT]` generics
   - All clients must use `EnvClient[ActT, ObsT, StateT]` generics
   - Types must match between client and server

3. **Pydantic serialization**
   - All wire types (Action, Observation, State) must be Pydantic models
   - Serialization must be JSON-compatible

## Security Invariants

1. **Agent isolation**
   - Agents cannot access reset/simulation controls
   - The WebSocket interface for reset/step is for orchestration only
   - MCP tools must not expose simulation control to agents

2. **Container isolation**
   - Environments run in isolated Docker containers
   - Containers must not have access to host filesystem (except explicitly mounted volumes)
   - Network access must be explicitly configured

3. **No credential exposure**
   - Never log API keys, tokens, or secrets
   - Never include credentials in error messages
   - Use environment variables for sensitive configuration

## Architectural Invariants

1. **Dual API boundary** (see RFC 001, RFC 004)

   OpenEnv exposes two distinct APIs to two different boundaries:

   | Boundary | API | Purpose |
   |----------|-----|---------|
   | **Agent** | MCP (Model Context Protocol) | Tools the agent uses to interact with the environment |
   | **Infrastructure** | Gym-like (`reset`, `step`, `state`) | Simulation control for training orchestration |

   **Critical**: The Gym-like API is NOT accessible to the agent being trained.

   **Why?** The agent must not be able to call `reset()`. If an agent could reset after crashing a car, it would learn that consequences are reversible - which breaks the training paradigm. The infrastructure calls `reset()` to clean up for the next episode, but from the agent's perspective, the episode simply ends.

   **Violations to flag:**
   - Exposing `reset()`, `step()`, or `state()` via MCP tools
   - Giving agents direct access to the Gym-like WebSocket API
   - Any mechanism that lets an agent trigger simulation control

2. **Client-server separation**
   - Clients must never import from `server/` directory
   - Server code must never import client code
   - Shared code goes in `models.py`

3. **Rewards in environment**
   - Reward computation must stay inside environment boundary
   - External reward augmentation uses Transform pipeline
   - Transforms are server-side only

4. **Communication patterns**
   - WebSocket for all environment communication (Gym-like API + metadata)
   - No custom protocols

   **Note**: We are in the process of deprecating HTTP (see PR #252) in favor of WebSocket-only, but we are still transitioning and both protocols are currently available.

## Breaking Change Policy

- **Pre-1.0**: Breaking changes acceptable if documented in release notes
- **Post-1.0**: Semantic versioning strictly enforced
  - MAJOR: Breaking changes
  - MINOR: New features, backward compatible
  - PATCH: Bug fixes only

### Deprecations

Before removing an API, CLI flag, or environment, deprecate it first: keep it working until a stated release and warn whoever uses it.

The warning must say:

- **When it goes away**: the OpenEnv release that removes it.
- **What to use instead**: concrete enough to copy (the new import, call, or command), plus a docs link when the replacement is more than a one-line change.

```python
warnings.warn(
    "`foo_env` is deprecated and will be removed in OpenEnv 0.8.0. "
    "Use `bar_env` instead: `BarFactory(server_url)`. See https://...",
    FutureWarning,
    stacklevel=2,
)
```

- Use `FutureWarning`, not `DeprecationWarning`. Python hides `DeprecationWarning` unless it is raised from `__main__`, so users importing from a module or a notebook would never see it.
- Say the same in the docs (README, doc page, tutorials), with the same version.
- Pre-1.0, leave at least two minor releases between the first release that warns and the removal, more for widely used pieces. Remove in the announced release, and list the removal in its release notes.

## Violation Response

If you identify a potential invariant violation:

1. **Stop** - Do not proceed with the change
2. **Flag** - Create an ALIGNMENT FLAG with:
   - Which invariant is at risk
   - Why the change might violate it
   - Suggested reviewer
3. **Wait** - Get human approval before proceeding

Example:
```
**ALIGNMENT FLAG**: Client importing server module
- **Invariant at risk**: Client-server separation
- **The concern**: client.py imports from server/environment.py
- **Suggested reviewer**: @darktex
```
