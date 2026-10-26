# OpenEnv Env Generation Checklist

Use this file while running `/generate-openenv-env`.

> **Notation:** This skill is invoked as `/generate-openenv-env` in Claude Code.
> The `$` prefix may appear in older agent configs (e.g., `agents/openai.yaml`)
> where it denotes a tool/skill reference in that platform's convention.

## Source Priorities

Read in this order:

1. `references/openenv-tutorial-01-environments.md` (Part 10 pattern)
2. `references/openenv-docs-environment-builder.md` (scaffold + validation flow)
3. `assets/openenv_env_template/` (canonical scaffold and file defaults)
4. A close environment match in `envs/`

## Example Mapping

Pick the closest template first, then one simpler baseline.

- External text or turn-based game libraries: `envs/textarena_env`
- Gym-like RL wrappers: `envs/snake_env`, `envs/dm_control_env`
- MCP-first tool-calling environments: `envs/echo_env`, `envs/finqa_env`
- Multi-tool reasoning wrappers: `envs/reasoning_gym_env`, `envs/calendar_env`
- Browser/web task wrappers: `envs/browsergym_env`, `envs/websearch_env`

## Archetype Selection

Select exactly one baseline architecture:

1. Typed step/reset env:
   - `EnvClient`
   - typed `Action`, `Observation`, optional custom `State`
2. MCP tool env:
   - `MCPEnvironment`
   - `MCPToolClient`
   - MCP tool actions/observations
3. Specialized client:
   - only if typed/MCP clients are insufficient (for example local+remote hybrid execution)

## Discovery Commands

Use fast repository search:

```bash
rg --files envs -g '!**/.venv/**' -g '!**/build/**' -g '!**/__pycache__/**' -g '!**/site-packages/**' \
  | rg 'openenv.yaml|models.py|client.py|server/(app.py|.*environment.py|Dockerfile)$'
rg -n "class .*Environment|class .*Env\\(|create_app\\(" envs/<candidate_env>
```

## Compatibility Checks

Verify these before finalizing:

- `server/app.py` passes a class or factory to `create_app` (not an instance).
- `reset` and `step` signatures remain compatible with OpenEnv `Environment` expectations.
- Concurrency settings are coherent:
  - `SUPPORTS_CONCURRENT_SESSIONS=True` only when session isolation is safe.
  - `max_concurrent_envs > 1` only when the above is true.
- `openenv.yaml` uses current manifest style (`spec_version: 1`, `name`, `type`, `runtime`, `app`, `port`).
- `__init__.py` exports the expected public client and model symbols.

## Question Bank

Ask only what changes architecture or contracts.

1. Which upstream environment/library object should be wrapped?
2. What exact action payload should agents send?
3. What observation fields are mandatory for policy decisions?
4. How should reward be computed, and what ends an episode?
5. Which runtime knobs should be env vars?
6. Should the environment be deterministic or stochastic by default?
7. Are there dependency limits (GPU, system packages, download size)?
8. Is deployment local-only, HF Space, or both?

## Done Criteria

Mark complete only when all are true:

- `envs/<name>_env/openenv.yaml` exists, uses `spec_version: 1`, and points to `server.app:app`.
- `models.py` defines typed action/observation/state.
- `server/<name>_environment.py` implements `reset`, `step`, and `state`.
- `server/app.py` calls `create_app` with action/observation classes.
- `client.py` matches the selected archetype and correctly serializes/parses data.
- `__init__.py` exports the public API.
- `README.md` includes quickstart and configuration.
- `openenv build` and `openenv validate --level static --skip-build` pass, or failures are documented.
- Runtime smoke check is executed (`/health`; optionally `openenv validate --url`).
