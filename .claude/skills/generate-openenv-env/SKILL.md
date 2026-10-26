---
name: generate-openenv-env
description: Generate OpenEnv environments from a concrete use case (for example, "generate an env for the library textarena"). Use when asked to design or implement a new environment under envs/ by researching a target library/API, selecting matching OpenEnv examples, asking key implementation questions, and building models/client/server/openenv.yaml. Do not use for model training or evaluation tasks.
---

# /generate-openenv-env

Build a production-ready OpenEnv environment from a use-case prompt.

## Execute Workflow

When invoked, execute this workflow end-to-end.

### 1. Parse the use case and name the environment

Derive a repo path in the form `envs/<name>_env/`.

- Normalize to snake_case.
- Keep names short and domain-specific.
- Example: "generate an env for the library textarena" -> `envs/textarena_env/`.

### 2. Research the target library/API before coding

Gather the minimum interface facts needed to implement `reset`, `step`, and state serialization.

- Search local docs/examples first.
- Search upstream docs/repo for the target library when local context is insufficient.
- Extract only implementation-critical details:
  - installation/dependency requirements
  - environment creation API
  - action format
  - observation format
  - reward and done semantics
  - special setup (model files, downloads, auth, etc.)

### 3. Mine matching OpenEnv examples

Select 2-3 existing environments as implementation templates.

- Always read `references/openenv-tutorial-01-environments.md` (Part 10) and `references/openenv-docs-environment-builder.md`.
- Prefer `envs/textarena_env` for external-library wrappers with richer state.
- Add one simpler baseline (for example `envs/snake_env` or `envs/echo_env`) to keep the implementation minimal.
- Follow patterns, do not copy blindly.
- Exclude generated or vendored files when mining examples (`.venv/`, `build/`, `site-packages/`, `__pycache__/`).

For a compact checklist and mapping, read `references/env-generation-checklist.md`.

### 4. Ask focused implementation questions

Ask only the questions that materially affect architecture. Use the question bank in `references/env-generation-checklist.md`.

Cover at least:
- action space contract
- observation fields needed by agents
- reward design and terminal conditions
- episode/session configuration knobs
- deployment target and dependency constraints

If answers are unavailable, proceed with explicit assumptions and document them.

### 5. Choose the environment archetype

Choose one archetype before scaffolding:

- Typed step/reset environment (default): use `EnvClient` + typed `Action/Observation[/State]` models.
- MCP tool environment: use `MCPEnvironment` + `MCPToolClient` and MCP action/observation types.
- Specialized client flow (rare): only when the standard clients cannot express required behavior (for example local+remote hybrid clients).

### 6. Scaffold the environment

Use the CLI to scaffold:

```bash
PYTHONPATH=src uv run openenv init <name>_env --output-dir envs
```

This generates all files with correct placeholders replaced, including `pyproject.toml`, `Dockerfile`, and `uv.lock`.

If the CLI is unavailable (import errors, missing dependencies), create the structure manually matching:

```text
envs/<name>_env/
├── __init__.py
├── client.py
├── models.py
├── openenv.yaml
├── pyproject.toml
└── server/
    ├── __init__.py
    ├── app.py
    ├── <name>_environment.py
    └── Dockerfile
```

Use `assets/openenv_env_template/` as a reference for file contents when scaffolding manually.

### 7. Implement with OpenEnv contracts

Implement these files in order:

1. `models.py`
2. `server/<name>_environment.py`
3. `server/app.py`
4. `client.py`
5. `openenv.yaml`
6. `README.md`

Use these standards:
- Use typed models (Action/Observation/State).
- Use `create_app(<factory_or_class>, ActionType, ObservationType, env_name=...)` in `server/app.py`. Pass a class or factory callable, not an instantiated environment.
- **Dual-import pattern** (required in `server/app.py` and `server/<name>_environment.py`): Use `try: from ..models import X / except ImportError: from models import X`. Relative imports work in-repo (`PYTHONPATH=src:envs`); bare imports work in Docker (`PYTHONPATH=/app/env`). The same pattern applies to intra-server imports (e.g., `from .foo import Bar` vs `from server.foo import Bar`).
- `client.py` uses `EnvClient[ActionType, ObservationType, State]` (three type parameters).
- Keep server logic in `server/`, keep client parsing in `client.py`.
- Expose config through environment variables when behavior is likely to vary.
- Keep reward logic inside the environment.
- Prefer reset/step signatures compatible with `Environment`:
  - `reset(seed=None, episode_id=None, **kwargs)`
  - `step(action, timeout_s=None, **kwargs)`
- Set `SUPPORTS_CONCURRENT_SESSIONS=True` only when isolation is real. Set `max_concurrent_envs` in `create_app` accordingly (1 when `False`, >1 when `True`).
- For MCP/tool-call UIs that send stringified JSON arguments, add action validators/parsers in `server/app.py`.
- Export public client/models symbols in `__init__.py`.
- Keep `openenv.yaml` aligned with current scaffold format (`spec_version: 1`, `name`, `type`, `runtime`, `app`, `port`).
- Avoid training/evaluation code paths in this skill.

### 8. Validate before handoff

Run the narrowest useful checks:

```bash
# Verify in-repo imports work (catches missing dual-import pattern)
PYTHONPATH=src:envs uv run python -c "from envs.<name>_env.server.<name>_environment import <ClassName>Environment"

# Build and validate
cd envs/<name>_env
openenv build
openenv validate --level static --skip-build
PYTHONPATH=src:envs uv run pytest envs/<name>_env -q
```

If tests do not exist, run a smoke check:

```bash
PYTHONPATH=src:envs uv run uvicorn envs.<name>_env.server.app:app --port 8000
curl http://localhost:8000/health
openenv validate --url http://localhost:8000
```

### 9. Deliver with assumptions and gaps

Report:
- files created/updated
- chosen archetype (typed vs MCP vs specialized)
- assumptions made due to missing answers
- validation commands executed and outcomes
- remaining risks or follow-up questions

## Guardrails

- Do not route into model training/evaluation workflows.
- Do not invent library APIs; confirm against source docs.
- Do not skip reading at least one existing OpenEnv env before implementation.
- Do not copy outdated manifest patterns from older envs (`name/version/action/observation`-only manifests).
- Do not copy build artifacts or virtualenv files from example envs.
- Do not set `max_concurrent_envs > 1` unless the environment explicitly supports concurrent sessions.
