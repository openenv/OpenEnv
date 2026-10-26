# AGENTS.md

See `CLAUDE.md` for the full agentic workflow, skills, and standard build/test/lint commands. This file adds cloud-environment specifics on top of that.

## Cursor Cloud specific instructions

### Project shape
OpenEnv is a Python (`uv`) monorepo: a core library + `openenv` CLI in `src/openenv/`, and ~35 self-contained RL "environments" under `envs/` (each its own `pyproject.toml`/`Dockerfile`/`openenv.yaml`). There is **no Node/JS, no shared database, and no single always-on backend**. The product is the framework plus per-environment FastAPI servers; `envs/echo_env/` is the canonical reference env.

### Environment / dependencies
- The startup update script installs `uv` (if missing) and runs `uv sync --all-extras`, which creates `.venv/` with all core + CLI + provider extras. You do not need to reinstall on a fresh session.
- `uv` lives at `~/.local/bin` and is already on `PATH` in login shells (added to `.bashrc`/`.profile`).
- Core deps are enough to run/test the core library and `echo_env`. Most other envs have optional, heavy deps (`torch`, simulators, etc.); their tests auto-skip when deps are absent. Install per-env extras only when working on that env, e.g. `uv pip install -e "envs/coding_env[dev]"`.

### Lint / test / build / run (standard commands live in `CLAUDE.md`)
- Tests need `PYTHONPATH=src:envs`, e.g. `PYTHONPATH=src:envs uv run pytest tests/ -q`. Full suite is ~80s; ~129 tests skip without optional env deps (expected).
- Lint: `uv run usort check src/ tests/`, `uv run ruff format src/ tests/ --check`, `uv run ruff check src/ tests/`. Note: `usort check` currently flags two pre-existing files (`tests/envs/test_grid_world.py`, `tests/envs/test_julia_env.py`) on a clean tree — not introduced by your change.

### Running an environment end-to-end (no Docker needed)
- Start a server (echo_env example): `PYTHONPATH=src:envs uv run python -m echo_env.server.app` (listens on `0.0.0.0:8000`; health at `/health`).
- `import echo_env` resolves only when `envs/` is on `PYTHONPATH`.
- `openenv serve` is a stub that just prints instructions; run the server module (or `uvicorn echo_env.server.app:app`) directly instead.
- Interact via the client: `EchoEnv(base_url="http://localhost:8000").sync()` then `.reset()` / `.list_tools()` / `.step(CallToolAction(...))`.

### Optional Gradio web UI
- The debug web UI is off by default; enable with `ENABLE_WEB_INTERFACE=true` and open `/web/`.

### Docker
- Docker is not required for local dev/test of the core library or pure-Python envs, and is not installed by the update script. Tests marked `@pytest.mark.docker`/`network` are skipped without it. Only set Docker up if you specifically work on container build/run flows.
