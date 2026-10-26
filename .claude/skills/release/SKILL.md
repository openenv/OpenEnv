---
name: release
description: Release workflow for deploying OpenEnv environments to Hugging Face Spaces and keeping canonical references in sync.
---

# Release Skill

This skill orchestrates the repo-embedded deployment tooling and documents the `openenv`-namespace canonicalization process.

## Prerequisites
- `hf` CLI authenticated (run `hf auth login` or set `HF_TOKEN`).
- Local build dependencies installed so `scripts/prepare_hf_deployment.sh` can stage Docker contexts.
- Access to `scripts/manage_hf_collection.py` for collection updates and discovery.

## Primary Flow
1. **Stage envs with `scripts/prepare_hf_deployment.sh`.** Default arguments deploy every *deployable* env from `envs/`. Pass `--env <name>` to target a subset. The script:
   - Resolves the requested OpenEnv ref for staged dependency rewrites. If `0.2.2` is only a release-candidate label and no `v0.2.2` tag exists yet, the script should fall back to `main` for env dependency rewrites while keeping the Hub suffix at `-0.2.2`.
   - Rewrites loose `openenv-core[core]>=...` specs and direct Dockerfile installs to `git+https://github.com/huggingface/OpenEnv.git@<resolved-ref>` so the sweep does not silently install `0.2.1` from PyPI.
   - Builds a staging tree with `src/`, `envs/<env>/`, and a rewritten `Dockerfile` that sets `BASE_IMAGE` to `ghcr.io/huggingface/openenv-base:latest` unless a hash is supplied.
   - Generates a README with Hub metadata, enforces `openenv`/`openenv-<version>` tags, and adds the `HUB_TAG` used in collection sync.
   - Uses `hf repo create`/`hf upload` plus visibility flags to push Docker spaces.
2. **Suffix naming and privacy.** Deploy to private spaces named `<env>-0.2.2` (set `SPACE_SUFFIX=-0.2.2` or rely on the version-derived default). Use `--private` to keep the collection private for now. The repo deploy script should update only the versioned collection during this phase, not the global tagged collection.
3. **Runtime verification.** For private Spaces, verify through authenticated `hf.space` domains, not anonymous browser URLs. Use `scripts/verify_private_spaces.py --hf-namespace openenv --suffix -0.2.2` or inspect `hf spaces info <space>` plus the runtime domain from `runtime.raw.domains`. Prefer `hf spaces ls --expand=runtime` to confirm `RUNNING`/`SLEEPING`. For stuck or error states, consult `scripts/prepare_hf_deployment.sh` logic or fall back to the `hf-space-recovery` skill.
   - Do not treat `/health` alone as sufficient. This sweep found false positives where `/health` was `200` but runtime use still failed: `chat_env-0.2.2` returned `500` on `/reset`, and `tbench2_env-0.2.2` also returned `500` on `/reset`.
   - For HTTP envs, an authenticated `POST /reset` is the minimum usability gate. If the action schema is simple, follow with a schema-correct `POST /step` probe as well; `snake_env`, `finrl_env`, and `sumo_rl_env` all surfaced step-time failures that a health-only check would miss.
   - Some envs are slow enough that `POST /reset` may time out on a 20-second probe (`git_env`, `unity_env`). Treat those as inconclusive until manually retried with a longer timeout or an env-specific probe.
   - Certain canonical spaces such as `openenv/echo_env` and the TextArena variants expose multiple `READY` runtime domains. The verifier now probes every `READY` domain and reports success once any domain returns usable data, so the canonical status is based on the first passing domain rather than the first listed domain.

## Canonical update decision
- Canonical environments (no suffix) should point to `main` only when the private `-0.2.2` candidate builds and passes health checks.
- If a suffixed space fails to start or its Docker build is broken, leave the canonical reference untouched and pin that env’s `pyproject.toml` dependency to `<0.2.2` to prevent inadvertent upgrades.
- When a suffixed space succeeds, add it to the private versioned collection with `scripts/manage_hf_collection.py --version 0.2.2 --collection-namespace openenv --skip-global-collection --space-id <space>`.
- Existing unsuffixed canonicals are only a subset of `envs/`. Before promotion, map repo envs to actual canonical repos on the Hub:
  - direct matches: `atari_env`, `browsergym_env`, `chat_env`, `coding_env`, `echo_env`, `openspiel_env`, `sumo_rl_env`
  - repo-name mismatches: `repl_env -> openenv/repl`, `tbench2_env -> openenv/tbench2`
  - textarena aliases: `textarena_env` promotes into productized canonicals such as `openenv/sudoku` and `openenv/wordle`, not `openenv/textarena_env`
  - do not infer repo names from the env directory when the namespace already has a canonical alias; list the current `openenv` spaces first and promote into the existing repo when one exists
  - Observed canonical spaces in `openenv` as of this sweep include `openenv/atari_env`, `openenv/browsergym_env`, `openenv/chat_env`, `openenv/coding_env`, `openenv/echo_env`, `openenv/finqa_env`, `openenv/openspiel_env`, `openenv/repl`, `openenv/sudoku`, `openenv/wordle`, `openenv/sumo_rl_env`, and `openenv/tbench2`; keep this list current when planning canonical promotions.

## Collection handling
- Run `scripts/manage_hf_collection.py` with a single release tuple (namespace `openenv`, version `0.2.2`, suffix `-0.2.2`).
- Default behavior adds discovered tagged spaces to the versioned collection, creating or reusing the slug `OpenEnv Environment Hub <version>`.
- Supply `--space-id` explicitly for each successfully redeployed suffixed space to avoid relying on discovery heuristics and to document the tested set.
- For canonical updates, handle repo-name mismatches explicitly. `repl_env` maps to `openenv/repl`, `tbench2_env` maps to `openenv/tbench2`, and `textarena_env` may be represented by productized aliases such as `openenv/sudoku` or `openenv/wordle` rather than an env-dir-matching Space.

## Toolchain reminders
- Always run the release script from the repo root so relative paths to `envs/`, `src/`, and `pyproject.toml` resolve.
- Keep `uv sync` detached at the start of Docker builds: the helper script injects `uv install`/`uv sync` edits automatically.
- `scripts/prepare_hf_deployment.sh --skip-collection` can be used when only validating builds without touching collections.
- For private-space verification, anonymous `https://<space>.hf.space/...` requests return a generic 404. Use an authenticated header from the locally logged-in `hf` token or `scripts/verify_private_spaces.py`.
- Canonical spaces may expose multiple `READY` domains. Do not stop at the first one. In this sweep, `openenv/echo_env` advertised both `openenv-echo-env-v2.hf.space` and `openenv-echo-env.hf.space`, and `openenv/sudoku` advertised both `openenv-textarena.hf.space` and `openenv-sudoku.hf.space`; the first domain in each pair returned 404 while the second served the env correctly. Probe every `READY` domain until one passes.
- If a suffixed repo already contains the intended Dockerfile/source fix but HF still shows an old `BUILD_ERROR`, use `HfApi().restart_space(..., factory_reboot=True)` to force a fresh rebuild of the same commit.
- When `scripts/prepare_hf_deployment.sh` stages legacy Dockerfiles that `COPY src/core/`, it now injects `COPY src/openenv/ /app/src/openenv/` and limits `PYTHONPATH` to `/app/src` because exposing `/app/src/core` earlier shadowed the stdlib `types` module and triggered `ImportError: cannot import name 'GenericAlias'`.

## Env-Specific Fix Patterns
- `browsergym_env`: avoid live Hub-time `git clone` of MiniWoB++ when possible. A pinned tarball snapshot is more reproducible than cloning the repo during every Space build.
- `websearch_env`: staged `openenv-core` rewrites can become `git+https://...` dependencies, so the builder image must have `git` available before `uv sync`.
- `maze_env` and `snake_env`: when a Dockerfile stages the repo root under `/app/env`, add `/app/env/src/core:/app/env/src:/app/env` to `PYTHONPATH` so the Space can use the checked-out repo sources and compatibility shims during validation.
- Legacy Dockerfiles that add `/app/src/core` directly to `PYTHONPATH` can shadow the Python stdlib with `/app/src/core/types.py`, causing startup failures like `ImportError: cannot import name 'GenericAlias' from partially initialized module 'types'`. Stage `src/openenv/` into `/app/src/openenv/` and put only `/app/src` on `PYTHONPATH`.

## Post-release cleanup
- Record which envs passed vs. failed. Failed envs should stay pinned below `0.2.2` until a fix is committed.
- Archive or delete test suffix spaces once their artifacts are promoted to canonical releases to reduce clutter in the `openenv` namespace.
- Capture `hf spaces info`/`curl .../health` output for the final success set so the release briefing notes the exact runtime status used to flip canonical references.
