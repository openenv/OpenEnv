# HF Space Troubleshooting Reference

Use this file when a Space is not `RUNNING` and the root cause is unclear.

## Symptom -> Fix Map

### `RUNTIME_ERROR` with import traceback from `openenv/__init__.py` or `openenv/core/__init__.py`

Likely cause:
- Eager imports trigger unrelated client dependencies (`websockets.asyncio`, etc.) during server boot.

Fix pattern:
- Make package exports lazy (`__getattr__`) in `src/openenv/__init__.py` and `src/openenv/core/__init__.py`.
- Avoid importing full client stack at module import time.

### `RUNTIME_ERROR` with `ModuleNotFoundError` from third-party env package (`finrl`, etc.)

Likely cause:
- Library-level imports pull optional/transitive dependencies not in the env Dockerfile.

Fix pattern:
- Add missing dependency directly in the environment Dockerfile (for example `exchange_calendars`, `wrds`).
- Redeploy only the affected env.

### `RUNTIME_ERROR` or command parse issues with malformed `CMD` in Dockerfile

Likely cause:
- Missing newline at Dockerfile EOF before appended `ENV` lines in staging.

Fix pattern:
- Ensure Dockerfile has trailing newline before appending.
- Redeploy after staging rebuild.

### `BUILD_ERROR` around `uv sync --frozen` or lock mismatch

Likely cause:
- Staged lockfile mismatch or lock not intended for release image.

Fix pattern:
- Remove irrelevant staged `uv.lock`.
- Replace `uv sync --frozen` with `uv sync` in staged Dockerfile when lock strictness is invalid for Space builds.

### `APP_STARTING` for extended periods with no `errorMessage`

Likely cause:
- Runtime orchestrator stall or startup readiness not transitioning.

Fix pattern:
- Check event stream and metrics to confirm activity.
- Run `restart_space(..., factory_reboot=True)`.
- If still stuck, delete and recreate the Space, then redeploy.

### `APP_STARTING` + no web UI requirements

Likely cause:
- Web interface path introduces startup burden or dependency failures.

Fix pattern:
- Set `ENV ENABLE_WEB_INTERFACE=false` in env Dockerfile.
- Keep HTTP server endpoints and `/health` path available.

## Command Snippets

### Fetch event stream quickly

```bash
curl -sS -m 10 https://huggingface.co/api/spaces/openenv/<space-id>/events \
  | sed -n '1,160p'
```

### Restart with factory reboot

```bash
uv run --with huggingface_hub python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
print(api.restart_space("openenv/<space-id>", factory_reboot=True))
PY
```

### Delete and recreate (last resort)

```bash
hf repo delete openenv/<space-id> --repo-type space
scripts/prepare_hf_deployment.sh --hf-namespace openenv --env <env_name> --skip-collection
```

## Deployment Guardrails

- Deploy with version suffix by default (`-vX-Y-Z`).
- Use targeted redeploys first (`--env ...`) before full-fleet reruns.
- Keep collection version (`--version vX.Y.Z`) aligned with deployed suffix set.
