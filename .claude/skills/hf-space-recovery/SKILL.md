---
name: hf-space-recovery
description: Diagnose and recover failing or stuck Hugging Face Space deployments for OpenEnv environments. Use when deploying envs from `envs/` to the Hub (`openenv` namespace with version suffixes), when Spaces are in `BUILDING`/`APP_STARTING`/`RUNTIME_ERROR`, or when release collections need to be reconciled after targeted redeploys.
---

# HF Space Recovery

Use this skill to recover OpenEnv Hub deployments quickly with minimal blast radius.

## Execute This Workflow

### 1) Confirm release tuple

Use a single release tuple across all commands:
- Namespace: `openenv`
- Version: `vX.Y.Z`
- Space suffix: `-vX-Y-Z`

Default to a version suffix and treat unsuffixed Spaces as legacy.

### 2) Snapshot runtime status

Collect all versioned spaces and isolate non-running ones:

```bash
hf spaces ls --author openenv --limit 500 --expand=runtime \
  | jq -r '.[] | select(.id|test("-v[0-9]+-[0-9]+-[0-9]+$")) \
    | [.id, .runtime.stage, (.runtime.raw.errorMessage // "")] | @tsv' \
  | sort
```

Treat `RUNNING` and `SLEEPING` as healthy. Triage everything else.

### 3) Classify and extract signal

- `RUNTIME_ERROR`: read traceback from `.runtime.raw.errorMessage`.
- `BUILD_ERROR`: read build error text from runtime info, then patch Dockerfile/deps.
- `APP_STARTING` longer than 10 minutes: inspect event stream and metrics before changing code.

```bash
hf spaces info openenv/<space-id> --expand=runtime
curl -sS -m 10 https://huggingface.co/api/spaces/openenv/<space-id>/events | sed -n '1,140p'
curl -sS -m 10 -i https://huggingface.co/api/spaces/openenv/<space-id>/metrics | sed -n '1,120p'
```

Read `references/troubleshooting.md` for symptom-to-fix mappings.

### 4) Apply minimal fix and targeted redeploy

Prefer targeted redeploys over full-fleet pushes:

```bash
scripts/prepare_hf_deployment.sh \
  --hf-namespace openenv \
  --env <env_name> \
  --skip-collection
```

Use `openenv` CLI as a supplement, not a replacement, for release triage:
- Validate env layout quickly (`uv run openenv validate ...`) when applicable.
- Keep release deploys on `scripts/prepare_hf_deployment.sh` to preserve suffix/pinning behavior.

### 5) Unstick runtime when code is already good

If Space remains in `APP_STARTING` with no actionable error:

```bash
uv run --with huggingface_hub python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.restart_space("openenv/<space-id>", factory_reboot=True)
PY
```

If still stuck, force recreation as last resort:

```bash
hf repo delete openenv/<space-id> --repo-type space
scripts/prepare_hf_deployment.sh --hf-namespace openenv --env <env_name> --skip-collection
```

### 6) Verify and close

Verify both runtime stage and health endpoint:

```bash
hf spaces info openenv/<space-id> --expand=runtime
curl -sS -m 10 https://<space-subdomain>.hf.space/health
```

Then verify fleet-wide:

```bash
hf spaces ls --author openenv --limit 500 --expand=runtime \
  | jq -r '.[] | select(.id|test("-v[0-9]+-[0-9]+-[0-9]+$")) \
    | select(.runtime.stage!="RUNNING" and .runtime.stage!="SLEEPING") \
    | [.id, .runtime.stage] | @tsv' | sort
```

### 7) Reconcile collection

When targeted deploys are done, update collection membership for the same version:

```bash
python3 scripts/manage_hf_collection.py \
  --version vX.Y.Z \
  --collection-namespace openenv \
  --space-id openenv/<space-id>
```

Add one `--space-id` per redeployed space.
