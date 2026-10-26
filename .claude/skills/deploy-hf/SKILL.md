---
name: deploy-hf
description: Deploy an OpenEnv environment to Hugging Face Spaces. Use when asked to deploy, push to Hugging Face, or update a space.
allowed-tools: Bash, Read
---

# Deploy to Hugging Face Spaces

Deploy an OpenEnv environment to Hugging Face Spaces using the OpenEnv CLI.

## When to Use This Skill

- User asks to "deploy to Hugging Face"
- User says "push to Hugging Face Spaces"
- User wants to update an existing space
- After implementing new features that need to be tested in production

## Prerequisites

Before deploying, ensure:
1. The environment has an `openenv.yaml` file
2. The environment has a `server/Dockerfile`
3. You have Hugging Face credentials configured (automatic via huggingface-cli)

## Instructions

### 1. Identify the Environment

Determine which environment to deploy:
- If user specifies: use that environment (e.g., "carla_env", "browser_env")
- If in environment directory: use current directory
- Otherwise: ask the user

### 2. Determine the Repository ID

Repository ID format: `username/space-name`

- If user provides full ID: use it (e.g., "sergiopaniego/carla-env-real-updated")
- If user provides only space name: construct ID with their username
- Check `openenv.yaml` for default repo-id
- Otherwise: ask the user

### 3. Pre-Deployment Setup

**IMPORTANT**: Always run from the project root directory.

Before deploying, ensure OpenEnv is installed:

```bash
cd /path/to/OpenEnv  # Navigate to project root if needed
uv pip install -e .
```

If this fails with "does not appear to be a Python project", you're not in the project root.

### 4. Run the Deployment Command

Execute the deployment:

```bash
PYTHONPATH=src uv run python -m openenv.cli push <environment-dir> --repo-id <username/space-name>
```

**Parameters**:
- `<environment-dir>`: Path to environment (e.g., `envs/carla_env`)
- `--repo-id`: Hugging Face Spaces repository ID (e.g., `sergiopaniego/carla-env-real-updated`)

**Optional flags**:
- `--private`: Deploy as a private space
- `--no-interface`: Disable the web interface (deploy API-only)
- `--base-image <image>`: Override the base Docker image
- `--hardware <hw>` / `-H <hw>`: Request Hugging Face Space hardware (e.g. `t4-medium`, `a10g-small`, `cpu-basic`)

### 5. Verify Deployment

After successful deployment:
1. Note the Space URL returned by the command
2. **Wait for build to complete:**
   - CPU environments: ~5 minutes
   - GPU environments (CARLA): ~30-60 minutes
3. Check the space status at the URL
4. Test with a simple health check once build completes:
   ```bash
   curl https://<username>-<space-name>.hf.space/health
   ```

## Example Usage

### Deploy carla_env to existing space

```bash
PYTHONPATH=src uv run python -m openenv.cli push envs/carla_env --repo-id sergiopaniego/carla-env-real-updated
```

### Deploy echo_env as private space

```bash
PYTHONPATH=src uv run python -m openenv.cli push envs/echo_env --repo-id username/my-echo-env --private
```

### Deploy with GPU hardware

```bash
PYTHONPATH=src uv run python -m openenv.cli push envs/carla_env --repo-id username/carla-env --hardware t4-medium
```

### Deploy with custom base image

```bash
PYTHONPATH=src uv run python -m openenv.cli push envs/browser_env --repo-id username/browser-env --base-image nvidia/cuda:11.8.0-runtime-ubuntu22.04
```

## Output Format

Report deployment status:

```
## Hugging Face Deployment

### Environment
- Environment: <env-name>
- Directory: <path>
- Dockerfile: <path-to-dockerfile>

### Deployment
- Repository ID: <username/space-name>
- Space URL: <https://huggingface.co/spaces/username/space-name>
- Status: ✓ Deployed successfully

### Next Steps
1. Wait for space to build (5 min for CPU, 30-60 min for GPU/CARLA)
2. Visit space URL to check build status
3. Test environment once build completes

### Testing Commands
```bash
# Health check
curl https://<username>-<space-name>.hf.space/health

# Reset environment
curl -X POST https://<username>-<space-name>.hf.space/reset

# Step action
curl -X POST https://<username>-<space-name>.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "observe"}'
```
```

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'openenv'"

**Solution**: Install OpenEnv first (must be run from project root):
```bash
cd /path/to/OpenEnv  # Navigate to project root
uv pip install -e .
```

### Error: "does not appear to be a Python project"

**Cause**: You're not in the project root directory.

**Solution**: Navigate to the OpenEnv project root where `pyproject.toml` exists:
```bash
cd /Users/sergiopaniegoblanco/Documents/Projects/OpenEnv  # Adjust path
uv pip install -e .
```

### Error: "Directory does not exist"

**Solution**: Ensure you're passing the correct environment directory path:
```bash
# Correct
PYTHONPATH=src uv run python -m openenv.cli push envs/carla_env --repo-id ...

# Incorrect
PYTHONPATH=src uv run python -m openenv.cli push carla_env --repo-id ...
```

### Error: "Authentication required"

**Solution**: Login to Hugging Face CLI first:
```bash
huggingface-cli login
```

### Space build fails

**Solutions**:
1. Check Dockerfile syntax and dependencies
2. Verify hardware requirements (GPU spaces need `--hardware` setting on HF)
3. Check space logs on Hugging Face for detailed errors
4. Ensure `openenv.yaml` is valid

## Common Environments

| Environment | Path | Typical Repo ID | Hardware |
|-------------|------|-----------------|----------|
| carla_env (standalone) | envs/carla_env | username/carla-env-real | GPU (T4/A10G) |
| carla_env (mock) | envs/carla_env | username/carla-env-mock | CPU |
| echo_env | envs/echo_env | username/echo-env | CPU |
| browser_env | envs/browser_env | username/browser-env | CPU |
| tbench2_env | envs/tbench2_env | username/tbench2-env | CPU |

## Notes

- Deployment requires Hugging Face authentication (automatic if `huggingface-cli` is logged in)
- By default, spaces are **public** (use `--private` for private spaces)
- By default, **web interface is enabled** (use `--no-interface` for API-only)
- GPU spaces can request hardware via `--hardware` (e.g. `--hardware t4-medium`)
- Build times vary: CPU (~5 min), GPU with CARLA (~30-60 min)
- The CLI automatically moves Dockerfile to repository root for Hugging Face compatibility

## Related Documentation

- [DEPLOYMENT_GUIDE.md](../../envs/carla_env/DEPLOYMENT_GUIDE.md) - Detailed deployment modes
- [README.md](../../README.md) - OpenEnv overview
- Hugging Face Spaces Docs: https://huggingface.co/docs/hub/spaces
