# Deploying an Environment

This page covers packaging and sharing an environment once it runs locally: dependencies and the Dockerfile, `openenv build`, `openenv validate`, `openenv push` and connecting to the deployed environment. To create the environment first, follow [Your First Environment](../guides/first-environment.md).

| Command | Description |
|---------|-------------|
| `openenv build` | Build the Docker image locally |
| `openenv validate --level static --skip-build` | Validate the manifest contract in `openenv.yaml` |
| `openenv push` | Deploy to a Hugging Face Space |
| `openenv push --repo-id NAME` | Deploy to a specific Space |
| `openenv push --private` | Deploy as a private Space |
| `openenv push --registry ghcr.io/ORG` | Push the image to a Docker registry instead |

See the [CLI reference](../reference/cli.md) for every command and flag.

## Dependencies and Dockerfile

Declare Python dependencies in the environment's `pyproject.toml` and regenerate `uv.lock` with `uv lock`. Install anything else (system packages, binaries) in `server/Dockerfile`, before the step that removes temporary files.

The template's [`server/Dockerfile`](https://github.com/huggingface/OpenEnv/blob/main/src/openenv/cli/templates/openenv_env/server/Dockerfile) is a multi-stage build on `ghcr.io/huggingface/openenv-base` that installs your dependencies with `uv sync` and serves `server.app:app` on port 8000. Keep building from `openenv-base` so shared tooling stays available.

## Build the image

From the environment folder (needs Docker):

```bash
openenv build
```

`openenv build` works for standalone environments and for ones inside the OpenEnv repo, and sets the build arguments accordingly. Useful flags:

- `--tag/-t`: override the default tag, `openenv-<env_name>` without its `_env` suffix (`openenv-my` for `my_env`)
- `--build-arg KEY=VALUE`: pass Docker build arguments (repeatable)
- `--dockerfile/-f` / `--context/-c`: custom locations when experimenting
- `--no-cache`: force fresh dependency installs

## Validate

```bash
openenv validate --level static --skip-build
```

`openenv validate` reads the `validation:` contract in `openenv.yaml`, checks the normalized manifest against the selected severity policy and exits non-zero when a required check fails. The report's `levels_run` field records which levels ran. To validate a running server, use `openenv validate --url http://localhost:8000`.

## Push to Hugging Face Spaces

```bash
# Push the environment in the current folder to <your-username>/<env_name>
openenv push

# Push to a specific repo or namespace
openenv push --repo-id my-org/my-env

# Push to a Docker registry (web UI disabled by default)
openenv push --registry ghcr.io/my-org

# Override the base image and make the Space private
openenv push --base-image ghcr.io/huggingface/openenv-base:latest --private

# Set Space variables and secrets at push time
openenv push -e OPENSPIEL_GAME=tic_tac_toe --secret OPENAI_API_KEY=sk-...
```

Options:

- `DIRECTORY` (positional): path to the environment (defaults to the current directory)
- `--repo-id/-r`: Space name, as `name` or `namespace/name`
- `--registry`: push the image to Docker Hub, GHCR, etc.
- `--interface/--no-interface`: toggle the web UI (on by default for Spaces)
- `--base-image/-b`: override the Dockerfile `FROM`
- `--private`: make the Space private
- `--env-var/-e KEY=VALUE`: set a public Space variable (repeatable), overriding matching keys from `variables:` in `openenv.yaml`
- `--secret KEY=VALUE`: set a private Space secret (repeatable). The value is never logged
- `--hardware/-H`: Space hardware (for example `t4-medium`)
- `--count/-n`: deploy several Space instances, each with a numeric suffix
- `--create-pr`: open a Pull Request instead of pushing to the default branch
- `--exclude`: an ignore file with globs to leave out of the upload
- `--delete <pattern>`: delete matching remote Space files that are not uploaded again (repeatable), as in `hf upload --delete`

The command logs you in if needed, validates `openenv.yaml`, adds the Hugging Face frontmatter to the README when needed and uploads the bundle. Like `hf upload`, it never deletes remote files on its own. Pass `--delete <pattern>` (repeatable) to remove matching Space files that are not uploaded again, e.g. `--delete "*"` to make the Space match the bundle exactly. Local build artifacts (`build/`, `*.egg-info`, `__pycache__`, dotfiles) are never uploaded.

Space variables and secrets are only applied on direct Hugging Face Space pushes. They are not available with `--registry` and cannot be staged through `--create-pr`.

### Declare public variables in `openenv.yaml`

Defaults that belong with the environment (game name, benchmark, max steps) go in a `variables:` block in `openenv.yaml`. `openenv push` applies them to the Space:

```yaml
variables:
  OPENSPIEL_GAME: catch
```

CLI `-e` overrides matching keys. Put secrets (API keys, tokens) only on the CLI with `--secret KEY=VALUE`, never in the yaml.

To fork or update someone else's environment, see [Contributing Environments](contributing-envs.md).

## Use the deployed environment

```python
from my_env import MyAction, MyEnv

# Pull the Space's image and run it locally (needs Docker)
client = MyEnv.from_env("my-org/my-env").sync()
# Or start a container from a local image
client = MyEnv.from_docker_image("openenv-my:latest").sync()
# Or connect to a server that is already running
client = MyEnv(base_url="http://localhost:8000").sync()

with client:
    result = client.reset()
    result = client.step(MyAction(message="Hello!"))
    state = client.state()
```

`from_docker_image()` and `from_env()` return a lazy bootstrap handle, not a connected client. Nothing starts until you resolve it: chain `.sync()` for a synchronous client, or `await` the handle from async code. Using the handle directly in a `with` block raises `TypeError: '_BootstrapResult' object does not support the context manager protocol`.

The async version:

```python
import asyncio

from my_env import MyAction, MyEnv


async def main():
    client = await MyEnv.from_docker_image("openenv-my:latest")
    async with client:
        result = await client.reset()
        result = await client.step(MyAction(message="Hello!"))


asyncio.run(main())
```

See [Async vs Sync](../guides/async-sync.md) for when to prefer each style and [Runtime Providers](../guides/runtime-providers.md) to run the image somewhere other than local Docker.

## Build in CI (OpenEnv repo only)

For an environment inside the OpenEnv repo, add it to the matrix in `.github/workflows/docker-build.yml` to build its image on every push to `main`:

```yaml
strategy:
  matrix:
    image:
      - name: echo-env
        dockerfile: envs/echo_env/server/Dockerfile
        context: envs/echo_env
      - name: my-env  # Add your environment here
        dockerfile: envs/my_env/server/Dockerfile
        context: envs/my_env
```
