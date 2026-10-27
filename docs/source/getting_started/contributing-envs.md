# Contributing Environments

OpenEnv environments are designed to be shared. The `openenv` CLI provides first-class
commands for publishing, forking, and contributing to environments hosted as
[Hugging Face Spaces](https://huggingface.co/spaces).

Envs are deployed as Hugging Face Spaces, which are Git repositories, Docker images, Python packages and Gradio apps at once.

This guide covers three workflows:

1. **Push** a new environment you built to the Hub.
2. **Fork** someone else's environment to your Hugging Face account to make changes.
3. **Download** an environment, make changes, and open a Pull Request.

## Prerequisites

Before you start, make sure you have:

- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv) installed
- The OpenEnv CLI: `pip install openenv` (or install from source)
- A [Hugging Face account](https://huggingface.co/join) with a [write token](https://huggingface.co/settings/tokens)

Authenticate with the Hub:

```bash
hf auth login
```

The `openenv` CLI will also prompt you to log in automatically if you haven't already.

## 1. Push a New Environment to the Hub

Once you've [built an environment](../guides/first-environment.md), publishing it to a Hugging Face Space is a single command.

```bash
# Push the env at '.' to the hub with config in openenv.yaml
openenv push

# Push the env to a specific repo
openenv push --repo-id my-org/my-custom-env

# Push the env as private
openenv push --private

# Push the env at 'path/to/my_env' to the hub with config in openenv.yaml
openenv push path/to/my_env
```

That's it. The CLI validates your environment, stages the files, adds the Hugging Face Space frontmatter, enables the web interface, and uploads everything. Your environment will be live at
`https://huggingface.co/spaces/<your-username>/my_env`.

> [!WARNING]
> If you are getting errors on deployment, first run
> `openenv validate --level static --skip-build`. This checks the `validation:`
> contract in `openenv.yaml` against the current severity policy. Validate a
> running server separately with `openenv validate --url http://localhost:8000`.

## 2. Fork Someone Else's Environment

Forking creates a copy of a Hugging Face Space under your own account. This is
the fastest way to start experimenting with an existing environment.

```bash
# Fork the openenv/wordle environment to your account
openenv fork owner/space-name
```

This duplicates the Space to `<your-username>/space-name` using the same name and config. To make changes, you can fork to a specific repo name, set environment variables and secrets, and request hardware.

```bash
# Fork to a specific repo name
openenv fork openenv/wordle --repo-id my-username/my-wordle

# Fork the openenv/coding_env environment to your account with environment variables and secrets
openenv fork openenv/coding_env \
  --set-env MODEL_ID=meta-llama/Llama-3-8B \
  --set-secret HF_TOKEN=hf_xxxxxxxxxxxxx

# Fork the openenv/coding_env environment to your account with a GPU
openenv fork openenv/coding_env --hardware t4-medium
```


Once forked, you have a fully independent copy. You can:

- Visit it at `https://huggingface.co/spaces/<your-username>/<space-name>`
- Clone it locally to make changes (see the next section)
- Push updates with `openenv push`

## 3. Pull, Modify, and Open a Pull Request

The contribution workflow lets you improve an existing environment and submit
your changes for review, just like a GitHub Pull Request but on the Hugging
Face Hub.

### 3.1 Download the Space locally

Hugging Face Spaces are Git repositories. Download the one you want to contribute to:

```bash
hf download owner/space-name --local-dir space-name --repo-type space
cd space-name
```

> [!WARNING]
> If the Space is private and you have access, make sure you're logged in with
> `hf auth login` first.

#### 3.2 Make your changes
Edit the environment files as needed.

> [!TIP]
> You can test your changes locally before submitting:
>
> ```bash
> # Run the server locally
> cd space-name
> uvicorn server.app:app --host 0.0.0.0 --port 8000
>
> # Or build and run in Docker
> openenv build
> openenv validate --level static --skip-build
> ```

#### 3.3 Push your changes as a Pull Request

From the cloned directory, use `openenv push` with the `--create-pr` flag:

```bash
openenv push --repo-id owner/space-name --create-pr
```

This uploads your modified files and opens a Pull Request on the Hub. The environment owner can review your changes, leave comments, and merge them.

> [!WARNING]
> When using `--create-pr`, the CLI uploads your changes to a new branch and
> opens a PR on the **original** Space. You do not need to create the Space
> yourself.

### Alternative: Fork-then-PR workflow

If you prefer to develop against your own fork first, you can combine the fork
and PR workflows:

```bash
# 1. Fork the environment to your account
openenv fork owner/space-name --repo-id my-username/space-name

# 2. Download the forked environment to your local directory
hf download my-username/space-name --local-dir space-name --repo-type space
cd space-name

# 3. Make and test your changes
#    ... edit files, run locally, validate ...

# 4. Push the changes back to your fork
openenv push

# 5. Submit a PR to the original Space
openenv push --repo-id owner/space-name --create-pr
```

## End-to-End Example

Here's a complete example: forking the Echo environment, adding a feature, and
submitting a PR.

```bash
# Fork the echo environment
openenv fork openenv/echo_env --repo-id my-username/echo-env-improved

# Clone your fork
git clone https://huggingface.co/spaces/my-username/echo-env-improved
cd echo-env-improved

# Make changes (e.g., add a timestamp to observations)
# ... edit server/echo_environment.py ...

# Test locally
openenv validate --level static --skip-build

# Push your improvement as a PR to the original
openenv push --repo-id openenv/echo_env --create-pr
```

## Next Steps

- [Build your own environment from scratch](../guides/first-environment.md)
- [Customize the web UI](../guides/customizing-web-ui.md)
- [Browse available environments](../environments.md)
- [Hello World](../tutorials/openenv-tutorial.md)
