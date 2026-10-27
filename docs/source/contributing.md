# Contributing to OpenEnv

We welcome contributions from the community! OpenEnv is an open-source project and we're excited to have you join us.

## Ways to Contribute

### 🐛 Report Bugs

Found a bug? Please [open an issue](https://github.com/huggingface/OpenEnv/issues/new) with:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, OS, etc.)

### 💡 Suggest Features

Have an idea? Open a [feature request](https://github.com/huggingface/OpenEnv/issues/new) describing:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

### 🌍 Add an Environment

One of the best ways to contribute is to add a new environment! See the [Building Environments](guides/first-environment.md) guide to get started.

### ☁️ Add a Runtime Provider

To support a new runtime (a cloud sandbox, a scheduler, etc.), subclass
`ContainerProvider` and follow the existing providers (`DaytonaProvider`,
`ACASandboxProvider`) as references. To keep it consistent:

- Put the SDK in a named `[project.optional-dependencies]` extra in `pyproject.toml` and import it lazily, so core OpenEnv stays SDK-free.
- Add an `[[autodoc]]` entry under "Container providers" in `docs/source/reference/core.md`.
- In the [Runtime Providers guide](guides/runtime-providers.md), add a row to the table and a "Per-provider setup" subsection (kept in alphabetical order).
- Add a test (the providers under `tests/test_core/` inject a fake SDK, so no network is needed).

See the Cloud Sandbox Providers amendment proposed in RFC 002 for the
provider-neutral invariants (direct base URL, WebSocket conformance, network posture).

### 📝 Improve Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples
- Write tutorials
- Translate content

### 🔧 Submit Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `PYTHONPATH=src:envs uv run pytest tests/`
5. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/OpenEnv.git
cd OpenEnv

# Install in development mode, with the dev tools
uv sync

# Run tests
PYTHONPATH=src:envs uv run pytest tests/

# Run linting
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

## Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Write docstrings for public functions
- Add type hints
- Write tests for new functionality

## Coordination

OpenEnv is openly governed by a technical committee that coordinates project direction, major technical decisions, RFCs, and release planning in public. The [charter](https://github.com/huggingface/OpenEnv/blob/main/GOVERNANCE.md) explains how it works, and the [README](https://github.com/huggingface/OpenEnv#community-support--acknowledgments) lists the committee members and supporters.

Use the [OpenEnv repository](https://github.com/huggingface/OpenEnv) to file
issues, discuss substantial changes, and submit pull requests.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing! 🙏
