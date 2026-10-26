# Contributing to OpenEnv

OpenEnv is an **agentic-first project** designed for Claude Code contributions.

## Quick Links

- **Contribution workflow**: See [.claude/docs/CONTRIBUTING.md](.claude/docs/CONTRIBUTING.md) for the agentic workflow, RFC process, and review expectations
- **Design principles**: See [.claude/docs/PRINCIPLES.md](.claude/docs/PRINCIPLES.md)
- **System invariants**: See [.claude/docs/INVARIANTS.md](.claude/docs/INVARIANTS.md)
- **Claude Code guidance**: See [CLAUDE.md](CLAUDE.md)

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes: `PYTHONPATH=src:envs uv run pytest tests/ -v`
5. Make sure your code lints: `uv run ruff format src/ tests/ --check`
6. For significant changes, write an RFC first (see `rfcs/README.md`)

## Issues

We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to reproduce the issue.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe disclosure of security bugs. In those cases, please go through the process outlined on that page and do not file a public issue.

## License

By contributing to OpenEnv, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.
