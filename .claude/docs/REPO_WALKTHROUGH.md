# Repository Walkthrough

This document provides a navigational guide to the OpenEnv codebase.

## Top-Level Structure

```
OpenEnv/
├── CLAUDE.md                 # Entry point for Claude Code - build commands, architecture overview
├── README.md                 # Project overview and getting started
├── pyproject.toml            # Python package configuration (uv/pip)
├── uv.lock                   # Locked dependencies
│
├── src/                      # Core library code (installed as `openenv`)
├── envs/                     # Example environments (not installed, used via PYTHONPATH)
├── tests/                    # Test suite
├── examples/                 # Usage examples and tutorials
├── docs/                     # Documentation (Sphinx)
├── rfcs/                     # Design documents and architectural decisions
├── scripts/                  # Utility scripts
│
├── .claude/                  # Claude Code configuration (skills, agents, docs)
├── .github/                  # GitHub Actions, PR templates, issue templates
└── .gitignore
```

## Source Code (`src/`)

```
src/
├── openenv/                  # Main package
│   ├── __init__.py
│   │
│   ├── core/                 # Core abstractions - the heart of OpenEnv
│   │   ├── env_client.py         # EnvClient base class (WebSocket client)
│   │   ├── client_types.py       # Client-side type definitions
│   │   ├── utils.py              # Shared utilities
│   │   │
│   │   ├── harness/              # Agent protocols and captured trace collection
│   │   │   └── capture/              # Dialect adapters, session routing, exact tokens, SSE replay
│   │   │
│   │   ├── env_server/           # Server-side components
│   │   │   ├── interfaces.py         # Environment abstract base class
│   │   │   ├── http_server.py        # HTTPEnvServer (FastAPI + WebSocket)
│   │   │   ├── types.py              # Wire types (Action, Observation, State, WS messages)
│   │   │   ├── serialization.py      # Pydantic serialization helpers
│   │   │   ├── base_transforms.py    # Transform pipeline for rewards/observations
│   │   │   ├── web_interface.py      # Web UI for debugging environments
│   │   │   ├── route_config.py       # FastAPI route configuration
│   │   │   └── exceptions.py         # Server-side exceptions
│   │   │
│   │   ├── containers/           # Container lifecycle management
│   │   │   ├── runtime/              # Provider implementations
│   │   │   │   ├── providers.py           # ContainerProvider/RuntimeProvider ABCs + LocalDockerProvider
│   │   │   │   ├── daytona_provider.py    # DaytonaProvider (Daytona cloud sandboxes)
│   │   │   │   └── uv_provider.py         # UVProvider (for local dev)
│   │   │   └── images/               # Base Docker images
│   │   │       └── Dockerfile            # openenv-base image
│   │   │
│   │   └── tools/                # Reusable tool implementations
│   │       ├── local_python_executor.py  # Python code execution
│   │       └── git_server_client.py      # Git operations
│   │
│   ├── harbor/               # Harbor tasks, harness seams, provider qualification, live UI
│   │
│   ├── discovery/            # RFC 011 metadata-only repository catalogs
│   │   ├── models.py             # Declaration profile and identity invariants
│   │   ├── repository.py         # Bounded reads from one committed Git tree
│   │   ├── producer.py           # Inventory, provenance and record generation
│   │   ├── search.py             # Lexical selection, exact lookup and lifecycle
│   │   ├── serialization.py      # Complete-snapshot loading and atomic writes
│   │   └── schemas/              # Packaged versioned JSON schemas
│   │
│   ├── validation/           # RFC 008 environment validation contracts
│   │   ├── types.py              # Shared validation enums and protocol types
│   │   ├── manifest.py           # Normalized environment manifest models
│   │   ├── signature.py          # Package signature detection
│   │   ├── report.py             # Versioned validation report models
│   │   ├── policy.py             # Severity-policy loading and application
│   │   ├── runner.py             # Local validation orchestration
│   │   ├── runtime/              # Bounded plans, launch requests and raw evidence
│   │   ├── parsers/              # Manifest parser registry
│   │   ├── providers/            # Validation provider contracts
│   │   ├── graders/              # Grader registry and protocols
│   │   ├── policies/             # Versioned severity policies
│   │   └── schemas/              # Generated JSON schemas
│   │
│   └── cli/                  # Command-line interface
│       ├── __main__.py           # Entry point (`python -m openenv.cli`)
│       ├── commands/             # CLI subcommands
│       │   ├── init.py               # `openenv init` - scaffold new env
│       │   ├── serve.py              # `openenv serve` - run server locally
│       │   ├── build.py              # `openenv build` - build Docker image
│       │   ├── catalog.py            # `catalog` and `discover` - metadata only
│       │   ├── push.py               # `openenv push` - deploy to HF Spaces
│       │   └── validate.py           # `openenv validate` - check config
│       └── templates/            # Scaffolding templates
│           └── openenv_env/          # Template for `openenv init`
│
└── openenv_core/             # Legacy compatibility shim (imports from openenv.core)
```

## Environments (`envs/`)

Each environment follows a consistent structure:

```
envs/
├── echo_env/                 # Minimal reference environment
│   ├── client.py                 # EnvClient subclass
│   ├── models.py                 # Action, Observation, State models
│   ├── openenv.yaml              # Environment manifest
│   ├── pyproject.toml            # Environment-specific dependencies
│   ├── README.md
│   └── server/
│       ├── app.py                    # FastAPI app setup
│       ├── echo_environment.py       # Environment implementation
│       └── Dockerfile                # Container definition
│
├── harbor_env/              # Thin Harbor package and trainer session factory
├── thinkingbox_env/           # Stateful MCP business-workflow benchmark adapter
├── coding_env/               # Python code execution environment
├── chat_env/                 # Conversational environment
├── textarena_env/            # Text-based games (TextArena)
├── browsergym_env/           # Browser automation (BrowserGym)
├── openspiel_env/            # Game theory environments (OpenSpiel)
├── atari_env/                # Atari games via Gymnasium
├── finrl_env/                # Financial RL environment
├── git_env/                  # Git operations environment
├── snake_env/                # Classic Snake game
├── sumo_rl_env/              # Traffic simulation (SUMO)
├── connect4_env/             # Connect Four game
├── dipg_safety_env/          # Safety-focused environment
├── reasoning_gym_env/        # Reasoning problems and puzzles
└── websearch_env/            # Web search environment
```

## Tests (`tests/`)

```
tests/
├── conftest.py               # Pytest fixtures
├── test_*.py                 # Core library tests
│
├── envs/                     # Per-environment integration tests
│   ├── test_echo_environment.py
│   ├── test_coding_environment.py
│   ├── test_thinkingbox_env.py  # Focused wire, fidelity, provenance, and evaluator suite
│   └── ...
│
├── test_cli/                 # CLI command tests
└── scripts/                  # Test utility scripts
```

## RFCs (`rfcs/`)

Design documents that capture architectural decisions:

```
rfcs/
├── README.md                 # RFC process and template
├── 000-project-phases.md     # Project vision and phases
├── 001-abstractions.md       # Core abstractions (Environment, Client, two-interface model)
├── 002-env-spec.md           # Environment specification
└── 003-mcp-support.md        # MCP integration design
```

## Claude Code Configuration (`.claude/`)

```
.claude/
├── docs/                     # Alignment documents
│   ├── PRINCIPLES.md             # Design principles and trade-offs
│   ├── INVARIANTS.md             # System invariants (must never violate)
│   ├── PATTERNS.md               # Code patterns and conventions
│   ├── CONTRIBUTING.md           # Agentic contribution workflow
│   └── REPO_WALKTHROUGH.md       # This file
│
├── skills/                   # Auto-discovered skills
│   ├── alignment-review/
│   │   └── SKILL.md              # Two-tier code review
│   ├── implement/
│   │   └── SKILL.md              # Make tests pass (Green phase)
│   ├── pre-submit-pr/
│   │   └── SKILL.md              # PR readiness validation
│   ├── rfc-check/
│   │   └── SKILL.md              # RFC requirement analysis
│   ├── simplify/
│   │   └── SKILL.md              # Refactor after tests pass
│   ├── sprint/
│   │   └── SKILL.md              # Parallel multi-issue batch (Agent Teams)
│   ├── update-docs/
│   │   └── SKILL.md              # Fix stale docs after API changes
│   ├── watch-pr/
│   │   └── SKILL.md              # Monitor CI + Greptile review after PR
│   ├── work-on-issue/
│   │   └── SKILL.md              # Start TDD on a single issue
│   └── write-tests/
│       └── SKILL.md              # Write failing tests (Red phase)
│
├── agents/                   # Specialized subagents
│   ├── alignment-reviewer.md     # Review for bugs + alignment
│   ├── build-validator.md        # Validate builds
│   ├── docs-updater.md           # Fix stale docs after API changes
│   ├── env-validator.md          # Validate environments e2e
│   ├── implementer.md            # Make tests pass with minimal code
│   ├── issue-worker.md           # Extract requirements from GitHub issues
│   ├── openenv-architect.md      # Design new features
│   ├── pr-planner.md             # Plan stacked PRs for complex features
│   └── tester.md                 # Write high-signal, failing tests
│
└── hooks/                    # Automation scripts
    ├── lint.sh                   # Run ruff format check
    ├── test.sh                   # Run pytest
    ├── check-debug.sh            # Find debug code
    ├── post-push-pr.sh           # Validate PR after push (freshness, CI, conflicts)
    ├── tdd-state.sh              # Shared TDD state helpers (is_tdd_active, activate, deactivate)
    ├── tdd-deactivate.sh         # Standalone TDD deactivation script
    ├── install.sh                # Install git hooks (pre-commit, pre-push, etc.)
    ├── session-start.sh          # SessionStart banner (3-state: TDD/worktree/explore)
    ├── no-direct-code.sh         # PreToolUse: block direct edits when TDD active
    ├── pre-commit-check.sh       # PreToolUse: warn on git commit in TDD mode
    ├── pre-pr-check.sh           # PreToolUse: block gh pr create if branch stale
    ├── delegate-todos.sh         # PostToolUse: TDD workflow reminder on TodoWrite
    ├── after-tester.sh           # SubagentStop: next steps after tester
    ├── after-implementer.sh      # SubagentStop: next steps after implementer
    ├── ci-wait.sh                # CI polling: block until checks complete or timeout
    └── after-docs-updater.sh     # SubagentStop: next steps after docs-updater
```

## Documentation (`docs/`)

Sphinx-based documentation:

```
docs/
├── Makefile                  # Sphinx build targets (html, html-noplot, html-stable)
├── README.md                 # Local build instructions
│
└── source/                   # Sphinx source root
    ├── conf.py               # Sphinx configuration
    ├── index.md              # Home page
    ├── core.md               # Core API reference (autodoc)
    ├── cli.md                # CLI reference (autodoc)
    ├── auto_discovery.md     # Auto-discovery API docs
    ├── customizing-web-ui.md # Web UI customization guide
    ├── environments.md       # Environments catalog page
    │
    ├── environments/         # Per-environment documentation
    │   ├── echo.md
    │   ├── coding.md
    │   ├── thinkingbox.md    # Generated from envs/thinkingbox_env/README.md
    │   └── ...
    │
    ├── getting_started/      # Sphinx Gallery executable tutorials
    │   ├── plot_01_introduction_quickstart.py
    │   ├── plot_02_using_environments.py
    │   ├── plot_03_building_environments.py
    │   ├── contributing-envs.md
    │   └── environment-builder.md
    │
    ├── tutorials/            # Additional tutorials
    │   ├── openenv-tutorial.md
    │   ├── wordle-grpo.md
    │   └── rl-training-2048.md
    │
    └── _static/              # Static assets (versions.json, etc.)
```

ThinkingBox examples: `example_usage.py` is a public-client smoke test, while `eval_testlist.py` invokes the packaged evaluator.

## Key Files to Know

| File | Purpose |
|------|---------|
| `src/openenv/core/env_server/interfaces.py` | `Environment` abstract base class |
| `src/openenv/core/env_client.py` | `EnvClient` WebSocket client |
| `src/openenv/core/env_server/http_server.py` | `HTTPEnvServer` FastAPI wrapper |
| `src/openenv/core/env_server/types.py` | All wire types and WebSocket messages |
| `envs/echo_env/` | Reference implementation - start here |
| `rfcs/001-abstractions.md` | Core architectural decisions |
| `.claude/docs/INVARIANTS.md` | Rules that must never be broken |

## Harbor integration

`src/openenv/core/harness/capture/` records inference calls independently of a trainer or tokenizer. `src/openenv/harbor/` runs Harbor tasks, reconciles their trajectories, and exposes the shared training contract through clients and the Gradio playground. `envs/harbor_env/` is the installable environment wrapper. `examples/harbor/nemo_shell_profile/` provides the explicitly qualified NeMo shell workflow.

See `docs/source/guides/harbor-provider-qualification.md` for provider and adapter evidence, `rfcs/012-harbor-capture-providers.md` for explicit evaluation/training and session ownership, and `tests/envs/test_harbor*.py` / `test_capture*.py` for deterministic capture regressions.
