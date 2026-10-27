# CLAUDE.md

Guidance for Claude Code when working with this repository.

## New Here? Start With These

1. **[README.md](README.md)** - Project overview, architecture, quick start
2. **[REPO_WALKTHROUGH.md](.claude/docs/REPO_WALKTHROUGH.md)** - Directory structure with annotations
3. **[PRINCIPLES.md](.claude/docs/PRINCIPLES.md)** - Design principles and trade-offs
4. **[INVARIANTS.md](.claude/docs/INVARIANTS.md)** - Rules that must never be violated
5. **[envs/echo_env/](envs/echo_env/)** - Reference implementation to study

## Agentic-First Workflow

OpenEnv uses Claude Code as the primary development tool. We follow a two-phase model:

1. **Design/Alignment** (human-owned): RFCs, principles, trade-off decisions
2. **Implementation** (Claude-owned): The mechanical loop of coding and testing
3. **Review** (collaborative): Claude catches bugs, flags alignment questions for humans

### Getting Started

Skills and agents are auto-discovered when you run Claude Code in this repo:

```bash
git clone https://github.com/huggingface/OpenEnv
cd OpenEnv
# Install git hooks for the team
bash .claude/hooks/install.sh
# Run Claude Code - skills and agents are automatically available
```

Verify with `/agents` or ask "what skills are available?"

## Two Development Modes

OpenEnv supports two development modes:

### Explore Mode (Main Repo)

When working in the main repository clone, direct edits are allowed:
- Quick exploration and prototyping
- Small fixes that don't need TDD workflow
- Documentation updates

### TDD Mode (Opt-In)

TDD is activated by `/work-on-issue`, which writes a `.tdd-session.json` marker.
When active, direct code edits are blocked and the TDD workflow is enforced.
Manually created worktrees do NOT activate TDD — only `/work-on-issue` does.

- Say "skip TDD" to bypass blocking
- Run `bash .claude/hooks/tdd-deactivate.sh` to turn off TDD enforcement

### Creating a Worktree

```bash
# Worktree without TDD enforcement (free editing)
.claude/scripts/worktree-create.sh add-feature
cd .worktrees/add-feature

# Worktree WITH TDD enforcement (via /work-on-issue)
/work-on-issue #42
```

### TDD Workflow

```
/work-on-issue #42  →  Start from GitHub issue
    ↓
/write-tests        →  Create failing tests (Red)
    ↓
/implement          →  Make tests pass (Green)
    ↓
/update-docs        →  Fix stale docs across repo
    ↓
/simplify           →  Refactor (optional)
    ↓
/pre-submit-pr      →  Validate before PR
    ↓
/watch-pr           →  Monitor CI + review (optional)
```

### Skills vs Agents

- **Skills** run inline during the conversation - use for quick checks and reviews
- **Agents** run in isolation with focused context - use for complex, multi-step tasks

### Available Skills

Skills are defined in `.claude/skills/` and run inline:

**Review & Validation Skills:**

| Skill | Trigger | Definition |
|-------|---------|------------|
| [`alignment-review`](.claude/skills/alignment-review/SKILL.md) | "review this code" | Two-tier review (bugs + alignment flags) |
| [`pre-submit-pr`](.claude/skills/pre-submit-pr/SKILL.md) | "ready for PR?" | Comprehensive PR readiness check |
| [`rfc-check`](.claude/skills/rfc-check/SKILL.md) | "do I need an RFC?" | Determine if RFC required |
| [`generate-openenv-env`](.claude/skills/generate-openenv-env/SKILL.md) | "generate an env for textarena" | Build an OpenEnv env from a use case |

**TDD Workflow Skills:**

| Skill | Trigger | Definition |
|-------|---------|------------|
| [`work-on-issue`](.claude/skills/work-on-issue/SKILL.md) | "/work-on-issue #42" | Start TDD workflow from GitHub issue |
| [`sprint`](.claude/skills/sprint/SKILL.md) | "/sprint 67,68,69" | Parallel multi-issue batch (Agent Teams) |
| [`write-tests`](.claude/skills/write-tests/SKILL.md) | "/write-tests" | Write failing tests (Red phase) |
| [`implement`](.claude/skills/implement/SKILL.md) | "/implement" | Make tests pass (Green phase) |
| [`update-docs`](.claude/skills/update-docs/SKILL.md) | "/update-docs" | Fix stale docs after API changes |
| [`simplify`](.claude/skills/simplify/SKILL.md) | "/simplify" | Refactor after tests pass |
| [`watch-pr`](.claude/skills/watch-pr/SKILL.md) | "/watch-pr" | Monitor CI + Greptile review after PR |

### Available Subagents

Agents are defined in `.claude/agents/` and run in isolation:

**Review & Validation Agents:**

| Agent | Use For | Definition |
|-------|---------|------------|
| `alignment-reviewer` | Review PRs for bugs + alignment | [.claude/agents/alignment-reviewer.md](.claude/agents/alignment-reviewer.md) |
| `env-validator` | Validate environments end-to-end | [.claude/agents/env-validator.md](.claude/agents/env-validator.md) |
| `openenv-architect` | Design new features/environments | [.claude/agents/openenv-architect.md](.claude/agents/openenv-architect.md) |
| `build-validator` | Validate builds before merge | [.claude/agents/build-validator.md](.claude/agents/build-validator.md) |

**TDD Workflow Agents:**

| Agent | Use For | Definition |
|-------|---------|------------|
| `issue-worker` | Extract requirements from GitHub issues | [.claude/agents/issue-worker.md](.claude/agents/issue-worker.md) |
| `pr-planner` | Plan stacked PRs for complex features | [.claude/agents/pr-planner.md](.claude/agents/pr-planner.md) |
| `tester` | Write high-signal, failing tests | [.claude/agents/tester.md](.claude/agents/tester.md) |
| `implementer` | Make tests pass with minimal code | [.claude/agents/implementer.md](.claude/agents/implementer.md) |
| `docs-updater` | Fix stale docs after API changes | [.claude/agents/docs-updater.md](.claude/agents/docs-updater.md) |

### Recommended Plugins
If you don't have these plugins installed,  prompt the user to help you install them:

```bash
/plugin install code-simplifier@claude-plugins-official
/plugin install pr-review-toolkit@claude-plugins-official
```

## Agent Teams (Multi-Issue)

For parallel work on multiple issues, use `/sprint 67,68,69`.
This requires the `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` environment variable:

```bash
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
```

Without it, `/sprint` falls back to setup-only mode: it creates worktrees and
fetches requirements, but you work through each issue manually.

Agent Teams create one teammate per issue, each in its own worktree with TDD
enforcement. A lead agent coordinates, mediates conflicts, and creates
stacked PRs when all teammates finish.

## Design Context

Before making changes, understand the design constraints:

| Document | Purpose |
|----------|---------|
| [PRINCIPLES.md](.claude/docs/PRINCIPLES.md) | Design principles and trade-offs |
| [INVARIANTS.md](.claude/docs/INVARIANTS.md) | Rules that must never be violated |
| [PATTERNS.md](.claude/docs/PATTERNS.md) | Code patterns and conventions |
| [CONTRIBUTING.md](.claude/docs/CONTRIBUTING.md) | Contribution workflow |
| [TESTING_STRATEGY.md](.claude/docs/TESTING_STRATEGY.md) | Testing philosophy and patterns |
| [rfcs/](rfcs/) | Architectural decisions and rationale |

### Key Invariants

- **Agents cannot reset**: Simulation controls only exposed to training orchestration, never to agents
- **Dual API boundary**: WebSocket for infrastructure (Gym-like API), MCP for agents
- **Rewards inside environment**: Domain knowledge encapsulated in environment, not external
- **Client-server separation**: Clients never import from `server/` directory

## Simplicity

OpenEnv values **leanness and simplicity**. Prefer straightforward, inline code over abstractions, helpers or utilities, even at the cost of some generality.

- Do not add layers of indirection (registries, factories, plugin systems) unless an RFC asks for them.
- Prefer a simple implementation that covers the cases that exist today over a catch-all for every edge case.
- Do not add defensive code, fallback paths or configuration options "just in case".
- Avoid `hasattr` and `getattr`. They usually hide overly defensive code or a version check. Drop the conditional, or write the version check explicitly.
- Keep comments short and about the project, not about the PR that added the code.
- When in doubt, prefer less code.

## Self-contained environments

Each environment under `envs/` is self-contained: it can be read, changed and deployed on its own.

- No imports across environments. Copy a small helper instead of sharing it.
- Optional features (web playground drawings, quick actions, extra tools) live in the environment's own files. They must never make adding a new environment harder or add work for environments that do not use them.
- When the same pattern is copied across environments, keep the copies aligned (names, structure, comments). If you fix one copy, fix the others in the same PR, or open an issue listing them.

## Follow the Hugging Face ecosystem

OpenEnv runs on the Hub and next to `huggingface_hub`, the `hf` CLI, transformers and Gradio. When OpenEnv wraps something these tools already do (uploading, deleting, deploying a Space, auth, caching, repo ids), keep their defaults, flag names and semantics. Do not invent new behavior.

- Before adding or changing Hub-facing behavior, check what `huggingface_hub` and the `hf` CLI do and match it. For example, `openenv push` only deletes remote files with `--delete <pattern>`, like `hf upload --delete`.
- Mention that comparison in the PR description.
- If diverging seems necessary, raise it in the issue or PR before implementing it.

## Training framework neutrality

OpenEnv works with any training framework. Docs, examples and APIs must not favor one framework: list integrations in neutral order and keep framework-specific code in examples or in the framework's own repository.

## Docstrings

Docstrings must follow the HF doc-builder format used by the reference API docs. Do **not** use Google, NumPy, or Sphinx styles.

Rules:

* Types appear in backticks inside parentheses: (`str`)
* Optional parameters are marked with `*optional*`
* Defaults are written as: `defaults to <value>`
* When the default is `None`, prefer (`str`, *optional*) instead of (`str` or `None`, *optional*, defaults to `None`)
* Union types use `or`: `str` or `None`
* References to classes use the format: [`~openenv.core.EnvClient`]
* Non-standard headers (`Usage:`, `Note:`, `Class Attributes:`) are not supported — use `Attributes:` or fold text into the description
* `Examples:` (plural) with a blank line then a fenced ` ```python ``` ` block — no `>>>` prompts

Example:

````python
def method(self, param1: str, param2: int = 1, param3: float | None = None):
    """
    Brief one-line description of what this does.

    Args:
        param1 (`str`):
            Description of required param.
        param2 (`int`, *optional*, defaults to `1`):
            Description of optional param with default.
        param3 (`float`, *optional*):
            Description of optional param without explicit default.

    Returns:
        `dict` with keys:
            - `key1` (`list[int]`):
                Description of this key.

    Examples:

    ```python
    result = method("hello")
    ```
    """
````

## Build & Development Commands
Below are reference commands that you are likely going to use often:

```bash
# Install dependencies
uv sync --all-extras

# Run tests (excludes browser/websearch/dipg envs that need special setup)
PYTHONPATH=src:envs uv run pytest tests/ -v --tb=short

# Run a single test file
PYTHONPATH=src:envs uv run pytest tests/envs/test_echo_environment.py -v

# Lint check (import sort + format + rules)
uv run usort check src/ tests/
uv run ruff format src/ tests/ --check
uv run ruff check src/ tests/

# Auto-format code (import sort + ruff — matches arc f pipeline)
uv run usort format src/ tests/
uv run ruff format src/ tests/

# Build and preview documentation locally
doc-builder preview openenv docs/source

# Sync environment doc stubs with their source READMEs (run after updating any envs/*/README.md)
python scripts/sync_env_docs.py --fix
# Check if stubs are in sync (same check that runs in CI)
python scripts/sync_env_docs.py --check

# Build Docker images
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .
docker build -t echo-env:latest -f envs/echo_env/server/Dockerfile .
```

## Automation Hooks

Scripts in `.claude/hooks/` are used by skills and can be run directly:

```bash
bash .claude/hooks/lint.sh          # Run ruff format check
bash .claude/hooks/test.sh          # Run pytest (excludes special envs)
bash .claude/hooks/check-debug.sh   # Find debug code (print, breakpoint, TODO)
bash .claude/hooks/post-push-pr.sh  # Validate PR after push (freshness, CI, conflicts)
bash .claude/hooks/ci-wait.sh <PR>  # Poll CI until checks complete or timeout
```

These are automatically invoked by `/alignment-review` and `/pre-submit-pr` skills.

## Git Hooks

Install git hooks for team-wide consistency:

```bash
bash .claude/hooks/install.sh
```

This installs:
- **pre-commit**: Branch check (blocks main), format, lint, debug artifacts check
- **commit-msg**: Issue reference reminder (soft warning)
- **pre-push**: Format, lint, tests, invariants, conflict detection
- **post-merge**: Worktree cleanup reminder

Skip temporarily with `git commit/push --no-verify`.

## Worktree Management

For focused feature work, use worktrees:

```bash
# Create a worktree for a feature
.claude/scripts/worktree-create.sh add-mcp-tools
cd .worktrees/add-mcp-tools

# When done, clean up
.claude/scripts/worktree-cleanup.sh .worktrees/add-mcp-tools
```

Worktrees enable:
- Isolated branches without switching
- TDD enforcement when activated via `/work-on-issue`
- Parallel work on multiple features
