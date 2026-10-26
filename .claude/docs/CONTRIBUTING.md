# Contributing with Claude Code

OpenEnv is an agentic-first project. We expect most contributions to use Claude Code or similar tools. This document describes the workflow.

## The Two-Phase Model

### Phase 1: Design & Alignment (Human-Owned)

Humans own the "what" and "why":
- Major architectural decisions require RFCs
- Discuss trade-offs in issues before implementation
- Establish acceptance criteria and invariants
- Review for alignment, not just correctness

### Phase 2: Implementation (Claude-Owned)

Claude handles the mechanical loop:
```
while not working:
    try_some_shit()
    test()
```

Humans intervene only for alignment questions.

## TDD Workflow

OpenEnv uses Test-Driven Development (TDD) enforced through Claude Code hooks.

### Quick Start

```bash
# Start working on an issue with TDD enforcement
/work-on-issue #42

# Or create a plain worktree (no TDD — free editing)
.claude/scripts/worktree-create.sh my-feature
cd .worktrees/my-feature
```

### The Red-Green-Refactor Cycle

1. **Red**: `/write-tests` - Create failing tests that encode requirements
2. **Green**: `/implement` - Write minimal code to make tests pass
3. **Docs**: `/update-docs` - Fix stale references across the repo
4. **Refactor**: `/simplify` - Clean up without changing behavior
5. **Validate**: `/pre-submit-pr` - Ensure everything passes before PR

### When to Use TDD Mode

TDD is opt-in — it is activated only by `/work-on-issue`, not by being in a worktree.

**Use TDD (`/work-on-issue`) for:**
- New features with clear acceptance criteria
- Bug fixes where you can write a failing test first
- Refactoring where tests ensure nothing breaks

**Skip TDD (stay in main repo or use a plain worktree) for:**
- Quick exploration and prototyping
- Documentation updates
- Simple config changes
- Discussing approaches before implementing

### Multi-Issue Work

For parallel work on a batch of issues:
```bash
/sprint 67,68,69
```
This uses Agent Teams (if enabled) to work on all issues in parallel,
each in its own worktree with TDD enforcement, then creates stacked PRs.
Without Agent Teams, it prepares worktrees and requirements for manual work.

### Bypassing TDD

When TDD is active, say "skip TDD" in your message to bypass the edit blocking.
This is useful for:
- Fixing typos in code you just wrote
- Making quick adjustments during iteration
- Emergency hotfixes

To deactivate TDD entirely: `bash .claude/hooks/tdd-deactivate.sh`

## When to Write an RFC

**Required for:**
- New core APIs in `src/openenv/core/`
- Breaking changes to existing APIs
- Major architectural decisions
- New abstractions or design patterns
- Changes affecting the two-interface model (WebSocket/MCP)

**Not required for:**
- Bug fixes, documentation, minor refactoring
- New example environments (unless introducing new patterns)
- Dependency updates, test additions

See `rfcs/README.md` for the RFC process.

## Review Expectations

### What Claude Catches (Tier 1)
- Bugs, uninitialized variables, type errors
- Lint failures, test failures
- Security issues (credential exposure, injection)
- Debug code left in (print statements, breakpoints)

### What Humans Review (Tier 2)
- Does this align with our principles in PRINCIPLES.md?
- Does this maintain our invariants in INVARIANTS.md?
- Is this the right trade-off for the project?
- Should this decision be documented in an RFC?

### Alignment Flags

When Claude identifies a potential alignment issue, it formats as:
```
**ALIGNMENT FLAG**: [Brief description]
- **Principle at stake**: [Which principle]
- **The concern**: [What seems misaligned]
- **Suggested reviewer**: @[maintainer]
```

## Adding or Updating an Environment

When adding a new environment or updating an existing `envs/<name>/README.md`:

1. Write or update `envs/<name>/README.md` — this is the source of truth for the environment's docs page.
2. Run the sync script to create or refresh the stub in `docs/source/environments/`:

```bash
python scripts/sync_env_docs.py --fix
```

3. For a **new** environment, also add manually:
   - An HTML card in `docs/source/environments.md`
   - A `local: environments/<slug>` entry in `docs/source/_toctree.yml`

4. Commit the updated stubs along with your other changes.

CI (`check-env-docs` job in `test.yml`) will fail if a README was changed without running `--fix`. To check locally before pushing:

```bash
python scripts/sync_env_docs.py --check
```

## Available Tools

For the full list of available skills, subagents, and recommended plugins, see [CLAUDE.md](../../CLAUDE.md#available-skills).
