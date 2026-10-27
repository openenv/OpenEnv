---
name: simplify
description: Refactor code after tests pass. The "Refactor" phase of Red-Green-Refactor.
context: fork
agent: code-simplifier
---

# /simplify

Refactor and clean up code after tests pass.

## Usage

```
/simplify
/simplify src/openenv/core/client.py
```

## When to Use

- After `/implement` makes tests pass
- When code is correct but could be cleaner
- Before creating a PR (optional polish step)

## When NOT to Use

- Tests are failing (fix tests first)
- You want to add new functionality (use `/write-tests` first)
- Code is already clean and simple

## What It Does

1. Runs tests to ensure they pass (baseline)
2. Identifies opportunities for simplification
3. Refactors while keeping tests green
4. Runs tests after each change to verify nothing broke

## Philosophy

This is TDD's third phase: Red → Green → **Refactor**.

The goal is NOT to add features or change behavior. The goal is to make the code:
- Easier to read
- Easier to maintain
- More consistent with project patterns
- Less code

## Guidelines

### Good Simplifications

- Inline needless helpers and indirection (see Simplicity in `CLAUDE.md`)
- Rename variables for clarity
- Remove dead code
- Simplify complex conditionals
- Use more Pythonic idioms

### NOT Simplifications (Avoid)

- Adding new features
- Changing public APIs
- "Improving" code that works and is readable
- Adding abstractions for hypothetical future needs

## Completion Criteria

1. All tests still pass
2. Code is cleaner/simpler than before
3. No new functionality was added
4. Changes follow project patterns (see PATTERNS.md)

## Integration with TDD Workflow

```
/write-tests  →  create failing tests (Red)
    ↓
/implement    →  make tests pass (Green)
    ↓
/simplify     →  clean up code (Refactor)
    ↓
/pre-submit-pr   →  validate before PR
```
