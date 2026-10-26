#!/bin/bash
# Install git hooks for OpenEnv
#
# Usage: .claude/hooks/install.sh
#
# This installs pre-commit, pre-push, commit-msg, and post-merge hooks.

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
# Use --git-common-dir to get the shared hooks directory (works in worktrees too)
GIT_COMMON_DIR="$(git rev-parse --git-common-dir)"
HOOKS_DIR="$GIT_COMMON_DIR/hooks"

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

echo "Installing git hooks..."

# Pre-commit hook: format, lint, branch check
cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash
# Installed by .claude/hooks/install.sh

echo "Running pre-commit checks..."

REPO_ROOT="$(git rev-parse --show-toplevel)"

# === Branch Check (BLOCKING) ===
echo ""
echo "=== Branch Check ==="
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
    echo "ERROR: Cannot commit directly to $BRANCH"
    echo ""
    echo "Create a worktree first:"
    echo "  $REPO_ROOT/.claude/scripts/worktree-create.sh <name>"
    exit 1
fi
echo "On branch: $BRANCH"

# === Import Sort + Format Check ===
echo ""
echo "=== Import Sort + Format Check ==="
# Run the arc f pipeline: usort then ruff format
uv run usort format src/ tests/ >/dev/null 2>&1
uv run ruff format src/ tests/ >/dev/null 2>&1
CHANGED=$(git diff --name-only -- '*.py' 2>/dev/null || true)
if [ -n "$CHANGED" ]; then
    echo "Files need formatting (usort + ruff format):"
    echo "$CHANGED"
    echo ""
    echo "Auto-formatting and staging changes..."
    git add $CHANGED
    echo "Fixed! Changes staged."
else
    echo "Import sort + format check passed!"
fi

# === Lint Check ===
echo ""
echo "=== Lint Check ==="
"$REPO_ROOT/.claude/hooks/lint.sh" || {
    echo "Lint failed. Fix issues before committing."
    exit 1
}

# === Debug Artifacts (non-blocking) ===
echo ""
echo "=== Debug Artifacts ==="
"$REPO_ROOT/.claude/hooks/check-debug.sh"

echo ""
echo "Pre-commit checks passed"
EOF
chmod +x "$HOOKS_DIR/pre-commit"
echo "  Installed pre-commit hook"

# Commit-msg hook: require issue reference
cat > "$HOOKS_DIR/commit-msg" << 'EOF'
#!/bin/bash
# Installed by .claude/hooks/install.sh
# Require issue reference in commit message

COMMIT_MSG_FILE="$1"
COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")

# Check for issue reference (#123, Fixes #123, Part of #123, etc.)
if echo "$COMMIT_MSG" | grep -qE '#[0-9]+'; then
    exit 0
fi

# Allow WIP commits without issue reference
if echo "$COMMIT_MSG" | grep -qiE '^WIP'; then
    exit 0
fi

echo ""
echo "WARNING: Commit message should reference an issue (#123)"
echo "  Examples: 'Fix bug in parser #45'"
echo "            'Fixes #123'"
echo "            'Part of #99'"
echo ""
echo "Proceeding anyway (this is a soft warning)..."
exit 0
EOF
chmod +x "$HOOKS_DIR/commit-msg"
echo "  Installed commit-msg hook"

# Pre-push hook: comprehensive validation
cat > "$HOOKS_DIR/pre-push" << 'EOF'
#!/bin/bash
# Installed by .claude/hooks/install.sh
# Comprehensive pre-push validation

echo "Running pre-push checks..."

REPO_ROOT="$(git rev-parse --show-toplevel)"
FAILED=0

# 0. BLOCK PUSHES TO MAIN/MASTER (most critical check)
echo ""
echo "=== Protected Branch Check ==="
# Read the remote and refs being pushed from stdin
while read local_ref local_sha remote_ref remote_sha; do
    # Extract branch name from remote ref (refs/heads/main -> main)
    remote_branch="${remote_ref#refs/heads/}"

    if [ "$remote_branch" = "main" ] || [ "$remote_branch" = "master" ]; then
        echo "ERROR: Direct push to '$remote_branch' is blocked!"
        echo ""
        echo "  You are trying to push to a protected branch."
        echo "  Create a PR instead:"
        echo ""
        echo "    # Push to a feature branch"
        echo "    git push -u origin HEAD:feature/your-branch-name"
        echo ""
        echo "    # Then create a PR"
        echo "    gh pr create"
        echo ""
        echo "  To bypass (not recommended): git push --no-verify"
        exit 1
    fi
done
echo "Not pushing to protected branch - OK"

# 1. Import sort + format check
echo ""
echo "=== Import Sort + Format Check ==="
uv run usort format src/ tests/ >/dev/null 2>&1
uv run ruff format src/ tests/ >/dev/null 2>&1
CHANGED_FMT=$(git diff --name-only -- '*.py' 2>/dev/null || true)
if [ -n "$CHANGED_FMT" ]; then
    echo "Files not properly formatted:"
    echo "$CHANGED_FMT"
    echo ""
    echo "Run: uv run usort format src/ tests/ && uv run ruff format src/ tests/"
    git checkout -- $CHANGED_FMT 2>/dev/null || true
    FAILED=1
fi

# 2. Lint check
echo ""
echo "=== Lint Check ==="
"$REPO_ROOT/.claude/hooks/lint.sh" || {
    echo "Lint failed"
    FAILED=1
}

# 3. Test check
echo ""
echo "=== Test Check ==="
"$REPO_ROOT/.claude/hooks/test.sh" || {
    echo "Tests failed"
    FAILED=1
}

# 4. Debug artifacts
echo ""
echo "=== Debug Artifacts ==="
"$REPO_ROOT/.claude/hooks/check-debug.sh"

# 5. Invariant: Client should not import from server
echo ""
echo "=== Invariant Checks ==="
# Check if any client file imports from server directory
# Pattern matches actual imports: "from .server", "from ..server", "import server"
# Excludes comments and string literals mentioning "server"
VIOLATIONS=$(grep -rE "^[[:space:]]*(from [.]+server|import server)" --include="*.py" envs/*/client.py envs/*/__init__.py 2>/dev/null | grep -v "# noqa" || true)
if [ -n "$VIOLATIONS" ]; then
    echo "INVARIANT VIOLATION: Client imports from server"
    echo "$VIOLATIONS"
    echo ""
    echo "   Client code must not import server code. Check INVARIANTS.md."
    echo "   Add '# noqa' comment to suppress if this is intentional (e.g., for local testing)."
    # Note: This is a warning for now due to pre-existing violations
    # TODO: Make this blocking once all violations are fixed (issue #XXX)
    echo "   (Currently warning-only - see pre-existing violations)"
else
    echo "Client-server separation maintained"
fi

# 6. Check branch freshness with main (warning only, non-blocking)
echo ""
echo "=== Branch Freshness Check ==="
# Fetch latest main silently
git fetch origin main --quiet 2>/dev/null || true

# Check how many commits behind main we are
BEHIND_COUNT=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo "0")
if [ "$BEHIND_COUNT" -gt 0 ]; then
    echo "WARNING: Your branch is $BEHIND_COUNT commit(s) behind main!"
    echo ""
    echo "  GitHub will show 'This branch is out-of-date with the base branch'"
    echo ""
    echo "  To update before pushing:"
    echo "    git fetch origin main"
    echo "    git merge origin/main"
    echo "    git push"
    echo ""
    echo "  Pushing anyway (update before merging PR)"
else
    echo "Branch is up to date with main"
fi

# 7. Check for conflicts with main (warning only, non-blocking)
echo ""
echo "=== Conflict Check with main ==="
# Try a test merge to detect conflicts (then abort)
MERGE_OUTPUT=$(git merge --no-commit --no-ff origin/main 2>&1) || true
MERGE_EXIT=$?
git merge --abort 2>/dev/null || true

if echo "$MERGE_OUTPUT" | grep -q "CONFLICT"; then
    echo "WARNING: Your branch has conflicts with main!"
    echo ""
    echo "$MERGE_OUTPUT" | grep "CONFLICT" | head -5
    echo ""
    echo "  To resolve before PR review:"
    echo "    git fetch origin main"
    echo "    git merge origin/main"
    echo "    # resolve conflicts"
    echo "    git push"
    echo ""
    echo "  Pushing anyway (fix conflicts before merging PR)"
else
    echo "No conflicts with main detected"
fi

# Summary
echo ""
if [ $FAILED -eq 1 ]; then
    echo "Pre-push checks FAILED. Fix issues before pushing."
    exit 1
else
    echo "Pre-push checks passed"
fi
EOF
chmod +x "$HOOKS_DIR/pre-push"
echo "  Installed pre-push hook"

# Post-merge hook: remind about worktree cleanup
cat > "$HOOKS_DIR/post-merge" << 'EOF'
#!/bin/bash
# Installed by .claude/hooks/install.sh
# Remind about worktree cleanup after merge

echo ""
echo "=== Post-Merge Reminder ==="

# Check if we're in a worktree
TOPLEVEL=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -f "$TOPLEVEL/.git" ]; then
    echo "You're in a worktree: $TOPLEVEL"
    echo ""
    echo "If this PR is complete, clean up with:"
    echo "  .claude/scripts/worktree-cleanup.sh $TOPLEVEL"
fi
EOF
chmod +x "$HOOKS_DIR/post-merge"
echo "  Installed post-merge hook"

echo ""
echo "Git hooks installed successfully!"
echo ""
echo "Hooks installed:"
echo "  - pre-commit: branch check, usort+format, lint, check-debug"
echo "  - commit-msg: issue reference reminder (soft warning)"
echo "  - pre-push: usort+format, lint, tests, check-debug, invariant checks, conflict detection"
echo "  - post-merge: worktree cleanup reminder"
echo ""
echo "To skip hooks temporarily: git commit/push --no-verify"
