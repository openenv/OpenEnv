#!/bin/bash
# PreToolUse hook for Bash: Block PR creation if branch is stale
#
# Intercepts `gh pr create` and checks branch freshness against the
# base branch. Unlike git hooks, this cannot be bypassed with --no-verify.

# Read JSON from stdin
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)

# Only check gh pr create commands
if [[ "$COMMAND" != *"gh pr create"* ]]; then
    exit 0
fi

# Check the repo the command runs in: the target of a leading `cd`, else the session's cwd
REPO_DIR=$(echo "$COMMAND" | sed -nE 's/^[[:space:]]*cd[[:space:]]+([^;&|[:space:]]+).*/\1/p')
REPO_DIR="${REPO_DIR/#\~/$HOME}"
if [[ -z "$REPO_DIR" ]]; then
    REPO_DIR=$(echo "$INPUT" | jq -r '.cwd // empty' 2>/dev/null)
fi
REPO_DIR="${REPO_DIR:-.}"

# Determine base branch (default: main) and the PR's head branch (default: the checked-out one).
# sed -E rather than grep -P, which macOS grep does not support.
BASE=$(echo "$COMMAND" | sed -nE 's/.*--base[[:space:]=]+([^[:space:]]+).*/\1/p')
BASE="${BASE:-main}"
HEAD_BRANCH=$(echo "$COMMAND" | sed -nE 's/.*--head[[:space:]=]+([^[:space:]]+).*/\1/p')
HEAD_BRANCH="${HEAD_BRANCH:-$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null)}"

# Fetch latest base and check freshness
git -C "$REPO_DIR" fetch origin "$BASE" --quiet 2>/dev/null || true
BEHIND=$(git -C "$REPO_DIR" rev-list --count "$HEAD_BRANCH".."origin/$BASE" 2>/dev/null || echo "?")

if [[ "$BEHIND" != "0" && "$BEHIND" != "?" ]]; then
    cat >&2 << EOF

===================================================================
  PR BLOCKED: Branch $HEAD_BRANCH is $BEHIND commit(s) behind $BASE
  (checked in $REPO_DIR)
===================================================================

  Your PR will show "out of date with base branch" on GitHub.

  Fix with:
    git fetch origin $BASE
    git rebase origin/$BASE
    git push --force-with-lease

  Then retry gh pr create.

===================================================================

EOF
    exit 2
fi

# Check we're not opening a PR from main/master
if [[ "$HEAD_BRANCH" == "main" || "$HEAD_BRANCH" == "master" ]]; then
    cat >&2 << EOF

===================================================================
  PR BLOCKED: Cannot create PR from $HEAD_BRANCH
===================================================================

  Create a feature branch first:
    git checkout -b <branch-name>
    git push -u origin <branch-name>

===================================================================

EOF
    exit 2
fi

exit 0
