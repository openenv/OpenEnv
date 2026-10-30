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

# Parse only the command line itself: join backslash continuations and keep the first line, since a heredoc
# body follows it. Quoted strings (the --title and --body values) are dropped before reading flags, so text
# inside them cannot match.
CMD_LINE=$(printf '%s\n' "$COMMAND" | sed -e ':a' -e '/\\$/N' -e 's/\\\n//' -e 'ta' | head -n1)

# Check the repo the command runs in: the target of a leading `cd`, else the session's cwd
REPO_DIR=$(printf '%s\n' "$CMD_LINE" | sed -nE "s/^[[:space:]]*cd[[:space:]]+(\"([^\"]+)\"|'([^']+)'|([^;&|[:space:]]+)).*/\2\3\4/p")
REPO_DIR="${REPO_DIR/#\~/$HOME}"
if [[ -z "$REPO_DIR" ]]; then
    REPO_DIR=$(echo "$INPUT" | jq -r '.cwd // empty' 2>/dev/null)
fi
REPO_DIR="${REPO_DIR:-.}"

# Determine base branch (default: main) and the PR's head branch (default: the checked-out one).
# grep -oE rather than grep -P, which macOS grep does not support.
FLAGS=$(printf '%s\n' "$CMD_LINE" | sed -E "s/\"[^\"]*\"//g; s/'[^']*'//g")
BASE=$(printf '%s\n' "$FLAGS" | grep -oE -- '--base[[:space:]=]+[^[:space:]]+' | head -n1 | sed -E 's/^--base[[:space:]=]+//')
BASE="${BASE:-main}"
HEAD_BRANCH=$(printf '%s\n' "$FLAGS" | grep -oE -- '--head[[:space:]=]+[^[:space:]]+' | head -n1 | sed -E 's/^--head[[:space:]=]+//')
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
