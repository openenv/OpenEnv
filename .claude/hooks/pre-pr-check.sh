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

# Parse the command as one line, with backslash continuations joined. Quoted strings (the --title and --body
# values, including a heredoc inside "$(...)") and a bare heredoc are dropped before reading flags, so text inside
# them cannot match. --base/--head values are unquoted first so they survive.
ONE_LINE=$(printf '%s\n' "$COMMAND" | sed -e ':a' -e '/\\$/N' -e 's/\\\n//' -e 'ta' | tr '\n' ' ')
FLAGS=$(printf '%s\n' "$ONE_LINE" | sed -E "s/((--base|-B|--head|-H)[[:space:]=]+)\"([^\"]*)\"/\1\3/g; s/((--base|-B|--head|-H)[[:space:]=]+)'([^']*)'/\1\3/g; s/\"[^\"]*\"//g; s/'[^']*'//g; s/<<.*//")

# Only a command that runs gh pr create, not one that mentions it (a commit message, a grep)
RUNS_PR_CREATE='(^|[;&|(])[[:space:]]*gh[[:space:]]+pr[[:space:]]+create'
if ! [[ "$FLAGS" =~ $RUNS_PR_CREATE ]]; then
    exit 0
fi

# Check the repo the command runs in: the target of the last `cd` before gh pr create, else the session's cwd.
# A relative target is resolved against the session's cwd.
CWD=$(echo "$INPUT" | jq -r '.cwd // empty' 2>/dev/null)
BEFORE="; ${ONE_LINE%%gh pr create*}"
REPO_DIR=$(printf '%s\n' "$BEFORE" | sed -nE "s/.*[;&|][[:space:]]*cd[[:space:]]+(\"([^\"]+)\"|'([^']+)'|([^;&|[:space:]]+)).*/\2\3\4/p")
REPO_DIR="${REPO_DIR/#\~/$HOME}"
if [[ -n "$REPO_DIR" && "$REPO_DIR" != /* && -n "$CWD" ]]; then
    REPO_DIR="$CWD/$REPO_DIR"
fi
REPO_DIR="${REPO_DIR:-${CWD:-.}}"

# Determine base branch (default: main) and the PR's head branch (default: the checked-out one). gh uses the last
# value of a repeated flag, and --head may be owner:branch. grep -oE rather than grep -P, which macOS grep lacks.
BASE=$(printf '%s\n' "$FLAGS" | grep -oE -- '(^|[[:space:]])(--base|-B)[[:space:]=]+[^[:space:];&|)]+' | tail -n1 | sed -E 's/^[[:space:]]*(--base|-B)[[:space:]=]+//')
BASE="${BASE:-main}"
HEAD_BRANCH=$(printf '%s\n' "$FLAGS" | grep -oE -- '(^|[[:space:]])(--head|-H)[[:space:]=]+[^[:space:];&|)]+' | tail -n1 | sed -E 's/^[[:space:]]*(--head|-H)[[:space:]=]+//')
HEAD_BRANCH="${HEAD_BRANCH#*:}"
HEAD_BRANCH="${HEAD_BRANCH:-$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null)}"

# A value that looks like an option would be passed to git as one, so skip the check instead
if [[ "$BASE" == -* || "$HEAD_BRANCH" == -* ]]; then
    exit 0
fi

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
