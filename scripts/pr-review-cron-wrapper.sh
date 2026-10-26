#!/bin/bash
# Cron wrapper for PR review bot
# Invokes Claude to review open PRs
#
# Crontab entry (every 6 hours):
#   0 */6 * * * /path/to/OpenEnv/scripts/pr-review-cron-wrapper.sh >> ~/.openenv-review-cron.log 2>&1

set -euo pipefail

# Source user's shell profile for PATH
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc" 2>/dev/null || true
elif [ -f "$HOME/.profile" ]; then
    source "$HOME/.profile" 2>/dev/null || true
fi

export PATH="$HOME/.local/bin:$HOME/bin:/usr/local/bin:$PATH"

# Get repo directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# Acquire lock to prevent concurrent runs
LOCK_FILE="/tmp/openenv-pr-review.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Another instance is running. Exiting."
    exit 0
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting PR review..."

# Invoke Claude to handle the reviews
exec claude --print --dangerously-skip-permissions --max-budget-usd 100.00 "
Review ONLY the PRs I was pinged on in the last 6 hours (i.e. where I was
@-mentioned or added as a reviewer). Do NOT review PRs I wasn't pinged on.

1. Run: python3 scripts/pr_tracker.py --list --since 6h --use-state
   This returns JSON for ONLY the open PRs where I have a GitHub notification
   with reason 'mention' or 'review_requested'. Each item has a 'number',
   'repo', and 'reason'. If the list is empty, stop — there is nothing to do.

2. For each PR, spawn a subagent (alignment-reviewer) to review it.
   The subagent should:
   - Read the diff and changed files
   - Apply the two-tier review model (Tier 1: bugs/lint, Tier 2: alignment)
   - Determine verdict: approve, comment, or request_changes

3. After each review, call pr_tracker.post_review(pr_number, verdict, body) to
   post it, then pr_tracker.record_review(pr_number, head_sha, verdict) so the
   same commit is not re-reviewed on the next run.

Run subagent reviews in parallel when possible.
"
