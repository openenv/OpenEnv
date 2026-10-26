# PR Review Bot

Automated PR review system using Claude Code. Runs as a cron job to review **only
the PRs the maintainer was pinged on** — i.e. PRs with a GitHub notification whose
reason is `mention` or `review_requested`. It does **not** review every open PR.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Cron (every 6 hours)                     │
│                pr-review-cron-wrapper.sh                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Claude Code                            │
│  1. pr_tracker.py --list --since 6h  (mention/review-req)   │
│  2. Spawns alignment-reviewer subagent for each pinged PR   │
│  3. Posts reviews via pr_tracker.post_review()              │
└─────────────────────────────────────────────────────────────┘
```

**Key insight**: Claude handles orchestration. The only code needed is `pr_tracker.py` for GitHub API access.

## Files

| File | Purpose |
|------|---------|
| `pr_tracker.py` | GitHub API wrapper (PyGithub) - fetches PRs, posts reviews |
| `pr-review-cron-wrapper.sh` | Cron entry point - invokes Claude |

## Quick Start

### 1. Install PyGithub

```bash
pip install PyGithub
```

### 2. Test the Tracker

```bash
# List PRs you were mentioned / review-requested on in the last 6 hours
python3 scripts/pr_tracker.py --list --since 6h

# ...in the last day
python3 scripts/pr_tracker.py --list --since 1d

# Get details for a specific PR
python3 scripts/pr_tracker.py --details 123
```

### 3. Test a Review (manually)

```bash
claude "Review PR #123 using the alignment-reviewer agent.
Use scripts/pr_tracker.py to get PR details and post the review."
```

### 4. Set Up Cron

```bash
crontab -e

# Add this line
0 */6 * * * /home/davidet/OpenEnv/scripts/pr-review-cron-wrapper.sh >> ~/.openenv-review-cron.log 2>&1
```

## pr_tracker.py API

```python
from scripts.pr_tracker import (
    get_prs_needing_review,
    get_pr_details,
    post_review,
    parse_since,
)

# Get PRs you were pinged on (mention / review_requested) in the last 6 hours
since = parse_since("6h")
prs = get_prs_needing_review(since=since)
# Returns: [{"number": 123, "repo": "...", "reason": "mention", ...}]

# Get detailed info about a PR
details = get_pr_details(123)
# Returns: {"number": 123, "files": [...], "body": "...", ...}

# Post a review
post_review(pr_number=123, verdict="approve", body="LGTM!")
```

## Filtering

A PR is reviewed **only** when the authenticated user (the bot's token owner) has
a GitHub notification for it whose reason is one of:

- `mention` — you were @-mentioned in the PR body or a comment
- `review_requested` — you were added as a requested reviewer

This is enforced in `get_prs_needing_review()`, which reads the notifications API
(not the full open-PR list). `--since` only bounds how far back to look for those
notifications; it never widens the set to PRs you weren't pinged on:

- `6h` - pings in the last 6 hours
- `1d` - pings in the last day
- `2024-01-13T00:00:00Z` - pings since a specific timestamp

The cron runs every 6 hours with `--since 6h --use-state`. `--use-state` records
each reviewed head SHA so an unread ping isn't re-reviewed every run.

> **History:** the bot originally filtered by update time alone, so it reviewed
> *every* open PR touched in the window — spamming unrelated PRs. The
> notification-reason filter above replaces that behaviour.

## Review Model

| Issues Found | Verdict |
|--------------|---------|
| Tier 1 issues (bugs, lint, security) | `request_changes` |
| Only Tier 2 flags (alignment concerns) | `comment` |
| No issues | `approve` |

## Logs

```bash
tail -50 ~/.openenv-review-cron.log
```
