#!/usr/bin/env python3
"""
PR Tracker - Fetches PRs that need review using GitHub API.

This module provides data about open PRs. Claude handles orchestration,
review logic, and posting reviews.

Only PRs where the authenticated user was pinged are listed — i.e. a GitHub
notification with reason ``mention`` or ``review_requested`` — never every open
PR in the repo.

Usage:
    # Get PRs you were mentioned / review-requested on in the last 6 hours
    python3 scripts/pr_tracker.py --list --since 6h

    # Get details for a specific PR
    python3 scripts/pr_tracker.py --details 123
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from github import Github, Auth

# Configuration
DEFAULT_REPO = "huggingface/OpenEnv"
DEFAULT_STATE_FILE = Path.home() / ".openenv-review-state.json"


def _get_github_client() -> Github:
    """Get authenticated GitHub client."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return Github(auth=Auth.Token(token))

    # Try gh CLI token
    try:
        import subprocess

        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            check=True,
        )
        token = result.stdout.strip()
        return Github(auth=Auth.Token(token))
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise RuntimeError(
        "No GitHub token found. Set GITHUB_TOKEN env var or authenticate with 'gh auth login'"
    )


def parse_since(since_str: str) -> datetime:
    """
    Parse a 'since' argument into a datetime.

    Accepts:
    - Duration: "6h", "1d", "30m", "2w"
    - ISO timestamp: "2024-01-13T00:00:00Z"
    """
    # Try duration format first (e.g., "6h", "1d", "30m")
    duration_match = re.match(r"^(\d+)([mhdw])$", since_str.lower())
    if duration_match:
        value = int(duration_match.group(1))
        unit = duration_match.group(2)
        unit_map = {
            "m": timedelta(minutes=value),
            "h": timedelta(hours=value),
            "d": timedelta(days=value),
            "w": timedelta(weeks=value),
        }
        return datetime.now(timezone.utc) - unit_map[unit]

    # Try ISO format
    try:
        # Handle various ISO formats
        if since_str.endswith("Z"):
            since_str = since_str[:-1] + "+00:00"
        return datetime.fromisoformat(since_str)
    except ValueError:
        pass

    raise ValueError(
        f"Invalid 'since' format: {since_str}. "
        "Use duration (6h, 1d, 30m, 2w) or ISO timestamp (2024-01-13T00:00:00Z)"
    )


# Notification reasons that mean "the maintainer was pinged on this PR".
# https://docs.github.com/en/rest/activity/notifications#about-notification-reasons
REVIEW_REASONS = ("mention", "review_requested")


def get_prs_needing_review(
    repo: str = DEFAULT_REPO,
    since: Optional[datetime] = None,
    state_file: Optional[Path] = None,
    reasons: tuple[str, ...] = REVIEW_REASONS,
) -> list[dict]:
    """
    Get open PRs the authenticated user was *pinged* on.

    A PR is returned only when GitHub delivered a notification for it whose
    ``reason`` is in ``reasons`` (by default ``mention`` or ``review_requested``).
    This is the whole point of the bot: review PRs where the maintainer was
    explicitly @-mentioned or added as a reviewer — never every open PR in the
    repo (the previous behaviour, which spammed unrelated PRs).

    Args:
        repo (`str`):
            Repository name (``owner/repo``). Org-transfer redirects are followed.
        since (`datetime`, *optional*):
            Only consider notifications updated after this time.
        state_file (`Path`, *optional*):
            When provided, skip PRs whose current head SHA was already reviewed
            (recorded via :func:`record_review`), so a still-unread ping is not
            re-reviewed on every run.
        reasons (`tuple[str, ...]`, *optional*):
            Notification reasons that trigger a review. Defaults to
            ``("mention", "review_requested")``.

    Returns:
        `list[dict]`: One dict per PR with keys ``number``, ``repo``, ``title``,
        ``author``, ``url``, ``head_sha``, ``updated_at`` and ``reason``.
    """
    gh = _get_github_client()
    repo_obj = gh.get_repo(repo)
    repo_full = repo_obj.full_name  # canonical name (follows org-transfer redirects)

    # Load state for SHA tracking if provided
    repo_state = {}
    if state_file and state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            repo_state = state.get(repo_full, state.get(repo, {}))
        except json.JSONDecodeError:
            pass

    wanted = set(reasons)
    # Unread notifications only (``all=False``) so handled pings drop off; combined
    # with ``since`` this is the set of fresh pings since the last run.
    kwargs = {"all": False}
    if since is not None:
        kwargs["since"] = since
    notifications = gh.get_user().get_notifications(**kwargs)

    prs = []
    seen: set[int] = set()
    for note in notifications:
        # 1. Only act on pings (mention / review_requested) — this is the fix.
        if note.reason not in wanted:
            continue
        subject = note.subject
        if subject is None or subject.type != "PullRequest":
            continue
        # 2. Scope to the target repo (compare canonical full names).
        if (
            note.repository is None
            or note.repository.full_name.lower() != repo_full.lower()
        ):
            continue
        # 3. Subject URL looks like .../repos/<owner>/<repo>/pulls/<number>
        match = re.search(r"/pulls/(\d+)$", subject.url or "")
        if not match:
            continue
        pr_number = int(match.group(1))
        if pr_number in seen:
            continue
        seen.add(pr_number)

        pr = repo_obj.get_pull(pr_number)
        if pr.draft or pr.state != "open":
            continue
        # 4. Skip if this exact commit was already reviewed.
        if state_file:
            if (
                repo_state.get(str(pr_number), {}).get("last_reviewed_sha")
                == pr.head.sha
            ):
                continue

        prs.append(
            {
                "number": pr.number,
                "repo": repo_full,
                "title": pr.title,
                "author": pr.user.login,
                "url": pr.html_url,
                "head_sha": pr.head.sha,
                "updated_at": pr.updated_at.isoformat(),
                "reason": note.reason,
            }
        )

    return prs


def get_pr_details(pr_number: int, repo: str = DEFAULT_REPO) -> dict:
    """Get detailed information about a specific PR."""
    gh = _get_github_client()
    repo_obj = gh.get_repo(repo)
    pr = repo_obj.get_pull(pr_number)

    return {
        "number": pr.number,
        "title": pr.title,
        "author": pr.user.login,
        "url": pr.html_url,
        "head_sha": pr.head.sha,
        "updated_at": pr.updated_at.isoformat(),
        "body": pr.body or "",
        "files": [f.filename for f in pr.get_files()],
        "diff_url": pr.diff_url,
        "additions": pr.additions,
        "deletions": pr.deletions,
        "changed_files": pr.changed_files,
    }


def record_review(
    pr_number: int,
    commit_sha: str,
    verdict: str,
    repo: str = DEFAULT_REPO,
    state_file: Path = DEFAULT_STATE_FILE,
):
    """Record that a PR was reviewed (for SHA-based tracking)."""
    state = {}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except json.JSONDecodeError:
            pass

    if repo not in state:
        state[repo] = {}

    state[repo][str(pr_number)] = {
        "last_reviewed_sha": commit_sha,
        "review_timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
    }

    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2))
    state_file.chmod(0o600)


def post_review(
    pr_number: int,
    verdict: str,
    body: str,
    repo: str = DEFAULT_REPO,
):
    """
    Post a review to a PR.

    Args:
        pr_number: PR number
        verdict: One of "approve", "comment", "request_changes"
        body: Review body (markdown)
    """
    gh = _get_github_client()
    repo_obj = gh.get_repo(repo)
    pr = repo_obj.get_pull(pr_number)

    event_map = {
        "approve": "APPROVE",
        "comment": "COMMENT",
        "request_changes": "REQUEST_CHANGES",
    }
    event = event_map.get(verdict, "COMMENT")

    # Can't approve own PR
    current_user = gh.get_user().login
    if event == "APPROVE" and pr.user.login == current_user:
        pr.create_issue_comment(body)
        return

    formatted_body = f"""> **Note**: This is an automated review by **Claude Code**, not a human review.

---

{body}

---
*Automated review by Claude Code | [Learn more](https://github.com/huggingface/OpenEnv/blob/main/CLAUDE.md)*"""

    pr.create_review(body=formatted_body, event=event)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PR Tracker - Fetch PRs needing review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PRs you were mentioned / review-requested on in the last 6 hours
  python3 pr_tracker.py --list --since 6h

  # ...in the last day
  python3 pr_tracker.py --list --since 1d

  # ...since a specific time
  python3 pr_tracker.py --list --since 2024-01-13T00:00:00Z

  # Get details for PR #123
  python3 pr_tracker.py --details 123
""",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List PRs you were mentioned / review-requested on",
    )
    parser.add_argument(
        "--details", type=int, metavar="PR", help="Get details for specific PR"
    )
    parser.add_argument(
        "--since",
        type=str,
        help="Only pings since (e.g., 6h, 1d, 2024-01-13T00:00:00Z)",
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help="Repository (default: %(default)s)"
    )
    parser.add_argument(
        "--use-state", action="store_true", help="Also filter by SHA state file"
    )
    args = parser.parse_args()

    if args.list:
        since = parse_since(args.since) if args.since else None
        state_file = DEFAULT_STATE_FILE if args.use_state else None
        prs = get_prs_needing_review(repo=args.repo, since=since, state_file=state_file)
        print(json.dumps(prs, indent=2))
    elif args.details:
        details = get_pr_details(args.details, repo=args.repo)
        print(json.dumps(details, indent=2))
    else:
        parser.print_help()
