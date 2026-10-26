---
name: watch-pr
description: Monitor a PR's CI checks and Greptile code review after submission. Polls CI status, auto-fixes failures via ralph-loop, waits for Greptile review, addresses comments, and iterates until green.
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, Skill
---

# /watch-pr

Monitor a submitted PR until CI passes and code reviews are addressed.

## EXECUTE THESE STEPS NOW

When this skill is invoked, you MUST execute these steps immediately. Do NOT just describe what will happen — actually do it.

### Step 0: Resolve PR Number and Repo

Extract the PR number from `$ARGUMENTS`. If no argument was provided, detect from the current branch:

```bash
gh pr view --json number -q '.number'
```

If no PR is found, stop with: "No PR found for current branch. Create one with `gh pr create` or pass a PR number: `/watch-pr 123`"

Also resolve the repo identifier:

```bash
gh repo view --json nameWithOwner -q '.nameWithOwner'
```

Store as `PR_NUMBER` and `REPO`. Initialize counters:
- `CI_FIX_COUNT = 0` (max 5)
- `REVIEW_FIX_COUNT = 0` (max 3)

Report to the user:

```
## Watching PR #<PR_NUMBER>
Monitoring CI and reviews for https://github.com/<REPO>/pull/<PR_NUMBER>
```

---

### Step 1: WAITING_CI — Poll CI Checks

Run the CI polling script with a 30-minute timeout:

```bash
bash .claude/hooks/ci-wait.sh <PR_NUMBER> 1800
```

**Important**: Set the Bash tool timeout to 600000ms (10 minutes). If the script exceeds
this, re-invoke it with the remaining timeout: `bash .claude/hooks/ci-wait.sh <PR_NUMBER> <REMAINING_SECONDS>`.

Evaluate the exit code:
- **Exit 0** (all checks passed): Go to **Step 3 (WAITING_REVIEW)**.
- **Exit 1** (checks failed): Go to **Step 2 (CI_FAILED)**.
- **Exit 2** (timeout): Report to user: "CI checks did not complete within 30 minutes. Check manually." Stop.
- **Exit 3** (error): Report error and stop.

---

### Step 2: CI_FAILED — Fix and Retry

Increment `CI_FIX_COUNT`. If `CI_FIX_COUNT > 5`, stop with:

```
CI has failed 5 times. Manual intervention required.
PR: https://github.com/<REPO>/pull/<PR_NUMBER>
```

**2a. Identify failed checks and get logs:**

```bash
# Get the head SHA for this PR
HEAD_SHA=$(gh pr view <PR_NUMBER> --json headRefOid -q '.headRefOid')

# List failed workflow runs for this commit
gh run list --commit "$HEAD_SHA" --json databaseId,name,conclusion --jq '.[] | select(.conclusion == "failure")'
```

For each failed run, fetch the failure logs:

```bash
gh run view <RUN_ID> --log-failed 2>&1 | tail -200
```

**2b. Try ralph-loop first:**

Invoke the ralph-loop plugin to fix the failures:

```
Skill tool: skill: "ralph-loop"
```

If ralph-loop successfully fixes the issues (tests pass locally), commit and push
the changes, then go to **Step 1**.

**2c. Fallback to inline fix (if ralph-loop unavailable or fails):**

1. Read the failure logs from step 2a
2. Identify the root cause (test failure, lint error, build error, etc.)
3. Read the relevant source files
4. Fix the code
5. Verify locally:
   ```bash
   bash .claude/hooks/lint.sh && bash .claude/hooks/test.sh
   ```
6. Stage, commit, and push:
   ```bash
   git add <specific-files>
   git commit -m "fix: address CI failure (<brief description>)"
   git push
   ```

**2d. Return to Step 1 (WAITING_CI).**

After pushing the fix, CI will re-run. Go back to Step 1.

---

### Step 3: WAITING_REVIEW — Poll for Greptile Review

All CI checks have passed. Now wait for Greptile's code review.

Poll every 120 seconds for up to 3 hours (max 90 polls). On each poll:

```bash
# Check for reviews from greptile-apps[bot]
REVIEW_COUNT=$(gh api repos/<REPO>/pulls/<PR_NUMBER>/reviews \
  --jq '[.[] | select(.user.login == "greptile-apps[bot]")] | length')
```

```bash
# Check for line-level comments from greptile-apps[bot]
COMMENT_COUNT=$(gh api repos/<REPO>/pulls/<PR_NUMBER>/comments \
  --jq '[.[] | select(.user.login == "greptile-apps[bot]")] | length')
```

**Decision logic:**

- If both counts are 0: print "Waiting for Greptile review... (poll N/90)", sleep 120 seconds, and poll again.
- If 3-hour timeout reached with no review: go to **Step 5 (DONE)** with note "Greptile review did not arrive within 3 hours."
- If review exists, check if actionable:
  - **Actionable** (go to Step 4): The latest review state is `CHANGES_REQUESTED`, OR there are line-level comments from `greptile-apps[bot]`.
  - **Not actionable** (go to Step 5): Review state is `APPROVED` or `COMMENTED` with no line-level comments.

To check review state:

```bash
gh api repos/<REPO>/pulls/<PR_NUMBER>/reviews \
  --jq '[.[] | select(.user.login == "greptile-apps[bot]")] | last | .state'
```

---

### Step 4: REVIEW_RECEIVED — Address Greptile Comments

Increment `REVIEW_FIX_COUNT`. If `REVIEW_FIX_COUNT > 3`, stop with:

```
Greptile has requested changes 3 times. Manual intervention required.
PR: https://github.com/<REPO>/pull/<PR_NUMBER>
```

**4a. Fetch the review body:**

```bash
gh api repos/<REPO>/pulls/<PR_NUMBER>/reviews \
  --jq '[.[] | select(.user.login == "greptile-apps[bot]")] | last | .body'
```

**4b. Fetch all line-level comments:**

```bash
gh api repos/<REPO>/pulls/<PR_NUMBER>/comments \
  --jq '[.[] | select(.user.login == "greptile-apps[bot]")] | .[] | {id: .id, path: .path, line: .line, body: .body}'
```

**4c. Address each comment:**

For each line-level comment:
1. Read the file at the specified path and line
2. Understand the suggestion
3. If it aligns with project principles: apply the fix
4. If it conflicts with `.claude/docs/PRINCIPLES.md` or `.claude/docs/INVARIANTS.md`: do NOT apply it. Reply explaining why:
   ```bash
   gh api repos/<REPO>/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
     -f body="Not applied: <reason based on project principles>"
   ```
5. For applied fixes, reply to acknowledge:
   ```bash
   gh api repos/<REPO>/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
     -f body="Fixed in latest push."
   ```

**4d. Human approval checkpoint:**

Before committing review fixes, present a summary to the user for approval:

```
## Greptile Review Changes Summary

| # | Comment | Action | File |
|---|---------|--------|------|
| 1 | <brief description> | Applied / Declined | <path> |
| 2 | ... | ... | ... |

Approve these changes before pushing? (y/n)
```

Use the AskUserQuestion tool to get explicit approval. If the user declines,
stop and let them handle the review manually.

**4e. Verify, commit, and push:**

After user approval:

```bash
bash .claude/hooks/lint.sh && bash .claude/hooks/test.sh
```

If local checks pass:

```bash
git add <specific-files>
git commit -m "fix: address Greptile review comments"
git push
```

**4f. Return to Step 1 (WAITING_CI).**

CI will re-run after the push. Go back to Step 1.

---

### Step 5: DONE — Final Report

```
## Watch PR Complete

### PR
https://github.com/<REPO>/pull/<PR_NUMBER>

### CI Status: PASSED
- CI fix iterations: <CI_FIX_COUNT>

### Greptile Review
| Status | Details |
|--------|---------|
| Received | YES / NO (timed out) |
| Actionable comments | N |
| Comments addressed | N |
| Comments declined | N (with reasons) |
| Review fix iterations | <REVIEW_FIX_COUNT> |

### Final Status: READY FOR HUMAN REVIEW / NEEDS ATTENTION
```

If final status is READY (CI green, reviews addressed), report:

```
PR is ready for human review.
```

If final status is NEEDS ATTENTION (hit iteration limits), explain what remains.

---

## Safeguards

| Limit | Value | Behavior when exceeded |
|-------|-------|----------------------|
| CI fix iterations | 5 | Stop, report failures, ask user |
| Greptile wait timeout | 3 hours | Continue without review |
| Review fix iterations | 3 | Stop, report outstanding comments |

## When to Use

- After `/pre-submit-pr` creates and pushes a PR
- After pushing fixes to an existing PR
- When you want automated CI monitoring and review handling

## When NOT to Use

- Before a PR exists (run `/pre-submit-pr` first)
- For draft PRs that aren't ready for review
- When you want manual control over CI fixes

## Workflow Integration

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
/watch-pr           →  Monitor CI + Greptile review  ← THIS SKILL
```
