#!/bin/bash
# CI polling script. Blocks until all CI checks complete or timeout.
#
# Usage: bash .claude/hooks/ci-wait.sh <PR_NUMBER> [TIMEOUT_SECONDS]
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed
#   2 - Timeout exceeded
#   3 - Error (could not fetch PR)
#
# Polls every 120 seconds. Prints status updates to stdout.

set -e

PR_NUMBER="${1:?Usage: ci-wait.sh <PR_NUMBER> [TIMEOUT_SECONDS]}"
TIMEOUT="${2:-1800}"
POLL_INTERVAL=120
ELAPSED=0

echo ""
echo "==================================================================="
echo "  CI Wait: Monitoring PR #$PR_NUMBER"
echo "==================================================================="
echo "  Timeout: ${TIMEOUT}s | Poll interval: ${POLL_INTERVAL}s"
echo ""

while true; do
    # Fetch current check status
    PR_JSON=$(gh pr view "$PR_NUMBER" --json statusCheckRollup 2>/dev/null || true)
    if [[ -z "$PR_JSON" ]]; then
        echo "ERROR: Could not fetch PR #$PR_NUMBER"
        exit 3
    fi

    CHECK_COUNT=$(echo "$PR_JSON" | jq '.statusCheckRollup | length' 2>/dev/null || echo "0")

    if [[ "$CHECK_COUNT" -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] No CI checks found yet. Waiting..."
    else
        PENDING=$(echo "$PR_JSON" | jq '[.statusCheckRollup[] | select(.status != "COMPLETED")] | length' 2>/dev/null || echo "0")
        FAILED_CHECKS=$(echo "$PR_JSON" | jq '[.statusCheckRollup[] | select(.conclusion == "FAILURE")] | length' 2>/dev/null || echo "0")
        PASSED_CHECKS=$(echo "$PR_JSON" | jq '[.statusCheckRollup[] | select(.conclusion == "SUCCESS")] | length' 2>/dev/null || echo "0")

        echo "[$(date +%H:%M:%S)] Checks: $PASSED_CHECKS passed, $FAILED_CHECKS failed, $PENDING pending (of $CHECK_COUNT)"

        # If no checks are pending, we have a final result
        if [[ "$PENDING" -eq 0 ]]; then
            echo ""
            if [[ "$FAILED_CHECKS" -gt 0 ]]; then
                echo "==================================================================="
                echo "  CI FAILED: $FAILED_CHECKS check(s) failed"
                echo "==================================================================="
                echo ""
                echo "Failed checks:"
                echo "$PR_JSON" | jq -r '.statusCheckRollup[] | select(.conclusion == "FAILURE") | "  - \(.name)"'
                echo ""
                exit 1
            elif [[ "$PASSED_CHECKS" -ne "$CHECK_COUNT" ]]; then
                echo "==================================================================="
                echo "  CI INCOMPLETE: $((CHECK_COUNT - PASSED_CHECKS - FAILED_CHECKS)) check(s) cancelled/skipped"
                echo "==================================================================="
                echo ""
                echo "Non-success checks:"
                echo "$PR_JSON" | jq -r '.statusCheckRollup[] | select(.conclusion != "SUCCESS" and .conclusion != null) | "  - \(.name): \(.conclusion)"'
                echo ""
                exit 1
            else
                echo "==================================================================="
                echo "  CI PASSED: All $PASSED_CHECKS check(s) passed"
                echo "==================================================================="
                echo ""
                exit 0
            fi
        fi
    fi

    # Check timeout
    if [[ "$ELAPSED" -ge "$TIMEOUT" ]]; then
        echo ""
        echo "==================================================================="
        echo "  CI TIMEOUT: Exceeded ${TIMEOUT}s waiting for checks"
        echo "==================================================================="
        echo ""
        if [[ "$CHECK_COUNT" -gt 0 ]]; then
            echo "Pending checks:"
            echo "$PR_JSON" | jq -r '.statusCheckRollup[] | select(.status != "COMPLETED") | "  - \(.name): \(.status)"'
            echo ""
        fi
        exit 2
    fi

    # Sleep and increment
    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done
