#!/bin/bash
# Test runner for OpenEnv
# Runs pytest excluding environments that need special setup

set -e

# Check for required tools
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed or not in PATH"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "=== Running tests ==="
# Note: Using timeout to prevent hanging tests from blocking indefinitely (5 min max)
# Matches .github/workflows/test.yml exactly to catch CI failures before push
PYTHONPATH=src:envs timeout 300 uv run pytest tests/ \
    --ignore=tests/envs/test_browsergym_environment.py \
    --ignore=tests/envs/test_dipg_environment.py \
    --ignore=tests/envs/test_websearch_environment.py \
    --ignore=tests/envs/test_python_codeact_reset.py \
    --ignore=tests/envs/test_python_codeact_rewards.py \
    --ignore=tests/envs/test_textarena_environment.py \
    -m "not integration and not network and not docker" \
    -v \
    --tb=short

TEST_EXIT_CODE=$?
if [ $TEST_EXIT_CODE -eq 124 ]; then
    echo "ERROR: Tests timed out after 5 minutes"
    exit 1
elif [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "=== Tests failed ==="
    exit $TEST_EXIT_CODE
fi

echo "=== Tests completed ==="
