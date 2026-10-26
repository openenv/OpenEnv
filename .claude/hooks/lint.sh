#!/bin/bash
# Lint check for OpenEnv
# Replicates the exact arc f pipeline from fbsource:
#   1. usort format — sort imports (matches arc f's usort pass)
#   2. ruff format  — code formatting, line-length 88 (matches arc f's ruff-api pass)
#   3. ruff check   — lint rules (E, F, W)
#
# usort is scoped to src/ and tests/ only. envs/ uses ruff format only
# because standalone usort and pyfmt's usort disagree on import ordering
# inside try/except blocks in some env files.

set -e

# Check for required tools
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed or not in PATH"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "=== Running import sort + format check ==="
# Ask the same tools as arc f whether the tree is formatted, without writing to
# it. Formatting in place and then running `git checkout --` to undo it also
# reverts the author's uncommitted edits in those files, and leaves reformatted
# Markdown behind, because the restore only ever covered *.py.
FORMAT_FAILED=0
uv run usort check src/ tests/ || FORMAT_FAILED=1
uv run ruff format --check src/ tests/ envs/ || FORMAT_FAILED=1

if [ "$FORMAT_FAILED" -ne 0 ]; then
    echo ""
    echo "ERROR: the files listed above need formatting."
    echo "Run: uv run usort format src/ tests/ && uv run ruff format src/ tests/ envs/"
    exit 1
fi
echo "Import sort + format check passed!"

echo "=== Running lint rules check ==="
uv run ruff check src/ tests/

echo "=== Lint check passed ==="
