#!/bin/bash
# Match CI: usort format, then ruff format, then ruff check on src/ and tests/.
# Format disposable copies so uncommitted edits are never overwritten.

set -e

# Check for required tools
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed or not in PATH"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "=== Running import sort + format check ==="
# CI checks the combined result: ruff can undo an intermediate usort change.
FORMAT_DIR=$(mktemp -d)
trap 'rm -rf "$FORMAT_DIR"' EXIT
cp -RL src tests pyproject.toml "$FORMAT_DIR/"
if [ -f .gitignore ]; then
    cp .gitignore "$FORMAT_DIR/"
fi
# Ruff only respects .gitignore inside a Git repository.
git -C "$FORMAT_DIR" init -q
uv run usort format "$FORMAT_DIR/src/" "$FORMAT_DIR/tests/"
uv run ruff format "$FORMAT_DIR/src/" "$FORMAT_DIR/tests/"

FORMAT_FAILED=0
while IFS= read -r -d '' file; do
    if ! cmp -s "$file" "$FORMAT_DIR/$file"; then
        echo "$file"
        FORMAT_FAILED=1
    fi
done < <(find -L src tests -type f -name '*.py' -print0)

if [ "$FORMAT_FAILED" -ne 0 ]; then
    echo ""
    echo "ERROR: the files listed above need formatting."
    echo "Run: uv run usort format src/ tests/ && uv run ruff format src/ tests/"
    exit 1
fi
echo "Import sort + format check passed!"

echo "=== Running lint rules check ==="
uv run ruff check src/ tests/

echo "=== Lint check passed ==="
