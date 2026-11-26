#!/bin/bash
# Check for CRLF line endings in text files
# Uses portable constructs that work in sandboxed environments

set -e

# Get the directory to check (default to current directory)
CHECK_DIR="${1:-.}"

# Find all tracked text files with CRLF line endings
CRLF_FILES=()

# Check if we're in a git repository
if git -C "$CHECK_DIR" rev-parse --git-dir > /dev/null 2>&1; then
    # In a git repo - check only tracked files
    # Use a temp file for portability (avoids process substitution issues in sandboxes)
    TEMP_FILE=$(mktemp)
    trap "rm -f '$TEMP_FILE'" EXIT

    (cd "$CHECK_DIR" && git ls-files) > "$TEMP_FILE"

    while IFS= read -r file; do
        # Skip if file doesn't exist
        if [[ ! -f "$file" ]]; then
            continue
        fi

        # Check if file is binary using git
        if git diff --no-index --numstat /dev/null "$file" 2>/dev/null | grep -q "^-"; then
            continue
        fi

        # Check for CRLF line endings
        if grep -qU $'\r' "$file" 2>/dev/null; then
            CRLF_FILES+=("$file")
        fi
    done < "$TEMP_FILE"
else
    # Not a git repo - check all text files
    # Use a temp file for portability
    TEMP_FILE=$(mktemp)
    trap "rm -f '$TEMP_FILE'" EXIT

    find "$CHECK_DIR" -type f -print > "$TEMP_FILE" 2>/dev/null || true

    while IFS= read -r file; do
        # Skip if file doesn't exist or is a directory
        if [[ ! -f "$file" ]]; then
            continue
        fi

        # Simple binary file check - skip files with null bytes
        if grep -qP '\x00' "$file" 2>/dev/null; then
            continue
        fi

        # Check for CRLF line endings
        if grep -qU $'\r' "$file" 2>/dev/null; then
            CRLF_FILES+=("$file")
        fi
    done < "$TEMP_FILE"
fi

# Report results
if [[ ${#CRLF_FILES[@]} -gt 0 ]]; then
    echo "ERROR: Found ${#CRLF_FILES[@]} file(s) with CRLF line endings:" >&2
    for file in "${CRLF_FILES[@]}"; do
        echo "  - $file" >&2
    done
    echo "" >&2
    echo "To fix, convert these files to LF line endings:" >&2
    echo "  dos2unix <file>  # or use your editor's line ending conversion" >&2
    exit 1
fi

exit 0
