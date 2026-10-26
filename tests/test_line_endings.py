# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for consistent line endings across the repository.

These tests ensure:
1. All tracked text files use LF line endings (no CRLF)
2. .gitattributes exists with proper LF normalization rules
3. A check script exists to detect CRLF files
"""

import subprocess
from pathlib import Path

import pytest


def get_repo_root() -> Path:
    """Get the repository root directory."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def get_tracked_files() -> list[Path]:
    """Get all git-tracked files."""
    repo_root = get_repo_root()
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
    )
    return [repo_root / f for f in result.stdout.strip().split("\n") if f]


def is_binary_file(file_path: Path) -> bool:
    """Check if a file is binary (not text)."""
    # Use git's built-in detection
    result = subprocess.run(
        ["git", "diff", "--no-index", "--numstat", "/dev/null", str(file_path)],
        capture_output=True,
        text=True,
        cwd=file_path.parent,
    )
    # Binary files show "-" for additions/deletions
    return result.stdout.startswith("-\t-\t")


def has_crlf_line_endings(file_path: Path) -> bool:
    """Check if a file contains CRLF line endings."""
    try:
        content = file_path.read_bytes()
        return b"\r\n" in content
    except (OSError, IOError):
        return False


class TestLineEndings:
    """Tests for consistent LF line endings."""

    def test_no_crlf_in_tracked_files(self):
        """All tracked text files should use LF line endings, not CRLF."""
        tracked_files = get_tracked_files()
        crlf_files = []

        for file_path in tracked_files:
            if not file_path.exists():
                continue
            if is_binary_file(file_path):
                continue
            if has_crlf_line_endings(file_path):
                crlf_files.append(file_path)

        assert not crlf_files, (
            f"Found {len(crlf_files)} files with CRLF line endings. "
            f"These should be converted to LF:\n"
            + "\n".join(f"  - {f}" for f in crlf_files)
        )


class TestGitAttributes:
    """Tests for .gitattributes configuration."""

    def test_gitattributes_exists(self):
        """Repository should have a .gitattributes file."""
        repo_root = get_repo_root()
        gitattributes = repo_root / ".gitattributes"
        assert gitattributes.exists(), (
            ".gitattributes file not found at repository root. "
            "This file is needed to enforce consistent line endings."
        )

    def test_gitattributes_has_lf_normalization(self):
        """The .gitattributes file should configure LF normalization."""
        repo_root = get_repo_root()
        gitattributes = repo_root / ".gitattributes"

        if not gitattributes.exists():
            pytest.skip(".gitattributes does not exist")

        content = gitattributes.read_text()

        # Check for text normalization
        assert "text=auto" in content or "* text" in content, (
            ".gitattributes should configure text file normalization. "
            "Expected to find 'text=auto' or '* text' rule."
        )

        # Check for LF line ending configuration
        assert "eol=lf" in content, (
            ".gitattributes should enforce LF line endings. "
            "Expected to find 'eol=lf' configuration."
        )


class TestLineEndingCheckScript:
    """Tests for the line ending check script."""

    def test_check_script_exists(self):
        """The check-line-endings.sh script should exist."""
        repo_root = get_repo_root()
        script_path = repo_root / ".claude" / "hooks" / "check-line-endings.sh"
        assert script_path.exists(), (
            f"Line ending check script not found at {script_path}. "
            "This script is needed to detect CRLF files in hooks and CI."
        )

    def test_check_script_is_executable(self):
        """The check-line-endings.sh script should be executable."""
        repo_root = get_repo_root()
        script_path = repo_root / ".claude" / "hooks" / "check-line-endings.sh"

        if not script_path.exists():
            pytest.skip("Script does not exist")

        # Check if file has execute permission
        import os
        import stat

        mode = os.stat(script_path).st_mode
        assert mode & stat.S_IXUSR, (
            f"Script {script_path} is not executable. "
            "Run: chmod +x .claude/hooks/check-line-endings.sh"
        )

    def test_check_script_detects_crlf(self, tmp_path):
        """The check script should detect files with CRLF line endings."""
        repo_root = get_repo_root()
        script_path = repo_root / ".claude" / "hooks" / "check-line-endings.sh"

        if not script_path.exists():
            pytest.skip("Script does not exist")

        # Create a test file with CRLF endings
        test_file = tmp_path / "test_crlf.txt"
        test_file.write_bytes(b"line1\r\nline2\r\n")

        # Run the script on the temp directory
        result = subprocess.run(
            ["bash", str(script_path), str(tmp_path)],
            capture_output=True,
            text=True,
        )

        # Script should return non-zero when CRLF is found
        assert result.returncode != 0, (
            "check-line-endings.sh should return non-zero exit code "
            "when CRLF files are found"
        )
        assert "test_crlf.txt" in result.stdout or "test_crlf.txt" in result.stderr, (
            "check-line-endings.sh should report the file with CRLF endings"
        )

    def test_check_script_passes_with_lf(self, tmp_path):
        """The check script should pass when all files have LF endings."""
        repo_root = get_repo_root()
        script_path = repo_root / ".claude" / "hooks" / "check-line-endings.sh"

        if not script_path.exists():
            pytest.skip("Script does not exist")

        # Create a test file with LF endings
        test_file = tmp_path / "test_lf.txt"
        test_file.write_bytes(b"line1\nline2\n")

        # Run the script on the temp directory
        result = subprocess.run(
            ["bash", str(script_path), str(tmp_path)],
            capture_output=True,
            text=True,
        )

        # Script should return zero when no CRLF is found
        assert result.returncode == 0, (
            "check-line-endings.sh should return zero exit code "
            f"when all files have LF endings. Got: {result.stderr}"
        )
