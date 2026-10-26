# SPDX-License-Identifier: BSD-3-Clause

"""`.claude/hooks/lint.sh` must never modify the working tree.

The hook says so itself ("Undo the formatting so the working tree stays
as-is"), and `/alignment-review` and `/pre-submit-pr` run it automatically, so
an author can trigger it without meaning to. It previously formatted in place
and then ran `git checkout --` on whatever changed, which restores from HEAD:
that discarded the formatting *and* any uncommitted edits in the same files,
and left reformatted Markdown behind because the restore only covered `*.py`.

These run the real hook against a throwaway git repo with a stub `uv` on PATH,
so no formatter is actually invoked and the test is hermetic. The stub rewrites
files in "format" mode and reports status in "check" mode, which is enough to
tell a hook that writes from one that only reports.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LINT_HOOK = REPO_ROOT / ".claude" / "hooks" / "lint.sh"

# The stub stands in for `uv`. In check mode it reports whether any target file
# still carries the marker; otherwise it rewrites them, standing in for a
# formatter that modifies files in place.
STUB_UV = """#!/bin/bash
mode=report
for a in "$@"; do
    if [ "$a" = "--check" ] || [ "$a" = "check" ]; then mode=check; fi
done
targets=()
for a in "$@"; do
    if [ -e "$a" ]; then targets+=("$a"); fi
done
[ ${#targets[@]} -eq 0 ] && exit 0

found=1
while IFS= read -r f; do
    if grep -q NEEDSFORMAT "$f" 2>/dev/null; then
        found=0
        if [ "$mode" != "check" ]; then
            sed -i.bak 's/NEEDSFORMAT/WASFORMATTED/g' "$f" && rm -f "$f.bak"
        fi
    fi
done < <(find "${targets[@]}" -type f \\( -name '*.py' -o -name '*.md' \\) 2>/dev/null)

if [ "$mode" = "check" ] && [ $found -eq 0 ]; then exit 1; fi
exit 0
"""

MARKER = "def keep_me():\n    return 'uncommitted work'\n"


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """A git repo whose committed files are already 'unformatted'."""
    if shutil.which("git") is None:  # pragma: no cover
        pytest.skip("git is required")

    # The stub lives outside the repo so it cannot show up as untracked and
    # confuse the "did the hook dirty the tree" assertions.
    work = tmp_path / "repo"
    (work / "src").mkdir(parents=True)
    (work / "tests").mkdir()
    (work / "envs" / "demo_env").mkdir(parents=True)

    # Committed in a state the formatter wants to change, mirroring the 55
    # unformatted files that sit under envs/ on main.
    (work / "src" / "mod.py").write_text("# NEEDSFORMAT\nx = 1\n")
    (work / "envs" / "demo_env" / "README.md").write_text(
        "# Demo\n\n```python\n# NEEDSFORMAT\nx = 1\n```\n"
    )

    run = lambda *a: subprocess.run(  # noqa: E731
        a, cwd=work, check=True, capture_output=True
    )
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "t")
    # Keep the throwaway repo hermetic against developer global config
    # (commit signing / hooksPath), same pattern as tests/discovery/.
    run("git", "config", "commit.gpgsign", "false")
    run("git", "config", "core.hooksPath", "/dev/null")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "init")

    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir()
    stub = bin_dir / "uv"
    stub.write_text(STUB_UV)
    stub.chmod(0o755)
    return work


def _run_hook(repo: pathlib.Path) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": f"{repo.parent / 'stubbin'}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(repo.parent),
    }
    return subprocess.run(
        ["bash", str(LINT_HOOK)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def _dirty(repo: pathlib.Path) -> list[str]:
    out = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line.strip()]


def test_hook_preserves_uncommitted_python_edits(repo: pathlib.Path) -> None:
    """The author's uncommitted work must survive a hook run.

    This is the data-loss case: `git checkout -- $CHANGED` restores from HEAD,
    taking the edits with the formatting.
    """
    target = repo / "src" / "mod.py"
    target.write_text(target.read_text() + "\n" + MARKER)

    _run_hook(repo)

    assert "uncommitted work" in target.read_text(), (
        "lint.sh discarded uncommitted edits in src/mod.py. It formats in place "
        "and then runs `git checkout --` on every changed file, which restores "
        "from HEAD and takes the author's work with the formatting."
    )


def test_hook_leaves_no_files_modified(repo: pathlib.Path) -> None:
    """A run on a clean tree must leave it clean, Markdown included.

    The old restore step filtered to `*.py`, so reformatted Markdown was left
    behind on every run.
    """
    assert _dirty(repo) == []

    _run_hook(repo)

    assert _dirty(repo) == [], (
        "lint.sh modified tracked files. It must report formatting problems "
        "without writing to the working tree."
    )


def test_hook_still_fails_when_formatting_is_needed(repo: pathlib.Path) -> None:
    """Not writing must not mean not reporting: the gate still has to fail."""
    result = _run_hook(repo)

    assert result.returncode != 0
    assert "format" in (result.stdout + result.stderr).lower()


def test_hook_passes_when_everything_is_formatted(repo: pathlib.Path) -> None:
    """And it must still succeed when there is nothing to fix."""
    for path in repo.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md"}:
            path.write_text(path.read_text().replace("NEEDSFORMAT", "ok"))
    subprocess.run(["git", "commit", "-qam", "format"], cwd=repo, check=True)

    result = _run_hook(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _dirty(repo) == []
