# SPDX-License-Identifier: BSD-3-Clause

"""The lint hook must match CI without modifying the working tree.

`/alignment-review` and `/pre-submit-pr` run it automatically, so an author
can trigger it without meaning to. It previously formatted in place and then
ran `git checkout --`, discarding uncommitted edits in the same files.

The same holds for the git hooks that `install.sh` writes: they run on every
commit and push, with the author's uncommitted work in the tree.

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
# formatter that modifies files in place. PIPELINE_STABLE models a file that
# usort changes but ruff returns to its original formatting.
STUB_UV = """#!/bin/bash
if [ "$2 $3" = "usort format" ]; then
    [ -d "$4/../.git" ] || exit 1
fi
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
    if [ "$2 $3" = "usort check" ] && grep -q PIPELINE_STABLE "$f"; then
        found=0
    elif [ "$2 $3" = "usort format" ]; then
        sed -i.bak 's/PIPELINE_STABLE/USORT_INTERMEDIATE/g' "$f" && rm -f "$f.bak"
    elif [ "$2 $3" = "ruff format" ] && [ "$mode" != "check" ]; then
        sed -i.bak 's/USORT_INTERMEDIATE/PIPELINE_STABLE/g' "$f" && rm -f "$f.bak"
    elif [ "$2 $3" = "ruff check" ] && grep -q NEEDSLINT "$f"; then
        found=0
    fi
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

    (work / "pyproject.toml").write_text("[tool.ruff]\nline-length = 88\n")
    (work / "src" / "mod.py").write_text("# NEEDSFORMAT\nx = 1\n")
    (work / "envs" / "demo_env" / "mod.py").write_text("# NEEDSFORMAT\nx = 1\n")
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


def _run_hook(
    repo: pathlib.Path,
    hook: pathlib.Path = LINT_HOOK,
    stdin: str | None = None,
    **extra_env: str,
) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": f"{repo.parent / 'stubbin'}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(repo.parent),
    }
    env.update(extra_env)
    return subprocess.run(
        ["bash", str(hook)],
        cwd=repo,
        env=env,
        input=stdin,
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
    before = target.read_bytes()

    _run_hook(repo)

    assert target.read_bytes() == before, (
        "lint.sh discarded uncommitted edits in src/mod.py. It formats in place "
        "and then runs `git checkout --` on every changed file, which restores "
        "from HEAD and takes the author's work with the formatting."
    )


@pytest.mark.parametrize(
    "git_vars", [("GIT_DIR",), ("GIT_WORK_TREE",), ("GIT_DIR", "GIT_WORK_TREE")]
)
def test_hook_isolates_temporary_git_repository(
    repo: pathlib.Path, git_vars: tuple[str, ...]
) -> None:
    """Inherited Git paths must not redirect initialization to the author's repo."""
    (repo / "src" / "mod.py").write_text("x = 1\n\n" + MARKER)
    # Include .git/config and all other metadata alongside uncommitted files.
    before = {
        path.relative_to(repo): path.read_bytes()
        for path in repo.rglob("*")
        if path.is_file()
    }
    git_env = {"GIT_DIR": str(repo / ".git"), "GIT_WORK_TREE": str(repo)}

    result = _run_hook(repo, **{name: git_env[name] for name in git_vars})

    assert result.returncode == 0, result.stdout + result.stderr
    assert {
        path.relative_to(repo): path.read_bytes()
        for path in repo.rglob("*")
        if path.is_file()
    } == before


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


@pytest.mark.parametrize("scope", ["src", "tests"])
def test_hook_still_fails_when_formatting_is_needed(
    repo: pathlib.Path, scope: str
) -> None:
    """Not writing must not mean not reporting: the gate still has to fail."""
    (repo / "src" / "mod.py").write_text("x = 1\n")
    (repo / scope / "mod.py").write_text("# NEEDSFORMAT\nx = 1\n")

    result = _run_hook(repo)

    assert result.returncode != 0
    assert "format" in (result.stdout + result.stderr).lower()
    assert "uv run usort format src/ tests/ && uv run ruff format src/ tests/" in (
        result.stdout
    )
    assert "envs/" not in result.stdout


def test_hook_checks_final_pipeline_output(repo: pathlib.Path) -> None:
    """CI accepts files that usort changes and ruff restores."""
    target = repo / "src" / "mod.py"
    target.write_text("# PIPELINE_STABLE\nx = 1\n")
    before = target.read_bytes()

    result = _run_hook(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert target.read_bytes() == before


def test_hook_ignores_environment_formatting(repo: pathlib.Path) -> None:
    """The lint CI job only formats and lints src/ and tests/."""
    (repo / "src" / "mod.py").write_text("x = 1\n")

    result = _run_hook(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "NEEDSFORMAT" in (repo / "envs" / "demo_env" / "mod.py").read_text()


def test_hook_still_fails_on_lint_errors(repo: pathlib.Path) -> None:
    """Formatting success must not bypass the lint gate."""
    (repo / "src" / "mod.py").write_text("# NEEDSLINT\nx = 1\n")

    result = _run_hook(repo)

    assert result.returncode != 0
    assert "Import sort + format check passed!" in result.stdout
    assert "Lint check passed" not in result.stdout


def test_hook_passes_when_everything_is_formatted(repo: pathlib.Path) -> None:
    """And it must still succeed when there is nothing to fix."""
    for path in repo.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md"}:
            path.write_text(path.read_text().replace("NEEDSFORMAT", "ok"))
    subprocess.run(["git", "commit", "-qam", "format"], cwd=repo, check=True)

    result = _run_hook(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _dirty(repo) == []


def _git_state(repo: pathlib.Path) -> list[str]:
    """Status plus both diffs, so any change to the tree or the index shows."""
    return [
        subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout
        for args in (["status", "--porcelain"], ["diff"], ["diff", "--cached"])
    ]


@pytest.mark.parametrize("hook", ["pre-commit", "pre-push"])
def test_git_hooks_leave_uncommitted_work_alone(repo: pathlib.Path, hook: str) -> None:
    """The installed hooks must leave staged and unstaged work as they found it.

    pre-commit used to format in place and `git add` every changed `.py` file,
    staging edits the author had left unstaged. pre-push formatted in place and
    then ran `git checkout --` on them, discarding the edits.
    """
    git = lambda *a: subprocess.run(  # noqa: E731
        ["git", *a], cwd=repo, check=True, capture_output=True, text=True
    )
    # Both hooks stop on main and master before they get to the files.
    git("checkout", "-qb", "work")
    shutil.copytree(REPO_ROOT / ".claude" / "hooks", repo / ".claude" / "hooks")
    subprocess.run(
        ["bash", ".claude/hooks/install.sh"], cwd=repo, check=True, capture_output=True
    )
    (repo / "src" / "staged.py").write_text("y = 2\n")
    git("add", "src/staged.py")
    target = repo / "src" / "mod.py"
    target.write_text(target.read_text() + "\n" + MARKER)
    before = _git_state(repo)

    head = git("rev-parse", "HEAD").stdout.strip()
    result = _run_hook(
        repo,
        repo / ".git" / "hooks" / hook,
        stdin=f"refs/heads/work {head} refs/heads/work {'0' * 40}\n",
    )

    assert f"Running {hook} checks" in result.stdout, result.stdout + result.stderr
    assert _git_state(repo) == before, (
        f"The {hook} hook changed the working tree or the index. It must leave "
        "formatting to lint.sh, which only reports."
    )
