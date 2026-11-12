"""Which of the agent's files may be handed to a verifier, and which may not.

Verification in a fresh sandbox (#1232) copies the paths a task declares in
`submit_paths` out of the agent's sandbox into `/submission` there. The
submission is data: it is not the working directory and not on `PATH` or
`PYTHONPATH`. A task's own tests normally run against it, though, so a file a
shell or a test runner reads on its way to the first assertion is code rather
than data, and those are left behind.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

SUBMISSION_DIR = "/submission"
WORKSPACE = "/home/user"

# Read before anything a verify command asserts: a shell's login files, Python's
# import-time hooks, and a test runner's configuration and fixtures.
EXCLUDED_NAMES = frozenset(
    {
        # A login shell reads the first of these that it can, and
        # `.bash_logout` when it exits.
        ".bash_profile",
        ".bash_login",
        ".profile",
        ".bashrc",
        ".bash_logout",
        # Python runs these on the way to importing anything.
        "sitecustomize.py",
        "usercustomize.py",
        "conftest.py",
        # Every one of these can carry `addopts = --co`, which collects the
        # tests without running them and reports success.
        "pytest.ini",
        ".pytest.ini",
        "pytest.toml",
        ".pytest.toml",
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
    }
)

# A submission is an answer, not a dataset.
MAX_FILES = 200
MAX_TOTAL_BYTES = 10 * 1024 * 1024


@dataclass
class SubmissionFile:
    """A regular file to copy, as a path relative to the workspace."""

    relative_path: str
    size: int


def check_declared_path(path: str) -> Optional[str]:
    """Refuse a `submit_paths` entry or a path listed under one, or return `None`.

    Args:
        path (`str`):
            The entry as the task declared it, or a path [`parse_listing`] found
            under it.

    Returns:
        `str` or `None`: why the entry was refused, or `None` when it is usable.

    An excluded name is accepted here on purpose: declaring one exactly is the
    task author choosing to submit it, while sweeping a directory that contains
    one is an accident, which [`excluded_file`] refuses.
    """
    if not path:
        return "empty path"
    if path.startswith("/"):
        return f"not relative to {WORKSPACE}: {path!r}"
    if ".." in path.split("/"):
        return f"leaves the workspace: {path!r}"
    return None


def excluded_file(relative_path: str) -> Optional[str]:
    """Refuse a file found under a declared directory, or return `None`."""
    name = relative_path.rsplit("/", 1)[-1]
    if name in EXCLUDED_NAMES or name.endswith(".pth"):
        return f"runs before verification: {relative_path!r}"
    return None


def listing_command(declared_path: str) -> str:
    """Build the command that lists the regular files under a declared path.

    `-printf` is GNU find, which E2B's code-interpreter template has. The type
    letter comes before the size so a symlink, socket, fifo or device is
    reported rather than read: following one copies whatever it points at.
    """
    target = f"{WORKSPACE}/{declared_path}"
    return f"find '{target}' -maxdepth 8 -printf '%y\\t%s\\t%P\\n' 2>/dev/null || true"


def parse_listing(
    declared_path: str, stdout: str
) -> Tuple[List[SubmissionFile], List[str]]:
    """Turn one [`listing_command`] result into files to copy and refusals.

    The listing runs in the agent's sandbox, through a login shell the agent can
    rewrite, so its output is the agent's to choose. Each path in it is checked
    like a declared one before anything is copied to it.

    Args:
        declared_path (`str`):
            The entry being expanded, used to build workspace-relative paths.
        stdout (`str`):
            Output of [`listing_command`].

    Returns:
        `tuple` with:
            - `list[SubmissionFile]`: the regular files to copy.
            - `list[str]`: one reason per refused entry.
    """
    files: List[SubmissionFile] = []
    refusals: List[str] = []
    for line in stdout.splitlines():
        kind, _, rest = line.partition("\t")
        size, _, below = rest.partition("\t")
        if kind == "d":
            continue
        # find prints an empty %P for the path it was given, so an empty one
        # means the declared path is itself the file.
        relative = f"{declared_path}/{below}" if below else declared_path
        # A `..` here would write the copy outside the submission directory.
        reason = check_declared_path(relative)
        if reason:
            refusals.append(reason)
            continue
        if kind != "f":
            refusals.append(f"not a regular file: {relative!r}")
            continue
        reason = excluded_file(relative) if below else None
        if reason:
            refusals.append(reason)
            continue
        files.append(SubmissionFile(relative_path=relative, size=int(size)))
    if not files and not refusals:
        refusals.append(f"nothing to submit at {declared_path!r}")
    return files, refusals


def apply_caps(files: List[SubmissionFile]) -> Tuple[List[SubmissionFile], List[str]]:
    """Keep a submission inside `MAX_FILES` and `MAX_TOTAL_BYTES`."""
    if len(files) > MAX_FILES:
        return [], [f"more than {MAX_FILES} files declared: {len(files)}"]
    total = sum(entry.size for entry in files)
    if total > MAX_TOTAL_BYTES:
        return [], [f"submission is larger than {MAX_TOTAL_BYTES} bytes: {total}"]
    return files, []
