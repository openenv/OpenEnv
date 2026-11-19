# Sandbox status and deployment holds

Do not treat this environment as safe for hostile submissions yet.

The production image separates the server and sandbox UIDs and interpreters.
The import hook alone is removable and is not a security boundary. Local runs
without `SANDBOX_USER` do not provide this isolation.

The sandbox passes only PATH, LANG and HOME to submitted code. Execution input
comes from a server-owned descriptor. The persistent instance file is replaced
atomically relative to a pinned, no-follow directory descriptor; existing links
are never opened for writing. Solution reads reject links and nonregular files,
are bounded to 1 MiB, and treat malformed JSON as a failed submission.
If a submission leaves a directory at either reserved filename, subsequent calls
return a controlled setup failure without launching code. The directory is left
untouched until the workdir is reset; setup never traverses or recursively deletes it.

## Sessions and leftover processes

Each session's code runs as its own uid (60000 to 64999, no account behind it, given back when the session ends), so
sessions can't signal each other or read each other's files, and a session's
working directory belongs to its uid only. When a call ends, a helper running as
that uid calls `kill(-1, SIGKILL)`, which stops every process of the uid at once,
including children that called `setsid` to leave the process group. Each uid can
run at most 32 processes (`RLIMIT_NPROC`).

## Unresolved: aggregate resource containment

RLIMIT_AS applies per process, so a session can still use up to 32 times its
memory limit, and nothing bounds CPU, disk use or log-file growth per session.
Bounding them requires a delegated cgroup per session, which the current
Dockerfile/hosting contract does not supply. Adding privileges or mounting the
Docker socket into the server would weaken the boundary and is not an acceptable
narrow fix.

`tests/containers/check_ortools_sandbox.py` runs against the production image,
checks the UID/interpreter boundary after removing the hook, exercises privileged
file I/O with synthetic data, and checks that a detached child does not outlive
its call, that sessions get separate uids and that the process count is limited.
The CI container's PID/memory limits protect the test runner as a whole. They do
not isolate individual submissions.

## CI and release gate

The `test-ortools-env` job installs from the existing environment lockfile,
explicitly imports required dependencies before pytest, and builds/runs the
production container on Linux without network access during checks. Repository
administrators must make this job required in branch protection; adding the job
does not change branch protection. Passing CI does not clear the process/resource
containment limit above. Deployment also needs an explicit network policy and
proof that untrusted code cannot reach server orchestration endpoints.
