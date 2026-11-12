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

## Unresolved: detached processes and aggregate resource containment

`killpg` only cleans up the initial process group. A submission can start a child
with `start_new_session=True` (or call `setsid`) and continue running after the
submission returns, including after timeout. RLIMIT_AS applies per process; it
does not bound aggregate memory, CPU, process count, disk use or log-file growth.
The sandbox UID is also shared between calls. Do not run mutually untrusted
sessions concurrently under that UID.

Fixing this requires a per-submission Linux PID namespace or a delegated cgroup
with race-free child placement, group-wide kill and resource limits. The current
Dockerfile/hosting contract supplies neither a namespace launcher nor writable,
delegated cgroups. Adding privileges or mounting the Docker socket into the
server would weaken the boundary and is not an acceptable narrow fix. A process
tree scan or process-group change does not solve the `setsid` escape.

`tests/containers/check_ortools_sandbox.py` runs against the production image,
checks the UID/interpreter boundary after removing the hook, exercises privileged
file I/O with synthetic data, and contains an explicitly expected failing test
for the detached-child blocker. That reproduction writes a marker and exits
itself. An expected failure is evidence of an unresolved bug, not containment
proof. The CI container's PID/memory limits protect the test runner as a whole;
they do not isolate individual submissions.

## CI and release gate

The `test-ortools-env` job installs from the existing environment lockfile,
explicitly imports required dependencies before pytest, and builds/runs the
production container on Linux without network access during checks. Repository
administrators must make this job required in branch protection; adding the job
does not change branch protection. Passing CI does not clear the process/resource
containment hold above. Deployment also needs an explicit network policy and
proof that untrusted code cannot reach server orchestration endpoints.
