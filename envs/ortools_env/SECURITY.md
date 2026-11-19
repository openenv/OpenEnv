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

Each session's code runs as its own uid (60000 to 64999, no account behind it,
given back when the session ends), so sessions can't signal each other or read
each other's files, and a session's working directory belongs to its uid only.
When a call ends, the server kills every process of that uid at once with
`kill(-1, SIGKILL)` from a process whose effective uid is the session's but whose
real and saved uids are one used by nothing else (the session's uid minus 5000),
so neither the session's processes nor other sessions' cleanups can signal it,
and none can start another process in between. That includes children
that called `setsid` to leave the process group. The server then reaps them, and
a uid goes back to the pool only once none of its processes is left. Each uid
can run at most 256 processes and threads (`RLIMIT_NPROC`).

## Network

On Linux (x86_64 and aarch64), each session's code starts under a seccomp filter
that needs no capability and can't be removed from inside: `socket(2)` of any
address family and `io_uring` return `EPERM`, and so do `ptrace(2)`,
`process_vm_readv/writev(2)` and `pidfd_getfd(2)`, which could take a socket or
memory from another process. It also covers hosts where network namespaces are
not available, such as Hugging Face Spaces and Jobs. An `--internal` network
(see the README) is still recommended.

## Unresolved: aggregate resource containment

RLIMIT_AS applies per process, so a session can still use many times its
memory limit, and nothing bounds CPU, disk use or log-file growth per session.
Bounding them requires a delegated cgroup per session, which the current
Dockerfile/hosting contract does not supply. Adding privileges or mounting the
Docker socket into the server would weaken the boundary and is not an acceptable
narrow fix.

`tests/containers/check_ortools_sandbox.py` runs against the production image,
checks the UID/interpreter boundary after removing the hook, exercises privileged
file I/O with synthetic data, and checks that a detached child does not outlive
its call, that sessions get separate uids, that the process count is limited
and that the code can't create sockets.
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
