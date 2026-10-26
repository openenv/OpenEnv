"""Trusted PID-namespace probe for the Docker-local provider.

The probe runs from the validator's own digest-pinned image and joins only the
subject's PID namespace, so it never executes code from the subject's image. It reads
each subject process from `/proc` and reaches the subject's mount namespace through
`/proc/1/root`, as the subject's own user.
"""

# Measures process identity, mounts and writable paths. argv: allotted roots.
PROBE_SCRIPT = """
import json, os, sys, uuid
me = os.getpid()
allotted = sys.argv[1:]
processes = []
for entry in sorted(os.listdir("/proc"), key=lambda e: (len(e), e)):
    if not entry.isdigit() or int(entry) == me:
        continue
    try:
        with open(f"/proc/{entry}/status") as f:
            status = dict(l.rstrip("\\n").split(":", 1) for l in f if ":" in l)
    except OSError:
        continue
    processes.append({
        "pid": int(entry),
        "name": status["Name"].strip()[:64],
        "uids": [int(u) for u in status["Uid"].split()],
        "cap_eff": int(status["CapEff"], 16),
        "no_new_privs": status.get("NoNewPrivs", "0").strip() == "1",
        "seccomp": int(status.get("Seccomp", "0")),
    })
mounts = []
try:
    with open("/proc/1/mountinfo") as f:
        for line in f:
            fields = line.split()
            separator = fields.index("-")
            mounts.append({
                "mount_point": fields[4].replace("\\\\040", " "),
                "fstype": fields[separator + 1],
                "read_only": "ro" in fields[5].split(","),
            })
except (OSError, ValueError, IndexError):
    pass
root = "/proc/1/root"
def writable(path):
    target = os.path.join(root + path, ".openenv-containment-" + uuid.uuid4().hex)
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except OSError:
        return False
    os.close(fd)
    os.unlink(target)
    return True
candidates = {"/"}
try:
    for name in os.listdir(root):
        path = "/" + name
        if path not in ("/proc", "/sys") and os.path.isdir(root + path) and not os.path.islink(root + path):
            candidates.add(path)
except OSError:
    pass
for process in processes:
    try:
        candidates.add(os.readlink(f"/proc/{process['pid']}/cwd"))
    except OSError:
        pass
inside = lambda path: any(path == a or path.startswith(a + "/") for a in allotted)
print(json.dumps({
    "processes": processes,
    "mounts": mounts,
    "allotted_writable": [a for a in allotted if writable(a)],
    "writable": sorted(p for p in candidates if not inside(p) and writable(p)),
}))
"""

# Reads the subject's cgroup v2 limits and writable-root sizes. argv: writable roots.
LIMITS_SCRIPT = """
import json, os, sys
root = "/proc/1/root"
def read(name):
    with open(root + "/sys/fs/cgroup/" + name) as f:
        return f.read().split()
def limit(value):
    return None if value == "max" else int(value)
try:
    quota, period = read("cpu.max")
    cgroup = {
        "memory_max": limit(read("memory.max")[0]),
        "swap_max": limit(read("memory.swap.max")[0]),
        "cpu_quota": limit(quota),
        "cpu_period": int(period),
        "pids_max": limit(read("pids.max")[0]),
    }
except (OSError, ValueError):
    cgroup = None
writable = []
for path in sys.argv[1:]:
    try:
        info = os.statvfs(root + path)
    except OSError:
        continue
    writable.append({"path": path, "size_bytes": info.f_blocks * info.f_frsize})
print(json.dumps({"cgroup": cgroup, "writable": writable}))
"""

# Runs inside a disposable subject. argv: kind, budget. Prints the outcome; a
# kill by the kernel (exit 137) also means the limit held.
QUALIFY_SCRIPT = """
import errno, os, sys, time
kind, budget = sys.argv[1], int(sys.argv[2])
mib = 1 << 20
if kind == "memory":
    # Make the probe the OOM killer's first choice so the server survives.
    with open("/proc/self/oom_score_adj", "w") as f:
        f.write("1000")
    held = []
    try:
        while len(held) * 16 * mib < budget + 64 * mib:
            held.append(b"\\x01" * (16 * mib))
        print("escaped")
    except MemoryError:
        print("contained")
elif kind == "pids":
    children = []
    try:
        for _ in range(budget + 16):
            pid = os.fork()
            if pid == 0:
                time.sleep(30)
                os._exit(0)
            children.append(pid)
        print("escaped")
    except BlockingIOError:
        print("contained")
    finally:
        for pid in children:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
elif kind == "disk":
    # Fill every writable root; together they must stay within the budget.
    written = 0
    for index, path in enumerate(sys.argv[3:]):
        try:
            with open(os.path.join(path, f"openenv-fill-{index}"), "wb") as f:
                while written <= budget + 16 * mib:
                    f.write(b"\\x01" * mib)
                    f.flush()
                    written += mib
        except OSError as exc:
            if exc.errno not in (errno.ENOSPC, errno.EDQUOT):
                print("inconclusive")
                sys.exit()
    print("escaped" if written > budget else "contained")
"""
