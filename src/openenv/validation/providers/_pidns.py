"""Trusted PID-namespace probe for the Docker-local provider.

The probe runs from the validator's own digest-pinned image and joins only the
subject's PID namespace, so it never executes code from the subject's image. It reads
each subject process from `/proc` and reaches the subject's mount namespace through
`/proc/1/root`, as the subject's own user.
"""

# Measures process identity, the mount table and writable mounts. argv: allotted roots.
PROBE_SCRIPT = """
import errno, json, os, re, stat, sys, uuid
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
root = "/proc/1/root"
def size(path):
    try:
        info = os.statvfs(root + path)
    except OSError:
        return None
    return info.f_blocks * info.f_frsize or None
try:
    mounts = []
    with open("/proc/1/mountinfo") as f:
        for line in f:
            fields = line.split()
            separator = fields.index("-")
            mount_point = re.sub(r"\\\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4])
            mounts.append({
                "mount_point": mount_point,
                "fstype": fields[separator + 1],
                "device": fields[2],
                # Writes need both the mount and its filesystem to be read-write.
                "read_only": "ro" in fields[5].split(",") + fields[separator + 3].split(","),
                "size_bytes": size(mount_point),
            })
except (OSError, ValueError, IndexError):
    # A partial table could omit a writable filesystem.
    mounts = []
def writable(path):
    target = os.path.join(root + path, ".openenv-containment-" + uuid.uuid4().hex)
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except OSError:
        return False
    os.close(fd)
    os.unlink(target)
    return True
# Every writable filesystem is a mount, so try each one outside the allotted roots.
candidates = {
    m["mount_point"] for m in mounts if not m["read_only"] and m["mount_point"] not in allotted
}
writable_paths = {p for p in candidates if writable(p)}
# Only /dev's otherwise exempt tmpfs needs a subtree inventory. Separate mounts
# (including allotted /dev/shm), symlinks and device nodes are never traversed.
devices = {m["device"] for m in mounts if (m["mount_point"], m["fstype"]) == ("/dev", "tmpfs")}
device_storage = {d: {"complete": True, "writable": False} for d in devices}
boundaries = {m["mount_point"] for m in mounts}
remaining = 1024
def scan(path, device, parent_fd, depth=0):
    global remaining
    reading = device_storage[device]
    if depth and path in boundaries:
        return
    remaining -= 1
    if remaining < 0 or depth > 32:
        reading["complete"] = False
        return
    target = path.rsplit("/", 1)[-1]
    try:
        info = os.stat(target, dir_fd=parent_fd, follow_symlinks=False)
        directory = stat.S_ISDIR(info.st_mode)
        if depth == 0 and stat.S_ISLNK(info.st_mode):
            reading["complete"] = False
            return
        if not directory and not stat.S_ISREG(info.st_mode):
            return
        if directory and any(path == p or path.startswith(p + "/") for p in ("/proc", "/sys")):
            reading["complete"] = False
            return
        if f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}" != device:
            reading["complete"] = False
            return
        # An owner can restore write permission without changing the launch.
        flags = os.O_RDONLY | os.O_DIRECTORY if directory else os.O_WRONLY | os.O_APPEND | os.O_NONBLOCK
        try:
            fd = os.open(target, flags | os.O_NOFOLLOW, dir_fd=parent_fd)
        except OSError as error:
            if (directory or error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS)
                or (info.st_uid == os.geteuid() and error.errno in (errno.EACCES, errno.EPERM))):
                reading["complete"] = False
            return
        try:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                reading["complete"] = False
                return
            if directory:
                name = ".openenv-containment-" + uuid.uuid4().hex
                try:
                    probe = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
                except OSError as error:
                    if (error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS)
                        or (opened.st_uid == os.geteuid() and error.errno in (errno.EACCES, errno.EPERM))):
                        reading["complete"] = False
                else:
                    reading["writable"] = True
                    writable_paths.add(path)
                    os.close(probe)
                    os.unlink(name, dir_fd=fd)
                with os.scandir(fd) as entries:
                    for entry in entries:
                        if remaining <= 0:
                            reading["complete"] = False
                            break
                        scan(path + "/" + entry.name, device, fd, depth + 1)
            else:
                # Opening without truncation proves access without changing data.
                reading["writable"] = True
                writable_paths.add(path)
        finally:
            os.close(fd)
    except OSError:
        reading["complete"] = False
for mount in mounts:
    if mount["device"] in devices and not mount["read_only"] and mount["mount_point"] not in allotted:
        parent_fd = None
        try:
            parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            for part in mount["mount_point"].strip("/").split("/")[:-1]:
                child_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
                os.close(parent_fd)
                parent_fd = child_fd
            scan(mount["mount_point"], mount["device"], parent_fd)
        except OSError:
            device_storage[mount["device"]]["complete"] = False
        finally:
            if parent_fd is not None:
                os.close(parent_fd)
print(json.dumps({
    "processes": processes,
    "mounts": mounts,
    "allotted_writable": [a for a in allotted if writable(a)],
    "writable": sorted(p for p in writable_paths if len(p) <= 1024)[:8],
    "device_storage": device_storage,
}))
"""

# Read limits independently: an unavailable file must not erase known violations.
LIMITS_SCRIPT = """
import json
root = "/proc/1/root/sys/fs/cgroup/"
cgroup = {}
unavailable = []
for filename, keys in [
    ("memory.max", ["memory_max"]),
    ("memory.swap.max", ["swap_max"]),
    ("cpu.max", ["cpu_quota", "cpu_period"]),
    ("pids.max", ["pids_max"]),
]:
    try:
        with open(root + filename) as f:
            values = f.read().split()
        if len(values) != len(keys):
            raise ValueError()
        parsed = [None if value == "max" else int(value) for value in values]
        if any(value is not None and value < 0 for value in parsed):
            raise ValueError()
        if "cpu_period" in keys and (parsed[1] is None or parsed[1] == 0):
            raise ValueError()
    except (OSError, ValueError):
        parsed = [None] * len(keys)
        unavailable.extend(keys)
    cgroup.update(zip(keys, parsed))
cgroup["unavailable"] = unavailable
print(json.dumps({"cgroup": cgroup}))
"""

# Trusted pre/post stress evidence from the subject's cgroup and filesystem.
COUNTERS_SCRIPT = """
import json, os, sys
root = "/proc/1/root"
result = {}
for filename, key, counter in [
    ("memory.events", "oom_kill", "oom_kill"),
    ("pids.events", "pids_max", "max"),
]:
    try:
        with open(root + "/sys/fs/cgroup/" + filename) as f:
            result[key] = int(dict(line.split() for line in f)[counter])
    except (OSError, ValueError, KeyError):
        result[key] = None
result["free_bytes"] = {}
for path in sys.argv[1:]:
    try:
        info = os.statvfs(root + path)
        result["free_bytes"][path] = info.f_bavail * info.f_frsize
    except OSError:
        result["free_bytes"][path] = None
print(json.dumps(result))
"""

# Subject-image code is only a stimulus. Trusted counters corroborate its result.
QUALIFY_SCRIPT = """
import errno, os, sys, time
kind, budget, nonce = sys.argv[1], int(sys.argv[2]), sys.argv[3]
print("openenv-probe-start:" + nonce, flush=True)
mib = 1 << 20
if kind == "memory":
    with open("/proc/self/oom_score_adj", "w") as f:
        f.write("1000")
    held = []
    while len(held) * 16 * mib < budget + 64 * mib:
        held.append(b"\\x01" * (16 * mib))
    print("escaped", flush=True)
elif kind == "pids":
    children = []
    try:
        for _ in range(budget + 16):
            pid = os.fork()
            if pid == 0:
                time.sleep(30)
                os._exit(0)
            children.append(pid)
        print("escaped", flush=True)
    except BlockingIOError:
        print("contained", flush=True)
    finally:
        for pid in children:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
elif kind == "disk":
    written = 0
    for index, path in enumerate(sys.argv[4:]):
        try:
            with open(os.path.join(path, f"openenv-fill-{index}"), "wb") as f:
                while written <= budget + 16 * mib:
                    f.write(b"\\x01" * mib)
                    f.flush()
                    written += mib
        except OSError as exc:
            if exc.errno not in (errno.ENOSPC, errno.EDQUOT):
                print("inconclusive", flush=True)
                sys.exit()
    # Leave the files for trusted statvfs verification; this subject is disposable.
    print("escaped" if written > budget else "contained", flush=True)
elif kind == "exec_timeout":
    if os.fork() == 0:
        time.sleep(300)
        os._exit(0)
    print("openenv-probe-descendant:" + nonce, flush=True)
    time.sleep(300)
"""
