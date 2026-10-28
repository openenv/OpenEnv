"""Trusted PID-namespace probe for the Docker-local provider.

The probe runs from the validator's own digest-pinned image and joins only the
subject's PID namespace, so it never executes code from the subject's image. It reads
each subject process from `/proc` and reaches the subject's mount namespace through
`/proc/1/root`, as the subject's own user.
"""

# Measures process identity, the mount table and writable mounts. argv: allotted roots.
PROBE_SCRIPT = """
import json, os, re, sys, uuid
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
print(json.dumps({
    "processes": processes,
    "mounts": mounts,
    "allotted_writable": [a for a in allotted if writable(a)],
    "writable": sorted(p for p in candidates if writable(p)),
}))
"""
