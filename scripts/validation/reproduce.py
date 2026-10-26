#!/usr/bin/env python3
"""One pinned entry point for local and CI runtime-validation evidence."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "tests/validation_runtime"
FIXTURE = ROOT / "tests/fixtures/validation/runtime/served_probe"
ECHO_OVERLAY = ROOT / "tests/fixtures/validation/runtime/echo_canary"
MAX_LOG_BYTES = 1024 * 1024


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_digest(argv, *, cwd=ROOT, timeout=600):
    with tempfile.TemporaryFile() as output:
        subprocess.run(
            [str(arg) for arg in argv],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=True,
        )
        output.seek(0)
        return hashlib.file_digest(output, "sha256").hexdigest()


def hashes(root):
    return {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(argv, *, cwd=ROOT, timeout=600, log=None, env=None):
    process = subprocess.Popen(
        [str(arg) for arg in argv],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    stdout_tail = bytearray()
    stderr_tail = bytearray()

    def drain(stream, tail):
        try:
            while chunk := stream.read(8192):
                tail.extend(chunk)
                if len(tail) > MAX_LOG_BYTES:
                    del tail[:-MAX_LOG_BYTES]
        finally:
            stream.close()

    readers = [
        threading.Thread(target=drain, args=(stream, tail), daemon=True)
        for stream, tail in (
            (process.stdout, stdout_tail),
            (process.stderr, stderr_tail),
        )
    ]
    for reader in readers:
        reader.start()
    try:
        process.wait(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            # The process group already exited before cleanup could signal it.
            pass
        process.wait()
        raise
    finally:
        for reader in readers:
            reader.join(timeout=5)
        if any(reader.is_alive() for reader in readers):
            # A descendant may retain a pipe after the main command exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                # Descendants may exit while the output reader is finishing.
                pass
            for reader in readers:
                reader.join(timeout=5)
        output = bytes(stdout_tail).decode("utf-8", "replace")
        errors = bytes(stderr_tail).decode("utf-8", "replace")
        output = re.sub(
            r"(?:hf_|ghp_|github_pat_|sk-)[A-Za-z0-9_-]{8,}", "[REDACTED]", output
        )
        errors = re.sub(
            r"(?:hf_|ghp_|github_pat_|sk-)[A-Za-z0-9_-]{8,}", "[REDACTED]", errors
        )
        diagnostics = (
            (output + errors).encode("utf-8")[-MAX_LOG_BYTES:].decode("utf-8", "ignore")
        )
        if log:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(diagnostics)
    if process.returncode:
        raise RuntimeError(
            f"Command failed ({process.returncode}): {argv[0:3]}\n{diagnostics[-4000:]}"
        )
    return output.strip()


def snapshot_source(destination):
    """Hash exactly the files used for the wheel, including local source edits."""
    if any(path.is_symlink() for path in (ROOT / "src").rglob("*")):
        raise RuntimeError("Source snapshots do not permit symlinks")
    destination.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copyfile(ROOT / name, destination / name)
    shutil.copytree(
        ROOT / "src",
        destination / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("Source snapshots do not permit symlinks")
    return hashes(destination)


def select_linux_requirements(path, arch):
    """Evaluate lock markers for the target, not pip's macOS host environment."""
    from packaging.markers import default_environment
    from packaging.requirements import Requirement

    target = default_environment()
    target.update(
        {
            "sys_platform": "linux",
            "platform_system": "Linux",
            "os_name": "posix",
            "platform_machine": arch,
            "python_version": "3.12",
            "python_full_version": "3.12.13",
        }
    )
    blocks = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            blocks.append([line])
        else:
            blocks[-1].append(line)
    selected = []
    for block in blocks:
        requirement = Requirement(block[0].rstrip(" \\"))
        if requirement.marker is not None and not requirement.marker.evaluate(target):
            continue
        requirement.marker = None
        selected.append(str(requirement) + " \\\n" + "\n".join(block[1:]))
    path.write_text("\n".join(selected) + "\n")


def download_linux_wheels(
    requirements, wheelhouse, base_image, docker_platform, *, log
):
    # Linux compatibility depends on the target image's libc, not the host or
    # a fixed shortlist of manylinux versions. The image digest pins this pip.
    # Pull separately so cold-cache progress cannot contaminate the JSON probe.
    run(
        ["docker", "pull", "--platform", docker_platform, base_image],
        timeout=300,
        log=log.parent / "base-image-pull.log",
    )
    platforms = json.loads(
        run(
            [
                "docker",
                "run",
                "--rm",
                "--pull=never",
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--user=65532:65532",
                "--platform",
                docker_platform,
                base_image,
                "python",
                "-c",
                "import json; from pip._vendor.packaging.tags import platform_tags; "
                "print(json.dumps(list(platform_tags())))",
            ],
            timeout=120,
        )
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--require-hashes",
            "--only-binary=:all:",
            "--python-version",
            "3.12",
            "--implementation",
            "cp",
            *[arg for tag in platforms for arg in ("--platform", tag)],
            "-r",
            requirements,
            "--dest",
            wheelhouse,
        ],
        timeout=900,
        log=log,
    )
    return platforms


def stage_image(work, output, pins, manifest):
    source = work / "source"
    manifest["wheel_source_hashes"] = snapshot_source(source)
    wheel_dir = output / "wheel"
    try:
        run(
            [
                "uv",
                "build",
                "--wheel",
                "--no-build-isolation",
                "--python",
                sys.executable,
                "--out-dir",
                wheel_dir,
                source,
            ],
            log=output / "logs/wheel-build.log",
        )
    finally:
        # uv's output-directory marker is not evidence and artifact upload omits
        # hidden files. Remove it before checksumming, including failed builds.
        (wheel_dir / ".gitignore").unlink(missing_ok=True)
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected one exact-source OpenEnv wheel")
    wheel = wheels[0]
    manifest["wheel_sha256"] = digest(wheel)
    # Replace only OpenEnv in this dedicated, non-editable test environment.
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            wheel,
        ],
        log=output / "logs/wheel-install.log",
    )
    requirements = output / "requirements.txt"
    run(
        [
            "uv",
            "export",
            "--project",
            PROJECT,
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--no-emit-package",
            "openenv",
            "--output-file",
            requirements,
        ],
        log=output / "logs/dependency-export.log",
    )
    context = work / "subject"
    context.mkdir()
    wheelhouse = output / "wheelhouse"
    architecture = run(["docker", "info", "--format", "{{.Architecture}}"])
    arch = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }.get(architecture)
    if arch is None:
        raise RuntimeError(f"Unsupported Docker architecture: {architecture}")
    manifest["docker_platform"] = "linux/amd64" if arch == "x86_64" else "linux/arm64"
    select_linux_requirements(requirements, arch)
    manifest["wheel_platform_tags"] = download_linux_wheels(
        requirements,
        wheelhouse,
        pins["base_image"],
        manifest["docker_platform"],
        log=output / "logs/dependency-acquisition.log",
    )
    shutil.copyfile(wheel, wheelhouse / wheel.name)
    manifest["wheelhouse_hashes"] = hashes(wheelhouse)
    shutil.copytree(wheelhouse, context / "wheelhouse")
    # The exact wheel hash joins the locked third-party hashes for offline installation.
    with requirements.open("a") as stream:
        stream.write(
            f"\nopenenv=={wheel.name.split('-')[1]} --hash=sha256:{digest(wheel)}\n"
        )
    shutil.copyfile(requirements, context / "requirements.txt")
    shutil.copytree(
        FIXTURE, context / "served_probe", ignore=shutil.ignore_patterns("__pycache__")
    )
    for name in ("Dockerfile", "openenv.yaml"):
        shutil.copyfile(FIXTURE / name, context / name)
    shutil.copytree(FIXTURE / "validation", context / "validation")
    # Assert the checked-in Dockerfile and declared base pin cannot drift independently.
    if pins["base_image"] not in (context / "Dockerfile").read_text():
        raise RuntimeError("Fixture base image differs from toolchain.json")
    from openenv.validation.manifest import ExecutionDeclaration
    from openenv.validation.providers.docker import DockerValidationProvider

    provider = DockerValidationProvider(build_timeout_s=600)
    image = provider.build(
        context, ExecutionDeclaration(kind="openenv_ws", agent_boundary="api")
    )
    manifest["image_ref"] = image
    manifest["build_context_hashes"] = hashes(context)
    return context, image


def stage_echo_canary(work, context, output, manifest):
    """Stage unchanged reference-environment sources without importing them."""
    import yaml

    source = ROOT / "envs/echo_env"
    if any(path.is_symlink() for path in source.rglob("*")):
        raise RuntimeError("Echo source snapshots do not permit symlinks")
    target = work / "echo-subject"
    target.mkdir()
    shutil.copytree(
        source,
        target / "echo_env",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", "*.egg-info"),
    )
    copied = hashes(target / "echo_env")
    if any(digest(source / name) != value for name, value in copied.items()):
        raise RuntimeError("Echo source changed while the canary was staged")
    # The wheelhouse and pinned, offline installation are shared with the probe.
    shutil.copytree(context / "wheelhouse", target / "wheelhouse")
    shutil.copyfile(context / "requirements.txt", target / "requirements.txt")
    dockerfile = (context / "Dockerfile").read_text()
    substitutions = {
        "COPY served_probe /app/served_probe": "COPY echo_env /app/echo_env",
        '["python", "-m", "served_probe.app"]': '["python", "-m", "echo_env.server.app"]',
    }
    for before, after in substitutions.items():
        if dockerfile.count(before) != 1:
            raise RuntimeError("Shared image recipe changed; review the Echo canary")
        dockerfile = dockerfile.replace(before, after)
    (target / "Dockerfile").write_text(dockerfile)
    declaration = yaml.safe_load((source / "openenv.yaml").read_text())
    declaration["validation"]["execution"] = json.loads(
        (ECHO_OVERLAY / "execution.json").read_text()
    )
    (target / "openenv.yaml").write_text(yaml.safe_dump(declaration, sort_keys=False))
    shutil.copytree(ECHO_OVERLAY / "validation", target / "validation")
    provenance = {
        "source_directory": "envs/echo_env",
        "source_hashes": copied,
        "overlay_hashes": hashes(ECHO_OVERLAY),
        "dockerfile_sha256": digest(target / "Dockerfile"),
        "manifest_sha256": digest(target / "openenv.yaml"),
        "wheel_sha256": manifest["wheel_sha256"],
        "subject_imported_on_host": False,
    }
    manifest["echo_canary"] = provenance
    write_json(output / "echo-canary-source.json", provenance)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", required=True, choices=("fast", "protocol", "docker")
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/validation-runtime"
    )
    args = parser.parse_args()
    pins = json.loads((PROJECT / "toolchain.json").read_text())
    if platform.python_version() != pins["python"]:
        parser.error(f"Use pinned Python {pins['python']} via tests/validation_runtime")
    uv_version = run(["uv", "--version"])
    if uv_version.split()[1] != pins["uv"]:
        parser.error(f"Use pinned uv {pins['uv']}")
    run_id = f"l2-{uuid.uuid4().hex[:12]}"
    output = (args.output / run_id).resolve()
    output.mkdir(parents=True)
    inventory = PROJECT / "acceptance.json"
    shutil.copyfile(inventory, output / "acceptance.json")
    print(f"Evidence: {output}", flush=True)
    manifest = {
        "schema_version": "1",
        "run_id": run_id,
        "suite": args.suite,
        "acceptance_inventory_sha256": digest(inventory),
        "argv": sys.argv,
        "head_sha": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain"])),
        "diff_sha256": command_digest(["git", "diff", "HEAD", "--binary"]),
        "toolchain": pins,
        "python": platform.python_version(),
        "uv": uv_version,
        "platform": platform.platform(),
        "lock_sha256": digest(PROJECT / "uv.lock"),
        "fixture_hashes": hashes(FIXTURE),
        "require_complete": args.require_complete,
    }
    started = time.monotonic()
    status = 1
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTEST_ADDOPTS", None)
    environment["OPENENV_VALIDATION_ARTIFACTS"] = str(output)
    environment["OPENENV_REQUIRE_COMPLETE"] = "1" if args.require_complete else "0"
    try:
        with tempfile.TemporaryDirectory(prefix="openenv-validation-") as temporary:
            if args.suite == "docker":
                manifest["docker_info"] = json.loads(
                    run(
                        ["docker", "info", "--format", "{{json .}}"],
                        timeout=30,
                    )
                )
                # Exclude host-specific paths, labels and proxy settings from retained evidence.
                manifest["docker_info"] = {
                    key: manifest["docker_info"].get(key)
                    for key in (
                        "ServerVersion",
                        "Architecture",
                        "OperatingSystem",
                        "OSType",
                        "KernelVersion",
                        "Driver",
                        "CgroupDriver",
                        "CgroupVersion",
                        "MemoryLimit",
                        "SwapLimit",
                    )
                }
                context, image = stage_image(Path(temporary), output, pins, manifest)
                echo_context = stage_echo_canary(
                    Path(temporary), context, output, manifest
                )
                environment.update(
                    {
                        "OPENENV_VALIDATION_IMAGE": image,
                        "OPENENV_VALIDATION_CONTEXT": str(context),
                        "OPENENV_VALIDATION_ECHO_CONTEXT": str(echo_context),
                        "OPENENV_REQUIRE_DOCKER": "1",
                    }
                )
                test_args = [
                    str(ROOT / "tests/test_validation/integration"),
                    "-m",
                    "docker",
                ]
            elif args.suite == "protocol":
                test_args = [
                    str(ROOT / "tests/test_validation/integration"),
                    "-m",
                    "not docker",
                ]
            else:
                test_args = [
                    str(ROOT / "tests/test_validation"),
                    "--ignore=" + str(ROOT / "tests/test_validation/integration"),
                ]
            run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    *test_args,
                    "-q",
                    "--junitxml",
                    output / "junit.xml",
                ],
                cwd=output,
                env=environment,
                timeout=600,
                log=output / "logs/tests.log",
            )
            status = 0
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
        manifest["error"] = str(error)
        print(str(error), file=sys.stderr)
    finally:
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
        manifest["success"] = status == 0
        write_json(output / "run-manifest.json", manifest)
        entries = hashes(output)
        (output / "SHA256SUMS").write_text(
            "".join(f"{value}  {name}\n" for name, value in entries.items())
        )
    if status == 0 and args.require_complete:
        from verify_artifacts import verify

        try:
            verify(output)
        except ValueError as error:
            status = 1
            manifest.update(success=False, error=str(error))
            write_json(output / "run-manifest.json", manifest)
            entries = {
                name: value
                for name, value in hashes(output).items()
                if name != "SHA256SUMS"
            }
            (output / "SHA256SUMS").write_text(
                "".join(f"{value}  {name}\n" for name, value in entries.items())
            )
            print(str(error), file=sys.stderr)
    print(f"{'PASS' if status == 0 else 'FAIL'}: {output}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
