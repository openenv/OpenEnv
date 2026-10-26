"""Execute only inside the sandbox where Harbor installs this workflow package."""

import asyncio
import json
import os
import signal


async def execute(command: str, timeout: float = 60.0, cwd: str = "/workdir") -> str:
    proc = await asyncio.create_subprocess_exec(
        "bash",
        "-lc",
        command,
        cwd=cwd,
        start_new_session=True,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (TimeoutError, asyncio.CancelledError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()
        raise
    return json.dumps(
        {
            "exit_code": proc.returncode,
            "stdout": stdout.decode(errors="replace")[:24000],
            "stderr": stderr.decode(errors="replace")[:8000],
        }
    )
