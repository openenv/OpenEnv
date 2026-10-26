"""An owned X display and real pointer input inside the browser VM."""

import os
import select
import subprocess
import tempfile
from pathlib import Path


class VirtualDesktop:
    def __init__(self):
        self.process = None
        self.directory = tempfile.TemporaryDirectory(prefix="browser-display-")
        self.log = (Path(self.directory.name) / "xvfb.log").open("wb")
        try:
            self.process = subprocess.Popen(
                ["Xvfb", "-displayfd", "1", "-screen", "0", "801x601x24", "-nolisten", "tcp"],
                stdout=subprocess.PIPE,
                stderr=self.log,
            )
            if not select.select([self.process.stdout], [], [], 10)[0]:
                raise RuntimeError("virtual display startup timed out")
            number = self.process.stdout.readline().decode().strip()
            if not number.isdecimal() or self.process.poll() is not None:
                raise RuntimeError("virtual display failed to start")
            self.env = dict(os.environ, DISPLAY=":" + number)
        except BaseException:
            self.close()
            raise

    def focus_browser(self, process_id):
        window_ids = subprocess.check_output(
            ["xdotool", "search", "--onlyvisible", "--pid", str(process_id)],
            env=self.env,
            text=True,
            timeout=10,
        ).splitlines()
        if not window_ids or not window_ids[0].isdecimal():
            raise RuntimeError("browser window unavailable")
        subprocess.run(
            ["xdotool", "windowfocus", "--sync", window_ids[0]],
            env=self.env,
            check=True,
            timeout=10,
        )

    def click(self, x, y):
        subprocess.run(
            ["xdotool", "mousemove", str(x), str(y), "click", "1"],
            env=self.env,
            check=True,
            timeout=10,
        )

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            if self.process.stdout is not None:
                self.process.stdout.close()
            self.process = None
        self.log.close()
        self.directory.cleanup()
