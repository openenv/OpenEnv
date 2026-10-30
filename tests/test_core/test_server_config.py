# SPDX-License-Identifier: BSD-3-Clause

"""Test the text parser without importing optional container/runtime packages."""

import importlib.util
import unittest
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/openenv/core/containers/runtime/_server_config.py"
)
_SPEC = importlib.util.spec_from_file_location("openenv_server_config", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class TestDockerfileCmd(unittest.TestCase):
    def test_logical_instructions(self):
        cases = [
            ("CMD uvicorn app:app\n", "uvicorn app:app"),
            ('CMD ["uvicorn", "app:app"]\n', "uvicorn app:app"),
            ("CMD first\nCMD second\n", "second"),
            ("# CMD ignored\ncmd real\n", "real"),
            ("FROM python:3.12\nCMD \n", None),
            ("CMD [not valid json\n", "[not valid json"),
            ("CMD unfinished \\", "unfinished \\"),
            ("CMD uvicorn \\\n    app:app\n", "uvicorn     app:app"),
            ('CMD ["uvicorn", \\\n    "app:app"]\n', "uvicorn app:app"),
            ("CMD uvicorn \\\n# comment\n    app:app\n", "uvicorn     app:app"),
            ("CMD uvicorn \\\n\n    app:app\n", "uvicorn     app:app"),
            ("CMD printf hel\\\nlo\n", "printf hello"),
            ("CMD printf hel\\\r\nlo\r\n", "printf hello"),
            ("CMD printf hel\\ \t\nlo\n", "printf hello"),
            ("RUN echo \\\nCMD ignored\n", None),
            ("CMD first\nRUN echo \\\nCMD ignored\n", "first"),
            ("CMD first\nCMD printf \\\\ \nRUN echo next\n", "printf \\\\"),
            ("# escape=`\nCMD printf hel`\nlo\n", "printf hello"),
            (
                "# syntax=docker/dockerfile:1\n# escape=`\nCMD printf hel`\nlo\n",
                "printf hello",
            ),
            ("# comment\n# escape=`\nCMD printf hel`\n", "printf hel`"),
            ("\n# escape=`\nCMD printf hel`\n", "printf hel`"),
            ('CMD printf "hello \\\nworld"\n', 'printf "hello world"'),
            ('CMD echo "# text"\n', 'echo "# text"'),
        ]
        for dockerfile, expected in cases:
            with self.subTest(dockerfile=dockerfile):
                self.assertEqual(_MODULE.parse_dockerfile_cmd(dockerfile), expected)


if __name__ == "__main__":
    unittest.main()
