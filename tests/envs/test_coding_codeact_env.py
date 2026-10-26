# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from coding_env.models import CodeAction
from coding_env.server.python_codeact_env import PythonCodeActEnv
from coding_env.server.python_executor import DEFAULT_SAFE_IMPORTS
from openenv.core.env_server.interfaces import Observation, Transform

# `json` is authorized through DEFAULT_SAFE_IMPORTS, `base64` only when the
# caller asks for it; neither is part of the smolagents base allowlist.
IMPORT_JSON = "import json\nprint(json.dumps({'a': 1}))"
IMPORT_BASE64 = "import base64\nprint(base64.b64encode(b'a').decode())"


class RecordingTransform(Transform):
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, observation: Observation) -> Observation:
        self.calls += 1
        return observation


def test_additional_imports_keep_the_default_safe_imports():
    env = PythonCodeActEnv(additional_imports=["base64"])
    env.reset()

    for code in (IMPORT_JSON, IMPORT_BASE64):
        observation = env.step(CodeAction(code=code))
        assert observation.exit_code == 0, observation.stderr


def test_additional_imports_survive_reset():
    env = PythonCodeActEnv(additional_imports=["base64"])
    env.reset()
    env.reset()

    for code in (IMPORT_JSON, IMPORT_BASE64):
        observation = env.step(CodeAction(code=code))
        assert observation.exit_code == 0, observation.stderr


def test_default_constructor_authorizes_only_the_safe_imports():
    env = PythonCodeActEnv()
    env.reset()

    assert DEFAULT_SAFE_IMPORTS == ["json"]
    assert env.step(CodeAction(code=IMPORT_JSON)).exit_code == 0
    assert env.step(CodeAction(code=IMPORT_BASE64)).exit_code == 1


def test_explicit_transform_survives_reset():
    transform = RecordingTransform()
    env = PythonCodeActEnv(transform=transform)
    env.reset()

    assert env.transform is transform
