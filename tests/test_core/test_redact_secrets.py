# SPDX-License-Identifier: BSD-3-Clause

"""Tests for `redact_secrets`, run on agent traces before they leave the machine."""

import json

import pytest
from openenv.core.utils import redact_secrets

# Built at runtime so the repository holds no string that looks like a real key.
HF = "hf_" + "x" * 34
OPENAI = "sk-proj-" + "y" * 40
GITHUB = "ghp_" + "z" * 36
AWS = "AKIA" + "Q" * 16
PRIVATE_KEY = "-----BEGIN RSA PRIVATE KEY-----\\nabc\\n-----END RSA PRIVATE KEY-----"


@pytest.mark.parametrize("secret", [HF, OPENAI, GITHUB, AWS, PRIVATE_KEY])
def test_known_key_formats_are_redacted(secret):
    text, count = redact_secrets(f'{{"content": "env says {secret} ok"}}')
    assert text == '{"content": "env says <REDACTED> ok"}'
    assert count == 1


def test_values_of_secret_env_vars_are_redacted(monkeypatch):
    monkeypatch.setenv("MY_SERVICE_TOKEN", "custom-format-1234")
    text, count = redact_secrets(
        "printenv: custom-format-1234, twice: custom-format-1234"
    )
    assert text == "printenv: <REDACTED>, twice: <REDACTED>"
    assert count == 2


@pytest.mark.parametrize(
    "value", ['pa"ss\\wörd-1234', "secret-1234\\", "\\secret-1234"]
)
def test_env_values_are_redacted_inside_json(monkeypatch, value):
    """Traces are JSON, so a value with quotes, backslashes or non-ASCII appears escaped."""
    monkeypatch.setenv("DB_PASSWORD", value)
    text, count = redact_secrets(json.dumps({"content": f"env says {value}"}))
    assert json.loads(text) == {"content": "env says <REDACTED>"}
    assert count == 1


def test_other_env_vars_are_kept(monkeypatch):
    """`TOKENIZER_NAME` holds a model id, not a credential."""
    monkeypatch.setenv("TOKENIZER_NAME", "Qwen/Qwen3.5-9B")
    assert redact_secrets("model Qwen/Qwen3.5-9B") == ("model Qwen/Qwen3.5-9B", 0)
