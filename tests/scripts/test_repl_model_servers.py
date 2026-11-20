# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the documented REPL model server examples."""

import json
import runpy
import sys
from pathlib import Path

import httpx
import pytest
from openai import OpenAI


EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "repl_model_servers"
COUNT_WORDS = "print(FINAL(len(context.split())))"


@pytest.mark.parametrize(
    ("script", "args", "replies", "expected_answer"),
    [
        pytest.param(
            "vllm_runner.py",
            ["--model", "local-model"],
            [("http://127.0.0.1:8000/v1/chat/completions", "local-model", COUNT_WORDS)],
            "3",
            id="vllm",
        ),
        pytest.param(
            "separate_models.py",
            [
                "--parent-url",
                "https://parent.example.invalid/v1",
                "--parent-model",
                "parent-model",
                "--child-url",
                "https://child.example.invalid/v1",
                "--child-model",
                "child-model",
            ],
            [
                (
                    "https://parent.example.invalid/v1/chat/completions",
                    "parent-model",
                    "lines = context.splitlines()\n"
                    "first = rlm_query(lines[0])\n"
                    "second = rlm_query(lines[1])\n"
                    "print(FINAL(int(first) + int(second)))",
                ),
                (
                    "https://child.example.invalid/v1/chat/completions",
                    "child-model",
                    COUNT_WORDS,
                ),
                (
                    "https://child.example.invalid/v1/chat/completions",
                    "child-model",
                    COUNT_WORDS,
                ),
            ],
            "5",
            id="separate-models",
        ),
    ],
)
def test_repl_model_server_example(
    monkeypatch, capsys, script, args, replies, expected_answer
):
    pytest.importorskip("smolagents")
    requests = []

    def respond(request):
        assert request.method == "POST"
        model = json.loads(request.content)["model"]
        requests.append((str(request.url), model))
        _, _, code = replies[len(requests) - 1]
        return httpx.Response(
            200,
            json={
                "id": "scripted",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": f"```repl\n{code}\n```",
                        },
                    }
                ],
            },
        )

    init_client = OpenAI.__init__

    def init_with_fake_server(client, **kwargs):
        init_client(
            client,
            **kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
            max_retries=0,
        )

    monkeypatch.setattr(OpenAI, "__init__", init_with_fake_server)
    for key in ("REPL_API_KEY", "REPL_PARENT_API_KEY", "REPL_CHILD_API_KEY"):
        monkeypatch.setenv(key, "test-key")
    monkeypatch.syspath_prepend(str(EXAMPLES))
    monkeypatch.setattr(sys, "argv", [script, *args])

    runpy.run_path(str(EXAMPLES / script), run_name="__main__")

    assert capsys.readouterr().out.strip() == expected_answer
    assert requests == [(url, model) for url, model, _ in replies]
