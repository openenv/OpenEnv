# SPDX-License-Identifier: BSD-3-Clause
"""Check Sudoku client cleanup without model weights or a server."""

import runpy
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from textarena_env import TextArenaEnv


ROOT = Path(__file__).resolve().parents[2]


def test_grpo_reset_releases_previous_client_loop(monkeypatch):
    clients = []

    async def reset(self):
        return SimpleNamespace(
            observation=SimpleNamespace(
                messages=[SimpleNamespace(content="Game Board:\n. . .")]
            )
        )

    def client_factory(**kwargs):
        client = TextArenaEnv(**kwargs)
        clients.append(client)
        return client

    class Trainer:
        def __init__(self, *, environment_factory, **kwargs):
            self.env = environment_factory()
            self.model = SimpleNamespace(to=lambda dtype: None)

        def train(self):
            for _ in range(4):
                self.env.reset()
            loops = {
                thread
                for thread in threading.enumerate()
                if thread.name == "openenv-sync-client-loop"
            }
            assert len(loops - initial_loops) == 1

        def push_to_hub(self):
            pass

    initial_loops = {
        thread
        for thread in threading.enumerate()
        if thread.name == "openenv-sync-client-loop"
    }
    monkeypatch.setattr(TextArenaEnv, "_reset_async", reset)
    monkeypatch.setattr("textarena_env.TextArenaEnv", client_factory)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grpo.py",
            "--model",
            "example/model",
            "--hub-model-id",
            "example/output",
            "--env-url",
            "http://localhost:18854",
        ],
    )
    modules = {
        "torch": SimpleNamespace(bfloat16="bfloat16"),
        "datasets": SimpleNamespace(
            Dataset=SimpleNamespace(from_dict=lambda data: data)
        ),
        "trl": SimpleNamespace(GRPOConfig=SimpleNamespace, GRPOTrainer=Trainer),
        "huggingface_hub": SimpleNamespace(get_token=lambda: None),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    try:
        runpy.run_path(str(ROOT / "examples/sudoku/grpo.py"), run_name="__main__")
    finally:
        for client in clients:
            client.sync().close()
