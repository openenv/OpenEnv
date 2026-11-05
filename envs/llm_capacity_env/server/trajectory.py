"""Server-owned structured trajectory files, never exposed as agent tools."""

import json
import threading
from pathlib import Path
from uuid import uuid4


class TrajectoryRecorder:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else None
        self.path = None
        self.lock = threading.Lock()

    def start(self, seed, episode_id, observation):
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        # User-supplied episode IDs are data, never filesystem path components.
        self.path = self.directory / (uuid4().hex + ".jsonl")
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "type": "reset",
                        "seed": seed,
                        "episode_id": episode_id,
                        "observation": observation,
                    },
                    allow_nan=False,
                )
                + "\n"
            )

    def append(self, action, observation, step_count):
        if self.path is None:
            return
        payload = {
            "type": "transition",
            "action": action.model_dump(mode="json"),
            "observation": observation.model_dump(mode="json"),
            "reward": observation.reward,
            "terminal": observation.done,
            "step_count": step_count,
        }
        # Serialization may fail on an invalid nonfinite action. Preserve the rejection
        # without inventing a JSON number or changing the tool's action/error semantics.
        try:
            line = json.dumps(payload, allow_nan=False)
        except ValueError:
            payload["action"] = {"serialization_error": "nonfinite value in rejected action"}
            line = json.dumps(payload, allow_nan=False)
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
