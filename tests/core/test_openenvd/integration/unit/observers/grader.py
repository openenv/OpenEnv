"""Verdict observer: reads assets and the runner's output, never agent code."""

import json
import os

env = os.environ.get
with open(
    os.path.join(env("OPENENVD_ASSETS", "/assets"), "oracle", "oracle.json")
) as f:
    oracle = json.load(f)
with open(
    os.path.join(env("OPENENVD_INPUTS", "/inputs"), "runner", "report.json")
) as f:
    runner = json.load(f)
verdict = {
    "score": 1.0 if oracle and not runner["assets_visible"] else 0.0,
    "runner_saw_assets": runner["assets_visible"],
}
with open(os.path.join(env("OPENENVD_OUT", "/out"), "verdict.json"), "w") as f:
    json.dump(verdict, f)
