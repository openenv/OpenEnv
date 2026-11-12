"""Create three trusted tiny tasks and exercise validation/filtering with local Docker."""

import argparse
from pathlib import Path

from examples.hub_validation_pipeline.pipeline import PILOT, run


def create_tasks(root):
    """One valid task, one broken verifier, and one missing reference solution."""
    config = """version = "1.0"
[agent]
timeout_sec = 30.0
[verifier]
timeout_sec = 10.0
environment_mode = "shared"
[environment]
cpus = 1
memory_mb = 512
storage_mb = 1024
"""
    verifier = """#!/bin/bash
mkdir -p /logs/verifier
if [ "$(cat /app/answer.txt 2>/dev/null)" = "42" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""
    for name in ("good", "always-rewarding", "missing-oracle"):
        task = root / name
        (task / "environment").mkdir(parents=True)
        (task / "tests").mkdir()
        canaries = task / "validation/canaries"
        canaries.mkdir(parents=True)
        (canaries / "empty.sh").write_text("#!/bin/bash\n:\n")
        (canaries / "wrong-answer.sh").write_text(
            "#!/bin/bash\nprintf '41\\n' > /app/answer.txt\n"
        )
        (task / "task.toml").write_text(config)
        (task / "instruction.md").write_text("Write exactly 42 into /app/answer.txt.\n")
        (task / "environment/Dockerfile").write_text(
            "FROM python:3.12-slim\nWORKDIR /app\n"
        )
        (task / "tests/test.sh").write_text(
            "#!/bin/bash\nmkdir -p /logs/verifier\necho 1 > /logs/verifier/reward.txt\n"
            if name == "always-rewarding"
            else verifier
        )
        if name != "missing-oracle":
            (task / "solution").mkdir()
            (task / "solution/solve.sh").write_text(
                "#!/bin/bash\nprintf '42\\n' > /app/answer.txt\n"
            )
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New demo directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = create_tasks(args.output.resolve() / "dataset")
    options = argparse.Namespace(
        output=args.output / "run",
        contract=Path(__file__).with_name("contract.json"),
        profile=PILOT,
        source="both",
        dataset_limit=0,
        tasks_per_dataset=3,
        hf_dataset=[],
        harbor_dataset=[],
        local_dataset=[dataset],
        validation_timeout=300,
        llm_url="",
        model="",
        harness="opencode",
        expose="gradio",
        model_timeout=120,
        model_step_limit=32,
    )
    status = run(options)
    import json

    summary = json.loads((options.output / "summary.json").read_text())
    if status != 0 or summary["accepted"] != 1 or summary["excluded"] != 2:
        raise SystemExit("demo failed: expected one admitted task and two exclusions")
    print(
        "Demo passed: good task admitted; broken verifier and missing oracle excluded."
    )


if __name__ == "__main__":
    main()
