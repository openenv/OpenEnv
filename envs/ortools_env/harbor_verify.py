# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Verifier of an exported Harbor task, copied to `/tests/verify.py` next to `problem.py` (the problem module) and
`task.json` (level, references and, at level 2, the hidden evaluation instances). Writes the score to
`/logs/verifier/reward.txt`.

At level 2, the submitted `solve` runs as the unprivileged `sandbox` user of the task image, so it cannot read the
references in `/tests` or write the reward, and its whole process group is killed after each instance, so nothing it
starts outlives the call.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile

sys.path.insert(0, "/tests")
import problem  # noqa: E402

RUNNER = """\
import json, sys
sys.path.insert(0, "/app")
from solve import solve
with open(sys.argv[1]) as f:
    instance = json.load(f)
with open(sys.argv[2], "w") as f:
    json.dump(solve(instance), f, default=lambda o: o.tolist())
"""


def score(objective, reference, baseline):
    # Same as `problems.score` in the environment.
    if objective is None:
        return -1.0
    if baseline <= reference:
        return 1.0 if objective <= reference else -1.0
    return max(-1.0, (baseline - objective) / (baseline - reference))


def level_1(task):
    with open("/app/instance.json") as f:
        instance = json.load(f)
    with open("/app/solution.json") as f:
        solution = json.load(f)
    objective = problem.evaluate(instance, solution)
    print(
        f"objective={objective} naive={task['baseline']} reference={task['reference']}"
    )
    return score(objective, task["reference"], task["baseline"])


def level_2(task):
    scores = []
    for k, (instance, reference, baseline) in enumerate(
        zip(task["eval_instances"], task["eval_references"], task["eval_baselines"])
    ):
        workdir = tempfile.mkdtemp()
        instance_path, solution_path = (
            os.path.join(workdir, "instance.json"),
            os.path.join(workdir, "solution.json"),
        )
        with open(instance_path, "w") as f:
            json.dump(instance, f)
        shutil.chown(workdir, "sandbox", "sandbox")
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", RUNNER, instance_path, solution_path],
            cwd=workdir,
            start_new_session=True,
            user="sandbox",
            group="sandbox",
            extra_groups=[],
        )
        try:
            if proc.wait(timeout=task["solve_time_limit_s"]) != 0:
                raise RuntimeError(f"solve exited with code {proc.returncode}")
            with open(solution_path) as f:
                objective = problem.evaluate(instance, json.load(f))
        except Exception as e:  # noqa: BLE001 - any failure of the submitted code is infeasible
            print(f"instance {k}: {type(e).__name__}: {e}")
            scores.append(score(None, reference, baseline))
            continue
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        print(
            f"instance {k}: objective={objective} naive={baseline} reference={reference}"
        )
        scores.append(score(objective, reference, baseline))
    return sum(scores) / len(scores)


def main():
    os.chmod("/tests", 0o700)
    subprocess.run(["chmod", "-R", "a+rX", "/app"], check=True)
    os.makedirs("/logs/verifier", exist_ok=True)
    os.chmod("/logs", 0o700)
    with open("/tests/task.json") as f:
        task = json.load(f)
    try:
        reward = level_1(task) if task["level"] == 1 else level_2(task)
    except Exception as e:  # noqa: BLE001 - a missing or infeasible submission
        print(f"{type(e).__name__}: {e}")
        reward = score(None, 0, 0)
    print(f"reward={reward}")
    os.makedirs("/logs/verifier", exist_ok=True)
    with open("/logs/verifier/reward.txt", "w") as f:
        f.write(str(reward))


if __name__ == "__main__":
    main()
