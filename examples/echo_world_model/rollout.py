"""Collect rollouts in the mini-terminal env and tag them by token role.

`oracle_rollout` runs a known-good command sequence so the env produces *diverse,
complex* observations (file listings, matched log lines, counts) — the rich
"world" the model learns to predict. (In a full RL run the **policy itself**
generates the actions and even its *failed* commands return real, predictable
error observations that still teach; `model_rollout` shows that path.)

Each rollout is segmented into the four token roles in ``trajectory.py``:
``context`` (shell prompt / scaffolding), ``action`` (the agent's command),
``warning`` (a harness ``[stdout]`` tag — excluded from the env loss by default),
and ``env_output`` (the real terminal response — the ECHO target).
"""

from __future__ import annotations

from mini_terminal_env import MiniTerminalEnv, Task, run_command
from trajectory import ACTION, CONTEXT, ENV_OUTPUT, WARNING, Segment, Trajectory

PROMPT = "\n$ "  # shell prompt (context)
STDOUT_TAG = "\n[stdout] "  # harness boilerplate (warning role)


def oracle_rollout(task: Task) -> Trajectory:
    """Build a clean, role-tagged trajectory by running the task's oracle commands."""
    segs: list[Segment] = [Segment(CONTEXT, f"Task: {task.prompt}")]
    for cmd in task.oracle:
        out = run_command(cmd)
        segs.append(Segment(CONTEXT, PROMPT))
        segs.append(Segment(ACTION, cmd))
        segs.append(Segment(WARNING, STDOUT_TAG))
        segs.append(Segment(ENV_OUTPUT, out))
    segs.append(Segment(CONTEXT, PROMPT))
    segs.append(Segment(ACTION, f"answer: {task.solution}"))
    return Trajectory(segments=segs, reward=1.0, task_prompt=task.prompt)


def model_rollout(
    model, tokenizer, task: Task, *, max_turns: int = 4, max_new: int = 12
) -> Trajectory:
    """Roll out the *policy* itself: the model proposes each command, the env
    answers with a real observation (including realistic errors for bad
    commands). Used for the RL / "failures still teach" path."""
    import torch

    env = MiniTerminalEnv()
    env.reset(task)
    segs: list[Segment] = [Segment(CONTEXT, f"Task: {task.prompt}")]
    transcript = f"Task: {task.prompt}"
    reward = 0.0
    for _ in range(max_turns):
        transcript += PROMPT
        segs.append(Segment(CONTEXT, PROMPT))
        ids = tokenizer(transcript, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        action = tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
        action = action.splitlines()[0].strip() if action.strip() else "ls"
        segs.append(Segment(ACTION, action))
        obs, r, done = env.step(action)
        segs.append(Segment(WARNING, STDOUT_TAG))
        segs.append(Segment(ENV_OUTPUT, obs))
        transcript += action + STDOUT_TAG + obs
        reward = max(reward, r)
        if done:
            break
    return Trajectory(segments=segs, reward=reward, task_prompt=task.prompt)
