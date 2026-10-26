"""Run one Harbor trial through the capture proxy and return trainable output.

This is where every piece meets: a task from the dataset, a harness from the seam table, a sandbox
from Harbor, and a capture session whose id doubles as the agent's API key.

    task dir ──┐
    harness ───┼──> TrialConfig ──> Trial.run() ──> TrialResult (reward)
    sandbox ───┘                         │
                                         └──> agent talks to the intercept
                                              (api key == session id)
                                                     │
                                              RolloutGraph ──> HarborRolloutResult

**Nothing raises out of `run_rollout`.** A failed rollout returns `ok=False` with `reward=None`. That
is not defensive habit, it is the reason this layer exists: in the white-box predecessor a rollout
exception reached the trainer and hung every rank at the NCCL barrier forever, so every method there
had to be individually wrapped. Behind a result object that failure mode cannot occur.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import url2pathname

from openenv.core.harness.capture.export import export_session

from . import seams
from .atif import load_trace, reconcile
from .models import (
    conversations_from_document,
    HarborRolloutResult,
    HarborStepResult,
    HarborTurn,
    turns_from_document,
)

# Harbor's own retry is deliberately off. A Harbor-level retry re-runs the agent against the SAME
# capture session, so two attempts merge into one graph and the trace cross-check compares a
# two-attempt capture against a one-attempt trajectory. One attempt = one session = one rollout;
# retries belong to the caller, with a fresh session each time.
_NO_HARBOR_RETRY = 0


# Some agents read os.environ at CONSTRUCTION, before any sandbox exists — Harbor's claude-code
# wrapper decides there which credentials to forward. For those, `agent_env` alone is too late, so
# the seam also carries a process-env channel.
#
# Others read it at RUN time instead: Harbor's goose wrapper does so inside its own `run`
# (goose.py:653). So the override is held across `Trial.create` AND `trial.run()` whenever a seam
# carries `proc_env`, because restoring it in between handed goose the operator's real provider key
# and earned a 401 from our own proxy on every call.
#
# That channel is global, so two concurrent rollouts of a proc-env harness would overwrite each
# other's session key, and holding it across the whole trial means those rollouts serialise. That is
# not an implementation shortcut but a property of the agent: it reads a process-global variable at
# run time, so per-rollout isolation is impossible without a process per rollout. Seams that pass
# credentials through `agent_env` — most of them — neither take the lock nor wait on it.
#
# A `threading.Lock`, not an `asyncio.Lock`. What is being protected is `os.environ`, which is global
# to the PROCESS, not to an event loop, and rollouts arrive on several loops: the env server answers
# each request on its own loop, and any caller using `asyncio.run` per rollout creates another. An
# asyncio lock binds to the first loop that uses it and then raises
# "is bound to a different event loop" for every other one, which fails 100% of concurrent rollouts
# while passing every sequential test.
_PROC_ENV_LOCK = threading.Lock()


async def _acquire_proc_env() -> None:
    """Take `_PROC_ENV_LOCK` without leaking it if the caller is cancelled.

    `asyncio.to_thread` is not cancellable: cancelling the await abandons the future, but the worker
    thread runs on and still takes the lock. Nobody is left holding a frame that releases it, so a
    single cancelled proc-env rollout — a client timeout, a trainer shutting a worker down — would
    wedge every later goose rollout on a lock with no owner. Hence the done-callback: whoever ends up
    acquiring is the one who releases.
    """
    acquiring = asyncio.ensure_future(asyncio.to_thread(_PROC_ENV_LOCK.acquire))
    try:
        await asyncio.shield(acquiring)
    except asyncio.CancelledError:
        acquiring.add_done_callback(
            lambda fut: _PROC_ENV_LOCK.release() if not fut.cancelled() else None
        )
        raise


@contextlib.contextmanager
def process_env(seam_name: str, env: dict[str, str]):
    """Set the seam's process-level env vars for the duration of the block, then put them back.

    Restoring matters because this is global state. Without it the last rollout's session id stays
    in `os.environ` for the life of the process, so anything afterwards, including the next
    rollout's grader, sees another harness's credentials. That is the same class of cross-talk the
    lock exists to prevent, just spread over time instead of across threads.

    OPENAI_API_KEY is shared with the task grader: Harbor forwards it into the sandbox and the
    DataAgent grader's LLM-judge tier fires on `if os.environ.get("OPENAI_API_KEY")`. Overwriting it
    with a session id makes the judge 401, so every semantically-correct-but-not-exact answer scores
    0 — which reads as a weak model rather than a broken harness, and poisons the RL baseline.
    """
    grader_key = os.environ.get("OPENAI_API_KEY")
    previous = {key: os.environ.get(key) for key in env}
    for key, value in env.items():
        if key == "OPENAI_API_KEY" and grader_key and value != grader_key:
            print(
                f"[{seam_name}] WARNING: this seam overwrites OPENAI_API_KEY, which the grader "
                "uses for its LLM-judge tier. Judging is disabled for this run (exact-match and "
                "numeric tolerance still apply)."
            )
        os.environ[key] = value
    try:
        yield
    finally:
        for key, was in previous.items():
            if was is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = was


def _finite_reward(key: str, value: Any) -> float:
    """A verifier's value as a real float, or raise.

    Args:
        key (`str`):
            Reward name, for the error message — a caller needs to know WHICH key was unusable.
        value:
            Whatever the task's verifier put in its reward dict.

    Returns:
        `float`: The coerced value.

    Raises:
        TypeError: If the value cannot be a float at all.
        ValueError: If it coerces to inf or nan.
    """
    number = float(value)  # raises TypeError/ValueError, caught by the caller
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{key}={value!r} is not finite")
    return number


def _pick_reward(
    rewards: dict[str, float], reward_key: str = ""
) -> tuple[float | None, str]:
    """Collapse Harbor's reward dict to the one scalar an RL trainer consumes.

    Refuses rather than guesses. Harbor lets a task emit any keys it likes, and inventing a rule to
    combine them is inventing reward semantics — which is exactly how a previous run got reward
    hacked: a `+0.2 for submitting anything` term made the policy learn to quick-submit, training
    reward looked healthy, and eval collapsed from 0.740 to 0.178.

    Args:
        rewards (`dict[str, float]`):
            The verifier's reward dict, as Harbor produced it.
        reward_key (`str`, *optional*):
            Force a specific key, or a comma-separated preference order. Required when the dict
            has several and none is named `reward`. Selection happens on the same verifier result.

    Returns:
        `tuple[float | None, str]`: The chosen value and the key it came from.

    Raises:
        ValueError: If the key is ambiguous and none was given.
    """
    if not rewards:
        return None, ""
    if reward_key:
        # Preserve literal keys (including commas) before interpreting a preference list.
        candidates = [reward_key] + [key.strip() for key in reward_key.split(",")]
        for key in candidates:
            if key in rewards:
                return float(rewards[key]), key
        raise ValueError(f"reward_key {reward_key!r} not in {sorted(rewards)}")
    if len(rewards) == 1:
        key = next(iter(rewards))
        return float(rewards[key]), key
    if "reward" in rewards:
        return float(rewards["reward"]), "reward"
    raise ValueError(
        f"task produced several rewards {sorted(rewards)} and none is named 'reward'. "
        "Pass reward_key= to say which one is the training signal."
    )


def build_trial_config(
    *,
    task_dir: Path | str,
    harness: str,
    sandbox: str,
    intercept_url: str,
    session_id: str,
    model: str,
    trial_name: str,
    trials_dir: Path | str,
    keep_sandbox: bool = False,
    agent_timeout_sec: float | None = None,
    agent_step_limit: int | None = None,
    force_build: bool = False,
    training: bool = False,
    harness_profile: str | None = None,
) -> Any:
    """Assemble Harbor's `TrialConfig` for one rollout.

    The seam decides how this particular agent learns about the proxy — an env var, a constructor
    kwarg, or a config file written by a subclass. That is the only per-agent knowledge involved;
    everything downstream is identical.
    """
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
        VerifierConfig,
    )

    seam = seams.get(harness, profile=harness_profile)
    model_name, kwargs, agent_env, _proc_env = seam.resolve(
        base_url=intercept_url,
        session=session_id,
        model=model,
        step_limit=agent_step_limit,
        training=training,
    )

    # Deployment-level pins use Harbor's existing InstalledAgent(version=...) support.
    # Unlisted harnesses retain Harbor's default; malformed pins must not reach an installer.
    versions = json.loads(os.environ.get("OPENENV_HARBOR_AGENT_VERSIONS", "{}"))
    if not isinstance(versions, dict) or any(
        not isinstance(name, str)
        or not isinstance(version, str)
        or re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}(?:[-+][A-Za-z0-9.-]+)?", version)
        is None
        for name, version in versions.items()
    ):
        raise ValueError(
            "OPENENV_HARBOR_AGENT_VERSIONS must map harness names to exact numeric versions"
        )
    if harness in versions:
        kwargs = {**kwargs, "version": versions[harness]}

    agent = AgentConfig(
        name=seam.import_path or harness,
        import_path=seam.import_path,
        model_name=model_name,
        kwargs=kwargs,
        env=agent_env,
        override_timeout_sec=agent_timeout_sec,
    )
    return TrialConfig(
        task=TaskConfig(path=Path(str(task_dir))),
        agent=agent,
        environment=EnvironmentConfig(
            type=sandbox,
            import_path=(
                "openenv.harbor.e2b_stream:E2BStreamingEnvironment"
                if sandbox == "e2b"
                and os.environ.get("OPENENV_E2B_STREAM_UPLOADS") == "1"
                else None
            ),
            delete=not keep_sandbox,
            force_build=force_build,
        ),
        verifier=VerifierConfig(),
        trial_name=trial_name,
        trials_dir=Path(str(trials_dir)),
    )


async def run_rollout(
    *,
    task_dir: Path | str,
    harness: str,
    sandbox: str,
    registry: Any,
    intercept_url: str,
    model: str,
    trials_dir: Path | str,
    dataset: str = "",
    reward_key: str = "",
    require_reward: bool = True,
    keep_sandbox: bool = False,
    agent_timeout_sec: float | None = None,
    agent_step_limit: int | None = None,
    force_build: bool = False,
    session_prefix: str = "oe",
    capture_level: str = "tokens",
    upstream: Any = None,
    inference: Any = None,
    sampling: dict[str, Any] | None = None,
    purpose: str = "auto",
    eval_sampling: dict[str, Any] | None = None,
    harness_profile: str | None = None,
    on_session_created: Callable[[str], None] | None = None,
) -> HarborRolloutResult:
    """Run one rollout end to end. Never raises.

    Args:
        task_dir (`Path` or `str`):
            The Harbor task directory to run.
        harness (`str`):
            A seam name (`opencode`, `claude-code`, ...) or a `module:Class` import path.
        sandbox (`str`):
            A Harbor `EnvironmentType`, e.g. `e2b` or `modal`.
        registry (`SessionRegistry`):
            The live capture registry; a session is minted here and its id becomes the agent's key.
        intercept_url (`str`):
            Public URL of the capture proxy, as the sandbox must reach it.
        model (`str`):
            Served model id. Normalised per harness by the seam.
        trials_dir (`Path` or `str`):
            Where Harbor writes trial artifacts.
        reward_key (`str`, *optional*):
            Which reward key is the training signal, for multi-reward tasks.
        require_reward (`bool`, *optional*, defaults to `True`):
            Fail the rollout when the verifier's rewards leave no headline one. `False` keeps it,
            with every reward in `rewards` and a warning in `findings`: for a rollout to look at.
        keep_sandbox (`bool`, *optional*, defaults to `False`):
            Leave the sandbox alive after the run, for debugging.
        capture_level (`str`, *optional*, defaults to `"tokens"`):
            What the inference endpoint can return, as probed at startup. Below `tokens` this is an
            eval rollout: reward and full trace, no token fields.
        upstream (`Upstream`, *optional*):
            The engine this rollout's captured calls go to, already resolved and probed by the caller.
            `None` uses the capture server's default engine.
        inference (`InferenceClient`, *optional*):
            The proxy's upstream client, read at the end for the parameter workarounds it had to
            apply. Passed rather than looked up so this function keeps no handle on the server.
        sampling (`dict`, *optional*):
            Explicit training policy. Requires a finite positive temperature and full-vocabulary
            sampling. Overrides harness sampling settings and records both requested and submitted
            values. Must match the trainer's recompute temperature. Also selects the seam's native
            training settings; OpenCode disables delegation, questions and web fetch.
        on_session_created (`Callable[[str], None]`, *optional*):
            Receive this rollout's capture session ID before the agent starts. Live views must
            follow this ID rather than discover unrelated sessions from the shared registry.

    Returns:
        [`HarborRolloutResult`]: Reward, per-turn token ids and logprobs, and validation findings.
    """
    from openenv.core.harness.capture.sessions import rollout_type_for

    try:
        rollout_type = rollout_type_for(purpose, capture_level)
        if purpose == "eval" and sampling is not None:
            raise ValueError("eval purpose cannot apply a training sampling override")
    except ValueError as exc:
        return HarborRolloutResult(
            ok=False,
            error=str(exc),
            harness=harness,
            sandbox=sandbox,
            capture_level=capture_level,
            rollout_type="eval",
        )
    task_dir = Path(str(task_dir))
    started = time.monotonic()

    # The session id lands in Harbor's E2B sandbox metadata (it is derived from trial_name), which
    # is what later makes it possible to reap only the sandboxes this server created.
    trial_name = f"{session_prefix}-{task_dir.name[:28]}-{uuid.uuid4().hex[:8]}"
    # `upstream` names the engine for THIS rollout. Passing it here is what lets one server serve a
    # train-tier and an eval-tier engine at once: the tier is measured per engine when the session is
    # created, and travels on the session rather than on the server.
    try:
        session = registry.create(
            session_id=None,
            upstream=upstream,
            capture_level=capture_level,
            max_model_calls=agent_step_limit or 0,
            sampling=sampling,
            purpose=purpose,
            eval_sampling=eval_sampling,
            harness=harness,
            sandbox=sandbox,
            task=task_dir.name,
        )
    except ValueError as exc:
        return HarborRolloutResult(
            task_id=str(task_dir),
            task_name=task_dir.name,
            dataset=dataset,
            harness=harness,
            sandbox=sandbox,
            capture_level=capture_level,
            rollout_type=rollout_type,
            ok=False,
            error=str(exc),
            exception_type=type(exc).__name__,
        )
    try:
        # The session is the single source of truth for the level from here on: the caller resolved the
        # engine and its measured tier before calling, and a rollout must never claim a level the engine
        # was not measured at.
        # `getattr`, because `registry` is deliberately untyped: callers pass their own registry (the
        # trainer-side runner does), and a session object that predates per-session levels should degrade
        # to the caller's hint rather than crash the rollout.
        capture_level = getattr(session, "capture_level", "") or capture_level

        result = HarborRolloutResult(
            task_id=str(task_dir),
            task_name=task_dir.name,
            dataset=dataset,
            harness=harness,
            sandbox=sandbox,
            trial_name=trial_name,
            session_id=session.session_id,
            capture_level=capture_level,
            rollout_type=rollout_type,
        )

        trial_result = None
        try:
            if on_session_created is not None:
                on_session_created(session.session_id)
            from harbor.trial.trial import Trial

            # Collapse identical per-task templates onto one alias when asked. Must happen before the
            # first `Trial.create`, and is idempotent, so calling it per rollout is free.
            from .shared_template import enable_shared_templates

            enable_shared_templates()

            config_kwargs = dict(
                task_dir=task_dir,
                harness=harness,
                sandbox=sandbox,
                intercept_url=intercept_url,
                session_id=session.session_id,
                model=model,
                trial_name=trial_name,
                trials_dir=trials_dir,
                keep_sandbox=keep_sandbox,
                agent_timeout_sec=agent_timeout_sec,
                agent_step_limit=agent_step_limit,
                force_build=force_build,
                training=sampling is not None,
                harness_profile=harness_profile,
            )

            # Held across construction because Harbor's wrappers read `os.environ` while building the
            # agent, so the variables must still be in place when `Trial.create` runs. It is a blocking
            # acquire inside an async function, which is acceptable only because construction does no
            # I/O worth speaking of: the sandbox is booted later, by `trial.run()`, outside the lock.
            # Harbor's wrappers read `os.environ` while constructing the agent, so the seam's
            # process-level vars have to be in place across `Trial.create` and are put back after.
            seam = seams.get(harness, profile=harness_profile)
            *_, proc_env = seam.resolve(
                base_url=intercept_url, session=session.session_id, model=model
            )
            if not proc_env:
                config = build_trial_config(**config_kwargs)
                trial = await Trial.create(config)
                trial_result = await trial.run()
            else:
                # The override has to span `run()`, not just construction. Harbor's goose wrapper reads
                # `os.environ["OPENAI_API_KEY"]` inside its own `run` (goose.py:653), long after
                # construction is done — so restoring the env before `run()` handed it the operator's REAL
                # provider key instead of the capture session id, and our own proxy correctly answered
                #
                #     401 unknown API key; register a session via POST /sessions
                #
                # on every call. It failed identically against all five upstreams in a compatibility
                # matrix, which is what identified it as the key rather than any endpoint.
                #
                # The cost is real and unavoidable: `os.environ` is process-global, so two concurrent
                # rollouts of a proc-env seam genuinely cannot each have their own key. Those rollouts
                # serialise. Seams that carry no `proc_env` — the large majority, which pass credentials
                # through `agent_env` — take neither the lock nor the wait.
                #
                # Acquired via `to_thread` so a blocking `acquire()` cannot stall the event loop while
                # another rollout holds the lock for the length of a full trial. A bare `acquire()` here
                # would deadlock the server the first time two goose rollouts overlapped on one loop.
                from .proc_env_context import (
                    install as install_ctx_env,
                    overlay as ctx_env_overlay,
                )

                if install_ctx_env():
                    # No lock. `os.environ` reads are context-local now, so concurrent rollouts of a
                    # credential-by-env harness each see their OWN session key from the same expression.
                    # This is what stops claude-code, gemini-cli and goose serialising behind one another.
                    with ctx_env_overlay(proc_env):
                        config = build_trial_config(**config_kwargs)
                        trial = await Trial.create(config)
                        trial_result = await trial.run()
                else:
                    # Fallback: mutate the real environment under a lock, one rollout at a time.
                    await _acquire_proc_env()
                    try:
                        with process_env(seam.name, proc_env):
                            config = build_trial_config(**config_kwargs)
                            trial = await Trial.create(config)
                            trial_result = await trial.run()
                    finally:
                        _PROC_ENV_LOCK.release()
        except Exception as exc:  # noqa: BLE001 - a rollout failure is a RESULT, never an exception
            result.ok = False
            result.error = str(exc)[:600]
            result.exception_type = type(exc).__name__

        result.wall_s = round(time.monotonic() - started, 2)

        # --- reward, forwarded verbatim -------------------------------------
        if trial_result is not None:
            verifier = getattr(trial_result, "verifier_result", None)
            rewards = dict(getattr(verifier, "rewards", None) or {}) if verifier else {}
            # Inside a try, and finite-checked. A verifier is task-supplied code that can put anything in
            # this dict, and this line sits between the trial's try/except and `_pick_reward`'s, so a
            # value like None, "N/A" or a nested dict used to raise straight out of `run_rollout`, through
            # the MCP tool, and into the trainer — which is the every-rank-hangs-on-the-NCCL-barrier
            # failure this whole HTTP boundary exists to make impossible.
            #
            # inf and nan are rejected too, though they coerce fine: inf reads as solved downstream, and
            # nan poisons any average computed over a batch of rewards.
            #
            # PER KEY, not all-or-nothing. A dict comprehension inside one try meant a single unusable
            # key discarded EVERY key and failed the whole rollout -- so a suite emitting
            # `tool_efficiency: null` alongside a perfectly good `correctness` lost the correctness too.
            # That is not hypothetical: it dropped 86 of 250 tasks from one scored run while the summary
            # reported clean numbers over the remaining two thirds. An unmeasured key is an EXCLUSION,
            # never a zero, so the key is dropped and named in `findings` rather than coerced to 0.0.
            # The rollout only fails if the key actually being trained on is the unusable one, which
            # `_pick_reward` decides below.
            unusable: list[str] = []
            usable: dict[str, float] = {}
            for key, value in rewards.items():
                try:
                    usable[key] = _finite_reward(key, value)
                except (TypeError, ValueError) as exc:
                    unusable.append(f"{key}={value!r} ({exc})")
            result.rewards = usable
            if unusable:
                result.findings.append(
                    "[WARN] verifier keys dropped as unusable: " + "; ".join(unusable)
                )
            try:
                result.reward, result.reward_key = _pick_reward(
                    result.rewards, reward_key
                )
            except ValueError as exc:
                if require_reward:
                    result.ok = False
                    result.error = str(exc)
                else:
                    # A rollout to look at, not to train on: every reward stays in `rewards`.
                    result.findings.append(f"[WARN] no headline reward: {exc}")
            for step in getattr(trial_result, "step_results", None) or []:
                step_rewards = dict(
                    getattr(getattr(step, "verifier_result", None), "rewards", None)
                    or {}
                )
                try:
                    coerced = {k: _finite_reward(k, v) for k, v in step_rewards.items()}
                except (TypeError, ValueError):
                    # A bad per-step reward is worth dropping, not worth failing the rollout: the
                    # headline reward above is what trains, and gating on it is already handled.
                    coerced = {}
                result.step_results.append(
                    HarborStepResult(
                        name=getattr(step, "name", "") or "",
                        rewards=coerced,
                    )
                )
            info = getattr(trial_result, "exception_info", None)
            if info is not None and result.error is None:
                result.ok = False
                result.exception_type = getattr(info, "exception_type", None)
                result.error = str(getattr(info, "exception_message", ""))[:600]

        # --- capture --------------------------------------------------------
        try:
            # `include_messages` is what puts the assistant's own output in the result.
            # Only the response side is kept downstream (see `turns_from_document`), so
            # the payload grows by the completion text, not by the whole conversation.
            document = export_session(
                session, include_messages=True, capture_level=capture_level
            )
            stats = document.get("stats", {})
            result.n_turns = stats.get("n_turns", 0)
            result.n_roots = stats.get("n_roots", 0)
            result.budget_stop_count = document.get("budget_stop_count", 0)
            # Counted over the DEDUPED turns, not over sequences.
            #
            # `stats["n_trainable_tokens"]` sums `n_trainable` per sequence, and forked paths share their
            # prefix — so a node reached by several sequences is counted once per sequence. `turns` (below)
            # deliberately emits each node once, because a duplicated row is the same model call credited
            # twice and quietly doubles its weight in a gradient. Reporting the sequence-wise sum next to
            # the deduped turns meant the two disagreed: measured on a forked gemini-cli rollout, 6845
            # reported against 6201 actually present, and 32950 against 31720 on a longer one.
            #
            # A consumer comparing this field against the turns beside it has to find them consistent, so
            # this is the deduped figure. The sequence-wise total is still in `stats` for a consumer that
            # trains on `sequences` instead, where per-sequence counting is the correct reading.
            # Set once `result.turns` exists — see below, where it is filled in from the document after
            # aux masking. Reading it here made the total always 0, because `turns` is still empty at this
            # point in the function.
            result.multi_turn = result.n_turns > result.n_roots
            result.findings.extend(
                f for f in document.get("validation", []) if not f.startswith("[INFO]")
            )
            # Sequence-level findings were never surfaced. `check_sequence` FATALs — a positive logprob, a
            # length mismatch, nothing trainable — live on the row rather than in the document's own
            # validation list, so a rollout carrying one of them came back with a clean `findings` list and
            # `ok=True`. The row is already marked untrainable; this makes the reason visible.
            for row in document.get("sequences", []):
                for finding in row.get("validation", []):
                    if not finding.startswith("[INFO]"):
                        result.findings.append(
                            f"[sequence {row.get('root_id', '?')}] {finding}"
                        )

            trial_dir = _trial_dir(trial_result, trials_dir, trial_name)
            # Not every harness writes ATIF. Three of the sixteen (hermes, openclaw, pi) emit no
            # `trajectory.json`, which left them with no independent check at all. pi does record the
            # same information in its own session log, so `load_trace` falls back to that; hermes writes
            # a zero-byte file and openclaw only echoes back its config, so for those two there is
            # genuinely nothing to compare against and `none` is the honest answer.
            atif, trace_source = load_trace(trial_dir) if trial_dir else (None, "")
            report = reconcile(document, atif)
            result.atif = (
                "none" if atif is None else ("match" if report.ok else "MISMATCH")
            )
            result.trace_source = trace_source
            result.findings += [
                str(f) for f in report.findings if not str(f).startswith("[INFO]")
            ]

            # Calls the harness's own trace does not count as agent steps are auxiliary; drop them so
            # they cannot be credited with the reward earned by solving the task.
            if report.aux_node_ids:
                aux = set(report.aux_node_ids)
                for sequence in document["sequences"]:
                    if sequence["role"] != "agent":
                        continue
                    nodes = set(sequence["node_ids"])
                    if nodes <= aux:
                        sequence["role"] = "auxiliary"
                    elif nodes & aux:
                        # Detection is per node; demotion was per sequence, so a sequence MIXING an
                        # auxiliary node with real agent turns stayed `agent` in full and shipped the aux
                        # node as a training turn credited with the task's reward. That worked only under
                        # the unstated assumption that aux calls always form their own single-node root,
                        # and it erred unsafe when they did not.
                        #
                        # Masked at token level rather than dropped, because the rest of the sequence is
                        # genuine agent work: the aux node's sampled tokens stop being targets while
                        # remaining context, which is exactly how `sequence_for` treats a turn it cannot
                        # trust. Anything else would either discard real turns or train on an aux call.
                        _mask_out_nodes(document, sequence, nodes & aux)

            result.turns = turns_from_document(document)
            # Counted over the DEDUPED turns, not over sequences.
            #
            # `stats["n_trainable_tokens"]` sums `n_trainable` per sequence, and forked paths share their
            # prefix — so a node reached by several sequences is counted once per sequence, while `turns`
            # deliberately emits each node once (a duplicated row is the same model call credited twice,
            # which quietly doubles its weight in a gradient). Reporting the sequence-wise sum beside the
            # deduped turns meant the two disagreed: 6845 reported against 6201 present on one forked
            # rollout, 32950 against 31720 on a longer one.
            #
            # It has to be computed HERE rather than beside the other stats, because `turns` does not
            # exist until this line. The sequence-wise total stays in `stats` for a consumer that trains
            # on `sequences`, where counting per sequence is the correct reading.
            result.n_trainable_tokens = sum(
                sum(t.loss_mask)
                if t.loss_mask is not None
                else len(t.completion_token_ids)
                for t in result.turns
                if t.trainable and not t.discarded
            )
            # Every conversation, not only the trainable ones: an auxiliary call that
            # went wrong is exactly what someone reading a bad rollout needs to see.
            result.conversations = conversations_from_document(document)
            if not report.ok:
                result.ok = False
                result.error = result.error or "capture failed validation"

            # A FATAL from the document's own validation means the capture is unusable, and until now it
            # was recorded in `findings` without touching `ok`. Only the trace-reconciliation report
            # could set `ok`, so a rollout that reached the verifier but produced NO model calls came
            # back `ok=True` with zero trainable tokens, and reconciliation agreed because both sides
            # were empty. One such rollout even carried reward=1.0, which is the worst shape available:
            # a trainer filtering on `ok` would accept a row with nothing in it and a positive reward.
            fatal = [f for f in result.findings if "[FATAL" in f]
            if fatal:
                result.ok = False
                result.error = result.error or fatal[0]

            # A train rollout that produced no trainable sequence is a failed train rollout, however
            # healthy it looks: every row was masked out, or auxiliary, or discarded. Checked only on the
            # train path, since having nothing trainable is the defining property of an eval rollout.
            if (
                result.rollout_type == "train"
                and result.ok
                and not document.get("trainable")
            ):
                result.ok = False
                result.error = (
                    result.error
                    or "no trainable sequence survived: every captured path was masked out, "
                    "auxiliary or discarded"
                )
        except Exception as exc:  # noqa: BLE001
            result.ok = False
            result.error = (
                result.error or f"capture export failed: {type(exc).__name__}: {exc}"
            )

        # Read after the rollout, not before: the fixes are discovered from the provider's own 400s as
        # calls are made. A rewritten request is a changed experiment — dropping `temperature` alters the
        # sampling distribution — so it travels with the result rather than living only in a log.
        if inference is not None:
            result.param_fixes = [str(f) for f in getattr(inference, "param_fixes", [])]

        # A rollout that produced no reward is not a zero: the verifier never ran. Keeping the two
        # distinct is what stops a dead sandbox being scored as a wrong answer.
        if (
            result.ok
            and result.reward is None
            and not result.rewards  # several, none the headline, is a warning of its own above
            and trial_result is not None
        ):
            result.findings.append(
                "[WARN] ungraded: the verifier produced no reward for this trial"
            )
        return result
    finally:
        registry.delete(session.session_id)


def _mask_out_nodes(
    document: dict[str, Any], sequence: dict[str, Any], node_ids: set[str]
) -> None:
    """Zero the loss mask over the given nodes' sampled spans, in place.

    Their tokens stay in `input_ids` as context — the model did condition on them — but stop being
    targets, and the sequence's trainable count is corrected to match.
    """
    by_node = {t["node_id"]: t for t in document.get("turns", [])}
    mask = sequence.get("loss_mask")
    if not mask:
        return
    for node_id in sequence["node_ids"]:
        if node_id not in node_ids:
            continue
        node = by_node.get(node_id, {})
        n_prompt = int(node.get("n_prompt", 0))
        n_sampled = int(node.get("n_sampled", 0))
        # `n_prompt` IS the sequence-coordinate start of this turn's sampled span — no running offset
        # is needed, and tracking one was wrong.
        #
        # A node's `prompt_ids` is the whole conversation prefix, and `sequence_for` lays a child out as
        # (interstitial context, sampled) where the context is `prompt_ids[len(parent.end_ids):]`. The
        # two cancel: cumulative-before-sampled for turn k is
        #     n_prompt(k-1) + n_sampled(k-1) + (n_prompt(k) - n_prompt(k-1) - n_sampled(k-1)) = n_prompt(k)
        # The first version advanced an offset as if each turn were only prompt-plus-sampled, so from
        # the second turn on `start` landed on the interstitial context: it zeroed positions that were
        # already 0 and left the aux node's real completion tokens at mask 1, which meant this function
        # silently did nothing and aux calls kept being credited with the task's reward.
        #
        # `turns_from_document` slices the same way, so the two cannot disagree.
        for i in range(n_prompt, min(n_prompt + n_sampled, len(mask))):
            mask[i] = 0
    sequence["n_trainable"] = sum(mask)
    sequence["trainable"] = bool(sequence["n_trainable"]) and sequence.get(
        "trainable", True
    )


def _trial_dir(
    trial_result: Any, trials_dir: Path | str, trial_name: str
) -> Path | None:
    """Where Harbor wrote this trial's artifacts, including its trajectory."""
    uri = getattr(trial_result, "trial_uri", None) if trial_result is not None else None
    if uri:
        # `as_uri()` percent-encodes, so a space or `+` in the path has to be decoded back.
        return Path(url2pathname(urlparse(str(uri)).path))
    candidate = Path(str(trials_dir)) / trial_name
    return candidate if candidate.is_dir() else None


__all__ = ["run_rollout", "build_trial_config", "HarborRolloutResult", "HarborTurn"]
