"""Wire types for harbor_env.

Two shapes matter. `HarborTaskRef` is what the Task API hands out during discovery, one per dataset
item. `HarborRolloutResult` is what one `run_rollout` returns: the reward, and enough token detail to
train on.

Everything here is JSON-serialisable by construction — `run_rollout` returns
`result.model_dump_json()` and the client re-validates. There is no shared memory between server and
client.
"""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import State
from pydantic import BaseModel, Field


class HarborTaskRef(BaseModel):
    """One task, as returned by the Task API. What a trainer's dataset holds per row."""

    index: int
    task_id: str
    task_name: str
    dataset: str = ""
    instruction: str = ""


class HarborTurn(BaseModel):
    """One model call, captured exactly. The unit a trainer consumes.

    `prompt_token_ids` is the engine's own tokenisation of everything before this turn, not a local
    re-render. That distinction is the whole point of the capture layer: re-tokenising a prompt
    offline drifts from what the model actually saw (measured at 0/6 exact on Qwen3.5 until thinking
    was disabled), and a drifted prompt silently forks a long conversation into short fragments.
    """

    turn: int
    role: str = "agent"
    finish_reason: str | None = None
    prompt_token_ids: list[int] = Field(default_factory=list)
    completion_token_ids: list[int] = Field(default_factory=list)
    per_token_logps: list[float] = Field(default_factory=list)
    # Full prompt+completion mask; None supports older serialized Harbor results.
    loss_mask: list[int] | None = None
    node_id: str = ""
    n_tools: int = 0
    discarded: bool = False
    # Whether THIS turn's logprobs may be trained on. False when ingest rejected them (`check_turn`
    # drops `sampled_logprobs` but keeps the sampled ids, so the tokens remain valid context for later
    # turns) — after which `sequence_for` masks the turn out and zero-fills its logprobs. Without this
    # flag those zeros are indistinguishable from a genuine logprob of 0.0, i.e. a p=1.0 token, and a
    # per-turn trainer would take them at face value.
    trainable: bool = True
    # Submitted inference policy, after overrides and compatibility edits. A processed logprob
    # includes these transformations. Preserve the harness's original request separately.
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    requested_sampling_params: dict[str, Any] = Field(default_factory=dict)

    # What the model actually produced, in readable form. Only the assistant's own output is kept,
    # never the prompt side: the prompt is already present as token ids and repeating it as text
    # would roughly double the payload for no new information. This is what makes a result
    # inspectable without a tokenizer, and it is what a reward function keys on when it needs to
    # know which tool was called rather than how many tokens were spent.
    text: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    # What the harness ASKED for on this turn. Carried because a consumer that wants TRL's
    # `TraceEntry` shape needs `request.messages` and `request.tools`, and without them the wire
    # result cannot be converted at all — the token fields alone do not say what prompt produced
    # them. It also makes retokenization skew measurable against `prompt_token_ids` above, which is
    # the only way to know whether re-rendering a prompt locally is lossless for a model + harness
    # pair rather than assuming it either way.
    request_messages: list[dict[str, Any]] = Field(default_factory=list)
    request_tools: list[dict[str, Any]] | None = None


class HarborConversation(BaseModel):
    """One complete conversation from a rollout, exactly as the harness assembled it.

    A rollout can contain several: a root is a conversation that started from a fresh prompt, so
    subagents and auxiliary calls each get their own. `messages` is the full list including the
    system prompt and every tool result, which is what makes a finished rollout readable rather than
    a column of token counts.
    """

    root_id: str = ""
    role: str = "agent"  # agent | auxiliary | discarded
    n_turns: int = 0
    messages: list[dict[str, Any]] = Field(default_factory=list)


class HarborStepResult(BaseModel):
    """One step of a multi-step task. Harbor gates progression on `min_reward`."""

    name: str = ""
    rewards: dict[str, float] = Field(default_factory=dict)
    passed: bool = True


class HarborRolloutResult(BaseModel):
    """Everything one rollout produced.

    A failed rollout is still a valid result: `ok=False`, `error` set, `reward=None`. Nothing raises
    across the server boundary, because a rollout exception reaching a trainer is what hangs every
    rank at the NCCL barrier forever.
    """

    # identity
    task_id: str = ""
    task_name: str = ""
    dataset: str = ""
    harness: str = ""
    sandbox: str = ""
    trial_name: str = ""
    session_id: str = ""

    # outcome — forwarded from Harbor's verifier, never recomputed here
    reward: float | None = None
    rewards: dict[str, float] = Field(default_factory=dict)
    reward_key: str = ""
    step_results: list[HarborStepResult] = Field(default_factory=list)

    # What kind of rollout this is, and why. `train` carries the token fields below; `eval` carries
    # everything except them. This is not a flag anyone sets: it is decided by what the inference
    # endpoint could return, probed before the server started, and it travels with the result so that
    # no consumer has to infer trainability from the emptiness of a list.
    rollout_type: str = "train"
    capture_level: str = "tokens"
    # Params the upstream rejected and how the proxy worked around them. A dropped `temperature`
    # changes the sampling distribution, which makes an eval number irreproducible if unrecorded.
    param_fixes: list[str] = Field(default_factory=list)

    @property
    def trainable(self) -> bool:
        return self.rollout_type == "train"

    # capture
    turns: list[HarborTurn] = Field(default_factory=list)
    conversations: list[HarborConversation] = Field(default_factory=list)
    n_turns: int = 0
    n_roots: int = 0
    # Terminal responses emitted by the proxy; never model tokens or training turns.
    budget_stop_count: int = 0
    n_trainable_tokens: int = 0
    multi_turn: bool = False
    atif: str = "none"
    # Which independent record the capture was checked against: `atif` for a harness trajectory,
    # a reader name (e.g. `pi_session`) when the harness records the same thing under another
    # format, or empty when it records nothing comparable.
    trace_source: str = ""
    findings: list[str] = Field(default_factory=list)

    # timings and diagnostics. There is no metrics endpoint and no structured logging in the env
    # server, so observability has to ride back inside the payload or it does not exist.
    wall_s: float = 0.0
    phase_timings: dict[str, float] = Field(default_factory=dict)
    agent_log_tail: str = ""

    # failure
    ok: bool = True
    error: str | None = None
    exception_type: str | None = None

    @property
    def solved(self) -> bool:
        """Graded AND positive. `reward is None` means the verifier never ran, which is not a zero."""
        return self.reward is not None and self.reward > 0


class HarborState(State):
    """Per-session counters. Mutated inside the tool, since `step` only dispatches."""

    rollouts_completed: int = 0
    last_reward: float | None = None
    last_task_id: str | None = None
    last_trial_name: str | None = None
    llm_url: str = ""
    intercept_url: str = ""


def _assistant_text(response: dict[str, Any]) -> str:
    """The assistant's own words, flattened across the shapes the four dialects produce."""
    content = response.get("content")
    if isinstance(content, list):  # anthropic / responses send block lists
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("text")
        )
    return content if isinstance(content, str) else ""


def _tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Tool calls as `{name, arguments}`, normalised across dialects.

    Kept as data rather than a rendered string: a reward function that wants to check which tool ran
    should not have to parse a display format.
    """
    out: list[dict[str, Any]] = []
    for call in response.get("tool_calls") or []:
        function = call.get("function") or {}
        name = function.get("name") or call.get("name")
        if not name:
            continue
        out.append(
            {
                "name": str(name),
                "arguments": function.get("arguments", call.get("arguments", "")),
            }
        )
    # Anthropic does not use `tool_calls`: it puts tool use in the content block list. Reading only
    # the chat-completions shape leaves claude-code's actions out of the result entirely, so
    # `contract.json` and the rendered conversation both show it as a stream of text that did
    # nothing.
    content = response.get("content")
    if isinstance(content, list):
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name")
            ):
                out.append(
                    {"name": str(block["name"]), "arguments": block.get("input", "")}
                )
    return out


def conversations_from_document(document: dict[str, Any]) -> list[HarborConversation]:
    """Rebuild the full conversations, system prompt and tool results included.

    The deepest node of a chain already carries the whole conversation in `request_messages`, since
    each call replays everything before it. So the last node per root plus its own response is the
    complete transcript, with no stitching and no risk of drifting from what was actually sent.
    """
    by_node = {t["node_id"]: t for t in document.get("turns", [])}
    # One per root, not one per sequence. A fork produces several paths through the same root, and
    # each replays the same conversation up to the branch point, so emitting one per sequence shows
    # the reader near-identical transcripts and calls both of them the main conversation. The
    # longest path is the complete one.
    best: dict[str, HarborConversation] = {}

    for sequence in document.get("sequences", []):
        node_ids = sequence.get("node_ids") or []
        if not node_ids:
            continue
        last = by_node.get(node_ids[-1], {})
        messages = list(last.get("request_messages") or [])
        response = last.get("response_message") or {}
        if response:
            messages.append({**response, "role": response.get("role", "assistant")})
        if not messages:
            continue
        root_id = str(sequence.get("root_id", "")) or node_ids[0]
        candidate = HarborConversation(
            root_id=root_id,
            role=str(sequence.get("role", "agent")),
            n_turns=int(sequence.get("n_turns", len(node_ids))),
            messages=messages,
        )
        current = best.get(root_id)
        if current is None or len(candidate.messages) > len(current.messages):
            best[root_id] = candidate
    return list(best.values())


def turns_from_document(document: dict[str, Any]) -> list[HarborTurn]:
    """Flatten a capture document into per-turn training rows.

    Only `agent` sequences become turns. Auxiliary calls are dropped here rather than marked,
    because they are not the agent working on the task and must never carry its reward — a
    next-speaker classification credited with solving a task is a reward-hacking gift.
    """
    by_node = {t["node_id"]: t for t in document.get("turns", [])}
    rows: list[HarborTurn] = []
    # Forked paths share their prefix, and each live path is exported as its own sequence, so the
    # same node appears in more than one of them. Emit each node once: a duplicated row is the same
    # model call credited twice, which quietly doubles its weight in the gradient.
    seen: dict[str, list[int]] = {}
    index = 0

    for sequence in document.get("sequences", []):
        if sequence.get("role") != "agent":
            continue
        input_ids = sequence["input_ids"]
        logprobs = sequence["logprobs"]
        for node_id in sequence["node_ids"]:
            node = by_node.get(node_id, {})
            response = node.get("response_message") or {}

            # Each turn's own counts, not a walk over runs of the loss mask. A sequence is built as
            # (context, sampled) per node, so the cumulative offset where a turn's sampled tokens
            # begin is exactly the length of that turn's prompt, which the document already records.
            #
            # The previous version zipped `node_ids` against `turn_lengths`, where `turn_lengths`
            # counts runs of mask-1. A turn whose logprobs were missing contributes mask-0 and so no
            # run at all, which made the two lists different lengths: `zip` then stopped early and
            # every turn after the bad one was dropped or attributed to the wrong node. Turns with
            # no context between them merged into one run for the same reason.
            n_prompt = int(node.get("n_prompt", 0))
            n_sampled = int(node.get("n_sampled", 0))
            end = n_prompt + n_sampled

            # Reconciliation can mask only part of a completion. Preserve each bit; `trainable`
            # means some supervision remains, and cannot replace the per-token mask.
            mask = sequence.get("loss_mask") or []
            span = mask[n_prompt:end]
            if (
                n_prompt < 0
                or n_sampled < 0
                or end > len(input_ids)
                or len(span) != n_sampled
            ):
                raise ValueError(f"incomplete token or mask span for node {node_id}")
            if node_id in seen:
                if seen[node_id] != span:
                    raise ValueError(
                        f"inconsistent completion masks for shared node {node_id}"
                    )
                continue
            seen[node_id] = list(span)
            trainable = bool(span) and any(m == 1 for m in span)

            rows.append(
                HarborTurn(
                    turn=index,
                    node_id=node_id,
                    finish_reason=node.get("finish_reason"),
                    # Every turn, not just the first. This is the engine's own tokenisation of
                    # everything the model saw before it generated, which is what the training
                    # contract promises and what a per-turn trainer consumes.
                    prompt_token_ids=input_ids[:n_prompt],
                    completion_token_ids=input_ids[n_prompt:end],
                    # Straight off the node, which already recorded exactly what was sent upstream.
                    request_messages=list(node.get("request_messages") or []),
                    request_tools=node.get("request_tools"),
                    per_token_logps=node.get("sampled_logprobs")
                    or logprobs[n_prompt:end],
                    loss_mask=[0] * n_prompt + list(span),
                    n_tools=node.get("n_tools", 0),
                    discarded=bool(node.get("discarded")),
                    trainable=trainable,
                    sampling_params=node.get("sampling_params") or {},
                    requested_sampling_params=node.get("requested_sampling_params")
                    or {},
                    text=_assistant_text(response),
                    tool_calls=_tool_calls(response),
                )
            )
            index += 1
    return rows
