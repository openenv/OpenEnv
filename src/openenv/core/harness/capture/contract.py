"""What a rollout hands a trainer.

The typed training contract is `to_training_trace`: one engine prompt, sampled completion,
behavior logprobs, and a loss mask over prompt+completion per retained model call. Prompt ids and
sampled ids are never reconstructed from text. Masks are authoritative after reconciliation.

`to_turn_records` provides the older three-array tuple for fully supervised completions. It skips
fully masked turns and rejects partial masks because a tuple cannot represent their eligibility.

`measure_retokenization_skew` stays, not as a cost estimate for a re-render you were going to do
anyway, but as the measurement that catches a producer which has quietly stopped emitting ids.

WHAT THIS MODULE CANNOT FILL IN. `TraceEntry` also declares `reward`, and nothing here emits it: a
capture proxy sees model calls, not task outcomes, and inventing a number would be worse than
omitting one. An ENVIRONMENT fills it in from its own `verify()`. `TraceEntry` is `total=False`, so
read it with `.get("reward")` — indexing it raises on every entry this module produces.
"""

from __future__ import annotations

import logging
from typing import Any

from ..training import TrainingTrace
from .graph import RolloutGraph, TurnNode
from .validate import check_turn, validate_training_turn

logger = logging.getLogger(__name__)

# Control text emitted by the proxy, never sampled or eligible for training.
BUDGET_STOP_MESSAGE = "Step budget exhausted; stopping."


def _usable(node: TurnNode) -> bool:
    """Whether this turn's sampled ids and logprobs describe the same tokens.

    The same guard `sequence_for` applies before masking a turn in, applied here too. These adapters
    walk the graph directly, so they used to emit `completion_token_ids` with every sampled id and
    `per_token_logps` as `[]` whenever ingest had rejected the logprobs — an unequal pair with no
    assertion anywhere, which a consumer zipping the two silently misattributes across every token of
    the turn.

    A turn that fails this is skipped rather than emitted short, and the caller logs how many, because
    `n_trainable_tokens` already excludes them and dropping them keeps the two counts consistent.
    """
    logprobs = node.sampled_logprobs
    return (
        bool(node.sampled_ids)
        and bool(logprobs)
        and len(logprobs) == len(node.sampled_ids)
        and check_turn(node.prompt_ids, node.sampled_ids, logprobs).ok
    )


def _require_trainable(document: dict[str, Any]) -> None:
    """Refuse to build a training contract out of an eval rollout.

    An eval document has an empty `sequences` list, so every converter here would return `[]` and the
    caller would receive a well-formed, empty contract — the exact silent-nothing failure this whole
    layer exists to prevent. Raising is the only honest answer: the data a trainer is asking for was
    never captured, and no amount of downstream care can reconstruct it.

    Raises:
        ValueError: If the document came from an endpoint that could not return token ids.
    """
    if document.get("rollout_type", "train") == "eval":
        level = document.get("capture_level") or "unknown"
        raise ValueError(
            f"this is an EVAL rollout (capture_level={level!r}): it carries the reward and the full "
            "trace, but no token ids or logprobs, so there is no training contract to build. Point "
            "the capture proxy at vLLM (--return-tokens-as-token-ids --logprobs-mode "
            "processed_logprobs) or SGLang built from main to get trainable rollouts."
        )


def _warn_skipped(nodes: list[TurnNode]) -> None:
    """Say out loud how many turns are being left out, and why."""
    skipped = [n.node_id for n in nodes if not _usable(n)]
    if skipped:
        logger.warning(
            "omitting %d of %d agent turn(s) from the training contract: their logprobs were "
            "rejected on ingest, so ids and logprobs do not describe the same tokens (%s)",
            len(skipped),
            len(nodes),
            ", ".join(skipped[:5]),
        )


def _agent_nodes(graph: RolloutGraph, document: dict[str, Any]) -> list[TurnNode]:
    """Every agent turn, in arrival order, excluding auxiliary calls and discarded retries.

    All agent sequences, not just the first. A rollout can have several: a harness that rewrites its
    system prompt mid-run starts a new root, and a fork produces several paths that share a prefix.
    Taking `agent_rows[0]` dropped the rest, so `to_turn_records` and `to_trace_entries` silently
    returned part of the rollout while reporting nothing wrong.

    Nodes are deduplicated because forked paths share their common prefix, and ordered by arrival so
    a turn's position matches the order the model produced it in.
    """
    keep: set[str] = set()
    for row in document["sequences"]:
        if row["role"] == "agent":
            keep.update(row["node_ids"])
    if not keep:
        return []
    return [n for n in graph.nodes() if n.node_id in keep]


def to_trace_entries(
    graph: RolloutGraph, document: dict[str, Any]
) -> list[dict[str, Any]]:
    """Rollout graph -> `list[TraceEntry]`, carrying the ENGINE's own prompt tokenization.

    Every entry includes `prompt_token_ids` and `loss_mask`, so a consumer never re-tokenizes. This
    used to emit neither, on the reasoning that a consumer could re-derive the prompt from
    `request`. It cannot: measured on Qwen3.5-4B over 28 live turns, a local re-render matched the
    engine on ZERO of them, and training on the difference collapsed a run at its first weight
    update. The ids were always here in `node.prompt_ids`; they were simply dropped at this
    boundary.

    `loss_mask` spans `prompt_token_ids + completion_token_ids`: 0 across the prompt (context) and 1
    at eligible sampled positions. Zero bits from reconciliation remain zero, including partial
    completion masks; eligibility is not inferable downstream.

    Auxiliary roots and discarded retries are already excluded here, so the caller does not need an
    `agent_turn_fn`. That hook exists because a flat trace cannot tell an aux call from an agent
    turn; a graph can, structurally.

    Raises:
        ValueError: If `document` is an eval rollout. See `_require_trainable`.
    """
    _require_trainable(document)
    entries = []
    nodes = _agent_nodes(graph, document)
    masks = _completion_masks(graph, document)
    _warn_skipped(nodes)
    for node in nodes:
        if not _usable(node):
            continue
        mask = [0] * len(node.prompt_ids) + masks[node.node_id]
        validate_training_turn(
            node.prompt_ids, node.sampled_ids, node.sampled_logprobs, mask
        )
        entries.append(
            {
                "request": {
                    "messages": node.request_messages,
                    "tools": node.request_tools,
                },
                "response": {
                    "choices": [
                        {
                            "message": node.response_message,
                            "finish_reason": node.finish_reason,
                        }
                    ]
                },
                "prompt_token_ids": node.prompt_ids,
                "completion_token_ids": node.sampled_ids,
                "per_token_logps": node.sampled_logprobs or [],
                # Deterministic here because `_usable` already skipped every turn whose logprobs
                # were rejected on ingest -- so anything that reaches this line is fully trainable,
                # and the mask is simply context across the prompt, train across the sample. The
                # field is emitted anyway rather than left for the consumer to synthesise: a
                # consumer that assumes "all sampled tokens are trainable" is right only because of
                # a filter it cannot see from here.
                "loss_mask": mask,
                "metadata": {
                    "node_id": node.node_id,
                    "index": node.index,
                    "model": node.model,
                    "n_tools": node.n_tools,
                    "harness_session_id": node.harness_session_id,
                    "sampling_params": dict(node.sampling_params),
                    "requested_sampling_params": dict(node.requested_sampling_params),
                },
            }
        )
    return entries


def _completion_masks(
    graph: RolloutGraph, document: dict[str, Any]
) -> dict[str, list[int]]:
    """Read authoritative masks after reconciliation, deduplicating shared nodes."""
    masks: dict[str, list[int]] = {}
    for row in document["sequences"]:
        if row["role"] != "agent":
            continue
        mask = row.get("loss_mask")
        if mask is None:
            # Older callers supply only graph membership; use the graph's validated mask.
            mask = graph.sequence_for(row["node_ids"][-1]).loss_mask
        for node_id in row["node_ids"]:
            node = graph.get(node_id)
            start = len(node.prompt_ids)
            span = list(mask[start : start + len(node.sampled_ids)])
            if len(span) != len(node.sampled_ids):
                raise ValueError(f"incomplete completion mask for node {node_id}")
            if node_id in masks and masks[node_id] != span:
                raise ValueError(
                    f"inconsistent completion masks for shared node {node_id}"
                )
            masks[node_id] = span
    return masks


def to_training_trace(graph: RolloutGraph, document: dict[str, Any]) -> TrainingTrace:
    """Return validated agent calls once each, including shared graph prefixes."""
    if any("[FATAL" in finding for finding in document.get("validation", [])):
        raise ValueError("cannot train a capture with fatal validation findings")
    _require_trainable(document)
    if any(not _usable(node) for node in _agent_nodes(graph, document)):
        raise ValueError("agent capture has missing or invalid engine tokens/logprobs")
    return TrainingTrace.from_entries(to_trace_entries(graph, document))


def to_turn_records(
    graph: RolloutGraph, document: dict[str, Any]
) -> list[tuple[list[int], list[int], list[float]]]:
    """Rollout graph -> `(prompt_ids, output_ids, output_log_probs)` per turn, losslessly.

    For fully supervised completions only. Fully masked calls are skipped; partial masks raise
    rather than silently restoring supervision. Use `to_trace_entries` for the mask-aware contract.

    Raises:
        ValueError: If `document` is an eval rollout. See `_require_trainable`.
    """
    _require_trainable(document)
    nodes = _agent_nodes(graph, document)
    _warn_skipped(nodes)
    masks = _completion_masks(graph, document)
    if any(any(mask) and not all(mask) for mask in masks.values()):
        raise ValueError(
            "to_turn_records cannot represent partial masks; use to_trace_entries"
        )
    return [
        (node.prompt_ids, node.sampled_ids, node.sampled_logprobs or [])
        for node in nodes
        if _usable(node) and any(masks[node.node_id])
    ]


def measure_retokenization_skew(
    graph: RolloutGraph,
    document: dict[str, Any],
    tokenizer,
    *,
    chat_template: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare TRL's re-tokenized prompt against the engine's, per turn. The number nobody has had.

    Reproduces `_turns_from_trace` exactly, including the `_decode_tool_call_arguments` step (the
    trace stores tool-call `arguments` as a JSON string; XML-style templates such as Qwen3.5's
    iterate it and raise on a string).

    Returns per-turn exact-match, common-prefix length and length delta. `exact_match_frac == 1.0`
    means re-tokenization is safe for this model+harness pair and the hook buys nothing. Anything
    less means TRL is training on a prompt the model never saw, and `_chain_to_sequences` will fork
    the conversation at the first divergence.
    """
    import json as _json

    def decode_arguments(messages):
        out = []
        for message in messages:
            calls = message.get("tool_calls")
            if not calls:
                out.append(message)
                continue
            new = []
            for call in calls:
                function = call.get("function")
                arguments = (function or call).get("arguments")
                if not isinstance(arguments, str):
                    new.append(call)
                    continue
                try:
                    arguments = _json.loads(arguments)
                except _json.JSONDecodeError:
                    arguments = {}
                new.append(
                    {**call, "function": {**function, "arguments": arguments}}
                    if function
                    else {**call, "arguments": arguments}
                )
            out.append({**message, "tool_calls": new})
        return out

    def prefix_len(a, b):
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    turns = []
    for i, node in enumerate(_agent_nodes(graph, document)):
        try:
            rebuilt = tokenizer.apply_chat_template(
                decode_arguments(node.request_messages),
                tools=node.request_tools,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                chat_template=chat_template,
                **(chat_template_kwargs or {}),
            )
        except Exception as exc:  # noqa: BLE001 - a template that raises IS the finding
            turns.append(
                {
                    "turn": i,
                    "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                    "engine_len": len(node.prompt_ids),
                }
            )
            continue
        turns.append(
            {
                "turn": i,
                "engine_len": len(node.prompt_ids),
                "rebuilt_len": len(rebuilt),
                "delta": len(rebuilt) - len(node.prompt_ids),
                "prefix_match": prefix_len(node.prompt_ids, rebuilt),
                "exact": list(rebuilt) == list(node.prompt_ids),
            }
        )

    scored = [t for t in turns if "exact" in t]
    return {
        "n_turns": len(turns),
        "n_errors": len(turns) - len(scored),
        "exact_match_frac": (sum(t["exact"] for t in scored) / len(scored))
        if scored
        else 0.0,
        "max_abs_delta": max((abs(t["delta"]) for t in scored), default=0),
        "min_prefix_match_frac": min(
            (t["prefix_match"] / max(t["engine_len"], 1) for t in scored), default=0.0
        ),
        "turns": turns,
    }
