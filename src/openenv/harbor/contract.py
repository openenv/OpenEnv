# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the BSD-style license in the repository LICENSE file.

"""Authoritative Harbor training export shared by clients, trainers and the UI."""

from __future__ import annotations

from typing import Any

from openenv.core.harness.capture.validate import validate_training_turn
from openenv.core.harness.training import TrainingTrace, TrainingTurn

from .models import HarborRolloutResult


def _openai_tool_calls(flat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`{name, arguments}` -> OpenAI's `{id, type, function: {name, arguments}}`.

    `HarborTurn.tool_calls` is deliberately flattened: a reward function checking which tool ran should
    not have to walk a wire envelope. But TRL reads `message["tool_calls"]` verbatim, and both
    `has_tool_call` and `apply_chat_template` expect the nested form. Handing over the flat shape makes
    `has_tool_call` false for every turn, so `train_turn_fn=has_tool_call` — the documented default for
    a coding agent — discards the entire rollout while the run still looks healthy. Templates that
    iterate `call.function.name` would raise instead.

    `arguments` is left exactly as captured. It is a JSON *string* on the wire, and TRL's
    `_decode_tool_call_arguments` parses it before rendering, so parsing it here would hand the
    template a dict it does not expect.
    """
    out: list[dict[str, Any]] = []
    for index, call in enumerate(flat or []):
        if not isinstance(call, dict):
            continue
        # Already nested (a future capture change, or another dialect): pass it through untouched.
        if call.get("function"):
            out.append(call)
            continue
        name = call.get("name")
        if not name:
            continue
        out.append(
            {
                # An id is required by the schema and is what pairs a call with its tool result. The
                # capture does not keep the harness's own id, so a positional one is minted: within a
                # single assistant message that is enough to keep the pairing unambiguous.
                "id": str(call.get("id") or f"call_{index}"),
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": call.get("arguments", ""),
                },
            }
        )
    return out


def to_trace_entries(result: HarborRolloutResult) -> list[dict[str, Any]]:
    """`HarborRolloutResult` -> TRL `TraceEntry` records, one per trainable turn.

    Auxiliary calls and discarded retries are already excluded by the server, so a caller needs no
    `agent_turn_fn`: the capture layer can tell an aux call from an agent turn structurally, which a
    flat trace cannot.

    `request_messages` is what makes this possible at all. Without it the token fields say what was
    produced but not what produced them, and no `TraceEntry` can be built.
    """
    if result.rollout_type == "eval":
        raise ValueError("eval-only rollout has no exact-token training contract")
    fatal = [finding for finding in result.findings if "[FATAL" in finding]
    if fatal:
        raise ValueError(
            "cannot train a capture with fatal validation findings: " + fatal[0]
        )
    entries: list[dict[str, Any]] = []
    for turn in result.turns or []:
        if (
            turn.role != "agent"
            or turn.discarded
            or not turn.trainable
            or not turn.completion_token_ids
        ):
            continue
        mask = turn.loss_mask
        if mask is None:
            mask = [0] * len(turn.prompt_token_ids) + [1] * len(
                turn.completion_token_ids
            )
        validate_training_turn(
            turn.prompt_token_ids, turn.completion_token_ids, turn.per_token_logps, mask
        )
        entries.append(
            {
                "request": {
                    "messages": list(turn.request_messages),
                    "tools": turn.request_tools,
                },
                "response": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": turn.text,
                                "tool_calls": _openai_tool_calls(turn.tool_calls)
                                or None,
                            },
                            "finish_reason": turn.finish_reason,
                        }
                    ]
                },
                # The engine's own tokenization, carried through rather than dropped. HarborTurn has
                # held this since it was introduced ("not a local re-render", models.py); it simply
                # had nowhere to go until TraceEntry gained the field. A consumer that has it must
                # not call apply_chat_template -- that re-render matched the engine on 0 of 28
                # measured turns and collapsed a run at its first weight update.
                "prompt_token_ids": list(turn.prompt_token_ids),
                "completion_token_ids": list(turn.completion_token_ids),
                "per_token_logps": list(turn.per_token_logps),
                # Preserve partial completion masks after reconciliation; eligibility is not
                # inferable from token ids or the turn-level trainable flag.
                "loss_mask": list(mask),
                "metadata": {
                    "turn": turn.turn,
                    "node_id": turn.node_id,
                    "sampling_params": dict(turn.sampling_params),
                    "requested_sampling_params": dict(turn.requested_sampling_params),
                    "n_tools": turn.n_tools,
                    "finish_reason": turn.finish_reason,
                },
            }
        )
    return entries


def to_training_trace(result: HarborRolloutResult) -> TrainingTrace:
    """Return the strict training API; legacy results need explicit producer masks."""
    if result.rollout_type != "train":
        raise ValueError("eval-only rollout has no exact-token training contract")
    if any("[FATAL" in finding for finding in result.findings):
        raise ValueError("cannot train a capture with fatal validation findings")
    turns = []
    for turn in result.turns:
        if turn.role != "agent" or turn.discarded or not turn.completion_token_ids:
            continue
        if turn.loss_mask is None:
            raise ValueError("training capture requires an explicit loss_mask")
        if not turn.trainable and any(turn.loss_mask):
            raise ValueError("ineligible capture contains supervised tokens")
        turns.append(
            TrainingTurn(
                node_id=turn.node_id,
                prompt_token_ids=turn.prompt_token_ids,
                completion_token_ids=turn.completion_token_ids,
                per_token_logps=turn.per_token_logps,
                loss_mask=turn.loss_mask,
                request={
                    "messages": turn.request_messages,
                    "tools": turn.request_tools,
                },
                response={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": turn.text,
                                "tool_calls": _openai_tool_calls(turn.tool_calls)
                                or None,
                            },
                            "finish_reason": turn.finish_reason,
                        }
                    ]
                },
                metadata={
                    "turn": turn.turn,
                    "sampling_params": turn.sampling_params,
                    "requested_sampling_params": turn.requested_sampling_params,
                },
            )
        )
    return TrainingTrace(turns=turns)


def export_training_contract(result: HarborRolloutResult) -> dict[str, Any]:
    """Export validated supervision and an explicit audit of excluded turns.

    Args:
        result (`HarborRolloutResult`): The captured and reconciled rollout.

    Returns:
        `dict`: Versioned trace entries, masked turn records and task outcome.
    """
    entries = to_trace_entries(result)
    turn_ids = [turn.turn for turn in result.turns]
    if len(set(turn_ids)) != len(turn_ids):
        raise ValueError("duplicate turn identities in training contract")
    selected = {entry["metadata"]["turn"]: entry for entry in entries}
    turns = []
    for turn in result.turns:
        entry = selected.get(turn.turn)
        mask = (
            entry["loss_mask"]
            if entry
            else [0] * (len(turn.prompt_token_ids) + len(turn.completion_token_ids))
        )
        turns.append(
            {
                "turn": turn.turn,
                "node_id": turn.node_id,
                "prompt_token_ids": list(turn.prompt_token_ids),
                "completion_token_ids": list(turn.completion_token_ids),
                "per_token_logps": list(turn.per_token_logps),
                "loss_mask": list(mask),
                "finish_reason": turn.finish_reason,
                "discarded": turn.discarded,
                "trainable": bool(entry) and any(mask),
                "sampling_params": dict(turn.sampling_params),
                "requested_sampling_params": dict(turn.requested_sampling_params),
            }
        )
    return {
        "schema_version": 1,
        **{
            key: getattr(result, key)
            for key in (
                "task_id",
                "task_name",
                "dataset",
                "harness",
                "sandbox",
                "trial_name",
                "session_id",
                "reward",
                "rewards",
                "reward_key",
                "rollout_type",
                "capture_level",
            )
        },
        "n_trainable_tokens": sum(sum(entry["loss_mask"]) for entry in entries),
        "trace_entries": entries,
        "turns": turns,
    }
