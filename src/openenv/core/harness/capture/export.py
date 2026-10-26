"""Graph -> the JSON a trainer consumes.

One document per rollout. Every field a trainer needs is precomputed and validated; nothing
downstream has to re-derive, re-tokenize, or guess.

    {
      "session_id": ...,
      "stats": {...},                  graph shape: turns, roots, forks, discards
      "sequences": [                   one per root-to-leaf path
        {"input_ids", "loss_mask", "logprobs", "prompt_len", "n_turns",
         "turn_lengths",              sampled tokens per turn: the join key against a harness trace
         "role",                      "agent" | "auxiliary" | "discarded"
         "validation": [...]}
      ],
      "validation": [...],             rollout-level findings
      "trainable": bool                the single gate: did anything survive
    }

Sequences are labelled rather than filtered. A caller that silently drops rows cannot be
distinguished from one that had none to drop, and "the group quietly shrank" is far harder to
diagnose than "three rows were labelled auxiliary". The trainer picks by `role`.

ROLE ASSIGNMENT is structural first, heuristic only as a tiebreak. A rollout's real work is the
longest path with tool access; a title generator or a summariser is a short toolless root. The old
approach (matching known system-prompt strings) needed a new entry per harness and failed silently
on the harnesses nobody had profiled yet.
"""

from __future__ import annotations

from typing import Any

from .validate import check_rollout, check_sequence

AGENT, AUXILIARY, DISCARDED = "agent", "auxiliary", "discarded"


def _assign_roles(graph, sequences, *, trainable_capture: bool = True) -> list[str]:
    """Label each flattened path. Purely structural.

    - a path ending in a discarded node (a sibling that never continued) is a retry
    - a path whose turns carry a TOOL MANIFEST is the agent working
    - anything else is auxiliary: title generators, summarisers, classifiers

    **Multiple agent paths are normal and all of them are trainable.** A harness that rewrites its
    system prompt mid-run breaks token-prefix continuity and starts a new root, even though the
    conversation continued: claude-code does exactly this, swapping a 12118-char system prompt for a
    12541-char one at call 6 while its message list grows 2 -> 26 unbroken. Both roots are the agent
    doing real work on the same task and earn the same reward.

    An earlier version kept only the single longest tool-using path. On opencode that was
    indistinguishable from correct (its second root really is a title generator), but on claude-code
    it silently discarded 6 genuine agent turns. Tool access alone is the honest signal: aux calls
    essentially never pass a tool manifest, and coding agents essentially always do.
    """
    discarded_ids = {n.node_id for n in graph.discarded_nodes()}
    live = [
        (i, s) for i, s in enumerate(sequences) if s.node_ids[-1] not in discarded_ids
    ]

    # Tools only DISCRIMINATE when some paths have them and others do not. That is the opencode
    # shape: an agent chain with a manifest plus a toolless title generator.
    #
    # Some harnesses never send a manifest at all. terminus-2 parses tool calls out of raw model
    # text, so every one of its paths has n_tools == 0. Applying the tool rule there labels the whole
    # rollout auxiliary, and an earlier "keep the longest" fallback then kept exactly ONE of its 13
    # turns -- a harness-trace cross-check caught it as `captured [263] vs trace [167, ..., 136]`.
    #
    # So: if nothing in the rollout uses tools, tools carry no signal and every live path is agent
    # work. If something does, the toolless paths really are auxiliary.
    any_tools = any(graph.get(nid).n_tools > 0 for _, s in live for nid in s.node_ids)

    roles = []
    for i, seq in enumerate(sequences):
        if seq.node_ids[-1] in discarded_ids:
            roles.append(DISCARDED)
            continue
        if not any_tools:
            # `n_trainable` is the tiebreak only where it can mean something. On an eval endpoint it
            # is 0 for every sequence by construction, so using it there labelled EVERY path auxiliary
            # for a harness that sends no tool manifest — terminus-2 parses tool calls out of raw text
            # — which emptied `result.turns` and mistagged the conversations, on a rollout that had
            # captured perfectly well. With neither tools nor token counts to discriminate on, a live
            # path is the agent working: that is the same conclusion the tool rule reaches, and the
            # cost of being wrong is a mislabelled trace rather than a mistrained token, since nothing
            # here is trainable anyway.
            usable = seq.n_trainable if trainable_capture else bool(seq.node_ids)
            roles.append(AGENT if usable else AUXILIARY)
            continue
        has_tools = any(graph.get(nid).n_tools > 0 for nid in seq.node_ids)
        roles.append(AGENT if has_tools else AUXILIARY)
    return roles


def export_session(
    session,
    *,
    include_discarded: bool = False,
    include_messages: bool = False,
    capture_level: str = "tokens",
) -> dict[str, Any]:
    """Build the document for one rollout: training rows when there are any, the trace always.

    Args:
        session:
            The live capture session.
        include_discarded (`bool`, *optional*, defaults to `False`):
            Keep paths that led nowhere (retries, resamples) as labelled rows.
        include_messages (`bool`, *optional*, defaults to `False`):
            Add each turn's request messages, tools and response message. Off by default because it
            multiplies payload size by the full conversation text, on when you need to feed TRL's
            `TraceEntry` contract or measure re-tokenization skew.
        capture_level (`str`, *optional*, defaults to `"tokens"`):
            What the upstream could return. Below `tokens` this is an eval rollout: the trace and the
            graph structure are complete, no row is trainable, and the token arrays are empty — by
            construction rather than by accident.

    Returns:
        `dict[str, Any]`: The rollout document.
    """
    graph = session.graph
    rollout_report = check_rollout(
        graph,
        capture_level=capture_level,
        budget_stop_count=getattr(session, "budget_stop_count", 0),
    )
    trainable_capture = (
        capture_level == "tokens" and getattr(session, "purpose", "auto") != "eval"
    )

    # Sequences are built at every level, because they are the rollout's STRUCTURE — which calls
    # belong to which conversation, which path is the agent working, which branches died — and that
    # structure is real whether or not token ids came back. Everything that reads a rollout as a
    # trace (`conversations_from_document`, `turns_from_document`, the UI transcript) walks these
    # rows, so dropping them on an eval rollout deletes exactly the payload an eval rollout is for.
    #
    # What is withheld at a lower level is the *training* claim: `check_sequence` is skipped, since
    # every one of its findings is about token arrays that are empty by design, and no row is ever
    # marked trainable. The token fields stay as the empty lists the graph produced.
    sequences = graph.sequences()
    roles = _assign_roles(graph, sequences, trainable_capture=trainable_capture)

    rows: list[dict[str, Any]] = []
    for seq, role in zip(sequences, roles):
        if role == DISCARDED and not include_discarded:
            continue
        report = check_sequence(seq) if trainable_capture else None
        rows.append(
            {
                "role": role,
                "root_id": seq.root_id,
                "node_ids": seq.node_ids,
                "n_turns": seq.n_turns,
                "prompt_len": seq.prompt_len,
                "n_trainable": seq.n_trainable if trainable_capture else 0,
                "turn_lengths": seq.turn_lengths(),
                "input_ids": seq.input_ids,
                "loss_mask": seq.loss_mask
                if trainable_capture
                else [0] * len(seq.input_ids),
                "logprobs": seq.logprobs,
                "trainable": bool(report and report.ok and role == AGENT),
                "validation": [str(f) for f in report.findings] if report else [],
            }
        )

    # Every call in arrival order, including the ones excluded from training. This is what an
    # external trace can be reconciled against: the harness logged every LLM call it made,
    # so comparing only the surviving path would report a mismatch on any rollout that retried.
    discarded_ids = {n.node_id for n in graph.discarded_nodes()}
    turns = [
        {
            "node_id": node.node_id,
            "index": node.index,
            "root_id": graph.root_of(node.node_id),
            "n_sampled": len(node.sampled_ids),
            "n_prompt": len(node.prompt_ids),
            "n_tools": node.n_tools,
            "finish_reason": node.finish_reason,
            "harness_session_id": node.harness_session_id,
            # Travels with the turn because it decides whether a trainer's recompute is comparable to
            # the captured logprob at all. See `TurnNode.sampling_params`.
            "sampling_params": node.sampling_params,
            "requested_sampling_params": node.requested_sampling_params,
            "sampled_logprobs": node.sampled_logprobs,
            "discarded": node.node_id in discarded_ids,
            **(
                {
                    "request_messages": node.request_messages,
                    "request_tools": node.request_tools,
                    "response_message": node.response_message,
                }
                if include_messages
                else {}
            ),
        }
        for node in graph.nodes()
    ]

    trainable_rows = [r for r in rows if r["trainable"]]
    return {
        "session_id": session.session_id,
        "metadata": session.metadata,
        "budget_stop_count": getattr(session, "budget_stop_count", 0),
        "turns": turns,
        "stats": {
            **graph.stats(),
            "n_sequences": len(rows),
            "n_trainable_sequences": len(trainable_rows),
            "n_trainable_tokens": sum(r["n_trainable"] for r in trainable_rows),
        },
        "sequences": rows,
        "validation": [str(f) for f in rollout_report.findings] + session.findings,
        "trainable": bool(trainable_rows) and rollout_report.ok,
        # Why this rollout is or is not trainable, travelling with the data rather than living only
        # in the server's log. A consumer that reads `trainable` alone is safe; one that wants to
        # explain an empty `sequences` list to a human needs these two.
        "capture_level": capture_level,
        "rollout_type": "train" if trainable_capture else "eval",
    }


def summarise(document: dict[str, Any]) -> str:
    """One screen of text. What you actually read after a rollout."""
    stats = document["stats"]
    lines = [
        f"session {document['session_id']}  "
        f"{document.get('rollout_type', 'train')}  "
        f"trainable={document['trainable']}",
        f"  graph: {stats['n_turns']} turns, {stats['n_roots']} roots, "
        f"{stats['n_forks']} forks, {stats['n_discarded']} discarded",
        f"  training: {stats['n_trainable_sequences']} sequence(s), "
        f"{stats['n_trainable_tokens']} trainable tokens",
    ]
    for row in document["sequences"]:
        lines.append(
            f"  [{row['role']:<10}] turns={row['n_turns']:<3} prompt={row['prompt_len']:<6} "
            f"len={len(row['input_ids']):<6} trainable={row['n_trainable']:<5} "
            f"turn_lengths={row['turn_lengths']}"
        )
        for finding in row["validation"]:
            if not finding.startswith("[INFO]"):
                lines.append(f"      {finding}")
    for finding in document["validation"]:
        lines.append(f"  {finding}")
    return "\n".join(lines)
