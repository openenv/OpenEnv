#!/usr/bin/env python3
"""Are the captured logprobs the ones the model actually assigned?

Every other check in this project verifies the SHAPE of a rollout: ids present, lengths aligned,
prefix chains exact, ATIF agreeing. None of them can see whether `per_token_logps` are the right
NUMBERS — which is the entire claim the training contract makes, and exactly where the raw-vs-
processed logprobs bug lived: aligned, negative, correctly counted, and wrong.

This scores the captured sequence again with the SAME serving engine and compares:

    captured    the logprob the engine reported while SAMPLING each token
    recomputed  the logprob the engine reports for that same token when asked to SCORE it,
                via /v1/completions with prompt=[prompt_ids + completion_ids], max_tokens=0,
                prompt_logprobs

Equivalently, this asserts **the GRPO importance ratio is 1.0** on freshly captured on-policy data:
`exp(recomputed - captured)` must be 1 if the policy has not moved. Any drift — raw instead of
processed logprobs, a re-tokenised prompt, an off-by-one in the loss mask, the wrong tokenizer —
appears here as a ratio away from 1, which is the number a trainer actually multiplies by.

The engine does the scoring rather than a locally loaded copy of the model. That keeps the property
the whole design rests on: nothing is tokenised or re-implemented on this side.

Temperature is pinned to 1.0 for both the sampling and the scoring pass. At T=1 the processed
logprob equals the raw one (dividing by 1 is a no-op), so the two passes are directly comparable;
comparing a turn sampled at some other temperature would require knowing that temperature, which the
capture deliberately does not store.

WHAT THE RESIDUAL IS, AND WHY THERE IS A NEGATIVE CONTROL
---------------------------------------------------------
The captured logprobs are copied from the engine verbatim, so "captured == what the engine said while
sampling" holds by construction. What can genuinely go wrong is ALIGNMENT: the graph stitching the
wrong prompt in front of a completion, or an off-by-one between a logprob and its token. So the
number that matters is not the absolute residual but its size RELATIVE to a deliberately broken
alignment, which this measures in the same run rather than comparing against a magic threshold.

Measured on Qwen3.5-4B served by vLLM 0.25.1:

    aligned, as captured           0.03 - 0.14 nats     ratio 0.97 - 1.15
    prompt truncated by one token  0.35 nats            ratio 0.70
    completion rotated by one      5.89 nats            ratio 360

The honest residual is not zero, and it is not this layer's doing. Scoring the identical sequence
three times gives bitwise-identical logprobs (max diff 0.000000), so the engine is deterministic with
itself; but its SAMPLING path (incremental decode, KV cache, prefix-cache reuse) and its SCORING path
(one prefill over the whole sequence) do not agree exactly in bf16 — measured directly, with capture
out of the picture, at up to 0.026 nats for a first sampled token, and wider on later turns that reuse
more cached prefix.

The consequence is worth stating plainly because it affects training rather than testing: even with
perfect capture, the GRPO importance ratio at step 0 is NOT exactly 1.0. It was within 1.03 typically
and 1.15 on a long cache-reusing turn here. Ratio-clipping thresholds should be set knowing that.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Resolved from this file's own location, so the script runs from any checkout. It used to carry an
# absolute path from the machine it was written on, which meant it could not run anywhere else — and
# a check nobody can run is not a check. Prepended only when `openenv` is not already importable, so
# an installed package still wins over the working tree.
try:  # noqa: SIM105
    import openenv  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openenv.core.harness.capture.contract import to_turn_records  # noqa: E402
from openenv.core.harness.capture.export import export_session  # noqa: E402
from openenv.core.harness.capture.graph import RolloutGraph, TurnNode  # noqa: E402
from openenv.core.harness.capture.upstream import (  # noqa: E402
    normalise_engine_base,
    normalize_response,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def post(url: str, body: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def capture_conversation(
    base: str, model: str, n_turns: int, top_p: float | None = None
) -> RolloutGraph:
    """Drive a multi-turn tool conversation and build the graph exactly as the proxy would.

    The conversation grows the way a real agent's does — assistant reply, tool result, next call —
    so the turns form a token-prefix chain rather than n independent one-shot calls. A single-turn
    test would pass even with the stitching completely broken.
    """
    graph = RolloutGraph()
    messages: list[dict] = [
        {"role": "system", "content": "You are a shell agent. Use the bash tool."},
        {
            "role": "user",
            "content": "List /tmp, then report how many entries there are.",
        },
    ]

    for index in range(n_turns):
        body = {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "max_tokens": 96,
            "temperature": 1.0,
            "logprobs": True,
            "top_logprobs": 0,
            "return_token_ids": True,
        }
        # Sampling with top_p<1 is what a harness does by default, and under
        # `--logprobs-mode processed_logprobs` it changes what a captured logprob MEANS: vLLM masks
        # the truncated tail to -inf and takes the log-softmax afterwards
        # (`v1/sample/ops/topk_topp_sampler.py`, `apply_top_k_top_p` then `compute_logprobs`), so the
        # captured value is renormalised over the surviving set. `rescore` below scores over the full
        # vocabulary, which is what a trainer does, so the residual between them is the bias itself.
        if top_p is not None:
            body["top_p"] = top_p
        payload = normalize_response(post(f"{base}/v1/chat/completions", body))
        choice = (payload.get("choices") or [{}])[0]
        entries = ((choice.get("logprobs") or {}).get("content")) or []
        graph.add_turn(
            TurnNode(
                node_id=f"n{index}",
                prompt_ids=list(payload.get("prompt_token_ids") or []),
                sampled_ids=list(choice.get("token_ids") or []),
                sampled_logprobs=[e.get("logprob") for e in entries] or None,
                model=model,
                finish_reason=choice.get("finish_reason"),
                request_messages=list(messages),
                request_tools=TOOLS,
                response_message=choice.get("message") or {},
                n_tools=len(TOOLS),
            )
        )

        reply = choice.get("message") or {}
        messages = [*messages, reply]
        calls = reply.get("tool_calls") or []
        if calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": calls[0].get("id", "call_0"),
                    "content": "a.txt\nb.txt\nc.txt",
                }
            )
        else:
            messages.append({"role": "user", "content": "Now double-check that count."})
    return graph


def rescore(
    base: str, model: str, prompt_ids: list[int], completion_ids: list[int]
) -> list[float]:
    """The engine's logprob for each completion token, given the prompt that preceded it.

    Two engines, two spellings of "score this sequence", so both are tried:

    vLLM      `max_tokens: 0` + `prompt_logprobs`, returning one dict per prompt position keyed by
              token id (`{"11": {"logprob": ...}}`, or `token_id:11` when the server runs with
              --return-tokens-as-token-ids).
    SGLang    rejects `max_tokens: 0` outright ("max_tokens must be positive") and 500s on
              `echo` + `logprobs: 0` with a bare KeyError for `input_top_logprobs`. What works is
              `max_tokens: 1` + `echo` + `logprobs: 1`, whose `token_logprobs` array is positional
              rather than keyed. Verified to agree with SGLang's native /generate `input_token_logprobs`
              to the last digit, so it is the same number by a different route.

    Position i of either form is the logprob of token i given tokens < i, so the completion occupies
    the last len(completion_ids) positions.
    """
    sequence = prompt_ids + completion_ids
    try:
        payload = post(
            f"{base}/v1/completions",
            {
                "model": model,
                "prompt": sequence,
                "max_tokens": 0,
                "temperature": 1.0,
                "prompt_logprobs": 0,
                "echo": True,
            },
        )
        entries = (payload.get("choices") or [{}])[0].get("prompt_logprobs") or []
        if len(entries) == len(sequence):
            out: list[float] = []
            for offset, token_id in enumerate(completion_ids):
                slot = entries[len(prompt_ids) + offset] or {}
                info = slot.get(str(token_id)) or slot.get(f"token_id:{token_id}")
                if info is None:
                    raise SystemExit(
                        f"position {offset}: the engine scored {list(slot)[:4]} but not the token "
                        f"that was actually sampled (id {token_id})"
                    )
                out.append(float(info["logprob"]))
            return out
    except urllib.error.HTTPError:
        pass  # not a vLLM-shaped scoring route; fall through

    payload = post(
        f"{base}/v1/completions",
        {
            "model": model,
            "prompt": sequence,
            "max_tokens": 1,
            "temperature": 1.0,
            "logprobs": 1,
            "echo": True,
        },
    )
    values = ((payload.get("choices") or [{}])[0].get("logprobs") or {}).get(
        "token_logprobs"
    ) or []
    if len(values) < len(sequence):
        raise SystemExit(
            f"scored {len(values)} positions for a {len(sequence)}-token sequence; cannot align"
        )
    return [
        float(values[len(prompt_ids) + offset]) for offset in range(len(completion_ids))
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="sample with this top_p instead of the engine default. Use it to measure the "
        "truncation bias directly: a processed logprob is taken after top_p masks the tail, so "
        "the captured value is renormalised over the kept set while a trainer's recompute is not. "
        "Expect a systematic residual of about -log(kept_mass) at p<1, and none at p=1.0.",
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=3.0,
        help="how many times smaller the honest residual must be than a one-token misalignment. "
        "Scale-free on purpose: the absolute residual depends on the model, dtype and how much "
        "prefix cache the turn reused, so a fixed nat threshold would need retuning per engine.",
    )
    args = ap.parse_args()
    base = normalise_engine_base(args.llm_url)

    knob = "engine default" if args.top_p is None else f"top_p={args.top_p}"
    print(
        f"capturing a {args.turns}-turn tool conversation at temperature 1.0, {knob} ..."
    )
    graph = capture_conversation(base, args.model, args.turns, top_p=args.top_p)

    class Session:
        session_id = "parity"
        metadata: dict = {}
        findings: list[str] = []

    session = Session()
    session.graph = graph
    document = export_session(session, capture_level="tokens")
    if document["rollout_type"] != "train":
        raise SystemExit(f"endpoint is not trainable: {document['capture_level']}")

    records = to_turn_records(graph, document)
    print(
        f"captured {len(records)} turns, {sum(len(c) for _, c, _ in records)} sampled tokens\n"
    )

    signed: list[float] = []

    def compare(prompt_ids, completion_ids, captured):
        recomputed = rescore(base, args.model, prompt_ids, completion_ids)
        pairs = [(a, b) for a, b in zip(captured, recomputed) if b is not None]
        diffs = [abs(a - b) for a, b in pairs]
        # exp(new - old) is literally what GRPO multiplies by.
        ratios = [pow(2.718281828459045, b - a) for a, b in pairs]
        # Kept separately because a truncation bias and a misalignment look different: truncation is
        # systematic and one-directional (captured always too high, so `captured - recomputed > 0`),
        # while a misalignment is large and randomly signed. The max would report both; only the mean
        # of the signed residual distinguishes them.
        signed.extend(a - b for a, b in pairs)
        return max(diffs), max(ratios, key=lambda r: abs(r - 1.0)), len(pairs)

    worst = 0.0
    worst_ratio = 1.0
    total = 0
    for turn, (prompt_ids, completion_ids, captured) in enumerate(records):
        if not completion_ids:
            continue
        d, r, n = compare(prompt_ids, completion_ids, captured)
        total += n
        worst = max(worst, d)
        worst_ratio = (
            max(worst_ratio, r) if abs(r - 1) > abs(worst_ratio - 1) else worst_ratio
        )
        print(
            f"  turn {turn}: {len(completion_ids):>3} tokens  prompt={len(prompt_ids):>5}  "
            f"max|diff|={d:.6f}  worst ratio={r:.6f}"
        )

    aligned_signed = list(signed)  # before the negative controls pollute it

    # Negative control on the longest turn: the same comparison against a knowingly wrong alignment.
    # Without this the residual above is uninterpretable — it could mean "faithful" or "quietly off".
    longest = max(records, key=lambda rec: len(rec[1]))
    p_ids, c_ids, cap = longest
    truncated, _, _ = compare(p_ids[:-1], c_ids, cap)
    rotated, rot_ratio, _ = compare(p_ids, c_ids[1:] + c_ids[:1], cap)

    print()
    print(f"tokens compared              {total}")
    if aligned_signed:
        mean = sum(aligned_signed) / len(aligned_signed)
        print(
            f"mean signed residual         {mean:+.6f} nats  -> mean ratio "
            f"{pow(2.718281828459045, -mean):.4f}   ({knob})"
        )
    print(
        f"aligned, as captured         {worst:.6f} nats   worst ratio {worst_ratio:.4f}"
    )
    print(f"prompt truncated by 1 token  {truncated:.6f} nats")
    print(
        f"completion rotated by 1      {rotated:.6f} nats   worst ratio {rot_ratio:.2f}"
    )

    broken = min(truncated, rotated)
    if worst <= 0 or broken < args.margin * worst:
        print(
            f"\nFAIL: a one-token misalignment is only {broken / max(worst, 1e-9):.1f}x the residual "
            f"(need {args.margin}x). Either the capture is misaligned, or this check cannot tell."
        )
        raise SystemExit(1)
    print(
        f"\nPASS: misalignment shows up {broken / worst:.0f}x larger than the honest residual, so the"
    )
    print(
        "      prompt/completion/logprob alignment in the captured contract is correct."
    )
    print(
        "      The residual itself is the engine's sampling-vs-scoring gap, not this layer's."
    )


if __name__ == "__main__":
    main()
