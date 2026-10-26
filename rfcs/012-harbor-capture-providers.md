# RFC: Harbor capture purpose, provider fidelity and live-session ownership

**Status**: In Review
**Created**: 2026-09-16
**Authors**: @adithya-s-k
**RFC ID**: 012

## Summary

This addendum defines the evaluation and training APIs implemented in Harbor integration PR #1036. It extends RFC 005 and the capture design in [RFC 006 / PR #941](https://github.com/huggingface/OpenEnv/pull/941). Its change to RFC 006 decision D2 is explicit: hosted providers are supported for **evaluation**, while training export still requires exact engine tokens and processed log probabilities. Endpoint reachability is not training certification.

## Motivation

One long-lived environment serves multiple harnesses, sandboxes and inference endpoints. A hosted evaluation, a trainer-controlled rollout and a playground user can overlap. Capture purpose, upstream credentials, sampling policy and live traces must remain scoped to their own session. Prompt rewrites must not silently replace sampled tokens with re-tokenized text or fabricated log probabilities.

## Design

### Purpose and provider

`purpose` is `eval`, `train`, or legacy `auto`. An explicit evaluation remains evaluation even if its endpoint exposes tokens; training export rejects it. Explicit training fails when the endpoint cannot supply engine token capture. `auto` retains capability-based selection for existing clients.

The upstream descriptor adds `provider` (`openai`, `anthropic`, `hf`, `vllm`). Client caching includes the provider, endpoint, requested model, authentication header and credential identity. A session's credentials never become another session's defaults. Native Anthropic requests preserve signed blocks and supported metadata; cross-protocol conversion rejects semantics it cannot preserve.

`sampling` selects the training distribution and requires a positive finite temperature, full-vocabulary sampling and neutral unsupported penalties. Capture stores requested and submitted policies separately. A changed effective training policy is fatal. `eval_sampling` is an optional, validated override available only for explicit evaluation. Invalid combinations fail before sandbox allocation.

A server-side session may set a positive integer `metadata["max_output_tokens"]` before forwarding. Capture caps it at the server limit and validates it before inference. This lets one shared service reserve different output budgets for training and evaluation without mutating global configuration. The hook is not a client credential or a way to raise the server ceiling.

### Exact training contract

`openenv.harbor.contract.to_trace_entries` is the authoritative Harbor reader. The environment wrapper imports it, and the UI download uses `export_training_contract`, schema version 1. Each eligible entry contains engine `prompt_token_ids`, sampled `completion_token_ids`, aligned `per_token_logps`, and a `loss_mask` spanning prompt plus completion. Prompt positions are zero; partial completion masks survive export. Token IDs are nonnegative integers and supervised log probabilities are finite and nonpositive. Missing sampled probabilities are never filled with zeros.

Evaluation and fatal capture findings reject training export. Auxiliary calls, discarded retries and ineligible turns cannot gain supervision in the downloadable audit. A zero verifier reward remains a valid grade; missing or failed grading remains distinguishable from zero. The trainer owns advantages, weighting, batching, staleness and weight synchronization. More captured rows do not imply fair rollout weighting or a memory-safe batch.

### Streaming and session ownership

Upstream responses are buffered to preserve complete capture, then replayed as protocol-compatible SSE. Keepalive comments maintain delayed connections without adding model output or captured tokens. Fast errors retain their HTTP status. After headers are sent, late errors use an error event in the caller's protocol. Disconnects cancel pending inference capture.

`run_rollout(..., on_session_created=callback)` reports only that rollout's session ID. A live UI follows this callback, never the difference between shared registry listings. The callback is local process state, not an HTTP or MCP request field. A callback failure returns a failed rollout and releases its session. Session creation precedes sandbox setup, so the UI does not claim a ready sandbox solely because a session exists.

### Qualification and profiles

A versioned external evidence report supplies stable, experimental and unstable classifications for the recorded provider/model/adapter combination. No report means unqualified; the UI requires an explicit experimental opt-in. Recorded evidence does not qualify a newly entered endpoint or pin installed agent versions. `harness_profile` selects an explicitly supported ACP or NeMo workflow on a local seam copy, without mutating global defaults. Unknown profiles fail.

## Examples

```python
# Server-side rollout with exact training capture.
result = await run_rollout(
    task_dir=task_dir, harness="opencode", sandbox="daytona",
    registry=capture.registry, intercept_url=public_capture_url,
    model=model, trials_dir=trials_dir, upstream=upstream,
    capture_level="tokens", purpose="train",
    sampling={"temperature": 0.8, "top_p": 1.0, "top_k": -1},
    on_session_created=live_session_queue.put_nowait,
)

# Consumer and UI share the same validator and mask semantics.
entries = to_trace_entries(result)
contract = export_training_contract(result)
```

For a hosted evaluation, select its provider and use `purpose="eval"`, with no training sampling policy. Consume the verifier reward and captured messages; do not request a training contract.

## Validation and trade-offs

Deterministic tests cover invalid policies, exact masks/logprobs, provider conversion, signed content, SSE keepalives and disconnects, concurrent live-view ownership, cleanup and qualification filtering. Live provider smoke evidence is bounded by the tested tasks and versions. Optimizer replay is recorded separately from weight synchronization and reward-based learning. Buffering favors capture integrity over first-token latency; lossless prompt forks favor retention over row count.

This proposal preserves reset/step/state and the server/client boundary. It adds no trainer or tokenizer dependency to capture. Consolidating older environment-specific interceptors and implementing rollout-normalized trainer weighting remain separate work.
