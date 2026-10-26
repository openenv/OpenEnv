"""The intercept server.

    input   an OpenAI-spec endpoint you already host (vLLM or SGLang) + the served model name
    output  per rollout, a JSON document of exact token ids, logprobs and loss masks, ready to train

In between: a coding agent points at this URL, in whichever wire dialect it speaks, and
nothing about the agent changes except a base URL and an API key.

    agent (in an E2B sandbox, any of ~37)
       |  OPENAI_BASE_URL / ANTHROPIC_BASE_URL / provider config = this server
       |  API key = the rollout's session id       <- the entire multiplexing scheme
       v
    THIS  --detect dialect--> normalise to chat --inject capture params--> your engine
       <--replay in the agent's dialect (SSE if it asked for SSE)---------┘
       |
       └─ each call becomes a node in the rollout graph, linked by token prefix

WHY IT CAPTURES FAITHFULLY: we never tokenize. The engine tokenizes each prompt to serve it and hands
back `prompt_token_ids`, so turn k+1's prompt is the canonical tokenization of everything up to that
point, tool results included. Completions come back as sampled ids with aligned logprobs. Stitching
those along a graph path reproduces exactly what the model saw and produced, with no local chat
template involved. See `graph.py`.

TWO ASYMMETRIES, both learned from real failures:

  * **Capture is non-streaming, the reply is whatever the client asked for.** One complete response
    carries ids and logprobs whole; reassembling them from SSE deltas is error-prone in exactly the
    way that silently corrupts training data. But a harness that requested SSE and receives a JSON
    body does not error, it yields nothing: opencode reported `step-finish reason:"unknown"`, zero
    tokens, no error, having been handed a perfectly valid tool call. See `sse.py`.
  * **We validate on ingest, not on export.** A turn whose logprobs are misaligned must be caught
    while we still know which turn it was.

Run:  python -m intercept.server --llm-url http://127.0.0.1:8000 --model Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import sse
from .contract import BUDGET_STOP_MESSAGE, to_trace_entries
from .detection import APIType, detect
from .dialects import TransformManager
from .export import export_session
from .graph import TurnNode
from .sessions import (
    evaluation_sampling,
    extract_api_key,
    extract_harness_session,
    rollout_type_for,
    SessionRegistry,
    Upstream,
)
from .upstream import (
    InferenceClient,
    training_sampling,
    truncating_params,
    UpstreamError,
    UpstreamHTTPError,
    UpstreamRequestError,
)
from .validate import check_turn, check_turn_eval

logger = logging.getLogger("intercept")


def _system_digest(messages: list[dict[str, Any]]) -> str | None:
    """Cheap identity for 'which conversation is this'. Recorded, never used for routing."""
    import hashlib

    for message in messages or []:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, list):  # anthropic / responses send block lists
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            if isinstance(content, str) and content:
                return hashlib.sha256(content.encode()).hexdigest()[:16]
    return None


# Routes an agent calls that are NOT model turns. They must be answered, but must never become graph
# nodes: recording them adds a bogus root and corrupts the trajectory structure.
#
# Borrowed from verifiers, whose Dialect ABC carries `aux_routes` for exactly this
# (v1/dialects/anthropic.py:273, "relayed as native JSON, never recorded on the trace").
# claude-code calls count_tokens before sending a turn; without this the catch-all would hand it to
# `transform_request`, forward nonsense upstream, and file the result as a model call.
# Each entry maps a route suffix to the DIALECT whose reply shape the caller expects. Answering an
# aux route in the wrong shape is its own bug: gemini-cli's `:countTokens` used to fall through to the
# catch-all, get detected as GOOGLE, turn into a real chat completion, land in the graph as a bogus
# root that inflated n_turns, and hand the caller a `{"candidates": [...]}` envelope where it wanted
# `{"totalTokens": N}`.
AUX_ROUTES: dict[str, str] = {
    "/v1/messages/count_tokens": "anthropic",
    ":counttokens": "google",
}


def aux_dialect(path: str) -> str | None:
    """Which dialect's token-count reply this path expects, or `None` if it is not an aux route.

    Google puts the method after a colon on the model path
    (`/v1beta/models/gemini-2.5-pro:countTokens`), so suffix matching has to be case-insensitive
    rather than an exact path compare.
    """
    normalised = ("/" + path.lstrip("/")).lower()
    for route, dialect in AUX_ROUTES.items():
        if normalised.endswith(route):
            return dialect
    return None


def is_aux_route(path: str) -> bool:
    return aux_dialect(path) is not None


def aux_token_count_response(dialect: str, count: int) -> dict[str, Any]:
    """The token-count reply in the shape the calling dialect expects."""
    if dialect == "google":
        # gemini-cli reads `totalTokens`; the other two fields are part of the documented response and
        # cheap to include rather than have a client guess at their absence.
        return {
            "totalTokens": count,
            "totalBillableCharacters": count * 4,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": count}],
        }
    return {"input_tokens": count}


def approximate_token_count(body: dict[str, Any]) -> int:
    """Answer a count_tokens request without a tokenizer.

    We deliberately do not load one: the whole design keeps tokenization on the engine, and pulling a
    tokenizer in here just to serve a side request would reintroduce the "two sources of truth"
    problem this architecture exists to avoid. Agents use this figure for context-budget decisions,
    not for anything that reaches training, so a ~4-chars-per-token estimate is sufficient. If a
    harness turns out to depend on exactness, forward it to the engine's /tokenize endpoint instead.
    """
    text_len = 0
    for message in body.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            text_len += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text_len += len(str(part.get("text") or part.get("content") or ""))
    system = body.get("system")
    if isinstance(system, str):
        text_len += len(system)
    elif isinstance(system, list):
        text_len += sum(
            len(str(p.get("text", ""))) for p in system if isinstance(p, dict)
        )

    # Google puts the conversation in `contents` with `parts`, and the system prompt in
    # `systemInstruction` — neither of which the OpenAI/Anthropic branches above look at. Without this
    # a `:countTokens` call collapsed to the `max(1, ...)` floor and answered 1 on every request, so
    # gemini-cli got a useless context-budget signal and could mismanage compaction mid-rollout. The
    # aux route answers Google now, so the estimator has to speak Google too.
    for content in body.get("contents") or []:
        if not isinstance(content, dict):
            continue
        for part in content.get("parts") or []:
            if isinstance(part, dict):
                text_len += len(str(part.get("text") or ""))
            elif isinstance(part, str):
                text_len += len(part)
    instruction = body.get("systemInstruction") or body.get("system_instruction")
    if isinstance(instruction, dict):
        for part in instruction.get("parts") or []:
            if isinstance(part, dict):
                text_len += len(str(part.get("text") or ""))
    elif isinstance(instruction, str):
        text_len += len(instruction)

    # Tool manifests count too, under each dialect's own spelling.
    text_len += len(str(body.get("tools") or ""))
    return max(1, text_len // 4)


def wants_stream(path: str, body: dict[str, Any]) -> bool:
    """Did the client ask for SSE? Each dialect says so differently.

    OpenAI chat, Responses and Anthropic all set `stream: true` in the body. **Google does not.** It
    signals streaming in the URL: `:streamGenerateContent`, usually with `?alt=sse`. gemini-cli calls

        POST /v1beta/models/<model>:streamGenerateContent?alt=sse

    with no `stream` key anywhere in the body, so a body-only check returns False and we answer a
    streaming request with a plain JSON document. It arrives as HTTP 200 and the client dies parsing
    it:

        Error: Incomplete JSON segment at the end
            at ApiClient.processStreamResponse_1 (@google/gemini-cli/...)

    Same failure family as opencode's silent `reason:"unknown"`: a valid-looking response in the
    wrong envelope. verifiers models this as a per-dialect `Dialect.streaming(body)`; this is the
    same idea kept to one function.
    """
    if body.get("stream") is True:
        return True
    lowered = path.lower()
    return "streamgeneratecontent" in lowered or "alt=sse" in lowered


_MAX_TOKENS_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens")

# Sampling knobs that alter the distribution a processed logprob is taken over. Recorded per turn so a
# trainer can tell whether its own recompute is comparable; see `TurnNode.sampling_params`.
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
)


def clamp_output_tokens(chat_request: dict[str, Any], cap: int | None) -> int | None:
    """Cap the requested output length so prompt + completion fits the served context window.

    Harnesses ask for absurd output budgets. qwen-coder requests **64000** output tokens, which on a
    65536-token model leaves room for a 1536-token prompt and then fails on the next character:

        maximum context length is 65536 tokens. However, you requested 64000 output tokens and
        your prompt contains at least 1537 input tokens, for a total of at least 65537

    Every call 502s, the agent does nothing, and it presents as "reached the intercept, captured
    nothing". Polar caps this too (`proxy_max_tokens_cap = 16384`, noting opencode's ~32000 default
    "exceeds some provider limits"), so it is a known hazard rather than one harness misbehaving.

    A fixed cap rather than `context - len(prompt)`: computing the latter needs a tokenizer here, and
    keeping tokenization on the engine is the whole design. Agent turns are short (the longest seen
    across every validated harness is 874 tokens), so a few thousand is generous.

    Returns the value it replaced, for logging, or None if nothing changed.
    """
    if not cap:
        return None
    replaced = None
    for key in _MAX_TOKENS_KEYS:
        value = chat_request.get(key)
        if isinstance(value, int) and value > cap:
            chat_request[key] = cap
            replaced = max(replaced or 0, value)
    if not any(chat_request.get(key) is not None for key in _MAX_TOKENS_KEYS):
        chat_request["max_tokens"] = cap
    return replaced


def normalise_for_capture(chat_request: dict[str, Any]) -> None:
    """Force the upstream call into the one shape that yields complete, capturable responses.

    `stream_options` is not cosmetic: vLLM validates it against `stream` and rejects the pair with
    "Stream options can only be defined when `stream=True`", which 400s the ENTIRE request. opencode
    sends it on every call, so leaving it in is a total outage rather than a degradation.

    Both keys are set here rather than left to the inference client, because the client rewrites
    `stream` only after this point: a check against the incoming value sees `true` and leaves
    `stream_options` behind, which is precisely the bug this exists to prevent.

    An empty `tools` array is dropped for the same reason. vLLM rejects it outright:

        `tools` must not be an empty array. Either provide at least one tool or omit the field
        entirely.

    kimi-cli sends `tools: []` once its agent loop has no tools left to offer, which 400s the call.
    The two forms mean the same thing to the model, so dropping the key is lossless and keeps the
    rollout alive rather than truncating it mid-trajectory.

    Everything that only makes sense ALONGSIDE `tools` has to go with it. Dropping the list and
    leaving its companions behind trades one provider's 400 for another's:

        Invalid value for 'parallel_tool_calls': 'parallel_tool_calls' is only allowed when
        'tools' are specified.

    That is codex against OpenAI, every call, for the whole rollout — it sends `tools: []` plus
    `parallel_tool_calls`, so stripping only the list left the orphan behind. vLLM ignores the orphan,
    which is why this survived until a hosted provider was tried.
    """
    chat_request["stream"] = False
    chat_request.pop("stream_options", None)
    for key in ("tools", "functions"):
        if key in chat_request and not chat_request[key]:
            chat_request.pop(key)
    # `tool_choice` and `parallel_tool_calls` are equally invalid without `tools`, and meaningless
    # once the list is gone.
    if "tools" not in chat_request:
        chat_request.pop("tool_choice", None)
        chat_request.pop("parallel_tool_calls", None)


def normalise_response(response: dict[str, Any]) -> None:
    """Fill in usage sub-objects that vLLM leaves null but the OpenAI schema always returns.

    vLLM returns `"prompt_tokens_details": null` when prefix caching is off. OpenAI always returns
    the object, so a harness that reads `usage.prompt_tokens_details.cached_tokens` without guarding
    gets an AttributeError. trae-agent does exactly that and dies after its FIRST call:

        'NoneType' object has no attribute 'cached_tokens'

    which produced a clean single-turn capture and a task the agent never attempted.

    This touches ONLY accounting fields. No token id, logprob or message content is altered, so it
    cannot affect what gets captured or trained. It is a compatibility shim that makes us MORE
    OpenAI-conformant than the engine behind us, which is the safe direction: a client that already
    guarded for null sees a zeroed object instead, which reads the same.
    """
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return
    if usage.get("prompt_tokens_details") is None:
        usage["prompt_tokens_details"] = {"cached_tokens": 0, "audio_tokens": 0}
    if usage.get("completion_tokens_details") is None:
        usage["completion_tokens_details"] = {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        }


def normalise_client_payload(payload: dict[str, Any], api_type: APIType) -> None:
    """Fill in usage sub-objects the OUTBOUND dialect promises but the transformer omits.

    Sibling of `normalise_response`, one layer further out. That one repairs the chat-completions
    usage we get FROM vLLM; this repairs the usage we hand TO the client after translation.

    Polar's Responses transformer builds usage as exactly
    `{"input_tokens", "output_tokens", "total_tokens"}` (transform/openai_responses.py:43), with no
    detail sub-objects. The real Responses API always returns them, and trae-agent reads them without
    a guard (trae_agent/utils/llm_clients/openai_client.py):

        cache_read_input_tokens=response.usage.input_tokens_details.cached_tokens or 0,
        reasoning_tokens=response.usage.output_tokens_details.reasoning_tokens or 0,

    so `input_tokens_details` is None and it dies with
    `'NoneType' object has no attribute 'cached_tokens'` after its FIRST call.

    Note this is why trae-agent looked like a chat-completions harness for a whole night: its seam
    says `openai_chat`, but the access log shows exactly one `POST /v1/responses` against 465
    chat-completions calls. It speaks Responses.

    Accounting fields only. No token id, logprob or content is touched, so capture is unaffected.
    """
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return
    if api_type is APIType.OPENAI_RESPONSES:
        if usage.get("input_tokens_details") is None:
            usage["input_tokens_details"] = {"cached_tokens": 0}
        if usage.get("output_tokens_details") is None:
            usage["output_tokens_details"] = {"reasoning_tokens": 0}


class UpstreamPool:
    """One inference client and one capability probe per distinct engine.

    The engine a rollout talks to is a per-SESSION property, not a per-server one: a dataset server is
    long-lived (thousands of task files, prebuilt sandbox templates) while a vLLM restarts every
    training run, and a train-tier engine and an eval-tier one are usually both wanted at once. So
    callers name their engine when they mint a session, and this pool makes that cheap.

    Two things it exists to avoid:

      * **Re-probing.** Deciding a tier means sending real completions (`validate_llm`). Doing that per
        session would add several round trips to every rollout, so the measurement is cached per
        `(url, model, auth_header, credential_digest)` and shared by matching sessions.
      * **Client churn.** One `InferenceClient` per engine rather than per rollout, so connection
        pooling and the discovered `param_fixes` are shared.

    The tier is MEASURED, never assumed: an engine that cannot be probed is `text`, the weakest, so a
    rollout is never stamped trainable without evidence.
    """

    _probes = ThreadPoolExecutor(max_workers=4, thread_name_prefix="capture-probe")

    def __init__(self, *, default_client: InferenceClient, default_level: str) -> None:
        self._default = (default_client, default_level)
        self._by_engine: dict[
            tuple[str, str, str, str, str], tuple[InferenceClient, str]
        ] = {}
        self._pending: dict[tuple[str, str, str, str, str], Future] = {}
        self._lock = threading.Lock()

    @property
    def default(self) -> tuple[InferenceClient, str]:
        """The engine this server was booted with, for sessions that name none."""
        return self._default

    def known(self) -> list[dict[str, Any]]:
        """What has been probed so far, for `/health`. Never includes credentials."""
        with self._lock:
            return [
                {
                    "llm_url": url,
                    # The client's model, not the key's: a caller may have left it blank for a
                    # single-model endpoint and the probe resolved it, so the key holds "" while the
                    # requests actually being sent carry the real name.
                    "model": client.served_model or "",
                    "capture_level": level,
                }
                for (url, _requested, _header, _credential, _provider), (
                    client,
                    level,
                ) in self._by_engine.items()
            ]

    def forget(self, upstream: Upstream) -> None:
        """Drop the cached client for `upstream`, and with it the credential it holds.

        Every proxied call resolves its session's engine here, so the next call naming this engine
        probes it again. Call it only once no session uses the engine: for a caller whose key should
        not outlive their use of it.
        """
        with self._lock:
            self._by_engine.pop(upstream.cache_key, None)

    async def resolve(self, upstream: Upstream) -> tuple[InferenceClient, str]:
        """Client and measured capture level for `upstream`, probing once per engine."""
        key = upstream.cache_key
        with self._lock:
            hit = self._by_engine.get(key)
            if hit is not None:
                return hit
            pending = self._pending.get(key)
            if pending is None:
                pending = self._probes.submit(self._resolve_once, upstream)
                self._pending[key] = pending
        # Shield shared work: cancelling one waiter must not cancel another rollout's probe.
        return await asyncio.shield(asyncio.wrap_future(pending))

    def _resolve_once(self, upstream: Upstream) -> tuple[InferenceClient, str]:
        key = upstream.cache_key
        try:
            model, level = self._probe(upstream)
            client = InferenceClient(
                base_url=upstream.llm_url.rstrip("/"),
                served_model=model,
                api_key=upstream.api_key,
                auth_header=upstream.auth_header,
                capture_level=level,
                provider=upstream.provider,
            )
            with self._lock:
                self._by_engine[key] = (client, level)
            return client, level
        finally:
            with self._lock:
                self._pending.pop(key, None)

    def _probe(self, upstream: Upstream) -> tuple[str, str]:
        """`(served_model, capture_level)`. Blocking; always called on a worker thread."""
        from .validate_llm import list_models, validate_llm

        model = upstream.model
        try:
            if not model:
                served = list_models(
                    upstream.llm_url,
                    api_key=upstream.api_key,
                    auth_header=upstream.auth_header,
                )
                # Only when unambiguous. Guessing here makes the proxy rewrite `model` to the wrong
                # name, and every call then fails upstream for a reason that mentions neither.
                model = served[0] if len(served) == 1 else ""
            if not model:
                logger.warning(
                    "upstream %s serves several models and none was named; capture level is 'text'",
                    upstream.llm_url,
                )
                return "", "text"
            report = validate_llm(
                upstream.llm_url,
                model,
                api_key=upstream.api_key,
                auth_header=upstream.auth_header,
                provider=upstream.provider,
            )
            level = report.capture_level or "text"
            logger.warning(
                "probed upstream %s (%s): capture_level=%s rollout_type=%s",
                upstream.llm_url,
                model,
                level,
                report.rollout_type,
            )
            return model, level
        except Exception as exc:  # noqa: BLE001 - an unreachable engine is a tier, not a crash
            logger.warning(
                "could not probe upstream %s: %s: %s; capture level is 'text'",
                upstream.llm_url,
                type(exc).__name__,
                exc,
            )
            return model, "text"


def _budget_stop_response(model: str) -> dict[str, Any]:
    """A terminal chat-completion, in the shape every dialect transformer expects on the way out.

    The content is NON-EMPTY on purpose. An empty assistant message reads as a failed generation and
    the harness retries it, so the proxy answers the stop again and the rollout spins until its
    timeout -- measured, with opencode looping on exactly this. It is deliberately a statement about
    the BUDGET rather than about the task: it lands in the harness's own transcript, and capture never
    records it, so it must not look like something the model chose to say about the work.
    """
    return {
        "id": f"budget-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": BUDGET_STOP_MESSAGE,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def create_app(
    *,
    llm_url: str = "",
    model: str | None = None,
    engine: str = "",
    require_registered: bool = True,
    max_output_tokens: int | None = 8192,
    api_key: str | None = None,
    auth_header: str = "Authorization",
    provider: str = "openai",
    capture_level: str = "tokens",
    admin_key: str | None = None,
    max_model_calls: int = 0,
) -> FastAPI:
    """The capture proxy as an ASGI app.

    Args:
        llm_url (`str`):
            The OpenAI-spec endpoint to forward to.
        model (`str`, *optional*):
            Served model id to send upstream, overriding whatever the agent asked for.
        engine (`str`, *optional*):
            What is on the other end, for `/health` only. Defaults to `"unknown"` there rather than
            to `"vllm"`: the proxy cannot tell, nothing passes it on the harbor path, and a hosted
            endpoint reported as vLLM reads as a claim about capture that `capture_level` then
            contradicts.
        require_registered (`bool`, *optional*, defaults to `True`):
            Reject callers whose API key is not a minted session id. This port may be public.
        max_output_tokens (`int`, *optional*, defaults to `8192`):
            Cap on requested completion length; `0` or `None` disables.
        api_key (`str`, *optional*):
            Credential for the *upstream*. Never the agent-facing key — that one is the session id,
            and the sandbox never sees this value.
        auth_header (`str`, *optional*, defaults to `"Authorization"`):
            Header to send `api_key` under.
        capture_level (`str`, *optional*, defaults to `"tokens"`):
            What the upstream can return, as decided by `validate_llm`. Below `tokens` every rollout
            from this app is an eval rollout.
        admin_key (`str`, *optional*):
            Required by the session-management routes when set. Leave unset for a private port; set it
            whenever this app is reachable from outside, which `serve` and `rollout` do automatically.
        max_model_calls (`int`, *optional*, defaults to `0`):
            Default ceiling on model calls per session; `0` is unlimited, and a session may name its
            own. Once a rollout reaches it the proxy answers a terminal completion itself, which ends
            the agent's loop without recording a turn the model never generated. Worth setting on any
            training deployment: no agent harness honours its own step config (opencode 1.18.30 ran 61
            model calls at `steps=3`, `maxSteps=3` and unset alike), and one runaway rollout holds its
            whole GRPO group hostage.
    """
    if provider == "anthropic":
        capture_level = "text"
    app = FastAPI(title="openenv-capture")

    # Identifies this app instance on /health. A caller that binds a port cannot tell "my server is
    # up" from "someone else's server already held this port" by connecting alone, and answering the
    # wrong process is silent: sessions are minted here and rejected there, so the agent gets 401 and
    # the rollout reports no model calls.
    app.state.instance_id = uuid.uuid4().hex
    # `None` when the server was booted without an engine. That is a supported state, not a broken
    # one: the datasets are what makes this server worth keeping alive, and the engine is a per-rollout
    # detail that arrives with the session. A session that names no engine and finds no default gets a
    # clear 503 rather than calls to an empty base URL.
    app.state.inference = (
        InferenceClient(
            base_url=llm_url.rstrip("/"),
            served_model=model,
            api_key=api_key,
            auth_header=auth_header,
            capture_level=capture_level,
            provider=provider,
        )
        if llm_url
        else None
    )
    app.state.transforms = TransformManager()
    app.state.registry = SessionRegistry(require_registered=require_registered)
    app.state.model = model
    app.state.max_model_calls = max_model_calls
    app.state.llm_url = llm_url
    app.state.max_output_tokens = max_output_tokens
    app.state.capture_level = capture_level
    # Per-session engines. The client built above is the DEFAULT, used by sessions that name none, so
    # a server booted with --llm-url behaves exactly as before.
    app.state.upstreams = UpstreamPool(
        default_client=app.state.inference, default_level=capture_level
    )
    app.state.admin_key = admin_key or None

    async def _upstream_for(session) -> tuple[InferenceClient, str]:
        """The engine this session's calls go to, and the level it was measured at."""
        if session.upstream is not None:
            return await app.state.upstreams.resolve(session.upstream)
        return app.state.upstreams.default

    def _model_of(session) -> str:
        """The model name to send upstream for this session.

        The proxy rewrites `model` because harnesses mangle it: opencode is configured with
        `intercepted/<model>` and the provider layer forwards only the last path segment, so what
        arrives is `Qwen3.5-2B` where the engine serves `Qwen/Qwen3.5-2B` and answers 404. Rewriting
        it is therefore not a nicety, it is what makes the call work at all.

        With the engine per session, the name has to come from the SESSION's engine. Reading the
        server default here is what broke an engineless server: `app.state.model` was empty, the
        rewrite was skipped, and the agent's mangled name went upstream untouched.
        """
        if (
            session is not None
            and session.upstream is not None
            and session.upstream.model
        ):
            return session.upstream.model
        return app.state.model or ""

    def _level_of(session) -> str:
        """A session's capture level without touching the network.

        Reads what the probe recorded on the session, falling back to the server default. Used where
        the level is needed but no upstream call is being made — exporting, for instance.
        """
        return session.capture_level or app.state.capture_level

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        return {
            "status": "ok",
            "instance": app.state.instance_id,
            "upstream": llm_url,
            "engine": engine or "unknown",
            "model": app.state.model,
            "sessions": len(app.state.registry.list_ids()),
            "require_registered": app.state.registry.require_registered,
            # What this proxy can actually produce, so a caller never has to infer it from the
            # engine name. `upstream_auth` is a boolean on purpose: the key itself must not be
            # readable from an endpoint that, on a Space, is public.
            "capture_level": app.state.capture_level,
            "rollout_type": "train" if app.state.capture_level == "tokens" else "eval",
            # Engines named per session and already probed, so a caller can see what this server
            # measured without minting a session. Behind the admin key when one is set: on a public
            # Space these are other callers' endpoints, private tunnel URLs among them.
            "upstreams": app.state.upstreams.known() if _admin_ok(request) else [],
            "upstream_auth": bool(app.state.inference and app.state.inference.api_key),
            "param_fixes": (
                [str(f) for f in app.state.inference.param_fixes]
                if app.state.inference
                else []
            ),
        }

    def _admin_ok(request: Request) -> bool:
        """Whether a caller may use the session-management routes.

        The registered-session check guards the catch-all proxy route and nothing else, which is fine
        while this app owns a private port and wrong once it is mounted at `/capture` on a public
        Space: `GET /sessions` enumerated every live rollout, `GET /sessions/{id}/rollout` returned its
        full token-level training data, `DELETE` ended it, and `POST /sessions` let anyone mint a key
        that the proxy would then honour — turning the endpoint into the open relay the 401 exists to
        prevent.

        Gated on the ADMIN key rather than a session id, because these routes are the trainer's
        control plane, not the agent's data plane. `admin_key` defaults to unset, which keeps a
        local run on a private port exactly as convenient as before; `serve`/`push` set it whenever
        the proxy is reachable from outside.
        """
        expected = app.state.admin_key
        if not expected:
            return True
        offered = extract_api_key(dict(request.headers)) or ""
        # Constant-time: these are short strings and the comparison is cheap, but a timing oracle on a
        # public endpoint is free to exploit and free to close.
        return secrets.compare_digest(offered, expected)

    def _forbidden() -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "message": "this route requires the capture admin key",
                    "type": "invalid_request_error",
                }
            },
            status_code=401,
        )

    def _invalid_request(message: str) -> JSONResponse:
        return JSONResponse(
            {"error": {"message": message, "type": "invalid_request_error"}},
            status_code=400,
        )

    @app.post("/sessions")
    async def create_session(
        request: Request, payload: dict[str, Any] | None = None
    ) -> Any:
        """Mint a rollout id. Hand it to the agent as its API key; that is the whole integration.

        The caller may also name the engine this rollout should use — `llm_url`, and optionally
        `model`, `api_key`, `auth_header`. That engine is PROBED HERE, before the session is handed
        back, and the measured tier is returned with it. So a caller learns whether it is going to get
        a trainable rollout at submit time, not minutes later when the token fields come back empty.

        Omitting `llm_url` uses the server's default engine, which is what a server booted with
        `--llm-url` has always done.
        """
        if not _admin_ok(request):
            return _forbidden()
        payload = payload or {}
        metadata = payload.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            return _invalid_request("metadata must be a JSON object")
        metadata = metadata or {}
        reserved = {
            "session_id",
            "upstream",
            "capture_level",
            "max_model_calls",
            "sampling",
            "purpose",
            "eval_sampling",
        } & metadata.keys()
        if reserved:
            noun = "key" if len(reserved) == 1 else "keys"
            return _invalid_request(
                f"metadata cannot include reserved {noun}: {', '.join(sorted(reserved))}"
            )
        budget = payload.get("max_model_calls", app.state.max_model_calls)
        if type(budget) is not int or budget < 0:
            return _invalid_request("max_model_calls must be a non-negative integer")
        try:
            policy = training_sampling(payload.get("sampling"))
        except ValueError as exc:
            return _invalid_request(str(exc))
        upstream = None
        level = ""
        llm_url = str(payload.get("llm_url") or "").strip()
        if llm_url:
            upstream = Upstream(
                llm_url=llm_url,
                model=str(payload.get("model") or "").strip(),
                api_key=payload.get("api_key") or None,
                auth_header=str(payload.get("auth_header") or "Authorization"),
                provider=str(payload.get("provider") or "openai"),
            )
            # Probe now. The cache means this costs round trips only for an engine never seen before,
            # so a whole GRPO group naming the same vLLM pays for it once.
            client, level = await app.state.upstreams.resolve(upstream)
            # Carry the model the probe settled on: the caller may have left it blank for a
            # single-model endpoint, and the proxy needs the resolved name to rewrite requests.
            upstream = replace(upstream, model=client.served_model or upstream.model)
        purpose = payload.get("purpose", "auto")
        try:
            rollout_type = rollout_type_for(purpose, level or app.state.capture_level)
            eval_policy = evaluation_sampling(payload.get("eval_sampling"))
            if eval_policy and purpose != "eval":
                raise ValueError("eval_sampling requires explicit eval purpose")
            if purpose == "eval" and payload.get("sampling") is not None:
                raise ValueError(
                    "eval purpose cannot apply a training sampling override"
                )
        except (ValueError, TypeError) as exc:
            return _invalid_request(str(exc))
        session = app.state.registry.create(
            payload.get("session_id"),
            purpose=purpose,
            eval_sampling=eval_policy,
            upstream=upstream,
            capture_level=level,
            # Falls back to the server default, so a deployment can cap every rollout without its
            # callers knowing, and a caller can still tighten it per rollout.
            max_model_calls=budget,
            sampling=policy or None,
            **metadata,
        )
        effective = _level_of(session)
        return {
            "session_id": session.session_id,
            "capture_level": effective,
            "rollout_type": rollout_type,
            "max_model_calls": session.max_model_calls,
            "llm_url": upstream.llm_url if upstream else app.state.llm_url,
            "model": upstream.model if upstream else app.state.model,
        }

    @app.get("/sessions")
    async def list_sessions(request: Request) -> Any:
        if not _admin_ok(request):
            return _forbidden()
        return {"sessions": app.state.registry.summary()}

    @app.get("/sessions/{session_id}")
    async def session_status(session_id: str, request: Request) -> Any:
        if not _admin_ok(request):
            return _forbidden()
        """Live progress. `idle_s` is the cheapest wedge detector: turns arriving means progress."""
        session = app.state.registry.get(session_id)
        if session is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        return {
            "session_id": session_id,
            "idle_s": round(session.idle_seconds, 1),
            "upstream_errors": session.upstream_errors,
            **session.graph.stats(),
        }

    @app.get("/sessions/{session_id}/rollout")
    async def rollout(
        session_id: str,
        request: Request,
        include_discarded: bool = False,
        include_messages: bool = False,
    ) -> Any:
        """THE training endpoint: stitched, masked, logprob-aligned, validated."""
        if not _admin_ok(request):
            return _forbidden()
        session = app.state.registry.get(session_id)
        if session is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        return export_session(
            session,
            include_discarded=include_discarded,
            include_messages=include_messages,
            capture_level=_level_of(session),
        )

    @app.get("/sessions/{session_id}/trace_entries")
    async def trace_entries(session_id: str, request: Request) -> Any:
        """The LOOP-OWNING training endpoint: `list[TraceEntry]`, one per captured model call.

        `/rollout` returns the stitched document (input_ids + loss_mask + logprobs) for a consumer
        that trains on whole sequences. A loop-owning consumer wants the per-call records instead --
        the agent drove its own tool loop, so the unit is "one model call" and it re-derives the
        prompt from `request`. That is what `LoopOwningSession.fetch_proxy_trace` is defined to
        return, and `to_trace_entries` already produced this shape; it just had no way to be asked,
        because it needs the session's graph and that is server-side state.

        Same registry as every other session here, so this serves a rollout from ANY harness --
        opencode, codex, claude-code, a Harbor task -- without knowing which produced it.
        """
        if not _admin_ok(request):
            return _forbidden()
        session = app.state.registry.get(session_id)
        if session is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        # `to_trace_entries` refuses a non-trainable document rather than silently handing back
        # rows with empty token fields, which is the failure this whole contract exists to avoid.
        document = export_session(
            session, include_messages=True, capture_level=_level_of(session)
        )
        try:
            return {
                "session_id": session_id,
                "entries": to_trace_entries(session.graph, document),
            }
        except Exception as exc:
            return JSONResponse(
                {"error": f"{type(exc).__name__}: {exc}", "session_id": session_id},
                status_code=409,
            )

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request) -> Any:
        if not _admin_ok(request):
            return _forbidden()
        return {"deleted": app.state.registry.delete(session_id)}

    @app.get("/v1/models")
    async def models() -> Any:
        """The default engine's model list.

        Deliberately not session-scoped: this route is unauthenticated and answers before any session
        exists, and some harnesses call it to decide whether their configured model is available. With
        no default engine there is nothing to list, and an empty list is the honest answer rather than
        a 500.
        """
        if app.state.inference is None:
            return {"object": "list", "data": []}
        return await app.state.inference.list_models()

    @app.post("/{path:path}")
    async def proxy(path: str, request: Request) -> Any:
        """Catch-all: /v1/chat/completions, /v1/messages, /v1/responses, :generateContent."""
        headers = dict(request.headers)
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return _invalid_request("body must be JSON")
        if not isinstance(body, dict):
            return _invalid_request("body must be a JSON object")

        # Answered, never recorded. Must come before session routing and dialect handling: an aux
        # route is not a model turn, so it has no business creating a node or a session.
        aux = aux_dialect(path)
        if aux is not None:
            logger.info("aux route %s (answered as %s, not recorded)", path, aux)
            return JSONResponse(
                aux_token_count_response(aux, approximate_token_count(body))
            )

        session = app.state.registry.resolve(headers, body)
        if session is None:
            # Deliberately 401 rather than serving an unknown caller: this port is public.
            return JSONResponse(
                {
                    "error": {
                        "message": "unknown API key; register a session via POST /sessions",
                        "type": "invalid_request_error",
                    }
                },
                status_code=401,
            )

        api_type: APIType = detect(f"/{path}", headers, body)
        transformer = app.state.transforms.get(api_type)

        import copy

        original_request = copy.deepcopy(body)
        # Include the query string: Google puts `alt=sse` there, not in the body.
        full_target = (
            f"/{path}?{request.url.query}" if request.url.query else f"/{path}"
        )
        client_wants_stream = wants_stream(full_target, body)

        # BUDGET CHECK, BEFORE ANYTHING IS FORWARDED OR RECORDED.
        #
        # Placed after `client_wants_stream` deliberately. Answering a STREAMING request with a plain
        # JSON body does not end the agent's loop -- opencode streams, and it simply retried, so the
        # proxy answered the stop over and over while the rollout burned its whole timeout. The reply
        # has to go back in the dialect the caller asked for, which is what `sse.replay` is for.
        #
        # The same experiment that showed opencode ignores its own `steps` setting also showed what
        # DOES end its loop: a plain assistant message with `finish_reason="stop"` and no tool calls
        # terminates cleanly and `opencode run` exits 0. The content must be non-empty -- an empty
        # assistant message reads as a failed generation and gets retried.
        #
        # Because this returns before `_ingest`, CAPTURE NEVER SEES IT, so no turn the model did not
        # generate can enter the training data. It bounds cost and wall clock and nothing else: it is
        # not a nudge and says nothing about the task, because shaping what an agent does with its
        # last turns is the environment's business, not every environment's.
        if session.over_budget:
            logger.info(
                "session %s hit its model-call budget (%d); ending the agent loop",
                session.session_id,
                session.max_model_calls,
            )
            session.budget_stop_count += 1
            stop = _budget_stop_response(_model_of(session) or app.state.model)
            if client_wants_stream:
                return StreamingResponse(
                    sse.replay(api_type, transformer, stop, original_request),
                    media_type="text/event-stream",
                    headers=sse.SSE_HEADERS,
                )
            payload = transformer.transform_response(stop, original_request)
            normalise_client_payload(payload, api_type)
            return JSONResponse(payload)
        # The served model name has to be on the body BEFORE the transformer runs: each dialect
        # reads `_served_model` inside `transform_request` to decide per-model request fixes, and
        # `BaseTransformer._normalize_request` strips it again on the way out. Setting it afterwards,
        # as the upstream client used to, meant the transformers never saw it (so the Qwen3.5
        # thinking fix silently never applied) and the marker travelled on to the engine unused.
        incoming = dict(body)
        served_model = _model_of(session)
        if served_model:
            incoming["_served_model"] = served_model
        try:
            chat_request = transformer.transform_request(incoming)
        except UpstreamRequestError as exc:
            return _invalid_request(str(exc))
        if chat_request.get("n", 1) != 1 or isinstance(chat_request.get("n"), bool):
            return _invalid_request(
                "capture supports exactly one completion per request (n=1)"
            )
        if served_model:
            chat_request["model"] = served_model
        normalise_for_capture(chat_request)
        requested_sampling = {
            key: chat_request[key] for key in SAMPLING_KEYS if key in chat_request
        }
        if getattr(session, "eval_sampling", None):
            chat_request.update(session.eval_sampling)
        if session.sampling:
            # These alter logits beyond the full-vocabulary, temperature-only policy the
            # trainer recomputes. Capture cannot reconstruct an unreported grammar mask.
            constrained = [
                key
                for key in (
                    "logit_bias",
                    "allowed_token_ids",
                    "structured_outputs",
                    "guided_json",
                    "guided_regex",
                    "guided_choice",
                    "guided_grammar",
                    "min_tokens",
                )
                if chat_request.get(key)
            ]
            response_format = chat_request.get("response_format") or {}
            if not isinstance(response_format, dict):
                return _invalid_request("response_format must be a JSON object")
            if response_format.get("type", "text") != "text":
                constrained.append("response_format")
            tool_choice = chat_request.get("tool_choice")
            if tool_choice is not None and tool_choice not in ("auto", "none"):
                constrained.append("tool_choice")
            if constrained:
                return _invalid_request(
                    "training sampling cannot reproduce constrained logits: "
                    + ", ".join(constrained)
                )
            chat_request.update(session.sampling)
        output_cap = session.metadata.get("max_output_tokens")
        if output_cap is not None and (type(output_cap) is not int or output_cap < 1):
            return _invalid_request(
                "session max_output_tokens must be a positive integer"
            )
        if app.state.max_output_tokens:
            output_cap = (
                min(output_cap, app.state.max_output_tokens)
                if output_cap
                else app.state.max_output_tokens
            )
        clamped = clamp_output_tokens(chat_request, output_cap)
        if clamped:
            logger.info(
                "clamped requested output tokens %d -> %d",
                clamped,
                output_cap,
            )

        upstream_client, session_level = await _upstream_for(session)
        if session.sampling and session_level != "tokens":
            return _invalid_request(
                "training sampling requires an endpoint with token capture"
            )
        if upstream_client is None:
            # No engine on the session and none on the server. Saying so beats forwarding to an empty
            # base URL, which surfaces as a connection error that names neither cause.
            return JSONResponse(
                {
                    "error": {
                        "message": "no inference engine for this session: name one as `llm_url` "
                        "when creating the session, or boot the server with --llm-url"
                    }
                },
                status_code=503,
            )
        # Reserve after validation and upstream resolution. No await separates the check and
        # increment, so concurrent requests on the capture loop cannot exceed the session budget.
        if session.over_budget:
            session.budget_stop_count += 1
            stop = _budget_stop_response(_model_of(session) or app.state.model)
            if client_wants_stream:
                return StreamingResponse(
                    sse.replay(api_type, transformer, stop, original_request),
                    media_type="text/event-stream",
                    headers=sse.SSE_HEADERS,
                )
            payload = transformer.transform_response(stop, original_request)
            normalise_client_payload(payload, api_type)
            return JSONResponse(payload)
        if (
            getattr(upstream_client, "provider", "openai") == "anthropic"
            and api_type == APIType.ANTHROPIC
        ):
            chat_request["_openenv_native_request"] = {
                **original_request,
                **getattr(session, "eval_sampling", {}),
            }
            chat_request["_openenv_native_headers"] = {
                key: request.headers[key]
                for key in ("anthropic-beta", "anthropic-version")
                if key in request.headers
            }
        session.model_calls += 1

        async def complete_response():
            try:
                response = await upstream_client.completion(chat_request)
            except UpstreamError as exc:
                # A known context limit ends the rollout's compute budget. Let the
                # harness terminate normally so Harbor can grade its current answer.
                # Like the model-call stop, this control response is never ingested.
                if isinstance(exc, UpstreamHTTPError) and exc.status_code == 400:
                    error = (
                        exc.body.get("error", exc.body)
                        if isinstance(exc.body, dict)
                        else {}
                    )
                    code = error.get("code") if isinstance(error, dict) else None
                    message = str(exc).lower()
                    if code == "context_length_exceeded" or (
                        "maximum context length" in message and "tokens" in message
                    ):
                        session.findings.append(
                            "[WARN] context_budget_exhausted: " + str(exc)
                        )
                        session.budget_stop_count += 1
                        stop = _budget_stop_response(
                            _model_of(session) or app.state.model
                        )
                        if client_wants_stream:
                            return StreamingResponse(
                                sse.replay(
                                    api_type, transformer, stop, original_request
                                ),
                                media_type="text/event-stream",
                                headers=sse.SSE_HEADERS,
                            )
                        payload = transformer.transform_response(stop, original_request)
                        normalise_client_payload(payload, api_type)
                        return JSONResponse(payload)
                session.upstream_errors += 1
                logger.warning("upstream error [%s]: %s", session.session_id, exc)
                # Preserve a permanent request error: converting a 400 (for example,
                # an image sent to a text-only engine) into 502 makes harnesses retry
                # unchanged input for minutes. Transient failures retain the gateway
                # response, and auth statuses are not exposed as harness credentials.
                from .providers import ProviderConversionError

                status = (
                    400
                    if isinstance(exc, ProviderConversionError)
                    else exc.status_code
                    if isinstance(exc, UpstreamHTTPError)
                    and exc.status_code in (400, 413, 422)
                    else 502
                )
                return JSONResponse(
                    {"error": {"message": str(exc)}}, status_code=status
                )

            native_response = response.pop("_openenv_native_response", None)
            effective_sampling = response.pop("_openenv_sampling", None)
            if session.sampling and (
                effective_sampling is None
                or any(
                    effective_sampling.get(key) != value
                    for key, value in session.sampling.items()
                )
            ):
                session.findings.append(
                    "[FATAL] sampling_policy_changed: upstream did not preserve the session training policy"
                )
                session.upstream_errors += 1
                return JSONResponse(
                    {
                        "error": {
                            "message": "upstream did not preserve the session training sampling policy"
                        }
                    },
                    status_code=502,
                )
            if effective_sampling is not None:
                for key in SAMPLING_KEYS:
                    chat_request.pop(key, None)
                chat_request.update(effective_sampling)
            normalise_response(response)
            _ingest(
                session,
                chat_request,
                response,
                api_type,
                session_level,
                requested_sampling,
            )

            if native_response is not None and api_type == APIType.ANTHROPIC:
                if client_wants_stream:
                    from .providers import replay_anthropic

                    return StreamingResponse(
                        replay_anthropic(native_response),
                        media_type="text/event-stream",
                        headers=sse.SSE_HEADERS,
                    )
                return JSONResponse(native_response)
            if client_wants_stream:
                return StreamingResponse(
                    sse.replay(api_type, transformer, response, original_request),
                    media_type="text/event-stream",
                    headers=sse.SSE_HEADERS,
                )
            payload = transformer.transform_response(response, original_request)
            normalise_client_payload(payload, api_type)
            return JSONResponse(payload)

        if client_wants_stream:
            return await sse.keepalive_response(complete_response(), api_type)
        return await complete_response()

    def _ingest(
        session,
        chat_request: dict[str, Any],
        response: dict[str, Any],
        api_type: APIType,
        capture_level: str,
        requested_sampling: dict[str, Any] | None = None,
    ) -> None:
        """Turn one upstream response into a graph node, validating before it lands.

        Never raises. A capture problem must degrade one turn, not kill a rollout that is otherwise
        producing usable data, and certainly not take down the server serving every other rollout.
        """
        try:
            choices = response.get("choices") or []
            if not choices:
                # A 200 with no choices is not a turn. Recording it produced a node with no prompt and
                # no completion, which inflated n_turns and n_roots and could push a worthless rollout
                # past the `degenerate_rollout` FATAL that exists to catch exactly that.
                logger.warning(
                    "[%s] upstream returned 200 with no choices; not recording a turn",
                    session.session_id,
                )
                session.upstream_errors += 1
                return
            choice = choices[0] or {}
            logprob_entries = (choice.get("logprobs") or {}).get("content") or []
            logprobs = [e.get("logprob") for e in logprob_entries] or None
            # A None inside the list passes the length check and then crashes export on `None > 0.0`,
            # turning `GET /sessions/{id}/rollout` into a 500. Missing values mean the turn cannot be
            # trained on, which is what dropping them to None already signals.
            if logprobs is not None and any(lp is None for lp in logprobs):
                logger.warning(
                    "[%s] %d of %d logprobs are null; treating the turn as untrainable",
                    session.session_id,
                    sum(1 for lp in logprobs if lp is None),
                    len(logprobs),
                )
                logprobs = None
            sampled_ids = choice.get("token_ids") or []
            prompt_ids = response.get("prompt_token_ids") or []
            index = session.graph.stats()["n_turns"]

            if capture_level != "tokens":
                # An eval turn. Grade what an eval turn can be graded on and keep going: running
                # `check_turn` here would report `no_prompt_ids` and `no_logprobs` as FATAL on every
                # single turn, which is true and useless — it is the known, accepted property of the
                # endpoint, decided before the server started.
                #
                # Any logprobs that did come back are kept on the node for the confidence readout,
                # but they are not a training signal: with no token ids there is nothing to align
                # them to, so `export` never promotes them to `per_token_logps`.
                report = check_turn_eval(
                    choice.get("message"),
                    finish_reason=choice.get("finish_reason"),
                    index=index,
                )
                session.findings.extend(str(f) for f in report.findings)
            else:
                report = check_turn(
                    prompt_ids,
                    sampled_ids,
                    logprobs,
                    finish_reason=choice.get("finish_reason"),
                    index=index,
                )
                session.findings.extend(str(f) for f in report.findings)
                if not report.ok:
                    logger.warning(
                        "[%s] turn %d rejected: %s",
                        session.session_id,
                        index,
                        "; ".join(str(f) for f in report.fatal),
                    )
                    # Still recorded, with logprobs dropped: the tokens are real context for later
                    # turns, and `sequence_for` masks a turn whose logprobs it cannot trust.
                    logprobs = None

            # A rewritten request is how an eval number becomes unreproducible, so say so — once per
            # session, since every turn of a given harness sends the same knobs.
            if capture_level == "tokens":
                overridden = truncating_params(requested_sampling or chat_request)
                if overridden and not session.metadata.get("sampling_overridden"):
                    session.metadata["sampling_overridden"] = overridden
                    session.findings.append(
                        "WARN sampling_neutralised: the harness asked for "
                        + ", ".join(f"{k}={v}" for k, v in sorted(overridden.items()))
                        + "; these were sent upstream at their no-op values because a processed "
                        "logprob is taken after they are applied, which would bias a full-vocab "
                        "recompute. requested_sampling_params preserves the harness request; "
                        "sampling_params records the submitted policy."
                    )

            session.graph.add_turn(
                TurnNode(
                    node_id=uuid.uuid4().hex[:12],
                    prompt_ids=list(prompt_ids),
                    sampled_ids=list(sampled_ids),
                    sampled_logprobs=list(logprobs) if logprobs else None,
                    model=chat_request.get("model"),
                    finish_reason=choice.get("finish_reason"),
                    harness_session_id=extract_harness_session({}, chat_request),
                    system_digest=_system_digest(chat_request.get("messages") or []),
                    n_tools=len(chat_request.get("tools") or []),
                    request_messages=chat_request.get("messages") or [],
                    request_tools=chat_request.get("tools"),
                    sampling_params={
                        key: chat_request[key]
                        for key in SAMPLING_KEYS
                        if chat_request.get(key) is not None
                    },
                    requested_sampling_params=requested_sampling or {},
                    response_message=choice.get("message") or {},
                )
            )
            session.last_turn_at = __import__("time").time()
            session.metadata.setdefault("api_type", api_type.value)
        except Exception:  # noqa: BLE001
            session.findings.append(
                "[FATAL] capture_ingest_failed: an upstream response could not be recorded"
            )
            logger.exception("[%s] ingest failed; turn dropped", session.session_id)

    return app


def _is_loopback_host(host: str) -> bool:
    """Whether `host` is a loopback-only bind (private local port).

    `0.0.0.0` / `::` are intentionally not loopback: they publish on every interface.
    """
    return host in {"127.0.0.1", "::1", "localhost"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--llm-url", required=True, help="OpenAI-spec endpoint you host"
    )
    parser.add_argument(
        "--model", default=None, help="served model name to send upstream"
    )
    parser.add_argument(
        "--engine",
        default="",
        choices=["", "vllm", "sglang"],
        help="what is on the other end, for /health only. Left unset by default because the "
        "upstream may be a hosted provider, and reporting one of these two when it is not says "
        "something untrue about capture.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (default: 127.0.0.1). Non-loopback binds require --admin-key or "
        "$OPENENV_CAPTURE_ADMIN_KEY so /sessions* is never left open on a reachable interface.",
    )
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
        help="cap on requested completion length; 0 disables",
    )
    parser.add_argument(
        "--allow-unregistered",
        action="store_true",
        help="serve unknown API keys (local debugging only; this port may be public)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENENV_LLM_API_KEY", ""),
        help="credential for the UPSTREAM endpoint (defaults to $OPENENV_LLM_API_KEY). Not the "
        "agent-facing key: that is a capture session id, minted per rollout.",
    )
    parser.add_argument(
        "--auth-header",
        default="Authorization",
        help="header to send --api-key under; `Authorization` gets a Bearer prefix, anything else "
        "(e.g. x-api-key) gets the raw key",
    )
    parser.add_argument(
        "--capture-level",
        default="",
        choices=["", "tokens", "logprobs", "text"],
        help="what the upstream can return. Probed from the endpoint when omitted, which is the "
        "recommended path; pass it only to force a level.",
    )
    parser.add_argument(
        "--admin-key",
        default=os.environ.get("OPENENV_CAPTURE_ADMIN_KEY", ""),
        help="key the session-management routes (/sessions*) require (defaults to "
        "$OPENENV_CAPTURE_ADMIN_KEY). Required for non-loopback binds. Never printed or logged.",
    )
    args = parser.parse_args()

    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    level = args.capture_level
    if not level:
        from .validate_llm import validate_llm

        report = validate_llm(
            args.llm_url,
            args.model or "",
            api_key=args.api_key or None,
            auth_header=args.auth_header,
        )
        if not report.reachable:
            raise SystemExit(report.summary())
        level = report.capture_level
        print(f"capture level: {level} ({report.rollout_type} rollouts)")
        for fix in report.param_fixes:
            print(f"  upstream compat: {fix}")

    # Flag/env only — never mint-and-print (that would log a credential). Loopback may leave the
    # key unset for private local use; non-loopback must supply one explicitly.
    admin_key = args.admin_key or None
    if not admin_key and not _is_loopback_host(args.host):
        raise SystemExit(
            f"--admin-key or $OPENENV_CAPTURE_ADMIN_KEY is required when --host={args.host} "
            "(non-loopback bind would publish /sessions* ungated). "
            "Pass --host 127.0.0.1 for a private local port, or set a key."
        )

    uvicorn.run(
        create_app(
            llm_url=args.llm_url,
            model=args.model,
            engine=args.engine,
            require_registered=not args.allow_unregistered,
            max_output_tokens=args.max_output_tokens or None,
            api_key=args.api_key or None,
            auth_header=args.auth_header,
            capture_level=level,
            admin_key=admin_key,
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
