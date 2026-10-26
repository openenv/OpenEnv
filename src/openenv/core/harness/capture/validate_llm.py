"""Work out what an inference endpoint can actually return, before anything is spent on it.

This runs BEFORE a server binds a port, and grading honestly here is the entire point.

An engine missing `--return-tokens-as-token-ids --logprobs-mode processed_logprobs` still answers
every request perfectly well: it returns text, a `200`, and plausible-looking usage. What it does not
return is token ids. Every row rebuilt downstream is then empty, training silently does nothing, and
the first symptom is a loss curve that never moves days later. That failure has no loud edge, so the
check has to be up front — and the answer has to travel with everything the endpoint later produces.

The probe therefore settles a **capture level**, not a yes/no:

    tokens      prompt ids, sampled ids, aligned logprobs. Trainable. vLLM with the two flags, or
                SGLang from main.
    logprobs    logprobs, no ids. Not trainable: an unpaired logprob has no token to attach to.
    text        neither.

Below `tokens` the endpoint is an eval backend — reward and full trace, no token fields — which is a
real, useful thing rather than a failure, so the server runs instead of refusing. What it must never
do is present an eval rollout as a training one, which is why the level is stamped on the result and
no contract is written without `tokens`.

The probe also **negotiates**: hosted providers reject the capture params, and reject them per model
rather than per endpoint. `return_token_ids` is a 400 on OpenAI; `logprobs` is a 400 on every current
OpenAI model; `max_tokens` and `temperature: 0` are 400s there too. So a rejection is read (see
`compat.diagnose`), the offending param is dropped or renamed, and the request is retried — the level
is decided by what finally came back, never by what we hoped to send.

Two engines implement the contract. vLLM, served with
`--return-tokens-as-token-ids --logprobs-mode processed_logprobs`. And SGLang built from `main`:
sgl-project/sglang#30917 (merged 2026-07-23) added `return_token_ids` to the OpenAI-compatible
routes, which is what sgl-project/sglang#18378 had asked for. Released SGLang is still unusable —
v0.5.16 carries only `return_prompt_token_ids`, the prompt without the sampled ids — so the version
matters and nothing but a live probe can tell the two apart, which is what this file does. SGLang
also puts the prompt ids per choice rather than at the top level; `normalize_response` absorbs that,
so the probe below runs its payload through it before grading.

A hosted alternative exists but is narrow: fireworks-ai via the HF router honours vLLM's
`return_token_ids`, though every one of its live models is a reasoning model whose reasoning tokens
are dropped from history, so multi-turn stitching degrades to per-turn.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .compat import behaviour_warnings, diagnose, MAX_FIXES
from .upstream import (
    auth_headers,
    normalise_engine_base,
    normalize_response,
    openai_api_base,
)
from .validate import check_upstream_response

# The one place the engine requirements are spelled out. Three call sites used to carry their own
# copy of this text and two of them still said "vLLM" only, months after SGLang started working.
ENGINE_HINT = (
    "Token capture needs one of:\n"
    "  - vLLM, served with --return-tokens-as-token-ids --logprobs-mode processed_logprobs\n"
    "  - SGLang built from git main (sgl-project/sglang#30917); no serving flag needed, and its "
    "logprobs are already temperature-scaled\n"
    "Any other OpenAI-spec endpoint (OpenAI, Anthropic, HF Inference Providers) still works, as an "
    "EVAL backend: you get the reward and the full trace, but no token ids or logprobs, so nothing "
    "from it is trainable."
)


@dataclass
class LLMReport:
    """Outcome of the probe. `ok` means trainable; `reachable` means usable at all."""

    ok: bool
    llm_url: str
    model: str
    findings: list[str] = field(default_factory=list)
    n_prompt_ids: int = 0
    n_completion_ids: int = 0
    served_models: list[str] = field(default_factory=list)
    # "tokens" | "logprobs" | "text", or "" when nothing came back at all.
    capture_level: str = ""
    reachable: bool = False
    # Params this endpoint rejected and how we worked around them, as human-readable strings. Carried
    # so that a rewritten request is visible: dropping `temperature` changes the sampling
    # distribution, which makes an eval number irreproducible if nobody is told.
    param_fixes: list[str] = field(default_factory=list)
    # "ok" | "no-tool-call" | "rejected" | "unknown", or "" when not probed. Every validated harness
    # sends a tool manifest on every call, and the capture probe sends none — so this is the only
    # signal about whether a coding agent can work here at all.
    tool_support: str = ""
    # "processed" | "raw" | "unknown", or "" when the question was not asked (only `tokens`-level
    # endpoints are asked, since nothing below it is trainable anyway). See `probe_logprobs_mode`.
    logprobs_mode: str = ""

    @property
    def trainable(self) -> bool:
        return self.capture_level == "tokens"

    @property
    def rollout_type(self) -> str:
        return "train" if self.trainable else "eval"

    def summary(self) -> str:
        if self.ok:
            return (
                f"engine OK: {self.n_completion_ids} completion ids, "
                f"{self.n_prompt_ids} prompt ids"
            )
        if self.reachable:
            detail = (
                "logprobs but no token ids"
                if self.capture_level == "logprobs"
                else "no token ids and no logprobs"
            )
            return (
                f"endpoint is EVAL ONLY ({detail}); rollouts carry reward and trace but "
                "nothing trainable"
            )
        return "engine NOT reachable:\n  " + "\n  ".join(self.findings)


# Above this magnitude a "logprob" is an engine sentinel for -inf, not a measurement (vLLM uses
# -9999). Real values stay in the tens even in the tail of a 150k vocabulary.
_SENTINEL_LOGPROB = 100.0


def _raw_logprobs_allowed() -> bool:
    """Whether an operator has explicitly accepted raw logprobs on the training path.

    An escape hatch exists because the probe infers from behaviour: the two vLLM flags are
    independent, and someone may have a configuration this cannot see. A refusal that cannot be
    overridden becomes a reason to stop trusting the tool.
    """
    return os.environ.get("OPENENV_ALLOW_RAW_LOGPROBS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _post(
    url: str,
    body: dict,
    timeout: float,
    api_key: str | None = None,
    auth_header: str = "Authorization",
) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "OpenEnv-provider-probe/1.0",
            **auth_headers(api_key, auth_header),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def list_models(
    llm_url: str,
    timeout: float = 30.0,
    api_key: str | None = None,
    auth_header: str = "Authorization",
) -> list[str]:
    """Served model ids, or [] if the endpoint is unreachable or does not publish a list."""
    try:
        request = urllib.request.Request(
            f"{openai_api_base(llm_url)}/models",
            headers=auth_headers(api_key, auth_header),
        )
        with urllib.request.urlopen(request, timeout=timeout) as r:
            return [m.get("id", "") for m in json.loads(r.read()).get("data", [])]
    except Exception:  # noqa: BLE001 - unreachable is reported by the caller, not raised here
        return []


# Findings that say nothing beyond "this endpoint is not `tokens`", which the level already says.
# Deliberately does NOT include `no_choices`: an endpoint that answered without a `choices` array is
# broken rather than merely eval-only, and that has to keep surfacing.
_EXPECTED_BELOW_TOKENS = frozenset(
    {
        "no_prompt_token_ids",
        "no_completion_token_ids",
        "no_logprobs",
        "token_strings",
    }
)


def _grade(payload: dict) -> tuple[str, list[str]]:
    """The capture level a response payload supports, and the findings behind that verdict.

    Grades the payload the same way the capture path will see it. `InferenceClient.completion`
    normalises every response before it reaches `_ingest`, so grading the raw body instead would
    reject an SGLang endpoint (per-choice prompt ids) that the rollout path handles perfectly well —
    a false negative in the one check whose whole job is to be trusted.
    """
    report = check_upstream_response(payload)
    if report.ok:
        return "tokens", [str(f) for f in report.findings]

    # Not `tokens`, so the only remaining question is whether logprobs came back at all. Read from
    # the same place `_ingest` reads them, so the two cannot disagree.
    choice = (payload.get("choices") or [{}])[0]
    entries = ((choice.get("logprobs") or {}).get("content")) or []
    has_logprobs = any(
        isinstance(entry, dict) and entry.get("logprob") is not None
        for entry in entries
    )
    level = "logprobs" if has_logprobs else "text"

    # Drop the findings that merely restate the tier. "no top-level prompt_token_ids", reported as
    # FATAL, is the *definition* of an eval endpoint, and printing three fatal-looking lines under a
    # heading that already says EVAL ONLY reads as a broken endpoint rather than a correctly
    # classified one — which is how a findings list stops being read at all. Anything else the probe
    # noticed still comes through.
    expected = _EXPECTED_BELOW_TOKENS
    return level, [
        str(f) for f in report.findings if getattr(f, "code", None) not in expected
    ]


_PROBE_TOOL = [
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


def probe_tool_support(
    base: str,
    model: str,
    *,
    timeout: float = 60.0,
    api_key: str | None = None,
    auth_header: str = "Authorization",
    fixes: list | None = None,
) -> tuple[str, list]:
    """Can this endpoint take a tool manifest, and what does it demand in exchange?

    A separate probe from the capture one, and deliberately NON-FATAL, because the two questions have
    different consequences. The capture probe decides reachability and the training tier; this decides
    whether a *coding agent* can work here, which no other check looks at — every validated harness
    sends a tool manifest on every call, and the capture probe sends none.

    It is also the only place a specific class of problem is visible before a rollout is spent. Some
    models accept tools only on terms that change their behaviour:

        gpt-5.6: "Function tools with reasoning_effort are not supported ... use /v1/responses or set
                  reasoning_effort to 'none'."

    The shim satisfies that by turning reasoning off, after which the model emits a valid first tool
    call and then agentic loops die — goose and codex each managed exactly one model call and 0/3
    tasks, while scoring 3/3 against a non-reasoning model on the same endpoint. Without a tool in the
    probe body that demand is never made, so `harbor info` looked perfectly healthy and the failure
    only appeared minutes into a rollout. Hence: send a tool, and report what it cost.

    Returns:
        `tuple[str, list]`: one of `"ok"` (a tool call came back), `"no-tool-call"` (accepted the
        manifest but answered in prose), `"rejected"` (will not take tools at all) or `"unknown"`,
        plus any additional [`ParamFix`] objects the endpoint demanded.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Use the bash tool to list the files in /tmp."}
        ],
        "tools": _PROBE_TOOL,
        "tool_choice": "auto",
        # Generous on purpose. A reasoning model spends output tokens thinking before it emits the
        # call, so a small cap truncates it mid-thought and looks exactly like "will not use tools":
        # Qwen3.6-35B-A3B produced 224 tokens of reasoning and `finish_reason: length` at 64, and was
        # reported as tool-incapable while in fact working with all 16 harnesses.
        "max_tokens": 512,
        "temperature": 1.0,
    }
    extra: list = []
    for fix in fixes or []:
        fix.apply(body)

    while True:
        try:
            payload = _post(
                f"{openai_api_base(base)}/chat/completions",
                body,
                timeout,
                api_key=api_key,
                auth_header=auth_header,
            )
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:600].decode(errors="replace")
            fix = None
            if exc.code == 400 and len(extra) < MAX_FIXES:
                try:
                    fix = diagnose(json.loads(detail))
                except json.JSONDecodeError:
                    fix = diagnose(detail)
            if fix is None or not fix.apply(body):
                # `tools` is protected from being dropped, so an endpoint that simply cannot do tool
                # calling ends up here rather than silently having the manifest removed.
                return "rejected", extra
            extra.append(fix)
        except Exception:  # noqa: BLE001
            return "unknown", extra

    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    if message.get("tool_calls"):
        return "ok", extra
    # Truncated before it could decide is not evidence about tool support. Reporting it as a failure
    # is how a warning that fires on a healthy endpoint teaches everyone to ignore warnings.
    if choice.get("finish_reason") == "length":
        return "unknown", extra
    return "no-tool-call", extra


def probe_logprobs_mode(
    base: str,
    model: str,
    *,
    timeout: float = 60.0,
    api_key: str | None = None,
    auth_header: str = "Authorization",
    fixes: list | None = None,
) -> str:
    """Whether an endpoint's logprobs are the sampling distribution's or the raw pre-temperature ones.

    This closes the one hole no other check here can see. vLLM's `logprobs_mode` defaults to
    `raw_logprobs` (`vllm/config/model.py`), documented as the values "before applying any logit
    processors, **including temperature and top_k/top_p**". Raw logprobs are aligned, negative,
    correctly counted, and wrong: GRPO's importance ratio needs the logprob under the policy that
    actually sampled the token. `token_ids` arrives either way, because it comes from the request
    parameter rather than from a serving flag, so an engine launched with neither flag produces a
    rollout that grades as fully trainable and trains on the wrong numbers.

    The test follows from the definition, and compares the GAP between the top two tokens rather
    than absolute values. That matters: processed logprobs are `logsoftmax(logits / T)`, so a
    difference between two of them is `(logit_a - logit_b) / T` — the normalising constant cancels.
    The gap is therefore invariant under renormalisation and under any per-replica offset, which
    absolute comparison is not. A first attempt compared values directly and misread a
    data-parallel engine (DP=4) as processed, because consecutive calls landed on different replicas.

    Measured, three repeats each, on two live Qwen3.5-4B servers:

        --logprobs-mode processed_logprobs   gap 6.7500 @T=1.0  ->  3.3750 @T=2.0   ratio 0.500
        default (raw_logprobs)               gap 6.7500 @T=1.0  ->  6.7500 @T=2.0   ratio 1.000

    So the expected ratio is `T1 / T2` when processed and `1.0` when raw, and the two are a factor of
    two apart. No `seed` is sent and the sampled token is ignored: `top_logprobs` at the first
    position is a property of the prompt and the temperature alone.

    Args:
        base (`str`):
            OpenAI-compatible base URL, with or without `/v1`.
        model (`str`):
            Model id to probe.
        timeout (`float`, *optional*, defaults to `60.0`):
            Per-request ceiling.
        api_key (`str`, *optional*):
            Upstream credential.
        auth_header (`str`, *optional*, defaults to `"Authorization"`):
            Header to send it under.
        fixes (`list[ParamFix]`, *optional*):
            Workarounds already discovered for this endpoint, applied to both probes.

    Returns:
        `str`: `"processed"`, `"raw"`, or `"unknown"` when the endpoint cannot answer the question —
        which is not a failure, only an absence of evidence.
    """
    applied = list(fixes or [])
    # A provider that rejected `temperature` outright cannot be asked a question about temperature.
    if any(getattr(f, "param", "") == "temperature" for f in applied):
        return "unknown"

    cool, hot = 1.0, 2.0
    gaps: list[float] = []
    for temperature in (cool, hot):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
            "max_tokens": 1,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": 5,
        }
        for fix in applied:
            fix.apply(body)
        try:
            payload = _post(
                f"{openai_api_base(base)}/chat/completions",
                body,
                timeout,
                api_key=api_key,
                auth_header=auth_header,
            )
        except Exception:  # noqa: BLE001 - no evidence is a valid outcome, not an error
            return "unknown"
        entries = (
            ((payload.get("choices") or [{}])[0].get("logprobs") or {}).get("content")
        ) or []
        if not entries:
            return "unknown"
        values = sorted(
            (
                float(t["logprob"])
                for t in ((entries[0] or {}).get("top_logprobs") or [])
                if isinstance(t, dict) and t.get("logprob") is not None
            ),
            reverse=True,
        )
        if len(values) < 2:
            return "unknown"
        # Too PEAKED to divide by, the mirror of the flatness guard below. When the runner-up is
        # -inf the engine reports a sentinel (vLLM: -9999), and +-inf is unchanged by division, so
        # the ratio is 1.0 at every temperature and processed logprobs get misread as raw.
        #
        # This is not hypothetical. A reasoning model's chat template FORCES its first token --
        # `<think>` for Qwen3, at p~1.0 with every alternative at -inf -- and position 1 is exactly
        # where this probe measures, by design, because it is the only position independent of
        # sampling. Measured on a live Qwen3-8B served WITH --logprobs-mode processed_logprobs:
        #
        #     position 1 (forced <think>):  gap 9999.0 @T=1.0 -> 9999.0 @T=2.0   ratio 1.000  "raw"
        #     position 1 after prefilling past <think>:
        #                                   gap 3.7500 @T=1.0 -> 1.8750 @T=2.0   ratio 0.500  processed
        #
        # Three trials each; the second form is the same method at a non-degenerate position and
        # agrees with the flag the engine was actually launched with. Without this guard every
        # reasoning model with a forced opening token is demoted to the eval tier and every rollout
        # comes back 409, which reads as "this model cannot train" rather than "this probe cannot
        # measure". Real logprobs never approach this magnitude (a 150k-vocab tail bottoms out in the
        # tens), so the threshold separates sentinels from data without needing to know the engine's
        # chosen sentinel value.
        if abs(values[1]) > _SENTINEL_LOGPROB:
            return "unknown"
        gaps.append(values[0] - values[1])

    # Too flat to divide by. A near-uniform distribution makes the ratio noise, and guessing from
    # noise is how a check becomes something people override on principle.
    if gaps[0] < 0.5:
        return "unknown"

    ratio = gaps[1] / gaps[0]
    expected_processed = cool / hot
    return "raw" if abs(ratio - 1.0) < abs(ratio - expected_processed) else "processed"


def _validate_anthropic(llm_url, model, *, timeout, api_key, check_tools):
    """Probe native Messages without inferring token IDs from hosted text."""
    base = normalise_engine_base(llm_url)
    report = LLMReport(ok=False, llm_url=base, model=model)
    if not model:
        report.findings.append("Native Anthropic requires an explicit model")
        return report
    body = {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Reply with ok."}],
    }
    if check_tools:
        body["messages"][0]["content"] = "Call report_ok with value ok."
        body["tools"] = [
            {
                "name": "report_ok",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            }
        ]
        body["tool_choice"] = {"type": "tool", "name": "report_ok"}
    request = urllib.request.Request(
        f"{base}/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            **auth_headers(api_key, "x-api-key"),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
        if payload.get("type") != "message" or not isinstance(
            payload.get("content"), list
        ):
            raise ValueError("Native Messages returned an invalid response shape")
        if not payload["content"]:
            raise ValueError("Native Messages returned no content")
        report.reachable = True
        report.capture_level = "text"
        if check_tools:
            report.tool_support = (
                "ok"
                if any(
                    block.get("type") == "tool_use"
                    and block.get("name") == "report_ok"
                    and isinstance(block.get("input"), dict)
                    for block in payload["content"]
                )
                else "no-tool-call"
            )
    except urllib.error.HTTPError as exc:
        report.findings.append(f"Native Messages HTTP {exc.code}")
    except Exception as exc:
        report.findings.append(f"Native Messages probe failed: {type(exc).__name__}")
    return report


def validate_llm(
    llm_url: str,
    model: str,
    *,
    provider: str = "openai",
    timeout: float = 120.0,
    api_key: str | None = None,
    auth_header: str = "Authorization",
    check_logprobs_mode: bool = True,
    check_tools: bool = True,
) -> LLMReport:
    """Send one real completion and report what the endpoint can return.

    Deliberately a live probe rather than a flag inspection: launch flags are not readable over the
    API, and an engine can be started with the right arguments and still not behave (wrong version,
    a proxy in between that strips fields). The only trustworthy check is asking for a completion and
    looking at what comes back.

    Args:
        llm_url (`str`):
            OpenAI-spec endpoint. Accepts `http://host:8000`, `http://host:8000/v1`, or a
            provider base with its own path such as `.../v1beta/openai`.
        model (`str`):
            Model id to probe.
        timeout (`float`, *optional*, defaults to `120.0`):
            Per-request ceiling.
        api_key (`str`, *optional*):
            Upstream credential, for a token-gated endpoint.
        auth_header (`str`, *optional*, defaults to `"Authorization"`):
            Header to send the credential under.

    Returns:
        [`LLMReport`]: `capture_level`, whether it is trainable, and every workaround applied.
    """
    # Same base rule as the OpenAI SDK, so a URL that works there also works here instead of
    # probing `/v1/v1/models` or `/v1beta/openai/v1/models` and reporting it dead.
    if provider not in {"openai", "anthropic", "hf", "vllm"}:
        raise ValueError(f"Unknown upstream provider: {provider}")
    if provider == "anthropic":
        return _validate_anthropic(
            llm_url, model, timeout=timeout, api_key=api_key, check_tools=check_tools
        )
    base = openai_api_base(llm_url)
    served = list_models(
        base, timeout=min(timeout, 30.0), api_key=api_key, auth_header=auth_header
    )

    # An empty list is no longer fatal on its own. Some hosted gateways do not publish `/v1/models`,
    # or gate it behind different scopes than inference, and refusing there would reject an endpoint
    # that serves completions perfectly well. The completion probe below is the real test; a missing
    # list only costs us the model-name check.
    listed_model = model.rsplit(":", 1)[0] if provider == "hf" else model
    if served and model not in served and listed_model not in served:
        # Worth failing on rather than warning: a mismatched name is silently accepted by some
        # servers and then every request 404s at rollout time instead of at startup.
        return LLMReport(
            ok=False,
            llm_url=base,
            model=model,
            served_models=served,
            findings=[f"model {model!r} is not served here; available: {served}"],
        )

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 0,
        # vLLM >= 0.10.2 exposes this on the OpenAI route. Its ABSENCE from the response is exactly
        # the signal we are testing for; OpenAI rejects the param outright, which `compat` handles.
        "return_token_ids": True,
    }
    fixes: list[str] = []
    # The fix OBJECTS as well as their rendering: the logprobs-mode probe below issues its own
    # requests and has to carry the same workarounds, or it would rediscover every one of them.
    applied: list = []
    payload: dict | None = None
    failure = ""

    # Negotiate down. Each 400 that names a parameter costs one retry and removes one obstacle; the
    # capture params are the ones most likely to go, and losing them is what decides the level.
    while payload is None:
        try:
            payload = _post(
                f"{base}/chat/completions",
                body,
                timeout,
                api_key=api_key,
                auth_header=auth_header,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:600].decode(errors="replace")
            fix = None
            if (
                exc.code == 400
                or (
                    exc.code == 403
                    and "You are not allowed to request logprobs from this model"
                    in detail
                )
            ) and len(fixes) < MAX_FIXES:
                try:
                    fix = diagnose(json.loads(detail))
                except json.JSONDecodeError:
                    fix = diagnose(detail)
            if fix is None or not fix.apply(body):
                failure = f"probe failed: HTTP {exc.code}: {detail[:300]}"
                break
            fixes.append(str(fix))
            applied.append(fix)
        except Exception as exc:  # noqa: BLE001
            failure = f"probe failed: {type(exc).__name__}: {str(exc)[:300]}"
            break

    if payload is None:
        return LLMReport(
            ok=False,
            llm_url=base,
            model=model,
            served_models=served,
            findings=[failure or "probe failed"],
            param_fixes=fixes,
        )

    payload = normalize_response(payload)
    level, findings = _grade(payload)

    # Surfaced at VALIDATE time, before a sandbox or a token is spent. A fix that only respells a
    # field is noise; one that changes how the model behaves decides whether agent rollouts work at
    # all, and the user has no way to know which they got from the fix list alone.
    # A tool manifest is what every harness actually sends, so ask with one. Non-fatal: an endpoint
    # that cannot do tool calling is still a usable eval backend for non-agentic work, and its
    # reachability was already settled above.
    tool_support = ""
    if check_tools:
        tool_support, tool_fixes = probe_tool_support(
            base,
            model,
            timeout=min(timeout, 60.0),
            api_key=api_key,
            auth_header=auth_header,
            fixes=applied,
        )
        for fix in tool_fixes:
            fixes.append(f"{fix} (needed for tool calling)")
            applied.append(fix)
        if tool_support == "rejected":
            findings.append(
                "[FATAL] no_tool_calling: this endpoint will not accept a tool manifest. Every "
                "validated harness sends one on every call, so agent rollouts cannot work here."
            )
        elif tool_support == "no-tool-call":
            findings.append(
                "[WARN] no_tool_call_emitted: the endpoint accepted a tool manifest but answered in "
                "prose instead of calling the tool. Agent rollouts may stall on the first turn."
            )

    findings.extend(
        f"[WARN] behaviour_changed: {w}" for w in behaviour_warnings(applied)
    )
    choice = (payload.get("choices") or [{}])[0]

    # Only worth asking at `tokens`: below it nothing is trainable anyway, and the two extra calls
    # would buy an answer no decision depends on.
    mode = ""
    if level == "tokens" and check_logprobs_mode:
        mode = probe_logprobs_mode(
            base,
            model,
            timeout=min(timeout, 60.0),
            api_key=api_key,
            auth_header=auth_header,
            fixes=applied,
        )
        # `token_strings` inferred, from the logprob token field, exactly what has now been
        # MEASURED. Keeping both prints two paragraphs about one fact, the weaker one first.
        if mode in {"raw", "processed"}:
            findings = [f for f in findings if "token_strings" not in f]
        if mode == "raw":
            if _raw_logprobs_allowed():
                findings.append(
                    "[WARN] raw_logprobs_forced: these logprobs are RAW (pre-temperature) and "
                    "$OPENENV_ALLOW_RAW_LOGPROBS is set, so they are being treated as trainable "
                    "anyway. GRPO's importance ratio will be wrong."
                )
            else:
                # Demoted rather than failed. The endpoint answers perfectly well and is a fine eval
                # backend; what it cannot do is produce a training contract, and `capture_level` is
                # exactly the field that says so. Refusing outright would take away a usable server
                # over a problem that only affects training.
                level = "logprobs"
                findings.append(
                    "[FATAL] raw_logprobs: measured RAW (pre-temperature) logprobs — the same "
                    "token's logprob did not change between temperature 1.0 and 2.0. Token ids "
                    "are present, so nothing else here would have caught this, and the rollout "
                    "would have looked perfectly trainable while carrying a wrong importance "
                    "ratio. Restart the engine with --logprobs-mode processed_logprobs (SGLang "
                    "from main is already temperature-scaled). Downgraded to EVAL; set "
                    "OPENENV_ALLOW_RAW_LOGPROBS=1 to override."
                )

    return LLMReport(
        ok=level == "tokens",
        llm_url=base,
        model=model,
        served_models=served,
        findings=findings,
        n_prompt_ids=len(payload.get("prompt_token_ids") or []),
        n_completion_ids=len(choice.get("token_ids") or []),
        capture_level=level,
        reachable=True,
        param_fixes=fixes,
        logprobs_mode=mode,
        tool_support=tool_support,
    )


def require_llm(
    llm_url: str,
    model: str,
    *,
    timeout: float = 120.0,
    api_key: str | None = None,
    auth_header: str = "Authorization",
    require_tokens: bool = False,
) -> LLMReport:
    """`validate_llm`, but raises when the endpoint cannot be used at all.

    Args:
        require_tokens (`bool`, *optional*, defaults to `False`):
            Also raise when the endpoint is reachable but cannot return token ids. Off by default:
            such an endpoint is a working eval backend, and refusing it would rule out every hosted
            provider. Set it on a path where an eval rollout is worthless — a training run.

    Raises:
        RuntimeError: If the endpoint is unreachable, or `require_tokens` and it is eval-only.
    """
    report = validate_llm(
        llm_url, model, timeout=timeout, api_key=api_key, auth_header=auth_header
    )
    if not report.reachable:
        raise RuntimeError(report.summary() + "\n\n" + ENGINE_HINT)
    if require_tokens and not report.trainable:
        raise RuntimeError(
            report.summary() + "\n\nThis path needs trainable rollouts.\n" + ENGINE_HINT
        )
    return report
