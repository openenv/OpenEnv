"""Turn a provider's 400 into the one edit that would have made the request work.

Hosted providers reject parameters a local vLLM accepts, and they reject them per *model* rather
than per endpoint, so no static table can be right. What they do give is a machine-readable reason,
and it names the offending field:

    max_tokens     Unsupported parameter: 'max_tokens' is not supported with this model.
                   Use 'max_completion_tokens' instead.          (code: unsupported_parameter)
    temperature    Unsupported value: 'temperature' does not support 0 with this model. Only the
                   default (1) value is supported.               (code: unsupported_value)
    logprobs       Unsupported parameter: 'logprobs' is not supported with this model.
    return_token_ids
                   Unrecognized request argument supplied: return_token_ids   (code: null)
                   Unknown parameter: 'return_token_ids'.        (code: unknown_parameter)
    reasoning_effort
                   Function tools with reasoning_effort are not supported for gpt-5.6-sol in
                   /v1/chat/completions. To use function tools, use /v1/responses or set
                   reasoning_effort to 'none'.                   (param: null, code: null)

All were observed live: the first three on `gpt-5.5` and every `gpt-5.6-*`, the last on `gpt-4o-mini`
and `gpt-5.6-*` respectively. Note the last two — the *same* rejection of the *same* parameter,
phrased two ways by two models of the same vendor, one of them not populating `error.param` at all.
That is the reason this reads several shapes rather than matching one string, and the reason
`error.param` is preferred but not required.

It matters because the agent, not this layer, chooses most of these params — opencode, codex and
qwen-coder all send `max_tokens` and a temperature — so a harness that works against vLLM would
otherwise 400 on every single call against a current OpenAI model.

The last one is the sharpest, and the only fix here that changes model *behaviour* rather than
spelling: `reasoning_effort: "none"` turns reasoning off, so what gets evaluated is not quite what the
harness asked for. It is applied anyway because the alternative is that every tools-bearing call fails
and the agent does nothing at all — but for that reason every applied fix is recorded on the result,
in the startup report and in the UI, never silently. The better answer is the Responses route, which
the provider's own message recommends.

`diagnose` is deliberately conservative. It returns a fix only when it can name the parameter, it
never touches a field the request cannot lose (`model`, `messages`, `input`), and a caller is
expected to bound how many fixes it will accept before giving up. Guessing wrong here turns a clear
400 into a silently different request, which is worse than the 400.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Fields whose removal would change the request into a different request, or make it invalid. A
# provider complaining about one of these is telling us something real, and papering over it would
# replace a loud failure with a wrong result.
PROTECTED = frozenset({"model", "messages", "input", "stream", "tools", "tool_choice"})

# A ceiling on how many distinct fixes one endpoint may need. Six covers every provider seen (three
# on gpt-5.6 plus `return_token_ids`); beyond that the disagreement is not about parameter spelling
# and retrying is just spending money on the same 400.
MAX_FIXES = 6

_UNRECOGNISED = re.compile(
    r"[Uu]nrecognized request argument supplied:\s*([A-Za-z0-9_.\[\]]+)"
)
# A parameter name as providers quote it. Anthropic uses backticks where OpenAI uses single quotes,
# for the same job, so both count.
_QUOTED = re.compile(r"['`]([A-Za-z0-9_.]+)['`]")
_USE_INSTEAD = re.compile(r"[Uu]se ['`]([A-Za-z0-9_.]+)['`] instead")
# "... or set reasoning_effort to 'none'." — a provider naming the value that makes the request legal.
_SET_TO = re.compile(r"set ([A-Za-z0-9_.]+) to ['`]([A-Za-z0-9_.-]+)['`]")
# "`temperature` is deprecated for this model." — Anthropic's phrasing, under a generic
# `invalid_request_error` code with a null `param`, so neither the code nor the structured field
# identifies it. The vocabulary is the only signal.
#
# The quotes are REQUIRED, not optional. An earlier version made them optional and matched the bare
# word before "is not supported", which on
#
#     Setting 'max_tokens' and 'max_completion_tokens' at the same time is not supported.
#
# extracted the parameter name **"time"**. Inventing a field name out of prose is how a compat layer
# starts deleting things nobody asked it to; a provider that means a parameter always quotes it.
_REJECTED_WORDING = re.compile(
    r"['`]([A-Za-z0-9_.]+)['`]\s+is\s+(?:deprecated|not supported|unsupported)"
)
# "Setting 'max_tokens' and 'max_completion_tokens' at the same time is not supported." — two params
# that are individually fine and mutually exclusive. Observed on Anthropic's compat route with
# openclaw and openhands-sdk, which send both; it failed every call of those rollouts.
_CONFLICT = re.compile(
    r"[Ss]etting ['`]([A-Za-z0-9_.]+)['`] and ['`]([A-Za-z0-9_.]+)['`].*?"
    r"(?:not supported|cannot|at the same time)"
)
# Fixes that change how the MODEL BEHAVES, as opposed to how a field is spelled. Dropping a
# temperature changes the sampling distribution; forcing `reasoning_effort` off turns a reasoning
# model into a non-reasoning one. Both are applied because the alternative is a request that fails
# outright, but a caller has to be told, because the consequences are not cosmetic — see
# `BEHAVIOUR_WARNINGS`.
BEHAVIOUR_ALTERING = frozenset({"reasoning_effort", "temperature", "top_p", "top_k"})

# What each one costs, in the terms a user cares about. Keyed by parameter.
BEHAVIOUR_WARNINGS = {
    "reasoning_effort": (
        "reasoning has been turned OFF for this model, because it refuses function tools on "
        "/v1/chat/completions otherwise. Measured consequence: agentic harnesses may make a single "
        "model call and stop — goose and codex both did, 0/3 tasks each, while the same harnesses "
        "scored 3/3 against a non-reasoning model on the same endpoint. Prefer a non-reasoning "
        "model for agent rollouts, or use the Responses route, which keeps reasoning on."
    ),
    "temperature": (
        "the requested temperature was rejected and dropped, so sampling uses the provider default. "
        "The policy being evaluated is not quite the one the harness asked for, which makes an eval "
        "number hard to reproduce."
    ),
    "top_p": "the requested top_p was dropped; sampling uses the provider default.",
    "top_k": "the requested top_k was dropped; sampling uses the provider default.",
}


def behaviour_warnings(fixes) -> list[str]:
    """Human-readable consequences of any fix that changes model behaviour, not just field names.

    Separated from the fix list itself because "renamed max_tokens" and "reasoning turned off" are
    both `param_fixes` entries and only one of them changes what you are measuring.
    """
    out = []
    for fix in fixes or []:
        param = getattr(fix, "param", None) or str(fix)
        for known in BEHAVIOUR_ALTERING:
            if known in str(param) and known in BEHAVIOUR_WARNINGS:
                out.append(f"{known}: {BEHAVIOUR_WARNINGS[known]}")
                break
    return out


# Legacy spelling -> the modern one it collides with. Used only to decide which of a conflicting PAIR
# to drop, and deliberately tiny: guessing wrong here silently changes the request.
LEGACY_ALIASES = {
    "max_tokens": "max_completion_tokens",
    "functions": "tools",
    "function_call": "tool_choice",
}


@dataclass(frozen=True)
class ParamFix:
    """One edit to a request body: drop a parameter, rename it, or set it to a demanded value."""

    param: str
    replacement: str = ""
    value: str | None = None

    @property
    def action(self) -> str:
        if self.value is not None:
            return "set"
        return "rename" if self.replacement else "drop"

    def apply(self, body: dict[str, Any]) -> bool:
        """Edit `body` in place. Returns whether anything changed.

        A rename carries the value across rather than dropping it, because `max_tokens` ->
        `max_completion_tokens` is the same instruction under a different name; losing the value
        would silently uncap the completion length.

        A `set` is the only fix that ADDS a key, so it is the only one that applies to a body which
        does not already contain the parameter — that is the point of it: the request was rejected
        for lacking a value the provider requires, not for carrying a bad one.
        """
        if self.value is not None:
            if body.get(self.param) == self.value:
                return False
            body[self.param] = self.value
            return True
        if self.param == "logprobs" and not self.replacement:
            changed = "logprobs" in body or "top_logprobs" in body
            body.pop("logprobs", None)
            body.pop("top_logprobs", None)
            return changed
        if self.param not in body:
            return False
        value = body.pop(self.param)
        if self.replacement:
            body[self.replacement] = value
        return True

    def __str__(self) -> str:
        if self.value is not None:
            return f"set {self.param}={self.value}"
        if self.replacement:
            return f"renamed {self.param} -> {self.replacement}"
        return f"dropped {self.param}"


def diagnose(body: dict[str, Any] | str | None) -> ParamFix | None:
    """The fix a 400 response body is asking for, or `None` if it is not asking for one.

    Args:
        body (`dict` or `str`, *optional*):
            The parsed error payload, as [`UpstreamHTTPError`] carries it.

    Returns:
        [`ParamFix`] or `None`: The single edit to retry with.
    """
    error: Any = body
    if isinstance(body, dict):
        error = body.get("error", body)
    if isinstance(error, dict):
        param = error.get("param")
        message = str(error.get("message") or "")
        code = str(error.get("code") or "")
    elif isinstance(error, str):
        param, message, code = None, error, ""
    else:
        return None

    if (
        message.strip().lower()
        == "you are not allowed to request logprobs from this model"
    ):
        return ParamFix(param="logprobs")

    # Two parameters that conflict. Resolved before anything else because the message names both and
    # neither is individually wrong, so every other rule here would either miss it or pick a field out
    # of the surrounding prose.
    conflict = _CONFLICT.search(message)
    if conflict:
        a, b = conflict.group(1), conflict.group(2)
        for legacy, modern in ((a, b), (b, a)):
            if LEGACY_ALIASES.get(legacy) == modern and legacy not in PROTECTED:
                return ParamFix(param=legacy)
        # An unknown pair. Refuse rather than guess which one the caller meant to keep: dropping the
        # wrong half of a conflict is a silently different request, and the 400 is at least honest.
        return None

    # An explicit remediation, checked first because it is the provider telling us the answer rather
    # than us inferring one. gpt-5.6 refuses function tools on /v1/chat/completions with:
    #
    #   Function tools with reasoning_effort are not supported for gpt-5.6-sol in
    #   /v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort
    #   to 'none'.
    #
    # Every tools-bearing call fails, which for a coding agent is every call that matters: observed as
    # a rollout that made one model call and then sat idle until the 900s agent timeout, with an empty
    # agent log. `/v1/responses` is the better answer and is a separate piece of work; this keeps the
    # newest OpenAI models usable for eval on the chat route in the meantime.
    set_to = _SET_TO.search(message)
    if set_to and set_to.group(1) not in PROTECTED:
        return ParamFix(param=set_to.group(1), value=set_to.group(2))

    # `Unrecognized request argument supplied: X` names the field in the message and leaves `param`
    # null, which is how OpenAI reports a parameter it has never heard of (our `return_token_ids`).
    unrecognised = _UNRECOGNISED.search(message)
    if unrecognised:
        return _fix(unrecognised.group(1), message)

    if code in {
        "unsupported_parameter",
        "unsupported_value",
        "unknown_parameter",
    } or message.startswith(
        ("Unsupported parameter", "Unsupported value", "Unknown parameter")
    ):
        # `param` is authoritative when present; the quoted name in the message is the fallback for
        # providers that copy OpenAI's prose but not its structured fields.
        name = param or (
            _QUOTED.search(message).group(1) if _QUOTED.search(message) else ""
        )
        return _fix(name, message)

    # Last resort: the provider used a generic code and told us in prose. Gated on explicit rejection
    # vocabulary rather than on any mention of a parameter, because an arbitrary
    # `invalid_request_error` that happens to name a field ("`messages` must not be empty") is a real
    # error to surface, not a parameter to quietly delete.
    rejected = _REJECTED_WORDING.search(message)
    if rejected:
        return _fix(rejected.group(1), message)

    return None


def _fix(name: str, message: str) -> ParamFix | None:
    if not name or name in PROTECTED:
        return None
    # Only honour the suggested replacement when it is a *different* field. "Use 'temperature'
    # instead" in a message about temperature means "use another value", not "rename", and renaming
    # a field to itself would loop forever on the same 400.
    suggested = _USE_INSTEAD.search(message)
    replacement = suggested.group(1) if suggested else ""
    if replacement in {name, *PROTECTED}:
        replacement = ""
    return ParamFix(param=name, replacement=replacement)
