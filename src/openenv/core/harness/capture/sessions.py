"""Session routing: one server, one port, N concurrent rollouts.

The whole multiplexing scheme is one decision: **the API key IS the session id**. We mint a key per
rollout, hand it to the agent as its `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`, or a provider config
field), and every SDK forwards it unchanged on every request. So the bearer token that arrives is
already the rollout identifier, and no agent needs to know it is being recorded.

That is what makes one intercept server serve a whole GRPO group. The alternative, a proxy per
sandbox, means N processes, N ports, N forwards, and capture living inside the thing most likely to
die.

Two rules learned the hard way:

  * **A registered key beats every other hint.** Harnesses inject their own session headers, and
    opencode sends `x-session-id: ses_...` from the AI SDK. Letting that win files the trajectory
    under an id the caller has never seen, so lookups return nothing and it reads as "the agent made
    no model calls" while every turn was in fact captured. That cost real debugging time.
  * **The harness's own session id is kept, not discarded.** It is recorded on the node as
    `harness_session_id`. Our key scopes the ROLLOUT; theirs identifies the sub-conversation, which
    is exactly the ground truth needed to separate a subagent from the main agent.

Unknown keys are rejected when `require_registered` is on. It defaults to on because this server sits
behind a public forward in front of a GPU, and an open inference endpoint is a real cost, not a
theoretical one.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .graph import RolloutGraph
from .upstream import training_sampling

# Session ids become dict keys, filenames, and URL path segments.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def clean_session_id(value: Any) -> str | None:
    if isinstance(value, str) and _SESSION_ID_RE.fullmatch(value.strip()):
        return value.strip()
    return None


def extract_api_key(headers: dict[str, str]) -> str | None:
    """Every major SDK puts its key in one of these three places."""
    lower = {k.lower(): v for k, v in headers.items()}
    auth = lower.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return lower.get("x-api-key") or lower.get("x-goog-api-key")


def extract_harness_session(
    headers: dict[str, str], body: dict[str, Any]
) -> str | None:
    """The harness's own conversation id, when it volunteers one. Never used for routing."""
    lower = {k.lower(): v for k, v in headers.items()}
    return (
        clean_session_id(lower.get("x-session-id"))
        or clean_session_id(lower.get("proxy-x-session-id"))
        or clean_session_id(body.get("_session_id"))
        or clean_session_id(body.get("user"))
    )


@dataclass(frozen=True)
class Upstream:
    """Which engine one rollout's calls go to.

    Per session rather than per server because the durable thing here is the DATASET, not the engine.
    A dataset server downloads thousands of task files and builds sandbox templates; a vLLM restarts
    every training run, and a train-tier engine and an eval-tier one are usually both wanted at once.
    Pinning the engine at boot made the long-lived thing hostage to the short-lived one.

    `cache_key` is what lets N concurrent sessions on one engine share a single client and a single
    capability probe. A credential digest separates authenticated clients without putting the key
    itself in a cache key or repr: different credentials can select different tenants or capabilities.
    """

    llm_url: str
    model: str = ""
    api_key: str | None = field(default=None, repr=False)
    auth_header: str = "Authorization"
    provider: str = "openai"

    @property
    def cache_key(self) -> tuple[str, str, str, str, str]:
        credential = hashlib.sha256((self.api_key or "").encode()).hexdigest()
        return (
            self.llm_url.rstrip("/"),
            self.model,
            self.auth_header.lower(),
            credential,
            self.provider,
        )


def evaluation_sampling(value: dict[str, Any] | None) -> dict[str, Any]:
    """An explicit eval policy override, distinct from the train-only policy contract."""
    import math

    if value is None:
        return {}
    if not isinstance(value, dict) or set(value) - {"temperature", "top_p", "top_k"}:
        raise ValueError("eval_sampling supports temperature, top_p, and top_k only")
    for key, number in value.items():
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
        ):
            raise ValueError("eval_sampling values must be finite numbers")
        if key == "temperature" and not 0 <= number <= 2:
            raise ValueError("eval temperature must be between 0 and 2")
        if key == "top_p" and not 0 < number <= 1:
            raise ValueError("eval top_p must be in (0, 1]")
        if key == "top_k" and (
            type(number) is not int or (number != -1 and number < 1)
        ):
            raise ValueError("eval top_k must be -1 or a positive integer")
    return dict(value)


def rollout_type_for(purpose: str, capture_level: str) -> str:
    """Separate requested use from observed engine capability; keep legacy auto behavior."""
    if purpose not in {"auto", "eval", "train"}:
        raise ValueError("purpose must be auto, eval, or train")
    if purpose == "train" and capture_level != "tokens":
        raise ValueError("training requires exact engine token capture")
    return "eval" if purpose == "eval" or capture_level != "tokens" else "train"


@dataclass
class Session:
    """One rollout's capture buffer."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    graph: RolloutGraph = field(default_factory=RolloutGraph)
    metadata: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    last_turn_at: float | None = None
    upstream_errors: int = 0
    # Set when the caller named an engine for this rollout; `None` means use the server's default.
    upstream: Upstream | None = None
    # What that engine was MEASURED to support, filled in by the probe when the session is created.
    # Empty means "not measured here", so the server default applies. Never assumed optimistically:
    # claiming `tokens` without evidence is how an eval rollout gets stamped trainable.
    capture_level: str = ""
    purpose: str = "auto"
    eval_sampling: dict[str, Any] = field(default_factory=dict)
    # Model calls FORWARDED for this rollout, and the ceiling. 0 means unlimited.
    #
    # Counted on forward rather than read off the graph, because a turn whose logprobs were rejected
    # never becomes a node yet still cost an engine call — counting nodes would let a rollout that
    # fails validation on every turn run forever.
    #
    # A ceiling is needed at all because NO AGENT HARNESS HONOURS ITS OWN STEP CONFIG. Measured on
    # opencode 1.18.30 against a fake engine that always asks for one more tool call:
    # `agent.build.steps=3`, `maxSteps=3` and no setting at all each produced 61 model calls — the
    # fake server's own hard stop, i.e. nothing else ever ended the loop. The proxy is the only
    # component that sees every call, and the API key IS the rollout, so a counter here is exactly a
    # per-rollout step count.
    model_calls: int = 0
    max_model_calls: int = 0
    budget_stop_count: int = 0
    sampling: dict[str, float | int] = field(default_factory=dict)

    @property
    def over_budget(self) -> bool:
        """Whether this rollout has spent its model-call budget."""
        return self.max_model_calls > 0 and self.model_calls >= self.max_model_calls

    @property
    def idle_seconds(self) -> float:
        """Since the last captured turn. The cheapest signal that separates progress from a wedge."""
        return time.time() - (self.last_turn_at or self.created_at)


class SessionRegistry:
    """Thread-safe. Uvicorn serves concurrently and rollouts are independent."""

    def __init__(self, *, require_registered: bool = True) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.require_registered = require_registered

    def create(
        self,
        session_id: str | None = None,
        *,
        upstream: Upstream | None = None,
        capture_level: str = "",
        purpose: str = "auto",
        eval_sampling: dict[str, Any] | None = None,
        max_model_calls: int = 0,
        sampling: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> Session:
        if type(max_model_calls) is not int or max_model_calls < 0:
            raise ValueError("max_model_calls must be a non-negative integer")
        if purpose not in {"auto", "eval", "train"}:
            raise ValueError("purpose must be auto, eval, or train")
        if purpose == "eval" and sampling is not None:
            raise ValueError("eval purpose cannot apply a training sampling override")
        if capture_level:
            rollout_type_for(purpose, capture_level)
        eval_policy = evaluation_sampling(eval_sampling)
        if eval_policy and purpose != "eval":
            raise ValueError("eval_sampling requires explicit eval purpose")
        policy = training_sampling(sampling)
        sid = clean_session_id(session_id) or f"s{secrets.token_hex(12)}"
        with self._lock:
            session = self._sessions.get(sid) or Session(session_id=sid)
            session.metadata.update(metadata)
            session.purpose = purpose
            session.eval_sampling = eval_policy
            if upstream is not None:
                session.upstream = upstream
            if capture_level:
                session.capture_level = capture_level
            # Per session, not per server: one deployment serves a training run that wants a tight
            # cap and an evaluation run that wants none, at the same time.
            if max_model_calls:
                session.max_model_calls = max_model_calls
            if policy:
                session.sampling = policy
            self._sessions[sid] = session
        return session

    def get(self, session_id: str | None) -> Session | None:
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def resolve(self, headers: dict[str, str], body: dict[str, Any]) -> Session | None:
        """Route a request to its rollout. `None` means reject.

        Order matters and is the opposite of what looks natural: the registered API key wins over any
        session header the harness supplies. See the module docstring.
        """
        api_key = extract_api_key(headers)
        session = self.get(api_key)
        if session is not None:
            return session
        if self.require_registered:
            return None
        # Open mode (local debugging only): an unknown caller still gets a session, so a
        # misconfigured agent shows up as an orphan trajectory instead of vanishing.
        return self.create(api_key)

    def list_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._sessions)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def summary(self) -> list[dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [
            {
                "session_id": s.session_id,
                "turns": s.graph.stats()["n_turns"],
                "roots": s.graph.stats()["n_roots"],
                "idle_s": round(s.idle_seconds, 1),
                "upstream_errors": s.upstream_errors,
                **s.metadata,
            }
            for s in sessions
        ]
