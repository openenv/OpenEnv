"""How each Harbor harness is pointed at the intercept server.

This is the ONLY per-agent knowledge in the stack, and it is deliberately data. For every Harbor
agent the only thing that differs is which env var or config key carries the base URL and the API
key. Sandbox, capture, stitching, masking and validation are identical downstream.

A seam has:
    env         env vars set in OUR process. Harbor's installed agents read os.environ at
                construction and forward the provider-relevant subset into the sandbox.
    model_fmt   what to pass as Harbor's `model_name`. Several agents derive their provider from
                the prefix, so this is load-bearing rather than cosmetic.
    dialect     the wire format we expect. Informational: the server detects per request. Recorded
                so a surprise in capture is checkable against what we predicted.
    kwargs      optional (base_url, session, model) -> dict merged into AgentConfig.kwargs, for
                agents needing more than env vars.
    status      what to trust, and the only field a caller should filter on:

                  "validated"     an end-to-end run passed the capture contract AND the ATIF
                                  cross-check. Safe for cross-model and cross-harness comparison.
                  "unstable:<why>" runs and grades, but cannot be budgeted -- it ignores the step
                                  cap, so one rollout can consume a sweep. Reachable explicitly;
                                  deliberately absent from the validated set and the UI picker.
                  "unsupported:<why>" observed to produce unusable results. Worse than an error
                                  when it scores a countable 0.0, because that reads as a weak
                                  model rather than a broken harness.
                  "blocked:<why>" cannot run here at all -- missing credentials or registry entry.
                  "untested"      no end-to-end run yet, however plausible it looks.

                Measured 2026-09-01/02 over 16,000 rollouts on HuggingEnvs/data-agent-harbor-test:
                10 validated, 2 unstable, 3 unsupported.

The API key is always the intercept session id. That is the multiplexing scheme: one server, one
port, N concurrent rollouts, each identified by the key its agent was handed.

Per-harness findings live in README.md (per-harness findings). Add to it as each agent is brought up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

# opencode dispatches on the provider *id*, not the npm package: a provider named `openai` is routed
# to `provider.responses(model)` (the OpenAI Responses API), which @ai-sdk/openai-compatible does not
# implement, and the run dies with `Z.responses is not a function`. Any other name stays on
# chat-completions. Must match the prefix in the opencode seam's model_fmt.
OPENCODE_PROVIDER = "intercepted"

# Must match `install_fixes.PROVIDER`. Duplicated as a literal rather than imported: `pi_agent` imports
# harbor, and pulling that into this module would make the seam table unusable without Harbor
# installed. Kept honest by `tests/test_seams.py`.
PI_PROVIDER = "intercept"


def agent_facing_model(served_model: str) -> str:
    """The model name to hand a harness, derived from what the engine actually serves.

    THE PROBLEM. A vLLM started without `--served-model-name` serves under its full repo id, e.g.
    `Qwen/Qwen3.5-9B`. Every seam then formats that into its own provider prefix and produces a
    two-slash name:

        model_fmt="openai/{model}"  ->  "openai/Qwen/Qwen3.5-9B"

    Harnesses disagree about what that means. gemini-cli requires exactly `provider/model_name` and
    rejects anything else ("Model name must be in the format provider/model_name"); cline-cli wants
    `provider:model` with a colon; several split on the FIRST slash and several on the LAST, so the
    same string resolves to different models depending on the agent. All of them are "working as
    documented" — the string is just ambiguous.

    WHY STRIPPING IS SAFE. The harness-facing name and the upstream name are already decoupled: the
    intercept overwrites `chat_request["model"]` with the configured served id on every request
    (`capture/server.py`), so whatever a harness puts on the wire is replaced before it reaches the
    engine. The harness-facing name only has to be something the harness can parse and route to us;
    it never has to match the engine.

    So: take the leaf. `Qwen/Qwen3.5-9B` -> `Qwen3.5-9B`, and a name with no slash is unchanged.
    This is preferred over relaunching the engine with `--served-model-name` because it works against
    an engine you do not control, including a shared or hosted one.
    """
    if not served_model or not served_model.strip():
        raise ValueError(
            "served model name is empty; the engine reported no model to route to"
        )
    leaf = served_model.strip().rstrip("/").rsplit("/", 1)[-1]
    if not leaf:
        raise ValueError(
            f"cannot derive a harness-facing model name from {served_model!r}"
        )
    # HF provider routes (for example :together) are meaningful upstream, but
    # Harbor's hosted_vllm model parser rejects colons and identifiers >=64 chars.
    # The capture server restores the exact served ID; this is only a local alias.
    import hashlib
    import re

    if len(leaf) >= 64 or re.search(r"[^A-Za-z0-9._-]", leaf):
        stem = re.sub(r"[^A-Za-z0-9._-]", "-", leaf)[:54]
        digest = hashlib.sha256(leaf.encode()).hexdigest()[:8]
        return f"{stem}-{digest}"
    return leaf


def _pi(base_url: str, session: str, model: str) -> dict[str, Any]:
    """pi's provider config, carried to our `InterceptPi` subclass via AgentConfig.kwargs."""
    return {
        "intercept_config": {"base_url": base_url, "api_key": session, "model": model}
    }


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Seam:
    """One harness's wiring. Three channels, in order of preference.

    `agent_env` is the RIGHT one and should be the default choice. It becomes `AgentConfig.env`,
    which Harbor injects into the sandbox via `agent_environment.scoped_exec_env(agent.extra_env)`
    (trial.py:469 for run, :1212 for setup). Two properties make it strictly better than `env`:

      * it works for ANY installed agent, even one whose own code never forwards that variable.
        Harbor's `pi`, for instance, forwards only API keys and has no base-URL handling at all;
        `agent_env` reaches it anyway because Harbor sets it on every exec in the agent phase.
      * it is scoped to the AGENT PHASES ONLY. The verifier runs outside that `with` block, so
        setting OPENAI_API_KEY here cannot reach the grader.

    `env` sets variables in OUR process instead. Needed only where the agent reads them at
    construction time (before any sandbox exists). It is dangerous: OPENAI_API_KEY set this way
    leaks into the grader, whose LLM-judge tier then 401s and silently scores 0 on every answer that
    is correct but not an exact string match.

    `kwargs` becomes `AgentConfig.kwargs`, for agents needing structured config (opencode's provider
    block).
    """

    name: str
    dialect: str
    model_fmt: str = "{model}"
    # When set, Harbor builds the agent from this class rather than its registered name. The escape
    # hatch for a harness whose config Harbor has no seam to write (see pi).
    import_path: str | None = None
    agent_env: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    kwargs: Callable[[str, str, str], dict[str, Any]] | None = None
    # Native harness settings used only when a rollout requests an explicit training policy.
    training_kwargs: dict[str, Any] = field(default_factory=dict)
    # How THIS harness expresses "stop after N agent steps", if it can. Per-agent knowledge, so it
    # belongs here rather than in the caller.
    #
    # A cap matters for training, not just for cost. AsyncGRPO packs every turn of a rollout into one
    # row, and each turn re-sends the whole conversation, so packed length grows with the SQUARE of
    # the turn count: a 58-turn rollout is an order of magnitude larger than a 17-turn one and OOMs
    # the loss step while every log line looks healthy. Harbor itself has no step cap — only a
    # timeout — and an earlier Qwen3-4B run micro-stepped to 451 turns and 10.1M prompt tokens.
    step_limit: Callable[[int], dict[str, Any]] | None = None
    status: str = "untested"
    notes: str = ""

    def resolve(
        self,
        *,
        base_url: str,
        session: str,
        model: str,
        step_limit: int | None = None,
        training: bool = False,
    ) -> tuple[str, dict[str, Any], dict[str, str], dict[str, str]]:
        """-> (model_name, AgentConfig.kwargs, AgentConfig.env, os.environ vars).

        `model` is normalised through `agent_facing_model` first, so a served id like
        `Qwen/Qwen3.5-9B` cannot leak an extra slash into a harness's model string.

        A `step_limit` this seam cannot express is WARNED about rather than dropped: silently ignoring
        it would let a caller believe its rollouts are bounded when they are not, and the symptom
        surfaces much later as an OOM in the loss step.
        """
        model = agent_facing_model(model)
        fmt = {"base_url": base_url, "session": session, "model": model}
        agent_env = {k: v.format(**fmt) for k, v in self.agent_env.items()}
        proc_env = {k: v.format(**fmt) for k, v in self.env.items()}
        extra = self.kwargs(base_url, session, model) if self.kwargs else {}
        if training:
            extra = _deep_merge(extra, self.training_kwargs)
        if step_limit:
            if self.step_limit is None:
                logger.warning(
                    "%s has no native step-limit setting; the capture proxy enforces the "
                    "model-call budget and the agent timeout still applies",
                    self.name,
                )
            else:
                extra = _deep_merge(extra, self.step_limit(step_limit))
        return self.model_fmt.format(model=model), extra, agent_env, proc_env


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Merge nested config without clobbering a sibling key.

    A shallow update of `{"config": {...}}` would replace a seam's whole config block with the step
    limit alone, which is how a base_url quietly disappears.
    """
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _mini_swe_agent_step_limit(limit: int) -> dict[str, Any]:
    """mini-swe-agent takes a YAML config; Harbor's wrapper accepts it as a `config` mapping and dumps
    it (`harbor/agents/installed/mini_swe_agent.py`). `agent.step_limit` is the key it reads."""
    return {"config": {"agent": {"step_limit": limit}}}


def _grok_build(base_url: str, session: str, model: str) -> dict[str, Any]:
    """Use Harbor's supported config merge, including all auxiliary model routes."""
    return {
        "grok_config": {
            "models": {
                key: model
                for key in (
                    "default",
                    "session_summary",
                    "image_description",
                    "web_search",
                )
            },
            "model": {
                model: {
                    "name": model,
                    "model": model,
                    "base_url": f"{base_url}/v1",
                    "env_key": "OPENAI_API_KEY",
                    "api_backend": "chat_completions",
                }
            },
        }
    }


def _opencode(base_url: str, session: str, model: str) -> dict[str, Any]:
    """opencode needs a full provider block; Harbor exposes exactly the right seam for it.

    Harbor writes `provider.<prefix>.options.baseURL` and nothing else (opencode.py:440-447), leaving
    opencode to resolve the model through its built-in provider against the models.dev registry. A
    locally-served model is not in that registry, so opencode emits step-start/step-finish with zero
    tokens and never issues a request: a silent no-op, the worst kind to debug.

    So we supply the provider block outright. `opencode_config` deep-merges LAST (opencode.py:453),
    overriding Harbor's generated block without patching Harbor.
    """
    return {
        "opencode_config": {
            "provider": {
                OPENCODE_PROVIDER: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Harbor Intercept",
                    "options": {
                        "baseURL": f"{base_url}/v1",
                        "apiKey": session,
                        "timeout": 600_000,
                    },
                    "models": {model: {"name": model}},
                }
            }
        }
    }


def acp_opencode_config(base_url: str, session: str, model: str) -> dict[str, Any]:
    """Explicit ACP qualification profile using OpenCode's native ACP command.

    Qualifies Harbor's ACP transport with this pinned implementation only;
    it does not imply that arbitrary ACP registry entries support the proxy.
    """
    import json

    config = _opencode(base_url, session, model)["opencode_config"]
    config.update(
        model=f"{OPENCODE_PROVIDER}/{model}",
        small_model=f"{OPENCODE_PROVIDER}/{model}",
        permission="allow",
        autoupdate=False,
    )
    return {
        "auth_policy": "disabled",
        "registry_entry": {
            "id": "opencode",
            "name": "OpenCode (ACP qualification profile)",
            "version": "1.18.30",
            "description": "OpenCode native ACP, routed through OpenEnv capture",
            "distribution": {
                "npx": {
                    "package": "opencode-ai@1.18.30",
                    "args": ["acp"],
                    "env": {"OPENCODE_CONFIG_CONTENT": json.dumps(config)},
                }
            },
        },
    }


def _terminus(base_url: str, session: str, model: str) -> dict[str, Any]:
    """Terminus runs host-side, in OUR process, so there is no sandbox-side CLI to configure.

    It drives a TmuxSession over environment.exec and calls litellm directly: `api_base` is a
    constructor argument. The API key is not; LiteLLM collects surplus kwargs into `_llm_kwargs` and
    splats them into the call (lite_llm.py:77,649), so `llm_kwargs` is the channel.

    `model_info` is mandatory, not optional: litellm refuses to route a model name it has no
    cost/context metadata for, and a locally-served name is never in its registry.

    Summarisation off: `Terminus2.run` appends subagent rollout details to the main ones
    (terminus_2.py:1615), and a summariser is a different task. Training its turns with this
    rollout's reward is exactly the contamination to avoid. Compaction also breaks prefix stitching.
    """
    return {
        "api_base": f"{base_url}/v1",
        "llm_kwargs": {"api_key": session},
        "model_info": {
            "max_input_tokens": 65536,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
        "collect_rollout_details": True,  # native capture, as a second cross-check
        "enable_summarize": False,
        "proactive_summarization_threshold": 0,
    }


SEAMS: dict[str, Seam] = {
    # --- priority order for bring-up ---------------------------------------
    "opencode": Seam(
        name="opencode",
        dialect="openai_chat",
        model_fmt=OPENCODE_PROVIDER + "/{model}",
        # NO env vars. The provider block carries baseURL and apiKey, and setting OPENAI_API_KEY
        # here actively breaks grading: Harbor forwards it into the sandbox, where the DataAgent
        # grader's tier-3 LLM judge runs `if os.environ.get("OPENAI_API_KEY")` and gets our session
        # id instead of a real key. It 401s, the judge is skipped, and every answer that is right
        # but not exact-match silently scores 0. Reward corruption, not a crash.
        # Naming the provider something other than `openai` also stops Harbor forwarding the key at
        # all (agents/installed/opencode.py:516), which is the general trick: alias the provider and
        # the grader's key survives untouched.
        env={},
        kwargs=_opencode,
        # Reuse the DataAgent training profile: no delegation, interactive questions or web fetch.
        # ATIF omits nested OpenCode sessions, so those rollouts cannot be attributed for training.
        training_kwargs={
            "opencode_config": {
                "tools": {"task": False, "question": False, "webfetch": False}
            }
        },
        status="validated",
        notes="Needed 3 global server fixes: SSE replay, stream_options strip, session-id priority.",
    ),
    "pi": Seam(
        name="pi",
        status="validated",
        dialect="openai_chat",
        # Harbor's pi wrapper cannot express a custom endpoint at all, so this seam supplies the
        # agent CLASS instead: a local subclass that writes ~/.pi/agent/models.json in setup().
        # See harnesses/pi_agent.py. Harbor stays unmodified.
        import_path="openenv.harbor.install_fixes:InterceptPi",
        # pi requires `provider/model`; the provider must match the one in models.json.
        model_fmt=PI_PROVIDER + "/{model}",
        kwargs=_pi,
        notes="No base-URL seam in Harbor's wrapper; needs a models.json written into the sandbox. "
        "Defaults to the Responses API unless api=openai-completions is pinned.",
    ),
    "claude-code": Seam(
        name="claude-code",
        status="validated",
        dialect="anthropic",
        # No /v1 suffix: the Anthropic SDK appends its own path.
        # ANTHROPIC_* does not collide with the grader (which reads OPENAI_API_KEY), so the process
        # channel is safe here; agent_env is set too because it is what actually reaches the sandbox.
        agent_env={
            "ANTHROPIC_BASE_URL": "{base_url}",
            "ANTHROPIC_API_KEY": "{session}",
        },
        env={"ANTHROPIC_BASE_URL": "{base_url}", "ANTHROPIC_API_KEY": "{session}"},
        notes="Calls /v1/messages/count_tokens (handled as an aux route). Injects an env/time block "
        "in its system prompt: watch n_roots for nonce breakage.",
    ),
    "codex": Seam(
        name="codex",
        status="validated",
        dialect="openai_responses",
        # Key goes ONLY through agent_env: in the process env it would overwrite the grader's
        # OPENAI_API_KEY and silently disable the LLM-judge tier.
        agent_env={"OPENAI_BASE_URL": "{base_url}/v1", "OPENAI_API_KEY": "{session}"},
        # codex prefers a Responses WEBSOCKET transport, which the capture proxy does not serve, and
        # only falls back to HTTPS after five failed upgrades — by which point it has abandoned the
        # task. Not worked around here: `supports_websockets=false` needs a CUSTOM provider, since
        # codex refuses to override a built-in one ("model_providers contains reserved built-in
        # provider IDs: `openai`"), and a custom provider needs its base_url threaded through six
        # chained -c overrides. Affects hosted OpenAI only; every other upstream is fine.
        notes="Responses dialect: exercises a different transform than chat-completions.",
    ),
    "gemini-cli": Seam(
        name="gemini-cli",
        status="validated",
        dialect="google",
        # Requires `provider/model` (gemini_cli.py:781) or it raises before install even matters:
        #   ValueError: Model name must be in the format provider/model_name
        # The first failure here LOOKED like an nvm install problem because the job log echoes the
        # install command; the run never got that far. Read result.json's exception_info, not the log.
        model_fmt="google/{model}",
        # Harbor's wrapper declares only `curl` as a system dep, but nvm's installer pipes into
        # bash, so in an image without bash it fails with a misleading "NVM failed to load".
        # The subclass adds bash and otherwise defers to Harbor. See harnesses/install_fixes.py.
        import_path="openenv.harbor.install_fixes:InterceptGeminiCli",
        agent_env={
            "GOOGLE_GEMINI_BASE_URL": "{base_url}",
            "GEMINI_API_KEY": "{session}",
            "GOOGLE_API_KEY": "{session}",
        },
        env={
            "GOOGLE_GEMINI_BASE_URL": "{base_url}",
            "GEMINI_API_KEY": "{session}",
            "GOOGLE_API_KEY": "{session}",
        },
        notes="generateContent dialect; key arrives as x-goog-api-key. Model is carried in the URL "
        "path rather than the body: expect that to be the first thing to break.",
    ),
    # --- second wave --------------------------------------------------------
    "terminus-2": Seam(
        name="terminus-2",
        status="validated",
        dialect="openai_chat",
        model_fmt="hosted_vllm/{model}",
        kwargs=_terminus,
        notes="Host-side agent: no sandbox involved in LLM traffic. Also emits RolloutDetail.",
    ),
    "openhands": Seam(
        name="openhands",
        dialect="openai_chat",
        model_fmt="openai/{model}",
        # Harbor installs `openhands-ai` unpinned and verifies with `python -m openhands.core.main`,
        # but V1 moved the core out to openhands-sdk, so latest fails with
        # `ModuleNotFoundError: No module named 'openhands.core'`. The subclass pins the last V0
        # release (0.49.0). See harnesses/install_fixes.py.
        import_path="openenv.harbor.install_fixes:InterceptOpenHands",
        # OpenHands does NOT use OPENAI_*. It reads LLM_MODEL / LLM_BASE_URL / LLM_API_KEY, all via
        # `_get_env` (openhands.py:873,920,931), so agent_env reaches them. LLM_MODEL is taken from
        # model_name directly, and litellm needs the provider prefix to route.
        agent_env={"LLM_BASE_URL": "{base_url}/v1", "LLM_API_KEY": "{session}"},
        notes="LLM_* env vars, not OPENAI_*. Harbor has a 'dummy-key-for-local-vllm' fallback.",
    ),
    "mini-swe-agent": Seam(
        name="mini-swe-agent",
        status="validated",
        dialect="openai_chat",
        model_fmt="openai/{model}",
        # Needs `provider/model`. It reads MSWEA_API_KEY, and otherwise derives the key variable from
        # the model name via litellm (`openai/` -> OPENAI_API_KEY). Base URL goes to litellm, which
        # accepts OPENAI_API_BASE or OPENAI_BASE_URL; set both, the unused one is harmless.
        # All through agent_env so the grader's OPENAI_API_KEY is never touched.
        agent_env={
            "MSWEA_API_KEY": "{session}",
            "OPENAI_API_KEY": "{session}",
            "OPENAI_API_BASE": "{base_url}/v1",
            "OPENAI_BASE_URL": "{base_url}/v1",
        },
        step_limit=_mini_swe_agent_step_limit,
        notes="MSWEA_API_KEY plus a model-derived key var; litellm under the hood.",
    ),
    "qwen-coder": Seam(
        name="qwen-coder",
        status="validated",
        dialect="openai_chat",
        # Cleanest seam of the lot: declarative EnvVar descriptors for api_key -> OPENAI_API_KEY and
        # base_url -> OPENAI_BASE_URL with env_fallback (qwen_code.py:39-45), then passed explicitly
        # as --openai-api-key / --openai-base-url. agent_env feeds `_get_env` directly.
        agent_env={"OPENAI_API_KEY": "{session}", "OPENAI_BASE_URL": "{base_url}/v1"},
    ),
    "swe-agent": Seam(
        name="swe-agent",
        status="unstable:unbounded-turns",
        dialect="openai_chat",
        model_fmt="openai/{model}",
        agent_env={"OPENAI_API_KEY": "{session}", "OPENAI_BASE_URL": "{base_url}/v1"},
        # Harbor's repo argument has a quoting bug in its else-branch (`'--env.repo.path=$(pwd)'`
        # inside single quotes), giving `git.exc.NoSuchPathError: /workdir/$(pwd)`. The subclass makes
        # /workdir a git repo and exposes it as /testbed so Harbor takes its WORKING branch.
        import_path="openenv.harbor.install_fixes:InterceptSweAgent",
        # Second bug, and the one that made swe-agent produce exactly ONE turn per task while every
        # capture check passed. swe-agent asks litellm to cost each call; litellm has no pricing row
        # for a locally-served name and raises, which swe-agent treats as fatal:
        #
        #   sweagent.exceptions.ModelConfigurationError: Error calculating cost:
        #   This model isn't mapped yet. model=openai/Qwen3.5-9B ...
        #   please make sure you set `per_instance_cost_limit` and `total_cost_limit` to 0
        #
        # Harbor already sets exactly these to 0, but only under `if is_hosted_vllm:` (swe_agent.py
        # :437-443), and our model_fmt is `openai/` so that branch never runs. They are declared
        # CLI_FLAGS, so passing them as kwargs reaches the same flags. `build_cli_flags` skips only
        # None (base.py:651), so "0" is emitted rather than dropped as falsy.
        kwargs=lambda base_url, session, model: {
            "per_instance_cost_limit": "0",
            "total_cost_limit": "0",
            "max_input_tokens": "0",
        },
        notes=(
            "Works, but IGNORES the step cap: 12 requested, 37-45 turns run. Cost "
            "cannot be bounded, so it is unsafe in a sized sweep. Also needs a git "
            "repo, which DataAgent tasks are not."
        ),
    ),
    "goose": Seam(
        name="goose",
        status="unsupported:erratic-cost",
        dialect="openai_chat",
        model_fmt="openai/{model}",
        # goose is the one that reads `os.environ.get("OPENAI_API_KEY")` DIRECTLY (goose.py:678) and
        # raises if unset, so agent_env cannot reach it and the process env is forced. That disables
        # the grader's LLM-judge tier for goose runs; exact-match and numeric tolerance still apply.
        # `runner.apply_seam_env` warns when this happens so it is never a silent reward change.
        env={"OPENAI_API_KEY": "{session}", "OPENAI_BASE_URL": "{base_url}/v1"},
        agent_env={"OPENAI_BASE_URL": "{base_url}/v1"},
        notes=(
            "Unbudgetable: 6 turns on one rollout and 347 on another of the SAME task. "
            "A sweep cannot be sized when one harness can consume 50x its expected "
            "wall-clock."
        ),
    ),
    # --- tier 2: seams derived from each wrapper, none validated yet ----------
    # These follow the same two shapes seen everywhere: a key + base URL pair, delivered through
    # `_get_env` (so agent_env works) or `os.environ` (so the process env is forced). Where the
    # wrapper reads os.environ directly, both channels are set and the grader warning fires.
    # hermes: REMOVED, not merely untested. Measured across 5 DataAgent tasks it failed 5/5 before the
    # agent started — `exit 127` from
    # `curl -fsSL .../NousResearch/hermes-agent/main/scripts/install.sh | bash` — so every attempt cost
    # a sandbox and several minutes to learn nothing. The capture layer reported it correctly ("the
    # intercept saw no model calls: the agent never reached it"), which is how it was found. The
    # `InterceptHermes` subclass is gone with it; re-adding both needs the install to work first.
    "vibe": Seam(
        name="vibe",
        status="validated",
        dialect="openai_chat",
        model_fmt="openai/{model}",
        # vibe defaults to Mistral's API unless VIBE_API_BASE or OPENAI_BASE_URL is set (vibe.py:76-96),
        # and resolves its key from the var named by VIBE_API_KEY_ENV.
        # Its own error names the valid values:
        #   ValueError: Unknown Vibe backend 'openai'; valid backends are 'mistral' and 'generic'
        #               (use 'generic' for any OpenAI-compatible endpoint)
        agent_env={
            "VIBE_API_BASE": "{base_url}/v1",
            "OPENAI_BASE_URL": "{base_url}/v1",
            "VIBE_API_KEY_ENV": "OPENAI_API_KEY",
            "OPENAI_API_KEY": "{session}",
            "VIBE_BACKEND": "generic",
        },
    ),
    "openclaw": Seam(
        name="openclaw",
        status="unsupported:launch-failure",
        dialect="openai_chat",
        model_fmt="openai/{model}",
        agent_env={"OPENAI_BASE_URL": "{base_url}/v1", "OPENAI_API_KEY": "{session}"},
        # Harbor writes a merged config to openclaw.upload.json and copies it into the sandbox, but
        # only when there IS a config. With none, setup dies on
        #   cp: cannot stat '/logs/agent/openclaw.upload.json'
        # A provider block gives it something to write AND points openclaw at us.
        # Schema is `models.providers.<name>`, NOT a top-level `providers`. A top-level one is
        # accepted by Harbor's merge and then rejected by openclaw itself:
        #     OpenClaw config is invalid
        #     Problem:  - <root>: Invalid input
        # Harbor's own `_align_provider_models` reads cfg["models"]["providers"][provider] and fills
        # in a `models` array beside `baseUrl`, which is the shape mirrored here. The provider name
        # must match the model prefix so `_model_provider()` resolves it.
        kwargs=lambda base_url, session, model: {
            "openclaw_config": {
                "models": {
                    "providers": {
                        "openai": {
                            "baseUrl": f"{base_url}/v1",
                            "apiKey": session,
                            "api": "openai-completions",
                            "models": [{"id": model, "name": model}],
                        }
                    }
                }
            },
            # Harbor defaults `--thinking high`, which openclaw rejects for a custom provider:
            #   Error: Thinking level "high" is not supported for openai/Qwen3.5-9B. Use one of: off.
            # It is a CliFlag, so kwargs can override it. Also correct for us: we serve with thinking
            # disabled, so anything else would be asking for a mode the model is not running in.
            "thinking": "off",
        },
        # Harbor writes the config to the HOST logs dir then copies it from the CONTAINER path,
        # which assumes a bind mount. E2B has none, so the subclass uploads it into the sandbox.
        import_path="openenv.harbor.install_fixes:InterceptOpenClaw",
        notes=(
            "`nvm use 22 && openclaw agent --local ...` exits 1 and the agent never "
            "makes a model call, so a suite's missing-answer default scores it a "
            "countable 0.0 -- worse than an error."
        ),
    ),
    "kimi-cli": Seam(
        name="kimi-cli",
        status="unsupported:no-reward",
        notes=(
            "Reached 14 turns then returned no reward at all (`rewards={}`) on both "
            "probe tasks, so nothing it produces is scorable."
        ),
        dialect="openai_chat",
        model_fmt="openai/{model}",
        # Harbor DELIBERATELY unsets OPENAI_BASE_URL / OPENAI_API_KEY before spawning kimi
        # (`_KIMI_ENV_OVERRIDES_TO_NEUTRALIZE`), because kimi-cli's own
        # `augment_provider_with_env_vars` silently overrides its config file from those vars — a
        # globally-injected OpenAI key would hijack an OpenRouter run (MoonshotAI/kimi-cli#1165).
        # So an env-var seam is not merely ignored here, it is actively erased. That is why every
        # attempt showed 0 turns.
        #
        # The config file is what wins, and Harbor exposes both values as plain constructor kwargs:
        #   base_url = self._base_url or pcfg["base_url"]      (kimi_cli.py:208)
        #   api_key  = self._api_key or <env lookup>           (kimi_cli.py:169)
        kwargs=lambda base_url, session, model: {
            "base_url": f"{base_url}/v1",
            "api_key": session,
        },
        # Second, separate bug. Harbor's run command ends with `kill 0`, which takes the E2B exec
        # stream down along with the process group and raises RemoteProtocolError(StreamReset) in
        # OUR process. That killed all 11 kimi trials AFTER the agent had finished its work, so the
        # verifier never ran and every reward came back None. The subclass swallows only that one
        # error, mirroring Harbor's own handling of the exit-143 half of the same teardown.
        import_path="openenv.harbor.install_fixes:InterceptKimi",
    ),
    "mimo": Seam(
        # mimo is an opencode fork and inherits its provider dispatch, so it inherits the same trap:
        # a provider literally named `openai` routes to `provider.responses(model)`, which
        # @ai-sdk/openai-compatible does not implement, and the run dies with
        #   TypeError: Z.responses is not a function
        # having never reached the intercept (0 turns).
        #
        # Harbor tries to avoid this: it writes `npm: @ai-sdk/openai-compatible` into the provider
        # block, but only when it finds a base URL, and it looks in OUR PROCESS env
        # (`os.environ.get(f"{provider.upper()}_BASE_URL")`, mimo.py:389) rather than in the sandbox
        # env we set. `agent_env` never reaches os.environ, so that branch does not run and mimo
        # falls back to its built-in openai provider. Same shape as the goose problem.
        #
        # Fixed the way opencode is fixed, and for the same reason: a provider id that is NOT
        # "openai", with the block supplied outright via the `mimo_config` kwarg. That deep-merges
        # LAST (mimo.py:402), so it wins without patching Harbor and without putting OPENAI_API_KEY
        # in our process env where the verifier's LLM judge would inherit it.
        name="mimo",
        dialect="openai_chat",
        model_fmt=OPENCODE_PROVIDER + "/{model}",
        kwargs=lambda base_url, session, model: {
            "mimo_config": {
                "provider": {
                    OPENCODE_PROVIDER: {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "Harbor Intercept",
                        "options": {
                            "baseURL": f"{base_url}/v1",
                            "apiKey": session,
                            "timeout": 600_000,
                        },
                        "models": {model: {"name": model}},
                    }
                }
            }
        },
        agent_env={"OPENAI_BASE_URL": "{base_url}/v1", "OPENAI_API_KEY": "{session}"},
    ),
    "trae-agent": Seam(
        # RESPONSES, not chat. Mislabelled `openai_chat` for a night because the config says
        # `provider: openai` and everything else with that label speaks chat. The access log settled
        # it: exactly one `POST /v1/responses` against 465 chat-completions calls, and that one was
        # trae. Its client reads `usage.input_tokens_details.cached_tokens`, a Responses-only field.
        name="trae-agent",
        status="unstable:unbounded-turns",
        notes=(
            "Works, but IGNORES the step cap: 12 requested, 112-200 turns run -- ~20x "
            "the stable harnesses. One rollout can consume a whole sweep's budget."
        ),
        dialect="openai_responses",
        model_fmt="openai/{model}",
        agent_env={"OPENAI_BASE_URL": "{base_url}/v1", "OPENAI_API_KEY": "{session}"},
    ),
    # --- remaining ATIF agents ------------------------------------------------
    # Goal is every ATIF-capable agent Harbor registers (25). These nine had no seam; the ones with a
    # recognisable LLM config get one, and the vendor-service ones are attempted anyway so their block
    # reason is RECORDED rather than assumed.
    "computer-1": Seam(
        name="computer-1",
        dialect="openai_chat",
        model_fmt="hosted_vllm/{model}",
        # Host-side like terminus-2: drives litellm from our process with an `api_base` kwarg, so
        # there is no sandbox-side CLI to configure.
        kwargs=lambda base_url, session, model: {
            "api_base": f"{base_url}/v1",
            "llm_kwargs": {"api_key": session},
            "model_info": {
                "max_input_tokens": 262144,
                "max_output_tokens": 8192,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
        },
        notes="Host-side agent, litellm. The other RolloutDetail emitter besides terminus-2.",
    ),
    "eve": Seam(
        name="eve",
        dialect="openai_chat",
        model_fmt="openai/{model}",
        # NOT a general coding agent. `_validate_path` requires a local Eve PROJECT directory with
        # package.json plus an agent/ dir (or flat agent.ts / instructions.md), and raises before any
        # sandbox work -- the 7s failure with empty Harbor logs. Nothing to do with the intercept.
        status="blocked:needs-eve-project",
        # Reads OPENAI_BASE_URL / OPENAI_ENDPOINT / OPENAI_API_KEY.
        agent_env={
            "OPENAI_BASE_URL": "{base_url}/v1",
            "OPENAI_ENDPOINT": "{base_url}/v1",
            "OPENAI_API_KEY": "{session}",
        },
        env={"OPENAI_BASE_URL": "{base_url}/v1"},
    ),
    "cursor-cli": Seam(
        name="cursor-cli",
        dialect="openai_chat",
        model_fmt="openai/{model}",
        agent_env={"OPENAI_BASE_URL": "{base_url}/v1", "OPENAI_API_KEY": "{session}"},
        status="blocked:credentials",
        # CONFIRMED by running it, not assumed:
        #   ValueError: CURSOR_API_KEY environment variable is required.
        # It authenticates against Cursor's own service before any model call, so pointing it at a
        # local endpoint cannot help. Nothing about the intercept is implicated.
        notes="BLOCKED: requires CURSOR_API_KEY (Cursor account). Verified, not assumed.",
    ),
    "acp": Seam(
        name="acp",
        dialect="openai_chat",
        model_fmt="openai/{model}",
        # A LAUNCHER for ACP-speaking agents, not an agent. Needs `registry_entry` /
        # `registry_entry_path` describing a distribution ("ACP registry entry must define at least
        # one distribution"), and raises immediately without one.
        status="blocked:needs-registry-entry",
        agent_env={"OPENAI_BASE_URL": "{base_url}/v1", "OPENAI_API_KEY": "{session}"},
        notes="Agent Client Protocol runner; needs an ACP-speaking agent configured underneath.",
    ),
    "devin": Seam(
        name="devin",
        dialect="openai_chat",
        model_fmt="openai/{model}",
        agent_env={"OPENAI_BASE_URL": "{base_url}/v1", "OPENAI_API_KEY": "{session}"},
        notes="Cognition hosted service; expected to need vendor credentials.",
    ),
    "copilot-cli": Seam(
        name="copilot-cli",
        dialect="openai_chat",
        model_fmt="openai/{model}",
        agent_env={
            "COPILOT_PROVIDER_TYPE": "openai",
            "COPILOT_PROVIDER_BASE_URL": "{base_url}/v1",
            "COPILOT_PROVIDER_API_KEY": "{session}",
            "COPILOT_MODEL": "{model}",
            "COPILOT_OFFLINE": "true",
        },
        notes="Native BYOK provider configuration routes model requests through capture; no GitHub model service required.",
    ),
    "antigravity-cli": Seam(
        name="antigravity-cli",
        dialect="google",
        model_fmt="google/{model}",
        agent_env={
            "GOOGLE_GEMINI_BASE_URL": "{base_url}",
            "GEMINI_API_KEY": "{session}",
            "GOOGLE_API_KEY": "{session}",
            "AGY_ADC_AUTH": "false",
        },
        notes="Harbor's native Gemini base-URL configuration and custom model registration route requests through capture.",
    ),
    "grok-build": Seam(
        name="grok-build",
        dialect="openai_chat",
        model_fmt="openai/{model}",
        agent_env={"OPENAI_API_KEY": "{session}"},
        kwargs=_grok_build,
        notes="Harbor grok_config points the primary and auxiliary models to capture using its supported custom endpoint configuration.",
    ),
    "rovodev-cli": Seam(
        name="rovodev-cli",
        dialect="openai_chat",
        model_fmt="openai/{model}",
        agent_env={"OPENAI_BASE_URL": "{base_url}/v1", "OPENAI_API_KEY": "{session}"},
        notes="Atlassian; needs ROVODEV_USER_API_TOKEN + ROVODEV_USER_EMAIL.",
    ),
    # Found only via Harbor's SUPPORTS_ATIF flag; an import grep missed all four.
    "cline-cli": Seam(
        name="cline-cli",
        dialect="openai_chat",
        # COLON, not slash. Its own error is explicit:
        #   ValueError: model_name must be in format 'provider:model-id', got: 'openai/Qwen3.5-9B'
        # The only harness so far that does not use `provider/model`.
        model_fmt="openai:{model}",
        agent_env={"OPENAI_BASE_URL": "{base_url}/v1", "OPENAI_API_KEY": "{session}"},
        # Harbor gives cline NO base-URL channel (only PROVIDER/API_KEY/MODELID), so it calls the
        # real OpenAI. The subclass merges Cline's own settings store between Harbor's write and the
        # run. See harnesses/install_fixes.py.
        import_path="openenv.harbor.install_fixes:InterceptCline",
        kwargs=lambda base_url, session, model: {
            "intercept_config": {
                "base_url": base_url,
                "api_key": session,
                "model": model,
            }
        },
    ),
    "nemo-agent": Seam(
        name="nemo-agent",
        dialect="openai_chat",
        model_fmt="openai/{model}",
        # Defaults to `llm_type: "nim"` (NVIDIA NIM), so it never reads OPENAI_* and the OpenAI seam
        # was silently ignored. Its own module docstring gives the recipe:
        #   harbor run --agent nemo-agent --model openai/gpt-4o --ak llm_type=openai
        # `--ak` is AgentConfig.kwargs, so this selects the OpenAI-compatible provider.
        kwargs=lambda base_url, session, model: {"llm_type": "openai"},
        agent_env={
            "OPENAI_BASE_URL": "{base_url}/v1",
            "OPENAI_API_KEY": "{session}",
            "OPENAI_API_BASE": "{base_url}/v1",
        },
    ),
    "openhands-sdk": Seam(
        name="openhands-sdk",
        status="validated",
        dialect="openai_chat",
        model_fmt="openai/{model}",
        # The V1 SDK packaging, so it should NOT need the V0 pin the classic wrapper does.
        agent_env={
            "LLM_BASE_URL": "{base_url}/v1",
            "LLM_API_KEY": "{session}",
            "OPENAI_BASE_URL": "{base_url}/v1",
            "OPENAI_API_KEY": "{session}",
        },
    ),
    "antigravity-sdk": Seam(
        name="antigravity-sdk",
        dialect="google",
        model_fmt="google/{model}",
        # It said so itself: `GEMINI_API_KEY environment variable must be set`. My first seam gave it
        # only OPENAI_*, which is why it looked like a vendor-credential block when it is really a
        # Google-family agent. Same seam as gemini-cli, which is validated.
        agent_env={
            "GOOGLE_GEMINI_BASE_URL": "{base_url}",
            "GEMINI_API_KEY": "{session}",
            "GOOGLE_API_KEY": "{session}",
        },
        env={
            "GOOGLE_GEMINI_BASE_URL": "{base_url}",
            "GEMINI_API_KEY": "{session}",
            "GOOGLE_API_KEY": "{session}",
        },
        notes="Google Antigravity SDK; takes the gemini-cli seam, not an OpenAI one.",
    ),
}

# Every ATIF-capable agent Harbor registers. This IS the goal.
#
# Source of truth is Harbor's own `SUPPORTS_ATIF` class flag, not a grep for trajectory imports.
# The grep undercounted (25 vs 29): it missed antigravity-sdk, cline-cli, nemo-agent and
# openhands-sdk, which build ATIF without importing the models in a way grep could see. Re-derive with:
#
#   getattr(import_class(AgentFactory._AGENT_MAP[a], label="agent"), "SUPPORTS_ATIF", False)
#
# computer-1 is deliberately excluded: not needed for this goal.
# Every agent whose Harbor class sets SUPPORTS_ATIF, minus two deliberate exclusions.
#
# `computer-1` is out of scope by request.
#
# `openhands` (V0) is out because Harbor registers it and `openhands-sdk` as two SEPARATE agents,
# not as old and new names for one. V0 bundles the full `openhands-ai` with its own Docker runtime;
# the SDK runs directly in the container. Running a Docker runtime inside a sandbox that is already
# a container is the wrong shape for this work, and the SDK validated 5/5 with zero fixes while V0
# needed four packaging patches and still did not come through. Its Seam and InterceptOpenHands
# subclass are kept so `--agents openhands` still works; it is simply not a target.
# `nemo-agent` is out as well. Harbor generates its NAT config as
# `workflow: {_type: chat_completion}`: a single-shot LLM call with no tools and no loop, so ONE
# turn is correct behaviour, not a failure. Making it agentic needs a react_agent workflow plus
# NAT's `code_execution` tool, which requires a separate sandbox service (docker or Piston) that
# would not even have the task's CSV. Bring-your-own-workflow, like eve and acp.
# `antigravity-sdk` is out too. Its Go `localharness` binary receives a correctly-translated
# functionCall (verified by direct probe) and then ends the conversation without executing it, the
# SDK's "Received tool call %s but no tool runner is configured. Yielding to user." path. That is
# inside google-antigravity, and Harbor's runner sets the root logger to ERROR so the warning that
# would confirm it never reaches a log. Seam kept; not a target.
ATIF_AGENTS = [
    "acp",
    "antigravity-cli",
    "claude-code",
    "cline-cli",
    "codex",
    "copilot-cli",
    "cursor-cli",
    "devin",
    "eve",
    "gemini-cli",
    "goose",
    "grok-build",
    "kimi-cli",
    "mimo",
    "mini-swe-agent",
    "openclaw",
    "opencode",
    "openhands-sdk",
    "qwen-coder",
    "rovodev-cli",
    "swe-agent",
    "terminus-2",
    "trae-agent",
    "vibe",
]

PRIORITY = [
    "opencode",
    "pi",
    "claude-code",
    "codex",
    "gemini-cli",
    "terminus-2",
    "openhands-sdk",
    "mini-swe-agent",
]


def get(name: str, *, profile: str | None = None) -> Seam:
    if name not in SEAMS:
        raise KeyError(f"no seam for {name!r}. Known: {sorted(SEAMS)}")
    if profile is None:
        return SEAMS[name]
    if name == "acp" and profile == "opencode-1.18.30":
        from dataclasses import replace

        return replace(
            SEAMS[name],
            model_fmt=OPENCODE_PROVIDER + "/{model}",
            kwargs=acp_opencode_config,
        )
    if name == "nemo-agent" and profile == "shell-1.9.0":
        from dataclasses import replace
        from pathlib import Path

        package = (
            Path(__file__).resolve().parents[3] / "examples/harbor/nemo_shell_profile"
        )
        if not (package / "pyproject.toml").is_file():
            raise ValueError(
                "NeMo shell profile requires the example workflow package in this checkout"
            )
        return replace(
            SEAMS[name],
            import_path="openenv.harbor.nemo_profile:NemoShellProfile",
            kwargs=lambda base_url, session, model: {
                "llm_type": "openai",
                "version": "1.9.0",
                "workflow_package": str(package),
            },
        )
    raise ValueError(f"unsupported harness profile {profile!r} for {name!r}")
