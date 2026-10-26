# Dialect transformers

Translation between the four wire dialects coding agents speak and the OpenAI chat-completions
shape a vLLM server understands.

| dialect | spoken by |
|---|---|
| `openai_chat` | opencode, qwen-coder, goose, swe-agent, mini-swe-agent, terminus-2, vibe, mimo, kimi-cli, hermes, openclaw, openhands-sdk, pi |
| `openai_responses` | codex, trae-agent |
| `anthropic` | claude-code |
| `google` | gemini-cli, antigravity-sdk |

Without this layer the intercept only works for chat-completions agents, which is 13 of the 16
validated harnesses — but the three it loses (claude-code, codex, gemini-cli) are the ones whose
capture is hardest to get right, so they are also the ones most worth having covered.

## Provenance

Adapted from the Polar gateway (`polar/gateway/transform/`, Apache-2.0).
Upstream: https://github.com/NVIDIA-NeMo/ProRL-Agent-Server (paper: https://arxiv.org/abs/2605.24220).
Named here because the package called `polar` on PyPI is an unrelated project, which is the
reason this is vendored rather than depended on.

Changes made when
vendoring:

- import paths rewritten to be relative and self-contained; no dependency on Polar remains
- the internal request marker `_polar_model_served` renamed to `_served_model`
- reasoning-signature wire prefixes `polar:` / `sg_polar_` renamed to `oe:` / `sg_oe_`
  (an opaque, symmetric encode/decode pair — the value is arbitrary as long as both sides agree)
- `engine.py` and `proxy.py` were **not** vendored. They carried an SGLang backend that cannot
  support token capture at all, so they were replaced by `capture/upstream.py`, a vLLM-only client
  in ~160 lines.

`images.py` and `reasoning.py` are required: the anthropic, google and responses transformers all
import them for multimodal content blocks and thinking-block round-tripping respectively.
