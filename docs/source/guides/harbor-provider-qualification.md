# Harbor Qualification

An installed adapter is not evidence that a harness works with a given model provider. Qualify the harness version, model route, sandbox, capture implementation and task set together.

## Evaluation and training capture

- **`purpose="eval"`**: hosted OpenAI, native Anthropic and Hugging Face routes produce graded evaluation traces without engine token IDs. An eval trace never exports a training contract, even if its endpoint returns token IDs.
- **`purpose="train"`**: only with a verified token-capable endpoint. Training export keeps engine prompt IDs, sampled completion IDs, processed log probabilities and loss masks. `openenv.harbor.contract.to_trace_entries` rejects evaluation traces and fatal capture findings. Never rebuild token IDs by tokenizing rendered text or fill missing log probabilities with zeros.

A prompt rewrite can turn one rollout into several training rows without invalidating the sampled tokens. Report rows per rollout, repeated context, retained supervision and downstream weighting separately. Correct capture does not mean an efficient training configuration.

Native Anthropic requests keep their signed blocks and supported native metadata. Translation to another harness protocol rejects output that can't be preserved. The native streaming bridge buffers the upstream response and replays SDK-compatible events, so there is no upstream first-token latency.

## Evidence and support tiers

A qualification report has one cell per harness/provider pair, with providers `openai`, `anthropic`, `hf` and `vllm`. Keep the capture artifacts and attempt configuration next to it: model routes, revision pins, harness versions, task identities, sampling and source hashes.

| Status | Meaning |
|--------|---------|
| `eval_pass` | Completed, graded rollout with captured calls, no fatal capture findings and no training export. A task score of zero is a valid evaluation. An infrastructure failure or missing grade is not. |
| `capture_and_reader_pass` | Exact capture passed validation and the real training reader kept the expected supervision. |
| `optimizer_pass` | Current capture artifacts were consumed by a real optimizer diagnostic. Record model revision, input fingerprints, consumed rows, finite losses and finite nonzero gradients, and say whether it was diagnostic replay and whether weight sync was tested. |
| `failed`, `blocked`, `in_progress`, `not_run` | Kept as is. Never replaced by a pass from a different configuration. |

`harness_maturity_rows` derives the tier from validated cells:

- **stable**: `eval_pass` on OpenAI, Anthropic and HF, plus `optimizer_pass` on vLLM.
- **unstable**: all four profiles `failed` or `blocked`.
- **experimental**: anything in between (partial or pending).

Tiers describe the recorded coverage only, not universal compatibility or production-scale reliability.

## Using a report in the UI

Set `OPENENV_HARBOR_QUALIFICATION_REPORT` to the report JSON path to show the evidence in the Harbor Gradio UI. The UI offers stable harnesses by default, experimental ones behind an opt-in, and hides unstable ones. Without a report, every adapter is unqualified and needs the opt-in. Recorded results don't certify a newly entered endpoint or pin the harness installation.

Some results apply to a specific profile, which the UI passes to the rollout and shows in the agent label: ACP is qualified with `opencode-1.18.30`, NeMo with `shell-1.9.0` (when the example workflow package is in the checkout). Selecting a profile uses a local seam copy and leaves the global adapter registry unchanged. In code, pass `harness_profile=` to `run_rollout` or `build_trial_config`. Unknown profiles fail.

## Recorded results

The latest recorded qualification, on 15 September 2026, covers 29 adapters on four provider profiles with two tasks per pair. It lists each adapter's tier and per-provider status, the exact models and serving configuration, and the known limitations.

<details>
<summary>Qualification of 15 September 2026 (click to expand)</summary>

The completed qualification attempted all 29 adapters on four provider profiles, with two fixed tasks per pair (116 pairs). Results are compatibility smoke tests, not benchmark pass@1 scores. “Stable” means passing this recorded coverage; it does not certify arbitrary models, harness upgrades, or production-scale reliability.

| Provider profile | Model | Passing adapters |
|---|---|---:|
| OpenAI evaluation | `gpt-5.4-mini-2026-03-17` | 21/29 |
| Native Anthropic evaluation | `claude-sonnet-4-5-20250929` | 20/29 |
| Hugging Face evaluation | `Qwen/Qwen3.5-9B:together` | 19/29 |
| vLLM training capture and optimizer diagnostic | `Qwen/Qwen3.5-4B` | 21/29 |

The HF route is pinned, but its hosted weights are not an immutable revision. The vLLM model revision is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. That profile used vLLM 0.25.1, TP=1, DP=1, BF16, a 131072-token context, processed log probabilities, engine token IDs, Qwen3 XML tool parsing, Qwen3 reasoning parsing with thinking disabled, and no image/video inputs.

There are **14 stable, 9 experimental, and 6 unstable adapters**. A failed pair means the two-task qualification did not pass; it does not necessarily mean both tasks failed or that the adapter can never support that provider.

| Adapter | Tier | OpenAI | Anthropic | HF | vLLM |
|---|---|---|---|---|---|
| acp | experimental | failed | eval_pass | eval_pass | optimizer_pass |
| antigravity-cli | experimental | failed | failed | eval_pass | optimizer_pass |
| antigravity-sdk | unstable | failed | failed | failed | failed |
| claude-code | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| cline-cli | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| codex | experimental | eval_pass | eval_pass | failed | optimizer_pass |
| computer-1 | unstable | failed | failed | failed | failed |
| copilot-cli | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| cursor-cli | unstable | failed | failed | failed | failed |
| devin | unstable | failed | failed | failed | failed |
| eve | unstable | failed | failed | failed | failed |
| gemini-cli | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| goose | experimental | eval_pass | eval_pass | failed | optimizer_pass |
| grok-build | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| kimi-cli | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| mimo | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| mini-swe-agent | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| nemo-agent | experimental | eval_pass | eval_pass | failed | optimizer_pass |
| openclaw | experimental | eval_pass | eval_pass | failed | failed |
| opencode | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| openhands | experimental | eval_pass | eval_pass | eval_pass | failed |
| openhands-sdk | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| pi | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| qwen-coder | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| rovodev-cli | unstable | failed | failed | failed | failed |
| swe-agent | experimental | eval_pass | failed | eval_pass | optimizer_pass |
| terminus-2 | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| trae-agent | experimental | eval_pass | failed | eval_pass | optimizer_pass |
| vibe | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |

**Scope and known limitations**

The optimizer diagnostics consumed 99 current capture rows across 21 adapters using the real `AsyncGRPOTrainer`, with finite losses and finite nonzero gradients. They used a diagnostic advantage of +1 and did not synchronize weights. This establishes capture consumption by the trainer, not reward-normalized learning, long-run stability, or correct weighting when a rollout produces multiple rows. Claude Code and other prompt-rewriting harnesses still need row-budget and weighting checks for a particular training configuration.

ACP qualification applies only to the `opencode-1.18.30` profile, and NeMo qualification only to `shell-1.9.0`. ACP has partial native usage evidence; NeMo lacks independent native token counts. Engine capture remains authoritative, and these results do not qualify arbitrary ACP agents or NeMo workflows.

Codex, Goose, and NeMo retain HF failures. Antigravity CLI retains OpenAI and Anthropic failures. SWE-agent timed out on Anthropic; Trae-agent captured no Anthropic calls. OpenClaw and OpenHands retain training trajectory reconciliation failures. Antigravity SDK also failed strict reconciliation despite executing tools. Missing vendor credentials or application prerequisites prevented qualification of Cursor, Devin, Rovo Dev, and Eve. Computer-1 needs a separate desktop/vision qualification. Keep these failures visible; do not relax token checks to promote an adapter.

The final combined regression run passed 567 tests with two skips; native Anthropic SDK streaming replay was also checked separately. Live qualification and optimizer replay used separate services and source snapshots. Updating this documentation or a qualification report does not restart training, change an existing training snapshot, or deploy the adapter changes. A running process continues to use its configured source and services.

</details>

## Regression and live validation

Run the deterministic Harbor tests from the repository root:

```bash
PYTHONPATH=src:envs python -m pytest tests/envs/test_harbor*.py -q
```

They cover provider conversion, capture graphs, export masks, reconciliation, routing, lifecycle and evidence gates. They don't replace live harness runs.

For live qualification:

- Use isolated services and immutable source snapshots. Fix the task set and versions before launch, bound sandbox concurrency and save each result before moving on.
- Resume by scheduling only the missing cases into a new attempt directory, keeping prior failures and provenance.
- A capture counts as optimizer evidence only if its source hash and row fingerprint match. A newer retry doesn't inherit an older optimizer pass.
- If an adapter needs a separate application, workflow, vision input or vendor account, report the missing prerequisite. Don't substitute another agent, drop observations, relax token checks or count a reachable endpoint as success.
