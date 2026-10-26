---
title: Harbor
emoji: ⚓
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Harbor Environment

**Train one policy against many coding agents.** Pick a Harbor dataset, pick an agent, pick a
sandbox, and get back the exact token ids and per-token logprobs of every model call the agent made,
plus the task's own reward.

## Overview

An agent harness is a moving part you probably do not want to own. opencode, codex, claude-code and
gemini-cli each have their own loop, their own tool surface and their own wire format, and a policy
trained against exactly one of them learns that one's habits.

The usual cost of supporting several is one integration per agent. Here it is one integration total:

| | |
|---|---|
| **16 harnesses** | validated end to end, across 4 wire dialects |
| **23 sandbox backends** | from Harbor, 4 with credential checks wired in |
| **any Harbor dataset** | HF repo, local directory, or Harbor registry name |

All three are chosen **per rollout**, so one server covers the whole matrix and rotating the harness
during training is a config change rather than a new environment.

Harbor supplies the tasks, sandboxes, agents and verifiers. This environment adds the OpenEnv
surface: dataset discovery over the Task API, one `run_rollout` MCP tool, a capture proxy, and a UI.

### What you get back

Every rollout carries the reward from the task's verifier and the full trace: each conversation with
its messages, and per turn the text, tool calls and finish reason.

Against vLLM or SGLang you also get the training contract, per model call:

```python
turn.prompt_token_ids       # the engine's own tokenisation of everything before this turn
turn.completion_token_ids   # what it sampled
turn.per_token_logps        # the behaviour-policy logprob of each sampled token
```

Which of the two you got is on the result as `rollout_type` (`"train"` or `"eval"`) and
`capture_level`, decided by probing the endpoint before the server starts. Against a hosted provider
the token fields are empty and `rollout_type == "eval"`; nothing pretends otherwise, and
`to_turn_records` raises rather than handing a trainer empty lists.

## Quick Start

```bash
pip install "openenv[harbor]"      # Harbor and every sandbox backend, needs Python 3.12 or newer
```

`$LLM` below is an OpenAI-compatible endpoint: vLLM or SGLang for trainable rollouts, or any hosted
provider for evaluation rollouts (see [Prerequisites](#prerequisites)).

### 1. See what this machine can do

```bash
openenv harbor info \
  --llm-url $LLM \
  --dataset AdithyaSK/data_agent_rl_environment_eval
```

```
llm       Qwen/Qwen3.5-9B  [ok]
sandboxes 2 of 4 usable
  [ok]   e2b
  [ok]   modal
  [--]   docker     Docker daemon is not running.
  [--]   daytona    SDK not installed (daytona).
datasets  1 split(s), 366 tasks
harnesses 16 validated of 30 known
```

Read-only, boots nothing. It tells you which sandboxes have working credentials **and** an
importable SDK, so you find out here rather than 90 seconds into a rollout.

### 2. Run one rollout, no server

```bash
openenv harbor rollout \
  --llm-url $LLM \
  --dataset AdithyaSK/data_agent_rl_environment_eval \
  --task-index 0 --harness opencode --sandbox e2b \
  --out rollout.json
```

```
[opencode / e2b] task 0: 0000_369_369503_qa_1 ...
   ok    reward=1.00  turns=9  roots=2  multi-turn  tokens=1043  atif=match  48s
```

This path involves no env server, which makes it the one to reach for when something breaks: if
`rollout` works and `serve` does not, the fault is in the serving layer and nothing below it.

### 3. Serve it

```bash
openenv harbor serve --llm-url $LLM --dataset org/train,org/eval
```

You get a Task API for discovery, one long-running `run_rollout` MCP tool, and a UI at `/web` for
reading tasks, running agents on them and reading what they did (see [The web UI](#the-web-ui)).
`--llm-url` is optional: without it, whoever uses the UI connects a model of their own.

```python
from harbor_env import HarborEnv

with HarborEnv(base_url="http://localhost:8000") as env:
    split = env.splits()[0]["name"]
    result = env.run_rollout(split=split, task_index=0, harness="opencode", sandbox="e2b")

    print(result.reward, result.n_turns)
    for turn in result.turns:
        print(len(turn.completion_token_ids), sum(turn.per_token_logps))
```

`harness` and `sandbox` are per call, so consecutive rollouts against the same server can use
different agents and different backends.

## The intercept

The agent is a black box. It is a real CLI tool running in a sandbox, and it was never written with
training in mind. So instead of modifying it, an OpenAI-spec proxy is placed between it and your
model, and every call is recorded as it passes.

```
agent in a sandbox
   |  base URL points at the proxy, API key IS the capture session id
   v
capture proxy  --normalise to chat, ask for token ids-->  your endpoint
   ^                                                            |
   |  replay in the agent's own dialect  <----------------------+
   |
   +-- every call becomes a node in a rollout graph, linked by token prefix
       (by message prefix when the endpoint returns no token ids)
```

Three properties make this work across agents rather than for one:

**Nothing is tokenised locally.** The engine tokenises each prompt in order to serve it and hands
back `prompt_token_ids`, so turn *k+1*'s prompt is by construction the canonical tokenisation of
everything before it, tool results included. Re-rendering a prompt offline with a chat template
drifts from what the model actually saw, and a prompt that differs by one token silently splits one
long conversation into several short ones.

**Four wire dialects.** Coding agents did not converge on one API. chat-completions, OpenAI
Responses, Anthropic Messages and Google `generateContent` are all translated to a single upstream
shape and replayed in the dialect the agent expects, streaming included. That is what makes codex,
claude-code and gemini-cli work rather than only the chat-completions agents.

**The API key is the session id.** One proxy serves many concurrent rollouts with no port each, and
a caller without a registered session is rejected, so the proxy is safe to expose to a sandbox.

Turns are linked into a graph by **exact token prefix**: a call whose `prompt_token_ids` begin with
an existing node's full sequence becomes its child. Nothing else is consulted, because request ids
and timestamps are per-agent and the prefix is not. Conversations, retries and subagent branches
fall out of that for free, and a branch the agent abandoned is marked discarded so it is never
trained with the reward the main path earned.

## Prerequisites

You need an OpenAI-compatible endpoint. Which *kind* of rollout you get depends on what it can
return, and that is probed at startup rather than configured:

| | eval | train |
|---|---|---|
| reward, `rewards` dict, step results | yes | yes |
| full trace: conversations, per-turn text, tool calls | yes | yes |
| `prompt_token_ids`, `completion_token_ids`, `per_token_logps` | no | yes |
| `contract.json`, TRL rollout func | no | yes |

**For trainable rollouts** the endpoint must return token ids *and* the sampling distribution's
logprobs. Both are checked by probing, because both fail silently:

```bash
vllm serve <model> --return-tokens-as-token-ids --logprobs-mode processed_logprobs
# or SGLang built from git main (sgl-project/sglang#30917); no serving flag needed
```

Without them the endpoint answers every request normally and returns no token ids. Rollouts would
look perfect and contain nothing trainable, and that failure has no loud edge — so the level is
probed before the server binds a port, printed as `[EVAL ONLY]`, stamped on every result, and no
training contract is ever built from it.

**For eval rollouts** any reachable OpenAI-spec endpoint works, including hosted ones:

```bash
openenv harbor serve --llm-url https://api.openai.com/v1        --api-key $OPENAI_API_KEY    --model gpt-5.6-sol
openenv harbor serve --llm-url https://router.huggingface.co/v1 --api-key $HF_API_KEY        --model Qwen/Qwen3.6-35B-A3B
openenv harbor serve --llm-url https://api.anthropic.com/v1     --api-key $ANTHROPIC_API_KEY --model claude-sonnet-5
```

Anthropic needs no `--auth-header`: its OpenAI-compatible `/v1/chat/completions` accepts
`Authorization: Bearer`. Its `/v1/models` does not — that route wants `x-api-key` *and*
`anthropic-version` — so the model list comes back empty and `--model` is required. An endpoint that
publishes no usable model list is not treated as unreachable for exactly this reason; the completion
probe is what decides.

`--api-key` (or `$OPENENV_LLM_API_KEY`) authenticates the proxy to the endpoint; `--auth-header`
changes the header name when a provider wants something other than `Authorization: Bearer`. This is
not the key the agent receives — that one is a capture session id, minted per rollout — and it never
leaves the server process.

A hosted provider also rejects parameters a local vLLM accepts, per model rather than per endpoint.
Every current OpenAI model refuses `max_tokens` (wants `max_completion_tokens`), any `temperature`
other than 1, and `logprobs` outright, while opencode, codex and qwen-coder all send the first two.
`gpt-5.6` additionally refuses **function tools** on `/v1/chat/completions` unless
`reasoning_effort` is `"none"` — which for a coding agent is every call that matters; left unhandled
it presents as a rollout that makes one model call and then idles until the agent timeout.

The proxy reads each such 400, applies the one edit the provider names, retries, and caches the fix —
so the agents work unchanged. Every applied fix is reported on the result, at startup and in the UI,
because these are changed experiments rather than cosmetic rewrites: dropping `temperature` alters
the sampling distribution, and `reasoning_effort: "none"` turns reasoning off. For `gpt-5.6` the
better answer is the Responses route the error message itself recommends.

### What startup checks about *agents*, not just capture

The capture probe sends no tools. Every validated harness sends a tool manifest on every call, so a
second, non-fatal probe asks the way a harness asks and reports:

| finding | meaning |
|---|---|
| `tools: ok` | a tool call came back; agents can work here |
| `no_tool_calling` (FATAL) | the endpoint refuses a manifest outright; no agent rollout is possible |
| `no_tool_call_emitted` (WARN) | manifest accepted, but it answered in prose |
| `behaviour_changed` (WARN) | a compat fix changed how the MODEL behaves, not just a field name |

The last one is the reason this probe exists. `gpt-5.6` accepts function tools on
`/v1/chat/completions` only if `reasoning_effort` is `"none"`, and with reasoning off it emits one
valid tool call and then agentic loops die: goose and codex each managed a single model call and 0/3
tasks, while both scored 3/3 against a non-reasoning model on the same endpoint. Without a tool in the
probe body that demand is never made, so `harbor info` looked healthy and the failure only appeared
minutes into a rollout. Now it is reported before a sandbox is booted.

Truncation is deliberately treated as inconclusive rather than a failure: a reasoning model spends
output tokens thinking before it calls anything, and a small cap made `Qwen3.6-35B-A3B` look
tool-incapable when it in fact worked with all 16 harnesses.

### Why `--logprobs-mode processed_logprobs` is not optional

`token_ids` comes from the `return_token_ids` **request** parameter, not from either serving flag, so
a vLLM started with neither flag still returns aligned, negative, correctly-counted logprobs and would
grade as fully trainable. But vLLM's `logprobs_mode` defaults to `raw_logprobs` — the values *before*
temperature and top-k/top-p are applied — and GRPO's importance ratio needs the logprob under the
policy that actually sampled the token.

Startup measures which you have, rather than inferring it: the gap between the top two logprobs is
requested at temperature 1.0 and 2.0. Processed logprobs are `logsoftmax(logits / T)`, so the gap
scales by `1/T` while the normalising constant cancels; raw logprobs cannot move at all. Measured on
two live Qwen3.5-4B servers:

| | gap @T=1.0 | gap @T=2.0 | verdict |
|---|---|---|---|
| `--logprobs-mode processed_logprobs` | 6.7500 | 3.3750 | `processed` |
| default | 6.7500 | 6.7500 | `raw` |

A measured `raw` endpoint is downgraded to EVAL rather than refused — it is still a perfectly good
eval backend — and `OPENENV_ALLOW_RAW_LOGPROBS=1` overrides that if you know better than the probe.
The gap is compared rather than the values themselves because a data-parallel engine answers
consecutive calls from different replicas; comparing values directly misread one such engine.

## Training with TRL

TRL trains a policy on these rollouts with `AsyncGRPOTrainer` (`trl.experimental`, TRL 1.15 or
later): you pass a `HarborSessionFactory` (a `functools.partial` with the server URL), and a
`HarnessRolloutWorker` calls it with its sampling policy, opens one session per rollout and trains on each session's validated `TrainingTrace`.

- [`examples/async_grpo_harbor`](https://github.com/huggingface/trl/tree/main/examples/async_grpo_harbor):
  a complete training script, with local setup and a Hugging Face Jobs launcher.
- [TRL's OpenEnv guide](https://huggingface.co/docs/trl/openenv): how the worker, the reward and the
  capture fit together.
- [The ultimate guide to multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl):
  one policy trained across OpenCode, Claude Code, Codex and Mini-SWE-Agent through this
  environment. Its runnable code is the
  [FineEnvs multi-harness tutorial](https://github.com/adithya-s-k/FineEnvs/tree/main/05-multi-harness-rl),
  and its [models, datasets and environments](https://huggingface.co/collections/FineEnvs/smoldataenvs-multi-harness-rl-6abdfaaa8d74dacd481d5212)
  are in one collection.

## The web UI

`serve` (and a Space made with `push`) serves a UI at `/web`, in four tabs.

**Tasks.** Every task of every served dataset, as cards you can search and filter by dataset,
category, difficulty and tag. Opening one shows exactly what the agent will receive, the task's files
(with a full-window viewer), the environment and verifier settings from `task.toml`, and the runs of
that task. Metadata fields that hold the answer (`gold_answer`, `solution`, ...) are left out of the
summary. **Add from the Hub** lists public datasets tagged `harbor`, each with its task count and
size, and adds one in the background with its progress shown; a dataset added this way can be
removed again. Only datasets in Harbor's `tasks/<name>/` layout (flat or grouped) can be added.

**Running a task.** The run card on a task picks the model, the agent and the sandbox:

- **This server**: the endpoint `serve` was started with. Its key stays in the capture proxy.
- **Hugging Face**: a model on Inference Providers, chosen from a searchable list with providers,
  context length and price, reached with a Hugging Face token (locally, this machine's token; on a
  Space, optionally by signing in with Hugging Face).
- **Your endpoint**: any OpenAI-compatible URL (vLLM, SGLang, ...) or an Anthropic endpoint, probed
  before use. Training capture needs vLLM with `--return-tokens-as-token-ids`.

A rollout runs on the server whether or not the page stays open. Model usage is billed to the
account or token selected on the card; sandbox compute is billed to the server operator, including
Hugging Face Sandbox on a Space.

**Runs.** Every run you may see, filterable by status. A run shows what the agent did as a timeline
(its prompt, thinking, each tool call with its input and output, terminal sessions, the final
answer), the result and reward, the trace checks, and downloads of the result JSON and, for a
trainable rollout, the **Training trace** and **Capture audit**. The training trace is the same
validated `TrainingTrace` returned by `HarborSession.fetch_training_trace()`: selected agent calls,
engine token IDs, logprobs and explicit loss masks. Load it with
`TrainingTrace.model_validate_json(Path("run.training_trace.json").read_text())` after importing
`Path` from `pathlib` and `TrainingTrace` from `openenv.core.harness`. Task rewards and the full
rollout remain in Result JSON. Capture audit preserves the existing `contract.json` export,
including excluded turns. Older runs without explicit masks can still download their audit but
cannot produce the typed training trace. Tick two to four runs to compare them.

**Setup.** The server's endpoint, which sandboxes have working credentials, the deployment settings
below, the datasets and the agents.

### Deployment settings

The same UI runs on a laptop and as a public Space, and what a visitor may do differs. The defaults
depend on where the server runs: **local** is `serve --host 127.0.0.1`; **network** is any other bind
address, including the default `0.0.0.0`, since everyone who can reach it is then a visitor; a
**Space** is detected from `SPACE_ID`. Each setting is an environment variable (a Space variable on
a Space), and some have a flag on `serve` and `push` (`--private-urls` is on `serve` only).

| setting | variable | flag | local | network | Space |
|---|---|---|---|---|---|
| Visitors may start rollouts | `OPENENV_HARBOR_UI_ROLLOUTS` | `--rollouts` | on | on | on |
| Visitors may use the server's endpoint | `OPENENV_HARBOR_UI_SERVER_ENDPOINT` | `--share-endpoint` | on | on | off |
| Visitors may connect their own | `OPENENV_HARBOR_UI_VISITOR_ENDPOINTS` | `--visitor-endpoints` | on | on | on |
| A visitor's URL may be private or local | `OPENENV_HARBOR_UI_PRIVATE_URLS` | `--private-urls` | on | off | off |
| Offer this machine's HF token | `OPENENV_HARBOR_UI_LOCAL_TOKEN` | | on | off | never |
| Add and remove Hub datasets from the page | `OPENENV_HARBOR_UI_ADD_DATASETS` | `--add-datasets` | on | off | off |
| Who sees runs (`all` or `own`) | `OPENENV_HARBOR_RUN_VISIBILITY` | `--run-visibility` | all | all | own |
| Keep finished runs across restarts | `OPENENV_HARBOR_RUN_HISTORY` | `--run-history` | on | on | off |
| Headline reward of a task with several | `OPENENV_HARBOR_REWARD_KEY` | `--reward-key` | unset | unset | unset |
| Rollouts at once | `OPENENV_HARBOR_UI_MAX_RUNS` | | 4 | 4 | 4 |
| Rollouts at once per visitor | `OPENENV_HARBOR_UI_MAX_RUNS_PER_VISITOR` | | 4 | 4 | 2 |

`own` visibility ties runs to a random id kept in the visitor's browser; a run stores only a digest
of it. That keeps visitors' runs apart, but it is not sign-in: a visitor who clears the browser's
storage loses their runs, and it is no substitute for access control over traces you consider
private. The per-visitor limit counts a signed-in visitor's Hugging Face account, and otherwise that
browser id, so for anonymous visitors it stops one page from taking every slot, not someone set on
it; `OPENENV_HARBOR_UI_MAX_RUNS` bounds the total. Run history goes to `OPENENV_HARBOR_RUNS_DIR`, by default `~/.cache/openenv/harbor/runs`, or
`/data/harbor-runs` on a Space with the bucket mounted. `OPENENV_HARBOR_UI_MAX_ADD_GB` (default `5`)
caps the size of a dataset added from the page.

What the UI guarantees whatever the settings:

- A key or token typed into the page is sent to that endpoint by the server, held in server memory
  for that page only, and never written to disk, to run history or back to the page.
- With private URLs off, a visitor's URL, and every redirect it answers with, must resolve to public
  addresses, and is checked again before each rollout.
- Every dataset, task and file the page asks for is one the server serves or that was added from
  the page. Every file it reads, for a card, the task view, the file viewer or the environment
  check, must resolve inside the dataset's own folder (a link to that folder itself is followed),
  so a task reached through a link to anywhere else shows nothing and cannot run on a visitor's
  model. The Task API's instruction preview follows the same rule, and a registry dataset is
  anchored on Harbor's cache. A dataset added from the page may not contain a symbolic link at all.
- A task that reads the server's environment variables (`${VAR}` in `task.toml` or a compose file,
  or a bare name under a compose `environment:`), which is where the server's keys are, or its files
  (a compose `env_file`, `include`, `extends`, or a host path in a mount, secret, build context,
  cache, watch rule or device), or asks for more of the host than a folder (`privileged`,
  `cap_add`, the host's namespaces, the Docker socket, another container's volumes, a named volume
  or network with settings, the build's SSH agent) runs only on the server's own endpoint, never on
  a model a visitor connects: that model does what the visitor asks, printing the sandbox's
  environment included, into a trace the visitor reads. A dataset added from the page may not do
  either at all. Both files are checked as parsed, and one that doesn't parse counts as reading.
- A request that changes something in the UI (a rollout, an added dataset) is refused when a
  browser sends it from another site, so another page can't act through a visitor's browser. Behind
  a proxy that rewrites `Host`, list the public host in `OPENENV_HARBOR_UI_HOSTS` (comma-separated).
- Everything a model, a task or a tool produced is escaped before it is shown.

## CLI reference

Four commands. Every flag below is the complete set, with its type and default.
`openenv harbor <command> --help` prints the same thing.

Exit codes:

| code | meaning |
|---|---|
| `0` | success |
| `1` | ran, but failed. `rollout` returns this if **any** rollout in the batch was unusable |
| `2` | usage error: a missing or invalid flag. Nothing ran |

The `1` and `2` split matters if you are scripting this: `2` means the command never started, so
retrying it unchanged will fail the same way.

### `openenv harbor info`

Report what this machine can run. Read-only: boots no sandbox, starts no server, makes no rollout.

| flag | type | default | meaning |
|---|---|---|---|
| `--llm-url` | str | `""` | OpenAI-spec endpoint. Optional here; without it the LLM section is skipped and the rest still reports |
| `--model` | str | `""` | Served model id. Auto-detected when the endpoint serves exactly one |
| `--dataset` | str | none | Dataset spec. Repeatable, or comma-separated |
| `--env-file` | path | `""` | dotenv with provider credentials, loaded before the checks |
| `--verbose` | flag | off | List all 30 harnesses, not only the 16 validated |
| `--json` | flag | off | Emit machine-readable JSON instead of the text report |

```bash
openenv harbor info --llm-url $LLM --dataset org/tasks --json
```

The JSON has four top-level keys: `llm`, `sandboxes`, `datasets`, `harnesses`. Each sandbox entry is
`{name, available, detail}`, so a script can select a backend without parsing prose:

```bash
openenv harbor info --llm-url $LLM --json \
  | jq -r '.sandboxes[] | select(.available) | .name'
```

### `openenv harbor rollout`

Run rollouts with no env server involved. This is the debugging path: if `rollout` works and `serve`
does not, the fault is in the serving layer and nothing below it.

| flag | type | default | meaning |
|---|---|---|---|
| `--llm-url` | str | **required** | OpenAI-spec endpoint. No default and no env fallback, on purpose |
| `--dataset` | str | **required** | Dataset spec. Only the first is used by this command |
| `--task-index` | int | `0` | Index into the split. Stable: index is a task's identity |
| `-n`, `--n-tasks` | int | `1` | Run this many consecutive tasks from `--task-index` |
| `--harness` | str | `opencode` | A validated seam name, or `module:Class` for your own agent |
| `--sandbox` | str | `e2b` | Harbor environment type |
| `--model` | str | `""` | Served model id. Auto-detected when unambiguous |
| `--port` | int | `8100` | Local port for the capture proxy. One per concurrent process |
| `--expose` | str | `gradio` | How the sandbox reaches the proxy: `gradio`, `cloudflare`, `direct` |
| `--reward-key` | str | `""` | Which reward key is the training signal, for multi-reward tasks |
| `--trials-dir` | path | tmp | Where Harbor writes trial artifacts |
| `--keep-sandbox` | flag | off | Leave sandboxes alive for debugging |
| `--force-build` | flag | off | Rebuild the sandbox image, bypassing the content-hash cache |
| `--env-file` | path | `""` | dotenv with provider credentials |
| `--out` | path | `""` | Write the full result JSON, token ids and logprobs included |

```bash
openenv harbor rollout --llm-url $LLM --dataset org/tasks \
  --task-index 0 -n 5 --harness codex --sandbox modal --out results.json
```

`-n` runs tasks sequentially in one process, reusing one proxy and one forward. To parallelise, run
several processes and **give each its own `--port`**. Two processes sharing a port is refused with
an error naming the process that holds it.

Use `--force-build` when a task has never been built on this account, or when a cached image has
drifted because the task pins its dependencies loosely.

### `openenv harbor serve`

Start the env server: Task API for discovery, one long-running `run_rollout` MCP tool, and a UI at
`/web`.

| flag | type | default | meaning |
|---|---|---|---|
| `--llm-url` | str | `""` | OpenAI-spec endpoint. Optional: without one, UI visitors connect their own |
| `--dataset` | str | none | Dataset specs to serve as splits. Repeatable |
| `--model` | str | `""` | Served model id |
| `--host` | str | `0.0.0.0` | Bind address. `127.0.0.1` gives the UI its local defaults, see [Deployment settings](#deployment-settings) |
| `--port` | int | `8000` | Env server port. Faces the trainer and the browser |
| `--capture-port` | int | `8100` | Capture proxy port. Faces the sandbox |
| `--expose` | str | `gradio` | How the sandbox reaches the proxy |
| `--env-file` | path | `""` | dotenv with provider credentials |
| `--api-key` | str | `$OPENENV_LLM_API_KEY` | Credential for the endpoint, for a hosted provider |
| `--auth-header` | str | `Authorization` | Header to send it under, e.g. `x-api-key` |
| `--share-endpoint/--no-share-endpoint` | flag | unset | UI visitors may use this endpoint and its key |
| `--visitor-endpoints/--no-visitor-endpoints` | flag | unset | UI visitors may connect their own model |
| `--run-visibility` | str | unset | `all` or `own`: who sees runs in the UI |
| `--run-history/--no-run-history` | flag | unset | Keep finished UI runs across restarts |
| `--add-datasets/--no-add-datasets` | flag | unset | UI visitors may add and remove Hub datasets |
| `--rollouts/--no-rollouts` | flag | unset | UI visitors may start rollouts |
| `--private-urls/--no-private-urls` | flag | unset | UI visitors may connect an endpoint on a private or local address |
| `--reward-key` | str | unset | The reward UI runs report for tasks with several and none named `reward`. Unset, such a run lists each one |

An unset UI flag keeps the default for where the server runs.

Refuses to start only if the endpoint is unreachable. One that cannot return token ids starts as an eval deployment and says so.

### `openenv harbor push`

Deploy the same server to a Hugging Face Space.

| flag | type | default | meaning |
|---|---|---|---|
| `--llm-url` | str | `""` | Endpoint the deployed Space will use. Optional: without one, visitors connect their own |
| `--repo-id` | str | **required** | Target Space, e.g. `you/harbor-env` |
| `--dataset` | str | none | Dataset specs. Repeatable |
| `--model` | str | `""` | Served model id |
| `--bucket` | str | Space name | Storage bucket holding the task suites. `none` disables the mount and downloads instead |
| `--public-bucket/--private-bucket` | flag | private | Visibility of a new bucket. Given for an existing bucket, it changes it; not given, an existing bucket keeps its visibility |
| `--hf-login/--no-hf-login` | flag | on | Let visitors sign in with Hugging Face to use their own account for Inference Providers |
| `--hardware` | str | `""` | Space hardware, e.g. `cpu-basic` |
| `--private` | flag | off | Create the Space private. Rollouts then cannot work, see below |
| `--recreate` | flag | off | Delete the Space first, then deploy fresh |
| `--dry-run` | flag | off | Print exactly what would be sent and stop |
| `--env-file` | path | `""` | dotenv whose provider keys become Space **secrets** |

`push` also takes the UI flags of `serve` (`--share-endpoint`, `--visitor-endpoints`,
`--run-visibility`, `--run-history`, `--add-datasets`, `--rollouts`, `--reward-key`) and sets them as Space
variables. Unlike `serve`, it doesn't share the endpoint with UI visitors unless `--share-endpoint` is
given. Each push reconciles these UI variables: an omitted flag removes an earlier override and
returns to the Space default, so a one-time `--share-endpoint` does not stay enabled forever.

```bash
openenv harbor push --llm-url $LLM --dataset org/train,org/eval \
  --repo-id you/harbor-env --env-file .env --dry-run
```

`--private` is supported but rollouts will not work on a private Space: the capture proxy is served
at `<space-url>/capture`, and a private Space requires an auth header the sandboxed agent does not
send. Use it only to park a deployment.

### Migration notes for the new UI

- `serve` binds `0.0.0.0` by default and therefore uses the network-safe UI defaults. Visitor
  endpoints such as `http://localhost:8000/v1` are refused. Bind the whole server to
  `--host 127.0.0.1`, or pass `--private-urls` when the server still needs a remote trainer.
- An existing Space re-pushed with `--llm-url` no longer shares that endpoint with UI visitors
  unless `--share-endpoint` is passed. `--hf-login` also writes `hf_oauth: true` and the
  `inference-api` OAuth scope into the Space README.
- Capture `/health` still returns status and aggregate fields publicly, but its per-session
  `upstreams` list is empty when an admin key is configured. Set
  `OPENENV_CAPTURE_ADMIN_KEY` and send it as a bearer token to inspect that field.

## Supported harnesses

16 of the 30 known agents are validated end to end. "Validated" means a real rollout produced token
ids and logprobs, and where the agent emits a trajectory, its own record agreed with the capture.

| harness | dialect | runs |
|---|---|---|
| `opencode` | chat-completions | in sandbox |
| `goose` | chat-completions | in sandbox |
| `qwen-coder` | chat-completions | in sandbox |
| `swe-agent` | chat-completions | in sandbox |
| `mini-swe-agent` | chat-completions | in sandbox |
| `openhands-sdk` | chat-completions | in sandbox |
| `openclaw` | chat-completions | in sandbox |
| `hermes` | chat-completions | in sandbox |
| `kimi-cli` | chat-completions | in sandbox |
| `pi` | chat-completions | in sandbox |
| `vibe` | chat-completions | in sandbox |
| `terminus-2` | chat-completions | **host side** |
| `codex` | OpenAI Responses | in sandbox |
| `trae-agent` | OpenAI Responses | in sandbox |
| `claude-code` | Anthropic Messages | in sandbox |
| `gemini-cli` | Google generateContent | in sandbox |

Supporting four dialects rather than chat-completions alone is what makes the last four rows work.

`terminus-2` runs in the server process rather than inside the sandbox, so it reaches the proxy on
localhost and needs no public URL.

The other 14 known agents have a seam but are untested; run `openenv harbor info --verbose` to list
them. Anything Harbor supports can be reached with `--harness module:Class`.

## Supported sandboxes

`openenv harbor info` checks these four by default and reports why any is unusable:

| sandbox | credentials | validated |
|---|---|---|
| `e2b` | `E2B_API_KEY` | yes, extensively |
| `modal` | `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`, or `~/.modal.toml` | yes, extensively |
| `docker` | none, but the daemon must be running | works, not swept |
| `daytona` | `DAYTONA_API_KEY`, or `DAYTONA_JWT_TOKEN` + `DAYTONA_ORGANIZATION_ID` | not swept |

e2b and modal were compared on identical tasks and came out indistinguishable, which is the check
that matters: a backend-specific capture bug is exactly what a single-backend test hides.

Harbor registers 23 backends in total. Any of them can be passed to `--sandbox`; the four above are
the ones with a credential check wired in.

## Environment details

### Where the proxy runs

Locally there are two ports: the env server faces the trainer and the browser, the capture proxy
faces the sandbox and is the only one published. Sharing one port would expose the env server the
moment the proxy became reachable.

Hosted, that inverts. A Space has one port and one public URL, so the proxy is mounted on the env
server's own app at `/capture` and nothing is forwarded. It still rejects callers without a
registered session id, which is what keeps a public mount from being an open relay.

### Sandboxes and providers

Two different things share the word "sandbox", and mixing them up is a common early confusion:

- **Harbor backends** are where the *agent* runs. That is what `--sandbox` selects.
- **OpenEnv providers** (`local_docker`, `hf_sandbox`, `modal`, `aca`, ...) host the *env server*
  itself and have no `exec`.

A backend counts as usable only if its class imports **and** Harbor's own preflight passes.
Credentials alone are not enough: a provider with valid keys but no SDK installed would otherwise
report available and fail at rollout time.

### Rewards

The verifier produces a dictionary. OpenEnv wants one number. The dictionary is forwarded unchanged
and the scalar is chosen by an explicit rule:

1. one key, use it
2. a key named `reward`, use it
3. otherwise fail and ask for `--reward-key`

Combining several keys automatically would be inventing reward semantics, so it refuses instead.
All shaping belongs in the trainer.

**`reward=None` is not zero.** It means the verifier never ran. A dead sandbox scored as zero looks
like a wrong answer, which is how an infrastructure failure gets mistaken for a model result.

### Reading a result

| field | meaning |
|---|---|
| `ok` | the rollout is usable. `False` means something failed, and `error` says what |
| `reward` | the verifier's number, or `None` if it never ran |
| `n_turns` | model calls captured |
| `n_roots` | independent conversations. More than one means subagents or auxiliary calls |
| `turns[]` | per-call token ids, logprobs, text and tool calls |
| `conversations[]` | the full message list per conversation, system prompt included |
| `atif` | `match`, `MISMATCH`, or `none`. See below |
| `findings` | warnings worth reading before training on the rollout |

`atif` is an independent cross-check. Harbor's own trajectory file records what the *harness*
thought happened; the capture records what crossed the wire. Two measurements of the same rollout
through completely different paths. `match` means they agree call for call.

**A failed rollout returns a result, never an exception.** That is deliberate. In an in-process
design one exception on one rank hangs every rank at the distributed barrier, so behind an HTTP
boundary a failure has to come back as `ok=False` instead.

## Deploying to Hugging Face Spaces

```bash
openenv harbor push \
  --llm-url $LLM \
  --dataset org/train,org/eval \
  --repo-id you/harbor-env \
  --env-file .env
```

Configuration travels as Space variables, provider credentials as Space secrets. Add `--dry-run`
to print exactly what would be sent first, and `--recreate` to delete and redeploy for a clean test.

Two details that matter:

**Task suites are mounted, not downloaded.** A Harbor suite is thousands of small files and Space
disk is ephemeral, so a download is re-paid on every restart. `push` syncs the suites into a storage
bucket named after the Space and mounts it at `/data`. The copy is server side, and re-running
`push` copies only what is new. The bucket is created private (`--public-bucket` to change that),
since with run history on it also holds every visitor's runs. With `--add-datasets`, the `tasks/`
folder of a dataset added from the UI is copied into the same bucket, server side, and read through
the mount, so it survives restarts; removing it deletes it from the bucket.

**Visitors can sign in with Hugging Face.** `push` turns on OAuth for the Space (`hf_oauth: true`,
scope `inference-api`), and the UI offers "sign in with Hugging Face" as a way to use Inference
Providers on the visitor's own account instead of pasting a token. It is optional and not needed
for anything else; `--no-hf-login` leaves it out.

**Visitors bring their own model.** On a Space, a visitor runs on a model they connect: signing in
with Hugging Face, a token, or their own endpoint. `--share-endpoint` lets them use the Space's own
endpoint and key too, at your cost; the Task API and MCP use it either way. On that endpoint a
served task that reads the Space's environment does run, and the visitor who starts it reads its
trace, so don't combine `--share-endpoint` with such tasks on a public Space.

**Rollouts on a Space use the Space's credentials for sandboxes.** An `hf-sandbox` rollout is billed
to the `HF_TOKEN` the Space holds, whoever started it. `--no-rollouts` makes a public deployment a
read-only task browser.

**The Space must be public.** The capture proxy is served at `<space-url>/capture`, and a private
Space requires an auth header that the agent inside the sandbox does not send. This is safe because
the proxy rejects any caller without a registered session id, so a public mount is not an open
relay.

## Troubleshooting

**Rollout finishes with zero model calls.** The agent never reached the proxy. Usually the endpoint
URL is wrong, the model name did not resolve, or auth was rejected. Check the `findings` field,
which names the likely cause.

**`llm ... [FAILED]` at startup.** The endpoint is unreachable — wrong URL, wrong model name, or
a missing `--api-key` for a provider that needs one. The findings name which.

**`llm ... [EVAL ONLY]` at startup.** The endpoint answers but returns no token ids, so rollouts
carry the reward and the trace and nothing trainable. Expected for OpenAI, Anthropic and HF
Inference Providers. If you meant to train, restart the engine with
`--return-tokens-as-token-ids --logprobs-mode processed_logprobs`, or build SGLang from git main —
released SGLang carries only `return_prompt_token_ids`, which is not enough.

**`contract.json` is missing after a rollout.** The rollout was an eval rollout; check
`rollout_type` and `capture_level` on the result. There is deliberately no all-zero contract to
download, because a file named `contract.json` containing no contract is more convincing than an
empty list.

**A sandbox shows `[--]` in `info`.** The detail column says why, and it is usually a missing
credential or a missing SDK. Install everything with `pip install "openenv[harbor]"`.

**`atif=none`.** That harness writes no trajectory file, so no cross-check is possible. Capture is
unaffected.

**Many roots for one rollout.** Normal for agents that run subagents or auxiliary calls. Each root
is a separate conversation, and only agent conversations are counted as trainable.

**A dataset is greyed out in "Add from the Hub".** It has no `tasks/<name>/` folder, which is how
Harbor datasets are laid out, or it is over `OPENENV_HARBOR_UI_MAX_ADD_GB`.

**A URL is refused as "private or local".** The server does not call private addresses for visitors
unless `OPENENV_HARBOR_UI_PRIVATE_URLS` is on, which by default it is only for `--host 127.0.0.1`;
`serve --private-urls` turns it on for any bind address. To reach a vLLM on your own network from a
Space, run the UI yourself.

**Exit code 137.** The agent was killed inside the sandbox, almost always by the OOM killer on a
large input. That is a task failure, not a capture failure.

## References

- [Harbor](https://github.com/laude-institute/harbor), which provides the datasets, sandboxes,
  agents and verifiers
- [Polar](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server)
  ([paper](https://arxiv.org/abs/2605.24220)), the black-box approach this capture layer follows,
  and the source of the vendored dialect transformers
- [verifiers](https://github.com/willccbb/verifiers), whose `Dialect` model informed the auxiliary
  route and streaming handling
