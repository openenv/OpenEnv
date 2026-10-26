# CLI

The `openenv` CLI provides a set of commands for building, validating, and pushing environments to Hugging Face Spaces or a custom Docker registry. For a walkthrough of building an environment, see [Your First Environment](../guides/first-environment).

## `openenv init`

[[autodoc]] openenv.cli.commands.init.init

## `openenv import`

Import a supported third-party source environment into a generated OpenEnv
wrapper package. The command detects the source format from the directory
contents, so ORS/OpenReward and Prime Intellect Verifiers sources do not
require `--type` in the common case.

The generated wrapper vendors the source tree into the package and includes
vendored files as package data, so non-secret fixture/data files are available to
the environment server at runtime. The importer carries portable dependencies
from source `pyproject.toml` and `requirements.txt` files into the generated
environment, skips VCS/cache/build directories and common secret file patterns
such as `.env`, `secrets.yaml`, and private key files, and excludes compiled
binary artifacts; review the generated `vendor/` directory before publishing a
wrapper.

```bash
openenv import path/to/source --name my_env --output-dir ./envs
openenv import path/to/source --name my_env --output-dir ./envs --env-class MyEnv
```

[[autodoc]] openenv.cli.commands.import_env.import_env

## `openenv build`

[[autodoc]] openenv.cli.commands.build.build

## `openenv validate`

Run static declaration checks without Docker, or build and probe a package locally:

```bash
openenv validate ./my_env --level static
openenv validate ./my_env --level runtime --local --output report.json
```

Runtime validation requires Docker and a `validation.execution` declaration in
`openenv.yaml`. The declaration points to a bounded JSON replay plan containing
one reset and a sequence of actions. The validator builds an immutable image,
opens one WebSocket session, measures reward values, observation schemas and
state continuity, and removes the container even if a check fails.

The runtime level implements startup, reward, observation, state and network
policy checks. Other applicable Level 2 checks appear explicitly as `SKIP`; they make
the result `WARN`, which exits zero and does not mean Level 2 is complete.
`FAIL` exits 1, unsupported package formats exit 2, and internal or policy errors
exit 3. `--level semantic`
includes the available lower-level checks but does not claim semantic execution.
`--skip-build` runs declaration checks and skips runtime execution entirely.
The Docker provider supports CPU workloads in `public` and `no-network` modes;
`allowlist` and GPU requirements skip runtime before building. A `no-network`
subject has no network interface: the validator reaches it through a helper that
shares only its network namespace, built from a digest-pinned validator image
pulled on first use. `runtime.network_policy` probes a validator-owned sink from
the subject's namespace and from a control container; a probe counts only when
the control reaches the sink, and a denial that is not proven is a `SKIP`.
`--local` explicitly selects this default package mode and rejects a remote URL.
The declared `episode_timeout_s` bounds collection; a reset or judged step may use
its remaining budget. Collection failures appear once under `runtime.startup`,
with dependent contract checks skipped and the completed trace retained.

Credential delivery is deferred for this release. Set
`validation.execution.requires_credentials: true` when the environment needs an
externally supplied credential, such as a judge API key. This boolean declaration
causes a named `credential_delivery` SKIP before build or launch; dependent runtime
checks also SKIP. The resulting WARN does not establish complete Level 2 coverage.
The declaration accepts no secret values and defaults to false. Validation never
inherits host credentials or forwards API keys. Self-contained LLM judges can run
without this requirement; `llm_judged` alone does not imply a need for credentials.
An undeclared startup or reset failure still fails validation.

Observation validation uses the `reset_observation` field from `/schema` for resets
and `observation` for steps. Servers can declare `reset_observation_cls` when
reset returns a different observation type; it defaults to the step observation
class. Older servers without a reset schema use the step schema for both. Reset
validation is always required; the validator does not silently substitute a base
schema. These are output schemas, separate from any reset input declaration.
The Level 2 profile permits null rewards only on reset. Every step, including a
nonterminal step, requires a finite numeric reward within the declared range;
boolean rewards are invalid. This certification requirement is stricter than the
backward-compatible core `Observation.reward` type. Emit zero only when the intended
reward is zero; the validator never converts missing or null step rewards to zero.

Cleanup removes run-owned containers. Built images remain in Docker's local cache
for reuse; the report records their immutable image IDs. Remove an unwanted image
with `docker image rm <image-id>` after its validation runs have finished.

Runtime reports use schema version 2 and severity policy v2. Static reports for
v1 manifests retain schema version 1 and policy v1. An explicit v1 policy with a
runtime ceiling is rejected. With `--output report.json`, a sibling
`report.artifacts/` directory contains the replay plan, bounded redacted evidence,
coverage inventory, provider settings, cleanup outcome and checksums. Treat these
as author-validation evidence; this command does not issue certification.

For a pinned, installed-wheel reproduction of the shared fixture and its fault
cases, follow [the runtime lab](../../../tests/validation_runtime/README.md).
The implementation and remaining check inventory are tracked in
[the Level 2 umbrella](https://github.com/huggingface/OpenEnv/issues/1177).

[[autodoc]] openenv.cli.commands.validate.validate

## `openenv push`

[[autodoc]] openenv.cli.commands.push.push

## `openenv serve`

Local serving is not implemented in the CLI yet. This command exits non-zero
and prints alternative ways to run an environment server.

[[autodoc]] openenv.cli.commands.serve.serve

## `openenv fork`

[[autodoc]] openenv.cli.commands.fork.fork

## `openenv skills`

Installs an `openenv-cli` skill into your AI assistant's skills directory so
it knows the `openenv` CLI is available and what each command does. Supports
Claude Code, Cursor, Codex, and OpenCode.

**Install for a single assistant (project-local):**

```bash
openenv skills add --claude    # → .claude/skills/openenv-cli/
openenv skills add --cursor    # → .cursor/skills/openenv-cli/
openenv skills add --codex     # → .codex/skills/openenv-cli/
openenv skills add --opencode  # → .opencode/skills/openenv-cli/
```

Multiple flags can be combined — `openenv skills add --claude --cursor` installs
for both at once. The skill file is written to a central location
(`.agents/skills/openenv-cli/`) and each agent directory gets a symlink, so
there is only one copy to update.

**Install globally (user-level, across all projects):**

```bash
openenv skills add --claude --global  # → ~/.claude/skills/openenv-cli/
```

**Overwrite an existing installation** (e.g. after upgrading `openenv`):

```bash
openenv skills add --claude --force
```

**Preview the skill content without installing:**

```bash
openenv skills preview
```

**Install to a custom path** (for non-standard agent setups):

```bash
openenv skills add --dest /path/to/my-agent/skills/
```

[[autodoc]] openenv.cli.commands.skills.skills_add

[[autodoc]] openenv.cli.commands.skills.skills_preview

## `openenv collect`

Collect rollouts from a running environment with a teacher model and write
them as an SFT-ready `results.jsonl`. The teacher can be a hosted provider
(`--provider openai|anthropic`) or any self-hosted OpenAI-compatible server
such as vLLM, TGI or Ollama via `--llm-endpoint`.

```bash
# Scripted teacher, no API key needed
openenv collect openspiel:tic_tac_toe --base-url http://localhost:8001 \
    --output-dir ./rollouts -n 10 --provider scripted

# Self-hosted model (vLLM serving on port 8000)
openenv collect reasoning_gym:chain_sum --base-url http://localhost:8001 \
    --output-dir ./rollouts -n 50 \
    --llm-endpoint http://localhost:8000 --model Qwen/Qwen3-1.7B
```

`--llm-endpoint` takes a full base URL. `/v1` is appended when the URL has no
path, so `http://localhost:8000` and `http://localhost:8000/v1` are equivalent;
a URL with a path (for example a gateway prefix) is used as-is. `--llm-port` is
only needed when the URL does not include a port; it has no default, so
`--llm-endpoint http://localhost` means port 80 (earlier releases assumed 8000).
The resolved endpoint is printed when the run starts. Only `http(s)` URLs are
accepted, and credentials, query strings and fragments in the URL are rejected:
pass the key through `OPENAI_API_KEY` instead.

[[autodoc]] openenv.cli.commands.collect.collect

## `openenv harbor`

Runs [Harbor](../environments/harbor) tasks with real coding agents and
captures the token ids and logprobs of every model call. Requires
`pip install "openenv[harbor]"` (Python 3.12 or newer).

| Command | What it does |
|---------|--------------|
| `openenv harbor info` | Reports the endpoint's capture level and which sandboxes, datasets and harnesses this machine can use. Boots nothing. |
| `openenv harbor rollout` | Runs one or more rollouts end to end without a server. |
| `openenv harbor serve` | Serves Harbor tasks over the Task API and MCP, with a UI at `/web`. |
| `openenv harbor push` | Deploys the same server to a Hugging Face Space. |

```bash
openenv harbor info --llm-url $LLM --dataset AdithyaSK/data_agent_rl_environment_eval

openenv harbor rollout --llm-url $LLM \
    --dataset AdithyaSK/data_agent_rl_environment_eval \
    --task-index 0 --harness opencode --sandbox e2b --out rollout.json

openenv harbor serve --llm-url $LLM --dataset org/train,org/eval
```

`--llm-url` is any OpenAI-compatible endpoint. Rollouts carry token ids only
when it returns them (vLLM with `--return-tokens-as-token-ids
--logprobs-mode processed_logprobs`, or SGLang). `--api-key` defaults to
`$OPENENV_LLM_API_KEY`. Run `openenv harbor <command> --help` for every flag,
and see the [Harbor environment page](../environments/harbor#cli-reference)
for details.

## `openenv catalog` and `openenv discover`

Build a versioned, metadata-only catalog of environments from a Git
repository, then search it before installing or running anything. See
[Discover environments before running them](../guides/catalog-discovery).

```bash
openenv catalog build --repository . \
    --repository-uri https://github.com/huggingface/OpenEnv.git \
    --publisher example.org --output catalog.json

openenv discover "client smoke test" --catalog catalog.json
openenv discover "" --catalog catalog.json --filter license=BSD-3-Clause
openenv catalog inspect "<identifier from the result>" --catalog catalog.json
```

`catalog build` reads committed `openenv.yaml` files under `--root` (default
`envs`) at `--revision` (default `HEAD`). `discover` ranks entries by query
match (`--limit`, default 20) and `--json` emits the full entries.
`catalog inspect` prints one entry by its exact identifier.

# API Reference

## Entry point

[[autodoc]] openenv.cli.__main__.main

## CLI helpers

[[autodoc]] openenv.cli._cli_utils.validate_env_structure

## Validation utilities

[[autodoc]] openenv.cli._validation.validate_running_environment

[[autodoc]] openenv.cli._validation.validate_multi_mode_deployment

[[autodoc]] openenv.cli._validation.get_deployment_modes

[[autodoc]] openenv.cli._validation.format_validation_report

[[autodoc]] openenv.cli._validation.build_local_validation_json_report
