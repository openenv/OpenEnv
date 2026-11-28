# CLI

The `openenv` CLI provides a set of commands for building, validating, and pushing environments to Hugging Face Spaces or a custom Docker registry. For a walkthrough of building an environment, see [Your First Environment](../guides/first-environment).

## `openenv init`[[openenv.cli.commands.init]]

#### openenv.cli.commands.init[[openenv.cli.commands.init]]

```python
openenv.cli.commands.init(env_name: Annotated[str, typer.Argument(help="Name of the environment to create (snake_case, e.g., 'my_env')")], output_dir: Annotated[str | None, typer.Option('--output-dir', '-o', help='Output directory (defaults to current working directory)')] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/init.py#L404)

Initialize a new OpenEnv environment.

Creates a new directory with the environment name and generates all necessary
files based on the OpenEnv template structure.

Examples:

```bash
$ openenv init my_game_env
$ openenv init my_env --output-dir /path/to/projects
```

## `openenv import`[[openenv.cli.commands.import_env]]

Import a supported third-party source environment into a generated OpenEnv
wrapper package. The command detects the source format from the directory
contents, so ORS/OpenReward, Prime Intellect Verifiers and AgentEnv sources do
not require `--type` in the common case.

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

An [AgentEnv](https://github.com/scaleapi/agentenv-framework) bundle is a folder
whose `tasks/` holds task files. When a source holds several bundles, pick one
with `--env-class` and its folder name; only that folder is vendored. The
generated server runs it with `build_app` from `agentenv-framework`, which by
default deploys each episode using local Docker. The server needs a Docker CLI
and a running daemon accessible for image builds and episode containers. The
generated `server/Dockerfile` does not supply that runtime, so `openenv build`
and `openenv push` alone do not produce a working deployment. The generated
README describes local startup and these prerequisites. The generated client is
an `MCPToolClient`, since the served environment takes MCP actions. See
[AgentEnv's OpenEnv page](https://www.agentenvframework.com/docs/openenv) for
what it serves and what it needs.

```bash
openenv import path/to/bundle --name my_env --output-dir ./envs
openenv import path/to/plugin --name my_env --output-dir ./envs --env-class my-bundle
```

#### openenv.cli.commands.import_env[[openenv.cli.commands.import_env]]

```python
openenv.cli.commands.import_env(source: Annotated[str, typer.Argument(help='Local source repository or directory to import')], name: Annotated[str, typer.Option('--name', '-n', help='Name for the generated OpenEnv package')], output_dir: Annotated[str, typer.Option('--output-dir', '-o', help='Directory where the generated package will be created')], env_class: Annotated[str | None, typer.Option('--env-class', help='Environment class name or module:Class when detection is ambiguous')] = None, source_type: Annotated[str | None, typer.Option('--type', help="Optional source type override, such as 'ors'")] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/import_env.py#L58)

Deterministically import a third-party environment into OpenEnv.

## `openenv build`[[openenv.cli.commands.build]]

#### openenv.cli.commands.build[[openenv.cli.commands.build]]

```python
openenv.cli.commands.build(env_path: Annotated[str | None, typer.Argument(help='Path to the environment directory (default: current directory)')] = None, tag: Annotated[str | None, typer.Option('--tag', '-t', help='Docker image tag (default: openenv-<env_name>)')] = None, context: Annotated[str | None, typer.Option('--context', '-c', help='Build context path (default: <env_path>/server)')] = None, dockerfile: Annotated[str | None, typer.Option('--dockerfile', '-f', help='Path to Dockerfile (default: <context>/Dockerfile)')] = None, no_cache: Annotated[bool, typer.Option('--no-cache', help='Build without using cache')] = False, build_arg: Annotated[list[str] | None, typer.Option('--build-arg', help='Build arguments (can be used multiple times, format: KEY=VALUE)')] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/build.py#L332)

Build Docker images for OpenEnv environments.

This command builds Docker images using the environment's pyproject.toml
and uv for dependency management. Run from the environment root directory.

Examples:

```bash
# Build from environment root (recommended)
$ cd my_env
$ openenv build

# Build with custom tag
$ openenv build -t my-custom-tag

# Build without cache
$ openenv build --no-cache

# Build with custom build arguments
$ openenv build --build-arg VERSION=1.0 --build-arg ENV=prod

# Build from different directory
$ openenv build envs/echo_env
```

## `openenv validate`[[openenv.cli.commands.validate]]

Run static declaration checks without Docker, or build and probe a package locally:

```bash
openenv validate ./my_env --level static
openenv validate ./my_env --level runtime --local --output report.json
```

For served packages, runtime validation requires Docker and a `validation.execution` declaration in
`openenv.yaml`. The declaration points to a bounded JSON replay plan containing
one reset and a sequence of actions. The validator builds an immutable image,
collects the original episode, repeats the plan in a fresh WebSocket session and
with a different seed, then replays it in a fresh container when the provider
supports that capability. Docker validation uses two containers concurrently
during the fresh-container replay. Both use the same immutable image and are
removed even if a check fails.

Harbor `task.toml` directories are also recognized. Validation requires Python 3.12 or newer
and `harbor==0.25.0` installed alongside `openenv[harbor]`:

```bash
pip install "openenv[harbor]" "harbor==0.25.0"
```

A missing or mismatched Harbor installation exits 3 and names `harbor==0.25.0`.

Add a `validation/openenv.json` contract with `profile: "harbor-semantic-v1"`,
`reward_key`, `reward`, `resources` and `types`. The reward range, oracle tolerance
and floor margin must be explicit. Oracle tolerance and floor margin are fractions
of the declared reward range. Numeric strings in numeric fields, oversized integers
and excessive JSON nesting fail `static.manifest`. Reward and resource numbers must
retain their exact value when stored as floats. `resources.gpu_types` remains a list
of strings. This is a minimal complete contract:

```json
{
  "profile": "harbor-semantic-v1",
  "reward_key": "reward",
  "reward": {
    "range": [0, 1],
    "oracle_tolerance": 0,
    "floor_margin": 0.1
  },
  "resources": {
    "cpu": 1,
    "memory_mb": 512,
    "disk_mb": 1024,
    "episode_timeout_s": 300
  },
  "types": {"tags": ["swe"]}
}
```

The `harbor-semantic-v1` binding runs
fresh Harbor trials, in local Docker unless `--sandbox hf-sandbox` moves them to
HF Jobs: a do-nothing agent, the oracle (`solution/solve.sh`),
an oracle replay and any canaries declared in the contract. A canary is a known-wrong
`solve.sh` replacement. The results report under `runtime.startup`,
`runtime.reward_well_formed`, `runtime.episode_determinism`, `semantic.oracle_max`,
`semantic.floor_gap` and `semantic.canary_floor`, and `harbor-evidence.json` keeps
the raw trial evidence in the report bundle. Other checks are named SKIPs, so a
passing task still reports WARN. This validates task packages, not the outer
`harbor_env` server.

Supported tasks use Linux, a single step, a shared verifier and public networking.
The image comes from one Dockerfile or from a digest-pinned
`environment.docker_image`, such as `org/image@sha256:<64 hex digits>`. Harbor
prefers a declared image over a Dockerfile, so the do-nothing trial pulls the pin
and `runtime.startup` records it as `source_image`. The validator leaves the pulled
image in the local Docker cache. Unsupported shapes include multi-step or non-Linux
tasks, an `environment.docker_image` without a digest, tasks with neither a
Dockerfile nor a pinned image and task-authored Compose files. GPU/TPU or external
MCP requirements, credentials, non-public or differing phase networks, task
environment variables, healthchecks, skills and verifier collection are also
unsupported. Separate verifier environments and artifact extensions run only on
`hf-sandbox`. A served-package provider, including `--space`, is unsupported. Each case
gives `runtime.startup` SKIP with a named reason and runs no trials. With valid
static declarations and an oracle, the verdict is WARN and the exit code is 0.
A missing `solution/solve.sh` still FAILs `semantic.oracle_max` at the semantic level,
including with `--skip-build` or `--space`.

The initial image build and environment start share a separate 600-second validator
budget. Exceeding it gives `runtime.startup` ERROR. Later environment starts use the
episode budget. There is no shared outer trial deadline. Harbor bounds setup, agent
and verifier separately with the declared `resources.episode_timeout_s`. A timeout
in one of these phases is a package FAIL that names the phase.

The do-nothing trial builds from the task's Dockerfile, or pulls the digest-pinned
image when the task declares one. The validator keeps `openenv-validation:<run id>`
only when the run has no FAIL or ERROR and was not interrupted. It records the tag in `runtime.startup.measured` as
`retained_image_tag`. Otherwise it removes the tag. The run-owned Compose build tag
is removed on every run. Tag removal verifies image identity and uses `--no-prune`.
A removal failure marks cleanup incomplete and gives a startup ERROR. During an
abort, cleanup preserves the original exception and records the removal failure.

`--sandbox hf-sandbox` runs the same trials on Hugging Face Jobs instead, one job
per trial, through Harbor's `hf-sandbox` environment:

```bash
openenv validate path/to/task --sandbox hf-sandbox
openenv validate path/to/task --sandbox hf-sandbox --flavor cpu-upgrade --namespace my-org
```

A sandbox cannot build a Dockerfile, so the task must pin a pullable image as
`environment.docker_image = "name@sha256:<digest>"`. Build and push the image
first, then pin its digest. The image needs `/bin/bash`, which Harbor uses to run
commands, and either `sha256sum` or `openssl`, which the sandbox server uses to
verify its own download. With nothing
to build, every trial starts at once. `--flavor` and `--namespace` mean what they
mean for `hf jobs run`: the flavor sets the hardware, and the namespace is the user
or organization that owns and pays for the jobs, by default the token's user. The
flavor must provide the CPU, memory and disk the contract declares, or
`runtime.startup` reports ERROR before any job starts.

HF Sandbox cannot turn the network off or switch users. Tasks that declare
`no-network` for any phase, or set an agent or verifier user, are unsupported,
and `harbor-evidence.json` records that every phase ran with public networking
as the image's default user.

A task with `verifier.environment_mode = "separate"` runs its verifier in a second
job, as Harbor does on other backends. That job starts from the task's pinned
image. Harbor copies each `[[artifacts]]` path from the agent's job into it and
uploads `tests/` at grading time, so test files never need to be in the image.
A separate verifier with its own image, either `verifier.environment.docker_image`
or a `tests/Dockerfile`, is unsupported.

Each job, including a separate verifier's, carries `openenv-validation-run` and
`openenv-validation-role` labels. At
`AGENT_START` the validator reads the job back and requires the pinned image, the
requested flavor and a `RUNNING` stage. After each trial it cancels any labelled
job that is still running and waits up to 60 seconds for all of them to stop.
Otherwise cleanup is unverified and `runtime.startup` reports ERROR.
`runtime.startup.measured` records the flavor, namespace and job IDs, and
`harbor-evidence.json` states that the image is the digest requested from HF Jobs,
not one measured inside the sandbox. Check IDs, graders and thresholds are the
same as on Docker, and no image is retained. A served package gives a named
`runtime.startup` SKIP with `--sandbox hf-sandbox`; validate a served Space with
`--space` instead.

The 16 implemented runtime checks cover startup, rewards, observation schemas,
state continuity, network policy, host containment, resource bounds, episode isolation,
oracle containment, seed forwarding, determinism, emitted trajectory records, tool
and task discovery, rubric introspection and reward attribution. Applicable checks
with unsupported capabilities or incomplete evidence appear as `SKIP`; they make
the result `WARN`, which exits zero and does not mean Level 2 is complete.
`FAIL` exits 1, unsupported package formats exit 2, and internal or policy errors
exit 3. `--level semantic`
runs the semantic checks for Harbor tasks. Served packages have no oracle execution
yet, so their semantic checks are named SKIPs.
`--skip-build` runs declaration checks and skips runtime execution entirely.
The Docker provider supports CPU workloads in `public` and `no-network` modes;
`allowlist` and GPU requirements skip runtime before building. A `no-network`
subject has no network interface: the validator reaches it through a helper that
shares only its network namespace, built from a digest-pinned validator image
pulled on first use. `runtime.network_policy` probes a validator-owned sink from
the subject's namespace and from a control container; a probe counts only when
the control reaches the sink, and a denial that is not proven is a `SKIP`.
`runtime.host_containment` inspects the launch from outside the subject and runs a
probe from the validator image that joins only the subject's PID namespace, as the
subject's own user. It fails when the subject runs privileged, shares a host
namespace, holds capabilities, runs as root or has an unexpected mount. The probe
also reads every writable filesystem from the subject's mount table. The image root
must be read-only. Besides `/tmp` and `/dev/shm`, only Docker's `/proc`, `/dev/pts`
and `/dev/mqueue` pseudo-filesystems and its root-owned `/dev` may be writable, and
the subject must not be able to create files or write existing regular files on
that device filesystem. The helper inventories at most 1024 entries to depth 32,
without following symlinks or entering separate mounts such as `/dev/shm`.
An unreadable or exhausted inventory, a probe that cannot write an allotted root,
or a mount table that cannot be read in full leaves containment unproven and
reports `SKIP`.
`runtime.resource_bounds` reads the subject's cgroup v2 limits and writable-root
sizes and compares them with the declared `cpu`, `memory_mb` and `disk_mb`; swap
must be disabled and the process count bounded. Storage accounting uses the complete
mount inventory and the device-storage inventory above, charges any usable device
filesystem once across its aliases, and cannot pass with incomplete storage
evidence or an unbounded writable filesystem. Each bounded memory, process,
disk and Docker exec-timeout probe runs in its own disposable subject. A fresh
start marker and trusted cgroup events or filesystem measurements must corroborate
enforcement. Subject text cannot establish excess usage. Exit 137 alone proves no limit. Unreadable cgroup files preserve
independently readable limits and leave the affected qualification unproven.
Unavailable evidence reports `SKIP`; failed disposable teardown reports `ERROR`
and outer cleanup retries every run-owned subject and helper. The exec-timeout
probe requires a trusted helper to observe a live parent and child with the
fresh stimulus nonce before timeout, followed by verified removal. A printed
descendant marker alone leaves qualification unproven. It does not qualify
enforcement of the declared `episode_timeout_s` for a served episode.
`runtime.episode_isolation` runs the replay plan twice on one session, as episode A
and then a reset into episode B, and once more on a new session. Episode B must
match the fresh episode exchange for exchange. Messages are compared as decoded
JSON values, so key order and formatting do not count. If episode A or the fresh
episode differs from the original measured episode, the subject is not
deterministic under fixed inputs and the check reports `SKIP`. Each episode gets
its own `episode_timeout_s` deadline and trace size bound. When an episode stops
early before any response differs, the experiment is incomplete and the check
also reports `SKIP`.
`runtime.oracle_containment` applies when `capabilities.oracle` is declared. It
compares complete declared oracle files with collected reset, step and state
responses, the complete `/schema` response, and unauthenticated HTTP probes of
each inventoried file's package-relative URL. It decodes JSON string values and
compares structured JSON independent of formatting. Isolated source lines are
ignored because ordinary boilerplate does not establish disclosure. A matching
complete artifact fails regardless of HTTP status or incomplete evidence elsewhere.

Oracle inventory reads at most 64 files and 1 MiB. HTTP probes share a 10 second
wall deadline and 1 MiB body budget, follow at most three same-origin redirects,
and reject compressed responses before decoding. Missing or failed probes,
unresolved redirects, incomplete transcripts or schemas, unreadable inventory
entries, unsupported structured JSON comparisons, files too short to trace
reliably, and omitted files or bytes report
`SKIP` unless a disclosure was found. Coverage counts are recorded for each limit.
A pass establishes only that complete artifacts did not match
these measured responses. It does not test arbitrary MCP tool reads, partial or
transformed disclosures, or writes to reward artifacts; it does not certify the
full agent boundary.
`--local` explicitly selects this default package mode and rejects a remote URL.
Explicitly unsupported session telemetry yields a `SKIP` naming `session_telemetry`;
authorization refusals and malformed telemetry remain failures. Explicitly unsupported
`tools/list` similarly skips with `tool_discovery` named. Internal discovery errors
and malformed responses remain failures, even for an empty declared tool set. Seed control
reports observed forwarding to reset, separately from empirical determinism.
The declared `episode_timeout_s` bounds each episode; a reset or judged step may use
its remaining budget. Original and replay collection share a 300-second deadline
and a 32 MiB evidence budget. `llm_judged` requests 20 identical-input episodes,
including the original, plus the different-seed episode. Validation repeats the
actions and any external service calls they make, so budget for the additional
runtime and cost. Missing replay samples cannot pass determinism. Primary episode
collection failures appear under `runtime.startup`, with dependent contract checks
skipped and the completed trace retained.
Validator JSON numeric-range and depth limits report named capability `SKIP`s,
preserving the received evidence when it can be retained without validation credentials.

`--space` validates a Hugging Face Space on HF Jobs instead of a local directory:

```bash
openenv validate --space openenv/echo_env --level runtime
openenv validate --space my-org/my-env --level runtime --flavor t4-small
```

The validator resolves `--revision` (a branch, tag or commit hash; default `main`)
to one commit and downloads and hashes the Space files at that commit. Hub download
bookkeeping is excluded. The report target names the resolved source commit.

Runtime validation currently reports a named `immutable_image` SKIP without
creating an HF Job. The supported Hub API exposes a mutable Space image name and
current Space runtime metadata, but does not establish an immutable image digest
bound to the downloaded source commit. Matching repository and running-build
revisions alone cannot prove which image a Job would pull. A different repository
or reported runtime revision reports `runtime.startup` as `ERROR`. This provider
also refuses building, sleeping and paused Spaces: it requires reported stage
`RUNNING`, without treating that stage as proof of image provenance. Absent runtime
stage or revision metadata yields a named `space_runtime_metadata` SKIP; explicit
contradictory metadata still yields ERROR. Use
`--level static` to validate historical source commits. Reports retain the source
commit and source digest without claiming a verified runtime image or execution.

`--flavor` selects an HF Jobs hardware flavor (default `cpu-basic`; see
`hf jobs hardware`). A GPU flavor satisfies the declared GPU prerequisite, but
image verification still blocks execution. The provider only supports `public`
network mode; network policy, host containment and resource bounds require
capabilities it cannot inspect from outside the HF Job. CPU/GPU runtime and
WebSocket qualification, cleanup and the reported Sandbox startup issue remain
unverified on live HF infrastructure.

The serving implementation supports ordinary Dockerfile backslash continuations
and final-stage commands, including inheritance from a named earlier stage.
Heredocs are unsupported: a `<<` marker in any non-comment instruction, including
a literal marker, reports a named `dockerfile_heredoc` SKIP before command parsing.
`ENTRYPOINT`, custom `SHELL` commands, custom escape directives and commands
inherited from an external base image are unsupported. The server log retains at
most the first 64 KiB of combined stdout and stderr; further output is drained
and discarded. These serving paths have local regression coverage and do not
enable unverified image launches.

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
cases, follow [the runtime lab](https://github.com/openenv/OpenEnv/blob/main/tests/validation_runtime/README.md).
The implementation and remaining check inventory are tracked in
[the Level 2 umbrella](https://github.com/openenv/OpenEnv/issues/1177).

#### openenv.cli.commands.validate[[openenv.cli.commands.validate]]

```python
openenv.cli.commands.validate(target: Annotated = None, url: Annotated = None, level: Annotated = 'semantic', skip_build: Annotated = False, local: Annotated = False, policy_version: Annotated = None, json_output: Annotated = False, output: Annotated = None, timeout: Annotated = 5.0, space: Annotated = None, revision: Annotated = None, flavor: Annotated = 'cpu-basic', sandbox: Annotated = 'docker', namespace: Annotated = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/validate.py#L148)

Validate a local package or a running OpenEnv server.

Local validation detects the package format by its well-known file (the
formats this build can parse: `openenv.yaml` or Harbor `task.toml`), parses it into the
normalized manifest, runs the applicable graders up to the requested level,
applies the severity policy, and emits a report.

Native Harbor tasks require the optional Harbor dependency and an explicit
`validation/openenv.json` contract. Fresh trials grade the oracle, the
do-nothing floor and any declared canaries under the semantic check ids. They
run on the local Docker daemon, or with `--sandbox hf-sandbox` as one Hugging
Face Job per trial from the task's digest-pinned image. Checks without an
implementation are named SKIPs, so the report is WARN at best.

Exit codes: 0 pass/warn · 1 fail · 2 unrecognized/unsupported package · 3
internal or policy error.

Examples:

```bash
# Validate the current directory up to the semantic level
openenv validate

# Fast inner loop: static checks only, no image build
openenv validate envs/echo_env --level static --skip-build

# Machine-readable report
openenv validate envs/echo_env --json

# Probe a running server (legacy runtime probe)
openenv validate --url http://localhost:8000

# Validate a Hub Space on HF Jobs GPU hardware
openenv validate --space openenv/echo_env --level runtime --flavor t4-small

# Run a Harbor task's trials as HF Jobs instead of local Docker
openenv validate path/to/task --sandbox hf-sandbox --flavor cpu-upgrade
```

## `openenv push`[[openenv.cli.commands.push]]

#### openenv.cli.commands.push[[openenv.cli.commands.push]]

```python
openenv.cli.commands.push(directory: Annotated[str | None, typer.Argument(help='Directory containing the OpenEnv environment (default: current directory)')] = None, repo_id: Annotated[str | None, typer.Option('--repo-id', '-r', help="Repository ID as 'repo_name' or 'namespace/repo_name'. Defaults to 'username/env-name' from openenv.yaml.")] = None, base_image: Annotated[str | None, typer.Option('--base-image', '-b', help='Base Docker image to use (overrides Dockerfile FROM)')] = None, interface: Annotated[bool, typer.Option('--interface', help='Enable web interface (default: True if no registry specified)')] = None, no_interface: Annotated[bool, typer.Option('--no-interface', help='Disable web interface')] = False, registry: Annotated[str | None, typer.Option('--registry', help='Custom registry URL (e.g., docker.io/username). Disables web interface by default.')] = None, private: Annotated[bool, typer.Option('--private', help='Deploy the space as private')] = False, create_pr: Annotated[bool, typer.Option('--create-pr', help='Create a Pull Request instead of pushing to the default branch')] = False, exclude: Annotated[str | None, typer.Option('--exclude', help='Optional additional ignore file with newline-separated glob patterns to exclude from Hugging Face uploads')] = None, delete: Annotated[list[str] | None, typer.Option('--delete', help="Glob pattern of remote Space files to delete in the same commit, as in `hf upload --delete` (repeatable). Use '*' to remove every file not uploaded again.")] = None, hardware: Annotated[str | None, typer.Option('--hardware', '-H', help='Request hardware for Hugging Face Space (e.g. t4-medium, cpu-basic). See HF docs for options.')] = None, count: Annotated[int, typer.Option('--count', '-n', help='Number of Space instances to deploy. Each gets a numeric suffix (e.g. env-1, env-2).', min=1)] = 1, env_vars: Annotated[list[str] | None, typer.Option('--env-var', '-e', help='Public Space variable as KEY=VALUE (repeatable). Overrides matching keys from openenv.yaml variables:.')] = None, secrets: Annotated[list[str] | None, typer.Option('--secret', help='Private Space secret as KEY=VALUE (repeatable). Value is never logged.')] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/push.py#L511)

Push an OpenEnv environment to Hugging Face Spaces or a custom Docker registry.

This command:
1. Validates that the directory is an OpenEnv environment (openenv.yaml present)
2. Builds and pushes to Hugging Face Spaces or custom Docker registry
3. Optionally enables web interface for deployment

The web interface is enabled by default when pushing to HuggingFace Spaces,
but disabled by default when pushing to a custom Docker registry.

Examples:

```bash
# Push to HuggingFace Spaces from current directory (web interface enabled)
$ cd my_env
$ openenv push

# Push to HuggingFace repo and open a Pull Request
$ openenv push my-org/my-env --create-pr
$ openenv push --repo-id my-org/my-env --create-pr

# Push to HuggingFace without web interface
$ openenv push --no-interface

# Push to Docker Hub
$ openenv push --registry docker.io/myuser

# Push to GitHub Container Registry
$ openenv push --registry ghcr.io/myorg

# Push to custom registry with web interface
$ openenv push --registry myregistry.io/path1/path2 --interface

# Push to specific HuggingFace repo
$ openenv push --repo-id my-org/my-env

# Push privately with custom base image
$ openenv push --private --base-image ghcr.io/huggingface/openenv-base:latest

# Push with GPU hardware
$ openenv push --hardware t4-medium

# Set a public Space variable (overrides openenv.yaml variables:)
$ openenv push -e OPENSPIEL_GAME=tic_tac_toe -e MAX_STEPS=100

# Set a private Space secret (value never logged)
$ openenv push --secret OPENAI_API_KEY=sk-...
```

## `openenv serve`[[openenv.cli.commands.serve]]

Local serving is not implemented in the CLI yet. This command exits non-zero
and prints alternative ways to run an environment server.

#### openenv.cli.commands.serve[[openenv.cli.commands.serve]]

```python
openenv.cli.commands.serve(env_path: Annotated[str | None, typer.Argument(help='Path to the environment directory (default: current directory)')] = None, port: Annotated[int | None, typer.Option('--port', '-p', help='Port to bind (default: ``port`` in openenv.yaml, else 8000)')] = None, host: Annotated[str, typer.Option('--host', help='Host interface to bind (local development only)')] = '127.0.0.1', reload: Annotated[bool, typer.Option('--reload', help='Enable autoreload (development)')] = False)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/serve.py#L18)

Run the environment FastAPI app on the host for local development.

Uses `openenv.yaml` fields `app` (e.g. `server.app:app`), `port`, and
`runtime` (must be `fastapi`). Matches `uv run --project . server` layout:
the environment directory is the working directory. Package-qualified app
paths also support the environment package and the repo's `envs` package.

For production or training, use Docker (`openenv build`) — this command runs
on the host for local development only.

## `openenv fork`[[openenv.cli.commands.fork]]

#### openenv.cli.commands.fork[[openenv.cli.commands.fork]]

```python
openenv.cli.commands.fork(source_space: Annotated[str, typer.Argument(help="Source Space ID in format 'owner/space-name' (e.g. org/my-openenv-space)")], repo_id: Annotated[str | None, typer.Option('--repo-id', '-r', help='Target repo ID for the fork (default: created under your account with same name)')] = None, private: Annotated[bool, typer.Option('--private', help='Create the forked Space as private')] = False, set_env: Annotated[list[str], typer.Option('--set-env', '-e', help='Set Space variable (public). Can be repeated. Format: KEY=VALUE')] = [], set_secret: Annotated[list[str], typer.Option('--set-secret', '--secret', '-s', help='Set Space secret. Can be repeated. Format: KEY=VALUE')] = [], hardware: Annotated[str | None, typer.Option('--hardware', '-H', help='Request hardware (e.g. t4-medium, cpu-basic). See Hub docs for options.')] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/fork.py#L58)

Fork (duplicate) a Hugging Face Space to your account using the Hub API.

Uses the Hugging Face duplicate_space API. You can set environment variables
and secrets, and request hardware/storage/sleep time at creation time.

Examples:

```bash
$ openenv fork owner/source-space
$ openenv fork owner/source-space --private
$ openenv fork owner/source-space --repo-id myuser/my-fork
$ openenv fork owner/source-space --set-env MODEL_ID=user/model --set-secret HF_TOKEN=hf_xxx
$ openenv fork owner/source-space --hardware t4-medium
```

## `openenv skills`[[openenv.cli.commands.skills.skills_add]]

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

#### openenv.cli.commands.skills.skills_add[[openenv.cli.commands.skills.skills_add]]

```python
openenv.cli.commands.skills.skills_add(claude: Annotated[bool, typer.Option('--claude', help='Install for Claude.')] = False, codex: Annotated[bool, typer.Option('--codex', help='Install for Codex.')] = False, cursor: Annotated[bool, typer.Option('--cursor', help='Install for Cursor.')] = False, opencode: Annotated[bool, typer.Option('--opencode', help='Install for OpenCode.')] = False, global_: Annotated[bool, typer.Option('--global', '-g', help='Install globally (user-level) instead of in the current project directory.')] = False, dest: Annotated[Path | None, typer.Option(help='Install into a custom destination (skills directory path).')] = None, force: Annotated[bool, typer.Option('--force', help='Overwrite existing skills in the destination.')] = False)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/skills.py#L143)

Install OpenEnv CLI skill for AI assistants.

#### openenv.cli.commands.skills.skills_preview[[openenv.cli.commands.skills.skills_preview]]

```python
openenv.cli.commands.skills.skills_preview()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/skills.py#L137)

Print generated SKILL.md content.

## `openenv collect`[[openenv.cli.commands.collect]]

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

#### openenv.cli.commands.collect[[openenv.cli.commands.collect]]

```python
openenv.cli.commands.collect(env: Annotated[str, typer.Argument(help="Env spec in 'family:variant' form (e.g. openspiel:tic_tac_toe).")], base_url: Annotated[str, typer.Option('--base-url', help='Env server URL (local Docker or Hugging Face Space).')], output_dir: Annotated[Path, typer.Option('--output-dir', '-o', help='Directory to write results.jsonl + metadata.json.')], num_episodes: Annotated[int, typer.Option('--num-episodes', '-n', help='Number of episodes to collect.')] = 10, max_turns: Annotated[int, typer.Option('--max-turns', help='Max tool/model turns per episode.')] = 9, episode_id_prefix: Annotated[str, typer.Option('--episode-id-prefix', help='Prefix for serialized episode ids.')] = 'ep', resume: Annotated[bool, typer.Option('--resume/--no-resume', help='Skip episodes already present in results.jsonl.')] = True, provider: Annotated[str, typer.Option('--provider', help='Teacher provider: scripted | openai | anthropic.')] = 'scripted', model: Annotated[str | None, typer.Option('--model', help='Model id (required when provider != scripted).')] = None, llm_endpoint: Annotated[str | None, typer.Option('--llm-endpoint', help='Base URL of a self-hosted OpenAI-compatible server (vLLM/TGI/Ollama), e.g. http://localhost:8000. /v1 is appended when the URL has no path; a URL with a path is used as-is.')] = None, llm_port: Annotated[int | None, typer.Option('--llm-port', help='Port appended to --llm-endpoint when the URL does not include one. No default: earlier releases assumed 8000.')] = None, temperature: Annotated[float, typer.Option('--temperature', help='Sampling temperature.')] = 0.2, max_tokens: Annotated[int, typer.Option('--max-tokens', help='Max completion tokens.')] = 200, keep_losses: Annotated[bool, typer.Option('--keep-losses', help='Keep losing rollouts (default: filter rollouts with reward < 0).')] = False, push_to_hub: Annotated[str | None, typer.Option('--push-to-hub', '-H', help="Destination dataset repo id ('user/name'). Uploads after collect.")] = None, private: Annotated[bool, typer.Option('--private', help='Create the Hub dataset repo as private.')] = False, commit_message: Annotated[str | None, typer.Option('--commit-message', help='Commit message for the Hub upload.')] = None, dataset_config: Annotated[str | None, typer.Option('--dataset-config', help='JSON string of dataset config for envs that support it (e.g. reasoning_gym). Example: \'{"min_terms": 2, "max_terms": 3}\'')] = None, system_prompt: Annotated[str | None, typer.Option('--system-prompt', help='Custom system prompt for the teacher model.')] = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/collect.py#L228)

Collect rollouts from a deployed OpenEnv environment.

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
    --repository-uri https://github.com/openenv/OpenEnv.git \
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

## Entry point[[openenv.cli.__main__.main]]

#### openenv.cli.__main__.main[[openenv.cli.__main__.main]]

```python
openenv.cli.__main__.main()
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/__main__.py#L76)

Main entry point for the CLI.

## CLI helpers[[openenv.cli._cli_utils.validate_env_structure]]

#### openenv.cli._cli_utils.validate_env_structure[[openenv.cli._cli_utils.validate_env_structure]]

```python
openenv.cli._cli_utils.validate_env_structure(env_dir: Path, strict: bool = False)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/_cli_utils.py#L30)

**Parameters:**

env_dir (`Path`) : Path to the environment directory.

strict (`bool`, *optional*, defaults to `False`) : If `True`, enforce all optional requirements.

**Returns:**

`list` of validation warnings (empty if all checks pass).

**Raises:** ``FileNotFoundError``

- ``FileNotFoundError`` -- If required files are missing.

Validate that the directory follows OpenEnv environment structure.

## Validation utilities[[openenv.cli._validation.validate_running_environment]]

#### openenv.cli._validation.validate_running_environment[[openenv.cli._validation.validate_running_environment]]

```python
openenv.cli._validation.validate_running_environment(base_url: str, timeout_s: float = 5.0)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/_validation.py#L99)

Validate a running OpenEnv server against runtime API standards.

The returned JSON report contains an overall pass/fail result and
per-criterion outcomes that can be consumed in CI.

#### openenv.cli._validation.validate_multi_mode_deployment[[openenv.cli._validation.validate_multi_mode_deployment]]

```python
openenv.cli._validation.validate_multi_mode_deployment(env_path: Path)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/_validation.py#L505)

**Returns:**

`tuple` of `(is_valid, issues)` where `is_valid` is a `bool` and `issues` is a
`list` of issue strings found during validation.

Validate that an environment is ready for multi-mode deployment.

Checks:
1. pyproject.toml exists
2. uv.lock exists
3. pyproject.toml has [project.scripts] with server entry point
4. server/app.py has a main() function
5. Required dependencies are present

#### openenv.cli._validation.get_deployment_modes[[openenv.cli._validation.get_deployment_modes]]

```python
openenv.cli._validation.get_deployment_modes(env_path: Path)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/_validation.py#L581)

**Returns:**

`dict` mapping deployment mode names to whether they are supported.

Check which deployment modes are supported by the environment.

#### openenv.cli._validation.format_validation_report[[openenv.cli._validation.format_validation_report]]

```python
openenv.cli._validation.format_validation_report(env_name: str, is_valid: bool, issues: list[str])
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/_validation.py#L610)

**Returns:** `str`

formatted validation report.

Format a validation report for display.

#### openenv.cli._validation.build_local_validation_json_report[[openenv.cli._validation.build_local_validation_json_report]]

```python
openenv.cli._validation.build_local_validation_json_report(env_name: str, env_path: Path, is_valid: bool, issues: list[str], deployment_modes: dict[str, bool] | None = None)
```

[Source](https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/_validation.py#L628)

Build a JSON report for local environment validation.

