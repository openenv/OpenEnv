import"../chunks/DsnmJJEf.js";import{i as Re,h as xe,C as Ve,H as s,D as a,a as n,E as Ce,s as Le}from"../chunks/Dz7d4_v9.js";import{p as Fe,o as Ye,s as e,f as m,a as p,b as He,d as o,c as A,r as t,n as l}from"../chunks/BSuxAqoA.js";import{E as h}from"../chunks/CxooqZEj.js";const Xe='{"title":"CLI","local":"cli","sections":[{"title":"openenv init","local":"openenv.cli.commands.init","sections":[],"depth":2},{"title":"openenv import","local":"openenv.cli.commands.import_env","sections":[],"depth":2},{"title":"openenv build","local":"openenv.cli.commands.build","sections":[],"depth":2},{"title":"openenv validate","local":"openenv.cli.commands.validate","sections":[],"depth":2},{"title":"openenv push","local":"openenv.cli.commands.push","sections":[],"depth":2},{"title":"openenv serve","local":"openenv.cli.commands.serve","sections":[],"depth":2},{"title":"openenv fork","local":"openenv.cli.commands.fork","sections":[],"depth":2},{"title":"openenv skills","local":"openenv.cli.commands.skills.skills_add","sections":[],"depth":2},{"title":"openenv collect","local":"openenv.cli.commands.collect","sections":[],"depth":2},{"title":"openenv harbor","local":"openenv-harbor","sections":[],"depth":2},{"title":"openenv catalog and openenv discover","local":"openenv-catalog-and-openenv-discover","sections":[],"depth":2}],"depth":1}';var Oe=A('<meta name="hf:doc:metadata"/>'),u=A("<p>Examples:</p> <!>",1),Qe=A(`<p></p> <!> <!> <p>The <code>openenv</code> CLI provides a set of commands for building, validating, and pushing environments to Hugging Face Spaces or a custom Docker registry. For a walkthrough of building an environment, see <a href="../guides/first-environment">Your First Environment</a>.</p> <!> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Initialize a new OpenEnv environment.</p> <p>Creates a new directory with the environment name and generates all necessary
files based on the OpenEnv template structure.</p> <!></div> <!> <p>Import a supported third-party source environment into a generated OpenEnv
wrapper package. The command detects the source format from the directory
contents, so ORS/OpenReward, Prime Intellect Verifiers and AgentEnv sources do
not require <code>--type</code> in the common case.</p> <p>The generated wrapper vendors the source tree into the package and includes
vendored files as package data, so non-secret fixture/data files are available to
the environment server at runtime. The importer carries portable dependencies
from source <code>pyproject.toml</code> and <code>requirements.txt</code> files into the generated
environment, skips VCS/cache/build directories and common secret file patterns
such as <code>.env</code>, <code>secrets.yaml</code>, and private key files, and excludes compiled
binary artifacts; review the generated <code>vendor/</code> directory before publishing a
wrapper.</p> <!> <p>An <a href="https://github.com/scaleapi/agentenv-framework" rel="nofollow">AgentEnv</a> bundle is a folder
whose <code>tasks/</code> holds task files. When a source holds several bundles, pick one
with <code>--env-class</code> and its folder name; only that folder is vendored. The
generated server runs it with <code>build_app</code> from <code>agentenv-framework</code>, which by
default deploys each episode using local Docker. The server needs a Docker CLI
and a running daemon accessible for image builds and episode containers. The
generated <code>server/Dockerfile</code> does not supply that runtime, so <code>openenv build</code> and <code>openenv push</code> alone do not produce a working deployment. The generated
README describes local startup and these prerequisites. The generated client is
an <code>MCPToolClient</code>, since the served environment takes MCP actions. See <a href="https://www.agentenvframework.com/docs/openenv" rel="nofollow">AgentEnv’s OpenEnv page</a> for
what it serves and what it needs.</p> <!> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Deterministically import a third-party environment into OpenEnv.</p></div> <!> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Build Docker images for OpenEnv environments.</p> <p>This command builds Docker images using the environment’s pyproject.toml
and uv for dependency management. Run from the environment root directory.</p> <!></div> <!> <p>Run static declaration checks without Docker, or build and probe a package locally:</p> <!> <p>For served packages, runtime validation requires Docker and a <code>validation.execution</code> declaration in <code>openenv.yaml</code>. The declaration points to a bounded JSON replay plan containing
one reset and a sequence of actions. The validator builds an immutable image,
collects the original episode, repeats the plan in a fresh WebSocket session and
with a different seed, then replays it in a fresh container when the provider
supports that capability. Docker validation uses two containers concurrently
during the fresh-container replay. Both use the same immutable image and are
removed even if a check fails.</p> <p>Harbor <code>task.toml</code> directories are also recognized. Validation requires Python 3.12 or newer
and <code>harbor==0.25.0</code> installed alongside <code>openenv[harbor]</code>:</p> <!> <p>A missing or mismatched Harbor installation exits 3 and names <code>harbor==0.25.0</code>.</p> <p>Add a <code>validation/openenv.json</code> contract with <code>profile: "harbor-semantic-v1"</code>, <code>reward_key</code>, <code>reward</code>, <code>resources</code> and <code>types</code>. The reward range, oracle tolerance
and floor margin must be explicit. Oracle tolerance and floor margin are fractions
of the declared reward range. Numeric strings in numeric fields, oversized integers
and excessive JSON nesting fail <code>static.manifest</code>. Reward and resource numbers must
retain their exact value when stored as floats. <code>resources.gpu_types</code> remains a list
of strings. This is a minimal complete contract:</p> <!> <p>The <code>harbor-semantic-v1</code> binding runs
fresh Harbor trials, in local Docker unless <code>--sandbox hf-sandbox</code> moves them to
HF Jobs: a do-nothing agent, the oracle (<code>solution/solve.sh</code>),
an oracle replay and any canaries declared in the contract. A canary is a known-wrong <code>solve.sh</code> replacement. The results report under <code>runtime.startup</code>, <code>runtime.reward_well_formed</code>, <code>runtime.episode_determinism</code>, <code>semantic.oracle_max</code>, <code>semantic.floor_gap</code> and <code>semantic.canary_floor</code>, and <code>harbor-evidence.json</code> keeps
the raw trial evidence in the report bundle. Other checks are named SKIPs, so a
passing task still reports WARN. This validates task packages, not the outer <code>harbor_env</code> server.</p> <p>Supported tasks use Linux, a single step, a shared verifier and public networking.
The image comes from one Dockerfile or from a digest-pinned <code>environment.docker_image</code>, such as <code>org/image@sha256:&lt;64 hex digits&gt;</code>. Harbor
prefers a declared image over a Dockerfile, so the do-nothing trial pulls the pin
and <code>runtime.startup</code> records it as <code>source_image</code>. The validator leaves the pulled
image in the local Docker cache. Unsupported shapes include multi-step or non-Linux
tasks, an <code>environment.docker_image</code> without a digest, tasks with neither a
Dockerfile nor a pinned image and task-authored Compose files. GPU/TPU or external
MCP requirements, credentials, non-public or differing phase networks, task
environment variables, healthchecks, skills and verifier collection are also
unsupported. Separate verifier environments and artifact extensions run only on <code>hf-sandbox</code>. A served-package provider, including <code>--space</code>, is unsupported. Each case
gives <code>runtime.startup</code> SKIP with a named reason and runs no trials. With valid
static declarations and an oracle, the verdict is WARN and the exit code is 0.
A missing <code>solution/solve.sh</code> still FAILs <code>semantic.oracle_max</code> at the semantic level,
including with <code>--skip-build</code> or <code>--space</code>.</p> <p>The initial image build and environment start share a separate 600-second validator
budget. Exceeding it gives <code>runtime.startup</code> ERROR. Later environment starts use the
episode budget. There is no shared outer trial deadline. Harbor bounds setup, agent
and verifier separately with the declared <code>resources.episode_timeout_s</code>. A timeout
in one of these phases is a package FAIL that names the phase.</p> <p>The do-nothing trial builds from the task’s Dockerfile, or pulls the digest-pinned
image when the task declares one. The validator keeps <code>openenv-validation:&lt;run id&gt;</code> only when the run has no FAIL or ERROR and was not interrupted. It records the tag in <code>runtime.startup.measured</code> as <code>retained_image_tag</code>. Otherwise it removes the tag. The run-owned Compose build tag
is removed on every run. Tag removal verifies image identity and uses <code>--no-prune</code>.
A removal failure marks cleanup incomplete and gives a startup ERROR. During an
abort, cleanup preserves the original exception and records the removal failure.</p> <p><code>--sandbox hf-sandbox</code> runs the same trials on Hugging Face Jobs instead, one job
per trial, through Harbor’s <code>hf-sandbox</code> environment:</p> <!> <p>A sandbox cannot build a Dockerfile, so the task must pin a pullable image as <code>environment.docker_image = "name@sha256:&lt;digest&gt;"</code>. Build and push the image
first, then pin its digest. The image needs <code>/bin/bash</code>, which Harbor uses to run
commands, and either <code>sha256sum</code> or <code>openssl</code>, which the sandbox server uses to
verify its own download. With nothing
to build, every trial starts at once. <code>--flavor</code> and <code>--namespace</code> mean what they
mean for <code>hf jobs run</code>: the flavor sets the hardware, and the namespace is the user
or organization that owns and pays for the jobs, by default the token’s user. The
flavor must provide the CPU, memory and disk the contract declares, or <code>runtime.startup</code> reports ERROR before any job starts.</p> <p>HF Sandbox cannot turn the network off or switch users. Tasks that declare <code>no-network</code> for any phase, or set an agent or verifier user, are unsupported,
and <code>harbor-evidence.json</code> records that every phase ran with public networking
as the image’s default user.</p> <p>A task with <code>verifier.environment_mode = "separate"</code> runs its verifier in a second
job, as Harbor does on other backends. That job starts from the task’s pinned
image. Harbor copies each <code>[[artifacts]]</code> path from the agent’s job into it and
uploads <code>tests/</code> at grading time, so test files never need to be in the image.
A separate verifier with its own image, either <code>verifier.environment.docker_image</code> or a <code>tests/Dockerfile</code>, is unsupported.</p> <p>Each job, including a separate verifier’s, carries <code>openenv-validation-run</code> and <code>openenv-validation-role</code> labels. At <code>AGENT_START</code> the validator reads the job back and requires the pinned image, the
requested flavor and a <code>RUNNING</code> stage. After each trial it cancels any labelled
job that is still running and waits up to 60 seconds for all of them to stop.
Otherwise cleanup is unverified and <code>runtime.startup</code> reports ERROR. <code>runtime.startup.measured</code> records the flavor, namespace and job IDs, and <code>harbor-evidence.json</code> states that the image is the digest requested from HF Jobs,
not one measured inside the sandbox. Check IDs, graders and thresholds are the
same as on Docker, and no image is retained. A served package gives a named <code>runtime.startup</code> SKIP with <code>--sandbox hf-sandbox</code>; validate a served Space with <code>--space</code> instead.</p> <p>The 16 implemented runtime checks cover startup, rewards, observation schemas,
state continuity, network policy, host containment, resource bounds, episode isolation,
oracle containment, seed forwarding, determinism, emitted trajectory records, tool
and task discovery, rubric introspection and reward attribution. Applicable checks
with unsupported capabilities or incomplete evidence appear as <code>SKIP</code>; they make
the result <code>WARN</code>, which exits zero and does not mean Level 2 is complete. <code>FAIL</code> exits 1, unsupported package formats exit 2, and internal or policy errors
exit 3. <code>--level semantic</code> runs the semantic checks for Harbor tasks. Served packages have no oracle execution
yet, so their semantic checks are named SKIPs. <code>--skip-build</code> runs declaration checks and skips runtime execution entirely.
The Docker provider supports CPU workloads in <code>public</code> and <code>no-network</code> modes; <code>allowlist</code> and GPU requirements skip runtime before building. A <code>no-network</code> subject has no network interface: the validator reaches it through a helper that
shares only its network namespace, built from a digest-pinned validator image
pulled on first use. <code>runtime.network_policy</code> probes a validator-owned sink from
the subject’s namespace and from a control container; a probe counts only when
the control reaches the sink, and a denial that is not proven is a <code>SKIP</code>. <code>runtime.host_containment</code> inspects the launch from outside the subject and runs a
probe from the validator image that joins only the subject’s PID namespace, as the
subject’s own user. It fails when the subject runs privileged, shares a host
namespace, holds capabilities, runs as root or has an unexpected mount. The probe
also reads every writable filesystem from the subject’s mount table. The image root
must be read-only. Besides <code>/tmp</code> and <code>/dev/shm</code>, only Docker’s <code>/proc</code>, <code>/dev/pts</code> and <code>/dev/mqueue</code> pseudo-filesystems and its root-owned <code>/dev</code> may be writable, and
the subject must not be able to create files or write existing regular files on
that device filesystem. The helper inventories at most 1024 entries to depth 32,
without following symlinks or entering separate mounts such as <code>/dev/shm</code>.
An unreadable or exhausted inventory, a probe that cannot write an allotted root,
or a mount table that cannot be read in full leaves containment unproven and
reports <code>SKIP</code>. <code>runtime.resource_bounds</code> reads the subject’s cgroup v2 limits and writable-root
sizes and compares them with the declared <code>cpu</code>, <code>memory_mb</code> and <code>disk_mb</code>; swap
must be disabled and the process count bounded. Storage accounting uses the complete
mount inventory and the device-storage inventory above, charges any usable device
filesystem once across its aliases, and cannot pass with incomplete storage
evidence or an unbounded writable filesystem. Each bounded memory, process,
disk and Docker exec-timeout probe runs in its own disposable subject. A fresh
start marker and trusted cgroup events or filesystem measurements must corroborate
enforcement. Subject text cannot establish excess usage. Exit 137 alone proves no limit. Unreadable cgroup files preserve
independently readable limits and leave the affected qualification unproven.
Unavailable evidence reports <code>SKIP</code>; failed disposable teardown reports <code>ERROR</code> and outer cleanup retries every run-owned subject and helper. The exec-timeout
probe requires a trusted helper to observe a live parent and child with the
fresh stimulus nonce before timeout, followed by verified removal. A printed
descendant marker alone leaves qualification unproven. It does not qualify
enforcement of the declared <code>episode_timeout_s</code> for a served episode. <code>runtime.episode_isolation</code> runs the replay plan twice on one session, as episode A
and then a reset into episode B, and once more on a new session. Episode B must
match the fresh episode exchange for exchange. Messages are compared as decoded
JSON values, so key order and formatting do not count. If episode A or the fresh
episode differs from the original measured episode, the subject is not
deterministic under fixed inputs and the check reports <code>SKIP</code>. Each episode gets
its own <code>episode_timeout_s</code> deadline and trace size bound. When an episode stops
early before any response differs, the experiment is incomplete and the check
also reports <code>SKIP</code>. <code>runtime.oracle_containment</code> applies when <code>capabilities.oracle</code> is declared. It
compares complete declared oracle files with collected reset, step and state
responses, the complete <code>/schema</code> response, and unauthenticated HTTP probes of
each inventoried file’s package-relative URL. It decodes JSON string values and
compares structured JSON independent of formatting. Isolated source lines are
ignored because ordinary boilerplate does not establish disclosure. A matching
complete artifact fails regardless of HTTP status or incomplete evidence elsewhere.</p> <p>Oracle inventory reads at most 64 files and 1 MiB. HTTP probes share a 10 second
wall deadline and 1 MiB body budget, follow at most three same-origin redirects,
and reject compressed responses before decoding. Missing or failed probes,
unresolved redirects, incomplete transcripts or schemas, unreadable inventory
entries, unsupported structured JSON comparisons, files too short to trace
reliably, and omitted files or bytes report <code>SKIP</code> unless a disclosure was found. Coverage counts are recorded for each limit.
A pass establishes only that complete artifacts did not match
these measured responses. It does not test arbitrary MCP tool reads, partial or
transformed disclosures, or writes to reward artifacts; it does not certify the
full agent boundary. <code>--local</code> explicitly selects this default package mode and rejects a remote URL.
Explicitly unsupported session telemetry yields a <code>SKIP</code> naming <code>session_telemetry</code>;
authorization refusals and malformed telemetry remain failures. Explicitly unsupported <code>tools/list</code> similarly skips with <code>tool_discovery</code> named. Internal discovery errors
and malformed responses remain failures, even for an empty declared tool set. Seed control
reports observed forwarding to reset, separately from empirical determinism.
The declared <code>episode_timeout_s</code> bounds each episode; a reset or judged step may use
its remaining budget. Original and replay collection share a 300-second deadline
and a 32 MiB evidence budget. <code>llm_judged</code> requests 20 identical-input episodes,
including the original, plus the different-seed episode. Validation repeats the
actions and any external service calls they make, so budget for the additional
runtime and cost. Missing replay samples cannot pass determinism. Primary episode
collection failures appear under <code>runtime.startup</code>, with dependent contract checks
skipped and the completed trace retained.
Validator JSON numeric-range and depth limits report named capability <code>SKIP</code>s,
preserving the received evidence when it can be retained without validation credentials.</p> <p><code>--space</code> validates a Hugging Face Space on HF Jobs instead of a local directory:</p> <!> <p>The validator resolves <code>--revision</code> (a branch, tag or commit hash; default <code>main</code>)
to one commit and downloads and hashes the Space files at that commit. Hub download
bookkeeping is excluded. The report target names the resolved source commit.</p> <p>Runtime validation currently reports a named <code>immutable_image</code> SKIP without
creating an HF Job. The supported Hub API exposes a mutable Space image name and
current Space runtime metadata, but does not establish an immutable image digest
bound to the downloaded source commit. Matching repository and running-build
revisions alone cannot prove which image a Job would pull. A different repository
or reported runtime revision reports <code>runtime.startup</code> as <code>ERROR</code>. This provider
also refuses building, sleeping and paused Spaces: it requires reported stage <code>RUNNING</code>, without treating that stage as proof of image provenance. Absent runtime
stage or revision metadata yields a named <code>space_runtime_metadata</code> SKIP; explicit
contradictory metadata still yields ERROR. Use <code>--level static</code> to validate historical source commits. Reports retain the source
commit and source digest without claiming a verified runtime image or execution.</p> <p><code>--flavor</code> selects an HF Jobs hardware flavor (default <code>cpu-basic</code>; see <code>hf jobs hardware</code>). A GPU flavor satisfies the declared GPU prerequisite, but
image verification still blocks execution. The provider only supports <code>public</code> network mode; network policy, host containment and resource bounds require
capabilities it cannot inspect from outside the HF Job. CPU/GPU runtime and
WebSocket qualification, cleanup and the reported Sandbox startup issue remain
unverified on live HF infrastructure.</p> <p>The serving implementation supports ordinary Dockerfile backslash continuations
and final-stage commands, including inheritance from a named earlier stage.
Heredocs are unsupported: a <code>&lt;&lt;</code> marker in any non-comment instruction, including
a literal marker, reports a named <code>dockerfile_heredoc</code> SKIP before command parsing. <code>ENTRYPOINT</code>, custom <code>SHELL</code> commands, custom escape directives and commands
inherited from an external base image are unsupported. The server log retains at
most the first 64 KiB of combined stdout and stderr; further output is drained
and discarded. These serving paths have local regression coverage and do not
enable unverified image launches.</p> <p>Credential delivery is deferred for this release. Set <code>validation.execution.requires_credentials: true</code> when the environment needs an
externally supplied credential, such as a judge API key. This boolean declaration
causes a named <code>credential_delivery</code> SKIP before build or launch; dependent runtime
checks also SKIP. The resulting WARN does not establish complete Level 2 coverage.
The declaration accepts no secret values and defaults to false. Validation never
inherits host credentials or forwards API keys. Self-contained LLM judges can run
without this requirement; <code>llm_judged</code> alone does not imply a need for credentials.
An undeclared startup or reset failure still fails validation.</p> <p>Observation validation uses the <code>reset_observation</code> field from <code>/schema</code> for resets
and <code>observation</code> for steps. Servers can declare <code>reset_observation_cls</code> when
reset returns a different observation type; it defaults to the step observation
class. Older servers without a reset schema use the step schema for both. Reset
validation is always required; the validator does not silently substitute a base
schema. These are output schemas, separate from any reset input declaration.
The Level 2 profile permits null rewards only on reset. Every step, including a
nonterminal step, requires a finite numeric reward within the declared range;
boolean rewards are invalid. This certification requirement is stricter than the
backward-compatible core <code>Observation.reward</code> type. Emit zero only when the intended
reward is zero; the validator never converts missing or null step rewards to zero.</p> <p>Cleanup removes run-owned containers. Built images remain in Docker’s local cache
for reuse; the report records their immutable image IDs. Remove an unwanted image
with <code>docker image rm &lt;image-id&gt;</code> after its validation runs have finished.</p> <p>Runtime reports use schema version 2 and severity policy v2. Static reports for
v1 manifests retain schema version 1 and policy v1. An explicit v1 policy with a
runtime ceiling is rejected. With <code>--output report.json</code>, a sibling <code>report.artifacts/</code> directory contains the replay plan, bounded redacted evidence,
coverage inventory, provider settings, cleanup outcome and checksums. Treat these
as author-validation evidence; this command does not issue certification.</p> <p>For a pinned, installed-wheel reproduction of the shared fixture and its fault
cases, follow <a href="../../../tests/validation_runtime/README">the runtime lab</a>.
The implementation and remaining check inventory are tracked in <a href="https://github.com/openenv/OpenEnv/issues/1177" rel="nofollow">the Level 2 umbrella</a>.</p> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Validate a local package or a running OpenEnv server.</p> <p>Local validation detects the package format by its well-known file (the
formats this build can parse: <code>openenv.yaml</code> or Harbor <code>task.toml</code>), parses it into the
normalized manifest, runs the applicable graders up to the requested level,
applies the severity policy, and emits a report.</p> <p>Native Harbor tasks require the optional Harbor dependency and an explicit <code>validation/openenv.json</code> contract. Fresh trials grade the oracle, the
do-nothing floor and any declared canaries under the semantic check ids. They
run on the local Docker daemon, or with <code>--sandbox hf-sandbox</code> as one Hugging
Face Job per trial from the task’s digest-pinned image. Checks without an
implementation are named SKIPs, so the report is WARN at best.</p> <p>Exit codes: 0 pass/warn · 1 fail · 2 unrecognized/unsupported package · 3
internal or policy error.</p> <!></div> <!> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Push an OpenEnv environment to Hugging Face Spaces or a custom Docker registry.</p> <p>This command:</p> <ol><li>Validates that the directory is an OpenEnv environment (openenv.yaml present)</li> <li>Builds and pushes to Hugging Face Spaces or custom Docker registry</li> <li>Optionally enables web interface for deployment</li></ol> <p>The web interface is enabled by default when pushing to HuggingFace Spaces,
but disabled by default when pushing to a custom Docker registry.</p> <!></div> <!> <p>Local serving is not implemented in the CLI yet. This command exits non-zero
and prints alternative ways to run an environment server.</p> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Run the environment FastAPI app on the host for local development.</p> <p>Uses <code>openenv.yaml</code> fields <code>app</code> (e.g. <code>server.app:app</code>), <code>port</code>, and <code>runtime</code> (must be <code>fastapi</code>). Matches <code>uv run --project . server</code> layout:
the environment directory is the working directory. Package-qualified app
paths also support the environment package and the repo’s <code>envs</code> package.</p> <p>For production or training, use Docker (<code>openenv build</code>) — this command runs
on the host for local development only.</p></div> <!> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Fork (duplicate) a Hugging Face Space to your account using the Hub API.</p> <p>Uses the Hugging Face duplicate_space API. You can set environment variables
and secrets, and request hardware/storage/sleep time at creation time.</p> <!></div> <!> <p>Installs an <code>openenv-cli</code> skill into your AI assistant’s skills directory so
it knows the <code>openenv</code> CLI is available and what each command does. Supports
Claude Code, Cursor, Codex, and OpenCode.</p> <p><strong>Install for a single assistant (project-local):</strong></p> <!> <p>Multiple flags can be combined — <code>openenv skills add --claude --cursor</code> installs
for both at once. The skill file is written to a central location
(<code>.agents/skills/openenv-cli/</code>) and each agent directory gets a symlink, so
there is only one copy to update.</p> <p><strong>Install globally (user-level, across all projects):</strong></p> <!> <p><strong>Overwrite an existing installation</strong> (e.g. after upgrading <code>openenv</code>):</p> <!> <p><strong>Preview the skill content without installing:</strong></p> <!> <p><strong>Install to a custom path</strong> (for non-standard agent setups):</p> <!> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Install OpenEnv CLI skill for AI assistants.</p></div> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Print generated SKILL.md content.</p></div> <!> <p>Collect rollouts from a running environment with a teacher model and write
them as an SFT-ready <code>results.jsonl</code>. The teacher can be a hosted provider
(<code>--provider openai|anthropic</code>) or any self-hosted OpenAI-compatible server
such as vLLM, TGI or Ollama via <code>--llm-endpoint</code>.</p> <!> <p><code>--llm-endpoint</code> takes a full base URL. <code>/v1</code> is appended when the URL has no
path, so <code>http://localhost:8000</code> and <code>http://localhost:8000/v1</code> are equivalent;
a URL with a path (for example a gateway prefix) is used as-is. <code>--llm-port</code> is
only needed when the URL does not include a port; it has no default, so <code>--llm-endpoint http://localhost</code> means port 80 (earlier releases assumed 8000).
The resolved endpoint is printed when the run starts. Only <code>http(s)</code> URLs are
accepted, and credentials, query strings and fragments in the URL are rejected:
pass the key through <code>OPENAI_API_KEY</code> instead.</p> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Collect rollouts from a deployed OpenEnv environment.</p></div> <!> <p>Runs <a href="../environments/harbor">Harbor</a> tasks with real coding agents and
captures the token ids and logprobs of every model call. Requires <code>pip install "openenv[harbor]"</code> (Python 3.12 or newer).</p> <table><thead><tr><th>Command</th><th>What it does</th></tr></thead><tbody><tr><td><code>openenv harbor info</code></td><td>Reports the endpoint’s capture level and which sandboxes, datasets and harnesses this machine can use. Boots nothing.</td></tr><tr><td><code>openenv harbor rollout</code></td><td>Runs one or more rollouts end to end without a server.</td></tr><tr><td><code>openenv harbor serve</code></td><td>Serves Harbor tasks over the Task API and MCP, with a UI at <code>/web</code>.</td></tr><tr><td><code>openenv harbor push</code></td><td>Deploys the same server to a Hugging Face Space.</td></tr></tbody></table> <!> <p><code>--llm-url</code> is any OpenAI-compatible endpoint. Rollouts carry token ids only
when it returns them (vLLM with <code>--return-tokens-as-token-ids --logprobs-mode processed_logprobs</code>, or SGLang). <code>--api-key</code> defaults to <code>$OPENENV_LLM_API_KEY</code>. Run <code>openenv harbor &lt;command&gt; --help</code> for every flag,
and see the <a href="../environments/harbor#cli-reference">Harbor environment page</a> for details.</p> <!> <p>Build a versioned, metadata-only catalog of environments from a Git
repository, then search it before installing or running anything. See <a href="../guides/catalog-discovery">Discover environments before running them</a>.</p> <!> <p><code>catalog build</code> reads committed <code>openenv.yaml</code> files under <code>--root</code> (default <code>envs</code>) at <code>--revision</code> (default <code>HEAD</code>). <code>discover</code> ranks entries by query
match (<code>--limit</code>, default 20) and <code>--json</code> emits the full entries. <code>catalog inspect</code> prints one entry by its exact identifier.</p> <!> <!> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Main entry point for the CLI.</p></div> <!> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Validate that the directory follows OpenEnv environment structure.</p></div> <!> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Validate a running OpenEnv server against runtime API standards.</p> <p>The returned JSON report contains an overall pass/fail result and
per-criterion outcomes that can be consumed in CI.</p></div> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Validate that an environment is ready for multi-mode deployment.</p> <p>Checks:</p> <ol><li>pyproject.toml exists</li> <li>uv.lock exists</li> <li>pyproject.toml has [project.scripts] with server entry point</li> <li>server/app.py has a main() function</li> <li>Required dependencies are present</li></ol></div> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Check which deployment modes are supported by the environment.</p></div> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Format a validation report for display.</p></div> <div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"><!> <p>Build a JSON report for local environment validation.</p></div> <!> <p></p>`,1);function $e(ye,ge){Fe(ge,!1),Ye(()=>{new URLSearchParams(window.location.search).get("fw")}),Re();var N=Qe();xe("hdrvib",i=>{var c=Oe();Le(c,"content",Xe),p(i,c)});var W=e(m(N),2);Ve(W,{containerStyle:"float: inline-end; margin-inline-start: 10px; display: inline-flex; position: relative; z-index: 10;"});var S=e(W,2);s(S,{title:"CLI",local:"cli",headingTag:"h1"});var E=e(S,4);s(E,{title:"openenv init",local:"openenv.cli.commands.init",headingTag:"h2"});var v=e(E,2),R=o(v);a(R,{name:"openenv.cli.commands.init",anchor:"openenv.cli.commands.init",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/init.py#L404",parameters:[{name:"env_name",val:`: Annotated[str, typer.Argument(help="Name of the environment to create (snake_case, e.g., 'my_env')")]`},{name:"output_dir",val:": Annotated[str | None, typer.Option('--output-dir', '-o', help='Output directory (defaults to current working directory)')] = None"}]});var Me=e(R,6);h(Me,{anchor:"openenv.cli.commands.init.example",children:(i,c)=>{var r=u(),d=e(m(r),2);n(d,{code:"JTI0JTIwb3BlbmVudiUyMGluaXQlMjBteV9nYW1lX2VudiUwQSUyNCUyMG9wZW5lbnYlMjBpbml0JTIwbXlfZW52JTIwLS1vdXRwdXQtZGlyJTIwJTJGcGF0aCUyRnRvJTJGcHJvamVjdHM=",highlighted:`$ openenv init my_game_env
$ openenv init my_env --output-dir /path/to/projects`,lang:"bash",wrap:!1}),p(i,r)},$$slots:{default:!0}}),t(v);var x=e(v,2);s(x,{title:"openenv import",local:"openenv.cli.commands.import_env",headingTag:"h2"});var V=e(x,6);n(V,{code:"b3BlbmVudiUyMGltcG9ydCUyMHBhdGglMkZ0byUyRnNvdXJjZSUyMC0tbmFtZSUyMG15X2VudiUyMC0tb3V0cHV0LWRpciUyMC4lMkZlbnZzJTBBb3BlbmVudiUyMGltcG9ydCUyMHBhdGglMkZ0byUyRnNvdXJjZSUyMC0tbmFtZSUyMG15X2VudiUyMC0tb3V0cHV0LWRpciUyMC4lMkZlbnZzJTIwLS1lbnYtY2xhc3MlMjBNeUVudg==",highlighted:`openenv import path/to/source --name my_env --output-dir ./envs
openenv import path/to/source --name my_env --output-dir ./envs --env-class MyEnv`,lang:"bash",wrap:!1});var C=e(V,4);n(C,{code:"b3BlbmVudiUyMGltcG9ydCUyMHBhdGglMkZ0byUyRmJ1bmRsZSUyMC0tbmFtZSUyMG15X2VudiUyMC0tb3V0cHV0LWRpciUyMC4lMkZlbnZzJTBBb3BlbmVudiUyMGltcG9ydCUyMHBhdGglMkZ0byUyRnBsdWdpbiUyMC0tbmFtZSUyMG15X2VudiUyMC0tb3V0cHV0LWRpciUyMC4lMkZlbnZzJTIwLS1lbnYtY2xhc3MlMjBteS1idW5kbGU=",highlighted:`openenv import path/to/bundle --name my_env --output-dir ./envs
openenv import path/to/plugin --name my_env --output-dir ./envs --env-class my-bundle`,lang:"bash",wrap:!1});var b=e(C,2),we=o(b);a(we,{name:"openenv.cli.commands.import_env",anchor:"openenv.cli.commands.import_env",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/import_env.py#L58",parameters:[{name:"source",val:": Annotated[str, typer.Argument(help='Local source repository or directory to import')]"},{name:"name",val:": Annotated[str, typer.Option('--name', '-n', help='Name for the generated OpenEnv package')]"},{name:"output_dir",val:": Annotated[str, typer.Option('--output-dir', '-o', help='Directory where the generated package will be created')]"},{name:"env_class",val:": Annotated[str | None, typer.Option('--env-class', help='Environment class name or module:Class when detection is ambiguous')] = None"},{name:"source_type",val:`: Annotated[str | None, typer.Option('--type', help="Optional source type override, such as 'ors'")] = None`}]}),l(2),t(b);var L=e(b,2);s(L,{title:"openenv build",local:"openenv.cli.commands.build",headingTag:"h2"});var y=e(L,2),F=o(y);a(F,{name:"openenv.cli.commands.build",anchor:"openenv.cli.commands.build",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/build.py#L332",parameters:[{name:"env_path",val:": Annotated[str | None, typer.Argument(help='Path to the environment directory (default: current directory)')] = None"},{name:"tag",val:": Annotated[str | None, typer.Option('--tag', '-t', help='Docker image tag (default: openenv-<env_name>)')] = None"},{name:"context",val:": Annotated[str | None, typer.Option('--context', '-c', help='Build context path (default: <env_path>/server)')] = None"},{name:"dockerfile",val:": Annotated[str | None, typer.Option('--dockerfile', '-f', help='Path to Dockerfile (default: <context>/Dockerfile)')] = None"},{name:"no_cache",val:": Annotated[bool, typer.Option('--no-cache', help='Build without using cache')] = False"},{name:"build_arg",val:": Annotated[list[str] | None, typer.Option('--build-arg', help='Build arguments (can be used multiple times, format: KEY=VALUE)')] = None"}]});var fe=e(F,6);h(fe,{anchor:"openenv.cli.commands.build.example",children:(i,c)=>{var r=u(),d=e(m(r),2);n(d,{code:"JTIzJTIwQnVpbGQlMjBmcm9tJTIwZW52aXJvbm1lbnQlMjByb290JTIwKHJlY29tbWVuZGVkKSUwQSUyNCUyMGNkJTIwbXlfZW52JTBBJTI0JTIwb3BlbmVudiUyMGJ1aWxkJTBBJTBBJTIzJTIwQnVpbGQlMjB3aXRoJTIwY3VzdG9tJTIwdGFnJTBBJTI0JTIwb3BlbmVudiUyMGJ1aWxkJTIwLXQlMjBteS1jdXN0b20tdGFnJTBBJTBBJTIzJTIwQnVpbGQlMjB3aXRob3V0JTIwY2FjaGUlMEElMjQlMjBvcGVuZW52JTIwYnVpbGQlMjAtLW5vLWNhY2hlJTBBJTBBJTIzJTIwQnVpbGQlMjB3aXRoJTIwY3VzdG9tJTIwYnVpbGQlMjBhcmd1bWVudHMlMEElMjQlMjBvcGVuZW52JTIwYnVpbGQlMjAtLWJ1aWxkLWFyZyUyMFZFUlNJT04lM0QxLjAlMjAtLWJ1aWxkLWFyZyUyMEVOViUzRHByb2QlMEElMEElMjMlMjBCdWlsZCUyMGZyb20lMjBkaWZmZXJlbnQlMjBkaXJlY3RvcnklMEElMjQlMjBvcGVuZW52JTIwYnVpbGQlMjBlbnZzJTJGZWNob19lbnY=",highlighted:`<span class="hljs-comment"># Build from environment root (recommended)</span>
$ <span class="hljs-built_in">cd</span> my_env
$ openenv build

<span class="hljs-comment"># Build with custom tag</span>
$ openenv build -t my-custom-tag

<span class="hljs-comment"># Build without cache</span>
$ openenv build --no-cache

<span class="hljs-comment"># Build with custom build arguments</span>
$ openenv build --build-arg VERSION=1.0 --build-arg ENV=prod

<span class="hljs-comment"># Build from different directory</span>
$ openenv build envs/echo_env`,lang:"bash",wrap:!1}),p(i,r)},$$slots:{default:!0}}),t(y);var Y=e(y,2);s(Y,{title:"openenv validate",local:"openenv.cli.commands.validate",headingTag:"h2"});var H=e(Y,4);n(H,{code:"b3BlbmVudiUyMHZhbGlkYXRlJTIwLiUyRm15X2VudiUyMC0tbGV2ZWwlMjBzdGF0aWMlMEFvcGVuZW52JTIwdmFsaWRhdGUlMjAuJTJGbXlfZW52JTIwLS1sZXZlbCUyMHJ1bnRpbWUlMjAtLWxvY2FsJTIwLS1vdXRwdXQlMjByZXBvcnQuanNvbg==",highlighted:`openenv validate ./my_env --level static
openenv validate ./my_env --level runtime --<span class="hljs-built_in">local</span> --output report.json`,lang:"bash",wrap:!1});var X=e(H,6);n(X,{code:"cGlwJTIwaW5zdGFsbCUyMCUyMm9wZW5lbnYlNUJoYXJib3IlNUQlMjIlMjAlMjJoYXJib3IlM0QlM0QwLjI1LjAlMjI=",highlighted:'pip install <span class="hljs-string">&quot;openenv[harbor]&quot;</span> <span class="hljs-string">&quot;harbor==0.25.0&quot;</span>',lang:"bash",wrap:!1});var O=e(X,6);n(O,{code:"JTdCJTBBJTIwJTIwJTIycHJvZmlsZSUyMiUzQSUyMCUyMmhhcmJvci1zZW1hbnRpYy12MSUyMiUyQyUwQSUyMCUyMCUyMnJld2FyZF9rZXklMjIlM0ElMjAlMjJyZXdhcmQlMjIlMkMlMEElMjAlMjAlMjJyZXdhcmQlMjIlM0ElMjAlN0IlMEElMjAlMjAlMjAlMjAlMjJyYW5nZSUyMiUzQSUyMCU1QjAlMkMlMjAxJTVEJTJDJTBBJTIwJTIwJTIwJTIwJTIyb3JhY2xlX3RvbGVyYW5jZSUyMiUzQSUyMDAlMkMlMEElMjAlMjAlMjAlMjAlMjJmbG9vcl9tYXJnaW4lMjIlM0ElMjAwLjElMEElMjAlMjAlN0QlMkMlMEElMjAlMjAlMjJyZXNvdXJjZXMlMjIlM0ElMjAlN0IlMEElMjAlMjAlMjAlMjAlMjJjcHUlMjIlM0ElMjAxJTJDJTBBJTIwJTIwJTIwJTIwJTIybWVtb3J5X21iJTIyJTNBJTIwNTEyJTJDJTBBJTIwJTIwJTIwJTIwJTIyZGlza19tYiUyMiUzQSUyMDEwMjQlMkMlMEElMjAlMjAlMjAlMjAlMjJlcGlzb2RlX3RpbWVvdXRfcyUyMiUzQSUyMDMwMCUwQSUyMCUyMCU3RCUyQyUwQSUyMCUyMCUyMnR5cGVzJTIyJTNBJTIwJTdCJTIydGFncyUyMiUzQSUyMCU1QiUyMnN3ZSUyMiU1RCU3RCUwQSU3RA==",highlighted:`<span class="hljs-punctuation">{</span>
  <span class="hljs-attr">&quot;profile&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-string">&quot;harbor-semantic-v1&quot;</span><span class="hljs-punctuation">,</span>
  <span class="hljs-attr">&quot;reward_key&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-string">&quot;reward&quot;</span><span class="hljs-punctuation">,</span>
  <span class="hljs-attr">&quot;reward&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-punctuation">{</span>
    <span class="hljs-attr">&quot;range&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-punctuation">[</span><span class="hljs-number">0</span><span class="hljs-punctuation">,</span> <span class="hljs-number">1</span><span class="hljs-punctuation">]</span><span class="hljs-punctuation">,</span>
    <span class="hljs-attr">&quot;oracle_tolerance&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-number">0</span><span class="hljs-punctuation">,</span>
    <span class="hljs-attr">&quot;floor_margin&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-number">0.1</span>
  <span class="hljs-punctuation">}</span><span class="hljs-punctuation">,</span>
  <span class="hljs-attr">&quot;resources&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-punctuation">{</span>
    <span class="hljs-attr">&quot;cpu&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-number">1</span><span class="hljs-punctuation">,</span>
    <span class="hljs-attr">&quot;memory_mb&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-number">512</span><span class="hljs-punctuation">,</span>
    <span class="hljs-attr">&quot;disk_mb&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-number">1024</span><span class="hljs-punctuation">,</span>
    <span class="hljs-attr">&quot;episode_timeout_s&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-number">300</span>
  <span class="hljs-punctuation">}</span><span class="hljs-punctuation">,</span>
  <span class="hljs-attr">&quot;types&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-punctuation">{</span><span class="hljs-attr">&quot;tags&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-punctuation">[</span><span class="hljs-string">&quot;swe&quot;</span><span class="hljs-punctuation">]</span><span class="hljs-punctuation">}</span>
<span class="hljs-punctuation">}</span>`,lang:"json",wrap:!1});var Q=e(O,12);n(Q,{code:"b3BlbmVudiUyMHZhbGlkYXRlJTIwcGF0aCUyRnRvJTJGdGFzayUyMC0tc2FuZGJveCUyMGhmLXNhbmRib3glMEFvcGVuZW52JTIwdmFsaWRhdGUlMjBwYXRoJTJGdG8lMkZ0YXNrJTIwLS1zYW5kYm94JTIwaGYtc2FuZGJveCUyMC0tZmxhdm9yJTIwY3B1LXVwZ3JhZGUlMjAtLW5hbWVzcGFjZSUyMG15LW9yZw==",highlighted:`openenv validate path/to/task --sandbox hf-sandbox
openenv validate path/to/task --sandbox hf-sandbox --flavor cpu-upgrade --namespace my-org`,lang:"bash",wrap:!1});var P=e(Q,16);n(P,{code:"b3BlbmVudiUyMHZhbGlkYXRlJTIwLS1zcGFjZSUyMG9wZW5lbnYlMkZlY2hvX2VudiUyMC0tbGV2ZWwlMjBydW50aW1lJTBBb3BlbmVudiUyMHZhbGlkYXRlJTIwLS1zcGFjZSUyMG15LW9yZyUyRm15LWVudiUyMC0tbGV2ZWwlMjBydW50aW1lJTIwLS1mbGF2b3IlMjB0NC1zbWFsbA==",highlighted:`openenv validate --space openenv/echo_env --level runtime
openenv validate --space my-org/my-env --level runtime --flavor t4-small`,lang:"bash",wrap:!1});var g=e(P,20),z=o(g);a(z,{name:"openenv.cli.commands.validate",anchor:"openenv.cli.commands.validate",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/validate.py#L148",parameters:[{name:"target",val:": Annotated = None"},{name:"url",val:": Annotated = None"},{name:"level",val:": Annotated = 'semantic'"},{name:"skip_build",val:": Annotated = False"},{name:"local",val:": Annotated = False"},{name:"policy_version",val:": Annotated = None"},{name:"json_output",val:": Annotated = False"},{name:"output",val:": Annotated = None"},{name:"timeout",val:": Annotated = 5.0"},{name:"space",val:": Annotated = None"},{name:"revision",val:": Annotated = None"},{name:"flavor",val:": Annotated = 'cpu-basic'"},{name:"sandbox",val:": Annotated = 'docker'"},{name:"namespace",val:": Annotated = None"}]});var Je=e(z,10);h(Je,{anchor:"openenv.cli.commands.validate.example",children:(i,c)=>{var r=u(),d=e(m(r),2);n(d,{code:"JTIzJTIwVmFsaWRhdGUlMjB0aGUlMjBjdXJyZW50JTIwZGlyZWN0b3J5JTIwdXAlMjB0byUyMHRoZSUyMHNlbWFudGljJTIwbGV2ZWwlMEFvcGVuZW52JTIwdmFsaWRhdGUlMEElMEElMjMlMjBGYXN0JTIwaW5uZXIlMjBsb29wJTNBJTIwc3RhdGljJTIwY2hlY2tzJTIwb25seSUyQyUyMG5vJTIwaW1hZ2UlMjBidWlsZCUwQW9wZW5lbnYlMjB2YWxpZGF0ZSUyMGVudnMlMkZlY2hvX2VudiUyMC0tbGV2ZWwlMjBzdGF0aWMlMjAtLXNraXAtYnVpbGQlMEElMEElMjMlMjBNYWNoaW5lLXJlYWRhYmxlJTIwcmVwb3J0JTBBb3BlbmVudiUyMHZhbGlkYXRlJTIwZW52cyUyRmVjaG9fZW52JTIwLS1qc29uJTBBJTBBJTIzJTIwUHJvYmUlMjBhJTIwcnVubmluZyUyMHNlcnZlciUyMChsZWdhY3klMjBydW50aW1lJTIwcHJvYmUpJTBBb3BlbmVudiUyMHZhbGlkYXRlJTIwLS11cmwlMjBodHRwJTNBJTJGJTJGbG9jYWxob3N0JTNBODAwMCUwQSUwQSUyMyUyMFZhbGlkYXRlJTIwYSUyMEh1YiUyMFNwYWNlJTIwb24lMjBIRiUyMEpvYnMlMjBHUFUlMjBoYXJkd2FyZSUwQW9wZW5lbnYlMjB2YWxpZGF0ZSUyMC0tc3BhY2UlMjBvcGVuZW52JTJGZWNob19lbnYlMjAtLWxldmVsJTIwcnVudGltZSUyMC0tZmxhdm9yJTIwdDQtc21hbGwlMEElMEElMjMlMjBSdW4lMjBhJTIwSGFyYm9yJTIwdGFzaydzJTIwdHJpYWxzJTIwYXMlMjBIRiUyMEpvYnMlMjBpbnN0ZWFkJTIwb2YlMjBsb2NhbCUyMERvY2tlciUwQW9wZW5lbnYlMjB2YWxpZGF0ZSUyMHBhdGglMkZ0byUyRnRhc2slMjAtLXNhbmRib3glMjBoZi1zYW5kYm94JTIwLS1mbGF2b3IlMjBjcHUtdXBncmFkZQ==",highlighted:`<span class="hljs-comment"># Validate the current directory up to the semantic level</span>
openenv validate

<span class="hljs-comment"># Fast inner loop: static checks only, no image build</span>
openenv validate envs/echo_env --level static --skip-build

<span class="hljs-comment"># Machine-readable report</span>
openenv validate envs/echo_env --json

<span class="hljs-comment"># Probe a running server (legacy runtime probe)</span>
openenv validate --url http://localhost:8000

<span class="hljs-comment"># Validate a Hub Space on HF Jobs GPU hardware</span>
openenv validate --space openenv/echo_env --level runtime --flavor t4-small

<span class="hljs-comment"># Run a Harbor task&#x27;s trials as HF Jobs instead of local Docker</span>
openenv validate path/to/task --sandbox hf-sandbox --flavor cpu-upgrade`,lang:"bash",wrap:!1}),p(i,r)},$$slots:{default:!0}}),t(g);var D=e(g,2);s(D,{title:"openenv push",local:"openenv.cli.commands.push",headingTag:"h2"});var M=e(D,2),q=o(M);a(q,{name:"openenv.cli.commands.push",anchor:"openenv.cli.commands.push",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/push.py#L511",parameters:[{name:"directory",val:": Annotated[str | None, typer.Argument(help='Directory containing the OpenEnv environment (default: current directory)')] = None"},{name:"repo_id",val:`: Annotated[str | None, typer.Option('--repo-id', '-r', help="Repository ID as 'repo_name' or 'namespace/repo_name'. Defaults to 'username/env-name' from openenv.yaml.")] = None`},{name:"base_image",val:": Annotated[str | None, typer.Option('--base-image', '-b', help='Base Docker image to use (overrides Dockerfile FROM)')] = None"},{name:"interface",val:": Annotated[bool, typer.Option('--interface', help='Enable web interface (default: True if no registry specified)')] = None"},{name:"no_interface",val:": Annotated[bool, typer.Option('--no-interface', help='Disable web interface')] = False"},{name:"registry",val:": Annotated[str | None, typer.Option('--registry', help='Custom registry URL (e.g., docker.io/username). Disables web interface by default.')] = None"},{name:"private",val:": Annotated[bool, typer.Option('--private', help='Deploy the space as private')] = False"},{name:"create_pr",val:": Annotated[bool, typer.Option('--create-pr', help='Create a Pull Request instead of pushing to the default branch')] = False"},{name:"exclude",val:": Annotated[str | None, typer.Option('--exclude', help='Optional additional ignore file with newline-separated glob patterns to exclude from Hugging Face uploads')] = None"},{name:"delete",val:": Annotated[list[str] | None, typer.Option('--delete', help=\"Glob pattern of remote Space files to delete in the same commit, as in `hf upload --delete` (repeatable). Use '*' to remove every file not uploaded again.\")] = None"},{name:"hardware",val:": Annotated[str | None, typer.Option('--hardware', '-H', help='Request hardware for Hugging Face Space (e.g. t4-medium, cpu-basic). See HF docs for options.')] = None"},{name:"count",val:": Annotated[int, typer.Option('--count', '-n', help='Number of Space instances to deploy. Each gets a numeric suffix (e.g. env-1, env-2).', min=1)] = 1"},{name:"env_vars",val:": Annotated[list[str] | None, typer.Option('--env-var', '-e', help='Public Space variable as KEY=VALUE (repeatable). Overrides matching keys from openenv.yaml variables:.')] = None"},{name:"secrets",val:": Annotated[list[str] | None, typer.Option('--secret', help='Private Space secret as KEY=VALUE (repeatable). Value is never logged.')] = None"}]});var Te=e(q,10);h(Te,{anchor:"openenv.cli.commands.push.example",children:(i,c)=>{var r=u(),d=e(m(r),2);n(d,{code:"JTIzJTIwUHVzaCUyMHRvJTIwSHVnZ2luZ0ZhY2UlMjBTcGFjZXMlMjBmcm9tJTIwY3VycmVudCUyMGRpcmVjdG9yeSUyMCh3ZWIlMjBpbnRlcmZhY2UlMjBlbmFibGVkKSUwQSUyNCUyMGNkJTIwbXlfZW52JTBBJTI0JTIwb3BlbmVudiUyMHB1c2glMEElMEElMjMlMjBQdXNoJTIwdG8lMjBIdWdnaW5nRmFjZSUyMHJlcG8lMjBhbmQlMjBvcGVuJTIwYSUyMFB1bGwlMjBSZXF1ZXN0JTBBJTI0JTIwb3BlbmVudiUyMHB1c2glMjBteS1vcmclMkZteS1lbnYlMjAtLWNyZWF0ZS1wciUwQSUyNCUyMG9wZW5lbnYlMjBwdXNoJTIwLS1yZXBvLWlkJTIwbXktb3JnJTJGbXktZW52JTIwLS1jcmVhdGUtcHIlMEElMEElMjMlMjBQdXNoJTIwdG8lMjBIdWdnaW5nRmFjZSUyMHdpdGhvdXQlMjB3ZWIlMjBpbnRlcmZhY2UlMEElMjQlMjBvcGVuZW52JTIwcHVzaCUyMC0tbm8taW50ZXJmYWNlJTBBJTBBJTIzJTIwUHVzaCUyMHRvJTIwRG9ja2VyJTIwSHViJTBBJTI0JTIwb3BlbmVudiUyMHB1c2glMjAtLXJlZ2lzdHJ5JTIwZG9ja2VyLmlvJTJGbXl1c2VyJTBBJTBBJTIzJTIwUHVzaCUyMHRvJTIwR2l0SHViJTIwQ29udGFpbmVyJTIwUmVnaXN0cnklMEElMjQlMjBvcGVuZW52JTIwcHVzaCUyMC0tcmVnaXN0cnklMjBnaGNyLmlvJTJGbXlvcmclMEElMEElMjMlMjBQdXNoJTIwdG8lMjBjdXN0b20lMjByZWdpc3RyeSUyMHdpdGglMjB3ZWIlMjBpbnRlcmZhY2UlMEElMjQlMjBvcGVuZW52JTIwcHVzaCUyMC0tcmVnaXN0cnklMjBteXJlZ2lzdHJ5LmlvJTJGcGF0aDElMkZwYXRoMiUyMC0taW50ZXJmYWNlJTBBJTBBJTIzJTIwUHVzaCUyMHRvJTIwc3BlY2lmaWMlMjBIdWdnaW5nRmFjZSUyMHJlcG8lMEElMjQlMjBvcGVuZW52JTIwcHVzaCUyMC0tcmVwby1pZCUyMG15LW9yZyUyRm15LWVudiUwQSUwQSUyMyUyMFB1c2glMjBwcml2YXRlbHklMjB3aXRoJTIwY3VzdG9tJTIwYmFzZSUyMGltYWdlJTBBJTI0JTIwb3BlbmVudiUyMHB1c2glMjAtLXByaXZhdGUlMjAtLWJhc2UtaW1hZ2UlMjBnaGNyLmlvJTJGaHVnZ2luZ2ZhY2UlMkZvcGVuZW52LWJhc2UlM0FsYXRlc3QlMEElMEElMjMlMjBQdXNoJTIwd2l0aCUyMEdQVSUyMGhhcmR3YXJlJTBBJTI0JTIwb3BlbmVudiUyMHB1c2glMjAtLWhhcmR3YXJlJTIwdDQtbWVkaXVtJTBBJTBBJTIzJTIwU2V0JTIwYSUyMHB1YmxpYyUyMFNwYWNlJTIwdmFyaWFibGUlMjAob3ZlcnJpZGVzJTIwb3BlbmVudi55YW1sJTIwdmFyaWFibGVzJTNBKSUwQSUyNCUyMG9wZW5lbnYlMjBwdXNoJTIwLWUlMjBPUEVOU1BJRUxfR0FNRSUzRHRpY190YWNfdG9lJTIwLWUlMjBNQVhfU1RFUFMlM0QxMDAlMEElMEElMjMlMjBTZXQlMjBhJTIwcHJpdmF0ZSUyMFNwYWNlJTIwc2VjcmV0JTIwKHZhbHVlJTIwbmV2ZXIlMjBsb2dnZWQpJTBBJTI0JTIwb3BlbmVudiUyMHB1c2glMjAtLXNlY3JldCUyME9QRU5BSV9BUElfS0VZJTNEc2stLi4u",highlighted:`<span class="hljs-comment"># Push to HuggingFace Spaces from current directory (web interface enabled)</span>
$ <span class="hljs-built_in">cd</span> my_env
$ openenv push

<span class="hljs-comment"># Push to HuggingFace repo and open a Pull Request</span>
$ openenv push my-org/my-env --create-pr
$ openenv push --repo-id my-org/my-env --create-pr

<span class="hljs-comment"># Push to HuggingFace without web interface</span>
$ openenv push --no-interface

<span class="hljs-comment"># Push to Docker Hub</span>
$ openenv push --registry docker.io/myuser

<span class="hljs-comment"># Push to GitHub Container Registry</span>
$ openenv push --registry ghcr.io/myorg

<span class="hljs-comment"># Push to custom registry with web interface</span>
$ openenv push --registry myregistry.io/path1/path2 --interface

<span class="hljs-comment"># Push to specific HuggingFace repo</span>
$ openenv push --repo-id my-org/my-env

<span class="hljs-comment"># Push privately with custom base image</span>
$ openenv push --private --base-image ghcr.io/huggingface/openenv-base:latest

<span class="hljs-comment"># Push with GPU hardware</span>
$ openenv push --hardware t4-medium

<span class="hljs-comment"># Set a public Space variable (overrides openenv.yaml variables:)</span>
$ openenv push -e OPENSPIEL_GAME=tic_tac_toe -e MAX_STEPS=100

<span class="hljs-comment"># Set a private Space secret (value never logged)</span>
$ openenv push --secret OPENAI_API_KEY=sk-...`,lang:"bash",wrap:!1}),p(i,r)},$$slots:{default:!0}}),t(M);var $=e(M,2);s($,{title:"openenv serve",local:"openenv.cli.commands.serve",headingTag:"h2"});var w=e($,4),je=o(w);a(je,{name:"openenv.cli.commands.serve",anchor:"openenv.cli.commands.serve",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/serve.py#L18",parameters:[{name:"env_path",val:": Annotated[str | None, typer.Argument(help='Path to the environment directory (default: current directory)')] = None"},{name:"port",val:": Annotated[int | None, typer.Option('--port', '-p', help='Port to bind (default: ``port`` in openenv.yaml, else 8000)')] = None"},{name:"host",val:": Annotated[str, typer.Option('--host', help='Host interface to bind (local development only)')] = '127.0.0.1'"},{name:"reload",val:": Annotated[bool, typer.Option('--reload', help='Enable autoreload (development)')] = False"}]}),l(6),t(w);var K=e(w,2);s(K,{title:"openenv fork",local:"openenv.cli.commands.fork",headingTag:"h2"});var f=e(K,2),ee=o(f);a(ee,{name:"openenv.cli.commands.fork",anchor:"openenv.cli.commands.fork",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/fork.py#L58",parameters:[{name:"source_space",val:`: Annotated[str, typer.Argument(help="Source Space ID in format 'owner/space-name' (e.g. org/my-openenv-space)")]`},{name:"repo_id",val:": Annotated[str | None, typer.Option('--repo-id', '-r', help='Target repo ID for the fork (default: created under your account with same name)')] = None"},{name:"private",val:": Annotated[bool, typer.Option('--private', help='Create the forked Space as private')] = False"},{name:"set_env",val:": Annotated[list[str], typer.Option('--set-env', '-e', help='Set Space variable (public). Can be repeated. Format: KEY=VALUE')] = []"},{name:"set_secret",val:": Annotated[list[str], typer.Option('--set-secret', '--secret', '-s', help='Set Space secret. Can be repeated. Format: KEY=VALUE')] = []"},{name:"hardware",val:": Annotated[str | None, typer.Option('--hardware', '-H', help='Request hardware (e.g. t4-medium, cpu-basic). See Hub docs for options.')] = None"}]});var Ue=e(ee,6);h(Ue,{anchor:"openenv.cli.commands.fork.example",children:(i,c)=>{var r=u(),d=e(m(r),2);n(d,{code:"JTI0JTIwb3BlbmVudiUyMGZvcmslMjBvd25lciUyRnNvdXJjZS1zcGFjZSUwQSUyNCUyMG9wZW5lbnYlMjBmb3JrJTIwb3duZXIlMkZzb3VyY2Utc3BhY2UlMjAtLXByaXZhdGUlMEElMjQlMjBvcGVuZW52JTIwZm9yayUyMG93bmVyJTJGc291cmNlLXNwYWNlJTIwLS1yZXBvLWlkJTIwbXl1c2VyJTJGbXktZm9yayUwQSUyNCUyMG9wZW5lbnYlMjBmb3JrJTIwb3duZXIlMkZzb3VyY2Utc3BhY2UlMjAtLXNldC1lbnYlMjBNT0RFTF9JRCUzRHVzZXIlMkZtb2RlbCUyMC0tc2V0LXNlY3JldCUyMEhGX1RPS0VOJTNEaGZfeHh4JTBBJTI0JTIwb3BlbmVudiUyMGZvcmslMjBvd25lciUyRnNvdXJjZS1zcGFjZSUyMC0taGFyZHdhcmUlMjB0NC1tZWRpdW0=",highlighted:`$ openenv fork owner/source-space
$ openenv fork owner/source-space --private
$ openenv fork owner/source-space --repo-id myuser/my-fork
$ openenv fork owner/source-space --set-env MODEL_ID=user/model --set-secret HF_TOKEN=hf_xxx
$ openenv fork owner/source-space --hardware t4-medium`,lang:"bash",wrap:!1}),p(i,r)},$$slots:{default:!0}}),t(f);var ne=e(f,2);s(ne,{title:"openenv skills",local:"openenv.cli.commands.skills.skills_add",headingTag:"h2"});var ae=e(ne,6);n(ae,{code:"b3BlbmVudiUyMHNraWxscyUyMGFkZCUyMC0tY2xhdWRlJTIwJTIwJTIwJTIwJTIzJTIwJUUyJTg2JTkyJTIwLmNsYXVkZSUyRnNraWxscyUyRm9wZW5lbnYtY2xpJTJGJTBBb3BlbmVudiUyMHNraWxscyUyMGFkZCUyMC0tY3Vyc29yJTIwJTIwJTIwJTIwJTIzJTIwJUUyJTg2JTkyJTIwLmN1cnNvciUyRnNraWxscyUyRm9wZW5lbnYtY2xpJTJGJTBBb3BlbmVudiUyMHNraWxscyUyMGFkZCUyMC0tY29kZXglMjAlMjAlMjAlMjAlMjAlMjMlMjAlRTIlODYlOTIlMjAuY29kZXglMkZza2lsbHMlMkZvcGVuZW52LWNsaSUyRiUwQW9wZW5lbnYlMjBza2lsbHMlMjBhZGQlMjAtLW9wZW5jb2RlJTIwJTIwJTIzJTIwJUUyJTg2JTkyJTIwLm9wZW5jb2RlJTJGc2tpbGxzJTJGb3BlbmVudi1jbGklMkY=",highlighted:`openenv skills add --claude    <span class="hljs-comment"># → .claude/skills/openenv-cli/</span>
openenv skills add --cursor    <span class="hljs-comment"># → .cursor/skills/openenv-cli/</span>
openenv skills add --codex     <span class="hljs-comment"># → .codex/skills/openenv-cli/</span>
openenv skills add --opencode  <span class="hljs-comment"># → .opencode/skills/openenv-cli/</span>`,lang:"bash",wrap:!1});var oe=e(ae,6);n(oe,{code:"b3BlbmVudiUyMHNraWxscyUyMGFkZCUyMC0tY2xhdWRlJTIwLS1nbG9iYWwlMjAlMjAlMjMlMjAlRTIlODYlOTIlMjB+JTJGLmNsYXVkZSUyRnNraWxscyUyRm9wZW5lbnYtY2xpJTJG",highlighted:'openenv skills add --claude --global  <span class="hljs-comment"># → ~/.claude/skills/openenv-cli/</span>',lang:"bash",wrap:!1});var te=e(oe,4);n(te,{code:"b3BlbmVudiUyMHNraWxscyUyMGFkZCUyMC0tY2xhdWRlJTIwLS1mb3JjZQ==",highlighted:"openenv skills add --claude --force",lang:"bash",wrap:!1});var se=e(te,4);n(se,{code:"b3BlbmVudiUyMHNraWxscyUyMHByZXZpZXc=",highlighted:"openenv skills preview",lang:"bash",wrap:!1});var re=e(se,4);n(re,{code:"b3BlbmVudiUyMHNraWxscyUyMGFkZCUyMC0tZGVzdCUyMCUyRnBhdGglMkZ0byUyRm15LWFnZW50JTJGc2tpbGxzJTJG",highlighted:"openenv skills add --dest /path/to/my-agent/skills/",lang:"bash",wrap:!1});var J=e(re,2),ke=o(J);a(ke,{name:"openenv.cli.commands.skills.skills_add",anchor:"openenv.cli.commands.skills.skills_add",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/skills.py#L143",parameters:[{name:"claude",val:": Annotated[bool, typer.Option('--claude', help='Install for Claude.')] = False"},{name:"codex",val:": Annotated[bool, typer.Option('--codex', help='Install for Codex.')] = False"},{name:"cursor",val:": Annotated[bool, typer.Option('--cursor', help='Install for Cursor.')] = False"},{name:"opencode",val:": Annotated[bool, typer.Option('--opencode', help='Install for OpenCode.')] = False"},{name:"global_",val:": Annotated[bool, typer.Option('--global', '-g', help='Install globally (user-level) instead of in the current project directory.')] = False"},{name:"dest",val:": Annotated[Path | None, typer.Option(help='Install into a custom destination (skills directory path).')] = None"},{name:"force",val:": Annotated[bool, typer.Option('--force', help='Overwrite existing skills in the destination.')] = False"}]}),l(2),t(J);var T=e(J,2),Ie=o(T);a(Ie,{name:"openenv.cli.commands.skills.skills_preview",anchor:"openenv.cli.commands.skills.skills_preview",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/skills.py#L137",parameters:[]}),l(2),t(T);var le=e(T,2);s(le,{title:"openenv collect",local:"openenv.cli.commands.collect",headingTag:"h2"});var ie=e(le,4);n(ie,{code:"JTIzJTIwU2NyaXB0ZWQlMjB0ZWFjaGVyJTJDJTIwbm8lMjBBUEklMjBrZXklMjBuZWVkZWQlMEFvcGVuZW52JTIwY29sbGVjdCUyMG9wZW5zcGllbCUzQXRpY190YWNfdG9lJTIwLS1iYXNlLXVybCUyMGh0dHAlM0ElMkYlMkZsb2NhbGhvc3QlM0E4MDAxJTIwJTVDJTBBJTIwJTIwJTIwJTIwLS1vdXRwdXQtZGlyJTIwLiUyRnJvbGxvdXRzJTIwLW4lMjAxMCUyMC0tcHJvdmlkZXIlMjBzY3JpcHRlZCUwQSUwQSUyMyUyMFNlbGYtaG9zdGVkJTIwbW9kZWwlMjAodkxMTSUyMHNlcnZpbmclMjBvbiUyMHBvcnQlMjA4MDAwKSUwQW9wZW5lbnYlMjBjb2xsZWN0JTIwcmVhc29uaW5nX2d5bSUzQWNoYWluX3N1bSUyMC0tYmFzZS11cmwlMjBodHRwJTNBJTJGJTJGbG9jYWxob3N0JTNBODAwMSUyMCU1QyUwQSUyMCUyMCUyMCUyMC0tb3V0cHV0LWRpciUyMC4lMkZyb2xsb3V0cyUyMC1uJTIwNTAlMjAlNUMlMEElMjAlMjAlMjAlMjAtLWxsbS1lbmRwb2ludCUyMGh0dHAlM0ElMkYlMkZsb2NhbGhvc3QlM0E4MDAwJTIwLS1tb2RlbCUyMFF3ZW4lMkZRd2VuMy0xLjdC",highlighted:`<span class="hljs-comment"># Scripted teacher, no API key needed</span>
openenv collect openspiel:tic_tac_toe --base-url http://localhost:8001 \\
    --output-dir ./rollouts -n 10 --provider scripted

<span class="hljs-comment"># Self-hosted model (vLLM serving on port 8000)</span>
openenv collect reasoning_gym:chain_sum --base-url http://localhost:8001 \\
    --output-dir ./rollouts -n 50 \\
    --llm-endpoint http://localhost:8000 --model Qwen/Qwen3-1.7B`,lang:"bash",wrap:!1});var j=e(ie,4),_e=o(j);a(_e,{name:"openenv.cli.commands.collect",anchor:"openenv.cli.commands.collect",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/commands/collect.py#L228",parameters:[{name:"env",val:`: Annotated[str, typer.Argument(help="Env spec in 'family:variant' form (e.g. openspiel:tic_tac_toe).")]`},{name:"base_url",val:": Annotated[str, typer.Option('--base-url', help='Env server URL (local Docker or Hugging Face Space).')]"},{name:"output_dir",val:": Annotated[Path, typer.Option('--output-dir', '-o', help='Directory to write results.jsonl + metadata.json.')]"},{name:"num_episodes",val:": Annotated[int, typer.Option('--num-episodes', '-n', help='Number of episodes to collect.')] = 10"},{name:"max_turns",val:": Annotated[int, typer.Option('--max-turns', help='Max tool/model turns per episode.')] = 9"},{name:"episode_id_prefix",val:": Annotated[str, typer.Option('--episode-id-prefix', help='Prefix for serialized episode ids.')] = 'ep'"},{name:"resume",val:": Annotated[bool, typer.Option('--resume/--no-resume', help='Skip episodes already present in results.jsonl.')] = True"},{name:"provider",val:": Annotated[str, typer.Option('--provider', help='Teacher provider: scripted | openai | anthropic.')] = 'scripted'"},{name:"model",val:": Annotated[str | None, typer.Option('--model', help='Model id (required when provider != scripted).')] = None"},{name:"llm_endpoint",val:": Annotated[str | None, typer.Option('--llm-endpoint', help='Base URL of a self-hosted OpenAI-compatible server (vLLM/TGI/Ollama), e.g. http://localhost:8000. /v1 is appended when the URL has no path; a URL with a path is used as-is.')] = None"},{name:"llm_port",val:": Annotated[int | None, typer.Option('--llm-port', help='Port appended to --llm-endpoint when the URL does not include one. No default: earlier releases assumed 8000.')] = None"},{name:"temperature",val:": Annotated[float, typer.Option('--temperature', help='Sampling temperature.')] = 0.2"},{name:"max_tokens",val:": Annotated[int, typer.Option('--max-tokens', help='Max completion tokens.')] = 200"},{name:"keep_losses",val:": Annotated[bool, typer.Option('--keep-losses', help='Keep losing rollouts (default: filter rollouts with reward < 0).')] = False"},{name:"push_to_hub",val:`: Annotated[str | None, typer.Option('--push-to-hub', '-H', help="Destination dataset repo id ('user/name'). Uploads after collect.")] = None`},{name:"private",val:": Annotated[bool, typer.Option('--private', help='Create the Hub dataset repo as private.')] = False"},{name:"commit_message",val:": Annotated[str | None, typer.Option('--commit-message', help='Commit message for the Hub upload.')] = None"},{name:"dataset_config",val:`: Annotated[str | None, typer.Option('--dataset-config', help='JSON string of dataset config for envs that support it (e.g. reasoning_gym). Example: \\'{"min_terms": 2, "max_terms": 3}\\'')] = None`},{name:"system_prompt",val:": Annotated[str | None, typer.Option('--system-prompt', help='Custom system prompt for the teacher model.')] = None"}]}),l(2),t(j);var de=e(j,2);s(de,{title:"openenv harbor",local:"openenv-harbor",headingTag:"h2"});var ce=e(de,6);n(ce,{code:"b3BlbmVudiUyMGhhcmJvciUyMGluZm8lMjAtLWxsbS11cmwlMjAlMjRMTE0lMjAtLWRhdGFzZXQlMjBBZGl0aHlhU0slMkZkYXRhX2FnZW50X3JsX2Vudmlyb25tZW50X2V2YWwlMEElMEFvcGVuZW52JTIwaGFyYm9yJTIwcm9sbG91dCUyMC0tbGxtLXVybCUyMCUyNExMTSUyMCU1QyUwQSUyMCUyMCUyMCUyMC0tZGF0YXNldCUyMEFkaXRoeWFTSyUyRmRhdGFfYWdlbnRfcmxfZW52aXJvbm1lbnRfZXZhbCUyMCU1QyUwQSUyMCUyMCUyMCUyMC0tdGFzay1pbmRleCUyMDAlMjAtLWhhcm5lc3MlMjBvcGVuY29kZSUyMC0tc2FuZGJveCUyMGUyYiUyMC0tb3V0JTIwcm9sbG91dC5qc29uJTBBJTBBb3BlbmVudiUyMGhhcmJvciUyMHNlcnZlJTIwLS1sbG0tdXJsJTIwJTI0TExNJTIwLS1kYXRhc2V0JTIwb3JnJTJGdHJhaW4lMkNvcmclMkZldmFs",highlighted:`openenv harbor info --llm-url <span class="hljs-variable">$LLM</span> --dataset AdithyaSK/data_agent_rl_environment_eval

openenv harbor rollout --llm-url <span class="hljs-variable">$LLM</span> \\
    --dataset AdithyaSK/data_agent_rl_environment_eval \\
    --task-index 0 --harness opencode --sandbox e2b --out rollout.json

openenv harbor serve --llm-url <span class="hljs-variable">$LLM</span> --dataset org/train,org/eval`,lang:"bash",wrap:!1});var pe=e(ce,4);s(pe,{title:"openenv catalog and openenv discover",local:"openenv-catalog-and-openenv-discover",headingTag:"h2"});var me=e(pe,4);n(me,{code:"b3BlbmVudiUyMGNhdGFsb2clMjBidWlsZCUyMC0tcmVwb3NpdG9yeSUyMC4lMjAlNUMlMEElMjAlMjAlMjAlMjAtLXJlcG9zaXRvcnktdXJpJTIwaHR0cHMlM0ElMkYlMkZnaXRodWIuY29tJTJGb3BlbmVudiUyRk9wZW5FbnYuZ2l0JTIwJTVDJTBBJTIwJTIwJTIwJTIwLS1wdWJsaXNoZXIlMjBleGFtcGxlLm9yZyUyMC0tb3V0cHV0JTIwY2F0YWxvZy5qc29uJTBBJTBBb3BlbmVudiUyMGRpc2NvdmVyJTIwJTIyY2xpZW50JTIwc21va2UlMjB0ZXN0JTIyJTIwLS1jYXRhbG9nJTIwY2F0YWxvZy5qc29uJTBBb3BlbmVudiUyMGRpc2NvdmVyJTIwJTIyJTIyJTIwLS1jYXRhbG9nJTIwY2F0YWxvZy5qc29uJTIwLS1maWx0ZXIlMjBsaWNlbnNlJTNEQlNELTMtQ2xhdXNlJTBBb3BlbmVudiUyMGNhdGFsb2clMjBpbnNwZWN0JTIwJTIyJTNDaWRlbnRpZmllciUyMGZyb20lMjB0aGUlMjByZXN1bHQlM0UlMjIlMjAtLWNhdGFsb2clMjBjYXRhbG9nLmpzb24=",highlighted:`openenv catalog build --repository . \\
    --repository-uri https://github.com/openenv/OpenEnv.git \\
    --publisher example.org --output catalog.json

openenv discover <span class="hljs-string">&quot;client smoke test&quot;</span> --catalog catalog.json
openenv discover <span class="hljs-string">&quot;&quot;</span> --catalog catalog.json --filter license=BSD-3-Clause
openenv catalog inspect <span class="hljs-string">&quot;&lt;identifier from the result&gt;&quot;</span> --catalog catalog.json`,lang:"bash",wrap:!1});var he=e(me,4);s(he,{title:"API Reference",local:"api-reference",headingTag:"h1"});var ue=e(he,2);s(ue,{title:"Entry point",local:"openenv.cli. main .main",headingTag:"h2"});var U=e(ue,2),Be=o(U);a(Be,{name:"openenv.cli.__main__.main",anchor:"openenv.cli.__main__.main",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/__main__.py#L76",parameters:[]}),l(2),t(U);var ve=e(U,2);s(ve,{title:"CLI helpers",local:"openenv.cli._cli_utils.validate_env_structure",headingTag:"h2"});var k=e(ve,2),Ze=o(k);a(Ze,{name:"openenv.cli._cli_utils.validate_env_structure",anchor:"openenv.cli._cli_utils.validate_env_structure",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/_cli_utils.py#L30",parameters:[{name:"env_dir",val:": Path"},{name:"strict",val:": bool = False"}],parametersDescription:[{anchor:"openenv.cli._cli_utils.validate_env_structure.env_dir",description:`<strong>env_dir</strong> (<code>Path</code>) &#x2014;
Path to the environment directory.`,name:"env_dir"},{anchor:"openenv.cli._cli_utils.validate_env_structure.strict",description:`<strong>strict</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
If <code>True</code>, enforce all optional requirements.`,name:"strict"}],returnDescription:`
<p><code>list</code> of validation warnings (empty if all checks pass).</p>
`,raiseDescription:`
<ul>
<li><code>FileNotFoundError</code> — If required files are missing.</li>
</ul>
`,raiseType:`
<p><code>FileNotFoundError</code></p>
`}),l(2),t(k);var be=e(k,2);s(be,{title:"Validation utilities",local:"openenv.cli._validation.validate_running_environment",headingTag:"h2"});var I=e(be,2),Ge=o(I);a(Ge,{name:"openenv.cli._validation.validate_running_environment",anchor:"openenv.cli._validation.validate_running_environment",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/_validation.py#L99",parameters:[{name:"base_url",val:": str"},{name:"timeout_s",val:": float = 5.0"}]}),l(4),t(I);var _=e(I,2),Ae=o(_);a(Ae,{name:"openenv.cli._validation.validate_multi_mode_deployment",anchor:"openenv.cli._validation.validate_multi_mode_deployment",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/_validation.py#L505",parameters:[{name:"env_path",val:": Path"}],returnDescription:`
<p><code>tuple</code> of <code>(is_valid, issues)</code> where <code>is_valid</code> is a <code>bool</code> and <code>issues</code> is a
<code>list</code> of issue strings found during validation.</p>
`}),l(6),t(_);var B=e(_,2),Ne=o(B);a(Ne,{name:"openenv.cli._validation.get_deployment_modes",anchor:"openenv.cli._validation.get_deployment_modes",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/_validation.py#L581",parameters:[{name:"env_path",val:": Path"}],returnDescription:`
<p><code>dict</code> mapping deployment mode names to whether they are supported.</p>
`}),l(2),t(B);var Z=e(B,2),We=o(Z);a(We,{name:"openenv.cli._validation.format_validation_report",anchor:"openenv.cli._validation.format_validation_report",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/_validation.py#L610",parameters:[{name:"env_name",val:": str"},{name:"is_valid",val:": bool"},{name:"issues",val:": list[str]"}],returnDescription:`
<p>formatted validation report.</p>
`,returnType:`
<p><code>str</code></p>
`}),l(2),t(Z);var G=e(Z,2),Se=o(G);a(Se,{name:"openenv.cli._validation.build_local_validation_json_report",anchor:"openenv.cli._validation.build_local_validation_json_report",source:"https://github.com/openenv/OpenEnv/blob/main/src/openenv/cli/_validation.py#L628",parameters:[{name:"env_name",val:": str"},{name:"env_path",val:": Path"},{name:"is_valid",val:": bool"},{name:"issues",val:": list[str]"},{name:"deployment_modes",val:": dict[str, bool] | None = None"}]}),l(2),t(G);var Ee=e(G,2);Ce(Ee,{source:"https://github.com/openenv/OpenEnv/blob/main/docs/source/reference/cli.md"}),l(2),p(ye,N),He()}export{$e as component};
