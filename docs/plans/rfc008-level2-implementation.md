# RFC 008 Level 2 implementation plan

Planning baseline: 2026-09-16, `huggingface/OpenEnv` main at
`e3eb3fa5bc11ff0a019c30c7a7b1b7039eb8b615`.

Delivery tracker: [Level 2 umbrella #1177](https://github.com/huggingface/OpenEnv/issues/1177).
This document describes the full roadmap. The first delivery implements PRs 1–3;
subsequent slices remain planned. See the umbrella and the shared case catalog for
current implementation coverage and validation evidence.

## Outcome and scope

Deliver a reproducible Docker-local implementation of all **15 existing local
`runtime.*` checks** in RFC 008, plus an explicit startup outcome. A reviewer should
be able to build the exact revision, run a known-good subject, introduce one known
defect, observe the expected finding, and inspect the same evidence bundle that CI
produces.

The reference environment is Linux, x86-64, Docker and a qualified cgroup/storage
configuration. Fast tests work without Docker. Docker Desktop and other platforms
can run the same harness, but their capability reports determine which isolation
and resource claims they can establish. They do not replace the reference evidence.

First support the served OpenEnv execution binding and CPU fixtures. Preserve a
format-neutral grader interface so subsequent Harbor/PostTrain execution bindings
and an HF Sandbox provider reuse the checks. Those implementations, GPU provider
qualification, Level 3 oracle scoring/floor checks, remaining Level 1 graders,
operator certification and publish gating are separate work.

The implementation must handle LLM-judged declarations honestly: implement the
specified bounded variance path using a controlled test judge; if a real provider
or judge cannot run, report its missing capability or prerequisite. Live model
inference is not a prerequisite for the deterministic reference suite.

## What exists and what must change

- `validation/runner.py` currently ignores `skip_build`, registers only
  `StaticManifestGrader`, and always reports level 1.
- `Subject` carries a manifest and running subject, but no replay plan or collected
  runtime evidence. Registry selection orders by name, not dependency graph.
- `ValidationProvider.start()` cannot receive resource limits; `RunningSubject`
  only exposes `base_url`, `exec` and `stop`. Neither interface describes effective
  policy, agent identity, metrics, fresh launches or timeout cleanup.
- The core `LocalDockerProvider` does not enforce the needed launch settings.
  Passing additional keyword arguments to it does not configure Docker security.
- No normalized runtime action fixture, session-bound rubric/attribution telemetry,
  or environment-emitted trajectory contract exists.
- The existing static fixtures contain declarations, not runnable subjects. The
  existing echo image uses a mutable base tag. There is no committed root
  `uv.lock`; the ordinary test workflow excludes Docker/network/integration tests.

Use the validation-specific provider protocol rather than widening the core
provider ABC. Reuse core transport and serialization contracts. Keep protocol and
Docker details out of graders.

## Architecture

```text
pure parser + data-only probe plan
              |
      validated RuntimePlan
              |
 runner: dependency plan -> build -> provider preflight -> start/readiness
              |
 session collector: reset / actions / state / discovery / telemetry
              |
 immutable RuntimeEvidence + independently inspected provider evidence
              |
 graders: manifest + evidence -> CheckResult
              |
 policy -> existing report + provenance/artifact bundle -> teardown verification
```

One collector owns the normal episode sequence. Graders consume immutable evidence
instead of independently stepping a shared environment in an order-dependent way.
Fresh-session and fresh-container replay are explicit experiments. Invasive
containment and resource probes get separate, disposable subjects.

`RuntimePlan` is not a second capability manifest. Rewards, resources, network
policy, execution binding, observable capabilities, applicability and type selection
remain authoritative in the normalized manifest. The plan supplies bounded actions,
reset inputs and replay schedules only; it cannot suppress a grader or override
policy. Missing actions produce an unmet prerequisite, never silent deselection.
Only collector dispatch uses the execution binding; graders receive manifest
declarations and measured evidence. Include the plan hash in the run bundle so a
replay is tied to those exact inputs.

Package code runs only inside the subject. Pure parsers do not import it; neither
does the host validator. A trusted probe worker may use the existing OpenEnv
protocol from an isolated helper when a host URL is unavailable. This is an
internal validation transport, not a new public environment API.

## PR 1: settle the missing contracts

Amend RFC 008 narrowly before implementing new abstractions or core wire fields.
Do not create a parallel validation architecture. The following are recommended
decisions for that review, with schemas and small fixtures making them concrete.

| Decision | Proposed contract |
|---|---|
| Runtime input | A versioned, data-only `validation/runtime.json` plan: bounded reset arguments, action sequence and seed/replay schedule. Normalize execution binding, agent boundary and observable capabilities into the manifest, not this sidecar. A trusted parser normalizes inputs into `RuntimePlan`; graders never inspect package signature. No arbitrary host callbacks, guessed legal actions, or reuse of privileged L3 oracle actions. |
| Schema compatibility | Add the small execution/probe declaration in manifest schema v2 and version the report embedding it as v2. Preserve v1 schemas/models/fixtures and static-report compatibility; provide explicit parsing/migration tests. Attach `RuntimePlan` and collected evidence internally to `Subject`; keep expanded provenance and coverage in a separate versioned sidecar. Never silently change schema 1 or put capability selection in the sidecar to avoid versioning. |
| Startup outcome | Add `runtime.startup` in severity policy v2. Failed subject build/start/readiness is FAIL with the failed phase; a validator/provider defect is ERROR. Missing prerequisites are named SKIPs. One successful build does not pass `static.reproducible_build`. |
| Policy versions | Static validation retains v1 support. Runtime requires the policy version defining its new lifecycle outcome; reject an incompatible explicit policy before execution. Make v2 the documented runtime default and pin it in all fixtures. Never emit an unknown ID into v1. |
| Launch and evidence | Validation-only typed launch specification: immutable image identity, manifest resource/network settings, explicit environment variables, effective execution identity, writable roots, independent deadlines. Provider returns inspected settings, verified capabilities, bounded metrics/logs and cleanup evidence. |
| Seed control | Report framework-observed acceptance of the seed/reset contract separately from empirical replay determinism. A successful reset alone is insufficient because current server filtering can silently drop kwargs. A deterministic environment is allowed to produce the same output for different seeds. |
| Reward and comparison | Check raw types before coercion. Reject bool/non-finite/out-of-range rewards; explicitly specify when reset reward may be null. Compare full episode content with only policy-owned volatile metadata exclusions. Authors cannot exclude reward, done, tool output or state changes. |
| Session telemetry | Orchestrator-only, typed telemetry for seed handling, named rubric/configuration, child reward attribution and an emitted trajectory reference. Use the existing replay connection with an opt-in, per-run/session scoped authorization capability; do not open a second environment connection. Reject unauthorized and cross-session reads, and expose none of it as agent MCP tools. A validator transcript is not evidence that the subject emitted a record. |
| Applicability | Explicit predicates for capabilities/collections; an empty declared tool set must still be checked. Missing subject features, missing provider support and implementation-unavailable checks are distinct inventory reasons. |
| Agent boundary | Declare whether agent access is API-only or includes a process UID/filesystem. Oracle and host containment must test that boundary. Privileged exec identity is not an agent-access test. Missing measurable identity produces a named incomplete outcome. |
| Resources | CPU means an allocation ceiling; memory includes explicit swap behavior. Define `disk_mb` as the aggregate permitted writable subject storage, excluding immutable image layers, and name writable roots. Episode timeout includes descendants and is externally enforced. PID/log/build caps are additional supervisor budgets. |
| Network | Keep `public` egress as the existing default. Define any host/private-address restrictions explicitly. For allowlists, settle DNS, IPv4/IPv6, exact/wildcard hostnames, CIDRs and protocol/port semantics; do not silently weaken a hostname rule into an IP-only guarantee. |

The allowlist recommendation is to specify a DNS-derived destination-address
policy, with controlled resolution and recorded address sets, explicitly disclosing
shared-IP limitations. If the RFC instead requires application-host identity,
PR 9 must use protocol-aware enforcement and reject unsupported protocols. This
decision must be made in PR 1; a permissive fallback is not an implementation.

For judged rewards, specify a policy-owned repeated-sample procedure, its units,
sample count and total budget. A proposed default is 20 identical-input replays
and empirical reward variance, compared with the declared bound. This is a bounded
runtime check, not a confidence claim or Level 4 statistical evaluation. Fewer
completed samples cannot silently count as a passing full check.

## Stacked delivery: ten reviewable PRs

Use `ben/rfc008-l2-01-contracts` through `ben/rfc008-l2-10-episode-oracle`
as proposed branch names. Each PR targets its immediate predecessor. Merge
bottom-up, refresh the next dependency edge, and re-run checks on the resulting
exact head. Keep at most two or three unmerged implementation slices in active
review rather than opening the entire stack immediately.

| PR | Incremental change | Independent acceptance gate |
|---|---|---|
| 1. Contracts and shared assets | RFC amendments above; `RuntimePlan`/launch/evidence types; policy v2; scenario catalog; small fake provider and fixture skeleton; schema compatibility. | Pure contract tests and the acceptance inventory run without Docker. Expected findings are written from the RFC, not generated by graders. |
| 2. Docker supervisor and reproducible build | Runnable shared fixture; pinned test project/wheelhouse; build snapshot; build/start/inspect/exec/stop; safe launch defaults, deadlines and idempotent cleanup; initial Linux CI lane. | Actual exact-head wheel and fixture image start, answer health and protocol probes, execute bounded commands and leave no run-owned resources on success, failure or cancellation. |
| 3. Runner and basic checks | Dependency scheduler, strict session collector, runtime CLI path, startup outcome; reward, observation and state graders. | Public CLI returns levels 1 and 2; good fixture passes these checks; malformed-wire, invalid-reward, wrong-state and failed-start cases produce expected IDs. No Docker calls under `--skip-build`. |
| 4. Session telemetry | Small reviewed core protocol additions for seed acceptance, rubric/configuration, attribution and subject-emitted record metadata. | Real session tests preserve the replayed instance; unauthorized and cross-session reads fail; records are independent from the collector transcript; production MCP cannot invoke telemetry/reset controls. Existing clients remain compatible. |
| 5. Repeatability and trajectories | `seed_control`, `episode_determinism`, `trajectory_record`; fixed-input fresh-session/fresh-container experiments; bounded variance mode. | Deterministic and seed-invariant cases pass; ignored-seed stochastic fixture, trace mismatch and nondeterminism fail; first divergent operation/path is reported. |
| 6. Discovery and rubric checks | `tool_declaration_accuracy`, `task_declaration_accuracy`, `rubric_introspectable`, `reward_attribution`. | Empty/extra/missing tools, discovery failure, wrong task counts, bounded task previews, missing rubric and inconsistent attribution are distinguishable. |
| 7. Host containment and resources | `host_containment`, `resource_bounds`; externally inspected settings, tested agent identity, bounded canaries and quota/deadline evidence. | Benign host sentinel inaccessible; missing/misconfigured limits detected; bounded stress contained; timeout kills descendants; unsupported quota backend cannot pass disk enforcement. |
| 8. Public/no-network modes | `network_policy` for these modes; trusted control transport; controlled reachable/blocked sinks. | Runtime communication works with network isolation; public positive connectivity succeeds; no-network has no egress; a control helper cannot bridge subject egress. |
| 9. Allowlist mode | Dedicated enforcement backend and qualification matrix for reviewed hostname/CIDR/DNS/IPv6 semantics. | Allowed destinations succeed; prohibited names, direct-IP and alternate-DNS/bypass cases fail as specified. Unsupported policy is refused before launch. Separate network/security review. |
| 10. Episode and oracle boundaries | `episode_isolation`, `oracle_containment`; agent-access probes, same-server A/reset/B experiments; final documentation and reference evidence. | Sticky memory/file/rubric/background-state cases and exposed harmless oracle canary are detected. Full reference inventory has all 15 original IDs plus startup, with no missing expected checks. |

Security settings belong in PR 2 even though their dedicated graders arrive later.
The provider initially advertises only modes it can actually enforce. No-network
and allowlist requests must not be launched under public networking while their
implementation is pending.

Every PR adds its fault cases, useful tests and reproduction instructions with the
behavior it implements. The required CI inventory expands immediately. During the
stack, expose unimplemented requested checks as explicit incomplete inventory and
named SKIPs rather than presenting a partial runtime run as complete. Preserve RFC
exit semantics: WARN remains exit 0. The test harness independently enforces the
expected check inventory.

Review routing: Sergio as proposed architecture/core-contract lead; a named Docker
and networking reviewer for PRs 2 and 7–9; an environment/rubric maintainer for PRs
4–6 and 10. HF Sandbox-specific review belongs to the subsequent provider stack.
These are suggested review roles, not assigned or requested reviews.

## One shared runtime fixture and scenario catalog

Use a tiny CPU-only served fixture with a seeded counter/task selector, two tools,
two fixed task splits, a non-LLM rubric, bounded rewards, finite termination and a
harmless withheld oracle canary. Add a controlled stochastic judge mode to exercise
variance logic. All runtime actions are public, bounded JSON data.

Fault modes live only in test assets. Each mode introduces one defect into a fresh
subject; no production bypass flags. Reuse the same image and collector across
unit, protocol and Docker tests. Preserve the existing manifest-only static
fixtures, and use `echo_env` as an additional compatibility smoke rather than the
reproducibility anchor.

Proposed shared layout:

```text
tests/fixtures/validation/runtime/
  README.md
  served_probe/                 # runnable package, Dockerfile, public runtime plan
  cases.json                   # check -> case -> expected status/evidence predicate
  expected/                    # small semantic expectations, never full noisy logs
tests/test_validation/
  support/                     # fake provider, evidence builders, case loader
  runtime/                     # grader and orchestration tests
  integration/                 # real protocol, Docker, installed-wheel tests
tests/validation_runtime/
  pyproject.toml
  uv.lock
  .python-version
  toolchain.json               # uv/Python/image/platform/build dependency pins
scripts/validation/
  reproduce.py                 # shared entry point for developers and CI
  verify_artifacts.py
.github/workflows/validation-runtime.yml
```

The case catalog also declares applicability, required provider features, the PR
that implements the case and whether its test needs Docker. A missing case or an
unexpected skip fails the relevant CI suite; expected subject validation failures
are successful tests only when their exact findings and evidence match.

## Check-by-check acceptance matrix

| Existing Level 2 ID | Measurement and essential negative case |
|---|---|
| `runtime.reward_well_formed` | Raw reward type/finite/range checks under the explicit null rule; bool, string, NaN/Inf and out-of-range faults. |
| `runtime.observation_schema` | Strict envelope checks, then reconstruct full observation including reward/done for advertised-schema validation; missing/wrong-typed fields fail. |
| `runtime.state_contract` | Minimum state contract: requested episode identity, reset count and coherent step increments within one session; wrong episode or sticky counter fails. |
| `runtime.trajectory_record` | Fetch the subject-emitted record and compare with independently captured actions/results; missing, truncated or mismatched record fails. |
| `runtime.reward_attribution` | Match child scores to declared rubric structure and aggregation semantics; absent/misidentified/inconsistent attribution fails. Do not assume every rubric is an unweighted sum. |
| `runtime.rubric_introspectable` | Session-correct named tree and serializable configuration when claimed; declared-but-uninspectable rubric fails its warning-level check. |
| `runtime.tool_declaration_accuracy` | Compare normalized declared/discovered sets, including empty declaration; extra/missing tools fail, discovery error cannot become an empty success. |
| `runtime.task_declaration_accuracy` | Compare declared split counts with `num_tasks`; use bounded item sampling, never `len(list_tasks)` as the count; missing split/wrong count fails. |
| `runtime.seed_control` | Framework-observed seed contract plus scheduled resets; rejected/silently discarded seed in a seed-dependent fixture is detected. Different seeds need not change a deterministic subject. |
| `runtime.episode_determinism` | Same seed/actions across fresh sessions and containers; compare complete traces with tightly defined volatile fields; injected observation/reward drift fails. Judged path uses its separately pinned variance procedure. |
| `runtime.network_policy` | Effective inspected rules plus positive/negative connectivity from the subject namespace; controlled endpoints establish reachability before denial is interpreted. |
| `runtime.host_containment` | Effective mounts/namespaces/security settings plus a harmless agent-identity sentinel probe; use synthetic unsafe inspection records instead of giving a test real host privilege. |
| `runtime.resource_bounds` | Effective CPU/memory/swap/PID/writable-storage configuration and external time budget; bounded qualification probes detect missing enforcement and surviving children. |
| `runtime.episode_isolation` | In the same server, episode A writes a declared marker, reset starts B, B cannot observe A's state/file/rubric/task effects; fresh containers alone do not establish this property. |
| `runtime.oracle_containment` | Oracle artifact is absent/inaccessible through the declared agent boundary at serve time; a harmless readable canary fails. General solution leakage in observations stays in Level 3. |

For each applicable check test: good input, one subject defect, missing prerequisite,
and malformed evidence/provider error where meaningful. A declared supported
capability that malfunctions is a failure/error, not an opportunistic SKIP.

## Protocol and isolation details that tests must preserve

Use the existing WebSocket session for reset/step/state. Do not compose an episode
out of HTTP reset/step calls, which can instantiate separate environments. Keep raw
responses before generic-client/Pydantic defaults can hide missing fields. The
wire envelope separates reward/done from observation; validate both representations
correctly. The current schema endpoint exposes base State, so check the minimum
state behavior directly.

Do not combine MCP tool-return values with Gym rewards. Discovery failures must
remain failures, even if the ordinary convenience client returns an empty list.
Task discovery may legally expose a bounded preview.

Each current WebSocket connection creates its own environment. Telemetry therefore
extends the already-open replay session, guarded by the reviewed opt-in authorization
mechanism. An independent telemetry connection must not create and inspect a fresh
instance while claiming evidence about the original. Existing clients need not
request or receive these additional messages; agents retain only their MCP boundary.

The provider owns readiness, bounded exec/logs, fresh launches, inspected settings
and idempotent teardown. It never mounts a Docker socket or credentials into the
subject. Use a non-privileged launch, no host namespaces or host-directory mounts,
dropped capabilities, `no-new-privileges`, read-only image and explicitly bounded
writable areas. Exposed control ports bind only to loopback with allocated ports.
Unsupported storage/identity restrictions produce explicit capability evidence;
there is no automatic retry with weaker isolation.

`--network none` leaves only loopback, so publishing port 8000 does not solve
orchestrator access. Implement an isolated trusted probe worker sharing only the
subject network namespace and using the ordinary loopback OpenEnv protocol; the
host controls it over bounded runtime exec. It has its own image/filesystem, no
Docker socket, no externally routed interface and no endpoint through which the
subject can request arbitrary outbound traffic. Review this arrangement in PR 1,
implement it in PR 8, and test that it cannot bridge egress.

An HTTP proxy setting alone does not implement allowlisting. The subject must lack
network-administration authority; trusted default-deny enforcement and DNS policy
remain outside its control. Record provider qualification separately from each
subject's sampled measurements. A blocked connection to an offline endpoint is
not evidence of isolation.

Memory settings must include swap behavior. A tmpfs limit consumes memory and is
not by itself a general filesystem quota; all writable paths must fit the reviewed
aggregate budget, or disk enforcement remains unsupported. Do not infer effective
limits solely from Docker CLI arguments.

Independent deadlines cover dependency acquisition, build, start, readiness,
request, exec, episode and whole run. Killing a Docker exec client does not prove
its child stopped. Terminate the process scope or destroy the subject, then verify
cleanup. Label every resource with a run ID and provide a scoped stale-run reaper;
never use a global Docker prune.

## Reproducible build and run recipe

Pin a dedicated validation test project instead of changing the whole repository's
dependency policy. Commit its lock, Python patch version, uv version, build
dependencies, image digest and reference architecture. Build a wheel from the exact
PR head and install that wheel both in the fixture and in the installed-package
test. Do not fetch `main`, use `:latest`, or depend on an existing editable install.

Acquire dependencies as an explicit stage; retain a hash-checked wheelhouse.
Fixture image installation is network-free after that acquisition. Cache keys
include source wheel, fixture, lock and base-image hashes. A clean cache must work.
Arbitrary subject image builds are separately bounded and their build-network
policy is explicit; runtime network settings do not retroactively isolate builds.

Stage a stable source/build context before hashing and building. Capture generated
wheel/helper inputs as additional digests so the reported source revision actually
matches the tested artifact. Local dirty runs record a patch/content digest;
review evidence uses a clean exact commit. Symlinks or referenced probe paths must
not escape the subject package. Keep generated outputs outside build inputs.

These are **proposed commands**, to be implemented by the stack:

```bash
uv sync --project tests/validation_runtime --frozen

# Pure logic and real local protocol tests; no Docker required.
uv run --project tests/validation_runtime --frozen \
  python scripts/validation/reproduce.py --suite fast
uv run --project tests/validation_runtime --frozen \
  python scripts/validation/reproduce.py --suite protocol

# Build the exact-head wheel and shared image, exercise all required Docker cases,
# repeat from clean subjects, verify check inventory and cleanup, write artifacts.
uv run --project tests/validation_runtime --frozen \
  python scripts/validation/reproduce.py --suite docker \
  --require-complete --output outputs/validation-runtime
```

`--require-complete` is a test-harness flag, not a change to public CLI exit codes.
During development it requires the inventory committed for that PR; the final
reference inventory includes every original L2 ID and the startup outcome.

The Docker suite must invoke the public command on staged packages, with explicit
local selection, runtime level and pinned policy; it must not validate only private
Python helpers. `--local` is part of the RFC but not the current CLI and lands in
PR 3. Runtime tests must not infer remote execution from an ambient HF token.

Each case writes its own standard report. The suite writes:

```text
outputs/validation-runtime/<run-id>/
  run-manifest.json
  coverage.json
  cases.jsonl
  junit.xml
  cases/<case-id>/report.json
  trajectories/                # collector traces and separate emitted records
  logs/                        # bounded, redacted
  cleanup.json
  SHA256SUMS
```

Record head/base SHA, dirty-state digest, exact argv, source/fixture/runtime-plan/
policy/lock/wheel hashes, base digest and resolved image ID, platform/kernel/
cgroup/storage/Docker/Python/uv versions, seeds, replay identities, effective
provider features and all artifact hashes. The final checksum file covers the
other artifacts, not itself. Redact secrets before retention and enforce size
limits, including on malformed output.

Reproducibility means fixed inputs and equivalent findings/traces under the
qualified platform. Do not promise byte-identical images or duration/log equality.
Preserve raw evidence and compare a narrowly defined normalized view; do not erase
meaningful nondeterminism to make golden files pass.

## Testing and CI

1. **Fast, every PR:** scheduler DAG/cycles/missing dependencies, capability and
   applicability decisions, skip-build, unexpected grader IDs, exceptions,
   cancellation/timeout/cleanup, policy and schema compatibility. Fakes test
   orchestration behavior; they do not establish Docker enforcement.
2. **Real protocol, relevant PRs:** a real test server and supported production
   transport, raw serialization faults, session identity, discovery and telemetry.
   Prefer protocol faults over monkeypatches that skip the actual wire path.
3. **Required Linux Docker, starting PR 2:** exact-head wheel/image, all implemented
   positive and single-fault cases, enforced limits/network modes, two clean repeat
   runs, no resource leaks. Missing Docker or required platform capability fails
   preflight; it must not silently skip the merge gate.
4. **Installed-package compatibility:** run outside the checkout with `PYTHONPATH`
   unset; verify schemas, policy files and trusted helpers are included in the
   wheel. Keep the existing suite and add an `echo_env` smoke without claiming it
   covers every optional capability.

Use ephemeral Linux CI workers, read-only repository permissions, checkout without
persisted credentials, ordinary PR events, and no publish tokens. Do not run fork
code through a privileged `pull_request_target` path or approve fork workflows
automatically. Keep the required workflow visible on every PR; an internal path
filter can decide whether to run expensive cases. Relevant core protocol/provider
changes must trigger them too.

Always retain the evidence bundle on failures. Explicitly distinguish expected
subject FAIL fixtures from harness/test failures. CI verifies the expected ID set,
statuses, applicability and evidence; CLI exit zero alone is insufficient because
WARN also exits zero. Test CI for this feature is within this plan; enforcing
validation on user publication remains outside it.

Initial performance targets, to measure rather than claim as achieved: fast suite
under one minute, protocol suite under two minutes, warm-cache reference Docker
suite under ten minutes. Give the Docker job a separate hard deadline and preserve
partial artifacts on timeout. Cold dependency acquisition/build time is recorded
separately; do not hide it in a warm-cache timing claim.

## Review and completion gates

Each PR has one short purpose, incremental diff against its parent, focused files
and an acceptance case in the shared catalog. Keep one stack checklist mapping
check IDs to PRs and evidence. Put detailed run recipes and evidence in the shared
assets/CI artifacts; follow the repository's short PR-description convention.
Avoid copied fixture directories and generated multi-thousand-line snapshots.

Review three waves: (1) contracts, provider and first real CLI path; (2) telemetry,
replay and discovery; (3) isolation, network enforcement and complete coverage.
Security review remains required even when the fixture suite passes.

Completion requires all of the following:

- All 15 existing L2 checks implemented, plus the approved startup outcome.
- A reference fixture declaring all optional features exercises every check; each
  check has an independently specified positive and targeted negative case.
- Required reference cases have no unexplained SKIP/ERROR, omitted check or missing
  evidence. Unsupported platforms/features report their actual limitations.
- Build/readiness failures, stuck requests, cancellation and process descendants
  leave correct reports and no run-owned resources.
- Two clean runs from the same commit and pins produce equivalent findings and
  deterministic traces, with recorded differences limited to approved metadata.
- Installed-wheel tests and the existing validation tests pass; schemas and package
  data are verified, not just editable-source imports.
- Documentation distinguishes tested runtime behavior from Level 3 semantics,
  statistical trainability and broad security certification.

## Sources

- [RFC 008 at the planning revision](https://github.com/huggingface/OpenEnv/blob/e3eb3fa5bc11ff0a019c30c7a7b1b7039eb8b615/rfcs/008-environment-auto-validation.md)
- [Validation provider contract](https://github.com/huggingface/OpenEnv/blob/e3eb3fa5bc11ff0a019c30c7a7b1b7039eb8b615/src/openenv/validation/providers/__init__.py)
- [Grader/Subject contract](https://github.com/huggingface/OpenEnv/blob/e3eb3fa5bc11ff0a019c30c7a7b1b7039eb8b615/src/openenv/validation/graders/__init__.py)
- [Core Docker provider](https://github.com/huggingface/OpenEnv/blob/e3eb3fa5bc11ff0a019c30c7a7b1b7039eb8b615/src/openenv/core/containers/runtime/providers.py)
- [Server transport and schemas](https://github.com/huggingface/OpenEnv/blob/e3eb3fa5bc11ff0a019c30c7a7b1b7039eb8b615/src/openenv/core/env_server/http_server.py)
- [Task API and bounded previews](https://github.com/huggingface/OpenEnv/blob/e3eb3fa5bc11ff0a019c30c7a7b1b7039eb8b615/docs/source/guides/task-api.md)
- [Docker network-none semantics](https://docs.docker.com/engine/network/drivers/none/)
- [Docker resource constraints](https://docs.docker.com/engine/containers/resource_constraints/)
- [Docker tmpfs behavior](https://docs.docker.com/engine/storage/tmpfs/)
- [Docker build and pinning guidance](https://docs.docker.com/build/building/best-practices/)
