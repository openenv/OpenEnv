# RFC: Environment Auto-Validation

**Status**: In Review
**Created**: 2026-08-04
**Authors**: @zkwentz
**RFC ID**: 008

## Summary

OpenEnv's hub hosts 4,300+ environments; manual review cannot scale and demonstrably misses real
defect classes — verifiers awarding max reward to an empty solution, correct solutions scoring
below max, answers leaked in observations, nondeterministic rewards. This RFC specifies
**environment auto-validation**: a single local command (`openenv validate`) that grades an
environment or task package against the quality bar defined in
[#778](https://github.com/huggingface/openenv/issues/778), plus the **contracts** — signature
detection, parser interface, normalized manifest schema, grader registry, report schema, versioned
severity policy, provider interface — that let any operator build a validation hub on top.

The central architectural commitment: **ship local validation and the contracts that let any team
build a hub; do not ship a hub.** There will be many hubs (Hugging Face, private lab intake
systems, future operators); a first-pass hub implementation in the shared repo would quietly make
one operator's answers the standard. The repo therefore ships no runner, queue, submission API, or
gate policy — only the pieces every operator needs to agree on for results to be comparable.

Validation says an environment *can* be trained on, not that it *should* be. It certifies
well-formedness and runnability, not research value.

## Motivation

### Problem statement

An environment is only useful for improving models if a frontier-lab intake would accept it. #778
defines that bar as six properties:

1. **Scales on infra** — builds reproducibly, delivers efficiently.
2. **Learnable** — a model can hill-climb on it within O(100s) of steps.
3. **Secure** — model-generated actions stay contained.
4. **Not prone to reward hacking** — the reward means what it claims.
5. **Observable & self-describing** — observations, state, tools, and tasks can be inspected and
   match what the environment declares.
6. **Reproducible** — same inputs yield same outcomes across runs, hosts, and time.

These decompose into 44 acceptance tests (#778). Today, `openenv validate` covers a small
structural subset, `openenv push` runs a *different, divergent* check as warnings only, nothing is
gated, and no manifest schema exists. Meanwhile the defect classes above ship to the hub
regularly — one submitted verifier returned reward 1.0 for the empty string.

A second, converging problem: task packages. PostTrain/BenchFlow packages (`task.md`) and Harbor
tasks (`task.toml`, already consumed by `envs/tbench2_env`) are one-shot lifecycles — agent runs
to completion, verifier reads artifacts — that need the same validation bar as served OpenEnv
environments, without pretending to be served environments.
[#898](https://github.com/huggingface/openenv/issues/898) is that lane; this RFC subsumes it under
one mechanism (the grader registry, below) rather than a parallel validation core.

### Goals

- Give authors a fast, honest local loop: `openenv validate .` with actionable red-to-green
  remediation, seconds without a build, minutes with one.
- Catch the motivating defect classes mechanically, before a human ever reviews.
- Define the contracts so that a local run and an operator's hub run produce comparable,
  reproducible verdicts.
- Support served environments and one-shot task packages through one command and one grader set.

### Non-goals

- No hub machinery: no runner, queue, coordinator, submission/auth API, result storage, or gate
  policy. Operators own these.
- No statistical-level implementation (level 4 below): those checks need reference models, many
  tasks, and inference compute that authors do not have. The repo reserves their check ids and
  report fields; operators run them.
- No grandfathering/grace policy for existing environments — that is operator policy, and the repo
  holds no opinion (this resolves #778's open question 2 by construction).
- No measurement of research value or training transfer.

## Design

### Architecture overview: four levels, three budgets

Validation cost divides along who runs it:

| Budget | Where | Cost | Levels |
|---|---|---|---|
| Author | laptop / CI | seconds–minutes, no inference | 1–3 |
| Hub gate | operator compute | minutes per submission | 1–3 + hub lane |
| Lab intake | operator compute | hours, inference, all data | + 4 |

The four levels:

1. **Static** — manifest schema, pinned dependencies, reproducible build, SBOM, OCI labels,
   resource/timeout/task-distribution declarations.
2. **Runtime** — the subject starts; seed and episode determinism; declaration accuracy (tools,
   tasks); reward well-formedness; declared network policy enforced; host/filesystem containment;
   resource bounds;
   episode isolation. **Security lives here.**
3. **Semantic** — oracle replay reaches max reward; the measured do-nothing floor sits a margin
   below max; no solution leakage in observations; verifier determinism and portability; canary
   trajectories stay at the floor; replayability.
4. **Statistical** — pass@k band, difficulty separation across reference-model tiers, headroom,
   reward-variance bound, adversarial panel. **Reserved: operator-side only.**

`openenv validate` runs levels 1–3 locally:

```
signature detection → parser → normalized manifest
    → static graders (L1, build optional)
    → provider.start(network=declared policy) → runtime graders (L2)
    → oracle/floor replay + semantic graders (L3)
    → severity policy → report (JSON) → exit code
```

Levels run in order and accumulate into **one report per run** — a level-1 failure does not stop
level-2 checks whose dependencies still hold, so an author gets maximum information per run.

### The local/remote boundary

**The repo ships (this RFC + its PR stack):**

- Signature detection rules and the parser interface
- The normalized manifest schema (capabilities spec + type spec)
- The grader interface and registry (third parties add graders with zero core changes, via the
  `openenv.validation.graders` entry-point group)
- The report JSON Schema and the versioned severity policy (v1)
- The validation provider interface, with Docker-local and HF Sandbox implementations
- The explicit unsupported-categories list (below)
- `openenv validate`, levels 1–3

**Operators ship:** auth and submission APIs, queues/scheduling/retries, compute and its cost,
result storage, gate policy, and every statistical-level check. Statistical checks run where data
and compute live — hub or lab, with a hub service preferred — unified by a common metric namespace
(`pass@k · headroom · variance_bound`) so results compare across operators. (The metric-namespace
specification is deferred to a future RFC.)

Hugging Face appears on both sides of this line in two distinct roles, and the design keeps them
distinct: HF-as-hub is an operator like any other and gets no special machinery in the repo;
HF-as-sandbox-provider is a local primitive behind the provider interface, alongside Docker-local.

### Core abstraction: the grader registry

Format and type are different axes. One `task.md` package may hold a SWE, cyber, or browsing
task — same signature, so detection alone cannot select graders. The flow:

```
SIGNATURE selects the parser
  → PARSER writes capability + type fields into ONE normalized manifest
    → CAPABILITIES select contract graders (oracle replay, floor gap, determinism, network policy …)
    → TYPE tags select domain graders (SWE compiles the test suite; browsing opens pages …)
```

**The registry rule: graders read the manifest, never the signature.** Write a grader once and it
works with every format whose parser supplies the fields. `openenv.yaml` (served), `task.toml`
(Harbor), and `task.md` (PostTrain) are three parsers under one registry — adding a format is only
a parser; adding a check is only a grader. The manifest's `signature` field is provenance for the
report and nothing else; no grader, core or third-party, may branch on it.

Signature detection is well-known-file detection **over the formats whose parsers ship in the
build**: `openenv.yaml` → served OpenEnv, then `task.toml` → Harbor and `task.md` (with
frontmatter) → PostTrain as their parsers land. The detection table lists only implemented
parsers — validation never claims support for a format it cannot parse; a package carrying none
of the supported well-known files is refused as unrecognized (exit 2). Exactly one entry must
match; two+ is a hard "ambiguous package" failure — **never a guess**. Parsers are pure reads:
they never import or execute package code.

Type tags are multi-tag: bare tags are shared (`swe`), prefixed tags are hub-scoped
(`hf:agentic-swe`); pass@k comparisons are valid only within a tag. The shared v1 tag set is
seeded from the environments already in `envs/`. Domain graders themselves are post-v1; the
selection axis ships now.

### The oracle and the semantic contract

The core promise is #778's verifier-sanity test, made precise:

- **No oracle → FAIL.** No skip, no warn: "an environment is only good if verifiable." An
  environment that cannot demonstrate its verifier awards max reward to a correct solution does
  not validate. This is a graded failure (it appears in the report with remediation), not a parse
  error.
- **The oracle admits two forms.** *Injected state* (preferred — most deterministic): the package
  declares the state a correct trajectory would produce, and validation sets it directly.
  Set-state is a declared per-environment(+provider) capability, not a requirement. *Executable
  oracle script* (the Harbor `solution/solve.sh` precedent): a privileged script executed inside
  the sandbox drives the environment to the solved state. Either form must drive the verifier to
  max reward.
- **Floor is measured, not declared.** The floor is the measured reward of the reset/do-nothing
  state — the declared range minimum may sit below it (an agent that deletes the repo earns a
  legal sub-floor reward, not a validation anomaly). The contract is a **gap check**:
  `reward(oracle) ≥ max − tolerance`, `reward(reset) = floor`, `max − floor ≥ margin`.
- **Oracle containment is required.** The oracle (state or script) must be withheld from or
  isolated against the agent at serve time, and validation checks that it is.
- **Reward range is declared** in the manifest, defaulting to `[0, 1]`; graders normalize against
  it.
- **Determinism is a precondition of semantics.** Level-2 episode determinism is a declared
  dependency of every level-3 semantic check; a nondeterministic environment gets its semantic
  checks SKIPped with the dependency named, not silently passed or misleadingly failed.

**LLM-as-judge is an allowed reward.** Judged environments declare `llm_judged` and pin the judge
configuration (model, version, params) in the manifest; the oracle check becomes
`≥ max − declared tolerance` and determinism becomes variance-within-declared-bound rather than
bit-exact. RFC 004 rubrics are **leveraged, not required**: the contract stays spec-neutral
(graders read the manifest), but for the served OpenEnv format the rubric tree is the native
satisfaction path — `LLMJudge` is the in-repo `llm_judged` implementation, and the
introspectability and reward-attribution graders read `named_rubrics()` / `state_dict()` /
per-child scores. Judge pinning stays a *manifest* declaration because the rubric object does not
serialize model/version/params today.

Tolerances, margins, and variance bounds are author-declared in the manifest, **bounded by the
versioned severity policy**, and carried verbatim in reports so hubs can apply stricter ceilings.

### Report and severity policy

Every check emits a status (`pass` / `fail` / `skip` / `error`) plus measured values, evidence,
and remediation. **Graders have no opinion about whether their failure blocks validation** — only
the severity policy, a versioned data artifact (`severity-v1`), maps a check id to
`fail` / `warn` / `advisory` and produces the verdict. Pin a policy version and a local run
reproduces the hub verdict, **modulo hub-lane checks**.

Checks carry a lane. **Local reports contain only local-lane check ids** — an author is never
shown a check they cannot red-to-green (cross-host reproducibility and immutable versioning, for
example, are inherently operator-side). Hub and statistical check ids are *reserved* in the report
schema and the policy's hub lane so that operator reports and local reports share one schema.

The report is JSON, schema-versioned, and embeds the normalized manifest verbatim plus a source
digest. The normative JSON Schemas (`manifest.schema.json`, `report.schema.json`) land in the
contracts PR at `src/openenv/validation/schemas/` as committed exports of the Pydantic models,
with a CI sync check.

CLI contract:

```
openenv validate [TARGET]
    [--level {static,runtime,semantic}]   # ceiling; default semantic
    [--skip-build]                        # inner loop; build-dependent checks SKIP with reason
    [--remote | --local | --always-remote]
    [--policy <version>]                  # default v1
    [--json | --output <path>]
```

Exit codes: `0` verdict PASS/WARN · `1` verdict FAIL · `2` ambiguous/unrecognized/unsupported
package · `3` internal error.

### Providers

Runtime and semantic checks need a sandbox with capabilities beyond start/stop: **network-policy
enforcement** and **in-sandbox exec** (oracle scripts, containment probes). Validation defines its
own provider protocol with declared capabilities and adapts the existing core providers rather
than widening the core ABCs. v1 ships **Docker-local** and **HF Sandbox**. A check whose required
provider capability is absent SKIPs with the capability named.

Network posture follows the Harbor `task.toml` (schema 1.4) precedent: the manifest declares a
network policy — mode `public` (the default; egress allowed), `no-network`, or `allowlist` with
`allowed_hosts` (exact hostnames, CIDR ranges, wildcards) — and the subject starts under that
declared policy. The `runtime.network_policy` check verifies the sandbox's effective network
access matches the declaration. Verifier hermeticity (#40) is unchanged: that check runs the
verifier with network denied regardless of the environment's declared policy.

GPU is a provider capability, not a package property to reject: a package declaring GPU
resources validates on a sandbox provider that offers GPUs (e.g. a remote sandbox); on a provider
without them, runtime+ checks SKIP with the capability named.

Provider selection is auto-detected from environment keys (e.g. an HF token suggests hf-sandbox),
with a first-run remote-vs-local confirmation, an always-remote flag, and Docker-local as the
fallback.

### Unsupported package categories

Local validation recognizes but does not attempt the following categories. Detection, where
possible at parse time, produces an explicit "unsupported" outcome (exit 2) with the category and
reason — recognized, never guessed at:

| Category | Reason local validation cannot run it |
|---|---|
| `hosted-verifier` | The verifier calls an external service; the run cannot be hermetic, and verifier portability (#40) is unmeasurable locally. Operator-side validation with declared endpoints may support it. |
| `multi-agent` | Multi-agent scenes need an orchestration harness local validation does not define. |
| `simulated-user` | Requires a user-simulator policy; validating the simulator is its own problem and no local reference exists. |
| `parser-not-implemented` | Defensive registry contract: a signature kind reaching dispatch without a registered parser is refused, never guessed at. Unreachable in normal operation, since detection only recognizes formats with implemented parsers. |

### The 44-test mapping

Every #778 acceptance test is assigned a level, a lane, a v1 severity, and a stable check id
(policy-addressed; reserved ids exist in the report schema and the policy's hub lane but are never
implemented or reported locally). Approved 2026-08-04.

#### Scales on infra — build & delivery

| # | Check | Level | Lane | Severity | Check id |
|---|---|---|---|---|---|
| 1 | Reproducible build | 1 | local | fail | `static.reproducible_build` |
| 2 | Layer-change isolation | 1 | hub | warn | `hub.layer_isolation` *(reserved)* |
| 3 | Multi-stage hygiene | 1 | local | warn | `static.image_hygiene` |
| 4 | Archive-free layout | 1 | local | warn | `static.layout` |
| 5 | Conversion clean (estargz/nydus) | — | out | — | dropped from v1 |
| 6 | Time-to-first-useful-work | 2 | hub | warn | `hub.time_to_first_work` *(reserved)* |
| 7 | Composition inspection | — | out | — | dropped from v1 |
| 8 | Signature + SBOM | 1 | local (SBOM) / hub (cosign) | SBOM fail; cosign warn (hub) | `static.sbom`; `hub.cosign_signature` *(reserved)* |
| 9 | OCI labels | 1 | local | fail | `static.oci_labels` |

Signing is a publish-time act — an author cannot fail cosign before publishing, so it is hub-lane
by the red-to-green rule.

#### Resources

| # | Check | Level | Lane | Severity | Check id |
|---|---|---|---|---|---|
| 10 | Resource declaration | 1 | local | fail | `static.resource_declaration` |
| 11 | Measured envelope | 3 | local | fail | `semantic.resource_envelope` |
| 12 | Timeout ceiling | 1 | local | fail | `static.timeout_ceiling` |

The measured envelope piggybacks the oracle-replay run, as #778 anticipates.

#### Learnable — all statistical, all operator-side

| # | Check | Level | Lane | Severity | Check id |
|---|---|---|---|---|---|
| 13 | Reward reachability | 4 | hub/lab | fail | `statistical.reward_reachability` *(reserved)* |
| 14 | Difficulty separation | 4 | hub/lab | fail | `statistical.difficulty_separation` *(reserved)* |
| 15 | Headroom | 4 | hub/lab | warn | `statistical.headroom` *(reserved)* |
| 16 | Reward signal-to-noise | 4 | hub/lab | fail | `statistical.variance_bound` *(reserved)* |
| 17 | Improvement signal | 4 | hub/lab | advisory | `statistical.training_signal` *(reserved)* |

#### Secure — all runtime, all local

| # | Check | Level | Lane | Severity | Check id |
|---|---|---|---|---|---|
| 18 | Declared network policy enforced (public / no-network / allowlist) | 2 | local | fail | `runtime.network_policy` |
| 19 | Filesystem / host containment | 2 | local | fail | `runtime.host_containment` |
| 20 | Resource bounds per episode | 2 | local | fail | `runtime.resource_bounds` |
| 21 | Cross-episode isolation | 2 | local | fail | `runtime.episode_isolation` |
| 22 | Reward / ground-truth containment | 2–3 | local | fail | `runtime.oracle_containment` |

#### Not prone to reward hacking

| # | Check | Level | Lane | Severity | Check id |
|---|---|---|---|---|---|
| 23 | Well-formed reward | 2 | local | fail | `runtime.reward_well_formed` |
| 24 | Rubric introspectability | 2 | local | warn | `runtime.rubric_introspectable` |
| 25 | Verifier sanity (oracle → max; floor gap) | 3 | local | fail | `semantic.oracle_max` + `semantic.floor_gap` |
| 26 | Adversarial floor | 4 | hub/lab | fail | `statistical.adversarial_floor` *(reserved)* |
| 27 | Gameability gap | 4 | hub/lab | warn | `statistical.gameability_gap` *(reserved)* |
| 28 | Canary suite (package-shipped) | 3 | local | fail | `semantic.canary_floor` |

Test #25 splits into two check ids — one measurement each, better diagnostics; the v1 policy fails
both. Test #26's adversarial-panel procedure is not yet specified anywhere; defining the panel is
future operator-side work, and the reserved check id is the only artifact this RFC commits to.
Test #28's canary trajectories are cheap local replays when shipped in the package; a hub may
additionally maintain its own canary corpus operator-side.

#### Observable & self-describing

| # | Check | Level | Lane | Severity | Check id |
|---|---|---|---|---|---|
| 29 | Observation schema conformance | 2 | local | fail | `runtime.observation_schema` |
| 30 | No solution leakage | 3 | local | fail | `semantic.no_solution_leakage` |
| 31 | `state()` contract (episode_id, step_count) | 2 | local | fail | `runtime.state_contract` |
| 32 | Trajectory record emitted | 2 | local | warn | `runtime.trajectory_record` |
| 33 | Reward attribution | 2 | local | warn | `runtime.reward_attribution` |
| 34 | Tool declaration accuracy | 2 | local | fail | `runtime.tool_declaration_accuracy` |
| 35 | Task declaration accuracy | 2 | local | fail | `runtime.task_declaration_accuracy` |

#### Reproducible

| # | Check | Level | Lane | Severity | Check id |
|---|---|---|---|---|---|
| 36 | Seed control | 2 | local | fail | `runtime.seed_control` |
| 37 | Episode determinism | 2 | local | fail | `runtime.episode_determinism` |
| 38 | Cross-host reproducibility | 2 | hub | fail | `hub.cross_host_determinism` *(reserved)* |
| 39 | Verifier determinism | 3 | local | fail | `semantic.verifier_determinism` |
| 40 | Verifier portability | 3 | local | fail | `semantic.verifier_portability` |
| 41 | Dependency pinning | 1 | local | fail | `static.dependency_pinning` |
| 42 | Task-distribution pinning | 1 | local | fail | `static.task_distribution_pinning` |
| 43 | Immutable versioning | 1 | hub | fail | `hub.immutable_versioning` *(reserved)* |
| 44 | Replayability | 3 | local | fail | `semantic.replayability` |

One check id has no #778 number: `static.manifest` (level 1, local, fail) — the manifest-schema
check that underpins #10, #12, and #42 and is the pipeline's first grader.

**Rollup:** local `openenv validate` covers 33 of 44 tests (levels 1–3); operators own 9 (all five
learnable tests, adversarial floor, gameability gap, cross-host reproducibility, immutable
versioning); 2 are dropped (estargz/nydus conversion — a step too far for v1). v1 severities over
the 44 tests: 32 fail, 9 warn, 1 advisory (plus the 2 dropped). As check ids, the committed
severity policy carries 45 entries — 34 fail / 10 warn / 1 advisory — because test #25 splits into
two fail ids, test #8 splits into a local fail (`static.sbom`) and a hub warn
(`hub.cosign_signature`), and `static.manifest` carries no #778 number.

## Examples

Author inner loop:

```bash
# Fast: static checks without a build (seconds)
openenv validate . --level static --skip-build

# Full local validation: static + runtime + semantic (minutes, includes build)
openenv validate .

# Machine-readable report for CI
openenv validate . --json > validation-report.json
```

Manifest declarations (illustrative; the normative schema is
`src/openenv/validation/schemas/manifest.schema.json`):

```yaml
# openenv.yaml (validation-relevant fields)
name: my-swe-env
validation:
  reward:
    range: [0.0, 1.0]
    oracle_tolerance: 0.0
    floor_margin: 0.5
  resources:
    cpu: 2.0
    memory_mb: 4096
    disk_mb: 2048
    episode_timeout_s: 300
  capabilities:
    oracle:
      form: injected_state
      location: oracle/solved_state.json
    verifier:
      kind: reward_channel
    set_state: true
  types:
    tags: [swe]
```

Report excerpt (normative schema: `src/openenv/validation/schemas/report.schema.json`):

```json
{
  "report_schema_version": "1",
  "target": "envs/my_swe_env",
  "policy_version": "v1",
  "lane": "local",
  "results": [
    {
      "check_id": "semantic.floor_gap",
      "status": "fail",
      "measured": {"oracle_reward": 1.0, "measured_floor": 1.0, "declared_margin": 0.5},
      "evidence": ["reset/do-nothing state scored 1.0; gap to max is 0.0 < margin 0.5"],
      "remediation": "The verifier awards max reward without any work. Check for vacuous success conditions (e.g. empty-output comparisons)."
    }
  ],
  "verdict": "fail"
}
```

That failing example is the motivating bug class: a verifier that awards max reward to an empty
solution now fails `semantic.floor_gap` mechanically.

## Delivery

Stacked PRs, each vertically testable:

- **PR1** — this RFC. Also updates #778 (companion-spec reference) and #898 (registry framing).
- **PR2 — contracts.** Signature detection rules; parser interface; normalized manifest schema
  (capabilities + type specs; reward range, tolerances, judge pinning, oracle location); report
  JSON Schema; severity policy v1 (every check id above, including reserved); provider interface;
  grader interface. All testable on day one via fixtures, schema round-trips, and test-only fakes.
- **PR3+ — implementations**, one vertical slice per PR: walking skeleton, static level, runtime
  level on Docker-local, semantic level, containment, the Harbor parser, the HF Sandbox provider.
  The PostTrain `task.md` parser and the LLM-judged grader path (variance-mode determinism, rubric
  deepening) are fully specified with contracts and fixtures in PR2 and implemented separately.

## Level 2 execution amendment

This amendment specifies the missing execution contracts for the Docker-local
runtime implementation. It preserves the existing parser → normalized manifest →
grader → policy → report architecture. The first delivery wave consists of three
stacked slices: versioned contracts/shared assets; reproducible Docker supervision;
and a runnable CLI with startup, reward, observation and state checks. Later slices
add the remaining runtime checks. A partial implementation must expose the missing
requested checks as named SKIPs, never imply full Level 2 coverage from a PASS on
the implemented subset. Existing WARN exit semantics remain unchanged.

### Versioned declarations and public probe inputs

`validation.execution` opts a package into normalized manifest schema **2**:

```yaml
validation:
  execution:
    kind: openenv_ws
    probe_path: validation/runtime.json
    dockerfile: Dockerfile
    context: .
    agent_boundary: api
    requires_credentials: false
```

The other manifest sections remain authoritative for capabilities, resources,
network policy, reward bounds and grader applicability. The initial execution
binding is `openenv_ws` with API-only agent access. Process/filesystem agent
identities require a later explicit declaration; privileged provider exec is not
evidence of agent access. Paths are package-relative and portable. Readers reject
parent traversal, absolute paths and resolved symlink escapes.

Packages without `validation.execution` continue to produce the unchanged schema-1
manifest and static report. `NormalizedManifestV2` and `ValidationReportV2` have
separate committed JSON schemas. Report 2 accepts manifest 1, manifest 2, or null
after a parse failure so a runtime request can report missing prerequisites without
dropping diagnostics. No field is silently added to schema 1.

The public sidecar is data, not imported Python or a second capability manifest:

```json
{
  "plan_schema_version": "1",
  "reset": {"episode_id": "validation-probe", "seed": 42, "options": {}},
  "actions": [{"increment": 1}, {"increment": 1}]
}
```

The trusted loader bounds the file at 65,536 bytes, JSON nesting at 32 levels,
node count at 10,000 and actions at 1–100. It rejects duplicate keys, non-finite
numbers, non-JSON objects and reset options overriding `seed` or `episode_id`.
The seed is an explicit unsigned 32-bit integer. Missing or invalid probe inputs
are reported as an unmet prerequisite or an input failure; they never deselect
graders. The plan's content digest belongs in the reproduction bundle. Neither
privileged oracle inputs nor host callbacks may be substituted for public actions.

One collector owns reset, ordered actions until termination, and state reads on
**one** WebSocket session. It records immutable raw request/response strings before
client defaults or Pydantic coercion can hide malformed responses. Graders consume
that evidence and cannot mutate the measured episode. `/schema.observation`
describes step observations. The additive `/schema.reset_observation` field
describes reset observations; core servers publish it from an explicit optional
`reset_observation_cls`, defaulting to `observation_cls`. Environments whose reset
returns a distinct model must declare it. For legacy servers that omit the field,
the validator applies the step schema to reset as well. An explicitly malformed
reset schema is a failure, never a reason to fall back. Both schemas are recorded
alongside the transcript; reconstruct observation plus the separate reward/done
envelope before validating against the operation's schema. Neither schema may
retrieve non-local references.

Reset validation is required even when reset and step return different models.
An explicit reset output schema describes that difference; the validator never
silently substitutes the base `Observation` schema or skips reset. Reset input
declarations are a separate contract from these observation output schemas.

Reset reward may be null. Every step, including a nonterminal step, must emit a
finite JSON number within the declared range. Every numeric reset reward is also
checked against that range; booleans are invalid. These are Level Two validation
requirements, stricter than the core API: the core `Observation.reward` type
continues to accept `bool | int | float | None` for compatibility. A valid core
environment can therefore fail Level Two validation. Authors should emit numeric
zero when a step's intended reward is zero; the validator never converts an absent
or null reward to zero. State must retain the requested episode identity, reset
its step count to zero and advance it coherently for successful steps.

### Startup, policy and provider supervision

Policy **v2** retains every v1 entry and bound and adds `runtime.startup` at level 2,
local lane, failure severity. Runtime defaults to v2 and rejects an explicitly
incompatible policy before executing a subject. Static v1 remains supported.
Build, start or readiness failure caused by the subject yields startup FAIL with
the failed phase; a validator/provider defect yields ERROR. A missing prerequisite
or unsupported enforcement capability yields a named SKIP. One successful build
does not establish `static.reproducible_build`.

The validation-specific `LaunchSpec` names an immutable image ID/digest, requested
resources and network policy, an explicit run identity, startup deadline and
explicit environment variables. It inherits no host environment or credentials.
The provider owns build, readiness, bounded exec/logs, inspected settings and
idempotent cleanup on success, failure, timeout and cancellation. Cleanup evidence
must establish that no run-owned subject remains. Core provider ABCs are unchanged.

Credential delivery is deferred for this release. An environment that needs an
externally supplied credential declares `validation.execution.requires_credentials:
true` (a strict boolean, default `false`). This records a prerequisite only; no
credential names, values, sources or injection mechanism are accepted. After
validating the public plan, the runner reports `runtime.startup` as SKIP with the
missing `credential_delivery` capability named, before building or launching the
subject. Dependent checks remain SKIP and the report cannot establish complete
Level Two conformance. Missing credentials are never inferred from exception text
or from `llm_judged`: a self-contained judge may run without external credentials.
An undeclared startup or reset failure remains FAIL. Host tokens are never a
fallback, and credentials must not be embedded in manifests, plans or images.
Supporting external judge credentials requires a later explicit contract for
delivery, access isolation, lifetime, network access and evidence redaction.

Launches use no privilege, host namespaces, host-directory mounts, Docker socket
or forwarded credentials. They drop Linux capabilities, enable no-new-privileges,
use a read-only image and explicitly bounded writable roots, and expose only a
loopback control port. CPU is an allocation ceiling; memory includes an explicit
swap setting. `disk_mb` means aggregate writable subject storage, excluding image
layers. The initial provider splits this allowance between bounded writable `/tmp`
and `/dev/shm`; another writable root requires its own accounted budget. Episode
deadlines are externally enforced and
cover descendants; PID, log and build limits are additional supervisor budgets.
Unsupported enforcement is disclosed and must never trigger a weaker retry.

The initial provider may support only `public` networking and CPU subjects.
`no-network`, `allowlist` and GPU requests must then be refused before launch with
their missing capability named. Public networking permits egress; it does not
establish a host/private-address deny policy. A later no-network provider uses a
trusted helper sharing only the subject network namespace, with separate image and
filesystem, no external interface, Docker socket or outbound-proxy API. Merely
publishing a port on a no-network container is insufficient for control access.

Allowlist semantics are **destination-address enforcement**: exact DNS hostnames,
`*.example.com` subdomain patterns (not the bare apex), and IPv4/IPv6 CIDRs resolve
through a controlled resolver; matching DNS requests add their recorded addresses
to the run-owned allow set. Direct addresses are allowed only if included in a CIDR
or the recorded set. Entries allow all ports/protocols at those destinations; they
do not authenticate TLS/HTTP host identity and therefore disclose shared-IP
limitations. Alternate DNS, unmatched addresses and unsupported address families
must fail closed. This requires a separate reviewed enforcement implementation;
parsing these declarations does not claim they are enforced.

### Later runtime evidence contracts

Seed acceptance and empirical determinism are separate findings. A reset that
silently drops its seed does not establish seed control, while a deterministic
environment may legitimately behave identically under different seeds. Replay
comparison includes observation, reward, done, tool output and state; only
policy-owned volatile metadata can be excluded. Authors cannot exclude fields.
For `llm_judged`, the bounded variance path uses 20 completed identical-input fresh
replays and population reward variance in reward-squared units, compared to the
declared bound. The total run budget bounds all samples; fewer than 20 is incomplete.
This procedure is a runtime check, not a statistical confidence claim.

Session telemetry for seed handling, named rubric/configuration, child attribution
and subject-emitted record references is orchestrator-only. A future protocol
slice must authorize access with an opt-in, random per-run/per-session capability
attached to the **same** replay connection, reject unauthorized/cross-session
reads and never expose telemetry as agent MCP tools. A second WebSocket creates
another environment and cannot supply evidence for the measured instance.
A validator transcript alone cannot pass subject-emitted trajectory recording.
These authorization requirements do not add new public wire messages in this slice.

Applicability predicates must distinguish empty declared sets from absent
capabilities. Missing subject features, missing provider support and checks whose
implementation has not shipped are distinct outcomes. Independent graders must
not step a shared environment; invasive containment and resource probes use
disposable subjects. Episode-isolation checks require same-server reset evidence,
not just a new container, and oracle access checks use the declared agent boundary.

### Shared reproducibility and review assets

`tests/fixtures/validation/runtime/cases.json` is the versioned acceptance catalog.
Every runtime policy ID has positive/negative expectations, applicability, required
provider features and its implementation slice. Future cases are labelled planned;
they do not count as measured coverage. A small shared served subject supplies
deterministic public actions and test-only faults. Preserve the existing static
fixtures and run their schema-1 round trips alongside schema-2 tests.

The Docker/reference lane and local reproduction command use the same locked test
project, digest-pinned base, exact-revision wheel and case catalog. An evidence
bundle records wheel/source/plan/image identities, platform details, reports,
transcripts and cleanup findings. CI separately asserts inventory and expected
findings: a WARN exit 0 does not prove completeness. Product publish gating and
operator certification remain outside this feature; the test workflow only proves
the implementation's own acceptance cases.

## Explicitly out of scope

No hub/runner/queue/coordinator; no submission or auth APIs; no statistical-level implementations
(ids reserved only); no adversarial-panel specification (future operator-side work); no
metric-namespace RFC (deferred); no estargz/nydus conversion checks; no `openenv push` gating or
CI-workflow wiring (this stack ships the command, not its enforcement); no grandfathering policy
(operator concern); no operator-function standardization (report storage, webhooks, task↔env
mapping).

## References

- [#778 — RFC 008: Environment auto validation](https://github.com/huggingface/openenv/issues/778)
  (the bar: six properties, 44 acceptance tests)
- [#898 — Task validation for external task packages](https://github.com/huggingface/openenv/issues/898)
  (the one-shot lane, subsumed here as parsers under the grader registry)
- [RFC 004 — Rubrics](./004-rubrics.md) (leveraged as the served-format `llm_judged` path)
- Design review: "Automatic validation of OpenEnv environments" (Zach Wentz · Reflection, 2026-08)
- Harbor task format: [Task Structure](https://www.harborframework.com/docs/tasks) —
  `task.toml`, `solution/solve.sh`, `tests/test.sh`; consumed in-repo by `envs/tbench2_env`
