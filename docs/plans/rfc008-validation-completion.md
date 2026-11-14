# Completing RFC 008 validation

Planning baseline: 2026-11-10, `openenv/OpenEnv` main at
`fb36d7d62ff9548ebd44f3323386cbea908e8706`.

Trackers: [RFC 008 #778](https://github.com/openenv/OpenEnv/issues/778),
[Level 2 umbrella #1177](https://github.com/openenv/OpenEnv/issues/1177) and
[task packages #898](https://github.com/openenv/OpenEnv/issues/898).

RFC 008 promises that `openenv validate` grades an environment against the 44
acceptance tests of #778. At the baseline, main ran every Level 2 runtime check for
served environments on Docker, but no Level 3 semantic check. That includes the
check the RFC opens with: a correct solution must earn maximum reward, and doing
nothing must not. The PR that adds this document delivers that check for Harbor
task packages and makes every report name the checks it did not run. This document
records where validation stands after that PR, defines what "complete" means, and
ranks the remaining work as PR-sized workstreams.

## Where validation stands

Each RFC check has a stable id, a level and a lane. The lane says who can run the
check. `local` checks run inside `openenv validate`. `hub` and `statistical` ids are
reserved for operators, the teams that run validation hubs or lab intake. The table
counts the local ids in the committed severity policy and the ids each execution
binding grades.

| Level | Local ids | Served packages (`openenv_ws`) | Harbor tasks (`harbor_task`) |
|---|---|---|---|
| 1 Static | 10 | 1 (`static.manifest`) | 1 (`static.manifest`) |
| 2 Runtime | 16 | 16, on Docker-local | 3: startup, reward well-formedness, episode determinism |
| 3 Semantic | 8 | 0 | 3: oracle reaches max, floor gap, canary floor |
| 4 Statistical | 0 (7 reserved) | Operator-side by design | Operator-side by design |
| Hub | 0 (5 reserved) | Operator-side by design | Operator-side by design |

Every requested local check that does not run is a named SKIP with its reason, at
every level. A report therefore reads WARN until all requested checks pass, and a
PASS on a subset cannot pass for full coverage.

Three limits apply across bindings:

- **Parsers.** `openenv.yaml` and Harbor `task.toml` ship. PostTrain `task.md`
  packages are refused as unrecognized.
- **Providers.** Docker-local enforces `public` and `no-network` networking and
  refuses allowlists and GPUs. The HF Sandbox provider refuses every launch with
  `missing provider capability: immutable_image`, because nothing yet binds a Space
  revision to an image digest. Harbor tasks with a digest-pinned image can run
  their trials on HF Jobs with `--sandbox hf-sandbox`.
- **Served oracles.** RFC 008 does not yet say how to execute an oracle against a
  served WebSocket environment, so served packages report their semantic checks as
  named SKIPs.

Work outside main measures more than the RFC's local checks. A cloud audit on HF
Jobs adds generated reference solutions, wrong-answer controls, repeat runs,
leakage probes and target-model pass rates.
[#1582](https://github.com/openenv/OpenEnv/pull/1582) adds training and validator
performance experiments. These results report under their own fields rather than
RFC check ids, so they do not count toward RFC coverage yet.

## What complete means

The RFC splits responsibility between the repo and operators, so completion has two
targets. The repo can reach the first one alone. The second one needs an operator
procedure that the repo documents but does not run.

**Local completeness.** Every local-lane id has a grader for each shipped execution
binding where it applies, and reports name every inapplicable or unimplemented id
with its reason. For served packages this means nine static graders, an oracle
execution contract and eight semantic graders. For Harbor tasks it means the same
static graders, the remaining semantic graders, and a decision on which runtime ids
apply to one-shot tasks.

**Operator completeness.** Each reserved `statistical.*` id has at least one
documented reference procedure that emits results in the shared report schema. The
repo still ships no hub, as the RFC requires, so the procedure lives in an example or
an operator repository. Comparable operator results also need the metric namespace
(`pass@k`, headroom, variance bound) that the RFC defers to a future RFC.

Validation still certifies that an environment can be trained on, not that it should
be. Measuring training signal comes last, and it belongs with the evaluation work in
[RFC 013 (#1256)](https://github.com/openenv/OpenEnv/pull/1256).

## Proposed workstreams

The workstreams are ranked by dependency and by how much each one improves
environment selection. Each one is one to three PRs. Costs are relative: S needs no
build or inference, M needs Docker builds and replays, and L needs model inference
or GPU time. Owners are suggestions for discussion.

| Rank | Workstream | #778 tests | Status | Cost | Suggested owner |
|---|---|---|---|---|---|
| 1 | Verifier sanity graders | #11, #25 | Harbor done; envelope and served open | M | Validation core |
| 2 | Harbor binding on main | Harbor parser slice | Done | M | Validation core |
| 3 | Verifier strength | #28, #26 procedure | Harbor canaries done; panel open | M, then L | Validation core and Hub pipeline |
| 4 | Determinism, replay and leakage graders | #30, #39, #40, #44 | Open | M | Validation core |
| 5 | Static graders and complete reports | #1, #3, #4, #8, #9, #10, #12, #41, #42 | Reports done; graders open | S to M | Any contributor |
| 6 | Cloud execution on HF Sandbox | Provider qualification | Harbor done; Spaces open | M | Hub pipeline |
| 7 | Statistical reference procedure | #13 to #16, #27 | Open | L | Hub pipeline |
| 8 | Wider task shapes | Workspace handoff, multi-container, credentials | Open | M to L | Hub pipeline and validation core |
| 9 | Training signal | #17 | Open | L | Validation experiments and RFC 013 |

### 1. Verifier sanity graders

`semantic.oracle_max` and `semantic.floor_gap` now grade Harbor tasks. The graders
read a format-neutral `OracleEvidence`, so another binding only has to measure the
rewards. Oracle tolerance and floor margin are fractions of the declared reward
range. Both the maximum and every oracle reward must exceed the measured floor
by at least the declared margin times the range width.

**Remaining.** Served packages need an oracle execution contract first: the RFC
names injected state and executable scripts, but not how either drives a served
WebSocket session or what doing nothing means there. That needs an RFC amendment.
`semantic.resource_envelope` also remains: record peak CPU, memory and disk during
the oracle run and compare them with the declared resources.

**Evidence.** Oracle and do-nothing rewards, the declared margin and tolerance, the
image digest and the cleanup outcome.

**Cost and owner.** M: Docker builds and short episodes, with no inference.
Validation core.

### 2. Harbor binding on main

Done. Harbor `task.toml` packages validate through the `harbor-semantic-v1`
profile and an operator-supplied `validation/openenv.json` contract. Fresh Docker
trials run a do-nothing agent, the oracle, an oracle replay and declared canaries,
all pinned to one immutable image. The binding requires Python 3.12 or newer and
`harbor==0.25.0` alongside `openenv[harbor]`. The NOP trial builds from the task's
Dockerfile, or pulls a task-declared `docker_image` that is pinned by digest
(`name@sha256:<digest>`). It refuses a `docker_image` without a digest, because a
tag can move between validation and training. The validator keeps
`openenv-validation:<run id>` only when the run has no FAIL or ERROR and was not
interrupted. It records the tag in `runtime.startup.measured` as
`retained_image_tag`. Otherwise it removes the tag.

**Follow-up.** Tasks that declare a prebuilt image by tag still cannot run under
`harbor-semantic-v1`. Some Terminal Bench tasks are affected. An operator can
resolve the tag to its digest in a validation overlay and record the original
bytes in `original_source_digest`. The validator does not resolve tags itself.

**Remaining.** Decide which runtime ids apply to one-shot tasks. Containment,
network policy, resource bounds and oracle containment could apply. Observation
schema, state contract and tool declaration accuracy do not.

**Cost and owner.** M. Validation core.

### 3. Verifier strength: canaries and an adversarial panel

A verifier can pass the oracle and floor checks and still reward wrong answers.
`semantic.canary_floor` now runs operator-declared canaries on Harbor tasks: each
known-wrong `solve.sh` replacement must earn no more than doing nothing.

**Remaining.** The reserved `statistical.adversarial_floor` (#26) still has no
procedure; the RFC notes that the panel "is not yet specified anywhere". Write an
RFC 008 amendment that specifies:

- where wrong answers come from: mutations of the reference solution, known-fault
  corpora, empty and partial outputs, and plausible solutions from a weaker model
  that an independent check confirms are wrong;
- how many are needed for a verdict;
- the pass rule, which is that every wrong answer scores at the floor.

The cloud audit's wrong-answer controls are a working prototype of this panel.

**Evidence.** Each wrong answer's provenance and reward, plus the verifier version.

**Cost and owner.** L for generated panels, which need inference. Validation core
writes the amendment, and the Hub pipeline team implements the operator panel.

### 4. Determinism, replay and leakage graders

`semantic.verifier_determinism`, `semantic.verifier_portability`,
`semantic.replayability` and `semantic.no_solution_leakage` have no graders.
Harbor tasks now establish `runtime.episode_determinism` when an oracle replay in a
fresh container reproduces every reward and the oracle exit code. Each trial is a
one-shot episode with no per-step observations. Served environments retain the
general comparison of observation, reward, done, tool output and state. The semantic
checks depend on the determinism finding.

**Approach.** Verifier determinism runs the verifier on one final state several
times and requires identical rewards. For `llm_judged` environments it requires
variance within the declared bound, reusing the Level 2 variance path. Portability
reruns the verifier with networking denied, as the RFC's hermeticity rule for #40
requires. Replayability re-executes a recorded trajectory and compares rewards. The
leakage check scans observations and agent-visible files for oracle artifacts such
as solution text, expected outputs and test files.

**Evidence.** Replay transcripts and reward vectors, the network mode of each
verifier run, and every leakage match with its location.

**Cost and owner.** M. Validation core.

### 5. Static graders and complete reports

Reports now name every unimplemented static check as a SKIP, so a static run reads
WARN rather than PASS after one check.

**Remaining.** Add the nine graders in two batches. The first batch reads the
manifest and lockfiles and needs no build: `static.dependency_pinning`,
`static.task_distribution_pinning`, `static.resource_declaration` and
`static.timeout_ceiling`. The second batch inspects built images:
`static.oci_labels`, `static.sbom`, `static.image_hygiene`, `static.layout` and
`static.reproducible_build`, which builds twice and compares digests.

**Evidence.** The files and image metadata each grader read.

**Cost and owner.** S for the first batch and M for the image batch. Any
contributor; these make good first issues.

### 6. Cloud execution on HF Sandbox

The HF Sandbox provider refuses every launch because it has no immutable image
binding. Runtime and semantic checks for served Spaces therefore cannot run in the
cloud, or on GPUs, through the RFC provider interface.

Harbor tasks do not have this gap when they pin `environment.docker_image` by
digest, because the task declares its own immutable image. `openenv validate
--sandbox hf-sandbox` now runs their trials as HF Jobs through Harbor's
`hf-sandbox` environment, so cloud audits can call `openenv validate` instead of
their own runner. The remaining work below covers served Spaces.

**Approach.** Resolve a Space revision to the registry digest of the image it
builds, verify it, and launch by digest. Qualify public Space images first, then add
an authenticated pull path for private ones. Once launches work, the cloud audit can
call the RFC graders instead of its own runner.

**Evidence.** The Space revision, the resolved digest, the pull receipt and the same
cleanup evidence Docker-local records.

**Cost and owner.** M. Hub pipeline team.

### 7. Statistical reference procedure

The learnable tests (#13 to #16) and the gameability gap (#27) are reserved ids with
no reference procedure. The RFC also defers the shared metric namespace.

**Approach.** Write the metric-namespace RFC, with definitions, sample sizes and
confidence intervals for `pass@k`, headroom and variance bound. Then publish a
reference operator procedure, outside the core package, that maps target-model
attempts onto the reserved ids:

- `statistical.reward_reachability`: some attempt reaches maximum reward;
- `statistical.difficulty_separation`: pass rates differ across model tiers;
- `statistical.headroom`: the target model's pass rate leaves room to improve;
- `statistical.variance_bound`: reward noise across repeats stays within a bound.

The cloud audit already measures target-model pass rates with confidence intervals.
It needs to emit them under these ids in hub-lane reports.

**Evidence.** Model identity and provider, attempt counts, per-attempt rewards and
intervals.

**Cost and owner.** L, since every task needs inference. Hub pipeline team, with the
metric-namespace RFC from validation core.

### 8. Wider task shapes

The Harbor binding refuses multi-step tasks, task-authored Docker Compose files and
tasks that need credentials. The cloud audit also hands off only individual output
files. The RFC defers credential delivery explicitly.

**Approach.** Add a workspace or git-patch handoff so repository tasks are graded on
the agent's full change. Treat multi-container support as a provider capability, as
the RFC does for GPUs, so providers without it SKIP with the capability named. Write
the credential-delivery amendment the RFC asks for, covering delivery, access
isolation, lifetime, network access and evidence redaction.

**Evidence.** Handoff manifests with hashes, and provider capability reports.

**Cost and owner.** M for workspace handoff and L for multi-container and
credentials. The Hub pipeline team builds the adapters, and validation core writes
the amendment.

### 9. Training signal

`statistical.training_signal` (#17, advisory) asks whether training on an environment
improves a model. Nothing measures it yet.

**Approach.** Use the evaluation harness proposed in RFC 013 to run short training
jobs on a sample of validated environments, then report the change in held-out
success. The training experiments in #1582 are a first measurement. Keep the result
advisory, as the RFC's severity policy says, because a short run cannot show that an
environment is unlearnable.

**Evidence.** Training configuration, seeds, and held-out success before and after
training.

**Cost and owner.** L: GPU time. Validation experiments, with the RFC 013 authors.

## Sequencing

The verifier sanity graders came first because canaries, verifier determinism and
replay all reuse oracle replay. The order of the remaining work follows from its
dependencies:

- Workstream 4 builds on the oracle replay the Harbor binding now runs.
- Workstream 5 is independent and can start now.
- Workstream 6 moves the Harbor and served checks to the cloud and is needed before
  workstream 7 runs at scale.
- The served half of workstream 1 waits on the oracle execution amendment.
- Workstreams 8 and 9 widen coverage after the core lands.

## Open questions for maintainers

These questions change the scope of a workstream, so they need an answer before its
first PR:

1. How should an oracle drive a served WebSocket environment, and what is its
   do-nothing episode?
2. Should `semantic.canary_floor` also accept canaries the validator generates, or
   only canaries shipped with the package?
3. Should the adversarial panel be an RFC 008 amendment or its own RFC?
4. Where should operator reference procedures live: `examples/` or a separate
   repository?
5. Should the runtime applicability table for one-shot task packages be normative
   in the RFC?

## Sources

- `rfcs/008-environment-auto-validation.md`, including the Harbor task binding
  section
- `docs/plans/rfc008-level2-implementation.md`
- `tests/fixtures/validation/runtime/cases.json`
- `src/openenv/validation/runner.py`, `harbor.py`, `graders/semantic/` and
  `providers/hf_sandbox.py`
- Issues #778, #898 and #1177; PRs #1256 and #1582
