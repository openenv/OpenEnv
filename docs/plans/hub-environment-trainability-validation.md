# Selecting OpenEnv/Harbor environments for training

This research plan separates native validation, supplemental audit evidence and
controlled learning experiments. Its scientific recommendations do not qualify
a private corpus, a provider or an OpenEnv package.

Choose environments for **trustworthy rewards, useful policy-dependent difficulty,
diversity, and measured held-out learning per unit cost**. Generate missing solutions,
tests, and scaffolding as candidates, then validate them independently. A generated
solution passing a generated grader establishes agreement between those components;
it does not establish task correctness or trainability.

## Current evidence boundary

The [example README](../../examples/hub_validation_pipeline/README.md),
[`pipeline.py`](../../examples/hub_validation_pipeline/pipeline.py), and
[`harbor.py`](../../src/openenv/validation/harbor.py) define the native boundary:

- `harbor-semantic-v1` requires `static.manifest`, `runtime.startup`,
  `runtime.reward_well_formed`, `runtime.episode_determinism`,
  `semantic.oracle_max` and `semantic.floor_gap` to PASS. Declared canaries also
  require `semantic.canary_floor` to PASS. Any FAIL/ERROR, required SKIP or missing
  required evidence blocks admission; other standard checks remain named SKIPs.
- Schema 2 native evidence binds fresh NOP, oracle, oracle-replay and declared
  canary Trials to original/effective task digests, the same retained image and
  inspected configuration, raw verifier rewards, successful oracle exits and
  verified cleanup. Replay checks concern the same exact submission.
- The maximum, floor gap and canary outcomes are checked against an explicit
  operator contract. They measure the frozen controls, not general specification
  correctness. The complete `full` admission inventory remains unqualified.
- `evaluate()` requests a configured model attempt per admitted task. It does not
  perform optimizer updates or a controlled multi-model training experiment.
- The native adapter rejects separate verifier environments, declared artifact
  extensions and non-public/differing phase network policies. Upstream capability
  alone does not qualify those contracts in this adapter.

The supplemental `harbor-coding-audit-v1` implementation lives in
[`examples/coding_audit`](../../examples/coding_audit/) and is invoked with
`python -m examples.coding_audit --task TASK --config CONFIG --output REPORT`.
Cloud and discovery orchestration import this example package. It generates bounded
candidate references, tests operator-reviewed negatives, repeats verifier runs and
measures target attempts after screening. Its reports always retain
`rfc008_complete: false` and `trainability_established: false`.

Real-task recipes require an explicit `--panel PATH`, including pinned source
identity, execution boundaries, full audit configuration and privately reviewed
counterexamples. The exact panel byte snapshot is hashed into preparation/bundle
provenance; native task/verifier files stay unchanged. Keep evaluation-only tasks,
private controls and populated results outside public source and training data.

Keep semantic checks in the validator and corpus/experiment selection in examples.
[The #1591 contract](https://github.com/openenv/OpenEnv/pull/1591) versions the
native evidence; new trial roles or guarantees require a versioned extension.
[RFC 008](../../rfcs/008-environment-auto-validation.md) also names containment,
verifier determinism, leakage, canaries and replayability. Implemented IDs or a
supplemental PASS cannot substitute for complete evidence of those contracts.

The example pins Harbor at `9c7a39b991d7e2205019f6be0598cbd22d84bfc3`.
Its Oracle agent executes a supplied solution script; its verifier executes
provided tests and reads their rewards. Those roles do not independently certify
the solution or verifier. [Pinned Oracle source][harbor-oracle],
[pinned verifier source][harbor-source]

## What the research supports

- **Verifier quality needs both acceptance and rejection controls.** SWE-bench's
  original verification process reviews task clarity and tests that reject valid
  implementations; its evaluation uses hidden issue-fixing and regression tests.
  Human screening improves reliability without proving complete correctness.
  [Original SWE-bench Verified methodology][verified]
- **Generated tests can find false positives, but are fallible.** UTBoost reports
  345 previously accepted erroneous patch evaluations exposed through test
  augmentation **and parser corrections**. Its behavioral comparisons depend on
  reference patches, so they do not independently prove specification completeness.
  R2E-Gym separately documents low discrimination of generated tests and biases in
  learned verifiers. [UTBoost][utboost], [R2E-Gym][r2e]
- **Public benchmark success needs contamination checks.** OpenAI's February 2026
  audit reports residual test-design defects and elicitable task/solution knowledge.
  Its 59.4% defect figure concerns a selected set of 138 difficult instances,
  not the full benchmark or arbitrary Harbor tasks. [2026 SWE-bench audit][audit]
- **Synthetic tasks have demonstrated training value in particular settings.**
  SWE-smith generates bugs that break previously passing repository tests and
  reports improved external benchmark performance after supervised fine-tuning.
  SWE-Gym trains agents and outcome verifiers; its repeated-attempt selection gains
  are distinct from single-attempt training gains. Neither result establishes
  online RL benefit for this corpus. [SWE-smith][smith], [SWE-Gym][gym]
- **RL and self-evolving tasks have direct, bounded evidence.** DAPO reports math
  RL results and filters uniform-reward groups to retain useful relative-advantage
  signal; this does not establish coding transfer or universal cost savings.
  Self-play SWE-RL reports coding RL gains on external benchmarks, validates
  injected bugs through execution/inverse mutation checks, and restores oracle
  tests before grading. It also discusses degenerate challengers and narrow
  curricula: solve-rate targeting alone can be gamed. These are results for the
  papers' recipes, not qualification of OpenEnv packages. [DAPO][dapo],
  [Self-play SWE-RL, June 2026 revision][ssr]

## Proposed stages, in priority order

### 1. Establish the specification and a frozen verifier

Select a small, relevant, training-permitted corpus. Record provenance, licenses,
repository/release, task lineage, bug/template family, and intended skills. Split
by repository **and task-generation/bug family before generating variants**;
near-duplicate prompts, related patches, and descendants of one source task stay
together. Keep an evaluation-only benchmark outside the training candidate pool.

Review each task's behavioral requirements before using its reward. Assemble:

- A known-bad starting state/NOP; known-good repairs checked against the requirements;
  and an independently implemented valid alternative where feasible. Reject
  implementation-specific assertions that the instructions do not justify.
- Issue-specific `FAIL_TO_PASS` checks plus `PASS_TO_PASS` regression coverage.
  Validate required test discovery/counts and individual results; a zero-test run,
  skipped required tests, swallowed collection error, or stale reward cannot pass.
- Hidden edge cases, property/metamorphic checks, and counterexamples justified by
  the specification. Differential comparison with a reference implementation is a
  useful signal; agreement with one potentially wrong reference is insufficient.

Freeze the task specification, verifier source/image/dependencies, test inventory,
and reward aggregation before comparative rollouts. Repeat grading of the **same
submission** in fresh environments to measure verifier flakiness separately from
policy sampling variance. Test ordering, filesystem state, clock/seed dependence,
and parser behavior need controls. These recommendations extend the failure modes
documented in [Verified][verified] and [UTBoost][utboost].

### 2. Separate public scaffolding from private evaluation

Use three explicit artifact classes:

- **Solver-visible:** instructions, starter repository, dependencies, public examples
  and development tests, generated API stubs/documentation, and tool scaffolding.
- **Training evaluator-private:** reference repairs, hidden reward tests, expected
  outputs, counterexamples, grader configuration, and candidate-generation traces.
  The policy may receive the declared reward/limited feedback, not these artifacts.
- **Sealed evaluation:** held-out task families and their independent verifier/tests;
  inaccessible to training, task generators, candidate selection, and tuning.

Harbor uploads shared-mode tests after the agent runs and has a separate-verifier
mode with explicit output artifact transfer. A clean verifier image can also avoid
runtime dependency installation. Qualify this adapter's support for that mode with
both agent and verifier image/configuration identities and run-owned cleanup.
Artifact contents remain untrusted. [Harbor verifier][harbor-verifier],
[separate verifier and artifact boundary][harbor-separate]

For code execution, a separate container alone is insufficient: submitted code
must not run with the grader's authority to read answer keys or write rewards.
Use a restricted execution worker and a protected result collector, with only
declared submission files crossing the boundary. Probe copied answers, forged
reward files, test-runner/PATH/import shadowing, disabled assertions, symlink/path
escapes, background-process interference, and grader-file modification. Inspect
the solver image, mounts, history/caches, logs, network access, and captured
observations for solutions, generation prompts, upstream fixed commits, or private
test material. Public-network v1 cannot establish an offline leakage boundary.
Runtime isolation and pretraining contamination are separate concerns; hidden
files do not remove public-source memorization. [Harbor boundary][harbor-separate],
[contamination evidence][audit]

### 3. Generate candidates and challenge them

Prefer existing reviewed solutions. When one is missing, generate a bounded number
of candidate repairs from solver-visible inputs, and record generator/model revision,
prompt/input digests, visibility, seeds, dependencies, cost, and patch digest. Replay
each from the same clean start; inspect privileged oracle scripts so writing reward
files or modifying tests cannot masquerade as solving the task. Independently
check a passing repair before promoting it to a reference candidate.
Generated components create a derived task version: retain the parent source digest
and new content/component digests, then revalidate its bound artifacts.

Generate counterexamples and task-relevant mutants: boundary-condition changes,
removed error handling, partial fixes, constant/hard-coded outputs, regressions,
and exploit submissions. Record which known-invalid behaviors the verifier rejects
and which valid alternatives it accepts. Review surviving mutants: equivalent or
out-of-scope mutations are not verifier failures. A mutation score is relative to
that panel, not a correctness probability. Search especially for submissions that
pass training rewards but fail the independent verifier. [UTBoost][utboost],
[SWE-smith][smith], [R2E-Gym][r2e]

**Prevent circular validation:** a component author cannot certify their own
solution/grader pair. Generate challenger tests from requirements and development
failures, independently review their assertions, then freeze them. Do not rewrite
the grader until a generated solution passes or obtain expected outputs solely
from that solution. A different generator model is useful diversity, not sufficient
independence. Disagreements trigger specification review/quarantine and a new
version; never silently repair the evaluator during a comparison. Final held-out
feedback must not enter this generation loop.

Start with generated wrappers/public scaffolds and candidate repairs; expand next
to validated mutants of working repositories. Try adaptive/self-evolving task
generation only after the static pipeline shows useful transfer. Keep generator
changes on training families, and audit diversity and exploitability as well as
difficulty. [SWE-smith][smith], [Self-play SWE-RL][ssr]

### 4. Measure rollout difficulty, variance, and efficiency

Run repeated attempts with the intended trainable policy and at least one other
model family. Pin model/checkpoint, harness, prompt/tool schemas, sampling settings,
seed where supported, task/verifier/image digests, token/step/time limits, hardware,
and concurrency. Randomize task/run ordering. Retain every attempt, including
failures and retries; provider seeds do not guarantee reproducibility.

Report per task/model: single-attempt solve rate with uncertainty, bounded-budget
pass@k separately, reward distribution, mixed-success group frequency, solution
diversity, truncation, environment/verifier error rates, and disagreement with the
independent evaluator. Regrade identical submissions to isolate verifier noise.
Infrastructure errors stay explicit rather than becoming model reward zero.

For binary rewards, `p(1-p)` describes Bernoulli reward variance; high variance
could reflect useful policy uncertainty **or a broken verifier**. Uniform-reward
groups have no relative reward advantage in GRPO-style updates, motivating DAPO's
dynamic sampling. Neither a tiny all-zero sample proves impossibility nor an
all-one sample proves the task useless to every policy. Select a mixture of
reliable easy/moderate/hard tasks; refresh difficulty as the policy changes.
[DAPO][dapo]

Measure build/cold-start, reset, agent, and verification time separately, with
median/tail latency, tokens, sandbox/GPU time, cache/storage, retries, and human
review. Rank reliable, diverse tasks by useful rollout yield and total cost;
include discarded generations and failed builds. Cache immutable builds and
preinstall verifier dependencies, while keeping fresh episode state. Timeouts and
long-horizon skills deserve explicit strata rather than deletion for cost alone.
[Harbor separate verifier][harbor-separate]

### 5. Demonstrate trainability in a controlled RL pilot

Pre-register the split, metric, budgets, and comparisons. Use identical initialization,
harness, optimizer recipe, and inference budgets. Compare the unchanged base model,
RL on a reviewed reference curriculum, and RL on a cost-matched curriculum with
the candidate subset replacing part of that reference. A matched-data SFT control
helps distinguish RL benefit from imitation of generated solutions. Repeat training
seeds and use development families for checkpoint selection.

Evaluate on the sealed family/repository split using the frozen independent
verifier. Report baseline-to-trained **pass@1 transfer**, uncertainty across seeds
and families, regressions, reward-versus-independent-score divergence, and total
cost/time. Cluster uncertainty by task family; repeated attempts on one task are
not independent new tasks. Report best-of-k selection separately so additional
inference spending cannot appear as a learning gain. These controls are proposed
experimental design, not a protocol established wholesale by the cited papers.

Advance only with positive held-out transfer against the base and useful improvement
against the matched reference, within a declared cost budget. Report incremental
cost per additional held-out solve and performance at equal total spend, including
generation/qualification/training/evaluation. If uncertainty includes no gain,
report inconclusive; if only training reward rises, trainability remains unproven.
Evidence qualifies the tested model/task distribution/recipe, not an environment
universally. [SWE-Gym][gym], [Self-play SWE-RL][ssr]

## Suggested starting budgets and implementation seam

These are **engineering suggestions, not scientific acceptance thresholds**:

- Screen about 20 tasks across several families. Try up to 2–3 repair candidates
  for missing oracles, 3 fresh replays per control, and 10–20 relevant faulty mutants
  per task. Expand targeted panels when survivors expose gaps.
- Begin with 2 models × 4 attempts × 20 tasks = 160 rollouts. On roughly 10
  finalists, target 3 models × 8 attempts = 240 observations, reusing compatible
  screening runs. Small samples provide coarse estimates; increase repeats/tasks
  according to observed uncertainty and cost.
- For an RL feasibility pilot, reserve disjoint training, development, and held-out
  families; for example, 60–100 training tasks, 20 development tasks, and 40 held-out
  tasks across multiple repositories. Start with 256–512 attempted training episodes
  per RL arm/seed and 2 training seeds; expand seeds and held-out families before
  making a broad claim. Fix an attempted-rollout/token/wall-clock ceiling so dynamic
  sampling cannot create an unbounded budget. Fewer tasks/seeds justify weaker claims.

Implement incrementally: first native determinism/containment/replay evidence and
separate-verifier support; then private artifact, mutation, and leakage audits;
then repeated `evaluate()` orchestration; finally an external training experiment
consuming the frozen corpus. Keep training-framework-specific machinery in examples.
Suggested sidecars are `verifier-audit.json`, `generated-candidates.jsonl`,
`rollout-matrix.jsonl`, and `trainability.json`, bound to corpus/profile/verifier
digests. Keep explicit exclusions and incomplete outcomes.

The resulting evidence labels should distinguish **runtime sanity**, **audited
verifier**, **rollout-characterized**, and **training benefit demonstrated for a
specified experiment**. These are proposed research labels, not new RFC PASS
statuses. No combined quality score should let a cheap rollout or oracle pass
compensate for leakage, an unreliable verifier, or missing required validation.

[verified]: https://openai.com/index/introducing-swe-bench-verified/
[audit]: https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
[utboost]: https://aclanthology.org/2025.acl-long.189.pdf
[r2e]: https://arxiv.org/html/2504.07164v1
[smith]: https://arxiv.org/html/2504.21798v1
[gym]: https://arxiv.org/html/2412.21139v2
[dapo]: https://arxiv.org/html/2503.14476v1
[ssr]: https://arxiv.org/html/2512.18552v3
[harbor-verifier]: https://docs.harborframework.com/tasks/verifier
[harbor-separate]: https://docs.harborframework.com/tasks/separate-verifier
[harbor-oracle]: https://github.com/harbor-framework/harbor/blob/9c7a39b991d7e2205019f6be0598cbd22d84bfc3/src/harbor/agents/oracle.py
[harbor-source]: https://github.com/harbor-framework/harbor/blob/9c7a39b991d7e2205019f6be0598cbd22d84bfc3/src/harbor/verifier/verifier.py
