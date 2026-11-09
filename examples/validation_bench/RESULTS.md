# Validation experiments: 2026-11-09

The experiments detected every activated injected defect in their measured
scope, exposed a saved-evidence fidelity defect, and measured the cost of local
Docker validation. Production validation code was unchanged throughout.

Production revision: `2781faaa59f00b7192145282950193d570d07659`.
Reproduction commands: [README.md](README.md). Private raw evidence:
[validation-bench-20261109](https://huggingface.co/datasets/burtenshaw/openenv-fineenvs-e2e-20261104/tree/main/validation-bench-20261109).
The evidence includes frozen expectations/plans, executed scripts, source and
dependency hashes, raw traces, bundles, failed attempts and independent reviews.

## 1. Paired defect detection on real sources

**48 trials; 24 activated faults; 21 discriminating PASS-to-FAIL pairs; three
masked FAIL-to-FAIL pairs.** There were no false PASSes on activated faults and
no false FAILs against the scoped, preregistered control expectations. The
independent audit reproduced all 240 check statuses.

ScottWuMath was pinned to `4de8125d314fca0a3f137fa1bab04b966a3efc15`;
PortSimEnv to `fddf26c4d84f9f9ca4dcfb2503ee188473e53be9`. Each environment
had three paired executions of four faults: a malformed observation, a string
reward, retained state after reset, and a withheld completed-step response.
The first, second and fourth mutate the wire around the original server; the
reset fault changes one server assignment. Independent witnesses confirm
activation before considering grader output. These are concrete instances of
the proposed fault classes, not an exhaustive nonfinite-reward or delayed-leak
sweep.

PortSim's unmodified reset already violates its advertised observation schema:
it returns generic `Observation` where the server advertises
`CallToolObservation`. Its three injected observation faults add a diagnostic,
but the scalar verdict remains FAIL. Those pairs cannot be credited as new
PASS-to-FAIL detections. All 12 unmodified PortSim controls retain this expected
schema FAIL. Its declared OpenEnv 0.7.0 dependency was explicitly overridden
with the measured framework revision.

The original sources also lack the complete validation execution configuration:
Scott's original static CLI report FAILs; PortSim's original static manifest
passes, while runtime validation remains unsupported. The experiment uses
explicit source-bootstrap configuration, without claiming published Space image
qualification. A separate stale PortSim rubric-introspection result is retained
outside the injected-fault statistics; future reward contamination was not shown.

Limits: the execution repeats reuse one task and fixed actions per environment.
They are descriptive, not independent-task sensitivity estimates. The experiment
constructs startup status from collection and does not exercise provider startup.
Its hangs establish response withholding and step timeout, not process-kill
enforcement; the measured artifacts did not freeze the exact numeric effective
timeout. Two harness provenance fixes apply only to future runs. Trace equality
does not establish that all hidden environment state resets correctly.

## 2. Incomplete evidence and reconstruction

**60 observations; 231/231 live assertions agree; all saved decisive report
statuses match live.** Missing evidence remains inconclusive and known FAILs
survive unrelated missing measurements. The immutable-image tests create zero
Jobs. These provider/Hub measurements are controlled doubles, not live HF
qualification.

Of 48 replay-capable runs, **42 agree and six mismatch**. Nine deliberately
redacted/inconclusive runs and three unsupported JSON runs have separate
denominators. Across supported runs, 378/384 check statuses agree; 219 are
SKIP-to-SKIP comparisons, leaving 159/165 non-SKIP agreements. All bundle
checksums verify. Those checksums do not establish semantic fidelity.

The independent review confirmed one production writer defect on two surfaces:
`runtime/artifacts.py` decodes JSON into last-key-wins dictionaries in
`_recording()` and `_http_body()`. A complete oracle canary in an earlier
duplicate-key value is discarded. The saved evidence still claims completeness
without redaction/omission flags. This reproduces on WebSocket and HTTP evidence
across all three seeds. **Live and saved reports remain FAIL; grading the
reconstructed saved evidence changes oracle FAIL to PASS.** No production replay
reader exists at this revision, so this result does not establish a live
admission bypass. The minimal repros and independent direct-writer comparisons
are retained.

Recommended correction: preserve decisive duplicate-key content safely, or
explicitly classify that evidence as unsupported/incomplete. A reconstructed
PASS must never be inferred from an unflagged lossy representation. Intentional
credential omission is separately flagged and is not counted as this defect.

A separate four-case, root-only follow-up is retained outside the frozen matrix.
Its attempted independent review was rejected by the provider's automated
security review. It has no independent HTTP collector-path verification and is
excluded from the headline counts and independently verified conclusions.

## 3. Validation cost and recovery

Local host: macOS 26.6.2 arm64; Docker Desktop Linux aarch64, Docker 29.9.2,
16 CPUs and 8 GiB VM RAM. The experiment uses Python 3.12.13 and uv 0.9.3.
The separate locked Docker acceptance baseline passes **44/44**, without skips,
at a clean checkout of the measured revision. This establishes no native Linux
amd64 or live HF qualification.

The warm matrix has **120 measured samples**, 20 per arm/concurrency cell,
plus two excluded warmups and eight excluded c8 padding runs. Every sample
uses the real production build/start/stop path. Full validation uses the
production runner and bundle writer; startup-only supplies no admission verdict.
Both arms use identical hardening and CPU1, memory256 MiB, disk64 MiB,
episode10-second declarations. The immutable 96.3 MB offline context includes
the exact baseline wheel and dependency wheelhouse.

| Concurrent runs | Startup p50 / p95 | Full p50 / p95 | Full throughput |
|---:|---:|---:|---:|
| 1 | 5.96 / 8.70 s | 30.29 / 32.85 s | 2.02 runs/min |
| 4 | 10.18 / 10.45 s | 40.24 / 41.93 s | 5.83 runs/min |
| 8 | 18.10 / 19.58 s | 59.10 / 64.10 s | 7.82 runs/min |

Quantiles use nearest rank; all measured rows are retained. Batch throughput
includes result/checksum persistence and, at c8, four padding completions per
arm. Latency excludes padding and final metrics/checksum persistence, but
includes independent cleanup verification. The arms are randomized within a
single-host run; this is a descriptive cost comparison, not a paired causal
speedup estimate.

The difference between serial-arm nearest-rank p50s is **24.33 seconds**. Eight
concurrent validations achieve **3.87 times** the serial throughput while
increasing per-run latency. All 130 healthy rows, including warmups/padding,
have zero errors and verified empty cleanup; all full reports retain identical
results: nine runtime PASS, four emitted SKIP, one static PASS, overall WARN.
Three other checks are capability-inapplicable and absent. The initial study's
seven-SKIP expectation was corrected in the frozen primary plan before execution.

At c1, p50 exclusive resource measurement is **17.54 s**, cached build/source
staging **2.47 s**, start **3.32 s**, grading **0.197 s**, and bundling **0.009 s**.
Phase medians need not sum to the total median. Resource spans exclude nested
containment, execution and cleanup spans. Full bundle size is about **32.2 KB**.
P50 validator calling-thread CPU is 1.10 s at c1 and 2.42 s at c8; Docker
engine/subject/subprocess CPU is not measured. These data point first to probe
orchestration and repeated context/build work for performance investigation,
rather than grading or report serialization. No checks were removed or weakened.

BuildKit creates a fresh attestation/index ID on cached builds, so the frozen
report's literal `same_primary_image=false` is retained. All 130 warm build logs
nevertheless show the same runnable manifest and config:

- Manifest: `182d221dfddbec5ebc09c70ec0723214ea1548dc5ba732cd728e26f2f567301d`
- Config: `8933613ed575386f8239f3da85ab0a9870bd6feda0b203906434fba59972e4da`
- Baseline wheel SHA256: `bad23e2a2d7418dc2e3f77978ddbca4baa672f7cc817c967c2d7c3f811534a90`

### Storage failure and separate recovery

All three subsequent cache-bypassing builds failed with **Docker-store space
exhaustion**. No usable cold-build latency was obtained. All nine primary failure
attempts then failed during build, before activating their intended faults.
They remain in the operational record and are not detection/recovery successes.
The frozen report's `summary.errors` also omits cold-build errors; inspect its
`cold_builds` entries as well as the nine failed fault attempts.
The frozen harness's startup-failure matching flag is too weak to distinguish
a build error from the intended server error; raw command/log witnesses are
therefore required.

The coordinator reclaimed only 22 precisely selected experiment-owned,
reclaimable `source.local` cache records (**773.6 MB**). No pre-existing/shared
image layers or global caches were pruned. A fresh frozen recovery cohort then
ran two warmups, two additional calibration samples, and nine fault trials.
Its warm timings are excluded from the primary matrix.

- Three startup faults: successful build/create/start followed by the fixture's
  deliberate startup-error log; failure plus independent cleanup completed in
  **33.83–34.05 s** from worker launch.
- Three hung steps: completed reset/state/step/state prefixes survive; timeout
  is retained; process exit **10.63–10.77 s** and cleanup **10.72–10.85 s** after
  the independently witnessed blocked second step.
- Three SIGINT trials: the same prefixes survive; interruption remains an
  explicit collection error; process exit **0.74–0.75 s**, verified cleanup
  **0.83–0.84 s** after the milestone.

All nine recovery trials reached their intended faults, produced the expected
FAIL verdict, and left no owned containers. Expected startup/interruption
exceptions remain in raw `errors`; they are not silently removed. The 15-second
post-milestone recovery budget was met in all six hang/SIGINT cases. This tests
Python harness workers using the production runner, not CLI signal semantics.

### What this establishes

Detection and conservative missing-evidence behavior work for the exercised
trajectories; saved-evidence fidelity needs correction. Warm Docker overhead and
bounded cleanup are measured on this host. Cold build cost remains unmeasured,
and the storage failures demonstrate a capacity constraint. No operational fault
prevalence or avoided-training-cost threshold was preregistered or measured, so
these experiments do **not** establish net monetary savings. No cloud/GPU jobs,
training, public messages, or published Space changes were performed.
