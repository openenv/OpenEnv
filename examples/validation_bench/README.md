# Validation experiments

These three experiments measure real-source defect detection, incomplete-evidence
handling, and local Docker validation cost. They call the production collector,
graders, runner and provider without changing the validation implementation.
See [RESULTS.md](RESULTS.md) for the measured results and qualification limits.

The measured production revision is
`2781faaa59f00b7192145282950193d570d07659`. The scripts in this directory are
research harnesses, not a new validation or replay API. Keep an unchanged
production checkout when comparing results. Use a fresh output directory for
each run; retain failed attempts as well as successful ones.

## 1. Paired defects in pinned Hub sources

Use an interpreter with this checkout's runtime dependencies, `uvicorn`,
`huggingface_hub`, `PyYAML`, and the lightweight environment dependencies
`sympy`, `gymnasium`, and `numpy`. Preparation downloads and verifies the public
source/task inputs; it installs nothing and launches no cloud compute.

```sh
.venv/bin/python examples/validation_bench/exp1/prepare.py --output /tmp/validation-exp1-inputs
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  examples/validation_bench/defects.py \
  --config /tmp/validation-exp1-inputs/config.json \
  --output /tmp/validation-exp1-results --replicates 3
```

The pinned sources are ScottWuMath and PortSimEnv. Each fault is paired with an
unmodified-source control and has independent activation witnesses. PortSim's
original dependency on OpenEnv 0.7.0 is deliberately overridden by the checkout.
Details, source pins and interpretation are in [exp1/README.md](exp1/README.md).
The measured artifacts retain the exact executed scripts, including the older
timeout-provenance limitation; subsequent harness fixes do not alter those data.

## 2. Incomplete evidence and saved-evidence fidelity

```sh
PYTHONPATH=src:envs .venv/bin/python examples/validation_bench/incomplete.py \
  --freeze-only --output /tmp/validation-exp2
PYTHONPATH=src:envs .venv/bin/python examples/validation_bench/incomplete.py \
  --output /tmp/validation-exp2
```

This uses bounded loopback servers and explicitly synthetic provider
measurements/Hub doubles. It creates no Jobs or Sandboxes. Exit status is nonzero
when a measured mismatch is present; inspect `exp2-report.json`. Its independent
reconstruction of saved evidence is experimental. OpenEnv has no production
replay reader at the measured revision.

The independently reproduced artifact-fidelity regression can also be run alone:

```sh
PYTHONPATH=src:envs .venv/bin/python \
  examples/validation_bench/exp2/reproduce_duplicate.py --surface wire
PYTHONPATH=src:envs .venv/bin/python \
  examples/validation_bench/exp2/reproduce_duplicate.py --surface http
```

## 3. Docker cost, concurrency and recovery

The frozen recipe requires Python **3.12.13**, uv **0.9.3**, and a Linux arm64
Docker engine. It checks installed OpenEnv bytes against the retained baseline
wheel. Remove `PYTHONPATH` so the interpreter uses that wheel. Dependencies and
base/helper images must already be acquired outside the measured interval.

The private evidence directory includes `baseline-source.tar.gz` (the complete
tracked baseline), `code/` (these research harnesses), and `raw-evidence.tar.gz`.
Extract the baseline source and add the research directory to its `examples/`.
The raw archive contains `study.json`, `performance-context/`,
`performance-context-manifest.json`, and `docker-baseline/l2-d2aada5dfcff/`
(the exact wheel, wheelhouse, requirements, baseline reports and acquisition
logs). Restore the archived tree at `/tmp/openenv-validation-bench-20261109`, or
pass `--context`, `--context-manifest`, `--study`, and `--lab` explicitly.
Recreate the pinned lab interpreter with the repository's locked lab recipe and
install the archived wheel with `uv pip install --python
tests/validation_runtime/.venv/bin/python --no-deps --force-reinstall
/tmp/openenv-validation-bench-20261109/docker-baseline/l2-d2aada5dfcff/wheel/openenv-*.whl`.
Verify the wheel SHA in `RESULTS.md` before using it. The archived context is
already prepared and hash-checked; no temporary Git worktree is required to use
it. The original full locked baseline command was:

```sh
uv sync --project tests/validation_runtime --frozen
env -u PYTHONPATH tests/validation_runtime/.venv/bin/python \
  scripts/validation/reproduce.py --suite docker --require-complete \
  --output /tmp/a-new-validation-baseline
```

That command builds a wheel from its checkout. For a historical rerun, run it at
the measured production revision and retain its own newly generated inputs;
do not silently substitute those for the archived wheel/context.

```sh
env -u PYTHONPATH tests/validation_runtime/.venv/bin/python \
  examples/validation_bench/performance.py --self-check
env -u PYTHONPATH tests/validation_runtime/.venv/bin/python \
  examples/validation_bench/performance.py --plan-only \
  --repeats 20 --concurrency 1 4 8 --seed 20261109 \
  --cold-builds 3 --failure-repeats 3 --output /tmp/validation-exp3
env -u PYTHONPATH tests/validation_runtime/.venv/bin/python \
  examples/validation_bench/performance.py \
  --repeats 20 --concurrency 1 4 8 --seed 20261109 \
  --cold-builds 3 --failure-repeats 3 --output /tmp/validation-exp3
```

The primary plan has 120 latency samples, two excluded warmups, eight excluded
padding runs, three cache-bypassing builds and nine failure attempts. Every warm
sample calls the real provider build; this includes source staging and cached
BuildKit work. `--no-cache` does not imply a cold base image or fresh downloads.
Check Docker-store capacity before requesting it. The experiment never globally
prunes caches or deletes images. Run-owned labels permit narrow cleanup only.

The separate recovery cohort used `--repeats 1 --concurrency 1 --cold-builds 0
--failure-repeats 3` after the primary cold builds exhausted Docker's store.
Those extra warm samples are excluded from primary latency results. For startup
failure, independently require a successful container create/start and the
fixture's deliberate-startup-error log; the frozen harness's
`expected_failure_match` flag alone also accepts a build failure. For all cases,
inspect `errors`, milestones, retained trace prefixes and independent cleanup.
A shell exit of zero is not an experiment success verdict.

Raw phase spans are inclusive; exclusive totals remove nested work on the
calling thread. CPU time covers the validator's calling thread, excluding
Docker engine/subject/subprocess CPU. Latency quantiles use nearest rank and
retain errors; batch throughput includes result persistence and c8 padding.
Neither the startup-only arm nor an overall WARN establishes admission.

## Evidence

Private Hub dataset: `burtenshaw/openenv-fineenvs-e2e-20261104`, prefix
`validation-bench-20261109/`. The archive, inventory, frozen plans, raw results,
executed harness snapshots, source/task hashes, wheelhouse and independent
reviews are retained there. Access requires the owner's permission. Nothing in
these experiments qualifies a published Space image or weakens `immutable_image`.
