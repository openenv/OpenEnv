# Reproducing runtime validation

This project locks the shared fixture and test toolchain separately from the main
repository. It uses Python **3.12.13**, uv **0.9.3**, a digest-pinned Python image,
and a hash-checked dependency wheelhouse. Install those Python and uv versions
first; CI installs both explicitly. Run from the repository root:

```bash
uv sync --project tests/validation_runtime --frozen
uv run --project tests/validation_runtime --frozen python scripts/validation/reproduce.py --suite fast --require-complete
uv run --project tests/validation_runtime --frozen python scripts/validation/reproduce.py --suite protocol --require-complete
uv run --project tests/validation_runtime --frozen python scripts/validation/reproduce.py --suite docker --require-complete
```

The Docker suite requires a running Linux Docker engine. Missing Docker is a
failure. The reference job uses Linux x86-64; Docker Desktop arm64 uses the same
recipe but records its different platform. No host Python code from the subject
is imported by the validator.

Docker test execution has a twenty-minute budget for the full fault/control
inventory; fast and protocol suites retain their ten-minute budgets.

The Docker suite snapshots the current source, builds its exact OpenEnv wheel,
and installs that wheel into this dedicated non-editable test environment. It
downloads only binary dependencies selected from the committed lock for the
Docker platform using compatibility tags queried from the pinned base image,
records those tags, verifies dependency hashes, and installs them in the fixture image
with networking disabled. A cold cache is supported. The tests run outside the
checkout with `PYTHONPATH` removed, exercising installed package data and the
production OpenEnv `/ws` endpoint. Each launch uses a fresh subject.

The image supports controlled `VALIDATION_FAULT` modes: `good`, `bad_reward`,
`bad_observation`, `missing_done`, `bad_state`, `hung_step`, `slow_step`,
`sticky_episode`, `sticky_file`, `sticky_background`, `exposed_oracle`, and
`startup_failure`. The sticky modes retain a counter in memory, a file or a worker
thread across reset. Fresh sessions start clean; `exposed_oracle` serves the fixture's harmless oracle canary
at its declared path. All fault
switches and wire corruption remain inside test assets. They share one fixture
and one public runtime plan, so a defect changes one property at a time.

The Docker suite contains 44 required cases: three provider lifecycle tests,
five network-policy tests, eight host-containment tests, fourteen resource-bounds
tests, thirteen CLI fault/control
cases, and one real `echo_env`
canary. The network tests run `public` and `no-network` subjects against a
validator-owned sink, detect a subject launched on the bridge while declared
`no-network`, and confirm the no-network helper serves `/health` and `/ws` without
giving the subject a route out. The host-containment tests measure the hardened
fixture in both network modes, detect a launch whose image root is writable,
detect writable nested directories and existing regular files on `/dev`, leave
owner-changeable permission denials inconclusive, and
detect a root process started inside the subject. The resource-bounds tests read
the subject's cgroup limits and complete filesystem inventory. Separate disposable
subjects qualify memory, process, disk and exec-timeout cleanup. Tests detect
limits above a smaller declaration even when storage measurement fails, an unaccounted writable root, nested device
directories and regular files, false exit-137
signals, and failed disposable subject or no-network helper cleanup, including
failed startup and recovered exec-timeout removal failures. Forged escape text and
descendant markers leave qualification unproven. The
slow-step case completes a tool call taking more than five seconds within
the declared episode budget. The hung-step case
checks the episode deadline; the interruption case sends SIGINT only after a
container log confirms the second step has begun. Both must retain the completed
reset/state/step/state prefix and remove their own containers. Each CLI case uses
a unique image label for independent cleanup verification.

The Echo canary copies the actual `envs/echo_env` sources unchanged and records
their hashes. A test overlay adds only the execution declaration, replay plan and
pinned offline image recipe. It runs `echo_message` and `echo_with_length` in one
session and verifies episode identity and state counts 0, 1, 2. Echo advertises a
separate reset-observation schema and emits null rewards on nonterminal steps.
Those rewards remain valid under the core API but fail Level Two's stricter
numeric-step-reward requirement. The canary expects schema **PASS**, reward
**FAIL**, report **FAIL** and CLI exit 1. This demonstrates the certification
boundary without changing Echo or coercing its rewards. Inspect
`cli/echo_canary/compatibility-findings.json` for the actual results.

Evidence is written to `outputs/validation-runtime/<run-id>/`, including source,
fixture, lock and wheel hashes; the retained wheelhouse; platform and toolchain
details; test results and bounded logs; provider cleanup evidence; and checksums.
The validation subject never receives a host socket, host credentials or host
directory mount. Tests verify run-owned containers disappear after normal completion,
failed readiness and exec timeout. No global prune is used.

```bash
python scripts/validation/verify_artifacts.py outputs/validation-runtime/<run-id>
```

`--require-complete` requires every selected acceptance test to execute without a
skip. Protocol and Docker runs must also execute every named case in the committed
`acceptance.json`; a nonempty filtered run cannot count as complete. The harness
clears inherited `PYTEST_ADDOPTS` and copies the inventory and its digest into the
bundle. Verify a bundle using its recorded source revision, since later revisions
can add required cases. This checks the implemented slice's inventory, not completion of all RFC
008 checks. A successful image build does not establish any runtime qualification. This layer
measures host-containment settings, process privileges, the complete mount table
and write access. Resource validation measures CPU, memory, PID and complete
writable-storage ceilings with positive stress evidence and Docker exec descendant
cleanup using a trusted kernel parent/child witness correlated by a fresh nonce
before timeout. It does not qualify served-episode deadline enforcement. Reproducible-build
qualification lands in a later slice. A host-read sentinel is not exercised: the current provider has
no positive control for a file on the Docker daemon host, which can differ from
the validator host on Docker Desktop or remote engines.

For changes after the first sync, reinstall the local wheel before fast/protocol
tests: `uv sync --project tests/validation_runtime --frozen --reinstall-package
openenv`. The Docker suite always builds and reinstalls its captured source.
Clean commit runs are the review evidence; dirty developer runs record exact
wheel input hashes and identify themselves as dirty.
