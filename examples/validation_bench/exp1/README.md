# Real-source defect experiment

`../defects.py` uses the production `collect_runtime_evidence`, scheduler and
runtime graders without changing them. It freezes seeds, within-pair order,
plans, expected results, source hashes and dependency identity before launch.
The effective collector timeout and invocation are frozen explicitly. An existing
output directory is refused before any provenance files are written.
Each case has three paired replicates. Complete trials include a reference
episode, two same-session episodes, and a fresh episode. Incomplete reference
collection blocks dependent checks through the production scheduler.

Prepare the pinned public inputs, then use an interpreter that already contains
the OpenEnv runtime dependencies:

```sh
python examples/validation_bench/exp1/prepare.py --output /tmp/exp1-inputs
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python examples/validation_bench/defects.py \
  --config /tmp/exp1-inputs/config.json --output /tmp/exp1-results
```

Preparation downloads approximately 5 MB of public source and task packs,
verifying every retained file. It installs nothing and submits no compute.
`source-pins.json` identifies the exact source files needed; PortSim viewer
binaries are excluded. The public bucket is pinned by compressed and decompressed
hashes, with the latter bound to the pinned source manifest.

The temporary local venv used on 2026-11-09 reads existing workspace dependencies
through a `.pth` file; it does not install either original package. Explicit
`PYTHONPATH` precedence loads the checked-out framework and original source
packages. This intentionally overrides PortSim's original `openenv==0.7.0`
dependency. Local Python 3.13/macOS results are distinct from the earlier
Python 3.12/Linux ScottWuMath qualification recipe. The original math lock and
image digest are retained in provenance, not claimed as the local runtime.

Three action-conditional faults run in an ASGI wrapper around each real server:
an invalid observation field type, a string reward, and a withheld completed-step
response. These are **wire mutations**, including the hung-response case. The
reset-leak variant copies the original source and patches exactly one server
state assignment: ScottWuMath retains `submitted`, PortSim retains `checks_used`.
Original snapshots remain unchanged; each mutant's diff and hashes are saved.
The witness records upstream responses before mutation. Activation assertions
inspect those records and raw collector traces without reading grader results.

PortSim's original reset advertises `CallToolObservation` but returns generic
`Observation`, so the unmodified source already fails its reset schema check.
Its frozen control expectation records that FAIL. An injected step-schema fault
adds a distinct diagnostic but cannot establish a change in scalar FAIL status.
ScottWuMath's elapsed-time field can make episode isolation SKIP when fresh
traces differ. SKIP is never counted as PASS or successful defect detection.
The internal case ID `healthy` means an unmodified-source control; it does not
imply all contracts pass. In particular, PortSim's reset-schema failure masks
the scalar status change in three pairs: the final measurement has 21 target
PASS-to-FAIL pairs and three FAIL-to-FAIL pairs with additional step diagnostics.

Original CLI results and source-bootstrap reports remain separate. The unchanged
`immutable_image` gate, published-image identity, resource enforcement, network,
host and oracle containment are outside this experiment's measured scope.
No tokens are forwarded to servers; no Docker operations or public writes occur.
