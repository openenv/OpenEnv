# Shared Level 2 validation assets

`cases.json` is the acceptance catalog for the RFC 008 runtime stack. Expected
findings are written from the RFC; they are not generated from grader output.
Every policy-v2 runtime check has a positive and a negative case, an applicability
description, required provider evidence and the PR that implements it.

Cases whose `implementation_pr` exceeds the current slice are planned acceptance
work, not evidence that a grader exists. During the first three slices, the
`good`, `startup_failure`, `bad_reward`, `bad_observation` and `bad_state` cases
exercise startup and the basic runtime contracts. `good` means those four checks
pass; it does not mean all Level 2 checks have been implemented.

The first-slice regression inventory additionally covers a missing `done` field,
a blocked second step, controlled CLI interruption and the unchanged reference
Echo environment. `tests/validation_runtime/acceptance.json` records the required
protocol and Docker test identities; a filtered subset cannot establish suite
completion. The Echo canary deliberately records its reward/schema compatibility
findings instead of modifying the environment to make it pass.

The `served_probe/` subject is shared by real-protocol, Docker and installed-wheel
tests. Its fault modes are test-only and each introduces one intentional defect.
The public `validation/runtime.json` file contains only reset inputs and actions.
Capabilities, resource budgets and network policy remain in `openenv.yaml`.

Preserve the existing manifest-only fixture directories. Schema 1 fixtures remain
the static compatibility baseline; adding runtime declarations selects schema 2.
Fast contract tests and the catalog inventory run without Docker. Docker evidence
must record an exact wheel and image identity, expected check inventory and verified
cleanup; a CLI zero exit status alone is insufficient because incomplete runs warn.
