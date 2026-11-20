# Tasksmith generation and downstream validation

[Repo2RLEnv Tasksmith](https://huggingface.github.io/Repo2RLEnv/pipelines/tasksmith/)
converts source pull requests into Harbor tasks. This directory documents the
handoff to a separate validation experiment. It includes no task payloads,
credentials, campaign ledgers, generated environments, or evaluation results.

## Freeze the upstream runtime

The workflow was inspected against Repo2RLEnv commit
`61903fe0ed805ea5418811b0fcf399e4144e95e4` (version 0.9.5). Check out that commit in
a separate, durable directory, then install its optional dependencies and author
runtime:

```bash
uv sync --frozen --extra tasksmith --extra modal --extra harbor --no-dev
uv run repo2rlenv tasksmith install-runtime
uv build --wheel --out-dir "$RUN_ROOT/dist"
```

Set `RUN_ROOT` to a private directory outside a checkout before running these
commands. Preserve the source commit, wheel checksum, author-runtime version,
panel, options, and any frozen source records with each campaign. The upstream
wheel check rejects a runtime that differs from its controller source.

## Configure credentials privately

This pinned version requires an Anthropic API key for authoring, even when a
different model is the downstream evaluation target. Its supported worker
providers are Modal and Daytona. Configure the selected worker through its
supported authentication flow, and keep the Anthropic key in a private
credentials file outside a checkout with mode `600`.

The CLI accepts `--env-file`, using dotenv without overriding existing variables.
Resolve conflicting ambient credentials before dispatch. Keep credential values
out of command arguments, logs, images, task bundles, and source control.
Authentication checks establish access; they do not qualify a generation run.

## Dispatch a bounded panel

Create a panel and options file using the upstream `Panel` and `Options` models.
Keep source diffs and frozen PR records private: they contain oracle material.
Exclude benchmark evaluation instances from training candidates and retain their
original licensing and holdout metadata.

Start with one source using the actual upstream command:

```bash
uv run repo2rlenv tasksmith run "$RUN_ROOT/panel.json" \
  --options "$RUN_ROOT/options.json" \
  --env-file "$CREDENTIALS_FILE" \
  --campaign "$RUN_ROOT/campaign" \
  --output "$RUN_ROOT/run" \
  --runtime-wheel "$RUN_ROOT/dist/repo2rlenv-0.9.5-py3-none-any.whl" \
  --stop-after 1 --json
```

Pass `--sources` when using previously validated frozen source records. Retain
the same output directory and frozen configuration when extending the panel.
Inspect each result, worker termination, and accounting before increasing
`--stop-after`. Preserve failed and incomplete attempts.

Upstream reservations govern admission; they are not guaranteed provider billing
caps. In particular, the blind solver can spend more than its reservation.
Establish enforceable limits or conservative call bounds before dispatch, and
keep unknown spend reserved. A normally terminated CPU worker can still require
explicit compute reconciliation. Do not release an experiment allocation solely
because the controller exited or its report shows a stopped worker.

## Export evidence with its exact scope

Read the upstream quality label and the task-bound controls and trial receipts.
Distinguish a generated bundle from a verified task; a prepared panel or failed
preflight is neither. Keep private verifiers, reference solutions, negatives,
source diffs, and results outside public repositories and learner images.

Run separate audits on the intended target model. An author or blind-solver
success does not measure another model's difficulty or establish training
benefit. Catalog identity searches do not prove that a task's behavior is absent
from existing environments. Full RFC 008 qualification requires its own evidence.
