# Operator coding audit

This example collects and scores bounded coding-task evidence. It belongs to the
operator workflow, separate from the installed OpenEnv validator and its RFC 008
checks. Acceptance here does not establish full RFC 008 conformance, containment,
served-model revision attestation, or training benefit.

From a repository checkout, install the tested SDK and run the example:

```bash
uv sync --group dev
uv pip install 'huggingface_hub==1.33.0'
PYTHONPATH=src:envs uv run --no-sync python -m examples.coding_audit \
  --task /path/to/task \
  --config examples/coding_audit/config.json \
  --output /path/to/new-audit-directory \
  --json
```

The output directory is required and must be new. It retains `report.json`,
`evidence.json`, `config.json`, raw evidence artifacts, and `SHA256SUMS`. Exit codes
are `0` for accepted evidence, `1` for rejected or incomplete evidence, and `2` for
configuration or input failure. `--json` prints the report as well.

`scoring.py` provides pure `assess_coding_audit` and `validate_audit_config`
functions. `runner.py` provides `normalize_coding_config`, `coding_budget`,
`validate_coding_task`, and `run_coding_validation`. `hf.py` contains
`HFTaskBackend`, the bounded Sandbox and Inference Providers implementation.
The schemas bind task, verifier, immutable image references and independent execution identities.

Models may declare an explicit boolean `enable_thinking`, forwarded through the
HF client's `extra_body.chat_template_kwargs`. It is omitted by default. A
candidate-only override leaves the target protocol unchanged and is recorded
alongside request parameters. Support depends on the selected provider/model.
Candidate generation streams bounded content and reasoning; only a complete
`stop` response with validated usage can produce an executable candidate. A
`length` response remains rejected even when its terminal usage is retained.
Recognized Bash/sh Markdown fences are parsed before execution; the raw response
and both raw/script hashes are retained. Other language fences are rejected.

Candidate generation can opt into public task API context by adding
`"context_paths": ["environment/source/package/api.py"]` to the `generator`
model configuration. Only those declared regular UTF-8 files are read; paths must
be unique, normalized, and below `environment/source/`. Up to 32 files and 65,536
combined bytes are allowed. Linked files or parents, hidden or private paths,
NUL bytes, invalid UTF-8 and oversized context are rejected before inference.
The existing message and token budgets still apply. No source files are added by
default, and `target.context_paths` is rejected.

The candidate prompt labels each source path and its runtime filename relative to
the configured workdir, with `environment/source/` removed. Each generation record
retains `candidate_context` entries containing `path`, `runtime_path`, `sha256`,
and `size`, bound to the frozen task and checksummed evidence. This option does
not enumerate task directories or include verifier, reference or specification
files.

The HF backend launches paid services when invoked without a supplied test
backend. Use reviewed tasks and distinct execution and inference credentials.
The Hub pipeline's cloud worker packages these production example modules with
its frozen source and task bundle; its tests stay outside the bundle.

Run the offline suite with:

```bash
PYTHONPATH=.:src:envs uv run pytest examples/coding_audit -q
```
