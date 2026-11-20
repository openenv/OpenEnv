# Pathway Analysis Environment

A toy pathway analysis environment for [OpenEnv](https://github.com/meta-pytorch/OpenEnv) that models a simple computational biology task: identifying the activated signaling pathway from synthetic omics data.

## Overview

Each episode hides a **true activated pathway** (e.g. "MAPK signaling"). **Mode A (pipeline)** cases ship a count matrix plus per-sample metadata in JSON; differential expression uses **PyDESeq2** (Wald test) on genes that pass a **minimum total-count prefilter** (DESeq2-style), and enrichment uses **Fisher ORA** against the **same gene universe** with **SciPy Benjamini–Hochberg** q-values. The DE gene list for ORA can be restricted by **direction** (up / down / both) and **minimum |log2FC|**, matching common pathway-activation workflows. **Legacy** cases keep static top-gene / top-pathway lists for fast tests.

### Intended agent workflow (RNA-seq style)

1. **Groups & design** — Learn how many **conditions / groups** exist and how samples map to them (`understand_experiment_design`, optionally `inspect_dataset`). The structured observation includes `samples_per_condition` and `conditions`.
2. **DGE (differential expression)** — Choose **which two groups to compare** (reference vs alternate) for DESeq2, then `run_differential_expression`. You can validate the contrast first with `understand_experiment_design` + `condition_a` / `condition_b`.
3. **Pathways** — Use DE genes for `run_pathway_enrichment`, optional `compare_pathways`, then `submit_answer` with a pathway hypothesis.

Optional per-case **`analysis_options`** (see `server/analysis.py` defaults): `min_total_count`, `padj_alpha`, `de_query_direction` (`up` | `down` | `both`), `min_abs_log2fc`, `ora_min_pathway_genes` (skip tiny pathways).

Observations include rich DE rows (`log2FoldChange`, `padj`), ORA tables, overlap summaries (genes supporting multiple top pathways), ambiguity flags when top hits are statistically close, an optional **`ask_expert`** budget with penalties, and an **HTML episode trace** path (`trace_path`) for auditing.

## Actions

| Action | Description | Reward (typical) |
|--------|-------------|------------------|
| `inspect_dataset` | View conditions, sample metadata, PyDESeq2 availability | +0.05 |
| `understand_experiment_design` | Structured design summary (`samples_per_condition`, defaults); optional `condition_a`/`condition_b` validate reference/alternate (does **not** run DESeq2). Valid pair is reused by `run_differential_expression` when DE omits conditions | +0.05 summary / +0.08 validated |
| `run_differential_expression` | PyDESeq2 contrast (`condition_a`/`condition_b`, or validated design, or `default_contrast` in case JSON) | +0.35 pipeline / +0.25 legacy |
| `run_pathway_enrichment` | Fisher ORA vs `pathway_genes` in the case | +0.5 |
| `compare_pathways` | Exclusive vs shared DE support for `pathway_a` vs `pathway_b` | +0.15 |
| `ask_expert` | Hint from case (`expert_hint`); uses budget; penalized | negative (see `expert_penalty`) |
| `submit_answer` | Submit `hypothesis` pathway name (correct: +2.0, incorrect: -1.0) | terminal |

**Strict mode** (`strict_mode` in JSON or `reset(strict=True)`): invalid contrasts or missing PyDESeq2 end the episode with a failure signal instead of silent fallback.

Stable machine-readable error labels: see **[docs/FAILURE_CODES.md](docs/FAILURE_CODES.md)** (`metadata["failure_code"]`).

## Scripts

- `scripts/generate_curriculum_cases.py` — writes ramped synthetic cases under `data/curriculum/`.
- `scripts/run_benchmark.py` — fixed policy + accuracy / action trace (`--json-out`).
- `bench/` — optional “gold-standard” RNA-seq toolchain (STAR/featureCounts/Salmon) for studies that require FASTQ→counts benchmarking.

## Quick Start

### Install

```bash
cd envs/pathway_analysis_env
pip install -e .
```

### Direct test (no server)

```bash
PYTHONPATH=src:envs python -c "
from pathway_analysis_env.server.pathway_environment import PathwayEnvironment
from pathway_analysis_env.models import PathwayAction

env = PathwayEnvironment()
obs = env.reset()
print(obs.message)

obs = env.step(PathwayAction(action_type='inspect_dataset'))
print(obs.message, 'reward:', obs.reward)

obs = env.step(PathwayAction(action_type='run_differential_expression'))
print(obs.message, obs.top_genes, 'reward:', obs.reward)

obs = env.step(PathwayAction(action_type='run_pathway_enrichment'))
print(obs.message, obs.top_pathways, 'reward:', obs.reward)

obs = env.step(PathwayAction(action_type='submit_answer', hypothesis='MAPK signaling'))
print(obs.message, 'done:', obs.done, 'reward:', obs.reward)
"
```

### Run the HTTP server

```bash
cd envs/pathway_analysis_env
uv run server
# or: uvicorn pathway_analysis_env.server.app:app --host 0.0.0.0 --port 8000
```

The Gradio UI (**Playground** + **Pathway lab**) is **on by default** for this app. To disable it (API-only server), set `ENABLE_WEB_INTERFACE=false` before starting.

Open **http://localhost:8000/web/** — use the **Visualization** tab (**Pathway lab**) for the guided workflow.

**Step-by-step UI guide:** [docs/USER_GUIDE_UI.md](docs/USER_GUIDE_UI.md) (what each button does, results panels, first-run example).

### Connect via client

```python
import asyncio
from pathway_analysis_env import PathwayEnv, PathwayAction

async def main():
    async with PathwayEnv(base_url="http://localhost:8000") as client:
        result = await client.reset()
        print(result.observation.message)

        result = await client.step(PathwayAction(action_type="inspect_dataset"))
        result = await client.step(PathwayAction(action_type="run_differential_expression"))
        result = await client.step(PathwayAction(action_type="run_pathway_enrichment"))
        result = await client.step(PathwayAction(action_type="submit_answer", hypothesis="MAPK signaling"))
        print("done:", result.done, "reward:", result.reward)

asyncio.run(main())
```

## Roadmap

| Version | Change | Why |
|---------|--------|-----|
| v2 | Replace hard-coded outputs with precomputed CSV/JSON from a real expression dataset | Keeps behavior deterministic while feeling more realistic |
| v3 | Swap analysis actions to call real DE and enrichment tools (PyDESeq2, GSEApy) | Turns the toy env into a real computational workflow env |
| v4 | Add multi-omics tasks, richer rewards, and BixBench-inspired task templates | Moves closer to useful scientific-agent evaluation |
