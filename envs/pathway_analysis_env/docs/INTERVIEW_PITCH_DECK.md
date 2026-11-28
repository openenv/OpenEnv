---
marp: true
theme: default
paginate: true
size: 16:9
---

## OpenEnv: Pathway Analysis Environment
### A realistic RNA-seq workflow environment for evaluating scientific agents

**Parul Sarma Aryasomayajula**

---

## The problem
- Scientific/biomed agent performance is hard to evaluate
- Real workflows require **multi-step tool use**, not single-shot answers
- We need **reproducible**, **instrumented** tasks with measurable outcomes

**Goal:** evaluate agents on end-to-end reasoning + tool execution.

---

## What OpenEnv provides
- A standardized **environment API** (reset / step / state)
- **Client/server separation** (FastAPI runtime; reusable clients)
- Structured observations & metadata for evaluation and debugging

OpenEnv makes complex tasks **testable like environments**.

---

## The task: pathway analysis from gene expression
Agent objective:
- infer what biological programs/pathways explain a phenotype

Typical workflow:
- understand experimental design
- run **Differential Gene Expression (DGE)**
- run **pathway enrichment**
- compare candidates, submit a hypothesis

---

## Agent interface (actions)
Supported actions in `pathway_analysis_env`:
- `understand_experiment_design`
- `inspect_dataset`
- `run_differential_expression`
- `run_pathway_enrichment`
- `compare_pathways`
- `ask_expert` (budgeted hint/curriculum)
- `submit_answer`

This maps to how real analysts operate—now made “agent-stepable”.

---

## Observability & safety rails
Each step returns structured outputs:
- DE table: log2FC, p-value, q-value, baseMean
- enriched pathways + overlap genes
- ambiguity/overlap summaries (when multiple pathways look similar)
- stable `failure_code`s for error analysis
- an **HTML episode trace** for debugging & review

---

## Real pipeline integrated (not a toy)
Differential expression:
- **DESeq2-style** statistics via **PyDESeq2**
- prefilter low-count genes (DESeq2 practice)
- configurable thresholds (FDR, min log2FC, direction)

Pathway analysis:
- ORA on query genes
- supports:
  - deterministic case-defined gene sets (offline)
  - Enrichr libraries via **gseapy** (quick realism)

---

## Real-data support (GEO-style inputs)
Added support for:
- counts files like `*.csv.gz` (GEO)
- sample metadata (condition labels)
- pipeline cases defined as JSON configs

So the same agent actions run on **real public studies**.

---

## Public study demo: GSE235417
Study:
- head & neck squamous cell carcinoma PDX (UCLHN04)
- baseline vs acquired cetuximab-resistant
- 3 biological replicates per condition

Inside the environment:
- DESeq2 contrast: resistant vs baseline
- enrichment libraries: Hallmark / KEGG / Reactome
- outputs exported as JSON + HTML trace + report

---

## Example results (headlines)
Top DE genes (example):
- DAPK1, CXCL8, SOX2, FKBP5, …

Top enriched pathways (example):
- EMT
- inflammatory / interferon response
- TNFα/NF-κB signaling
- cytokine signaling

(Emphasis: reproducible pipeline + measurable signals, not over-claiming biology.)

---

## What we can measure (evaluation metrics)
- **Correctness**: does the agent identify the right pathway/hypothesis?
- **Efficiency**: steps taken, tool calls, reward trajectory
- **Robustness**: handles invalid contrasts, missing metadata, low signal
- **Interpretability**: traces + structured evidence for postmortems

This becomes an agent benchmark with real failure modes.

---

## Why this is valuable
- Bridges “toy benchmarks” and “real science workflows”
- Makes agent behavior:
  - reproducible
  - debuggable
  - comparable across models
- Strong foundation for:
  - regression tests
  - eval harnesses
  - future RL-style training loops

---

## Roadmap
- Add **GSEA** (rank-based enrichment) alongside ORA
- richer experiment designs (covariates / batches)
- more datasets and tasks (perturbations, multi-omics, drug response)
- standardized scoring + reporting for agent comparisons

---

## Close
**OpenEnv environments turn real workflows into measurable evaluation problems.**

Happy to discuss:
- where this fits into your agent evaluation stack
- how you’d score tool-using agents
- what domains you’d want next

