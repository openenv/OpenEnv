# Pathway Analysis Environment — Web UI user guide

The browser UI is built with **Gradio** and mounts at **`/web`** when the server runs. You get **two tabs**:

| Tab | Purpose |
|-----|--------|
| **Playground** | Generic OpenEnv form: choose any `action_type` and fields, then step. Good for experiments or copying exact payloads. |
| **Visualization** (Pathway lab) | Guided buttons for the RNA-seq–style workflow: **groups → DGE → pathways**. Use this tab for the walkthrough below. |

There is nothing extra to “install” for the UI beyond running the server.

---

## 1. Start the server

From the environment directory (with `PYTHONPATH` including repo `src` and `envs` if you use `uvicorn` manually):

```bash
cd envs/pathway_analysis_env
uv run server
```

Or:

```bash
uvicorn pathway_analysis_env.server.app:app --host 0.0.0.0 --port 8000
```

This app turns the Gradio web UI **on by default**. To run API-only: `ENABLE_WEB_INTERFACE=false`.

---

## 2. Open the UI

In your browser go to:

**http://localhost:8000/web/**

If the page redirects, that is normal. Click the **Visualization** tab to open **Pathway lab**.

---

## 3. Case & episode (top of Pathway lab)

### **Case JSON (`data/`)**

- **What it does:** Chooses which curriculum / toy case file under `data/` is loaded for this episode.
- **Why:** Each JSON defines counts (or legacy lists), conditions, pathway gene sets, and the hidden **true pathway** you try to guess at the end.

### **Reset episode**

- **What it does:** Loads the selected case, starts a **new episode** (new episode id, clears DE/ORA state), and fills **Reference** / **Alternate** from the case when `default_contrast` exists.
- **Why:** Always reset after changing the case file or after you **Submit** (episode ends).

### **Episode state** (panel)

- **What it shows:** Pipeline vs legacy, strict vs lenient, **conditions** in the case, whether **Design / DE / ORA** steps ran, expert budget, and **validated contrast** if you used **Understand design**.
- **Why:** Quick read on progress without opening raw JSON.

### **Last action**

- **What it shows:** Short status from the last button you clicked.

---

## 4. Contrast fields (Reference / Alternate)

- **Reference** = baseline condition (denominator side of log2 fold change in DESeq2 terms).  
- **Alternate** = the comparison arm (e.g. treated vs control).

**What they do:** They are sent to **0 · Understand design** (optional validation) and **2 · Run DE** (differential expression). If you validated a pair with **Understand design**, you can leave these empty on **Run DE** and the server reuses the validated contrast.

---

## 5. Workflow buttons (what each step does)

### **0 · Understand design**

- **What it does:**  
  - **Both fields empty:** Returns a structured **experiment design** summary: number of groups (`conditions`), **samples per condition**, sample ids, optional `default_contrast` — **no DESeq2 run**.  
  - **Both filled:** **Validates** reference/alternate (must be real condition names, two different groups, and in pipeline cases each arm must have samples). On success, stores the pair for **Run DE** if you omit contrast there.  
  - **Only one filled:** Small penalty; asks you to provide both or neither.
- **Why:** Mirrors an agent (or analyst) **understanding the design** and **choosing which two groups to compare** before running DGE.

### **1 · Inspect**

- **What it does:** Returns **sample metadata**, sample ids, and whether PyDESeq2 is available — **no DE**.
- **Why:** Extra detail for debugging or teaching; optional if you already used **Understand design**.

### **2 · Run DE** (differential expression)

- **What it does:** Runs **PyDESeq2** on the count matrix for the contrast (reference vs alternate), after gene-level prefiltering. Produces **DE gene table** (log2FC, padj, etc.).
- **Why:** This is the **DGE** step that feeds pathway enrichment.
- **Legacy cases:** Uses fixed gene lists instead of real DESeq2 on counts.

### **3 · Run ORA**

- **What it does:** **Over-representation analysis** (Fisher) of DE genes against pathway gene sets in the case. Adds overlap and ambiguity summaries when applicable.
- **Why:** Turns gene-level results into **pathway-level** hypotheses.
- **Requires:** DE must have run first in pipeline mode.

### **4 · Compare**

- **What it does:** Compares **two pathway names** you type (pathway A vs B) using DE gene support: exclusive vs shared genes.
- **Why:** Optional deeper comparison between top pathways.

### **5 · Submit**

- **What it does:** Submits your **hypothesis** as the **pathway name** (must match the case’s naming, e.g. `MAPK signaling`). Ends the episode; **reward** reflects correct vs incorrect.
- **Why:** This is the **task goal**: identify the **true activated pathway** hidden in the case.

### **Expert (accordion)**

- **What it does:** Uses a **limited budget** (per case) to surface a **hint**; usually applies a **penalty** to reward.
- **Why:** Optional curriculum / ablation; skip if `expert_budget` is 0.

---

## 6. Results area

| Area | What it shows |
|------|----------------|
| **Message / reward** (summary markdown) | Latest observation message, reward, conditions. |
| **DE genes** tab | DE table from the last step that produced DE rows. |
| **ORA** tab | Pathway enrichment table. |
| **Compare** tab | Pathway-vs-pathway comparison table when you ran **Compare**. |
| **Overlap & ambiguity** | Cross-pathway overlap and statistical ambiguity JSON snippets. |
| **Trace** | Path to the **HTML episode trace** file on disk (audit trail). |
| **Raw JSON** | Full last payload (debugging). |

If something fails, check **Raw JSON** and the docs on **`failure_code`** in [FAILURE_CODES.md](FAILURE_CODES.md).

---

## 7. Recommended first run (`toy_case_001.json`)

1. **Reset episode** (case already `toy_case_001.json`).  
2. Optional: **0 · Understand design** with empty fields → read summary.  
3. Optional: **0 · Understand design** with `control` / `treated` → validate contrast.  
4. **2 · Run DE** (or leave fields empty if you validated).  
5. **3 · Run ORA**.  
6. Optional: **4 · Compare** e.g. `MAPK signaling` vs `PI3K-Akt signaling`.  
7. **5 · Submit** with **`MAPK signaling`** (matches `true_pathway` in that case).

---

## 8. Playground tab (short)

Select **`action_type`** from the dropdown (e.g. `understand_experiment_design`, `run_differential_expression`), fill JSON fields, **Step**. Same environment as Pathway lab; no numbered buttons.

---

## See also

- [FAILURE_CODES.md](FAILURE_CODES.md) — stable `failure_code` values in metadata.  
- [README.md](../README.md) — actions table, strict mode, install.
