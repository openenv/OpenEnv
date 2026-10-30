# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Gradio **Pathway lab** tab: case selection, guided RNA-seq / ORA workflow, tables.

Mount when ``ENABLE_WEB_INTERFACE=true`` and ``create_app(..., gradio_builder=...)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
from openenv.core.env_server.types import EnvironmentMetadata

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def _list_case_files() -> List[str]:
    if not _DATA_DIR.is_dir():
        return ["toy_case_001.json"]
    names = sorted(p.name for p in _DATA_DIR.glob("*.json"))
    return names if names else ["toy_case_001.json"]


def _list_saved_runs() -> List[str]:
    """Folders under outputs/ that contain summary.json."""
    if not _OUTPUTS_DIR.is_dir():
        return []
    runs = []
    for p in sorted(_OUTPUTS_DIR.iterdir()):
        if not p.is_dir():
            continue
        if (p / "summary.json").is_file():
            runs.append(p.name)
    return runs


def _load_saved_run(run_name: str) -> Dict[str, Any]:
    base = _OUTPUTS_DIR / run_name
    root = _OUTPUTS_DIR.resolve()
    files = (base / "summary.json", base / "de_top200.json", base / "enrichment_top50.json")
    if (
        not run_name
        or run_name in {".", ".."}
        or Path(run_name).name != run_name
        or not all(path.resolve().is_relative_to(root) for path in files)
    ):
        return {
            "run": run_name,
            "summary_error": "Invalid run name",
            "summary": {},
            "de": [],
            "enrichment": [],
        }
    out: Dict[str, Any] = {"run": run_name}
    try:
        out["summary"] = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    except Exception as e:
        out["summary_error"] = str(e)
        out["summary"] = {}
    try:
        out["de"] = json.loads((base / "de_top200.json").read_text(encoding="utf-8"))
    except Exception as e:
        out["de_error"] = str(e)
        out["de"] = []
    try:
        out["enrichment"] = json.loads(
            (base / "enrichment_top50.json").read_text(encoding="utf-8")
        )
    except Exception as e:
        out["enrichment_error"] = str(e)
        out["enrichment"] = []
    return out


def _saved_run_to_tables(run: Dict[str, Any]) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    s = run.get("summary") or {}
    md_lines = [
        f"**Run:** `{run.get('run','')}`",
        f"**Case:** `{s.get('case_id','')}`",
        f"**Contrast:** `{s.get('contrast')}`",
        f"**Genes:** in matrix `{s.get('genes_in_matrix')}` → after prefilter `{s.get('genes_after_prefilter')}`",
        f"**Trace:** `{s.get('trace_path','')}`",
    ]
    md = "\n\n".join(md_lines)

    de = run.get("de") or []
    df_de = pd.DataFrame(de) if isinstance(de, list) and de else pd.DataFrame({"info": ["No DE export found."]})

    enr = run.get("enrichment") or []
    if isinstance(enr, list) and enr:
        df_enr = pd.DataFrame(
            [
                {
                    "pathway": r.get("pathway"),
                    "p_value": r.get("p_value"),
                    "q_value": r.get("q_value"),
                    "odds_ratio": r.get("odds_ratio"),
                    "overlap_genes": ", ".join((r.get("overlap_genes") or [])[:40]),
                }
                for r in enr
                if isinstance(r, dict)
            ]
        )
    else:
        df_enr = pd.DataFrame({"info": ["No enrichment export found."]})

    return md, df_de, df_enr


def _pathway_comparison_df(obs: Dict[str, Any]) -> pd.DataFrame:
    pc = obs.get("pathway_comparison")
    if not isinstance(pc, dict) or not pc:
        return pd.DataFrame(
            {
                "": [
                    "Run **Compare pathways** with two pathway names to see exclusive vs shared DE support."
                ]
            }
        )
    a = pc.get("pathway_a", "")
    b = pc.get("pathway_b", "")
    a_size = pc.get("pathway_a_size")
    b_size = pc.get("pathway_b_size")
    rows = [
        {
            "pathway": f"A only ({a})",
            "count": len(pc.get("exclusive_to_a") or []),
            "genes (preview)": ", ".join((pc.get("exclusive_to_a") or [])[:40]),
        },
        {
            "pathway": f"B only ({b})",
            "count": len(pc.get("exclusive_to_b") or []),
            "genes (preview)": ", ".join((pc.get("exclusive_to_b") or [])[:40]),
        },
        {
            "pathway": "Shared DE support",
            "count": len(pc.get("shared_de_support") or []),
            "genes (preview)": ", ".join((pc.get("shared_de_support") or [])[:40]),
        },
        {
            "pathway": "Pathway gene-set sizes",
            "count": a_size + b_size
            if a_size is not None and b_size is not None
            else "Unknown",
            "genes (preview)": (
                f"A size={a_size if a_size is not None else 'Unknown'} · "
                f"B size={b_size if b_size is not None else 'Unknown'}"
            ),
        },
    ]
    return pd.DataFrame(rows)


def _state_markdown(st: Dict[str, Any]) -> str:
    """Episode banner from ``WebInterfaceManager.get_state()`` (PathwayState fields)."""
    if not st:
        return "*No state yet — reset an episode.*"
    eid = str(st.get("episode_id") or "")
    eid_short = f"`{eid[:10]}…`" if len(eid) > 10 else f"`{eid}`"
    pipe = "counts + PyDESeq2" if st.get("pipeline_mode") else "legacy lists"
    strict = "strict" if st.get("strict_mode") else "lenient"
    de_ok = "✓" if st.get("de_run") else "○"
    ora_ok = "✓" if st.get("enrichment_run") else "○"
    ex_u = int(st.get("expert_calls_used") or 0)
    ex_b = int(st.get("expert_budget") or 0)
    expert = f"{ex_u} / {ex_b}" if ex_b > 0 else "off"
    done = "**Episode ended.** Reset to start over." if st.get("is_done") else ""
    conds = st.get("conditions") or []
    cond_line = ", ".join(f"`{c}`" for c in conds[:12]) if conds else "—"
    vref = st.get("validated_reference")
    valt = st.get("validated_alternate")
    val_line = ""
    if vref and valt:
        val_line = f"\n\n**Validated contrast (for DE if fields empty):** `{vref}` → `{valt}`"
    des = "✓" if st.get("design_understood") else "○"
    return (
        f"**Episode** {eid_short} · step **{st.get('step_count', 0)}** · {pipe} · {strict}\n\n"
        f"**Conditions in case:** {cond_line}\n\n"
        f"**Pipeline:** Design {des} · DE {de_ok} · ORA {ora_ok} · **Expert calls** {expert}"
        f"{val_line}\n\n"
        f"{done}"
    ).strip()


def _observation_to_tables(
    data: Dict[str, Any],
) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str, str]:
    """Markdown summary, DE df, ORA df, compare df, overlap/ambiguity, trace, raw JSON."""
    obs = data.get("observation") or {}
    if not isinstance(obs, dict):
        obs = {}

    msg = obs.get("message", "") or ""
    reward = obs.get("reward")
    done = obs.get("done")
    lines = [
        f"**Message:** {msg}",
        f"**Reward:** `{reward}`  ·  **Done:** `{done}`",
    ]
    ac = obs.get("available_conditions") or []
    if ac:
        lines.append("**Conditions (from last step):** " + ", ".join(f"`{c}`" for c in ac[:20]))

    ed = obs.get("experiment_design")
    if isinstance(ed, dict) and ed:
        lines.append(
            "**Experiment design (structured):**\n```json\n"
            + json.dumps(ed, indent=2, default=str)[:8000]
            + "\n```"
        )

    comparison = obs.get("pathway_comparison") or {}
    if comparison.get("comparison_scope") == "enrichr_overlap_genes":
        lines.append(
            "**Comparison scope:** Enrichr overlap genes only; "
            "exclusive/shared support is relative to the returned query overlaps, "
            "and full pathway membership is unknown."
        )

    md = "\n\n".join(lines)

    de_rows = obs.get("de_genes") or []
    if de_rows and isinstance(de_rows, list):
        df_de = pd.DataFrame(de_rows[:200])
    else:
        top = obs.get("top_genes") or []
        if top:
            df_de = pd.DataFrame({"gene": top})
        else:
            df_de = pd.DataFrame(
                {"info": ["No DE table yet — run differential expression."]}
            )

    pe = obs.get("pathway_enrichment") or []
    if pe and isinstance(pe, list):
        rows_flat = []
        for r in pe[:80]:
            if not isinstance(r, dict):
                continue
            rows_flat.append(
                {
                    "pathway": r.get("pathway", ""),
                    "p_value": r.get("p_value"),
                    "q_value": r.get("q_value"),
                    "overlap_count": r.get("overlap_count"),
                    "pathway_size": r.get("pathway_size"),
                    "gene_ratio": r.get("gene_ratio", ""),
                }
            )
        df_pe = (
            pd.DataFrame(rows_flat)
            if rows_flat
            else pd.DataFrame({"info": ["No ORA results — run pathway enrichment."]})
        )
    else:
        tp = obs.get("top_pathways") or []
        if tp:
            df_pe = pd.DataFrame({"pathway": tp})
        else:
            df_pe = pd.DataFrame(
                {"info": ["No ORA table yet — run pathway enrichment."]}
            )

    df_cmp = _pathway_comparison_df(obs)

    ov = obs.get("overlap_summary") or {}
    amb = obs.get("statistical_ambiguity") or {}
    extra = []
    if ov:
        extra.append(
            "**Overlap across top pathways:**\n```json\n"
            + json.dumps(ov, indent=2)[:4000]
            + "\n```"
        )
    if amb:
        extra.append(
            "**Statistical ambiguity:**\n```json\n"
            + json.dumps(amb, indent=2)[:2000]
            + "\n```"
        )
    extra_txt = "\n\n".join(extra) if extra else "*No overlap / ambiguity data yet.*"

    em = obs.get("expert_message")
    if em:
        extra_txt = (extra_txt + f"\n\n**Expert:** {em}").strip()

    trace = obs.get("trace_path") or ""
    trace_md = (
        f"**HTML episode trace:** `{trace}`\n\n"
        f"Open the file locally to audit each step in a browser."
        if trace
        else "*Trace file path appears after environment steps.*"
    )

    raw = json.dumps(data, indent=2, default=str)
    return md, df_de, df_pe, df_cmp, extra_txt, trace_md, raw


def _response(
    data: Dict[str, Any],
    web_manager: Any,
    status: str,
    update_contrast: bool,
) -> Tuple[Any, ...]:
    """Shared outputs for all steps; optionally refresh contrast textboxes from state."""
    md, df_de, df_pe, df_cmp, extra, trace_md, raw = _observation_to_tables(data)
    st = web_manager.get_state()
    state_md = _state_markdown(st if isinstance(st, dict) else {})
    conds = (st or {}).get("conditions") or []
    if update_contrast and conds:
        design = (data.get("observation") or {}).get("experiment_design") or {}
        contrast = design.get("default_contrast") or {}
        ref_v = str(contrast.get("reference") or conds[0])
        alt_v = str(contrast.get("alternate") or (conds[1] if len(conds) > 1 else conds[0]))
        cref, calt = gr.update(value=ref_v), gr.update(value=alt_v)
    else:
        cref, calt = gr.update(), gr.update()
    return (
        md,
        df_de,
        df_pe,
        df_cmp,
        extra,
        trace_md,
        raw,
        state_md,
        status,
        cref,
        calt,
    )


def build_pathway_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """
    Second tab (**Visualization**) for pathway_analysis_env: interactive pathway lab.

    Uses ``web_manager.env.set_case_file`` before reset, and ``step_environment`` with
    structured ``PathwayAction`` payloads.
    """
    case_choices = _list_case_files()
    saved_runs = _list_saved_runs()
    display = metadata.name if metadata else title

    async def do_reset(case_file: str):
        try:
            if hasattr(web_manager.env, "set_case_file"):
                web_manager.env.set_case_file(case_file)
            data = await web_manager.reset_environment()
            return _response(
                data,
                web_manager,
                f"Loaded case `{case_file}` and reset.",
                update_contrast=True,
            )
        except Exception as e:
            empty = pd.DataFrame({"error": [str(e)]})
            z = gr.update()
            return (
                "",
                empty,
                empty,
                empty,
                "",
                "",
                "",
                f"*Error:* `{e}`",
                str(e),
                z,
                z,
            )

    async def step_inspect():
        data = await web_manager.step_environment({"action_type": "inspect_dataset"})
        return _response(
            data,
            web_manager,
            "Inspect complete.",
            update_contrast=False,
        )

    async def step_understand(cond_a: str, cond_b: str):
        payload: Dict[str, Any] = {
            "action_type": "understand_experiment_design",
            "condition_a": (cond_a or "").strip() or None,
            "condition_b": (cond_b or "").strip() or None,
        }
        data = await web_manager.step_environment(payload)
        return _response(
            data,
            web_manager,
            "Understand experiment design complete.",
            update_contrast=False,
        )

    async def step_de(cond_a: str, cond_b: str):
        payload: Dict[str, Any] = {
            "action_type": "run_differential_expression",
            "condition_a": (cond_a or "").strip() or None,
            "condition_b": (cond_b or "").strip() or None,
        }
        data = await web_manager.step_environment(payload)
        return _response(
            data,
            web_manager,
            "Differential expression complete.",
            update_contrast=False,
        )

    async def step_ora():
        data = await web_manager.step_environment(
            {"action_type": "run_pathway_enrichment"}
        )
        return _response(
            data,
            web_manager,
            "ORA complete.",
            update_contrast=False,
        )

    async def step_compare(pa: str, pb: str):
        data = await web_manager.step_environment(
            {
                "action_type": "compare_pathways",
                "pathway_a": (pa or "").strip(),
                "pathway_b": (pb or "").strip(),
            }
        )
        return _response(
            data,
            web_manager,
            "Compare complete.",
            update_contrast=False,
        )

    async def step_expert(q: str):
        data = await web_manager.step_environment(
            {
                "action_type": "ask_expert",
                "expert_question": (q or "").strip() or None,
            }
        )
        return _response(
            data,
            web_manager,
            "Expert step complete.",
            update_contrast=False,
        )

    async def step_submit(hyp: str):
        data = await web_manager.step_environment(
            {"action_type": "submit_answer", "hypothesis": (hyp or "").strip()}
        )
        return _response(
            data,
            web_manager,
            "Answer submitted.",
            update_contrast=False,
        )

    with gr.Blocks(title=f"{display} — Pathway lab") as blocks:
        gr.Markdown(
            f"# Pathway lab\n\n"
            f"**Agent-style flow:** **(1) Groups & design** — how many conditions and samples per group. "
            f"**(2) DGE** — pick reference vs alternate, then differential expression. "
            f"**(3) Pathways** — ORA, compare, submit hypothesis. "
            f"Buttons: Understand design → Inspect → Run DE → Run ORA → … Use **Playground** for raw actions.\n\n"
            f"---"
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                gr.Markdown("**Case & episode**")
                with gr.Row():
                    case_dd = gr.Dropdown(
                        choices=case_choices,
                        value=case_choices[0] if case_choices else None,
                        label="Case JSON (`data/`)",
                        scale=2,
                    )
                    reset_btn = gr.Button("Reset episode", variant="primary", scale=1)
            with gr.Column(scale=3):
                out_state = gr.Markdown(label="Episode state")
                out_status = gr.Textbox(label="Last action", max_lines=2)

        gr.Markdown("### Contrast (PyDESeq2 pipeline cases)")
        gr.Markdown(
            "*Reference* = baseline condition, *alternate* = treatment. "
            "Reset fills these from the case when possible; edit if needed. "
            "**Understand design:** leave both empty for a structured summary only, or fill both to validate the contrast (used by **Run DE** when those fields are left empty)."
        )
        with gr.Row():
            cond_ref = gr.Textbox(
                label="Reference condition",
                placeholder="e.g. control",
                lines=1,
            )
            cond_alt = gr.Textbox(
                label="Alternate condition",
                placeholder="e.g. treated",
                lines=1,
            )

        gr.Markdown("#### Workflow")
        with gr.Row():
            btn_ud = gr.Button("0 · Understand design", variant="secondary")
            btn_ins = gr.Button("1 · Inspect", variant="secondary")
            btn_de = gr.Button("2 · Run DE", variant="primary")
            btn_ora = gr.Button("3 · Run ORA", variant="primary")
        with gr.Row():
            pw_a = gr.Textbox(label="Pathway A", placeholder="MAPK signaling", scale=1)
            pw_b = gr.Textbox(label="Pathway B", placeholder="PI3K-Akt", scale=1)
            btn_cmp = gr.Button("4 · Compare", scale=0, min_width=120)
        with gr.Row():
            hyp = gr.Textbox(
                label="Hypothesis (pathway name)",
                placeholder="True activated pathway",
                scale=2,
            )
            btn_sub = gr.Button("5 · Submit", variant="stop", scale=0, min_width=120)

        with gr.Accordion("Expert (optional, budgeted per case)", open=False):
            with gr.Row():
                ex_q = gr.Textbox(
                    label="Question / note",
                    placeholder="Optional — uses expert budget",
                    scale=3,
                )
                btn_ex = gr.Button("Ask expert", scale=0, min_width=120)

        gr.Markdown("### Results")
        out_md = gr.Markdown()
        with gr.Tabs():
            with gr.Tab("DE genes"):
                out_de = gr.Dataframe(
                    label="Differential expression",
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("ORA"):
                out_ora = gr.Dataframe(
                    label="Pathway enrichment",
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("Compare"):
                out_cmp = gr.Dataframe(
                    label="Pathway vs pathway",
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("Saved run (GSE235417)"):
                gr.Markdown(
                    "Browse exported run artifacts under `envs/pathway_analysis_env/outputs/<run>/` "
                    "(e.g. `gse235417`). This view does **not** re-run DESeq2; it only loads saved JSON."
                )
                run_dd = gr.Dropdown(
                    choices=saved_runs,
                    value="gse235417" if "gse235417" in saved_runs else (saved_runs[0] if saved_runs else None),
                    label="Saved run folder (`outputs/`)",
                )
                load_btn = gr.Button("Load saved results", variant="primary")
                run_md = gr.Markdown()
                run_de = gr.Dataframe(label="Saved DE (top 200)", interactive=False, wrap=True)
                run_enr = gr.Dataframe(label="Saved enrichment (top 50)", interactive=False, wrap=True)
            with gr.Tab("Overlap & ambiguity"):
                out_extra = gr.Markdown()
            with gr.Tab("Trace"):
                out_trace = gr.Markdown()
            with gr.Tab("Raw JSON"):
                out_raw = gr.Code(label="Wire payload", language="json", interactive=False)

        ui_outputs = [
            out_md,
            out_de,
            out_ora,
            out_cmp,
            out_extra,
            out_trace,
            out_raw,
            out_state,
            out_status,
            cond_ref,
            cond_alt,
        ]

        reset_btn.click(fn=do_reset, inputs=[case_dd], outputs=ui_outputs)
        btn_ud.click(fn=step_understand, inputs=[cond_ref, cond_alt], outputs=ui_outputs)
        btn_ins.click(fn=step_inspect, outputs=ui_outputs)
        btn_de.click(fn=step_de, inputs=[cond_ref, cond_alt], outputs=ui_outputs)
        btn_ora.click(fn=step_ora, outputs=ui_outputs)
        btn_cmp.click(fn=step_compare, inputs=[pw_a, pw_b], outputs=ui_outputs)
        btn_sub.click(fn=step_submit, inputs=[hyp], outputs=ui_outputs)
        btn_ex.click(fn=step_expert, inputs=[ex_q], outputs=ui_outputs)

        def do_load_saved(run_name: str):
            run = _load_saved_run(run_name or "")
            md, df_de, df_enr = _saved_run_to_tables(run)
            return md, df_de, df_enr

        load_btn.click(fn=do_load_saved, inputs=[run_dd], outputs=[run_md, run_de, run_enr])

    return blocks
