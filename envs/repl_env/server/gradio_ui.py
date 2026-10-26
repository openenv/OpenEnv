# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom Gradio tab for the REPL environment."""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
from openenv.core.env_server.types import EnvironmentMetadata


_FINAL_RE = re.compile(r"FINAL\(([^\n]+?)\)\s*$", re.MULTILINE)


def _extract_final(stdout: str) -> Optional[str]:
    """Pull the final answer from the last `FINAL(...)` line of stdout."""
    matches = _FINAL_RE.findall(stdout or "")
    return matches[-1].strip() if matches else None


try:
    import pypdf
except ImportError:  # pragma: no cover - optional dep
    pypdf = None  # type: ignore[assignment]


_EX1_NEEDLE_CONTEXT = """ACME Robotics — Q1 internal changelog, build notes, and incident log.

Week 1. Fleet rollout reached 412 units across warehouses A and B. Latency regression on the manipulator firmware was traced back to the new servo driver; hotfix 1.4.2 shipped Wednesday. No customer impact.

Week 2. Vision team rotated the training dataset to include low-light scenes. Offline eval improved 3.1% on the occluded-grasping benchmark but regressed 0.8% on the clean-table benchmark. Product signed off on the trade-off.

Week 3. Security audit flagged credential handling in the pairing flow. The legacy MQTT broker was retired. Any robot still on firmware < 1.3.0 will refuse to pair after May 1 — document this in the migration note. The activation code is BANANA-747. Do not share externally. Re-keying ceremony for the HSM happens next sprint.

Week 4. Onboarding four new SREs. Runbook for the arm-joint recalibration was rewritten from scratch after the Tuesday incident. New runbook lives under ops/arm/joint-recal/v2.md. The Tuesday incident: a collision with a pallet rack during autonomous pickup. Root cause: stale depth-map cache. Mitigation: cache TTL dropped from 5 minutes to 20 seconds.

Week 5. Starting the unification work on the control plane. Two clusters are being collapsed into one, with staged traffic shifting. Rollback criteria are defined in the RFC. Early signal looks good: no increase in p99 pick latency. Next milestone: full cutover.

Week 6. Offsite in Porto. Platform team proposed a new interface between the planner and the low-level controller. Discussion deferred to the RFC process. Interesting prior art from the MIT paper on whole-body MPC. Follow-up reading list shared in the team channel."""

_EX1_TASK = "Find the activation code hidden somewhere in the changelog."

_EX1_CODE = """# Needle in a haystack: fan out direct LM calls over chunks in parallel.
chunks = [context[i:i+300] for i in range(0, len(context), 300)]
prompts = [
    "In the text below, find any hyphenated identifier of the form LETTERS-DIGITS "
    "(for example: ABC-123, FOO-42). Reply with just that identifier. "
    f"If there is none, reply with the single word NONE.\\n\\n---\\n{c}"
    for c in chunks
]
answers = llm_query_batched(prompts)
hit = next((a.strip() for a in answers if "NONE" not in a.upper()), "not found")
print(f"spawned {len(chunks)} children")
print(FINAL(hit))"""


_EX2_CONTEXT = ""
_EX2_TASK = "Show the recursive primitive: the child runs its own REPL agent."
_EX2_CODE = """# rlm_query spawns a full recursive REPL agent as a child.
# The child can reason, run code, and finalize with FINAL(...).
answer = rlm_query("What is 15 * 23? Work it out step by step and respond with only the final number.")
print(f"Child returned: {answer!r}")
print(FINAL(answer.strip()))"""


_EXAMPLES: Dict[str, tuple[str, str, str, str]] = {
    # 4th element is `expected_answer`: RFC 004 ground truth, matched
    # against FINAL(...) for reward scoring.
    "Needle in a haystack (llm_query_batched)": (
        _EX1_NEEDLE_CONTEXT,
        _EX1_TASK,
        _EX1_CODE,
        "BANANA-747",
    ),
    "Recursive child agent (rlm_query)": (
        _EX2_CONTEXT,
        _EX2_TASK,
        _EX2_CODE,
        "345",
    ),
}


_HELPERS_MD = """
### REPL helpers

These names are injected into the Python namespace once you Reset:

- `context` — the string you passed in the *Context* field, available as a Python variable.
- `llm_query(prompt, model=None)` — single direct call to the configured LLM.
- `llm_query_batched(prompts, model=None)` — fan out N direct LLM calls in parallel.
- `rlm_query(prompt)` / `rlm_query_batched(prompts)` — each child runs a full recursive REPL loop (deeper RLM pattern).
- `FINAL(value)` — finalize the episode with `value` as the answer.
- `FINAL_VAR("name")` — finalize with the value of the named variable.
- `answer = {"content": ..., "ready": True}` — dict-based finalization.

### Rewards (RFC 004)

Pass **Expected answer** at Reset to enable rubric-based scoring: the observation's `reward` field becomes `1.0` when `FINAL(...)` matches (exact match, case-insensitive strip) and `0.0` otherwise. Leave blank to skip scoring.

### Typical flow

1. Paste your **Hugging Face Token** — required for any example that calls `llm_query` / `rlm_query`.
2. Write Context + Task Prompt (or pick a demo from **💡 Load example**).
3. Click **🔁 Reset episode** — loads your context and wires up the helpers above.
4. Write Python in *Python Code* that uses those helpers and ends with `FINAL(...)`.
5. Click **▶ Run**. Inspect *Stdout* and *Raw JSON response* (the latter contains child-call metadata for recursive runs).
"""


_MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB — comfortably fits any arXiv PDF


def _extract_text_from_upload(file_path: str) -> str:
    """Read a .txt or .pdf upload from disk and return its text content.

    Runs in the Gradio server process (outside the smolagents sandbox),
    so it can use `pypdf` and the filesystem without triggering the
    REPL's import restrictions.
    """
    path = Path(file_path)
    size = path.stat().st_size
    if size > _MAX_UPLOAD_BYTES:
        return (
            f"[File too large ({size // (1024 * 1024)} MB). "
            f"Cap is {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB to protect "
            "the shared Space process from OOM.]"
        )
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        if pypdf is None:
            return "[pypdf is not installed in this deployment; cannot parse PDFs.]"
        with path.open("rb") as fh:
            reader = pypdf.PdfReader(io.BytesIO(fh.read()))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:  # pypdf can raise on malformed PDFs
                pages.append(f"[page extraction failed: {exc}]")
        text = "\n\n".join(pages).strip()
        return text or "[No extractable text — PDF may be image-only.]"
    return f"[Unsupported file type: {suffix}. Use .txt or .pdf.]"


def _code_block(title: str, content: str) -> str:
    if not content:
        return ""
    return f"**{title}:**\n```text\n{content}\n```"


def _format_repl_response(
    data: Dict[str, Any],
    elapsed_s: Optional[float] = None,
) -> str:
    """Render REPL observations as a compact, screenshot-friendly view.

    The FINAL answer is surfaced at the top as a quoted block, followed by
    the task prompt echo, then stdout / stderr, and a one-line metadata
    strip. Context preview / locals snapshot / available variables are
    still in the raw JSON pane for debugging.
    """
    observation = data.get("observation", {})
    result = observation.get("result", {})

    sections: List[str] = []

    stdout = (result.get("stdout") or "").rstrip()
    final_answer = _extract_final(stdout) if stdout else None
    if final_answer:
        sections.append(f"### 🎯 Answer\n\n> **{final_answer}**")

    task_prompt = observation.get("metadata", {}).get("task_prompt")
    if task_prompt:
        sections.append(f"**Task:** {task_prompt}")

    if stdout:
        sections.append(_code_block("Stdout", stdout))

    stderr = (result.get("stderr") or "").rstrip()
    if stderr:
        sections.append(_code_block("Stderr", stderr))

    meta: List[str] = []
    reward = data.get("reward")
    if reward is not None:
        meta.append(f"Reward `{reward}`")
    done = data.get("done")
    if done is not None:
        meta.append(f"Done `{done}`")
    iteration = observation.get("iteration")
    max_iter = observation.get("max_iterations")
    if iteration is not None and max_iter is not None:
        meta.append(f"Step `{iteration}/{max_iter}`")
    if elapsed_s is not None:
        meta.append(f"Elapsed `{elapsed_s:.1f}s`")
    if meta:
        sections.append(" · ".join(meta))

    if not sections:
        return "_No output yet. Reset and Run._"
    return "\n\n".join(sections)


def build_repl_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """Build the REPL-specific Gradio tab."""
    del action_fields, is_chat_env

    # Skip `metadata.description`: the core default is a generic
    # "<name> environment" placeholder that adds noise without content.
    readme_content = (
        metadata.readme_content if metadata and metadata.readme_content else ""
    )

    async def reset_repl(
        context: Optional[str],
        task_prompt: Optional[str],
        expected_answer: Optional[str],
        hf_token: Optional[str],
        llm_model: Optional[str],
    ):
        # Gradio can pass None for untouched password textboxes. Normalize here.
        reset_kwargs: Dict[str, Any] = {}
        if context and context.strip():
            reset_kwargs["context"] = context
        if task_prompt and task_prompt.strip():
            reset_kwargs["task_prompt"] = task_prompt
        if expected_answer and expected_answer.strip():
            reset_kwargs["expected_answer"] = expected_answer.strip()
        if hf_token and hf_token.strip():
            reset_kwargs["hf_token"] = hf_token.strip()
        if llm_model and llm_model.strip():
            reset_kwargs["llm_model"] = llm_model

        try:
            start = time.monotonic()
            data = await web_manager.reset_environment(reset_kwargs)
            elapsed_s = time.monotonic() - start
            state = web_manager.get_state()
            return (
                _format_repl_response(data, elapsed_s=elapsed_s),
                json.dumps(data, indent=2, sort_keys=True),
                json.dumps(state, indent=2, sort_keys=True),
                f"Reset complete in {elapsed_s:.1f}s.",
            )
        except Exception as exc:
            return ("", "", "", f"Error: {exc}")

    async def run_code(code: Optional[str]):
        if not code or not code.strip():
            return ("", "", "", "Enter Python code to run.")

        try:
            start = time.monotonic()
            data = await web_manager.step_environment({"code": code})
            elapsed_s = time.monotonic() - start
            state = web_manager.get_state()
            return (
                _format_repl_response(data, elapsed_s=elapsed_s),
                json.dumps(data, indent=2, sort_keys=True),
                json.dumps(state, indent=2, sort_keys=True),
                f"Code executed in {elapsed_s:.1f}s.",
            )
        except Exception as exc:
            return ("", "", "", f"Error: {exc}")

    def get_state_sync():
        try:
            return json.dumps(web_manager.get_state(), indent=2, sort_keys=True)
        except Exception as exc:
            return f"Error: {exc}"

    def load_example(label: str):
        if not label or label not in _EXAMPLES:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )
        ctx, task, code, expected = _EXAMPLES[label]
        # Auto-collapse the Context accordion once populated to keep the
        # viewport focused on Task + Code + Output.
        return ctx, task, code, expected, gr.update(open=False)

    def load_uploaded_document(file_obj):
        if file_obj is None:
            return gr.update(), gr.update(), gr.update()
        file_path = (
            file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
        )
        if not file_path:
            return gr.update(), "Upload error: could not read file path.", gr.update()
        try:
            text = _extract_text_from_upload(file_path)
        except Exception as exc:
            return gr.update(), f"Upload error: {exc}", gr.update()
        return (
            text,
            f"Loaded {Path(file_path).name} ({len(text)} chars) into Context.",
            gr.update(open=False),
        )

    with gr.Blocks(title=title) as blocks:
        gr.Markdown(
            "# REPL Control Panel\n"
            "*Recursive Language Model REPL — run agentic Python with recursive LM calls.*"
        )

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    example_dropdown = gr.Dropdown(
                        label="💡 Load example",
                        choices=list(_EXAMPLES.keys()),
                        value=None,
                    )
                    upload = gr.File(
                        label="📎 Or drop a .txt / .pdf",
                        file_types=[".txt", ".pdf"],
                        type="filepath",
                    )
                task_prompt = gr.Textbox(
                    label="Task Prompt",
                    placeholder="What should the agent solve?",
                    lines=2,
                )
                with gr.Accordion("Context", open=True) as context_accordion:
                    context = gr.Textbox(
                        label="",
                        placeholder="Problem context or source text (auto-fills from Load example / Drop)...",
                        lines=3,
                        show_label=False,
                    )
                with gr.Accordion("Advanced options", open=False):
                    hf_token = gr.Textbox(
                        label="Hugging Face Token (required for llm_query / rlm_query)",
                        placeholder="hf_...  — used only for this reset; not persisted",
                        type="password",
                    )
                    expected_answer = gr.Textbox(
                        label="Expected answer (activates reward scoring)",
                        placeholder="e.g. BANANA-747  —  leave blank to skip",
                    )
                    llm_model = gr.Textbox(
                        label="LLM Model",
                        placeholder="e.g. Qwen/Qwen2.5-7B-Instruct",
                        info=(
                            "Blank = server default (Qwen/Qwen3.5-9B, tuned for the "
                            "`enable_thinking=False` chat template)."
                        ),
                    )
                with gr.Row():
                    reset_btn = gr.Button("🔁 Reset episode", variant="secondary")
                    status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=1,
                        scale=1,
                    )
            with gr.Column(scale=3):
                code = gr.Textbox(
                    label="Python Code",
                    placeholder="count = len(context.split())",
                    lines=8,
                )
                with gr.Row():
                    run_btn = gr.Button("▶ Run", variant="primary")
                    state_btn = gr.Button("📋 Get state", variant="secondary")
                session_view = gr.Markdown(value="_Reset and Run to see output._")
                with gr.Accordion("Raw JSON response", open=False):
                    raw_json = gr.Code(
                        label="",
                        language="json",
                        interactive=False,
                    )
                with gr.Accordion("Session state", open=False):
                    state_json = gr.Code(
                        label="",
                        language="json",
                        interactive=False,
                    )

        with gr.Accordion("REPL helpers reference & typical flow", open=False):
            gr.Markdown(_HELPERS_MD)

        if quick_start_md:
            with gr.Accordion("API quick start (Python client snippet)", open=False):
                gr.Markdown(quick_start_md)

        if readme_content:
            with gr.Accordion("Full README", open=False):
                gr.Markdown(readme_content)

        reset_btn.click(
            fn=reset_repl,
            inputs=[context, task_prompt, expected_answer, hf_token, llm_model],
            outputs=[session_view, raw_json, state_json, status],
        )
        run_btn.click(
            fn=run_code,
            inputs=[code],
            outputs=[session_view, raw_json, state_json, status],
        )
        code.submit(
            fn=run_code,
            inputs=[code],
            outputs=[session_view, raw_json, state_json, status],
        )
        state_btn.click(fn=get_state_sync, outputs=[state_json])
        example_dropdown.change(
            fn=load_example,
            inputs=[example_dropdown],
            outputs=[
                context,
                task_prompt,
                code,
                expected_answer,
                context_accordion,
            ],
        )
        upload.change(
            fn=load_uploaded_document,
            inputs=[upload],
            outputs=[context, status, context_accordion],
        )

    return blocks
