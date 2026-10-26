# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom Gradio tab for TextArena aliases."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import gradio as gr
from openenv.core.env_server.types import EnvironmentMetadata


def _wordle_demo_html() -> str:
    """Static Wordle-style grid HTML for the Custom tab (demo only)."""
    return """
<div class="wordle-demo" style="
  font-family: 'Clear Sans', 'Helvetica Neue', Arial, sans-serif;
  max-width: 320px;
  margin: 0 auto;
  padding: 16px;
  text-align: left;
">
  <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; margin-bottom: 8px;">
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;">C</div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;">R</div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;">A</div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;">N</div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;">E</div>
  </div>
  <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; margin-bottom: 8px;">
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold; background: #538d4e; color: white; border-color: #538d4e;">S</div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold; background: #538d4e; color: white; border-color: #538d4e;">T</div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold; background: #b59f3b; color: white; border-color: #b59f3b;">O</div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold; background: #538d4e; color: white; border-color: #538d4e;">N</div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold; background: #538d4e; color: white; border-color: #538d4e;">E</div>
  </div>
  <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px;">
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;"></div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;"></div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;"></div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;"></div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;"></div>
  </div>
  <p style="text-align: center; margin-top: 16px; color: #6b6b6b; font-size: 0.9rem;">
    Play in the <strong>Playground</strong> tab: Reset, then Step with guesses like <code>[crane]</code>.
  </p>
</div>
"""


def _sudoku_demo_html() -> str:
    """Static Sudoku-style grid HTML for the Custom tab (demo only)."""
    cells = []
    givens = {
        (0, 0): "5",
        (0, 1): "3",
        (0, 4): "7",
        (1, 0): "6",
        (1, 3): "1",
        (1, 4): "9",
        (1, 5): "5",
        (2, 1): "9",
        (2, 2): "8",
        (2, 7): "6",
    }
    for row in range(9):
        for col in range(9):
            value = givens.get((row, col), "")
            border_right = "3px solid #0f172a" if col in {2, 5} else "1px solid #94a3b8"
            border_bottom = "3px solid #0f172a" if row in {2, 5} else "1px solid #94a3b8"
            background = "#e2e8f0" if value else "#ffffff"
            cells.append(
                f"""
<div style="
  width: 100%;
  aspect-ratio: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
  font-weight: {'700' if value else '400'};
  color: #0f172a;
  background: {background};
  border-right: {border_right};
  border-bottom: {border_bottom};
">
  {value}
</div>
"""
            )
    return f"""
<div style="
  font-family: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
  max-width: 420px;
  margin: 0 auto;
  padding: 16px;
">
  <div style="
    display: grid;
    grid-template-columns: repeat(9, 1fr);
    border: 3px solid #0f172a;
    background: #ffffff;
  ">
    {''.join(cells)}
  </div>
  <p style="margin-top: 16px; color: #475569; font-size: 0.95rem; line-height: 1.45;">
    Use the <strong>Playground</strong> tab to reset the game and submit moves in the
    environment's expected text format.
  </p>
</div>
"""


def _resolve_textarena_ui(web_manager: Any) -> tuple[str, str, Optional[str]]:
    """Choose environment-specific copy and optional demo HTML."""
    env_id = getattr(getattr(web_manager, "env", None), "env_id", "") or ""
    normalized = env_id.lower()

    if normalized.startswith("wordle"):
        return (
            "Wordle Visualization",
            "This tab shows a **Wordle-style** view. Use the **Playground** tab to "
            "Reset and Step with guesses such as `[crane]` and `[stone]`.",
            _wordle_demo_html(),
        )
    if normalized.startswith("sudoku"):
        return (
            "Sudoku Overview",
            "This tab shows a static **Sudoku-style** board preview. Use the "
            "**Playground** tab to Reset and Step with the game's text actions.",
            _sudoku_demo_html(),
        )

    label = env_id or "TextArena"
    return (
        f"{label} Overview",
        "Use the **Playground** tab to Reset and Step through this TextArena game. "
        "The custom tab is environment-aware but only includes a static preview for "
        "selected aliases.",
        None,
    )


def build_textarena_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """Build the Custom tab Blocks for TextArena aliases."""
    heading, description, demo_html = _resolve_textarena_ui(web_manager)
    with gr.Blocks(title=f"{title} — Custom") as blocks:
        gr.Markdown(value=f"# {heading}")
        gr.Markdown(value=description)
        if demo_html is not None:
            gr.HTML(value=demo_html, show_label=False)
    return blocks
