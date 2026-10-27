# SPDX-License-Identifier: BSD-3-Clause

"""Theme for the OpenEnv Gradio UI: white, hairline borders, one green accent."""

from __future__ import annotations

import gradio as gr

_GREEN_HUE = gr.themes.Color(
    c50="#e6f4ea",
    c100="#ceead6",
    c200="#a8dab5",
    c300="#6fcc8b",
    c400="#3fb950",
    c500="#238636",
    c600="#1a7f37",
    c700="#116329",
    c800="#0a4620",
    c900="#033a16",
    c950="#04200d",
)

_NEUTRAL_HUE = gr.themes.Color(
    c50="#fafafa",
    c100="#f4f4f5",
    c200="#e4e4e7",
    c300="#d4d4d8",
    c400="#a1a1aa",
    c500="#71717a",
    c600="#52525b",
    c700="#3f3f46",
    c800="#27272a",
    c900="#18181b",
    c950="#0a0a0a",
)

OPENENV_GRADIO_THEME = gr.themes.Base(
    primary_hue=_GREEN_HUE,
    secondary_hue=_NEUTRAL_HUE,
    neutral_hue=_NEUTRAL_HUE,
    font=(gr.themes.GoogleFont("Geist"), "system-ui", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("Geist Mono"), "ui-monospace", "monospace"),
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#ffffff",
    body_text_color="#0a0a0a",
    background_fill_primary="#ffffff",
    background_fill_secondary="#fafafa",
    block_background_fill="#ffffff",
    block_border_width="0px",
    block_label_text_color="#52525b",
    block_label_background_fill="transparent",
    border_color_primary="#ecedef",
    input_background_fill="#ffffff",
    input_border_color="#e4e4e7",
    input_border_color_focus="#1a7f37",
    button_primary_background_fill="#0a0a0a",
    button_primary_background_fill_hover="#27272a",
    button_primary_text_color="#ffffff",
    button_primary_border_color="#0a0a0a",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#fafafa",
    button_secondary_text_color="#0a0a0a",
    button_secondary_border_color="#e4e4e7",
    button_border_width="1px",
    checkbox_background_color_selected="#1a7f37",
    checkbox_border_color_selected="#1a7f37",
    color_accent="#1a7f37",
    color_accent_soft="#f0f7f2",
    color_accent_soft_dark="#0d2416",
    body_text_color_subdued="#52525b",
    body_text_color_subdued_dark="#a1a1aa",
    body_background_fill_dark="#0a0a0a",
    body_text_color_dark="#fafafa",
    background_fill_primary_dark="#0a0a0a",
    background_fill_secondary_dark="#141414",
    block_background_fill_dark="#0a0a0a",
    block_label_text_color_dark="#a1a1aa",
    border_color_primary_dark="#27272a",
    input_background_fill_dark="#0a0a0a",
    input_border_color_dark="#3f3f46",
    button_primary_background_fill_dark="#fafafa",
    button_primary_background_fill_hover_dark="#e4e4e7",
    button_primary_text_color_dark="#0a0a0a",
    button_primary_border_color_dark="#fafafa",
    button_secondary_background_fill_dark="#0a0a0a",
    button_secondary_background_fill_hover_dark="#18181b",
    button_secondary_text_color_dark="#fafafa",
    button_secondary_border_color_dark="#3f3f46",
)

OPENENV_GRADIO_CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
.oe-header { padding: 24px 0 12px; }
.oe-header h1 { margin: 6px 0 4px; font-size: 40px; line-height: 1.1; font-weight: 700; letter-spacing: -.03em; }
.oe-header p, .oe-step p, .oe-episode small, .oe-muted { color: var(--body-text-color-subdued); }
.oe-header p { margin: 0; font-size: 16px; max-width: 640px; }
.oe-eyebrow { font-family: var(--font-mono); font-size: 13px; font-weight: 500; color: var(--color-accent); }
.oe-card { border: 1px solid var(--border-color-primary) !important; border-radius: 12px !important; padding: 18px 20px !important; background: var(--background-fill-primary) !important; gap: 12px !important; }
.oe-row { align-items: center !important; }
.oe-step h2, .oe-episode h2 { margin: 0; font-size: 17px; font-weight: 600; }
.oe-step p { margin: 2px 0 0; font-size: 14px; }
.oe-step code, .oe-episode code, .oe-result code { font-family: var(--font-mono); background: none; padding: 0; }
.oe-run { flex: none !important; align-self: flex-start !important; width: auto !important; min-width: 140px !important; padding: 0 22px !important; margin-left: var(--block-padding, 10px) !important; }
.oe-card textarea, .oe-card input[type=number], .oe-card input[type=password], .oe-card input[type=text] { border: 1px solid var(--input-border-color) !important; border-radius: 8px !important; padding: 10px 12px !important; font-family: var(--font-mono) !important; }
.oe-tools .wrap, .oe-actions .wrap { gap: 8px; }
.oe-tools .wrap { display: grid !important; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
.oe-actions .wrap { display: flex !important; flex-wrap: wrap; }
.oe-tools .wrap label, .oe-actions .wrap label { border: 1px solid var(--input-border-color) !important; border-radius: 10px !important; background: var(--background-fill-primary) !important; font-family: var(--font-mono); cursor: pointer; }
.oe-tools .wrap label { padding: 12px 14px !important; font-size: 13px; }
.oe-actions .wrap label { padding: 10px 16px !important; font-size: 14px; }
.oe-tools .wrap label.selected { border: 1.5px solid var(--color-accent) !important; background: var(--color-accent-soft) !important; }
.oe-actions input[type=radio] { display: none; }
.oe-result { border: 1px solid var(--border-color-primary); border-radius: 10px; overflow: hidden; }
.oe-output, .oe-fields { background: var(--background-fill-secondary); border-bottom: 1px solid var(--border-color-primary); }
.oe-output { margin: 0; padding: 14px 16px; font-family: var(--font-mono); font-size: 15px; white-space: pre-wrap; word-break: break-word; }
.oe-fields { padding: 10px 16px; }
.oe-fields div { display: grid; grid-template-columns: minmax(96px, 28%) minmax(0, 1fr); gap: 12px; padding: 3px 0; font-size: 13px; }
.oe-fields code { white-space: pre-wrap; word-break: break-word; }
.oe-fields span, .oe-stats span, .oe-episode li > span { font-family: var(--font-mono); color: var(--body-text-color-subdued); }
.oe-fields span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.oe-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); }
.oe-stats div { padding: 8px 16px; border-right: 1px solid var(--border-color-primary); }
.oe-stats div:last-child { border-right: none; }
.oe-stats span { display: block; font-size: 12px; }
.oe-stats code { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.oe-error { color: #b42318; }
.dark .oe-error { color: #ff8a80; }
.oe-episode ol { list-style: none; margin: 8px 0 0; padding: 0; }
.oe-episode li { display: flex; gap: 12px; padding: 8px 0; border-top: 1px solid var(--border-color-primary); }
.oe-episode li > span { flex: none; width: 20px; font-size: 12px; }
.oe-episode code { display: block; font-size: 13px; word-break: break-word; }
.oe-episode small { display: block; font-size: 13px; word-break: break-word; }
.oe-muted { margin: 8px 0 0; font-size: 14px; }
.oe-code pre code { white-space: pre-wrap !important; word-break: break-word; }
.oe-json .code_wrap, .oe-json .cm-editor { max-height: 360px; overflow: auto; }
.oe-code pre { background: var(--background-fill-secondary) !important; border: 1px solid var(--border-color-primary); border-radius: 8px; }
.oe-raw { border: none !important; }
.oe-visual { flex: none !important; width: auto !important; min-width: 0 !important; }
.dark { --color-accent: #3fb950; }
footer { display: none !important; }
"""
