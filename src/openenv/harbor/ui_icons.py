"""One small icon set for the Harbor UI (24px grid, 1.75 stroke, Lucide-style paths), shared by the
HTML rendered here and the components' JavaScript, so the page never falls back to emoji or glyphs."""

from __future__ import annotations

import html
import json
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def paths() -> dict[str, str]:
    return json.loads(
        resources.files("openenv.harbor")
        .joinpath("ui_assets", "icons.json")
        .read_text()
    )


def icon(name: str, size: int = 16, cls: str = "") -> str:
    return (
        f'<svg class="ic{" " + cls if cls else ""}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{paths().get(name) or paths()["file"]}</svg>'
    )


def json_tag() -> str:
    """The icon set, once per page, for every component's `icon()` to read.

    An attribute rather than a `<script type="application/json">`, which Gradio warns about in a
    `gr.HTML` because it cannot tell a data block from code that will never run.
    """
    return f'<div id="hb-icons" hidden data-icons="{html.escape(json.dumps(paths()))}"></div>'


def js_prelude() -> str:
    """`esc(text)` and `icon(name, size, cls)` for a component's `js_on_load`.

    The icons are read from the page (`json_tag`) the first time one is drawn, rather than inlined
    into each of the seven components' scripts.
    """
    return (
        "const esc = (s) => String(s ?? '').replace(/[&<>\"']/g, (c) => "
        "({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '\"': '&quot;', \"'\": '&#39;' }[c]));\n"
        "const icon = (n, s = 16, c = '') => {\n"
        "  let I = window.__hbIcons;\n"
        "  if (!I) { const el = document.getElementById('hb-icons'); I = el ? (window.__hbIcons = JSON.parse(el.dataset.icons)) : {}; }\n"
        '  return `<svg class="ic${c ? \' \' + c : \'\'}" width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" '
        "aria-hidden=\"true\">${I[n] || I.file || ''}</svg>`;\n"
        "};\n"
    )
