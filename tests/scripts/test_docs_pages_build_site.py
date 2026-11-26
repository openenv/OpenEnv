"""Regression checks for the GitHub Pages docs site built from doc-builder output."""

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "build_site", Path(__file__).parents[2] / "docs" / "pages" / "build_site.py"
)
build_site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_site)


@pytest.mark.parametrize("version", ["main", "v1.2.3"])
def test_rewrites_doc_builder_paths_for_every_version(tmp_path, monkeypatch, version):
    build = tmp_path / "build" / "openenv" / version / "en"
    (build / "_app").mkdir(parents=True)
    hub_base = f"/docs/openenv/{version}/en"
    (build / "_toctree.yml").write_text(
        "- title: Get Started\n  sections:\n  - local: index\n    title: Overview\n"
    )
    (build / "_redirects.yml").write_text("old-page: index\n")
    (build / "index.html").write_text(
        '<meta name="hf:doc:metadata" content="{&quot;title&quot;:&quot;Overview&quot;}"/>'
        "<p>OpenEnv runs <b>agentic</b> environments.</p>"
        f'<a href="{hub_base}/reference/core#x">core</a>'
        f'<script>import("{hub_base}/_app/start.js")</script>'
    )
    (build / "_app" / "start.js").write_text(f'const base = "{hub_base}";')
    (build / "llms.txt").write_text(
        f"- [Overview](https://huggingface.co/docs/openenv/{version}/index.md)\n"
    )
    out = tmp_path / "site"
    site_url = f"https://example.github.io/OpenEnv/{version}"
    monkeypatch.setattr(
        sys, "argv", ["build_site.py", str(build), str(out), "--site-url", site_url]
    )

    build_site.main()

    for path in out.rglob("*"):
        if path.suffix in {".html", ".js", ".md", ".txt"}:
            assert "/docs/openenv/" not in path.read_text(), path
    page = (out / "index.html").read_text()
    assert f'href="/OpenEnv/{version}/reference/core#x"' in page
    assert f'import("/OpenEnv/{version}/_app/start.js")' in page
    assert 'class="site-nav"' in page
    assert f'"/OpenEnv/{version}"' in (out / "_app" / "start.js").read_text()
    assert f"{site_url}/index.md" in (out / "llms.txt").read_text()
    assert f"url=/OpenEnv/{version}/" in (out / "old-page.html").read_text()
    assert (
        '<meta name="description" content="OpenEnv runs agentic environments.">' in page
    )
    assert f'<meta property="og:image" content="{site_url}/og-image.png">' in page
    assert 'name="robots"' not in page
    # Copy page / View as Markdown use the .md files, not huggingface.co's Accept header.
    assert 'input = page + ".md"' in page
    assert (out / "favicon.svg").exists() and (out / "og-image.png").exists()
