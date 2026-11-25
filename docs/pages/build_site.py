"""Turn a doc-builder HTML build into a standalone static site for GitHub Pages.

doc-builder renders each page as a fragment that huggingface.co wraps in its own
shell (navbar, sidebar, styles). This script adds that shell: a full HTML
document per page with a sidebar built from `_toctree.yml`, an "On this page"
list, the stylesheet compiled by `build_css.mjs`, and redirect pages from
`_redirects.yml`.

Usage:
    python docs/pages/build_site.py <doc-builder build dir> <output dir> \
        --site-url https://openenv.github.io/OpenEnv/main --version-label main \
        --versions-url https://openenv.github.io/OpenEnv/versions.json
"""

import argparse
import dataclasses
import html
import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

import yaml

REPO_URL = "https://github.com/openenv/OpenEnv"
SHELL_DIR = Path(__file__).parent

METADATA_RE = re.compile(r'<meta name="hf:doc:metadata" content="([^"]*)"\s*/?>')
CSS_PRELOAD_RE = re.compile(r'<link href="[^"]*\.css" rel="modulepreload">\n?')
PARAGRAPH_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.S)
# Icons and social card, copied to the site root.
ASSETS = ["favicon.svg", "favicon.png", "apple-touch-icon.png", "og-image.png"]


@dataclasses.dataclass
class Site:
    url: str  # e.g. https://openenv.github.io/OpenEnv/main
    base: str  # URL path of the site, e.g. /OpenEnv/main
    toctree: list
    version: str
    versions_url: str | None
    noindex: bool


def page_url(base: str, local: str) -> str:
    return f"{base}/" if local == "index" else f"{base}/{local}"


def render_sidebar(sections: list, base: str, current: str) -> tuple[str, bool]:
    """Returns the sidebar HTML for `sections` and whether it contains `current`."""
    items, has_current = [], False
    for section in sections:
        if "sections" in section:
            inner, inner_current = render_sidebar(section["sections"], base, current)
            has_current |= inner_current
            is_open = inner_current or section.get("isExpanded", True)
            items.append(
                f"<li><details{' open' if is_open else ''}>"
                f"<summary>{html.escape(section['title'])}</summary>{inner}</details></li>"
            )
        else:
            is_current = section["local"] == current
            has_current |= is_current
            items.append(
                f'<li><a href="{page_url(base, section["local"])}"'
                f"{' aria-current=page' if is_current else ''}>"
                f"{html.escape(section['title'])}</a></li>"
            )
    return f"<ul>{''.join(items)}</ul>", has_current


def render_toc(sections: list) -> str:
    items = []
    for section in sections:
        sub = render_toc(section["sections"]) if section.get("depth", 2) < 3 else ""
        items.append(
            f'<li><a href="#{section["local"]}">{html.escape(section["title"])}</a>{sub}</li>'
        )
    return f"<ul>{''.join(items)}</ul>" if items else ""


VERSION_SCRIPT = """<script>
  const select = document.querySelector(".version-select");
  fetch(select.dataset.versionsUrl)
    .then((response) => response.json())
    .then((versions) => {
      if (!versions.includes(select.dataset.current)) versions.unshift(select.dataset.current);
      select.replaceChildren(...versions.map((v) => new Option(v, v, false, v === select.dataset.current)));
    })
    .catch(() => {});
  select.addEventListener("change", () => {
    const root = select.dataset.versionsUrl.replace(/\\/versions\\.json$/, "");
    location.href = `${root}/${select.value}/${select.dataset.page}`;
  });
</script>"""


def render_version(version: str, versions_url: str | None, local: str) -> str:
    """Version picker when the site lists its versions, a plain label otherwise."""
    if not versions_url:
        return f'<span class="version-badge">{html.escape(version)}</span>'
    page = "" if local in {"index", "404"} else local
    return (
        f'<select class="version-select" aria-label="Docs version" data-versions-url="{versions_url}" '
        f'data-current="{html.escape(version)}" data-page="{page}">'
        f"<option>{html.escape(version)}</option></select>"
    )


def page_description(fragment: str) -> str:
    """First paragraph of the page as plain text, for search and link previews."""
    for paragraph in PARAGRAPH_RE.findall(fragment):
        text = " ".join(html.unescape(re.sub(r"<[^>]+>", "", paragraph)).split())
        if text:
            return text if len(text) <= 200 else text[:197].rsplit(" ", 1)[0] + "..."
    return ""


def render_page(fragment: str, local: str, site: Site) -> str:
    match = METADATA_RE.search(fragment)
    metadata = json.loads(html.unescape(match.group(1))) if match else {}
    title = metadata.get("title", "OpenEnv")
    if "OpenEnv" not in title:
        title = f"{title} | OpenEnv"
    title = html.escape(title)
    description = html.escape(page_description(fragment))
    base = site.base
    sidebar, _ = render_sidebar(site.toctree, base, local)
    toc = render_toc(metadata.get("sections", []))
    toc_block = (
        f'<aside class="page-toc"><p>On this page</p>{toc}</aside>' if toc else ""
    )
    # The kit's CSS files are empty in production builds, build_css.mjs compiles the real one.
    fragment = CSS_PRELOAD_RE.sub("", fragment)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
{'<meta name="robots" content="noindex">' if site.noindex else ""}
<meta property="og:type" content="website">
<meta property="og:site_name" content="OpenEnv">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:url" content="{page_url(site.url, local)}">
<meta property="og:image" content="{site.url}/og-image.png">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="{base}/favicon.svg" type="image/svg+xml">
<link rel="icon" href="{base}/favicon.png" sizes="32x32" type="image/png">
<link rel="apple-touch-icon" href="{base}/apple-touch-icon.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Source+Sans+3:wght@400;600;700&display=swap">
<link rel="stylesheet" href="{base}/docs.css">
<link rel="stylesheet" href="{base}/shell.css">
<script>
  try {{
    const theme = localStorage.getItem("theme");
    if (theme === "dark" || (!theme && matchMedia("(prefers-color-scheme: dark)").matches)) {{
      document.documentElement.classList.add("dark");
    }}
  }} catch {{}}
</script>
</head>
<body data-sveltekit-reload>
<header class="site-header">
  <button class="menu-button" aria-label="Toggle navigation" onclick="document.body.classList.toggle('nav-open')">&#9776;</button>
  <a class="site-title" href="{base}/">OpenEnv</a>
  {render_version(site.version, site.versions_url, local)}
  <span class="spacer"></span>
  <a href="{REPO_URL}">GitHub</a>
  <button class="theme-button" aria-label="Toggle dark mode" onclick="const dark = document.documentElement.classList.toggle('dark'); try {{ localStorage.setItem('theme', dark ? 'dark' : 'light'); }} catch {{}}">&#9680;</button>
</header>
<div class="site-layout">
<nav class="site-nav">{sidebar}</nav>
<main class="site-main">
<div class="prose prose-doc dark:prose-light max-w-4xl mx-auto break-words relative">
{fragment}
</div>
</main>
{toc_block}
</div>
{VERSION_SCRIPT if site.versions_url else ""}
</body>
</html>
"""


def redirect_page(target: str) -> str:
    return f"""<!doctype html>
<meta charset="utf-8">
<title>Redirecting</title>
<meta http-equiv="refresh" content="0; url={target}">
<link rel="canonical" href="{target}">
<a href="{target}">Moved to {target}</a>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "build_dir", type=Path, help="doc-builder output, e.g. build/openenv/main/en"
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--site-url", required=True, help="e.g. https://openenv.github.io/OpenEnv"
    )
    parser.add_argument(
        "--version-label", default="main", help="e.g. main, v0.15.0, PR #12"
    )
    parser.add_argument(
        "--versions-url", help="versions.json listing the published versions"
    )
    parser.add_argument(
        "--noindex", action="store_true", help="keep search engines out (PR previews)"
    )
    args = parser.parse_args()
    site_url = args.site_url.rstrip("/")
    base = urlparse(site_url).path
    # doc-builder writes to <build>/openenv/<version>/en and links pages under the
    # same path on huggingface.co.
    version = args.build_dir.resolve().parent.name
    doc_builder_base = f"/docs/openenv/{version}/en"
    hub_docs_url = f"https://huggingface.co/docs/openenv/{version}"

    shutil.rmtree(args.output_dir, ignore_errors=True)
    shutil.copytree(args.build_dir, args.output_dir)
    out = args.output_dir
    site = Site(
        url=site_url,
        base=base,
        toctree=yaml.safe_load((out / "_toctree.yml").read_text()),
        version=args.version_label,
        versions_url=args.versions_url,
        noindex=args.noindex,
    )

    for path in out.rglob("*"):
        if path.suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            text = text.replace(hub_docs_url, site_url).replace(
                doc_builder_base, site_url
            )
            path.write_text(text, encoding="utf-8")
        if path.suffix not in {".html", ".js"}:
            continue
        text = path.read_text(encoding="utf-8").replace(doc_builder_base, base)
        if path.suffix == ".html":
            local = path.relative_to(out).with_suffix("").as_posix()
            text = render_page(text, local, site)
        path.write_text(text, encoding="utf-8")

    redirects = yaml.safe_load((out / "_redirects.yml").read_text()) or {}
    for source, target in redirects.items():
        page = out / f"{source}.html"
        if not page.exists():
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text(redirect_page(page_url(base, target)), encoding="utf-8")

    (out / "404.html").write_text(
        render_page(
            "<h1>Page not found</h1><p>This page does not exist. "
            f'Go back to the <a href="{base}/">OpenEnv docs</a>.</p>',
            "404",
            site,
        ),
        encoding="utf-8",
    )
    for name in ["shell.css", *ASSETS]:
        shutil.copy(SHELL_DIR / name, out / name)
    (out / ".nojekyll").touch()
    print(f"site written to {out}")


if __name__ == "__main__":
    main()
