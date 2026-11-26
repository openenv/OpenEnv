# Building the Docs Locally

The documentation is built with [HF doc-builder](https://github.com/huggingface/doc-builder).

## Prerequisites

- Python 3.11+
- Node.js 18+

## Setup

Install the package and doc-builder:

```bash
pip install -e ".[core,cli]"
pip install hf-doc-builder watchdog
```

## Preview

From the repo root:

```bash
doc-builder preview openenv docs/source
```

Then open the URL printed in the terminal and navigate to `/index` (usually `http://localhost:5173/index`).

## Publishing to GitHub Pages

The docs are built with doc-builder in GitHub Actions and published to GitHub Pages from the `gh-pages` branch, with no Hugging Face token:

| Trigger | Published at |
|---|---|
| Push to `main` | `https://openenv.github.io/OpenEnv/main/` |
| Release tag `vX.Y.Z` | `https://openenv.github.io/OpenEnv/vX.Y.Z/` |
| Pull request | `https://openenv.github.io/OpenEnv/pr-preview/pr-N/`, linked from a comment on the PR and removed when it closes |

The site root redirects to `main`, and the version picker in the header lists `main` and every published release. To publish an older release, run the **Docs** workflow from `main` with its tag as `version`.

Pull requests from outside contributors build without write access and get a preview only after a maintainer adds the `docs-preview` label and re-runs the PR's **Docs** check, since previews are served from the docs domain. The label is removed once the preview is published, so each new commit needs a new review.

How it fits together:

- [`docs.yml`](../.github/workflows/docs.yml) builds the docs and publishes `main` and releases.
- [`docs_pr_preview.yml`](../.github/workflows/docs_pr_preview.yml) publishes and removes PR previews. It runs from `main` and only copies the files the PR build uploaded.
- [`docs_builder_update.yml`](../.github/workflows/docs_builder_update.yml) checks every week whether `huggingface/doc-builder` moved ahead of the commit pinned in `docs.yml` and keeps an issue open until the pin is updated.
- doc-builder renders each page as a fragment that huggingface.co wraps in its own layout, so [`docs/pages/`](pages/) adds that layout. `build_site.py` wraps each page with a sidebar from `_toctree.yml`, an "On this page" list, the version picker and redirect pages from `_redirects.yml`. `build_css.mjs` compiles the doc-builder stylesheet, which doc-builder leaves empty in production builds. `shell.css` styles the layout and the code highlighting, with the fonts in `fonts/` (Lato, the typeface of the OpenEnv wordmark, and IBM Plex Mono) served by the site itself. `logo.svg` and `logo-white.svg` are the OpenEnv logo for light and dark backgrounds. `publish.sh` updates one folder of `gh-pages`.

To build the site locally, clone [doc-builder](https://github.com/huggingface/doc-builder) at the commit pinned in `docs.yml` next to this repo and run the steps of the `build` job.

## Adding an Environment to the Docs

Every environment page is generated from the environment's own `README.md`. There are three steps:

### 1. Write the environment README

Your environment must have a `README.md` at `envs/<name>/README.md`.

### 2. Generate the doc stub

Run the sync script from the repo root to create or refresh the stub:

```bash
python scripts/sync_env_docs.py --fix
```

This inlines the README content into `docs/source/environments/<slug>.md`.
To check if stubs are in sync without modifying files:

```bash
python scripts/sync_env_docs.py --check
```

### 3. Add a card and toctree entry

- Add an HTML card in `docs/source/environments.md`
- Add a `local: environments/<slug>` entry in `docs/source/_toctree.yml`
