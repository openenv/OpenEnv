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
