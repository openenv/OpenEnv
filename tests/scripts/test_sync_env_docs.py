"""Regression checks for repository links in generated environment docs."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


spec = importlib.util.spec_from_file_location(
    "sync_env_docs", Path(__file__).parents[2] / "scripts" / "sync_env_docs.py"
)
sync_env_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync_env_docs)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    env = root / "envs" / "example_env"
    env.mkdir(parents=True)
    (env / "client.py").touch()
    (env / "image.png").touch()
    (root / "LICENSE").touch()
    (root / "examples").mkdir()
    (root / "examples" / "with spaces.py").touch()
    (root / "docs" / "source" / "environments").mkdir(parents=True)
    monkeypatch.setattr(sync_env_docs, "ROOT", str(root))
    monkeypatch.setattr(sync_env_docs, "ENVS_DIR", str(root / "envs"))
    monkeypatch.setattr(
        sync_env_docs,
        "DOCS_ENVS_DIR",
        str(root / "docs" / "source" / "environments"),
    )
    return root


def test_rewrites_files_directories_images_and_preserves_url_suffixes(repo):
    source = """[client](client.py?plain=1#L2 "source")
[examples](../../examples/)
[spaced](<../../examples/with spaces.py>)
![image](image.png)
<img src="image.png" alt="sample">
<a href="client.py">client</a>
[![License](https://img.shields.io/badge/license-BSD-blue)](../../LICENSE)
"""
    result = sync_env_docs._rewrite_relative_links(source, "example_env")
    github = "https://github.com/huggingface/OpenEnv"
    raw = "https://raw.githubusercontent.com/huggingface/OpenEnv/main"
    assert (
        result
        == f"""[client]({github}/blob/main/envs/example_env/client.py?plain=1#L2 "source")
[examples]({github}/tree/main/examples)
[spaced](<{github}/blob/main/examples/with%20spaces.py>)
![image]({raw}/envs/example_env/image.png)
<img src="{raw}/envs/example_env/image.png" alt="sample">
<a href="{github}/blob/main/envs/example_env/client.py">client</a>
[![License](https://img.shields.io/badge/license-BSD-blue)]({github}/blob/main/LICENSE)
"""
    )


def test_leaves_code_anchors_external_missing_and_outside_paths_unchanged(repo):
    (repo.parent / "outside.txt").touch()
    (repo / "envs" / "example_env" / "outside-link").symlink_to(
        repo.parent / "outside.txt"
    )
    source = """`[client](client.py)`
    [client](client.py)
````markdown
```python
[client](client.py)
```
````
~~~markdown
<img src="image.png">
~~~
> ```markdown
> [client](client.py)
> ```
- Example:
    ```markdown
    [client](client.py)
    <a href="client.py">client</a>
    <img src="image.png">
    ```
![plot [v2]](image.png)
[anchor](#usage)
[external](https://example.com/client.py)
[protocol-relative](//example.com/client.py)
[absolute](/client.py)
[null](%00)
[missing](missing.py)
[outside](../../../outside.txt)
[symlink](outside-link)
"""
    assert sync_env_docs._rewrite_relative_links(source, "example_env") == source


def test_rewrites_links_in_deeply_indented_list_items(repo):
    source = """    - Client files:
      [client](client.py)

    [client](client.py)
"""
    github = "https://github.com/huggingface/OpenEnv"
    assert sync_env_docs._rewrite_relative_links(source, "example_env") == (
        "    - Client files:\n"
        f"      [client]({github}/blob/main/envs/example_env/client.py)\n"
        "\n"
        "    [client](client.py)\n"
    )


def test_generate_stub_keeps_inline_code_labels_and_strips_frontmatter(repo):
    readme = repo / "envs" / "example_env" / "README.md"
    readme.write_text(
        "---\ntitle: Example\n---\n# Example\n\n[`client.py`](client.py)\n"
    )
    assert sync_env_docs.generate_stub("example_env") == (
        "<!-- openenv-source: example_env -->\n# Example\n\n"
        "[`client.py`](https://github.com/huggingface/OpenEnv/blob/main/envs/example_env/client.py)\n"
    )


def test_run_fix_handles_malformed_url_without_truncating_stub(repo):
    readme = repo / "envs" / "example_env" / "README.md"
    readme.write_text("# Example\n\n[bad](http://[)\n")
    stub = repo / "docs" / "source" / "environments" / "example.md"
    stub.write_text("previous content\n")

    sync_env_docs.run_fix([], [], [("example_env", "example")])

    assert stub.read_text() == (
        "<!-- openenv-source: example_env -->\n# Example\n\n[bad](http://[)\n"
    )


def test_run_fix_preserves_stub_when_generation_fails(repo, monkeypatch):
    stub = repo / "docs" / "source" / "environments" / "example.md"
    stub.write_text("previous content\n")
    monkeypatch.setattr(
        sync_env_docs,
        "generate_stub",
        Mock(side_effect=ValueError("invalid README")),
    )

    with pytest.raises(ValueError, match="invalid README"):
        sync_env_docs.run_fix([], [], [("example_env", "example")])

    assert stub.read_text() == "previous content\n"
