"""Saved-run reads must remain inside the outputs directory."""

import json

import pytest
from pathway_analysis_env.server import gradio_ui


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    root.mkdir()
    monkeypatch.setattr(gradio_ui, "_OUTPUTS_DIR", root)
    return root


def write_run(path):
    path.mkdir(exist_ok=True)
    for name, value in (
        ("summary.json", {"case_id": "safe"}),
        ("de_top200.json", []),
        ("enrichment_top50.json", []),
    ):
        (path / name).write_text(json.dumps(value))


def test_load_saved_run(outputs):
    write_run(outputs / "run1")
    result = gradio_ui._load_saved_run("run1")
    assert result == {
        "run": "run1",
        "summary": {"case_id": "safe"},
        "de": [],
        "enrichment": [],
    }


@pytest.mark.parametrize(
    "kind",
    [
        "parent",
        "absolute",
        "directory_symlink",
        "summary.json",
        "de_top200.json",
        "enrichment_top50.json",
    ],
)
def test_saved_run_rejects_escape_before_reading(outputs, monkeypatch, kind):
    outside = outputs.parent / "outside"
    write_run(outside)
    run_name = "run1"
    if kind == "parent":
        run_name = "../outside"
    elif kind == "absolute":
        run_name = str(outside)
    elif kind == "directory_symlink":
        (outputs / run_name).symlink_to(outside, target_is_directory=True)
    else:
        write_run(outputs / run_name)
        (outputs / run_name / kind).unlink()
        (outputs / run_name / kind).symlink_to(outside / kind)

    def unexpected_read(*args, **kwargs):
        pytest.fail("Rejected saved run must not read any files")

    monkeypatch.setattr(type(outputs), "read_text", unexpected_read)
    result = gradio_ui._load_saved_run(run_name)
    assert result["summary_error"] == "Invalid run name"
    assert result["summary"] == {}
    assert result["de"] == result["enrichment"] == []
