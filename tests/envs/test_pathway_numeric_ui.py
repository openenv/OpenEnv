# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offline regressions for DE serialization and the UI's contrast direction."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from pathway_analysis_env.models import PathwayObservation
from pathway_analysis_env.server import analysis, gradio_ui


def test_author_de_unknown_statistics_are_json_null(tmp_path):
    table = tmp_path / "de.csv"
    table.write_text(
        "gene;log2FoldChange;pvalue;padj\n"
        "TP53;2;0.001;0.01\n"
        "MISSING;NaN;inf;NaN\n"
        "INFINITE;-inf;-inf;inf\n"
    )
    rows = analysis.load_author_de_table_csv(table)
    payload = PathwayObservation(de_genes=rows).model_dump()
    json.dumps(payload, allow_nan=False)
    by_gene = {row["gene"]: row for row in rows}
    assert by_gene["TP53"]["baseMean"] is None
    assert by_gene["TP53"]["log2FoldChange"] == 2.0
    for gene in ("MISSING", "INFINITE"):
        assert by_gene[gene]["log2FoldChange"] is None
        assert by_gene[gene]["pvalue"] is None
        assert by_gene[gene]["padj"] == 1.0


def test_deseq_nonfinite_statistics_serialize_and_sort(monkeypatch):
    result = pd.DataFrame(
        {
            "baseMean": [10.0, float("nan"), float("inf")],
            "log2FoldChange": [2.0, float("nan"), float("-inf")],
            "lfcSE": [0.2, float("inf"), float("nan")],
            "pvalue": [0.001, float("nan"), float("inf")],
            "padj": [0.01, float("nan"), float("-inf")],
        },
        index=["TP53", "MISSING", "INFINITE"],
    )
    monkeypatch.setattr(analysis, "_PYDESQ2_AVAILABLE", True)
    monkeypatch.setattr(
        analysis, "DeseqDataSet", lambda **kwargs: SimpleNamespace(deseq2=lambda: None)
    )
    monkeypatch.setattr(
        analysis,
        "DeseqStats",
        lambda *args, **kwargs: SimpleNamespace(
            summary=lambda: None, results_df=result
        ),
    )
    rows, error = analysis.run_deseq2_contrast(
        pd.DataFrame(),
        pd.DataFrame({"condition": ["control", "treated"]}),
        "treated",
        "control",
    )
    assert error is None
    json.dumps(PathwayObservation(de_genes=rows).model_dump(), allow_nan=False)
    assert rows[0]["gene"] == "TP53"
    for row in rows[1:]:
        assert all(
            row[key] is None
            for key in ("baseMean", "log2FoldChange", "lfcSE", "pvalue")
        )
        assert row["padj"] == 1.0
        assert not row["significant"]
    assert analysis.pick_de_query_genes(rows, direction="up") == ["TP53"]


@pytest.mark.parametrize("update", [True, False])
def test_ui_reset_preserves_default_contrast_direction(update):
    manager = SimpleNamespace(get_state=lambda: {"conditions": ["FAR", "control"]})
    data = {
        "observation": {
            "experiment_design": {
                "default_contrast": {"reference": "control", "alternate": "FAR"}
            }
        }
    }
    outputs = gradio_ui._response(data, manager, "reset", update_contrast=update)
    if update:
        assert outputs[-2]["value"] == "control"
        assert outputs[-1]["value"] == "FAR"
    else:
        assert "value" not in outputs[-2]
        assert "value" not in outputs[-1]


def test_counts_like_reset_supplies_ui_contrast(monkeypatch):
    from pathway_analysis_env.server import pathway_environment

    monkeypatch.setattr(pathway_environment, "_write_html_trace", lambda *args: "")
    env = pathway_environment.PathwayEnvironment(
        case_file="gse227102_t47d_counts_like_case.json"
    )
    observation = env.reset()
    manager = SimpleNamespace(get_state=lambda: env.state.model_dump())
    outputs = gradio_ui._response(
        {"observation": observation.model_dump()},
        manager,
        "reset",
        update_contrast=True,
    )
    assert outputs[-2]["value"] == "control"
    assert outputs[-1]["value"] == "FAR"


@pytest.mark.parametrize("sizes", [(None, None), (None, 3), (2, None), (2, 3), (0, 0)])
def test_comparison_ui_preserves_unknown_and_known_sizes(sizes):
    a_size, b_size = sizes
    comparison = analysis.compare_pathways_detail(
        "P1", "P2", {"P1": ["A", "B"], "P2": ["B", "C"]}, ["A", "B", "C"]
    )
    comparison.update(pathway_a_size=a_size, pathway_b_size=b_size)
    obs = PathwayObservation(pathway_comparison=comparison)
    outputs = gradio_ui._response(
        {"observation": obs.model_dump()},
        SimpleNamespace(get_state=lambda: {}),
        "Compare complete.",
        update_contrast=False,
    )
    table = outputs[3]
    assert table["count"].tolist()[:3] == [1, 1, 1]
    assert table["genes (preview)"].tolist()[:3] == ["A", "C", "B"]
    expected = "Unknown" if None in sizes else sum(sizes)
    assert table.iloc[3]["count"] == expected
    for label, size in zip(("A", "B"), sizes):
        rendered = "Unknown" if size is None else str(size)
        assert f"{label} size={rendered}" in table.iloc[3]["genes (preview)"]
    assert json.loads(outputs[6])["observation"]["pathway_comparison"] == comparison
