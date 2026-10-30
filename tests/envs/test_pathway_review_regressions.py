# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from pathway_analysis_env.models import PathwayAction
from pathway_analysis_env.server import pathway_environment as pe


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(pe, "OUTPUT_TRACE_DIR", tmp_path / "traces")
    monkeypatch.setattr(pe, "pydeseq2_available", lambda: True)
    return pe.PathwayEnvironment()


@pytest.mark.parametrize("strict", [False, True])
def test_author_table_without_pydeseq2(env, tmp_path, monkeypatch, strict):
    table = tmp_path / "author.csv"
    table.write_text("Gene,name;log2FoldChange;pvalue;padj\nTP53;2;0.001;0.01\n")
    env._case["de_table_file"] = str(table)
    env._state.strict_mode = strict
    monkeypatch.setattr(pe, "pydeseq2_available", lambda: False)
    obs = env.step(PathwayAction(action_type="run_differential_expression"))
    assert obs.metadata.get("precomputed") is True
    assert obs.de_genes[0]["gene"] == "TP53"
    assert not obs.done


@pytest.mark.parametrize(
    "fields",
    [
        {"condition_a": "treated"},
        {"condition_b": "control"},
        {"condition_a": "control", "condition_b": "control"},
        {"condition_a": "missing", "condition_b": "treated"},
    ],
)
@pytest.mark.parametrize("strict", [False, True])
def test_invalid_de_contrast_never_fits(env, monkeypatch, fields, strict):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid contrast reached fitting")

    monkeypatch.setattr(pe, "run_deseq2_contrast", unexpected)
    env._state.strict_mode = strict
    obs = env.step(PathwayAction(action_type="run_differential_expression", **fields))
    assert obs.metadata["failure_code"]
    assert obs.reward < 0
    assert obs.done is strict
    assert not env.state.de_run


def test_contrast_pair_precedence(env, monkeypatch):
    seen = []

    def fit(counts, metadata, alt, ref, **kwargs):
        seen.append((ref, alt))
        return [], None

    monkeypatch.setattr(pe, "run_deseq2_contrast", fit)
    env.step(PathwayAction(action_type="run_differential_expression"))
    env._state.validated_reference = "treated"
    env._state.validated_alternate = "control"
    env.step(PathwayAction(action_type="run_differential_expression"))
    env.step(
        PathwayAction(
            action_type="run_differential_expression",
            condition_a="control",
            condition_b="treated",
        )
    )
    assert seen == [
        ("control", "treated"),
        ("treated", "control"),
        ("control", "treated"),
    ]


@pytest.mark.parametrize("strict", [False, True])
def test_metadata_failure_code(env, strict):
    env._state.strict_mode = strict
    env._case["sample_metadata"].pop(env._case["sample_ids"][0])
    obs = env.step(PathwayAction(action_type="run_differential_expression"))
    assert obs.metadata["failure_code"] == "case_sample_metadata_mismatch"
    assert obs.done is strict


def test_compare_enrichr_support(env, monkeypatch):
    env._case.pop("pathway_genes")
    env._case["enrichr_libraries"] = ["mock_library"]
    env._state.de_run = True
    env._query_genes = ["A", "B", "C"]
    env._de_rows = [
        {"gene": g, "padj": 0.01, "log2FoldChange": 2} for g in env._query_genes
    ]
    monkeypatch.setattr(pe, "gseapy_available", lambda: True)
    monkeypatch.setattr(
        pe,
        "enrichr_ora",
        lambda *a, **k: (
            [
                {
                    "pathway": "P1",
                    "overlap_genes": ["A", "B"],
                    "p_value": 0.01,
                    "q_value": 0.02,
                },
                {
                    "pathway": "P2",
                    "overlap_genes": ["B", "C"],
                    "p_value": 0.02,
                    "q_value": 0.03,
                },
            ],
            None,
        ),
    )
    env.step(PathwayAction(action_type="run_pathway_enrichment"))
    obs = env.step(
        PathwayAction(action_type="compare_pathways", pathway_a="P1", pathway_b="P2")
    )
    d = obs.pathway_comparison
    assert d["exclusive_to_a"] == ["A"]
    assert d["exclusive_to_b"] == ["C"]
    assert d["shared_de_support"] == ["B"]
    assert d["pathway_a_size"] is None  # hits do not give full pathway membership
    assert d["comparison_scope"] == "enrichr_overlap_genes"

    from types import SimpleNamespace

    from pathway_analysis_env.server import gradio_ui

    outputs = gradio_ui._response(
        {"observation": obs.model_dump()},
        SimpleNamespace(get_state=lambda: env.state.model_dump()),
        "Compare complete.",
        update_contrast=False,
    )
    table = outputs[3]
    assert table["count"].tolist() == [1, 1, 1, "Unknown"]
    assert table["genes (preview)"].tolist()[:3] == ["A", "C", "B"]
    assert "Enrichr overlap genes only" in outputs[0]
    assert "full pathway membership is unknown" in outputs[0]


def test_compare_unknown_pathways_does_not_reward(env):
    obs = env.step(
        PathwayAction(
            action_type="compare_pathways", pathway_a="absent", pathway_b="missing"
        )
    )
    assert obs.reward < 0
    assert obs.pathway_comparison is None


@pytest.mark.parametrize("strict", [False, True])
def test_missing_counts_file_is_failure(env, tmp_path, strict):
    env._case["counts_file"] = str(tmp_path / "missing.csv.gz")
    env._state.strict_mode = strict
    obs = env.step(PathwayAction(action_type="run_differential_expression"))
    assert obs.metadata["failure_code"] == "de_invalid_counts_matrix"
    assert obs.done is strict
    assert not env.state.de_run


@pytest.mark.parametrize("strict", [False, True])
def test_author_table_rejects_reversed_contrast(env, tmp_path, strict):
    table = tmp_path / "author.csv"
    table.write_text("Gene,name;log2FoldChange;pvalue;padj\nTP53;2;0.001;0.01\n")
    env._case["de_table_file"] = str(table)
    env._state.strict_mode = strict
    obs = env.step(
        PathwayAction(
            action_type="run_differential_expression",
            condition_a="treated",
            condition_b="control",
        )
    )
    assert obs.metadata["failure_code"] == "de_unsupported_precomputed_contrast"
    assert obs.done is strict
    assert not env.state.de_run
    assert not obs.de_genes


def test_author_table_caps_observation_but_keeps_analysis_rows(env, tmp_path):
    table = tmp_path / "author.csv"
    table.write_text(
        "Gene,name;log2FoldChange;pvalue;padj\n"
        + "".join(f"G{i};2;0.001;0.01\n" for i in range(251))
    )
    env._case["de_table_file"] = str(table)
    obs = env.step(PathwayAction(action_type="run_differential_expression"))
    assert len(obs.de_genes) == 200
    assert len(env._de_rows) == 251
    assert len(env._query_genes) == 200  # existing query selection cap


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("failure", ["missing", "runtime", "definitions"])
def test_enrichment_failure_codes(env, monkeypatch, strict, failure):
    env._state.strict_mode = strict
    env._case.pop("pathway_genes")
    env._case["enrichr_libraries"] = [] if failure == "definitions" else ["mock"]
    env._de_rows = [{"gene": "TP53", "padj": 0.01, "log2FoldChange": 2}]
    monkeypatch.setattr(pe, "gseapy_available", lambda: failure != "missing")
    monkeypatch.setattr(pe, "enrichr_ora", lambda *a, **k: ([], "service failed"))
    obs = env.step(PathwayAction(action_type="run_pathway_enrichment"))
    assert (
        obs.metadata["failure_code"]
        == {
            "missing": "ora_gseapy_unavailable",
            "runtime": "ora_enrichr_failed",
            "definitions": "ora_no_pathway_definitions",
        }[failure]
    )
    assert obs.done is strict
    assert not env.state.enrichment_run


@pytest.mark.parametrize("mode", ["validated_reverse", "missing_default"])
def test_author_table_rejects_unsupported_implicit_contrast(env, mode):
    env._case["de_table_file"] = "unused.csv"
    env._state.validated_reference = "treated"
    env._state.validated_alternate = "control"
    if mode == "missing_default":
        env._case.pop("default_contrast")
    obs = env.step(PathwayAction(action_type="run_differential_expression"))
    assert obs.metadata["failure_code"] == "de_unsupported_precomputed_contrast"
    assert not env.state.de_run


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("source", ["author", "counts"])
@pytest.mark.parametrize("failure", ["missing", "unreadable", "invalid"])
def test_de_source_failures_have_distinct_codes(
    env, tmp_path, monkeypatch, strict, source, failure
):
    path = tmp_path / "input.csv"
    if failure == "unreadable":
        from pathway_analysis_env.server import analysis

        path.write_text("unreadable fixture")

        def denied(*args, **kwargs):
            raise PermissionError(f"Cannot read {path}")

        # Exercise the actual loaders with a deterministic read failure, even as root.
        monkeypatch.setattr(analysis.pd, "read_csv", denied)
    elif failure == "invalid":
        path.write_text("wrong,columns\nfoo,bar\n")
    env._case[f"{'de_table' if source == 'author' else 'counts'}_file"] = str(path)
    env._state.strict_mode = strict

    def unexpected(*args, **kwargs):
        pytest.fail("failed source must not fall back to fitting counts")

    monkeypatch.setattr(pe, "run_deseq2_contrast", unexpected)
    obs = env.step(PathwayAction(action_type="run_differential_expression"))
    expected = (
        "de_invalid_author_table" if source == "author" else "de_invalid_counts_matrix"
    )
    assert obs.metadata["failure_code"] == expected
    assert obs.reward < 0
    assert obs.done is strict
    assert obs.metadata.get("strict_failure", False) is strict
    assert not env.state.de_run
    assert not obs.de_genes
