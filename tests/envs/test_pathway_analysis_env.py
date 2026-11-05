# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for pathway_analysis_env (DE, ORA, compare, expert, trace)."""

from __future__ import annotations

import json

import pytest
from pathway_analysis_env.models import PathwayAction
from pathway_analysis_env.server.analysis import (
    adjust_pvalues_bh,
    build_sample_metadata,
    ora_fisher,
    overlap_genes_across_top_pathways,
    pydeseq2_available,
    run_deseq2_contrast,
    validate_counts_case,
)
from pathway_analysis_env.server.pathway_environment import (
    DATA_DIR,
    load_case,
    PathwayEnvironment,
)

requires_pydeseq2 = pytest.mark.skipif(
    not pydeseq2_available(),
    reason="PyDESeq2 required for pathway pipeline tests",
)


def test_set_case_file():
    env = PathwayEnvironment(case_file="toy_case_001.json")
    env.set_case_file("toy_case_legacy.json")
    assert env._case_file == "toy_case_legacy.json"


def test_load_pipeline_case():
    case = load_case("toy_case_001.json")
    assert "counts" in case
    assert "pathway_genes" in case
    assert case["true_pathway"] == "MAPK signaling"


def test_load_gse235417_case_is_pipeline_mode():
    case = load_case("gse235417_case.json")
    assert "counts_file" in case
    assert "sample_ids" in case
    assert "sample_metadata" in case
    assert case["default_contrast"]["reference"] == "baseline"
    assert case["default_contrast"]["alternate"] == "resistant"


@requires_pydeseq2
def test_deseq2_mapk_case():
    case = json.loads((DATA_DIR / "toy_case_001.json").read_text(encoding="utf-8"))
    from pathway_analysis_env.server.analysis import (
        build_sample_metadata,
        counts_dict_to_samples_by_genes,
    )

    cdf = counts_dict_to_samples_by_genes(case["counts"], case["sample_ids"])
    meta = build_sample_metadata(case["sample_ids"], case["sample_metadata"])
    rows, err = run_deseq2_contrast(
        cdf,
        meta,
        case["default_contrast"]["alternate"],
        case["default_contrast"]["reference"],
    )
    assert err is None
    top = [r["gene"] for r in rows[:5]]
    assert "DUSP6" in top or "FOS" in top


def test_ora_fisher_structure():
    universe = ["A", "B", "C", "D", "E", "F"]
    pathways = {"P1": ["A", "B", "C"], "P2": ["C", "D"]}
    de = ["A", "B", "C"]
    ora = ora_fisher(de, pathways, universe, min_pathway_genes=2)
    assert len(ora) == 2
    assert ora[0]["pathway"] in ("P1", "P2")
    assert "q_value" in ora[0]


def test_adjust_pvalues_bh_matches_scipy():
    ps = [0.01, 0.05, 0.1]
    q = adjust_pvalues_bh(ps)
    assert len(q) == 3
    assert all(0.0 <= x <= 1.0 for x in q)


def test_validate_counts_case():
    assert validate_counts_case({}) is None
    bad = {
        "counts": {"G1": [1, 2], "G2": [1]},
        "sample_ids": ["a", "b"],
    }
    assert validate_counts_case(bad) is not None


def test_build_sample_metadata_missing_sample():
    with pytest.raises(ValueError, match="missing"):
        build_sample_metadata(["S1", "S2"], {"S1": "a"})


def test_understand_experiment_design_summary():
    env = PathwayEnvironment(case_file="toy_case_001.json")
    env.reset()
    obs = env.step(PathwayAction(action_type="understand_experiment_design"))
    assert obs.experiment_design
    assert obs.experiment_design.get("samples_per_condition")
    assert env.state.design_understood is True
    assert env.state.validated_reference is None


@requires_pydeseq2
def test_understand_validated_contrast_matches_explicit_de():
    env = PathwayEnvironment(case_file="toy_case_001.json")
    env.reset()
    u = env.step(
        PathwayAction(
            action_type="understand_experiment_design",
            condition_a="control",
            condition_b="treated",
        )
    )
    assert u.experiment_design and u.experiment_design.get("validated_contrast")
    assert env.state.validated_reference == "control"
    assert env.state.validated_alternate == "treated"
    a = env.step(
        PathwayAction(
            action_type="run_differential_expression",
            condition_a="control",
            condition_b="treated",
        )
    )
    env2 = PathwayEnvironment(case_file="toy_case_001.json")
    env2.reset()
    env2.step(
        PathwayAction(
            action_type="understand_experiment_design",
            condition_a="control",
            condition_b="treated",
        )
    )
    b = env2.step(PathwayAction(action_type="run_differential_expression"))
    assert a.de_genes and b.de_genes
    assert [r.get("gene") for r in a.de_genes[:10]] == [
        r.get("gene") for r in b.de_genes[:10]
    ]


def test_no_step_after_episode_done():
    env = PathwayEnvironment(case_file="toy_case_legacy.json")
    env.reset()
    env.step(PathwayAction(action_type="submit_answer", hypothesis="MAPK signaling"))
    late = env.step(PathwayAction(action_type="inspect_dataset"))
    assert late.done
    assert late.metadata.get("error") == "episode_done"
    assert late.metadata.get("failure_code") == "episode_already_done"


def test_overlap_summary():
    ora = [
        {
            "pathway": "a",
            "p_value": 0.01,
            "overlap_genes": ["G1", "G2"],
        },
        {
            "pathway": "b",
            "p_value": 0.02,
            "overlap_genes": ["G2", "G3"],
        },
    ]
    ov = overlap_genes_across_top_pathways(ora, top_k=2)
    assert "G2" in ov["genes_supporting_multiple_top_pathways"]


@requires_pydeseq2
def test_episode_pipeline_success():
    env = PathwayEnvironment(case_file="toy_case_001.json")
    obs0 = env.reset(episode_id="ep-test-1")
    assert obs0.metadata.get("pipeline_mode") is True
    assert obs0.trace_path

    a = PathwayAction(
        action_type="run_differential_expression",
        condition_a="control",
        condition_b="treated",
    )
    obs1 = env.step(a)
    assert obs1.de_genes
    assert obs1.top_genes

    b = PathwayAction(action_type="run_pathway_enrichment")
    obs2 = env.step(b)
    assert obs2.pathway_enrichment
    assert "MAPK signaling" in obs2.top_pathways[:3]
    assert obs2.overlap_summary is not None

    c = PathwayAction(
        action_type="compare_pathways",
        pathway_a="MAPK signaling",
        pathway_b="ERK cascade",
    )
    obs3 = env.step(c)
    assert obs3.pathway_comparison
    assert "shared_de_support" in obs3.pathway_comparison

    d = PathwayAction(action_type="ask_expert", expert_question="hint?")
    obs4 = env.step(d)
    assert obs4.expert_message
    assert obs4.reward is not None and obs4.reward < 0

    e = PathwayAction(action_type="submit_answer", hypothesis="MAPK signaling")
    obs5 = env.step(e)
    assert obs5.done
    assert obs5.metadata.get("correct") is True


def test_legacy_fixture():
    env = PathwayEnvironment(case_file="toy_case_legacy.json")
    obs0 = env.reset()
    assert obs0.metadata.get("pipeline_mode") is False
    env.step(PathwayAction(action_type="run_differential_expression"))
    env.step(PathwayAction(action_type="run_pathway_enrichment"))
    fin = env.step(
        PathwayAction(action_type="submit_answer", hypothesis="MAPK signaling")
    )
    assert fin.done
    assert fin.metadata.get("correct") is True


@requires_pydeseq2
def test_strict_invalid_counts_matrix():
    env = PathwayEnvironment(case_file="toy_case_001.json")
    env.reset(strict=True)
    env._case["counts"]["DUSP6"] = [1, 2]
    obs = env.step(
        PathwayAction(
            action_type="run_differential_expression",
            condition_a="control",
            condition_b="treated",
        )
    )
    assert obs.done and obs.metadata.get("strict_failure") is True
    assert obs.metadata.get("failure_code") == "de_invalid_counts_matrix"


@requires_pydeseq2
def test_strict_mode_missing_contrast():
    env = PathwayEnvironment(case_file="toy_case_no_default.json")
    env.reset(strict=True)
    obs = env.step(PathwayAction(action_type="run_differential_expression"))
    assert obs.done is True
    assert obs.metadata.get("strict_failure") is True
    assert obs.metadata.get("failure_code") == "de_missing_contrast"
