# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from pathway_analysis_env.bench import geo_openenv_deseq2 as bench
from pathway_analysis_env.server import analysis


@pytest.mark.parametrize("retained", [4, 5])
def test_benchmark_prefilter_and_background(tmp_path, monkeypatch, retained):
    repo = Path(__file__).resolve().parents[2]
    (tmp_path / "data").mkdir()
    counts = {f"G{i}": [5, 5] for i in range(retained)}
    counts["low"] = [1, 1]
    pd.DataFrame(counts, index=["c", "t"]).T.to_csv(tmp_path / "data/counts.csv")
    case = {
        "counts_file": "counts.csv",
        "sample_ids": ["c", "t"],
        "sample_metadata": {
            "c": {"condition": "control"},
            "t": {"condition": "treated"},
        },
        "default_contrast": {"reference": "control", "alternate": "treated"},
        "analysis_options": {"min_total_count": 10},
    }
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    out = tmp_path / "results"
    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "--repo-root",
            str(repo),
            "--env-dir",
            str(tmp_path),
            "--case-json",
            str(case_path),
            "--out-dir",
            str(out),
        ],
    )
    calls = []

    def fit(counts, *args, **kwargs):
        calls.append("fit")
        assert list(counts.columns) == [f"G{i}" for i in range(retained)]
        return [
            {"gene": "G0", "padj": 0.01, "log2FoldChange": 2, "significant": True}
        ], None

    def enrich(query, *, background, **kwargs):
        calls.append("enrich")
        assert background == [f"G{i}" for i in range(retained)]
        return [], None

    monkeypatch.setattr(analysis, "run_deseq2_contrast", fit)
    monkeypatch.setattr(analysis, "enrichr_ora", enrich)
    if retained < 5:
        with pytest.raises(RuntimeError, match="only 4 genes remain"):
            bench.main()
        assert calls == []
        assert not out.exists()
    else:
        bench.main()
        assert calls == ["fit", "enrich"]
        summary = json.loads((out / "summary.json").read_text())
        assert summary["genes_in_matrix"] == retained + 1
        assert summary["genes_after_prefilter"] == retained
