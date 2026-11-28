# SPDX-License-Identifier: BSD-3-Clause
"""Exercise data loading from a built artifact, outside the source checkout."""

import gzip
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ENV_DIR = Path(__file__).resolve().parents[2] / "envs" / "pathway_analysis_env"


def test_wheel_contains_nested_case_assets(tmp_path):
    pytest.importorskip("setuptools")
    pytest.importorskip("wheel")
    source = tmp_path / "source"
    shutil.copytree(
        ENV_DIR,
        source,
        ignore=shutil.ignore_patterns(
            ".venv", "build", "*.egg-info", "__pycache__", "outputs"
        ),
    )
    # External GEO matrices are not checked in. Synthetic values test packaging
    # and parsing only, at the paths referenced by the real case definitions.
    for case_file in (source / "data").rglob("*.json"):
        case = json.loads(case_file.read_text())
        if "counts_file" in case:
            asset = source / "data" / case["counts_file"]
            contents = (
                "gene,"
                + ",".join(case["sample_ids"])
                + "\nTP53,"
                + ",".join(["10"] * len(case["sample_ids"]))
                + "\n"
            )
        elif "de_table_file" in case:
            asset = source / "data" / case["de_table_file"]
            contents = "Gene,name;log2FoldChange;pvalue;padj\nTP53;2;0.001;0.01\n"
        else:
            continue
        asset.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(asset, "wt") as stream:
            stream.write(contents)
    (source / "data" / "geo_eval" / "plain.csv").write_text("gene,sample\nTP53,10\n")
    built = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel; build_wheel('dist')",
        ],
        cwd=source,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    installed = tmp_path / "installed"
    with zipfile.ZipFile(next((source / "dist").glob("*.whl"))) as wheel:
        wheel.extractall(installed)
    shutil.rmtree(source)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
from pathway_analysis_env.server.pathway_environment import DATA_DIR, load_case
from pathway_analysis_env.server.analysis import (
    load_counts_csv_as_samples_by_genes, load_author_de_table_csv,
)
assert str(DATA_DIR).startswith(INSTALLED)
nested = 'geo_eval/gse128911_mda_mb_134_vi_fulvestrant_vs_dmso/gse128911_case.json'
assert load_case(nested)['case_id'].startswith('GSE128911')
counts = tables = 0
for path in DATA_DIR.rglob('*.json'):
    case = json.loads(path.read_text())
    if 'counts_file' in case:
        frame = load_counts_csv_as_samples_by_genes(
            DATA_DIR / case['counts_file'], sample_ids=case['sample_ids'],
        )
        assert frame.shape == (len(case['sample_ids']), 1)
        counts += 1
    elif 'de_table_file' in case:
        rows = load_author_de_table_csv(DATA_DIR / case['de_table_file'])
        assert rows[0]['gene'] == 'TP53'
        tables += 1
assert counts == 4 and tables == 2, (counts, tables)
assert (DATA_DIR / 'geo_eval/plain.csv').read_text().startswith('gene,sample')
""".replace("INSTALLED", repr(str(installed))),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(installed), str(ENV_DIR.parents[1] / "src")]
            ),
        },
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
