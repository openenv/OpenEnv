# SPDX-License-Identifier: BSD-3-Clause
"""Reference download failures use tiny local payloads; no network or genome data."""

import gzip
import subprocess
import sys
from pathlib import Path

import pytest
from pathway_analysis_env.bench import download_hg38_gencode as download


NAMES = (
    "GRCh38.primary_assembly.genome.fa.gz",
    "gencode.v47.annotation.gtf.gz",
    "gencode.v47.transcripts.fa.gz",
)
VALID = gzip.compress(b">fixture\nACGT\n")
BAD_CRC = VALID[:-8] + bytes([VALID[-8] ^ 1]) + VALID[-7:]


@pytest.fixture
def invoke(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["download", "--out-dir", str(tmp_path)])
    return download.main


@pytest.mark.parametrize(
    "failure", ["http", "interrupted", "html", "truncated", "crc", "empty"]
)
def test_failed_download_is_not_published(tmp_path, monkeypatch, invoke, failure):
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep")
    calls = []

    def run(cmd):
        calls.append(cmd)
        output = Path(cmd[cmd.index("-o") + 1])
        output.write_bytes(
            {
                "http": b"<html>503</html>",
                "interrupted": VALID[:12],
                "html": b"<html>error</html>",
                "truncated": VALID[:-4],
                "crc": BAD_CRC,
                "empty": gzip.compress(b""),
            }[failure]
        )
        if failure in ("http", "interrupted"):
            raise subprocess.CalledProcessError(22 if failure == "http" else 18, cmd)

    monkeypatch.setattr(download, "_run", run)
    with pytest.raises((subprocess.CalledProcessError, OSError, EOFError, ValueError)):
        invoke()
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == [unrelated]


def test_success_is_validated_then_cached(tmp_path, monkeypatch, invoke):
    calls = []

    def run(cmd):
        calls.append(cmd)
        assert "--fail" in cmd
        assert int(cmd[cmd.index("--retry") + 1]) > 0
        output = Path(cmd[cmd.index("-o") + 1])
        assert output.parent == tmp_path
        assert output.name not in NAMES
        assert not any((tmp_path / name).exists() for name in NAMES[len(calls) - 1 :])
        output.write_bytes(VALID)

    monkeypatch.setattr(download, "_run", run)
    invoke()
    invoke()
    assert len(calls) == 3  # Valid references are reused without network traffic.
    assert {p.name for p in tmp_path.iterdir()} == set(NAMES)
    assert all((tmp_path / name).read_bytes() == VALID for name in NAMES)


@pytest.mark.parametrize("payload", [b"<html>404</html>", VALID[:-4], BAD_CRC, b""])
def test_corrupt_cache_fails_safely_without_downloading(
    tmp_path, monkeypatch, invoke, payload
):
    cached = tmp_path / NAMES[0]
    cached.write_bytes(payload)
    unrelated = tmp_path / "unrelated.gz"
    unrelated.write_bytes(b"preserve")

    def unexpected(cmd):
        pytest.fail(
            "Corrupt cache requires explicit recovery, never a silent huge download"
        )

    monkeypatch.setattr(download, "_run", unexpected)
    with pytest.raises(ValueError, match="cached reference"):
        invoke()
    assert cached.read_bytes() == payload
    assert unrelated.read_bytes() == b"preserve"
