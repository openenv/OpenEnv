# SPDX-License-Identifier: BSD-3-Clause

import os
import shutil
import socket
import stat
import tempfile
from pathlib import Path

import pytest
from openenv.core.openenvd import custody
from openenv.core.openenvd.custody import copy_validated, CustodyError

LIMITS = {"max_bytes": 1 << 20, "max_files": 100}


@pytest.fixture
def src(tmp_path):
    root = tmp_path / "out"
    (root / "sub" / "deep").mkdir(parents=True)
    (root / "a.txt").write_text("alpha")
    (root / "sub" / "b.bin").write_bytes(b"\x00" * 10)
    (root / "sub" / "deep" / "c").write_text("c")
    os.chmod(root / "a.txt", 0o777)
    return root


def test_copies_plain_tree_with_private_modes(src, tmp_path):
    dst = tmp_path / "dst"
    report = copy_validated(src, dst, **LIMITS)
    assert (report.files, report.bytes, report.dirs) == (3, 16, 2)
    assert (dst / "a.txt").read_text() == "alpha"
    assert (dst / "sub" / "deep" / "c").read_text() == "c"
    assert stat.S_IMODE((dst / "a.txt").stat().st_mode) == 0o600
    assert stat.S_IMODE((dst / "sub").stat().st_mode) == 0o700
    assert stat.S_IMODE(dst.stat().st_mode) == 0o700


def test_empty_existing_dst_is_allowed(src, tmp_path):
    dst = tmp_path / "dst"
    dst.mkdir()
    assert copy_validated(src, dst, **LIMITS).files == 3


def test_non_empty_dst_is_rejected(src, tmp_path):
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "x").write_text("")
    with pytest.raises(ValueError):
        copy_validated(src, dst, **LIMITS)


def test_symlink_to_asset_is_rejected(src, tmp_path):
    asset = tmp_path / "answer_key"
    asset.write_text("secret")
    (src / "sub" / "link").symlink_to(asset)
    with pytest.raises(CustodyError, match="sub/link: symlink"):
        copy_validated(src, tmp_path / "dst", **LIMITS)


def test_dangling_symlink_is_rejected(src, tmp_path):
    (src / "dangling").symlink_to(tmp_path / "nowhere")
    with pytest.raises(CustodyError, match="dangling"):
        copy_validated(src, tmp_path / "dst", **LIMITS)


def test_symlinked_directory_is_rejected(src, tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (src / "dirlink").symlink_to(assets, target_is_directory=True)
    with pytest.raises(CustodyError, match="dirlink: symlink"):
        copy_validated(src, tmp_path / "dst", **LIMITS)


def test_source_symlink_is_rejected(src, tmp_path):
    link = tmp_path / "srclink"
    link.symlink_to(src, target_is_directory=True)
    with pytest.raises(CustodyError):
        copy_validated(link, tmp_path / "dst", **LIMITS)


def test_fifo_is_rejected_without_blocking(src, tmp_path):
    os.mkfifo(src / "pipe")
    with pytest.raises(CustodyError, match="pipe: FIFO"):
        copy_validated(src, tmp_path / "dst", **LIMITS)


def test_socket_is_rejected(tmp_path):
    # AF_UNIX paths are capped near 104 bytes, so bind in a short temp dir.
    root = Path(tempfile.mkdtemp(dir="/tmp" if os.path.isdir("/tmp") else None))
    sock = socket.socket(socket.AF_UNIX)
    try:
        sock.bind(str(root / "sk"))
        with pytest.raises(CustodyError, match="sk: socket"):
            copy_validated(root, tmp_path / "dst", **LIMITS)
    finally:
        sock.close()
        shutil.rmtree(root)


def test_too_many_bytes(src, tmp_path):
    (src / "big").write_bytes(b"x" * 5000)
    with pytest.raises(CustodyError, match="big: copy exceeds 4096 bytes"):
        copy_validated(src, tmp_path / "dst", max_bytes=4096, max_files=100)


def test_too_many_files(src, tmp_path):
    for i in range(10):
        (src / f"f{i}").write_text("")
    with pytest.raises(CustodyError, match="more than 5 entries"):
        copy_validated(src, tmp_path / "dst", max_bytes=1 << 20, max_files=5)


def _swap_after_listing(monkeypatch, name, swap):
    real = custody._list_dir

    def listing(dfd):
        entries = real(dfd)
        if name in [n for n, _ in entries]:
            swap()
        return entries

    monkeypatch.setattr(custody, "_list_dir", listing)


def test_symlink_swapped_in_after_listing(src, tmp_path, monkeypatch):
    asset = tmp_path / "answer_key"
    asset.write_text("secret")
    target = src / "a.txt"

    def swap():
        target.unlink()
        target.symlink_to(asset)

    _swap_after_listing(monkeypatch, "a.txt", swap)
    dst = tmp_path / "dst"
    with pytest.raises(CustodyError, match="a.txt: changed type"):
        copy_validated(src, dst, **LIMITS)
    assert not (dst / "a.txt").exists()


def test_fifo_swapped_in_after_listing(src, tmp_path, monkeypatch):
    target = src / "a.txt"

    def swap():
        target.unlink()
        os.mkfifo(target)

    _swap_after_listing(monkeypatch, "a.txt", swap)
    with pytest.raises(CustodyError, match="a.txt: FIFO"):
        copy_validated(src, tmp_path / "dst", **LIMITS)


def test_directory_swapped_for_symlink_after_listing(src, tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "key").write_text("secret")
    target = src / "sub"

    def swap():
        target.rename(tmp_path / "moved")
        target.symlink_to(assets, target_is_directory=True)

    _swap_after_listing(monkeypatch, "sub", swap)
    dst = tmp_path / "dst"
    with pytest.raises(CustodyError, match="sub: changed type"):
        copy_validated(src, dst, **LIMITS)
    assert not (dst / "sub" / "key").exists()
