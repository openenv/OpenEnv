# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The `server` console script calls `main()` bare, so `main()` must parse argv itself."""

from __future__ import annotations

import ast
import pathlib

import pytest

ENVS_DIR = pathlib.Path(__file__).resolve().parents[2] / "envs"
APPS = sorted(ENVS_DIR.glob("*/server/app.py"))


@pytest.mark.parametrize("path", APPS, ids=lambda p: p.parent.parent.name)
def test_server_main_parses_port(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    main_def = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assert not main_def.args.args
    if "--port" in ast.unparse(main_def):
        assert "parse_args" in ast.unparse(main_def)
