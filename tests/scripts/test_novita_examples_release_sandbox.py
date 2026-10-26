"""Every Novita example must release its sandbox if readiness never arrives."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
NOVITA_EXAMPLES = sorted(EXAMPLES_DIR.glob("novita_*.py"))


def _calls(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Attribute) and child.attr == name
        for child in ast.walk(node)
    )


def _guarded_by_stop_container(tree: ast.AST) -> list[ast.Try]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(_calls(handler, "stop_container") for handler in node.finalbody)
    ]


@pytest.mark.parametrize("path", NOVITA_EXAMPLES, ids=lambda path: path.name)
def test_wait_for_ready_runs_under_the_stop_container_guard(path: Path):
    tree = ast.parse(path.read_text())

    if not _calls(tree, "wait_for_ready"):
        pytest.skip(f"{path.name} does not wait for readiness")

    guards = _guarded_by_stop_container(tree)
    assert guards, f"{path.name} calls wait_for_ready without a stop_container finally"
    assert any(
        _calls(statement, "wait_for_ready")
        for guard in guards
        for statement in guard.body
    ), (
        f"{path.name} calls wait_for_ready outside the try that stops the sandbox; "
        "a readiness timeout would leak a paid sandbox"
    )
