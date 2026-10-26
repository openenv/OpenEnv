"""Tests that calendar_env UserManager does not leak access tokens to logs."""

import hashlib
import importlib.util
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CALENDAR_SERVER = REPO_ROOT / "envs" / "calendar_env" / "server"
USER_MANAGER_PATH = CALENDAR_SERVER / "database" / "managers" / "user_manager.py"


def _load_user_manager_module():
    """Import user_manager.py directly"""
    # Make `from database.x import ...` inside user_manager.py resolve.
    server_path = str(CALENDAR_SERVER)
    if server_path not in sys.path:
        sys.path.insert(0, server_path)

    spec = importlib.util.spec_from_file_location(
        "calendar_env_user_manager", str(USER_MANAGER_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def user_manager_module():
    if not USER_MANAGER_PATH.exists():
        pytest.skip("calendar_env user_manager.py not present in this checkout")
    try:
        return _load_user_manager_module()
    except Exception as exc:  # pragma: no cover - import-time failure
        pytest.skip(f"calendar_env server deps unavailable: {exc}")


def test_token_fingerprint_is_deterministic_short_sha256(user_manager_module):
    """Fingerprint must be the first 8 chars of sha256(token), no exceptions."""
    token = "ya29.A0ARrdaM-k9Vq7GzY2pL4mQf8sN1xT0bR3uHcJWv5yKzP6eF2"
    expected = hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]

    fp = user_manager_module._token_fingerprint(token)

    assert fp == expected
    assert len(fp) == 8
    assert token not in fp


def test_token_fingerprint_handles_empty(user_manager_module):
    assert user_manager_module._token_fingerprint("") == "<empty>"
    assert user_manager_module._token_fingerprint(None) == "<empty>"


def test_get_user_by_access_token_does_not_log_token(
    user_manager_module, monkeypatch, caplog
):
    """When the DB session raises, the raw token must not appear in logs."""
    UserManager = user_manager_module.UserManager
    token = "ya29.A0ARrdaM-SECRET-TOKEN-VALUE-do-not-leak"

    failing_session = MagicMock()
    failing_session.query.side_effect = RuntimeError("simulated DB failure")
    monkeypatch.setattr(user_manager_module, "get_session", lambda _id: failing_session)
    monkeypatch.setattr(user_manager_module, "init_database", lambda _id: None)

    manager = UserManager.__new__(UserManager)
    manager.database_id = "test-db"

    with caplog.at_level(logging.ERROR, logger=user_manager_module.__name__):
        with pytest.raises(RuntimeError, match="simulated DB failure"):
            manager.get_user_by_access_token(token)

    assert token not in caplog.text, "raw access token leaked into log output"
    expected_fp = hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]
    assert f"fingerprint={expected_fp}" in caplog.text, (
        "log line should include a token fingerprint for correlation"
    )
    failing_session.close.assert_called_once()


def test_get_user_by_access_token_does_not_leak_token_via_exception_str(
    user_manager_module, monkeypatch, caplog
):
    """The token must not leak via str(exc) either.

    SQLAlchemy StatementError / DBAPIError include the bound parameters dict
    in their string form, e.g.:
        '... [parameters: {"static_token_1": "ya29..."}] ...'
    A naive `logger.error(f"...: {e}")` would emit the raw token even with
    the f-string's other slots sanitized. The current implementation logs
    only `type(e).__name__`, so this test pins that behavior.
    """
    UserManager = user_manager_module.UserManager
    token = "ya29.A0ARrdaM-SECRET-INSIDE-EXC-STR"

    class FakeStatementError(Exception):
        def __str__(self) -> str:
            return (
                "(sqlite3.OperationalError) no such column "
                "[SQL: SELECT users.* FROM users WHERE users.static_token = ?] "
                f"[parameters: ({token!r},)]"
            )

    failing_session = MagicMock()
    failing_session.query.side_effect = FakeStatementError()
    monkeypatch.setattr(user_manager_module, "get_session", lambda _id: failing_session)
    monkeypatch.setattr(user_manager_module, "init_database", lambda _id: None)

    manager = UserManager.__new__(UserManager)
    manager.database_id = "test-db"

    with caplog.at_level(logging.ERROR, logger=user_manager_module.__name__):
        with pytest.raises(FakeStatementError):
            manager.get_user_by_access_token(token)

    assert token not in caplog.text, "raw token leaked via str(exception) in log output"
    expected_fp = hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]
    assert f"fingerprint={expected_fp}" in caplog.text
    assert "FakeStatementError" in caplog.text, (
        "log should still identify the exception type for diagnostics"
    )
    assert "[parameters:" not in caplog.text, (
        "the SQLAlchemy-style parameter dump must not reach the log sink"
    )


def test_get_user_by_access_token_propagates_original_exception(
    user_manager_module, monkeypatch
):
    """Sanity: redacting the log line must not swallow the underlying error."""
    UserManager = user_manager_module.UserManager

    failing_session = MagicMock()
    failing_session.query.side_effect = ValueError("boom")
    monkeypatch.setattr(user_manager_module, "get_session", lambda _id: failing_session)
    monkeypatch.setattr(user_manager_module, "init_database", lambda _id: None)

    manager = UserManager.__new__(UserManager)
    manager.database_id = "test-db"

    with pytest.raises(ValueError, match="boom"):
        manager.get_user_by_access_token("any-token")
