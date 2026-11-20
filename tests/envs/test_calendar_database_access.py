"""calendar_env runs SQL sent by clients, so it must stay inside the session's own database."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

CALENDAR_SERVER = (
    Path(__file__).resolve().parents[2] / "envs" / "calendar_env" / "server"
)
if str(CALENDAR_SERVER) not in sys.path:
    sys.path.insert(0, str(CALENDAR_SERVER))

from database.session_manager import CalendarSessionManager  # noqa: E402


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # databases live in ./mcp_databases
    return CalendarSessionManager()


@pytest.mark.parametrize(
    "db_id", ["../victim", "X/../../victim", "/tmp/victim", "", "a" * 65, "a.b"]
)
def test_database_id_cannot_name_a_path(manager, db_id):
    with pytest.raises(ValueError, match="Invalid database id"):
        manager.get_db_path(db_id)


def test_database_id_names_a_file_in_the_databases_folder(manager):
    assert (
        manager.get_db_path("alice-2_B") == "./mcp_databases/calendar_alice-2_B.sqlite"
    )


def test_orm_connections_cannot_attach_other_files(manager, tmp_path):
    from sqlalchemy import exc, text

    with manager.get_engine("alice").connect() as conn:
        with pytest.raises(exc.DatabaseError):
            conn.execute(text(f"ATTACH DATABASE '{tmp_path / 'x.sqlite'}' AS x"))


def test_orm_connections_cannot_call_denied_functions(manager):
    from sqlalchemy import exc, text

    with manager.get_engine("alice").connect() as conn:
        with pytest.raises(exc.DatabaseError):
            conn.execute(text("SELECT fts3_tokenizer('simple')"))
