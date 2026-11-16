"""calendar_env runs SQL sent by clients, so it must stay inside the session's own database."""

import sqlite3
import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

CALENDAR_SERVER = (
    Path(__file__).resolve().parents[2] / "envs" / "calendar_env" / "server"
)
if str(CALENDAR_SERVER) not in sys.path:
    sys.path.insert(0, str(CALENDAR_SERVER))

from database.session_manager import (  # noqa: E402
    CalendarSessionManager,
    connect_database,
)


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


@pytest.mark.parametrize(
    "sql",
    [
        "ATTACH DATABASE '{other}' AS other",
        "VACUUM INTO '{copy}'",
    ],
)
def test_client_sql_cannot_open_or_write_other_files(manager, tmp_path, sql):
    other = tmp_path / "other.sqlite"
    sqlite3.connect(other).execute("create table secrets(v)")
    copy = tmp_path / "copy.sqlite"
    conn = connect_database(manager.get_db_path("alice"))
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute(sql.format(other=other, copy=copy))
    assert not copy.exists()


def test_client_sql_still_runs_on_its_own_database(manager):
    conn = connect_database(manager.get_db_path("alice"))
    conn.execute("create table events(title)")
    conn.execute("insert into events values ('standup')")
    assert conn.execute("select title from events").fetchall() == [("standup",)]


def test_orm_connections_cannot_attach_other_files(manager, tmp_path):
    from sqlalchemy import exc, text

    with manager.get_engine("alice").connect() as conn:
        with pytest.raises(exc.DatabaseError):
            conn.execute(text(f"ATTACH DATABASE '{tmp_path / 'x.sqlite'}' AS x"))
