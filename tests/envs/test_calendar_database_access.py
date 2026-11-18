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


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT fts3_tokenizer('simple')",
        "SELECT FTS3_TOKENIZER('simple')",
        "SELECT load_extension('anything')",
        "PRAGMA temp_store_directory = '/tmp'",
        "PRAGMA data_store_directory = '/tmp'",
    ],
)
def test_client_sql_cannot_reach_outside_sql(manager, sql):
    conn = connect_database(manager.get_db_path("alice"))
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute(sql)


def test_orm_connections_cannot_call_denied_functions(manager):
    from sqlalchemy import exc, text

    with manager.get_engine("alice").connect() as conn:
        with pytest.raises(exc.DatabaseError):
            conn.execute(text("SELECT fts3_tokenizer('simple')"))


def test_ordinary_functions_and_pragmas_still_work(manager):
    conn = connect_database(manager.get_db_path("alice"))
    conn.execute("create table events(title)")
    assert conn.execute("select upper('a'), length('abc')").fetchall() == [("A", 3)]
    assert [row[1] for row in conn.execute("PRAGMA table_info(events)")] == ["title"]
    try:
        conn.execute("create virtual table notes using fts5(body)")
    except sqlite3.OperationalError:
        pytest.skip("this sqlite has no fts5")
    conn.execute("insert into notes values ('team standup')")
    assert conn.execute(
        "select body from notes where notes match 'standup'"
    ).fetchall() == [("team standup",)]
