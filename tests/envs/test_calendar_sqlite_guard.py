"""calendar_env runs SQL sent by clients, so each connection must stay inside its own database.

Only the standard library is needed, so these run in CI without the calendar server deps.
"""

import importlib.util
import sqlite3
from pathlib import Path

import pytest

GUARD = (
    Path(__file__).resolve().parents[2]
    / "envs"
    / "calendar_env"
    / "server"
    / "database"
    / "sqlite_guard.py"
)
_spec = importlib.util.spec_from_file_location("calendar_sqlite_guard", GUARD)
sqlite_guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sqlite_guard)


@pytest.fixture
def conn(tmp_path):
    conn = sqlite_guard.connect_database(str(tmp_path / "calendar_alice.sqlite"))
    yield conn
    conn.close()


@pytest.mark.parametrize(
    "sql",
    [
        "ATTACH DATABASE '{other}' AS other",
        "ATTACH DATABASE 'file:{other}?mode=ro' AS other",
        "ATTACH DATABASE ':memory:' AS scratch",
        "VACUUM INTO '{copy}'",
    ],
)
def test_client_sql_cannot_open_or_write_other_files(conn, tmp_path, sql):
    other = tmp_path / "other.sqlite"
    sqlite3.connect(other).execute("create table secrets(v)")
    copy = tmp_path / "copy.sqlite"
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute(sql.format(other=other, copy=copy))
    assert not copy.exists()


def test_executescript_cannot_attach(conn, tmp_path):
    with pytest.raises(sqlite3.DatabaseError):
        conn.executescript(
            f"create table t(v); ATTACH DATABASE '{tmp_path / 'x.sqlite'}' AS x;"
        )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT fts3_tokenizer('simple')",
        "SELECT FTS3_TOKENIZER('simple')",
        "SELECT load_extension('anything')",
        "PRAGMA temp_store_directory = '/tmp'",
        "PRAGMA main.TEMP_STORE_DIRECTORY = '/tmp'",
        "PRAGMA data_store_directory = '/tmp'",
        # Harmless values: if the guard ever let these through, the test process keeps working.
        "PRAGMA hard_heap_limit = 1000000000000",
        "PRAGMA soft_heap_limit = 1000000000000",
    ],
)
def test_client_sql_cannot_change_the_process(conn, sql):
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute(sql)


def test_ordinary_sql_still_runs(conn):
    conn.execute("create table events(title)")
    conn.execute("insert into events values ('standup')")
    assert conn.execute(
        "select upper(title), length(title) from events"
    ).fetchall() == [("STANDUP", 7)]
    assert [row[1] for row in conn.execute("PRAGMA table_info(events)")] == ["title"]
    conn.execute("create virtual table notes using fts3(body)")
    conn.execute("insert into notes values ('team standup')")
    assert conn.execute(
        "select body from notes where notes match 'standup'"
    ).fetchall() == [("team standup",)]
