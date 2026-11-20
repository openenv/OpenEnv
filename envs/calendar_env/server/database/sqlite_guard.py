"""SQLite connections for SQL that may come from a client.

Kept free of SQLAlchemy so the guard can be tested with the standard library alone.
"""

import sqlite3

# Functions that reach outside SQL: extension loading and raw tokenizer pointers.
_DENIED_FUNCTIONS = {"fts3_tokenizer", "load_extension"}
# Pragmas that change settings for the whole process, not just this database.
_DENIED_PRAGMAS = {
    "temp_store_directory",
    "data_store_directory",
    "hard_heap_limit",
    "soft_heap_limit",
}


def _deny_other_files(action: int, arg1, arg2, *_args) -> int:
    """Keep client SQL inside its own database file.

    ATTACH and DETACH would open other files. VACUUM INTO, and plain VACUUM,
    attach a database internally, so they are denied too.
    """
    if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() in _DENIED_PRAGMAS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def connect_database(db_path: str) -> sqlite3.Connection:
    """Open a session database for SQL that may come from a client."""
    conn = sqlite3.connect(db_path)
    conn.set_authorizer(_deny_other_files)
    return conn
