"""
SQLite database management for AWM environments.

Creates databases from schema + sample data, manages snapshots for
verifier comparison (initial vs final state).
"""

import logging
import os
import shutil
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)


def create_database(db_path: str, db_schema: dict, sample_data: Any) -> str:
    """Create a SQLite database from schema and populate with sample data.

    Args:
        db_path: Path where the .db file will be created.
        db_schema: Schema dict from gen_db.jsonl (contains "tables" list with "ddl" and "indexes").
        sample_data: Sample data from gen_sample.jsonl (list of SQL INSERT dicts or raw statements) .

    Returns:
        The db_path on success.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    try:
        _create_schema(conn, db_schema)
        _insert_sample_data(conn, db_path, sample_data)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    logger.info(f"Created database: {db_path}")
    return db_path


def _create_schema(conn: sqlite3.Connection, db_schema: dict) -> None:
    cursor = conn.cursor()
    tables = db_schema.get("tables", [])
    for table in tables:
        ddl = table.get("ddl", "").strip()
        if ddl:
            try:
                cursor.execute(ddl)
            except sqlite3.Error as e:
                logger.warning(f"Failed to execute DDL: {e}\n  DDL: {ddl}")

        indexes = table.get("indexes", [])
        for idx in indexes:
            idx_stmt = str(idx).strip()
            if idx_stmt:
                try:
                    cursor.execute(idx_stmt)
                except sqlite3.Error as e:
                    logger.warning(f"Failed to create index: {e}\n  Index: {idx_stmt}")


def _insert_sample_data(
    conn: sqlite3.Connection, db_path: str, sample_data: Any
) -> None:
    """Insert sample data. Handles the AWM sample_data format which is a list
    of dicts with 'table_name' and 'insert_statements' keys."""
    if not sample_data:
        return

    cursor = conn.cursor()

    # AWM format wraps the list inside {"tables": [...]};  unwrap if needed.
    if isinstance(sample_data, dict) and "tables" in sample_data:
        sample_data = sample_data["tables"]

    if isinstance(sample_data, list):
        for item in sample_data:
            if isinstance(item, dict):
                statements = item.get("insert_statements", [])
                table_name = item.get("table_name", "unknown")
                for stmt in statements:
                    stmt = str(stmt).strip()
                    if stmt:
                        try:
                            cursor.execute(stmt)
                        except sqlite3.Error as e:
                            logger.warning(
                                f"Failed to insert into {table_name}: {e}\n  SQL: {stmt}"
                            )
            elif isinstance(item, str):
                item = item.strip()
                if item:
                    try:
                        cursor.execute(item)
                    except sqlite3.Error as e:
                        logger.warning(f"Failed to execute SQL: {e}\n  SQL: {item}")


def save_snapshot(db_path: str, snapshot_path: str) -> str:
    """Copy the database file as a snapshot for later verifier comparison."""
    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
    shutil.copy2(db_path, snapshot_path)
    return snapshot_path


def cleanup_session_dir(session_dir: str) -> None:
    """Remove the session temp directory and all contents."""
    if session_dir and os.path.isdir(session_dir):
        shutil.rmtree(session_dir, ignore_errors=True)
        logger.debug(f"Cleaned up session dir: {session_dir}")
