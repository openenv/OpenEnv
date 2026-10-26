"""
Base Manager - Common CRUD operations for all Calendar services
"""

import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseManager:
    """Base manager for common database operations"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute a SELECT query and return results"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT query and return the last row ID"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError as e:
            logger.error(f"Database integrity error: {e}")
            raise ValueError(f"Database constraint violation: {e}")
        finally:
            conn.close()

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an UPDATE query and return the number of affected rows"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def execute_delete(self, query: str, params: tuple = ()) -> int:
        """Execute a DELETE query and return the number of affected rows"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_by_id(self, table: str, record_id: int) -> Optional[Dict]:
        """Get a record by ID from any table"""
        query = f"SELECT * FROM {table} WHERE id = ?"
        results = self.execute_query(query, (record_id,))
        return results[0] if results else None

    def get_all(
        self, table: str, limit: int = 100, offset: int = 0, order_by: str = "id DESC"
    ) -> List[Dict]:
        """Get all records from a table with pagination"""
        query = f"SELECT * FROM {table} ORDER BY {order_by} LIMIT ? OFFSET ?"
        return self.execute_query(query, (limit, offset))

    def count_records(
        self, table: str, where_clause: str = "", params: tuple = ()
    ) -> int:
        """Count records in a table"""
        query = f"SELECT COUNT(*) as count FROM {table}"
        if where_clause:
            query += f" WHERE {where_clause}"
        result = self.execute_query(query, params)
        return result[0]["count"] if result else 0

    def update_record(
        self, table: str, record_id: int, updates: Dict[str, Any]
    ) -> bool:
        """Update a record with given field values"""
        if not updates:
            return False

        # Add updated_at timestamp if the table has this column
        updates["updated_at"] = datetime.now().isoformat()

        set_clauses = []
        params = []

        for field, value in updates.items():
            set_clauses.append(f"{field} = ?")
            params.append(value)

        params.append(record_id)

        query = f"UPDATE {table} SET {', '.join(set_clauses)} WHERE id = ?"
        affected_rows = self.execute_update(query, tuple(params))
        return affected_rows > 0

    def delete_record(self, table: str, record_id: int) -> bool:
        """Delete a record by ID"""
        query = f"DELETE FROM {table} WHERE id = ?"
        affected_rows = self.execute_delete(query, (record_id,))
        return affected_rows > 0

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = self.execute_query(query, (table_name,))
        return len(result) > 0
