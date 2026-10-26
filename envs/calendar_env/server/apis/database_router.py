"""
Database Management API endpoints for Calendar API
Handles database operations like seeding, viewing, and schema management
"""

import logging
import os
import random
import shutil
import sqlite3
import string
import time
from datetime import datetime
from typing import Any, Dict, Optional

from data.google_colors import GOOGLE_CALENDAR_COLORS
from data.multi_user_sample import get_multi_user_sql
from database.base_manager import BaseManager
from database.managers.calendar_manager import CalendarManager
from database.seed_database import get_seed_session, init_seed_database, SeedData
from database.session_manager import CalendarSessionManager
from database.session_utils import _session_manager
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Initialize session manager
calendar_session_manager = CalendarSessionManager()

router = APIRouter(prefix="/api", tags=["database"])


@router.get("/sample-data")
async def get_sample_data():
    """Get multi-user Calendar sample data as SQL script for download/inspection"""
    try:
        sql_content = get_multi_user_sql()

        return {
            "success": True,
            "message": "Sample calendar data.",
            "format": "sql",
            "sql_content": sql_content,
        }
    except Exception as e:
        logger.error(f"Error retrieving sample data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve sample data: {str(e)}"
        )


@router.post("/seed-database")
async def seed_database(body: dict):
    """Seed database with user-provided SQL content or default sample data"""
    try:
        database_id = body.get("database_id")
        if not database_id:
            raise HTTPException(status_code=400, detail="database_id is required")
        sql_content = body.get("sql_content")

        if not sql_content:
            raise HTTPException(status_code=400, detail="sql_content is required")

        # Initialize database
        calendar_session_manager.init_database(database_id, create_tables=True)
        db_path = calendar_session_manager.get_db_path(database_id)

        # Execute SQL content
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()

            # Drop all existing tables to ensure fresh schema from SQLAlchemy models
            # Get all table names dynamically
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = cursor.fetchall()

            # Drop all tables to get fresh schema
            for table in tables:
                table_name = table[0]
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

            conn.commit()
            conn.close()

            # Reinitialize database with fresh schema from SQLAlchemy models
            calendar_session_manager.init_database(database_id, create_tables=True)

            # Reconnect to database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Execute custom SQL statements individually
            statements = []
            for line in sql_content.split("\n"):
                line = line.strip()
                if line and not line.startswith("--"):
                    statements.append(line)

            # Join and split by semicolon
            full_sql = " ".join(statements)
            individual_statements = [
                stmt.strip() for stmt in full_sql.split(";") if stmt.strip()
            ]

            # Execute each statement
            for statement in individual_statements:
                try:
                    # Skip empty statements
                    if not statement.strip():
                        continue
                    cursor.execute(statement)
                except Exception as e:
                    logger.error(f"Error executing statement: {statement[:100]}...")
                    logger.error(f"Error details: {e}")
                    raise e

            conn.commit()

            # Enforce primary calendar constraints and ensure ACL entries exist
            try:
                calendar_manager = CalendarManager(database_id)
                calendar_manager.enforce_primary_calendar_constraint_for_all_users()

                # Ensure ACL entries exist for all calendar owners
                # This is important for the new ACL validation system
                logger.info("Ensuring ACL entries exist for all calendar owners...")

                # Get all calendars and ensure their owners have ACL entries
                cursor.execute("SELECT calendar_id, user_id FROM calendars")
                calendars = cursor.fetchall()

                for calendar_row in calendars:
                    calendar_id = calendar_row[0]
                    user_id = calendar_row[1]

                    # Check if ACL entry already exists for this calendar owner
                    cursor.execute(
                        """
                        SELECT COUNT(*) as count FROM acls a
                        JOIN scopes s ON a.scope_id = s.id
                        JOIN users u ON u.email = s.value
                        WHERE a.calendar_id = ? AND u.user_id = ? AND a.role = 'owner'
                    """,
                        (calendar_id, user_id),
                    )

                    acl_exists = cursor.fetchone()[0] > 0

                    if not acl_exists:
                        # Get user email
                        cursor.execute(
                            "SELECT email FROM users WHERE user_id = ?", (user_id,)
                        )
                        user_result = cursor.fetchone()

                        if user_result:
                            user_email = user_result[0]

                            # Create scope for this user if it doesn't exist
                            scope_id = f"scope-{user_id}"
                            cursor.execute(
                                "INSERT OR IGNORE INTO scopes (id, type, value) VALUES (?, 'user', ?)",
                                (scope_id, user_email),
                            )

                            # Create ACL entries for calendar owner (both owner and writer roles)
                            owner_acl_id = f"acl-{calendar_id}-owner"
                            writer_acl_id = f"acl-{calendar_id}-writer"

                            cursor.execute(
                                """
                                INSERT OR IGNORE INTO acls (id, calendar_id, user_id, role, scope_id, etag, created_at, updated_at)
                                VALUES (?, ?, ?, 'owner', ?, ?, datetime('now'), datetime('now'))
                            """,
                                (
                                    owner_acl_id,
                                    calendar_id,
                                    user_id,
                                    scope_id,
                                    f"etag-{owner_acl_id}",
                                ),
                            )

                            cursor.execute(
                                """
                                INSERT OR IGNORE INTO acls (id, calendar_id, user_id, role, scope_id, etag, created_at, updated_at)
                                VALUES (?, ?, ?, 'writer', ?, ?, datetime('now'), datetime('now'))
                            """,
                                (
                                    writer_acl_id,
                                    calendar_id,
                                    user_id,
                                    scope_id,
                                    f"etag-{writer_acl_id}",
                                ),
                            )

                            logger.info(
                                f"Created owner and writer ACL entries for calendar {calendar_id} owner {user_email}"
                            )

                conn.commit()
                logger.info("ACL validation completed successfully")

            except Exception as constraint_error:
                logger.warning(
                    f"Error enforcing constraints and ACL validation: {constraint_error}"
                )

            # Store SQL content in seed_data table for future resets
            sql_stored = False
            try:
                seed_db_session = get_seed_session()
                try:
                    existing_seed = (
                        seed_db_session.query(SeedData)
                        .filter(SeedData.database_id == database_id)
                        .first()
                    )

                    name = body.get("name", f"Database {database_id}")
                    description = body.get("description", "")

                    if existing_seed:
                        existing_seed.sql_content = sql_content
                        existing_seed.name = name
                        existing_seed.description = description
                        logger.info(f"Updated seed SQL for database {database_id}")
                    else:
                        seed_entry = SeedData(
                            database_id=database_id,
                            name=name,
                            description=description,
                            sql_content=sql_content,
                        )
                        seed_db_session.add(seed_entry)
                        logger.info(f"Stored new seed SQL for database {database_id}")

                    seed_db_session.commit()
                    sql_stored = True
                finally:
                    seed_db_session.close()
            except Exception as e:
                logger.warning(f"Failed to store seed SQL in database: {e}")
                sql_stored = False

            result = {
                "success": True,
                "message": f"Database seeded successfully",
                "database_id": database_id,
                "sql_stored": sql_stored,
            }

            logger.info(f"Successfully seeded database {database_id}")
            return result

        finally:
            conn.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to seed database: {str(e)}"
        )


@router.get("/database-state")
async def get_database_state(x_database_id: str = Header(alias="x-database-id")):
    """Get current database state and all data for frontend tabular display"""
    try:
        database_id = x_database_id

        # Initialize database if needed
        calendar_session_manager.init_database(database_id)
        db_path = calendar_session_manager.get_db_path(database_id)

        if not os.path.exists(db_path):
            return {
                "success": True,
                "message": "Database not initialized yet",
                "database_info": {"status": "not_created"},
                "table_counts": {},
                "table_data": {},
            }

        # Connect to database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.cursor()

            # Get all user-defined tables
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """
            )
            tables = cursor.fetchall()

            # Get data from each table
            table_data = {}
            table_counts = {}

            for table in tables:
                table_name = table["name"]

                # Get count
                cursor.execute(f"SELECT COUNT(*) as count FROM '{table_name}'")
                count = cursor.fetchone()["count"]
                table_counts[table_name] = count

                # Get data (limit to 100 rows for performance)
                cursor.execute(f"SELECT * FROM '{table_name}' LIMIT 100")
                rows = cursor.fetchall()

                # Convert to list of dictionaries
                table_data[table_name] = [dict(row) for row in rows]

            # Get database info
            db_file_size = os.path.getsize(db_path)
            db_modified = datetime.fromtimestamp(os.path.getmtime(db_path)).isoformat()

            return {
                "success": True,
                "database_id": database_id,
                "service": "calendar-api-server",
                "database_info": {
                    "path": db_path,
                    "size_bytes": db_file_size,
                    "size_mb": round(db_file_size / (1024 * 1024), 2),
                    "last_modified": db_modified,
                    "total_tables": len(tables),
                },
                "table_counts": table_counts,
                "table_data": table_data,
                "timestamp": datetime.now().isoformat(),
            }

        finally:
            conn.close()

    except Exception as e:
        logger.error(f"Error getting database state: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get database state: {str(e)}"
        )


@router.get("/schema")
async def get_schema():
    """Get database schema as JSON"""
    try:
        # Return static schema definition from SQLAlchemy models
        return {
            "success": True,
            "message": "Database schema retrieved successfully",
            "schema": calendar_session_manager.get_database_schema(),
        }

    except Exception as e:
        logger.error(f"Error getting database schema: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve database schema: {str(e)}"
        )


@router.get("/download-db-file")
async def download_database_file(
    x_database_id: str = Header(default="default", alias="x-database-id"),
):
    """Download the actual SQLite database file (.db)"""
    try:
        database_id = x_database_id.strip()

        # Basic input validation
        if not database_id or database_id == "":
            raise HTTPException(status_code=400, detail="Database ID cannot be empty")

        logger.info(f"-------Download DB File for {database_id}------")

        # Initialize database if needed
        calendar_session_manager.init_database(database_id)
        db_path = calendar_session_manager.get_db_path(database_id)

        # Check if file exists and is readable
        if not os.path.exists(db_path):
            raise HTTPException(
                status_code=404, detail=f"Database with ID '{database_id}' not found"
            )

        if not os.path.isfile(db_path):
            raise HTTPException(
                status_code=400, detail=f"Database path '{db_path}' is not a valid file"
            )

        if not os.access(db_path, os.R_OK):
            raise HTTPException(
                status_code=403, detail=f"Database file is not readable"
            )

        # Generate filename for download with sanitized database_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_database_id = "".join(c for c in database_id if c.isalnum() or c in "_-")
        filename = f"ola_database_{safe_database_id}_{timestamp}.db"

        logger.info(f"Serving database file: {db_path} as {filename}")

        # Return the actual database file
        return FileResponse(
            path=db_path,
            filename=filename,
            media_type="application/x-sqlite3",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading database file: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to download database file: {str(e)}"
        )


class SQLQueryRequest(BaseModel):
    """Request model for executing SQL queries"""

    query: str = Field(..., min_length=1, description="SQL query to execute")
    limit: Optional[int] = Field(
        100, ge=1, le=1000, description="Maximum number of rows to return"
    )


@router.post("/sql-runner", response_model=Dict[str, Any])
def execute_custom_query(request: SQLQueryRequest, x_database_id: str = Header(...)):
    """Execute a custom SQL query with limit"""
    try:
        # Ensure database exists
        calendar_session_manager.init_database(x_database_id)
        db_path = calendar_session_manager.get_db_path(x_database_id)

        # Use BaseManager to execute the query
        manager = BaseManager(db_path)

        # Add LIMIT clause to query if not present
        query = request.query.strip().rstrip(";")

        # Execute the query
        data = manager.execute_query(query)

        return {
            "success": True,
            "message": "Query executed successfully",
            "data": data,
            "row_count": len(data),
            "query": query,
        }
    except Exception as e:
        logger.error(f"Error executing custom query: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reset-database")
async def reset_database(body: dict):
    """Reset a database to its original seeded state."""
    try:
        database_id = body.get("database_id")
        if not database_id:
            raise HTTPException(status_code=400, detail="database_id is required")

        sql_content = body.get("sql_content")

        # Get or store seed SQL
        seed_session = get_seed_session()
        try:
            seed_data = (
                seed_session.query(SeedData)
                .filter(SeedData.database_id == database_id)
                .first()
            )

            if seed_data:
                sql_content = seed_data.sql_content
                logger.info(f"Using stored SQL for database {database_id}")
            elif sql_content:
                new_seed = SeedData(
                    database_id=database_id,
                    name=f"calendar_{database_id}",
                    description="Seed SQL for Calendar database (lazy migration)",
                    sql_content=sql_content,
                )
                seed_session.add(new_seed)
                seed_session.commit()
                logger.info(f"Stored SQL via lazy migration for database {database_id}")
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No stored SQL found and no sql_content provided. Cannot reset database.",
                )
        finally:
            seed_session.close()

        # Close existing session and get db path
        calendar_session_manager.close_session(database_id)
        db_path = calendar_session_manager.get_db_path(database_id)

        # Delete the database file completely (faster than dropping tables)
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info(f"Deleted database file: {db_path}")

        # Reinitialize with fresh tables
        calendar_session_manager.init_database(database_id, create_tables=True)

        # Execute seed SQL in one go using executescript
        conn = sqlite3.connect(db_path)
        try:
            # executescript is much faster than executing statements one by one
            conn.executescript(sql_content)
            conn.commit()
            logger.info(f"Database {database_id} reset successfully")

            return {
                "success": True,
                "message": "Database reset successfully",
                "database_id": database_id,
            }
        finally:
            conn.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to reset database: {str(e)}"
        )


@router.post("/clone-database")
async def clone_database(body: dict):
    """
    Clone an existing database to a new database.
    Returns the new database_id.
    """
    try:
        # Accept both "database_id" and "source_database_id" for compatibility
        source_database_id = body.get("database_id") or body.get("source_database_id")
        if not source_database_id:
            raise HTTPException(status_code=400, detail="database_id is required")

        logger.info(f"Clone database request for: {source_database_id}")

        # Get source database path
        source_db_path = calendar_session_manager.get_db_path(source_database_id)

        if not os.path.exists(source_db_path):
            raise HTTPException(
                status_code=404,
                detail=f"Source database '{source_database_id}' not found",
            )

        # Generate new database ID
        timestamp = int(time.time() * 1000)
        random_suffix = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=9)
        )
        new_db_id = f"db_clone_{timestamp}_{random_suffix}"

        # Get new database path
        new_db_path = calendar_session_manager.get_db_path(new_db_id)

        # Close any existing session on source to ensure file is not locked
        calendar_session_manager.close_session(source_database_id)

        # Clone the database file
        shutil.copy2(source_db_path, new_db_path)

        # Copy associated files (WAL, SHM, journal) if they exist
        for ext in ["-wal", "-shm", "-journal"]:
            source_extra = source_db_path + ext
            if os.path.exists(source_extra):
                dest_extra = new_db_path + ext
                shutil.copy2(source_extra, dest_extra)
                logger.debug(f"Copied {source_extra} to {dest_extra}")

        # Initialize the session for the cloned database
        calendar_session_manager.init_database(new_db_id, create_tables=False)

        # Get cloned database size
        cloned_size_bytes = (
            os.path.getsize(new_db_path) if os.path.exists(new_db_path) else 0
        )

        result = {
            "success": True,
            "message": "Database cloned successfully",
            "source_database_id": source_database_id,
            "cloned_database_id": new_db_id,
            "cloned_db_path": new_db_path,
            "cloned_size_bytes": cloned_size_bytes,
        }

        logger.info(
            f"Successfully cloned database '{source_database_id}' to '{new_db_id}'"
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cloning database: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to clone database: {str(e)}"
        )


@router.delete("/delete-database")
async def delete_database(body: dict):
    """
    Delete a cloned database.
    WARNING: This permanently deletes the database and cannot be undone.
    """
    try:
        database_id = body.get("database_id")
        if not database_id:
            raise HTTPException(
                status_code=400, detail="database_id is required in request body"
            )

        logger.info(f"Delete database request for: {database_id}")

        # Safety check: don't allow deleting the default database
        if database_id == "default":
            raise HTTPException(
                status_code=400, detail="Cannot delete the default database"
            )

        # Get database path
        db_path = calendar_session_manager.get_db_path(database_id)

        if not os.path.exists(db_path):
            raise HTTPException(
                status_code=404, detail=f"Database with ID '{database_id}' not found"
            )

        # Close any existing sessions
        calendar_session_manager.close_session(database_id)

        # Get file size before deletion
        db_size = os.path.getsize(db_path)

        # Delete the database file
        os.remove(db_path)

        # Delete associated files (WAL, SHM, journal) if they exist
        for ext in ["-wal", "-shm", "-journal"]:
            extra_file = db_path + ext
            if os.path.exists(extra_file):
                os.remove(extra_file)
                logger.debug(f"Deleted {extra_file}")

        # Also remove from seed_data table if exists
        try:
            seed_db_session = get_seed_session()
            try:
                seed_entry = (
                    seed_db_session.query(SeedData)
                    .filter(SeedData.database_id == database_id)
                    .first()
                )
                if seed_entry:
                    seed_db_session.delete(seed_entry)
                    seed_db_session.commit()
                    logger.info(f"Deleted seed data for database {database_id}")
            finally:
                seed_db_session.close()
        except Exception as e:
            logger.warning(f"Failed to delete seed data: {e}")

        result = {
            "success": True,
            "message": "Database deleted successfully",
            "database_id": database_id,
            "deleted_size_bytes": db_size,
        }

        logger.info(f"Successfully deleted database {database_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting database: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete database: {str(e)}"
        )
