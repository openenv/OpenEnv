"""
Separate database for seed data storage
This keeps seed SQL content isolated from the main Calendar database
"""

import logging
import os
from datetime import datetime

from sqlalchemy import Column, create_engine, DateTime, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# Separate Base for seed database
SeedBase = declarative_base()

# Seed database path - stored in mcp_databases folder for persistence
SEED_DATABASE_PATH = os.getenv("SEED_DATABASE_PATH", "./mcp_databases/seed_store.db")
SEED_DATABASE_URL = f"sqlite:///{SEED_DATABASE_PATH}"

# Create engine for seed database
seed_engine = create_engine(
    SEED_DATABASE_URL, connect_args={"check_same_thread": False}, echo=False
)

# Create session factory for seed database
SeedSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=seed_engine)


class SeedData(SeedBase):
    """Model to store seed SQL content for databases"""

    __tablename__ = "seed_data"

    id = Column(Integer, primary_key=True, index=True)
    database_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    sql_content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self):
        return f"<SeedData(database_id={self.database_id}, name={self.name})>"


def init_seed_database():
    """Initialize the seed database and create tables"""
    # Ensure the mcp_databases directory exists
    db_dir = os.path.dirname(SEED_DATABASE_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Created directory: {db_dir}")

    logger.info(f"Initializing seed database at {SEED_DATABASE_PATH}")
    SeedBase.metadata.create_all(bind=seed_engine)
    logger.info("Seed database initialized successfully")


def get_seed_session():
    """Get a session for the seed database"""
    return SeedSessionLocal()


__all__ = ["SeedData", "get_seed_session", "init_seed_database"]
