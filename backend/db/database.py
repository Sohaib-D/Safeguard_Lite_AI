import os
import logging
from sqlalchemy import create_engine, exc
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

# Logger configuration
logger = logging.getLogger("backend.db")

# Database URL with SSL enforcement
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")

# Engine configuration (Production-safe)
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

def check_db_connection():
    try:
        with engine.connect() as conn:
            return True
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return False
