"""
db.py — SQLite long-term user profile store via SQLAlchemy.
Stores interest weights, reading history, and read counts persistently.
"""
import json
import os
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/hypernews.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserRecord(Base):
    __tablename__ = "users"
    user_id       = Column(String, primary_key=True, index=True)
    interests     = Column(Text, default="{}")       # JSON: {category: float}
    reading_history = Column(Text, default="[]")    # JSON: [news_id, ...]
    total_reads   = Column(Integer, default=0)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def init_db():
    """Create tables if they don't exist."""
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)


def save_user(user_id: str, interests: dict, reading_history: list, total_reads: int):
    db = SessionLocal()
    try:
        record = db.query(UserRecord).filter(UserRecord.user_id == user_id).first()
        if record:
            record.interests       = json.dumps(interests)
            record.reading_history = json.dumps(reading_history)
            record.total_reads     = total_reads
            record.updated_at      = datetime.utcnow()
        else:
            record = UserRecord(
                user_id=user_id,
                interests=json.dumps(interests),
                reading_history=json.dumps(reading_history),
                total_reads=total_reads,
            )
            db.add(record)
        db.commit()
    finally:
        db.close()


def load_user(user_id: str) -> Optional[dict]:
    db = SessionLocal()
    try:
        record = db.query(UserRecord).filter(UserRecord.user_id == user_id).first()
        if not record:
            return None
        return {
            "user_id":         record.user_id,
            "interests":       json.loads(record.interests),
            "reading_history": json.loads(record.reading_history),
            "total_reads":     record.total_reads,
        }
    finally:
        db.close()


def delete_user(user_id: str):
    db = SessionLocal()
    try:
        record = db.query(UserRecord).filter(UserRecord.user_id == user_id).first()
        if record:
            db.delete(record)
            db.commit()
    finally:
        db.close()
