"""
db.py — User profile persistence.

Two backends:
  • PostgreSQL + pgvector  (DATABASE_URL starts with postgres)
    - users table: interests, reading_history (JSON)
    - reading_history_vectors table: per-article embeddings for semantic
      long-term memory (cosine similarity via pgvector)
  • SQLite fallback (default, no extra deps)
    - Same users table; no vector column (embeddings stored as JSON blobs).

The public API is identical regardless of backend:
  init_db(), save_user(), load_user(), delete_user(),
  save_reading_vector(), query_similar_history()
"""
import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import (
    Column, DateTime, Integer, String, Text, create_engine, text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/hypernews.db")
_IS_PG = DATABASE_URL.startswith("postgres")

engine = create_engine(
    DATABASE_URL,
    connect_args={} if _IS_PG else {"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ── ORM models ────────────────────────────────────────────────────────────────

class UserRecord(Base):
    __tablename__ = "users"
    user_id         = Column(String, primary_key=True, index=True)
    interests       = Column(Text, default="{}")    # JSON: {category: float}
    reading_history = Column(Text, default="[]")    # JSON: [news_id, ...]
    total_reads     = Column(Integer, default=0)
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ── DB init ───────────────────────────────────────────────────────────────────

def init_db():
    """Create tables (and pgvector extension if on Postgres)."""
    os.makedirs("data", exist_ok=True)
    if _IS_PG:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        # reading_history_vectors: stores per-article embedding for long-term
        # semantic memory. Uses pgvector's cosine distance (<=>).
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS reading_history_vectors (
                    id          SERIAL PRIMARY KEY,
                    user_id     VARCHAR NOT NULL,
                    article_id  VARCHAR NOT NULL,
                    read_at     TIMESTAMP DEFAULT NOW(),
                    feedback_weight FLOAT DEFAULT 1.0,
                    embedding   vector(384),
                    UNIQUE (user_id, article_id)
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_rhv_user
                ON reading_history_vectors (user_id, read_at DESC)
            """))
            conn.commit()
    else:
        # SQLite: store embeddings as JSON blobs (no cosine search, just retrieval)
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS reading_history_vectors (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id         TEXT NOT NULL,
                    article_id      TEXT NOT NULL,
                    read_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    feedback_weight REAL DEFAULT 1.0,
                    embedding_json  TEXT,
                    UNIQUE (user_id, article_id)
                )
            """))
            conn.commit()
    Base.metadata.create_all(bind=engine)


# ── User CRUD ─────────────────────────────────────────────────────────────────

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
        # Also purge vector history
        with engine.connect() as conn:
            conn.execute(
                text("DELETE FROM reading_history_vectors WHERE user_id = :uid"),
                {"uid": user_id},
            )
            conn.commit()
    finally:
        db.close()


# ── Vector history (pgvector / SQLite-blob) ───────────────────────────────────

def save_reading_vector(
    user_id: str,
    article_id: str,
    embedding: np.ndarray,
    feedback_weight: float = 1.0,
):
    """Persist an article embedding into long-term vector history.

    On Postgres uses pgvector; on SQLite stores JSON-serialised floats.
    Upserts on (user_id, article_id) so re-reading updates the weight.
    """
    vec = embedding.astype(np.float32).tolist()
    with engine.connect() as conn:
        if _IS_PG:
            conn.execute(
                text("""
                    INSERT INTO reading_history_vectors
                        (user_id, article_id, read_at, feedback_weight, embedding)
                    VALUES (:uid, :aid, NOW(), :fw, :emb::vector)
                    ON CONFLICT (user_id, article_id) DO UPDATE
                        SET read_at = NOW(),
                            feedback_weight = EXCLUDED.feedback_weight,
                            embedding = EXCLUDED.embedding
                """),
                {"uid": user_id, "aid": article_id, "fw": feedback_weight, "emb": str(vec)},
            )
        else:
            conn.execute(
                text("""
                    INSERT OR REPLACE INTO reading_history_vectors
                        (user_id, article_id, read_at, feedback_weight, embedding_json)
                    VALUES (:uid, :aid, CURRENT_TIMESTAMP, :fw, :emb)
                """),
                {"uid": user_id, "aid": article_id, "fw": feedback_weight, "emb": json.dumps(vec)},
            )
        conn.commit()


def query_similar_history(
    user_id: str,
    query_embedding: np.ndarray,
    top_k: int = 20,
) -> list[dict]:
    """Return the top-k past articles whose stored embedding is closest to
    query_embedding (cosine distance on Postgres, dot-product fallback on SQLite).

    Returns list of {"article_id": str, "feedback_weight": float, "similarity": float}.
    """
    vec = query_embedding.astype(np.float32)

    with engine.connect() as conn:
        if _IS_PG:
            rows = conn.execute(
                text("""
                    SELECT article_id, feedback_weight,
                           1 - (embedding <=> :emb::vector) AS similarity
                    FROM reading_history_vectors
                    WHERE user_id = :uid
                    ORDER BY embedding <=> :emb::vector
                    LIMIT :k
                """),
                {"uid": user_id, "emb": str(vec.tolist()), "k": top_k},
            ).fetchall()
            return [
                {"article_id": r[0], "feedback_weight": r[1], "similarity": float(r[2])}
                for r in rows
            ]
        else:
            # SQLite: load all vectors and compute dot-product in Python
            rows = conn.execute(
                text("""
                    SELECT article_id, feedback_weight, embedding_json
                    FROM reading_history_vectors
                    WHERE user_id = :uid
                    ORDER BY read_at DESC
                    LIMIT 200
                """),
                {"uid": user_id},
            ).fetchall()
            if not rows:
                return []
            results = []
            for article_id, fw, emb_json in rows:
                try:
                    stored = np.array(json.loads(emb_json), dtype=np.float32)
                    sim = float(np.dot(vec, stored) / (
                        np.linalg.norm(vec) * np.linalg.norm(stored) + 1e-9
                    ))
                    results.append({"article_id": article_id, "feedback_weight": fw, "similarity": sim})
                except Exception:
                    continue
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
