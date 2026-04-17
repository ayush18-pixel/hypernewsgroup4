"""
user_profile.py — In-memory user model with Redis session layer (falls back to
in-process dict when Redis is unavailable) and SQLite for long-term persistence.
"""
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

# ── Redis with graceful fallback ──────────────────────────────────────────────
try:
    import redis as _redis_lib
    _r = _redis_lib.Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        decode_responses=True,
        socket_connect_timeout=1,
    )
    _r.ping()
    _REDIS_OK = True
except Exception:
    _r = None
    _REDIS_OK = False

# Simple TTL-aware in-memory fallback
_mem_store: dict = {}   # {key: (value, expires_at)}


def _redis_set(key: str, value: str, ttl: int = 3600):
    if _REDIS_OK:
        _r.setex(key, ttl, value)
    else:
        _mem_store[key] = (value, time.time() + ttl)


def _redis_get(key: str) -> Optional[str]:
    if _REDIS_OK:
        return _r.get(key)
    entry = _mem_store.get(key)
    if entry and time.time() < entry[1]:
        return entry[0]
    return None


def _redis_delete(key: str):
    if _REDIS_OK:
        _r.delete(key)
    else:
        _mem_store.pop(key, None)


# ── User Profile ──────────────────────────────────────────────────────────────
@dataclass
class UserProfile:
    user_id:         str
    interests:       Dict[str, float] = field(default_factory=dict)
    reading_history: List[str]        = field(default_factory=list)
    avg_dwell_time:  float            = 0.0
    mood:            str              = "neutral"
    time_of_day:     str              = "morning"
    recent_clicks:   List[str]        = field(default_factory=list)
    recent_skips:    List[str]        = field(default_factory=list)
    session_topics:  List[str]        = field(default_factory=list)


# ── Context helpers ───────────────────────────────────────────────────────────
def get_time_of_day() -> str:
    hour = datetime.now().hour
    if 5  <= hour < 12: return "morning"
    elif 12 <= hour < 17: return "afternoon"
    elif 17 <= hour < 21: return "evening"
    else:                  return "night"


MOOD_WEIGHTS = {
    "stressed": {"entertainment": 1.5, "sports": 1.3, "politics": 0.4},
    "curious":  {"technology": 1.5, "science": 1.4, "business": 1.2},
    "tired":    {"entertainment": 1.4, "lifestyle": 1.3, "politics": 0.3},
    "happy":    {"sports": 1.2, "entertainment": 1.2, "technology": 1.2},
    "neutral":  {},
}

TIME_WEIGHTS = {
    "morning":   {"politics": 1.3, "business": 1.2, "technology": 1.1},
    "afternoon": {"sports": 1.2, "business": 1.2},
    "evening":   {"entertainment": 1.3, "lifestyle": 1.2},
    "night":     {"entertainment": 1.5, "lifestyle": 1.4, "politics": 0.5},
}


def compute_context_score(article_category: str, mood: str, time_of_day: str) -> float:
    category_key = str(article_category or "").strip().lower()
    score = 1.0
    score *= MOOD_WEIGHTS.get(mood, {}).get(category_key, 1.0)
    score *= TIME_WEIGHTS.get(time_of_day, {}).get(category_key, 1.0)
    return score


# ── Session store (Redis / fallback) ─────────────────────────────────────────
def update_user_session(user: UserProfile):
    """Persist short-term session data to Redis (or memory fallback) for 1 hour."""
    key = f"session:{user.user_id}"
    data = {
        "recent_clicks":  user.recent_clicks[-20:],
        "recent_skips":   user.recent_skips[-30:],
        "session_topics": user.session_topics[-20:],
        "mood":           user.mood,
    }
    _redis_set(key, json.dumps(data), ttl=3600)


def load_user_session(user_id: str) -> dict:
    """Load short-term session from Redis / fallback."""
    raw = _redis_get(f"session:{user_id}")
    if raw:
        return json.loads(raw)
    return {}


def clear_user_session(user_id: str):
    """Remove short-term session state for a user."""
    _redis_delete(f"session:{user_id}")
