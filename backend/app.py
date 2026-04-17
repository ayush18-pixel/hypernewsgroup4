"""
app.py — FastAPI backend for HyperNews.
Loads .env, boots from disk (FAISS index + KG + bandit), exposes REST API.
"""
import sys, os

# Fix tokenizer deadlock on Mac when running inside Uvicorn
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Path setup ────────────────────────────────────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# ── Load .env ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(_BACKEND_DIR, "..", ".env"))

# ── Imports ───────────────────────────────────────────────────────────────────
import faiss
import numpy as np
import pandas as pd
from typing import Optional, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from user_profile import UserProfile, get_time_of_day, update_user_session, load_user_session, compute_context_score
from bandit import LinUCBBandit
from ranker import rank_articles, cold_start_recommendations, build_context_vector
from rag_pipeline import generate_personalized_summary, retrieve_articles
from graph import build_knowledge_graph, get_related_articles, get_graph_stats
from db import init_db, save_user, load_user

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE       = os.path.join(_BACKEND_DIR, "..")
DATA_DIR    = os.path.join(_BASE, "data")
PARQUET     = os.path.join(DATA_DIR, "articles.parquet")      # full MIND dataset
EMB_FILE    = os.path.join(DATA_DIR, "article_embeddings.npy") # 65 k × 384
FAISS_FILE  = os.path.join(DATA_DIR, "faiss_mind.index")       # pre-built FAISS
BANDIT_FILE = os.path.join(_BASE, "models", "bandit_model.pkl")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="HyperNews Recommendation API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
USERS: Dict[str, UserProfile] = {}
BANDIT: Optional[LinUCBBandit]  = None
DF:            pd.DataFrame       = pd.DataFrame()
EMBEDDINGS:    np.ndarray         = np.array([])
FAISS_INDEX:   Optional[object]   = None
KG_GRAPH:      Optional[object]   = None
MODEL:         Optional[object]   = None

# ── Pydantic models ───────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    user_id: str
    mood: str = "neutral"
    n: int = 10
    query: Optional[str] = None

class FeedbackRequest(BaseModel):
    user_id: str
    article_id: str
    action: str          # click | read_full | skip | save
    dwell_time: float = 0.0   # seconds spent on article

# ── Serialization helper ─────────────────────────────────────────────────────
_SAFE_COLS = {"news_id", "category", "subcategory", "title", "abstract", "url", "popularity", "score", "source"}

def _to_python(v):
    """Convert any numpy / pandas scalar to a native Python type."""
    if v is None:
        return None
    # numpy scalars expose item()
    if hasattr(v, "item"):
        return v.item()
    # pandas Timestamp / datetime
    if hasattr(v, "isoformat"):
        return v.isoformat()
    # NaN float
    if isinstance(v, float) and v != v:
        return None
    return v

def sanitize_article(article: dict) -> dict:
    """Keep only JSON-safe scalar columns and convert all values to native Python types."""
    out = {}
    for k, v in article.items():
        if k not in _SAFE_COLS:
            continue
        if isinstance(v, (list, dict)):
            continue
        out[k] = _to_python(v)
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_or_create_user(user_id: str) -> UserProfile:
    if user_id not in USERS:
        # Try loading from SQLite first
        stored = load_user(user_id)
        if stored:
            USERS[user_id] = UserProfile(
                user_id         = user_id,
                interests       = stored["interests"],
                reading_history = stored["reading_history"],
            )
        else:
            USERS[user_id] = UserProfile(user_id=user_id)
        # Merge short-term session from Redis / fallback
        session = load_user_session(user_id)
        if session:
            USERS[user_id].recent_clicks   = session.get("recent_clicks", [])
            USERS[user_id].session_topics  = session.get("session_topics", [])
            USERS[user_id].mood            = session.get("mood", "neutral")
    return USERS[user_id]

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global DF, EMBEDDINGS, FAISS_INDEX, KG_GRAPH, MODEL, BANDIT

    init_db()
    os.makedirs(os.path.join(_BASE, "models"), exist_ok=True)

    if os.path.exists(PARQUET) and os.path.exists(EMB_FILE):
        DF         = pd.read_parquet(PARQUET)
        EMBEDDINGS = np.load(EMB_FILE).astype("float32")

        # Load the pre-built MIND FAISS index directly (already normalized)
        if os.path.exists(FAISS_FILE):
            FAISS_INDEX = faiss.read_index(FAISS_FILE)
            print(f"📂 Loaded FAISS index ({FAISS_INDEX.ntotal} vectors)")
        else:
            # Fallback: build from embeddings if index is missing
            print("⚠️  faiss_mind.index not found — building from embeddings...")
            dimension   = EMBEDDINGS.shape[1]
            FAISS_INDEX = faiss.IndexFlatIP(dimension)
            emb_copy    = EMBEDDINGS.copy()
            faiss.normalize_L2(emb_copy)
            FAISS_INDEX.add(emb_copy)
            faiss.write_index(FAISS_INDEX, FAISS_FILE)
            print(f"✅ Built and saved FAISS index ({FAISS_INDEX.ntotal} vectors)")

        BANDIT   = LinUCBBandit.load_or_create(BANDIT_FILE, context_dim=387)
        KG_GRAPH = build_knowledge_graph(DF)

        print(f"✅ Backend ready: {len(DF):,} articles | KG: {KG_GRAPH.number_of_nodes()} nodes | Embeddings: {EMBEDDINGS.shape}")
    else:
        print(f"⚠️  Data files not found. Expected:\n   {PARQUET}\n   {EMB_FILE}\n   {FAISS_FILE}")

# ── Shutdown: persist bandit ──────────────────────────────────────────────────
@app.on_event("shutdown")
async def shutdown_event():
    if BANDIT:
        BANDIT.save(BANDIT_FILE)
        print("💾 Bandit state saved to disk.")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "articles_loaded": len(DF),
        "kg_nodes":        KG_GRAPH.number_of_nodes() if KG_GRAPH else 0,
        "users_active":    len(USERS),
        "groq_enabled":    bool(os.getenv("GROQ_API_KEY")),
        "redis_enabled":   _redis_status(),
    }

def _redis_status():
    from user_profile import _REDIS_OK
    return _REDIS_OK

@app.get("/articles")
async def get_articles(limit: int = 20):
    if len(DF) == 0:
        return {"articles": []}
    records = DF.sample(min(limit, len(DF))).to_dict("records")
    return {"articles": [sanitize_article(a) for a in records]}

@app.get("/graph")
async def graph_info():
    if KG_GRAPH is None:
        return {"error": "Graph not loaded"}
    return get_graph_stats(KG_GRAPH)

@app.get("/profile/{user_id}")
async def get_profile(user_id: str):
    user = get_or_create_user(user_id)
    return {
        "user_id":        user.user_id,
        "mood":           user.mood,
        "time_of_day":    user.time_of_day,
        "interests":      user.interests,
        "articles_read":  len(user.reading_history),
        "recent_clicks":  user.recent_clicks[-5:],
        "session_topics": user.session_topics[-10:],
    }

@app.post("/recommend")
async def recommend(req: RecommendRequest):
    global MODEL
    if len(DF) == 0:
        return {"error": "Data not loaded. Run generate_data.py first."}

    user            = get_or_create_user(req.user_id)
    user.mood       = req.mood
    user.time_of_day = get_time_of_day()

    if not user.reading_history and not req.query:
        articles = cold_start_recommendations(user, DF, req.n)
        mode     = "cold_start"
    elif req.query and FAISS_INDEX is not None:
        if MODEL is None:
            # Lazy load the semantic model
            MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        raw      = retrieve_articles(req.query, FAISS_INDEX, DF, MODEL, top_k=20)
        articles = rank_articles(user, raw, EMBEDDINGS, BANDIT, DF, KG_GRAPH)[:req.n]
        mode     = "rag"
    else:
        candidates = DF.sample(min(100, len(DF))).to_dict("records")
        articles   = rank_articles(user, candidates, EMBEDDINGS, BANDIT, DF, KG_GRAPH)[:req.n]
        mode       = "rl"

    explanation = generate_personalized_summary(
        {"mood": user.mood, "time_of_day": user.time_of_day},
        articles,
    )

    update_user_session(user)

    return {
        "articles":    [sanitize_article(a) for a in articles],
        "explanation": explanation,
        "user_id":     req.user_id,
        "mode":        mode,
    }

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    reward_map = {"click": 0.5, "read_full": 1.0, "skip": -0.2, "save": 2.0}
    reward     = reward_map.get(req.action, 0)

    user = get_or_create_user(req.user_id)

    try:
        match = DF[DF["news_id"] == req.article_id]
        if len(match) == 0:
            return {"status": "article_not_found"}
        idx  = match.index[0]
        emb  = EMBEDDINGS[idx]
        ctx  = build_context_vector(user, emb)
        if BANDIT:
            BANDIT.update(req.article_id, ctx, reward)

        if req.action in ("click", "read_full", "save"):
            if req.article_id not in user.reading_history:
                user.reading_history.append(req.article_id)
            user.recent_clicks.append(req.article_id)
            cat = match.iloc[0]["category"]
            user.interests[cat] = user.interests.get(cat, 0.0) + reward
            user.session_topics.append(cat)

        # Persist to SQLite + Redis
        save_user(user.user_id, user.interests, user.reading_history, len(user.reading_history))
        update_user_session(user)

    except Exception as e:
        print(f"Error in /feedback: {e}")

    return {"status": "updated", "reward": reward}

@app.post("/reset/{user_id}")
async def reset_user(user_id: str):
    if user_id in USERS:
        del USERS[user_id]
    return {"status": "reset", "user_id": user_id}
