"""
FastAPI backend for HyperNews.
Loads persisted data, serves recommendations, and records feedback.
"""

import os
import sys
from typing import Dict, Iterable, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(_BACKEND_DIR, "..", ".env"))

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from bandit import LinUCBBandit
from db import delete_user, init_db, load_user, save_user, save_reading_vector
from graph import build_knowledge_graph, get_graph_stats
from mind_data import parse_entity_list
from rag_pipeline import build_faiss_index, generate_personalized_summary, retrieve_articles
from ranker import (
    _apply_interest_ema_decay,
    _build_news_id_to_idx,
    _graph_bonus_map,
    _normalize_key,
    build_candidate_pool,
    build_context_vector,
    cold_start_recommendations,
    get_kg_related_ids,
    rank_articles,
)
from user_profile import (
    UserProfile,
    clear_user_session,
    get_time_of_day,
    load_user_session,
    push_recent_history,
    update_user_session,
)

_BASE = os.path.join(_BACKEND_DIR, "..")
DATA_DIR = os.path.join(_BASE, "data")
GRAPH_DIR = os.path.join(_BASE, "graph")
DEFAULT_PARQUET = os.path.join(DATA_DIR, "articles.parquet")
DEFAULT_FAISS = os.path.join(DATA_DIR, "faiss_mind.index")
PARQUET_FILES = [
    DEFAULT_PARQUET,
    os.path.join(DATA_DIR, "news_processed.parquet"),
]
FAISS_FILES = [
    DEFAULT_FAISS,
    os.path.join(DATA_DIR, "news_faiss.index"),
]
EMB_FILE = os.path.join(DATA_DIR, "article_embeddings.npy")
BANDIT_FILE = os.path.join(_BASE, "models", "bandit_model.pkl")
GRAPH_ENABLED = os.getenv("HYPERNEWS_ENABLE_GRAPH", "1").strip().lower() not in {"0", "false", "no"}
GRAPH_ARTICLE_LIMIT = int(os.getenv("HYPERNEWS_GRAPH_ARTICLE_LIMIT", "0") or 0)
STARTUP_MAX_ARTICLES = int(os.getenv("HYPERNEWS_MAX_ARTICLES", "0") or 0)

# ── Exploration / feedback policy ─────────────────────────────────────────────
# Stay in semi-cold-start until this many confirmed positive interactions.
# Must match _INTEREST_WARMUP_INTERACTIONS in ranker.py (both = 5).
_SEMI_COLD_START_THRESHOLD = 5

# Interest nudge: fraction of reward added to category weight per event.
# click: 0.5 * 0.25 = 0.125; save: 2.0 * 0.25 = 0.5. Prevents instant lock-in.
_INTEREST_NUDGE_FACTOR = 0.25

# Skip: relative decay + fixed penalty on the category weight.
_SKIP_DECAY_FACTOR = 0.15
_SKIP_FIXED_PENALTY = 0.10

# Reward map used for bandit training (not for interest accumulation).
_REWARD_MAP = {"click": 0.5, "read_full": 1.0, "skip": -0.3, "save": 2.0}

# Decay applied to sibling articles in the same category on positive feedback.
_CATEGORY_PROPAGATION_DECAY = 0.3
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="HyperNews Recommendation API", version="2.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

USERS: Dict[str, UserProfile] = {}
BANDIT: Optional[LinUCBBandit] = None
DF: pd.DataFrame = pd.DataFrame()
EMBEDDINGS: np.ndarray = np.array([])
FAISS_INDEX: Optional[object] = None
KG_GRAPH: Optional[object] = None
MODEL: Optional[object] = None


class RecommendRequest(BaseModel):
    user_id: str
    mood: str = "neutral"
    n: int = 10
    query: Optional[str] = None


class FeedbackRequest(BaseModel):
    user_id: str
    article_id: str
    action: str
    dwell_time: float = 0.0


_SAFE_COLS = {"news_id", "category", "subcategory", "title", "abstract", "url", "popularity", "score", "source"}


def _to_python(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and value != value:
        return None
    return value


def sanitize_article(article: dict) -> dict:
    sanitized = {}
    for key, value in article.items():
        if key not in _SAFE_COLS:
            continue
        if isinstance(value, (list, dict)):
            continue
        sanitized[key] = _to_python(value)
    return sanitized


def _resolve_existing_path(candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _ensure_article_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy().reset_index(drop=True)
    if "subcategory" not in prepared.columns:
        prepared["subcategory"] = ""
    for column in ("entities", "title_entities", "abstract_entities"):
        if column in prepared.columns:
            prepared[column] = prepared[column].apply(parse_entity_list)
    for column in ("entity_ids", "entity_labels"):
        if column in prepared.columns:
            prepared[column] = prepared[column].apply(
                lambda value: value if isinstance(value, list) else ([] if value is None else [value] if isinstance(value, str) and value.strip() else [])
            )
    if "text" not in prepared.columns:
        title = prepared["title"].fillna("").astype(str) if "title" in prepared.columns else ""
        abstract = prepared["abstract"].fillna("").astype(str) if "abstract" in prepared.columns else ""
        prepared["text"] = title + ". " + abstract
    return prepared


def _normalize_embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _redis_status():
    from user_profile import _REDIS_OK
    return _REDIS_OK


def _graph_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not GRAPH_ARTICLE_LIMIT or len(df) <= GRAPH_ARTICLE_LIMIT:
        return df
    if "popularity" in df.columns:
        return df.sort_values("popularity", ascending=False).head(GRAPH_ARTICLE_LIMIT).reset_index(drop=True)
    return df.head(GRAPH_ARTICLE_LIMIT).reset_index(drop=True)


def _graph_cache_path_for_frame(graph_df: pd.DataFrame) -> str | None:
    if not STARTUP_MAX_ARTICLES and not GRAPH_ARTICLE_LIMIT:
        return None
    os.makedirs(GRAPH_DIR, exist_ok=True)
    return os.path.join(GRAPH_DIR, f"knowledge_graph.startup_{len(graph_df)}.pkl")


def _balanced_startup_indices(df: pd.DataFrame, limit: int) -> np.ndarray:
    if limit <= 0 or len(df) <= limit:
        return df.index.to_numpy()

    if "category" not in df.columns:
        return df.head(limit).index.to_numpy()

    working = df.copy()
    if "popularity" in working.columns:
        working = working.sort_values("popularity", ascending=False)

    grouped: dict[str, list[int]] = {}
    category_priority: list[tuple[str, float]] = []
    for category, group in working.groupby(working["category"].fillna("").astype(str).str.lower(), sort=False):
        rows = list(group.index)
        if not rows:
            continue
        grouped[category] = rows
        top_popularity = float(group["popularity"].iloc[0]) if "popularity" in group.columns else 0.0
        category_priority.append((category, top_popularity))

    ordered_categories = [category for category, _ in sorted(category_priority, key=lambda item: item[1], reverse=True)]
    if not ordered_categories:
        return working.head(limit).index.to_numpy()

    selected: list[int] = []
    cursors = {category: 0 for category in ordered_categories}

    while len(selected) < limit:
        progress = False
        for category in ordered_categories:
            rows = grouped[category]
            cursor = cursors[category]
            if cursor >= len(rows):
                continue
            selected.append(rows[cursor])
            cursors[category] += 1
            progress = True
            if len(selected) >= limit:
                break
        if not progress:
            break

    return np.asarray(selected, dtype=np.int64)


def _limit_loaded_assets(df: pd.DataFrame, embeddings: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    if not STARTUP_MAX_ARTICLES or len(df) <= STARTUP_MAX_ARTICLES:
        return df.reset_index(drop=True), embeddings

    selected_idx = _balanced_startup_indices(df, STARTUP_MAX_ARTICLES)

    limited_df = df.loc[selected_idx].reset_index(drop=True)
    limited_embeddings = np.asarray(embeddings[selected_idx], dtype=np.float32)
    category_mix = (
        limited_df["category"].fillna("").astype(str).str.lower().value_counts().head(6).to_dict()
        if "category" in limited_df.columns
        else {}
    )
    print(
        f"Trimmed startup assets to {len(limited_df):,} diverse articles via HYPERNEWS_MAX_ARTICLES. "
        f"Top categories: {category_mix}"
    )
    return limited_df, limited_embeddings


def get_or_create_user(user_id: str) -> UserProfile:
    if user_id not in USERS:
        stored = load_user(user_id)
        if stored:
            USERS[user_id] = UserProfile(
                user_id=user_id,
                interests=stored["interests"],
                reading_history=stored["reading_history"],
            )
        else:
            USERS[user_id] = UserProfile(user_id=user_id)

        session = load_user_session(user_id)
        if session:
            USERS[user_id].recent_clicks  = session.get("recent_clicks", [])
            USERS[user_id].recent_skips   = session.get("recent_skips", [])
            USERS[user_id].session_topics = session.get("session_topics", [])
            USERS[user_id].mood           = session.get("mood", "neutral")
            USERS[user_id].total_positive_interactions = session.get("total_positive_interactions", 0)
            USERS[user_id].interest_update_count       = session.get("interest_update_count", 0)

    return USERS[user_id]


@app.on_event("startup")
async def startup_event():
    global BANDIT, DF, EMBEDDINGS, FAISS_INDEX, KG_GRAPH, MODEL

    init_db()
    os.makedirs(os.path.join(_BASE, "models"), exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    parquet_path = _resolve_existing_path(PARQUET_FILES)
    faiss_path = _resolve_existing_path(FAISS_FILES)

    if not parquet_path or not os.path.exists(EMB_FILE):
        print(
            "Data files not found. Expected one of:\n"
            f"  parquet: {PARQUET_FILES}\n"
            f"  embeddings: {EMB_FILE}\n"
            f"  faiss: {FAISS_FILES}"
        )
        return

    try:
        DF = _ensure_article_frame(pd.read_parquet(parquet_path))
        EMBEDDINGS = _normalize_embedding_matrix(np.load(EMB_FILE).astype("float32"))
        if len(DF) != len(EMBEDDINGS):
            raise ValueError(
                f"Dataset and embedding count mismatch: {len(DF)} rows vs {len(EMBEDDINGS)} vectors"
            )
        DF, EMBEDDINGS = _limit_loaded_assets(DF, EMBEDDINGS)

        if faiss_path:
            FAISS_INDEX = faiss.read_index(faiss_path)
            if FAISS_INDEX.ntotal != len(DF):
                print(
                    f"FAISS size mismatch for {os.path.basename(faiss_path)} "
                    f"({FAISS_INDEX.ntotal} vs {len(DF)}). Rebuilding index (will use HNSW)."
                )
                FAISS_INDEX = None
            else:
                print(f"Loaded FAISS index from {os.path.basename(faiss_path)} ({FAISS_INDEX.ntotal} vectors)")

        if FAISS_INDEX is None:
            FAISS_INDEX = build_faiss_index(EMBEDDINGS)
            if STARTUP_MAX_ARTICLES:
                print(f"Built in-memory HNSW index ({FAISS_INDEX.ntotal} vectors) for capped startup mode")
            else:
                faiss.write_index(FAISS_INDEX, DEFAULT_FAISS)
                print(f"Built and saved HNSW index ({FAISS_INDEX.ntotal} vectors)")

        # context_dim = 384 embedding + 7 context scalars = 391
        # The 7th scalar is kg_score (knowledge graph affinity), so the bandit
        # learns jointly from RL rewards and graph structure.
        # Note: any existing bandit_model.pkl at a different dim will be recreated fresh.
        BANDIT = LinUCBBandit.load_or_create(
            BANDIT_FILE,
            context_dim=EMBEDDINGS.shape[1] + 7,
            epsilon=0.05,
        )

        if GRAPH_ENABLED:
            graph_df = _graph_frame(DF)
            KG_GRAPH = build_knowledge_graph(
                graph_df,
                cache_path=_graph_cache_path_for_frame(graph_df),
            )
        else:
            KG_GRAPH = None
        MODEL = None

        # Load Transformer encoder if saved weights exist
        try:
            from transformer_encoder import load_encoder, _MODEL_SAVE_PATH as _ENC_PATH
            enc_path = os.path.join(_BASE, _ENC_PATH)
            if os.path.exists(enc_path):
                load_encoder(enc_path)
                print(f"Transformer encoder loaded from {enc_path}")
            else:
                print("Transformer encoder: no saved weights found, using random init (improves over time)")
        except ImportError:
            pass

        print(
            f"Backend ready: {len(DF):,} articles | "
            f"KG nodes: {KG_GRAPH.number_of_nodes() if KG_GRAPH else 0} | "
            f"Embeddings: {EMBEDDINGS.shape} | "
            f"Bandit context_dim: {BANDIT.context_dim}"
        )
    except Exception as exc:
        print(f"Startup failed while loading recommendation assets: {exc}")
        DF = pd.DataFrame()
        EMBEDDINGS = np.array([])
        FAISS_INDEX = None
        KG_GRAPH = None
        BANDIT = None
        MODEL = None


@app.on_event("shutdown")
async def shutdown_event():
    if BANDIT:
        BANDIT.save(BANDIT_FILE)
        print("Bandit state saved to disk.")
    try:
        from transformer_encoder import save_encoder, _MODEL_SAVE_PATH as _ENC_PATH
        save_encoder(os.path.join(_BASE, _ENC_PATH))
        print("Transformer encoder saved.")
    except ImportError:
        pass


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "articles_loaded": len(DF),
        "kg_nodes": KG_GRAPH.number_of_nodes() if KG_GRAPH else 0,
        "users_active": len(USERS),
        "groq_enabled": bool(os.getenv("GROQ_API_KEY")),
        "redis_enabled": _redis_status(),
        "graph_enabled": GRAPH_ENABLED,
        "graph_article_limit": GRAPH_ARTICLE_LIMIT,
        "startup_max_articles": STARTUP_MAX_ARTICLES,
        "bandit_context_dim": BANDIT.context_dim if BANDIT else 0,
    }


@app.get("/articles")
async def get_articles(limit: int = 20):
    if len(DF) == 0:
        return {"articles": []}
    records = DF.sample(min(limit, len(DF))).to_dict("records")
    return {"articles": [sanitize_article(article) for article in records]}


@app.get("/graph")
async def graph_info():
    if KG_GRAPH is None:
        return {
            "error": "Graph not loaded",
            "graph_enabled": GRAPH_ENABLED,
            "graph_article_limit": GRAPH_ARTICLE_LIMIT,
            "startup_max_articles": STARTUP_MAX_ARTICLES,
        }
    return get_graph_stats(KG_GRAPH)


@app.get("/profile/{user_id}")
async def get_profile(user_id: str):
    user = get_or_create_user(user_id)
    return {
        "user_id": user.user_id,
        "mood": user.mood,
        "time_of_day": user.time_of_day,
        "interests": user.interests,
        "articles_read": len(user.reading_history),
        "recent_clicks": user.recent_clicks[-5:],
        "recent_skips": user.recent_skips[-5:],
        "session_topics": user.session_topics[-10:],
        "total_positive_interactions": user.total_positive_interactions,
    }


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    global MODEL

    if len(DF) == 0:
        return {"error": "Data not loaded. Run generate_data.py first."}

    user = get_or_create_user(req.user_id)
    user.mood = req.mood
    user.time_of_day = get_time_of_day()

    n_positive = user.total_positive_interactions
    use_cold_start = (
        n_positive < _SEMI_COLD_START_THRESHOLD
        and not user.recent_skips
        and not req.query
    )

    if use_cold_start:
        articles = cold_start_recommendations(user, DF, req.n)
        user._last_candidate_pool = []
        mode = "cold_start"
    elif req.query and FAISS_INDEX is not None:
        if MODEL is None:
            MODEL = SentenceTransformer("all-MiniLM-L6-v2")

        raw = retrieve_articles(
            req.query,
            FAISS_INDEX,
            DF,
            MODEL,
            top_k=max(req.n * 5, 24),
        )
        if len(raw) < req.n:
            fallback_pool = build_candidate_pool(
                user,
                DF,
                EMBEDDINGS,
                FAISS_INDEX,
                graph=KG_GRAPH,
                max_candidates=max(req.n * 5, 24),
            )
            seen_ids = {article["news_id"] for article in raw}
            raw.extend([article for article in fallback_pool if article["news_id"] not in seen_ids])

        user._last_candidate_pool = raw
        articles = rank_articles(user, raw, EMBEDDINGS, BANDIT, DF, KG_GRAPH, n=req.n)
        mode = "rag"
    else:
        candidates = build_candidate_pool(
            user,
            DF,
            EMBEDDINGS,
            FAISS_INDEX,
            graph=KG_GRAPH,
            max_candidates=min(max(req.n * 25, 120), len(DF)),
        )
        user._last_candidate_pool = candidates
        articles = rank_articles(user, candidates, EMBEDDINGS, BANDIT, DF, KG_GRAPH, n=req.n)
        if not articles:
            articles = cold_start_recommendations(user, DF, req.n)
        mode = "rl"

    explanation = generate_personalized_summary(
        {
            "mood": user.mood,
            "time_of_day": user.time_of_day,
            "mode": mode,
            "query": req.query,
            "recent_topics": user.session_topics[-5:],
        },
        articles,
    )

    update_user_session(user)

    return {
        "articles": [sanitize_article(article) for article in articles],
        "explanation": explanation,
        "user_id": req.user_id,
        "mode": mode,
    }


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    reward = _REWARD_MAP.get(req.action, 0.0)

    user = get_or_create_user(req.user_id)

    try:
        match = DF[DF["news_id"] == req.article_id]
        if len(match) == 0:
            return {"status": "article_not_found"}

        idx = int(match.index[0])
        article_embedding = EMBEDDINGS[idx]

        # Build kg_score for this article so the context vector fed to the bandit
        # includes graph affinity — RL and KG learn together.
        news_id_to_idx_fb = _build_news_id_to_idx(DF)
        graph_bonus_fb = _graph_bonus_map(user, KG_GRAPH)
        article_kg_score = float(graph_bonus_fb.get(req.article_id, 0.0))

        context_vector = build_context_vector(user, article_embedding, kg_score=article_kg_score)

        if BANDIT:
            # KG-augmented reward: if the article has strong graph affinity we reinforce
            # that signal by slightly boosting the reward the bandit trains on.
            # This teaches the bandit that "KG-connected + clicked" is especially good.
            kg_reward_boost = 0.15 * article_kg_score if reward > 0 else 0.0
            augmented_reward = float(np.clip(reward + kg_reward_boost, -1.0, 2.5))
            BANDIT.update(req.article_id, context_vector, augmented_reward)

            # Update category-level stats for all actions (incl. skip)
            category_str = str(match.iloc[0]["category"])
            BANDIT._update_category_stats(category_str, BANDIT._normalize_reward(augmented_reward))

            # Propagate positive reward to category siblings from last pool
            if reward > 0:
                last_pool = getattr(user, "_last_candidate_pool", [])
                sibling_ids = [
                    str(art.get("news_id"))
                    for art in last_pool
                    if str(art.get("news_id")) != req.article_id
                    and _normalize_key(str(art.get("category", ""))) == _normalize_key(category_str)
                    and art.get("news_id")
                ]
                if sibling_ids:
                    BANDIT.propagate_category_reward(
                        category_str,
                        sibling_ids,
                        reward,
                        decay=_CATEGORY_PROPAGATION_DECAY,
                    )

                # KG reward propagation: also update bandit stats for KG-linked articles.
                # This is the key RL+KG joint learning: clicking article A tells the bandit
                # that graph-neighbours of A are also likely to be interesting.
                if KG_GRAPH is not None:
                    kg_linked_ids = get_kg_related_ids(
                        req.article_id, KG_GRAPH, news_id_to_idx_fb, limit=20
                    )
                    if kg_linked_ids:
                        BANDIT.propagate_category_reward(
                            category_str,
                            kg_linked_ids,
                            reward,
                            decay=0.15,   # lighter decay than category siblings (0.3)
                        )

        category = match.iloc[0]["category"]

        if req.action in ("click", "read_full", "save"):
            if req.article_id not in user.reading_history:
                user.reading_history.append(req.article_id)
            user.recent_clicks.append(req.article_id)
            user.recent_skips = [nid for nid in user.recent_skips if nid != req.article_id]
            user.session_topics.append(category)

            # Soft nudge: fraction of reward only — prevents instant category lock-in
            nudge = abs(reward) * _INTEREST_NUDGE_FACTOR
            user.interests[category] = user.interests.get(category, 0.0) + nudge
            user.total_positive_interactions += 1

            # Push to Redis recent-history sorted set
            push_recent_history(req.user_id, req.article_id)

            # Persist embedding to long-term vector store (pgvector or SQLite blob)
            try:
                save_reading_vector(req.user_id, req.article_id, article_embedding, feedback_weight=nudge)
            except Exception as vec_err:
                print(f"Vector history write failed (non-fatal): {vec_err}")

        elif req.action == "skip":
            user.recent_skips.append(req.article_id)
            user.recent_skips = user.recent_skips[-30:]

            # Relative decay + fixed penalty so repeated skips genuinely suppress the category
            current = user.interests.get(category, 0.0)
            if current > 0:
                user.interests[category] = max(0.0, current * (1.0 - _SKIP_DECAY_FACTOR) - _SKIP_FIXED_PENALTY)

        # EMA decay: all interest weights decay by 5% on every feedback event
        _apply_interest_ema_decay(user)
        user.interest_update_count += 1

        save_user(user.user_id, user.interests, user.reading_history, len(user.reading_history))
        update_user_session(user)
    except Exception as exc:
        print(f"Error in /feedback: {exc}")

    return {"status": "updated", "reward": reward}


@app.post("/reset/{user_id}")
async def reset_user(user_id: str):
    USERS.pop(user_id, None)
    clear_user_session(user_id)
    delete_user(user_id)
    return {"status": "reset", "user_id": user_id}
