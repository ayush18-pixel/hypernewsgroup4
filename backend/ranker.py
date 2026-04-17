import pandas as pd
import numpy as np

try:
    from backend.user_profile import compute_context_score
    from backend.graph import get_related_articles
except ImportError:
    from user_profile import compute_context_score
    from graph import get_related_articles


def build_context_vector(user, article_embedding: np.ndarray) -> np.ndarray:
    mood_map = {"neutral": 0, "happy": 1, "curious": 2, "stressed": 3, "tired": 4}
    time_map = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}

    context_signal = np.array([
        mood_map.get(user.mood, 0) / 4.0,
        time_map.get(user.time_of_day, 0) / 3.0,
        len(user.recent_clicks) / 10.0,
    ])

    return np.concatenate([article_embedding[:384], context_signal])


def rank_articles(user, candidate_articles: list, article_embeddings: np.ndarray, bandit, df: pd.DataFrame, G=None) -> list:
    scored = []

    # Vectorised lookup: news_id → positional row index in embeddings array
    # df.index is the positional index (0..N-1) matching embeddings rows
    news_id_to_idx = pd.Series(df.index, index=df["news_id"]).to_dict()

    # Get KG-boosted article ids from user's interests
    related = set()
    if G is not None:
        for interest in user.interests:
            related.update(get_related_articles(interest, G))
        for art_id in user.reading_history[-5:]:
            related.update(get_related_articles(art_id, G))

    for article in candidate_articles:
        nid = article["news_id"]
        if nid not in news_id_to_idx:
            continue

        idx = news_id_to_idx[nid]
        emb = article_embeddings[idx]
        ctx = build_context_vector(user, emb)

        # UCB score from RL bandit
        rl_score = bandit.score(nid, ctx)

        # Mood + time-of-day multiplier
        ctx_multiplier = compute_context_score(article["category"], user.mood, user.time_of_day)

        # Knowledge Graph relatedness boost
        kg_boost = 1.2 if nid in related else 1.0

        final_score = rl_score * ctx_multiplier * kg_boost
        scored.append({**article, "score": float(final_score)})

    return sorted(scored, key=lambda x: x["score"], reverse=True)


def cold_start_recommendations(user, df: pd.DataFrame, n: int = 10) -> list:
    time_category_map = {
        "morning":   ["politics", "business", "finance", "technology"],
        "afternoon": ["sports", "technology", "business"],
        "evening":   ["entertainment", "lifestyle", "sports", "health"],
        "night":     ["entertainment", "lifestyle", "autos"],
    }
    preferred_cats = time_category_map.get(user.time_of_day, ["news"])

    avoid_cats = {
        "stressed": ["politics", "health"],
        "tired":    ["politics", "business", "finance"],
    }.get(user.mood, [])

    cat_lower = df["category"].str.lower()
    filtered = df[cat_lower.isin(preferred_cats) & ~cat_lower.isin(avoid_cats)]

    if len(filtered) == 0:
        filtered = df

    # Sort by popularity if the column exists (MIND dataset provides this)
    if "popularity" in filtered.columns:
        filtered = filtered.sort_values("popularity", ascending=False)

    if len(filtered) < n:
        extra = df[~df["news_id"].isin(filtered["news_id"])]
        if "popularity" in extra.columns:
            extra = extra.sort_values("popularity", ascending=False)
        filtered = pd.concat([filtered, extra])

    return filtered.head(n).to_dict("records")
