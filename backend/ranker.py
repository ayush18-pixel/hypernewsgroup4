import numpy as np
import pandas as pd
from collections import Counter

try:
    from backend.graph import get_article_entities, get_related_articles
    from backend.mind_data import parse_entity_list
    from backend.user_profile import compute_context_score
except ImportError:
    from graph import get_article_entities, get_related_articles
    from mind_data import parse_entity_list
    from user_profile import compute_context_score


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return arr / norm


def _normalize_key(value: str) -> str:
    return str(value or "").strip().lower()


def _build_news_id_to_idx(df: pd.DataFrame) -> dict:
    return pd.Series(df.index, index=df["news_id"]).to_dict()


def _extract_entity_keys(value) -> list[str]:
    if isinstance(value, list) and value and isinstance(value[0], str):
        return [str(item).strip() for item in value if str(item).strip()]

    entities = parse_entity_list(value)
    keys = []
    for entity in entities:
        key = entity.get("wikidata_id") or entity.get("id") or entity.get("label")
        if key:
            keys.append(str(key).strip())
    return keys


def build_user_profile_vector(user, article_embeddings: np.ndarray, news_id_to_idx: dict) -> np.ndarray | None:
    weighted_ids = list(user.recent_clicks[-8:]) + list(user.recent_clicks[-8:]) + list(user.reading_history[-16:])
    vectors = []
    for news_id in weighted_ids:
        idx = news_id_to_idx.get(news_id)
        if idx is None:
            continue
        vectors.append(article_embeddings[idx])
    if not vectors:
        return None
    return _l2_normalize(np.mean(np.vstack(vectors), axis=0))


def _score_interest(user, category: str) -> float:
    positive_weights = {
        _normalize_key(cat): max(float(weight), 0.0)
        for cat, weight in user.interests.items()
    }
    total_weight = sum(positive_weights.values())
    if total_weight <= 0.0:
        return 0.0
    return positive_weights.get(_normalize_key(category), 0.0) / total_weight


def _score_popularity(article: dict) -> float:
    popularity = article.get("popularity")
    if popularity is None:
        return 0.0
    try:
        return float(np.clip(float(popularity), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def build_context_vector(user, article_embedding: np.ndarray) -> np.ndarray:
    mood_map = {"neutral": 0, "happy": 1, "curious": 2, "stressed": 3, "tired": 4}
    time_map = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}

    context_signal = np.array(
        [
            mood_map.get(user.mood, 0) / 4.0,
            time_map.get(user.time_of_day, 0) / 3.0,
            len(user.recent_clicks) / 10.0,
        ],
        dtype=np.float32,
    )

    article_vec = _l2_normalize(article_embedding.astype(np.float32))
    context_vec = np.concatenate([article_vec, context_signal])
    return _l2_normalize(context_vec)


def _build_history_profiles(user, df: pd.DataFrame, news_id_to_idx: dict, graph=None) -> tuple[Counter, Counter]:
    subcategory_weights: Counter = Counter()
    entity_weights: Counter = Counter()

    history_ids = list(user.recent_clicks[-8:]) + list(user.reading_history[-24:])
    weighted_history = list(enumerate(reversed(history_ids)))
    for offset, news_id in weighted_history:
        idx = news_id_to_idx.get(news_id)
        if idx is None:
            continue

        weight = 1.0 / (1.0 + (offset * 0.2))
        row = df.iloc[idx]
        subcategory = _normalize_key(row.get("subcategory", ""))
        if subcategory:
            subcategory_weights[subcategory] += weight

        entity_keys = _extract_entity_keys(row.get("entity_ids") or row.get("entities"))
        if not entity_keys and graph is not None:
            entity_keys = get_article_entities(news_id, graph)
        for entity_key in entity_keys:
            entity_weights[_normalize_key(entity_key)] += weight

    return subcategory_weights, entity_weights


def _build_skip_profiles(user, df: pd.DataFrame, news_id_to_idx: dict, graph=None) -> tuple[Counter, Counter]:
    category_skips: Counter = Counter()
    entity_skips: Counter = Counter()

    skipped_ids = list(getattr(user, "recent_skips", [])[-20:])
    weighted_skips = list(enumerate(reversed(skipped_ids)))
    for offset, news_id in weighted_skips:
        idx = news_id_to_idx.get(news_id)
        if idx is None:
            continue

        weight = 1.0 / (1.0 + (offset * 0.3))
        row = df.iloc[idx]
        category = _normalize_key(row.get("category", ""))
        if category:
            category_skips[category] += weight

        entity_keys = _extract_entity_keys(row.get("entity_ids") or row.get("entities"))
        if not entity_keys and graph is not None:
            entity_keys = get_article_entities(news_id, graph)
        for entity_key in entity_keys:
            entity_skips[_normalize_key(entity_key)] += weight

    return category_skips, entity_skips


def _normalize_counter_score(counter: Counter, key: str) -> float:
    if not counter:
        return 0.0
    total = float(sum(counter.values()))
    if total <= 0.0:
        return 0.0
    return float(counter.get(_normalize_key(key), 0.0) / total)


def _graph_bonus_map(user, graph) -> dict[str, float]:
    if graph is None:
        return {}

    scores: Counter = Counter()
    for category, weight in sorted(user.interests.items(), key=lambda item: item[1], reverse=True)[:5]:
        if float(weight) <= 0:
            continue
        for news_id in get_related_articles(category, graph, limit=250):
            scores[news_id] += float(weight)

    for offset, news_id in enumerate(reversed(user.reading_history[-8:])):
        article_weight = 1.0 / (1.0 + offset)
        for related_id in get_related_articles(news_id, graph, limit=200):
            scores[related_id] += article_weight

    if not scores:
        return {}

    max_score = max(scores.values())
    return {news_id: min(value / max_score, 1.0) for news_id, value in scores.items()}


def build_candidate_pool(
    user,
    df: pd.DataFrame,
    article_embeddings: np.ndarray,
    faiss_index=None,
    graph=None,
    max_candidates: int = 200,
) -> list[dict]:
    if len(df) == 0:
        return []

    news_id_to_idx = _build_news_id_to_idx(df)
    seen_ids: set[str] = set()
    read_ids = set(user.reading_history)
    skipped_ids = set(getattr(user, "recent_skips", [])[-30:])
    category_skips, _ = _build_skip_profiles(user, df, news_id_to_idx, graph=graph)
    disliked_categories = {
        category
        for category, weight in category_skips.items()
        if weight >= 1.5
    }
    candidates: list[dict] = []

    def add_records(records: list[dict]):
        for article in records:
            news_id = article.get("news_id")
            if not news_id or news_id in seen_ids or news_id in read_ids or news_id in skipped_ids:
                continue
            if _normalize_key(article.get("category", "")) in disliked_categories:
                continue
            seen_ids.add(news_id)
            candidates.append(article)
            if len(candidates) >= max_candidates:
                break

    profile_vector = build_user_profile_vector(user, article_embeddings, news_id_to_idx)
    if profile_vector is not None and faiss_index is not None:
        search_k = min(len(df), max(max_candidates * 3, 80))
        query = np.expand_dims(profile_vector.astype(np.float32), axis=0)
        _, indices = faiss_index.search(query, search_k)
        semantic_records = [
            df.iloc[int(idx)].to_dict()
            for idx in indices[0]
            if 0 <= int(idx) < len(df)
        ]
        add_records(semantic_records)

    if len(candidates) < max_candidates and graph is not None:
        graph_news_ids = []
        for history_id in user.reading_history[-5:]:
            graph_news_ids.extend(get_related_articles(history_id, graph, limit=80))
        for category, _ in sorted(user.interests.items(), key=lambda item: item[1], reverse=True)[:3]:
            graph_news_ids.extend(get_related_articles(category, graph, limit=60))
        graph_records = [
            df.iloc[news_id_to_idx[news_id]].to_dict()
            for news_id in graph_news_ids
            if news_id in news_id_to_idx
        ]
        add_records(graph_records)

    if len(candidates) < max_candidates and user.interests:
        preferred_categories = [
            cat
            for cat, weight in sorted(user.interests.items(), key=lambda item: item[1], reverse=True)
            if weight > 0 and _normalize_key(cat) not in disliked_categories
        ]
        for category in preferred_categories[:4]:
            cat_rows = df[df["category"].fillna("").str.lower() == _normalize_key(category)]
            if "popularity" in cat_rows.columns:
                cat_rows = cat_rows.sort_values("popularity", ascending=False)
            add_records(cat_rows.to_dict("records"))
            if len(candidates) >= max_candidates:
                break

    if len(candidates) < max_candidates:
        unread = df[~df["news_id"].isin(read_ids | skipped_ids)]
        if disliked_categories:
            unread = unread[~unread["category"].fillna("").str.lower().isin(disliked_categories)]
        if "popularity" in unread.columns:
            unread = unread.sort_values("popularity", ascending=False)
        add_records(unread.head(max_candidates).to_dict("records"))

    if len(candidates) < max_candidates:
        unread = df[~df["news_id"].isin(read_ids | skipped_ids | seen_ids)]
        if disliked_categories:
            unread = unread[~unread["category"].fillna("").str.lower().isin(disliked_categories)]
        sample_size = min(max_candidates - len(candidates), len(unread))
        if sample_size > 0:
            add_records(unread.sample(sample_size, random_state=42).to_dict("records"))

    if not candidates:
        return df.head(max_candidates).to_dict("records")

    return candidates[:max_candidates]


def rank_articles(user, candidate_articles: list, article_embeddings: np.ndarray, bandit, df: pd.DataFrame, G=None) -> list:
    scored = []
    news_id_to_idx = _build_news_id_to_idx(df)
    profile_vector = build_user_profile_vector(user, article_embeddings, news_id_to_idx)
    previously_seen = set(user.reading_history)
    subcategory_weights, entity_weights = _build_history_profiles(user, df, news_id_to_idx, graph=G)
    skipped_categories, skipped_entities = _build_skip_profiles(user, df, news_id_to_idx, graph=G)
    skipped_ids = set(getattr(user, "recent_skips", [])[-30:])
    graph_bonus = _graph_bonus_map(user, G)

    for article in candidate_articles:
        news_id = article.get("news_id")
        if not news_id or news_id not in news_id_to_idx:
            continue

        idx = news_id_to_idx[news_id]
        article_embedding = _l2_normalize(article_embeddings[idx])
        context_vector = build_context_vector(user, article_embedding)

        rl_score = float(bandit.score(news_id, context_vector)) if bandit else 0.0
        semantic_score = 0.0
        if profile_vector is not None:
            semantic_score = float((np.dot(profile_vector, article_embedding) + 1.0) / 2.0)

        interest_score = _score_interest(user, article.get("category", ""))
        subcategory_score = _normalize_counter_score(subcategory_weights, article.get("subcategory", ""))
        candidate_entities = _extract_entity_keys(article.get("entity_ids") or article.get("entities"))
        entity_score = 0.0
        if candidate_entities:
            entity_score = max(
                (_normalize_counter_score(entity_weights, entity_key) for entity_key in candidate_entities),
                default=0.0,
            )
        popularity_score = _score_popularity(article)
        context_multiplier = compute_context_score(article.get("category", ""), user.mood, user.time_of_day)
        kg_bonus = 0.20 * float(graph_bonus.get(news_id, 0.0))
        repeat_penalty = 0.40 if news_id in previously_seen else 0.0
        skipped_category_penalty = 0.18 * _normalize_counter_score(skipped_categories, article.get("category", ""))
        skipped_entity_penalty = 0.08 * max(
            (_normalize_counter_score(skipped_entities, entity_key) for entity_key in candidate_entities),
            default=0.0,
        )
        skipped_article_penalty = 0.45 if news_id in skipped_ids else 0.0

        base_score = (
            (0.30 * rl_score)
            + (0.25 * semantic_score)
            + (0.15 * interest_score)
            + (0.10 * subcategory_score)
            + (0.12 * entity_score)
            + (0.08 * popularity_score)
        )
        final_score = (
            (base_score * context_multiplier)
            + kg_bonus
            - repeat_penalty
            - skipped_category_penalty
            - skipped_entity_penalty
            - skipped_article_penalty
        )
        scored.append(
            {
                **article,
                "score": float(final_score),
                "rl_score": float(rl_score),
                "semantic_alignment": float(semantic_score),
                "entity_alignment": float(entity_score),
            }
        )

    return sorted(scored, key=lambda x: x["score"], reverse=True)


def cold_start_recommendations(user, df: pd.DataFrame, n: int = 10) -> list:
    time_category_map = {
        "morning": ["news", "politics", "business", "finance", "technology"],
        "afternoon": ["sports", "technology", "business", "news"],
        "evening": ["entertainment", "lifestyle", "sports", "health"],
        "night": ["entertainment", "lifestyle", "health", "movies"],
    }
    preferred_cats = time_category_map.get(user.time_of_day, ["news"])

    avoid_cats = {
        "stressed": ["politics", "health"],
        "tired": ["politics", "business", "finance"],
    }.get(user.mood, [])

    mood_boosts = {
        "curious": ["technology", "science", "business", "news"],
        "happy": ["sports", "entertainment", "lifestyle"],
        "stressed": ["entertainment", "lifestyle", "sports"],
        "tired": ["entertainment", "lifestyle", "health"],
    }.get(user.mood, [])

    news_id_to_idx = _build_news_id_to_idx(df)
    category_skips, _ = _build_skip_profiles(user, df, news_id_to_idx)
    disliked_categories = {
        category
        for category, weight in category_skips.items()
        if weight >= 1.5
    }
    skipped_ids = set(getattr(user, "recent_skips", [])[-30:])

    cat_lower = df["category"].fillna("").str.lower()
    filtered = df[
        ~cat_lower.isin(avoid_cats)
        & ~cat_lower.isin(disliked_categories)
        & ~df["news_id"].isin(skipped_ids)
    ]

    if len(filtered) == 0:
        filtered = df

    if "popularity" in filtered.columns:
        filtered = filtered.sort_values("popularity", ascending=False)

    available_categories = [
        category
        for category in filtered["category"].fillna("").astype(str).str.lower().unique().tolist()
        if category and category not in disliked_categories
    ]
    category_priority = sorted(
        available_categories,
        key=lambda category: (
            compute_context_score(category, user.mood, user.time_of_day)
            + (0.35 if category in preferred_cats else 0.0)
            + (0.45 if category in mood_boosts else 0.0)
        ),
        reverse=True,
    )

    category_buckets = {
        category: filtered[filtered["category"].fillna("").str.lower() == category].to_dict("records")
        for category in category_priority
    }

    picks: list[dict] = []
    used_ids: set[str] = set()
    while len(picks) < n:
        progressed = False
        for category in category_priority:
            bucket = category_buckets.get(category, [])
            while bucket and bucket[0].get("news_id") in used_ids:
                bucket.pop(0)
            if not bucket:
                continue
            article = bucket.pop(0)
            news_id = article.get("news_id")
            if not news_id or news_id in used_ids:
                continue
            used_ids.add(news_id)
            picks.append(article)
            progressed = True
            if len(picks) >= n:
                break
        if not progressed:
            break

    if len(picks) < n:
        extra = filtered[~filtered["news_id"].isin(used_ids)].head(n - len(picks)).to_dict("records")
        picks.extend(extra)

    return picks[:n]
