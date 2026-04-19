"""
Offline evaluator for MIND-style ranking metrics.

It compares:
1. A naive baseline that ignores dataset entity metadata and uses simple title-case extraction.
2. The structured pipeline that uses the normalized entity fields and structured graph.

Metrics follow the common MIND setup:
- AUC
- MRR
- nDCG@5
- nDCG@10
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from backend.bandit import LinUCBBandit
from backend.graph import build_knowledge_graph, get_related_articles
from backend.mind_data import BEHAVIOR_COLUMNS, load_mind_news
from backend.ranker import build_context_vector, rank_articles
from backend.user_profile import UserProfile, compute_context_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HyperNews on MIND ranking metrics.")
    parser.add_argument(
        "--train-news",
        default=os.path.join(_BASE, "data", "mind_full", "MIND-small", "train", "news.tsv"),
    )
    parser.add_argument(
        "--train-behaviors",
        default=os.path.join(_BASE, "data", "mind_full", "MIND-small", "train", "behaviors.tsv"),
    )
    parser.add_argument(
        "--dev-news",
        default=os.path.join(_BASE, "data", "mind_full", "MIND-small", "dev", "news.tsv"),
    )
    parser.add_argument(
        "--dev-behaviors",
        default=os.path.join(_BASE, "data", "mind_full", "MIND-small", "dev", "behaviors.tsv"),
    )
    parser.add_argument("--limit-impressions", type=int, default=1000)
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    return parser.parse_args()


def _normalize_key(value: str) -> str:
    return str(value or "").strip().lower()


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return arr / norm


def _load_behaviors(path: str, limit_impressions: int) -> pd.DataFrame:
    behaviors = pd.read_csv(path, sep="\t", header=None, names=BEHAVIOR_COLUMNS)
    valid_rows = []
    for row in behaviors.itertuples(index=False):
        impressions = str(row.impressions or "").split()
        labels = [int(item.rsplit("-", 1)[1]) for item in impressions if "-" in item]
        if not labels or sum(labels) == 0 or sum(labels) == len(labels):
            continue
        valid_rows.append(row)
        if len(valid_rows) >= limit_impressions:
            break
    return pd.DataFrame(valid_rows, columns=BEHAVIOR_COLUMNS)


def _required_news_ids(behaviors: pd.DataFrame) -> set[str]:
    news_ids: set[str] = set()
    for row in behaviors.itertuples(index=False):
        history = str(row.history or "").split()
        news_ids.update(history)
        for item in str(row.impressions or "").split():
            if "-" not in item:
                continue
            news_ids.add(item.rsplit("-", 1)[0])
    return news_ids


def _embedding_cache_path(news_ids: list[str], model_name: str) -> str:
    os.makedirs(os.path.join(_BASE, "data", "eval_cache"), exist_ok=True)
    digest = hashlib.sha1(("|".join(news_ids) + "|" + model_name).encode("utf-8")).hexdigest()[:12]
    return os.path.join(_BASE, "data", "eval_cache", f"embeddings_{digest}.npy")


def _compute_embeddings(df: pd.DataFrame, model_name: str) -> np.ndarray:
    news_ids = df["news_id"].astype(str).tolist()
    cache_path = _embedding_cache_path(news_ids, model_name)
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        if len(embeddings) == len(df):
            return embeddings.astype("float32")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")
    np.save(cache_path, embeddings)
    return embeddings


def _time_of_day_from_timestamp(value: str) -> str:
    try:
        hour = datetime.strptime(str(value), "%m/%d/%Y %I:%M:%S %p").hour
    except ValueError:
        return "morning"
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def _build_user(row, news_lookup: dict[str, dict]) -> UserProfile:
    history_ids = [news_id for news_id in str(row.history or "").split() if news_id in news_lookup]
    user = UserProfile(user_id=str(row.user_id))
    user.reading_history = history_ids
    user.recent_clicks = history_ids[-10:]
    user.time_of_day = _time_of_day_from_timestamp(row.time)
    user.mood = "neutral"

    for offset, news_id in enumerate(reversed(history_ids)):
        article = news_lookup.get(news_id)
        if not article:
            continue
        category = article.get("category", "")
        if not category:
            continue
        user.interests[category] = user.interests.get(category, 0.0) + (1.0 / (1.0 + offset * 0.15))

    user.session_topics = [
        news_lookup[news_id].get("category", "")
        for news_id in history_ids[-20:]
        if news_id in news_lookup
    ]
    return user


def _extract_naive_entities(title: str, abstract: str) -> list[str]:
    text = f"{title} {abstract}".strip()
    return [token.strip(".,!?;:()[]\"'") for token in text.split() if token.istitle() and len(token) > 3]


def _build_naive_graph(df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    alias_map: dict[str, str] = {}

    def remember(alias: str, node_id: str):
        key = _normalize_key(alias)
        if key and key not in alias_map:
            alias_map[key] = node_id

    for row in df.itertuples(index=False):
        graph.add_node(row.news_id, type="article", news_id=row.news_id, label=row.title)
        remember(row.news_id, row.news_id)
        graph.add_node(row.category, type="category", label=row.category)
        graph.add_edge(row.news_id, row.category)
        remember(row.category, row.category)
        if row.subcategory:
            graph.add_node(row.subcategory, type="subcategory", label=row.subcategory)
            graph.add_edge(row.news_id, row.subcategory)
            graph.add_edge(row.category, row.subcategory)
            remember(row.subcategory, row.subcategory)
        for entity in _extract_naive_entities(row.title, row.abstract):
            graph.add_node(entity, type="ENTITY", label=entity)
            graph.add_edge(row.news_id, entity)
            remember(entity, entity)

    graph.graph["aliases"] = alias_map
    return graph


def _naive_related_articles(reference: str, graph: nx.Graph, limit: int | None = None) -> list[str]:
    alias_map = graph.graph.get("aliases", {})
    node_id = alias_map.get(_normalize_key(reference))
    if not node_id or node_id not in graph:
        return []

    related: Counter = Counter()
    if graph.nodes[node_id].get("type") == "article":
        for neighbor in graph.neighbors(node_id):
            for candidate in graph.neighbors(neighbor):
                if candidate == node_id:
                    continue
                if graph.nodes[candidate].get("type") == "article":
                    related[candidate] += 1
    else:
        for neighbor in graph.neighbors(node_id):
            if graph.nodes[neighbor].get("type") == "article":
                related[neighbor] += 1
    return [news_id for news_id, _ in related.most_common(limit)]


def _score_interest(user: UserProfile, category: str) -> float:
    weights = {_normalize_key(cat): max(float(weight), 0.0) for cat, weight in user.interests.items()}
    total = float(sum(weights.values()))
    if total <= 0.0:
        return 0.0
    return float(weights.get(_normalize_key(category), 0.0) / total)


def _build_user_profile_vector(user: UserProfile, embeddings: np.ndarray, news_id_to_idx: dict[str, int]) -> np.ndarray | None:
    weighted_ids = list(user.recent_clicks[-5:]) + list(user.recent_clicks[-5:]) + list(user.reading_history[-10:])
    vectors = [embeddings[news_id_to_idx[news_id]] for news_id in weighted_ids if news_id in news_id_to_idx]
    if not vectors:
        return None
    return _l2_normalize(np.mean(np.vstack(vectors), axis=0))


def _baseline_rank_articles(user: UserProfile, candidate_articles: list[dict], df: pd.DataFrame, embeddings: np.ndarray, graph: nx.Graph) -> list[dict]:
    news_id_to_idx = pd.Series(df.index, index=df["news_id"]).to_dict()
    profile_vector = _build_user_profile_vector(user, embeddings, news_id_to_idx)
    previously_seen = set(user.reading_history)

    related = set()
    for category in user.interests:
        related.update(_naive_related_articles(category, graph, limit=250))
    for news_id in user.reading_history[-5:]:
        related.update(_naive_related_articles(news_id, graph, limit=200))

    scored = []
    for article in candidate_articles:
        news_id = article.get("news_id")
        if news_id not in news_id_to_idx:
            continue

        article_embedding = _l2_normalize(embeddings[news_id_to_idx[news_id]])
        semantic_score = 0.0
        if profile_vector is not None:
            semantic_score = float((np.dot(profile_vector, article_embedding) + 1.0) / 2.0)

        interest_score = _score_interest(user, article.get("category", ""))
        popularity_score = float(np.clip(article.get("popularity", 0.0) or 0.0, 0.0, 1.0))
        context_multiplier = compute_context_score(article.get("category", ""), user.mood, user.time_of_day)
        kg_bonus = 0.15 if news_id in related else 0.0
        repeat_penalty = 0.35 if news_id in previously_seen else 0.0

        base_score = (0.30 * semantic_score) + (0.15 * interest_score) + (0.10 * popularity_score)
        final_score = (base_score * context_multiplier) + kg_bonus - repeat_penalty
        scored.append({**article, "score": float(final_score)})

    return sorted(scored, key=lambda item: item["score"], reverse=True)


def _auc(scores: list[float], labels: list[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return 0.0
    total = 0.0
    comparisons = 0
    for pos in positives:
        for neg in negatives:
            comparisons += 1
            if pos > neg:
                total += 1.0
            elif pos == neg:
                total += 0.5
    return total / comparisons if comparisons else 0.0


def _mrr(scores: list[float], labels: list[int]) -> float:
    ranking = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    for rank, (_, label) in enumerate(ranking, start=1):
        if label == 1:
            return 1.0 / rank
    return 0.0


def _ndcg(scores: list[float], labels: list[int], k: int) -> float:
    ranking = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)[:k]
    dcg = sum((label / np.log2(rank + 1)) for rank, (_, label) in enumerate(ranking, start=1))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum((label / np.log2(rank + 1)) for rank, label in enumerate(ideal, start=1))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def _evaluate_variant(
    name: str,
    behaviors: pd.DataFrame,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    graph,
    ranking_fn,
) -> dict[str, float]:
    news_lookup = {row["news_id"]: row for row in df.to_dict("records")}
    metrics = {"auc": [], "mrr": [], "ndcg5": [], "ndcg10": []}

    for row in behaviors.itertuples(index=False):
        user = _build_user(row, news_lookup)
        impression_pairs = [item.rsplit("-", 1) for item in str(row.impressions or "").split() if "-" in item]
        candidate_ids = [news_id for news_id, _ in impression_pairs if news_id in news_lookup]
        if len(candidate_ids) < 2:
            continue

        candidate_articles = [news_lookup[news_id] for news_id in candidate_ids]
        ranked = ranking_fn(user, candidate_articles, df, embeddings, graph)
        score_map = {article["news_id"]: float(article.get("score", 0.0)) for article in ranked}
        scores = [score_map.get(news_id, -1e9) for news_id, _ in impression_pairs if news_id in news_lookup]
        labels = [int(label) for news_id, label in impression_pairs if news_id in news_lookup]

        if sum(labels) == 0 or sum(labels) == len(labels):
            continue

        metrics["auc"].append(_auc(scores, labels))
        metrics["mrr"].append(_mrr(scores, labels))
        metrics["ndcg5"].append(_ndcg(scores, labels, 5))
        metrics["ndcg10"].append(_ndcg(scores, labels, 10))

    return {
        "variant": name,
        "impressions": len(metrics["auc"]),
        "auc": float(np.mean(metrics["auc"])) if metrics["auc"] else 0.0,
        "mrr": float(np.mean(metrics["mrr"])) if metrics["mrr"] else 0.0,
        "ndcg5": float(np.mean(metrics["ndcg5"])) if metrics["ndcg5"] else 0.0,
        "ndcg10": float(np.mean(metrics["ndcg10"])) if metrics["ndcg10"] else 0.0,
    }


def _improved_ranker(user: UserProfile, candidate_articles: list[dict], df: pd.DataFrame, embeddings: np.ndarray, graph) -> list[dict]:
    return rank_articles(user, candidate_articles, embeddings, None, df, graph)


def _evaluate_neural_bandit_variant(
    name: str,
    behaviors: pd.DataFrame,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    graph,
) -> dict[str, float]:
    news_lookup = {row["news_id"]: row for row in df.to_dict("records")}
    news_id_to_idx = pd.Series(df.index, index=df["news_id"]).to_dict()
    metrics = {"auc": [], "mrr": [], "ndcg5": [], "ndcg10": []}
    bandit = LinUCBBandit(context_dim=embeddings.shape[1] + 3, alpha=0.25)

    for row in behaviors.itertuples(index=False):
        user = _build_user(row, news_lookup)
        impression_pairs = [item.rsplit("-", 1) for item in str(row.impressions or "").split() if "-" in item]
        candidate_ids = [news_id for news_id, _ in impression_pairs if news_id in news_lookup]
        if len(candidate_ids) < 2:
            continue

        candidate_articles = [news_lookup[news_id] for news_id in candidate_ids]
        ranked = rank_articles(user, candidate_articles, embeddings, bandit, df, graph)
        score_map = {article["news_id"]: float(article.get("score", 0.0)) for article in ranked}
        scores = [score_map.get(news_id, -1e9) for news_id, _ in impression_pairs if news_id in news_lookup]
        labels = [int(label) for news_id, label in impression_pairs if news_id in news_lookup]

        if sum(labels) == 0 or sum(labels) == len(labels):
            continue

        metrics["auc"].append(_auc(scores, labels))
        metrics["mrr"].append(_mrr(scores, labels))
        metrics["ndcg5"].append(_ndcg(scores, labels, 5))
        metrics["ndcg10"].append(_ndcg(scores, labels, 10))

        # Learn online from all clicked items and a small set of high-ranked negatives.
        label_map = {news_id: int(label) for news_id, label in impression_pairs if news_id in news_lookup}
        ranked_negatives = [
            article["news_id"]
            for article in ranked
            if label_map.get(article["news_id"], 0) == 0
        ][:4]
        clicked = [news_id for news_id, label in label_map.items() if label == 1]
        update_ids = clicked + ranked_negatives
        update_contexts = [
            build_context_vector(user, embeddings[news_id_to_idx[news_id]])
            for news_id in update_ids
            if news_id in news_id_to_idx
        ]
        update_rewards = [
            1.0 if label_map.get(news_id, 0) == 1 else 0.0
            for news_id in update_ids
            if news_id in news_id_to_idx
        ]
        filtered_ids = [news_id for news_id in update_ids if news_id in news_id_to_idx]
        if filtered_ids:
            bandit.update_batch(filtered_ids, update_contexts, update_rewards)

    return {
        "variant": name,
        "impressions": len(metrics["auc"]),
        "auc": float(np.mean(metrics["auc"])) if metrics["auc"] else 0.0,
        "mrr": float(np.mean(metrics["mrr"])) if metrics["mrr"] else 0.0,
        "ndcg5": float(np.mean(metrics["ndcg5"])) if metrics["ndcg5"] else 0.0,
        "ndcg10": float(np.mean(metrics["ndcg10"])) if metrics["ndcg10"] else 0.0,
    }


def main():
    args = parse_args()
    if not os.path.exists(args.train_news) or not os.path.exists(args.train_behaviors) or not os.path.exists(args.dev_news) or not os.path.exists(args.dev_behaviors):
        raise FileNotFoundError("MIND train/dev files are required. Run the raw data download step first.")

    print("Loading MIND train/dev data...")
    full_df = load_mind_news(
        [args.train_news, args.dev_news],
        behavior_paths=[args.train_behaviors],
    )
    dev_behaviors = _load_behaviors(args.dev_behaviors, args.limit_impressions)
    required_ids = _required_news_ids(dev_behaviors)
    eval_df = full_df[full_df["news_id"].isin(required_ids)].reset_index(drop=True)
    print(f"Evaluation sample: {len(dev_behaviors):,} impressions | {len(eval_df):,} unique articles")

    embeddings = _compute_embeddings(eval_df, args.model_name)

    naive_df = eval_df.copy()
    naive_df["entities"] = [[] for _ in range(len(naive_df))]
    naive_df["entity_ids"] = [[] for _ in range(len(naive_df))]
    naive_graph = _build_naive_graph(naive_df)

    with tempfile.TemporaryDirectory() as tmp_dir:
        graph_path = os.path.join(tmp_dir, "kg.pkl")
        structured_graph = build_knowledge_graph(eval_df, force_rebuild=True, cache_path=graph_path)

        baseline = _evaluate_variant("baseline_naive_graph", dev_behaviors, naive_df, embeddings, naive_graph, _baseline_rank_articles)
        improved = _evaluate_variant("improved_structured_graph", dev_behaviors, eval_df, embeddings, structured_graph, _improved_ranker)
        neural_bandit = _evaluate_neural_bandit_variant(
            "improved_structured_graph_neural_bandit",
            dev_behaviors,
            eval_df,
            embeddings,
            structured_graph,
        )

    print("\nResults")
    for result in (baseline, improved, neural_bandit):
        print(
            f"{result['variant']}: impressions={result['impressions']:,} | "
            f"AUC={result['auc']:.4f} | MRR={result['mrr']:.4f} | "
            f"nDCG@5={result['ndcg5']:.4f} | nDCG@10={result['ndcg10']:.4f}"
        )

    print("\nDelta (improved - baseline)")
    print(f"AUC:     {improved['auc'] - baseline['auc']:+.4f}")
    print(f"MRR:     {improved['mrr'] - baseline['mrr']:+.4f}")
    print(f"nDCG@5:  {improved['ndcg5'] - baseline['ndcg5']:+.4f}")
    print(f"nDCG@10: {improved['ndcg10'] - baseline['ndcg10']:+.4f}")

    print("\nDelta (neural bandit - structured graph)")
    print(f"AUC:     {neural_bandit['auc'] - improved['auc']:+.4f}")
    print(f"MRR:     {neural_bandit['mrr'] - improved['mrr']:+.4f}")
    print(f"nDCG@5:  {neural_bandit['ndcg5'] - improved['ndcg5']:+.4f}")
    print(f"nDCG@10: {neural_bandit['ndcg10'] - improved['ndcg10']:+.4f}")


if __name__ == "__main__":
    main()
