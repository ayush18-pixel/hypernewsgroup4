"""
graph.py — Knowledge Graph using vectorized pandas + NetworkX.
Persists the built graph to disk so it only builds once.
"""
import os
import pickle
from typing import List
import networkx as nx
import pandas as pd

_KG_PATH = os.path.join(os.path.dirname(__file__), "..", "graph", "knowledge_graph.pkl")

# ── spaCy (optional fallback) ─────────────────────────────────────────────────
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _SPACY_OK = True
except (OSError, ImportError):
    _SPACY_OK = False


def _extract_entities_spacy(text: str) -> List[str]:
    """spaCy NER fallback — only called when entities column is missing."""
    if _SPACY_OK:
        doc = _nlp(text)
        return [
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ in ("ORG", "PERSON", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART")
            and len(ent.text.strip()) > 2
        ]
    # Bare-minimum fallback: capitalised words
    return [w for w in text.split() if w.istitle() and len(w) > 3]


def build_knowledge_graph(df: pd.DataFrame, force_rebuild: bool = False) -> nx.DiGraph:
    """Build (or load from disk) the Knowledge Graph using vectorized operations."""
    os.makedirs(os.path.dirname(_KG_PATH), exist_ok=True)

    if not force_rebuild and os.path.exists(_KG_PATH):
        print("📂 Loading Knowledge Graph from disk...")
        with open(_KG_PATH, "rb") as f:
            return pickle.load(f)

    print(f"🔗 Building Knowledge Graph (vectorized, {len(df):,} articles)...")
    G = nx.DiGraph()

    # ── 1. Article nodes ─────────────────────────────────────────────────────
    for news_id, cat, title in zip(df["news_id"], df["category"], df["title"]):
        G.add_node(news_id, type="article", category=cat, title=title)

    # ── 2. Category → article edges (groupby, no iterrows) ───────────────────
    for cat, group in df.groupby("category"):
        G.add_node(cat, type="category")
        for nid in group["news_id"]:
            G.add_edge(cat, nid, relation="contains")
            G.add_edge(nid, cat, relation="belongs_to")

    # ── 3. Subcategory nodes & edges ─────────────────────────────────────────
    if "subcategory" in df.columns:
        for (cat, subcat), group in df.groupby(["category", "subcategory"]):
            if subcat:
                G.add_node(subcat, type="subcategory")
                G.add_edge(cat, subcat, relation="has_subcategory")
                for nid in group["news_id"]:
                    G.add_edge(nid, subcat, relation="in_subcategory")

    # ── 4. Entity nodes & edges (explode — no iterrows) ──────────────────────
    if "entities" in df.columns:
        ent_df = df[["news_id", "entities"]].copy()

        def _norm(val):
            """Normalise an entity cell to a flat list of strings."""
            if not isinstance(val, list):
                return []
            out = []
            for e in val:
                if isinstance(e, str) and e:
                    out.append(e)
                elif isinstance(e, dict):
                    lbl = e.get("Label") or e.get("label", "")
                    if lbl:
                        out.append(str(lbl))
            return out

        ent_df["entities"] = ent_df["entities"].apply(_norm)
        ent_df = ent_df.explode("entities").dropna(subset=["entities"])
        ent_df = ent_df[ent_df["entities"].astype(str).str.len() > 2]

        for ent_text in ent_df["entities"].unique():
            G.add_node(str(ent_text), type="ENTITY")
        for nid, ent_text in zip(ent_df["news_id"], ent_df["entities"]):
            G.add_edge(nid, str(ent_text), relation="mentions")
            G.add_edge(str(ent_text), nid, relation="mentioned_in")
    else:
        # Fallback: spaCy on title+abstract (slow — only if no entities column)
        print("⚠️  No 'entities' column — falling back to spaCy NER (slow)...")
        for _, row in df.iterrows():
            text = str(row["title"]) + " " + str(row.get("abstract", ""))
            for ent_text in _extract_entities_spacy(text):
                G.add_node(ent_text, type="ENTITY")
                G.add_edge(row["news_id"], ent_text, relation="mentions")
                G.add_edge(ent_text, row["news_id"], relation="mentioned_in")

    with open(_KG_PATH, "wb") as f:
        pickle.dump(G, f)
    print(f"✅ Knowledge Graph saved: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def get_related_articles(entity: str, G: nx.DiGraph) -> List[str]:
    """Articles connected to a given entity or category node."""
    if entity not in G:
        return []
    return [
        n for n in G.successors(entity)
        if G.nodes[n].get("type") == "article"
    ]


def get_graph_stats(G: nx.DiGraph) -> dict:
    """Return summary stats for the /graph API endpoint."""
    entity_nodes = [
        {"id": n, "type": G.nodes[n].get("type"), "connections": G.degree(n)}
        for n in G.nodes
        if G.nodes[n].get("type") not in ("article", "category", "subcategory")
    ]
    top_entities = sorted(entity_nodes, key=lambda x: x["connections"], reverse=True)[:30]
    categories   = [n for n in G.nodes if G.nodes[n].get("type") == "category"]
    return {
        "total_nodes":   G.number_of_nodes(),
        "total_edges":   G.number_of_edges(),
        "article_count": len([n for n in G.nodes if G.nodes[n].get("type") == "article"]),
        "entity_count":  len(entity_nodes),
        "top_entities":  top_entities,
        "categories":    categories,
    }
