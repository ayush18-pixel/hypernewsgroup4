"""
rag_pipeline.py — RAG retrieval with real Groq LLM explanation.
Falls back to a smart template if GROQ_API_KEY is not set.
"""
import os
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# ── Groq LLM (optional) ───────────────────────────────────────────────────────
_GROQ_KEY = os.getenv("GROQ_API_KEY", "")
_llm = None

if _GROQ_KEY:
    try:
        from langchain_groq import ChatGroq
        from langchain.schema import HumanMessage
        _llm = ChatGroq(model="llama-3.1-8b-instant", api_key=_GROQ_KEY, temperature=0.4)
    except Exception as e:
        print(f"⚠️  Groq init failed: {e}. Using fallback explanation.")


# ── FAISS helpers ─────────────────────────────────────────────────────────────
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    emb = embeddings.copy().astype("float32")
    faiss.normalize_L2(emb)
    index.add(emb)
    return index


def retrieve_articles(
    query: str,
    index: faiss.Index,
    df: pd.DataFrame,
    model: SentenceTransformer,
    top_k: int = 10,
) -> List[Dict]:
    query_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, top_k)
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(df):
            results.append(df.iloc[idx].to_dict())
    return results


# ── Personalized explanation ──────────────────────────────────────────────────
def generate_personalized_summary(
    user_context: dict,
    articles: List[dict],
    dummy_mode: bool = False,
) -> str:
    mood      = user_context.get("mood", "neutral")
    time_slot = user_context.get("time_of_day", "morning")
    categories = list({a["category"] for a in articles[:5]})

    # Try real Groq
    if _llm and not dummy_mode:
        try:
            article_lines = "\n".join(
                f"- {a['title']} ({a['category']})" for a in articles[:5]
            )
            prompt = (
                f"User context:\n"
                f"  Mood: {mood}\n"
                f"  Time of day: {time_slot}\n\n"
                f"Articles being recommended:\n{article_lines}\n\n"
                f"In exactly 2 engaging sentences, explain why these articles are "
                f"perfect for this user right now. Be specific about mood and time. "
                f"Don't start with 'Based on'."
            )
            response = _llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"⚠️  Groq call failed: {e}")

    # Fallback template
    cats_str = " and ".join(categories[:3]) if categories else "various topics"
    return (
        f"It's {time_slot} and you're feeling {mood} — "
        f"so we've curated a mix of {cats_str} to match your energy right now. "
        f"These {len(articles)} articles are ranked by your personal interest profile and today's context."
    )
