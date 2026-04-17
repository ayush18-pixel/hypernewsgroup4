"""
bandit.py — LinUCB Contextual Bandit
Handles RL scoring and weight updates from user click/skip feedback.
"""
import numpy as np
import pickle, os


class LinUCBBandit:
    """
    LinUCB Contextual Bandit.
    Each article gets its own A matrix and b vector.
    context_dim should match the size of the context vector you pass in.
    """

    def __init__(self, context_dim: int = 387, alpha: float = 0.5):
        self.context_dim = context_dim
        self.alpha = alpha
        self.A: dict[str, np.ndarray] = {}   # article_id → (d×d) matrix
        self.b: dict[str, np.ndarray] = {}   # article_id → (d,) vector

    # ── internal helpers ─────────────────────────────────────────────────────

    def _ensure(self, article_id: str):
        if article_id not in self.A:
            self.A[article_id] = np.eye(self.context_dim, dtype=np.float32)
            self.b[article_id] = np.zeros(self.context_dim, dtype=np.float32)

    # ── public API ───────────────────────────────────────────────────────────

    def score(self, article_id: str, context: np.ndarray) -> float:
        """Return UCB score for one article given a context vector."""
        self._ensure(article_id)
        A_inv = np.linalg.inv(self.A[article_id])
        theta = A_inv @ self.b[article_id]
        exploit = float(theta @ context)
        explore = float(self.alpha * np.sqrt(context @ A_inv @ context))
        return exploit + explore

    def update(self, article_id: str, context: np.ndarray, reward: float):
        """Update bandit weights based on observed reward."""
        self._ensure(article_id)
        ctx = context.astype(np.float32)
        self.A[article_id] += np.outer(ctx, ctx)
        self.b[article_id] += reward * ctx

    def rank(self, candidates: list[dict], context: np.ndarray) -> list[dict]:
        """Score and sort a list of article dicts by UCB score."""
        scored = []
        for art in candidates:
            s = self.score(art["id"], context)
            scored.append({**art, "ucb_score": s})
        return sorted(scored, key=lambda x: x["ucb_score"], reverse=True)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = "bandit.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"A": self.A, "b": self.b,
                         "alpha": self.alpha, "dim": self.context_dim}, f)

    @classmethod
    def load(cls, path: str = "bandit.pkl") -> "LinUCBBandit":
        with open(path, "rb") as f:
            data = pickle.load(f)
        inst = cls(context_dim=data["dim"], alpha=data["alpha"])
        inst.A = data["A"]
        inst.b = data["b"]
        return inst

    @classmethod
    def load_or_create(cls, path: str = "bandit.pkl", **kwargs) -> "LinUCBBandit":
        if os.path.exists(path):
            return cls.load(path)
        return cls(**kwargs)
