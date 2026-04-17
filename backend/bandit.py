"""
Neural contextual bandit used by the RL ranking path.

The public API intentionally preserves the previous LinUCB-style surface:
- score(article_id, context)
- update(article_id, context, reward)
- save(path)
- load(path)
- load_or_create(path, **kwargs)

Internally this now uses a 5-head MLP ensemble with UCB-style exploration.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_float32(vector) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        return arr.reshape(-1).astype(np.float32)
    return arr


class _BanditHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _EnsembleRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], ensemble_size: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [_BanditHead(input_dim, hidden_dims) for _ in range(ensemble_size)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = [head(x) for head in self.heads]
        return torch.cat(logits, dim=1)


class NeuralContextualBandit:
    """
    5-head MLP ensemble bandit with UCB-style exploration.

    The article id is kept in the API for compatibility and to maintain a small
    article-specific reward prior on top of the neural prediction.
    """

    def __init__(
        self,
        context_dim: int = 391,
        alpha: float = 0.35,
        ensemble_size: int = 5,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        train_steps_per_update: int = 3,
        replay_capacity: int = 8000,
        min_buffer_to_train: int = 32,
        epsilon: float = 0.05,
        device: str = "cpu",
    ):
        self.context_dim = int(context_dim)
        self.alpha = float(alpha)
        self.ensemble_size = int(ensemble_size)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.train_steps_per_update = int(train_steps_per_update)
        self.replay_capacity = int(replay_capacity)
        self.min_buffer_to_train = int(min_buffer_to_train)
        self.device = torch.device(device)

        self.model = _EnsembleRegressor(self.context_dim, self.hidden_dims, self.ensemble_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.epsilon = float(epsilon)

        self.context_buffer: deque[np.ndarray] = deque(maxlen=self.replay_capacity)
        self.reward_buffer: deque[float] = deque(maxlen=self.replay_capacity)
        self.article_stats: dict[str, dict[str, float]] = {}
        self.category_stats: dict[str, dict[str, float]] = {}
        self.total_updates = 0

    def _normalize_reward(self, reward: float) -> float:
        value = float(reward)
        return float(np.clip(value, 0.0, 1.0))

    def _article_prior(self, article_id: str) -> tuple[float, float]:
        stats = self.article_stats.get(article_id)
        if not stats:
            return 0.0, 0.0
        mean_reward = float(stats.get("mean_reward", 0.0))
        count = float(stats.get("count", 0.0))
        prior_bonus = 0.12 * mean_reward
        explore_bonus = 0.05 / np.sqrt(max(count, 1.0))
        return prior_bonus, explore_bonus

    def _update_article_stats(self, article_id: str, normalized_reward: float):
        stats = self.article_stats.setdefault(article_id, {"count": 0.0, "mean_reward": 0.0})
        count = float(stats["count"])
        mean_reward = float(stats["mean_reward"])
        new_count = count + 1.0
        stats["count"] = new_count
        stats["mean_reward"] = mean_reward + ((normalized_reward - mean_reward) / new_count)

    def _update_category_stats(self, category: str, normalized_reward: float):
        key = str(category).strip().lower()
        if not key:
            return
        stats = self.category_stats.setdefault(key, {"count": 0.0, "mean_reward": 0.0})
        count = float(stats["count"])
        mean = float(stats["mean_reward"])
        new_count = count + 1.0
        stats["count"] = new_count
        stats["mean_reward"] = mean + ((normalized_reward - mean) / new_count)

    def _category_prior(self, category: str) -> tuple[float, float]:
        key = str(category).strip().lower()
        stats = self.category_stats.get(key)
        if not stats:
            return 0.0, 0.0
        mean_reward = float(stats.get("mean_reward", 0.0))
        count = float(stats.get("count", 0.0))
        # Smaller coefficients than article-level (0.12, 0.05) — category signal is noisier
        cat_prior_bonus = 0.08 * mean_reward
        cat_explore_bonus = 0.03 / np.sqrt(max(count, 1.0))
        return cat_prior_bonus, cat_explore_bonus

    def propagate_category_reward(
        self,
        category: str,
        sibling_article_ids: list,
        reward: float,
        decay: float = 0.3,
    ):
        """Apply a decayed reward to sibling articles in the same category.

        Stat-only update — does NOT add to replay buffer or trigger training,
        which would corrupt the buffer with phantom observations.
        """
        propagated = float(np.clip(reward * decay, 0.0, 1.0))
        for article_id in sibling_article_ids:
            self._update_article_stats(str(article_id), propagated)
        self._update_category_stats(category, propagated)

    def _append_example(self, context: np.ndarray, normalized_reward: float):
        self.context_buffer.append(_as_float32(context))
        self.reward_buffer.append(float(normalized_reward))

    def _sample_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = min(self.batch_size, len(self.context_buffer))
        indices = np.random.choice(len(self.context_buffer), size=batch_size, replace=False)
        contexts = np.stack([self.context_buffer[idx] for idx in indices]).astype(np.float32)
        rewards = np.asarray([self.reward_buffer[idx] for idx in indices], dtype=np.float32).reshape(-1, 1)
        x = torch.from_numpy(contexts).to(self.device)
        y = torch.from_numpy(rewards).to(self.device)
        return x, y

    def _train_step(self):
        if len(self.context_buffer) < self.min_buffer_to_train:
            return

        self.model.train()
        x, y = self._sample_batch()
        predictions = self.model(x)
        probabilities = torch.sigmoid(predictions)

        # Bootstrap-style masking keeps ensemble heads diverse.
        head_mask = torch.bernoulli(
            torch.full_like(probabilities, 0.8, device=self.device)
        )
        loss_per_head = F.mse_loss(probabilities, y.expand_as(probabilities), reduction="none")
        masked_loss = (loss_per_head * head_mask).sum()
        normalizer = head_mask.sum().clamp_min(1.0)
        loss = masked_loss / normalizer

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

    def _train_after_update(self, update_count: int):
        if len(self.context_buffer) < self.min_buffer_to_train:
            return

        warmup_multiplier = 2 if len(self.context_buffer) < 256 else 1
        steps = max(1, self.train_steps_per_update * warmup_multiplier)
        # Scale slightly with batch updates without exploding latency.
        steps = min(steps + max(update_count // 16, 0), 12)
        for _ in range(steps):
            self._train_step()

    def _predict_ensemble(self, context: np.ndarray) -> tuple[float, float]:
        vector = _as_float32(context)
        if vector.shape[0] != self.context_dim:
            raise ValueError(f"Expected context_dim={self.context_dim}, got {vector.shape[0]}")

        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(vector).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32)
        return float(np.mean(probs)), float(np.std(probs))

    def score(self, article_id: str, context: np.ndarray, category: str = "") -> float:
        predicted_reward, uncertainty = self._predict_ensemble(context)
        prior_bonus, article_explore = self._article_prior(str(article_id))
        cat_prior_bonus, cat_explore = self._category_prior(category)

        if np.random.random() < self.epsilon:
            # ε-Thompson Sampling: sample from N(mean, std) for exploration
            base = float(np.clip(np.random.normal(predicted_reward, max(uncertainty, 1e-6)), 0.0, 1.0))
        else:
            # UCB exploitation (unchanged)
            base = predicted_reward + (self.alpha * uncertainty)

        return float(np.clip(base + prior_bonus + article_explore + cat_prior_bonus + cat_explore, 0.0, 1.5))

    def update(self, article_id: str, context: np.ndarray, reward: float):
        self.update_batch([article_id], [context], [reward])

    def update_batch(
        self,
        article_ids: Iterable[str],
        contexts: Iterable[np.ndarray],
        rewards: Iterable[float],
    ):
        update_count = 0
        for article_id, context, reward in zip(article_ids, contexts, rewards):
            normalized_reward = self._normalize_reward(reward)
            self._append_example(_as_float32(context), normalized_reward)
            self._update_article_stats(str(article_id), normalized_reward)
            update_count += 1
            self.total_updates += 1

        if update_count:
            self._train_after_update(update_count)

    def rank(self, candidates: list[dict], context: np.ndarray) -> list[dict]:
        scored = []
        for article in candidates:
            article_id = article.get("news_id") or article.get("id")
            if not article_id:
                continue
            category = str(article.get("category", ""))
            bandit_score = self.score(article_id, context, category=category)
            scored.append({**article, "ucb_score": bandit_score})
        return sorted(scored, key=lambda item: item["ucb_score"], reverse=True)

    def save(self, path: str = "bandit.pkl"):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        contexts = (
            np.stack(list(self.context_buffer)).astype(np.float32)
            if self.context_buffer
            else np.empty((0, self.context_dim), dtype=np.float32)
        )
        rewards = (
            np.asarray(list(self.reward_buffer), dtype=np.float32)
            if self.reward_buffer
            else np.empty((0,), dtype=np.float32)
        )
        payload = {
            "model_type": "NeuralContextualBandit",
            "config": {
                "context_dim": self.context_dim,
                "alpha": self.alpha,
                "ensemble_size": self.ensemble_size,
                "hidden_dims": self.hidden_dims,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
                "train_steps_per_update": self.train_steps_per_update,
                "replay_capacity": self.replay_capacity,
                "min_buffer_to_train": self.min_buffer_to_train,
                "epsilon": self.epsilon,
                "device": "cpu",
            },
            "state_dict": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "contexts": contexts,
            "rewards": rewards,
            "article_stats": self.article_stats,
            "category_stats": self.category_stats,
            "total_updates": self.total_updates,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str = "bandit.pkl") -> "NeuralContextualBandit":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("model_type") != "NeuralContextualBandit":
            raise ValueError(f"Unsupported bandit state in {path}")

        config = dict(payload.get("config", {}))
        inst = cls(**config)
        inst.model.load_state_dict(payload["state_dict"])
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state:
            inst.optimizer.load_state_dict(optimizer_state)

        for context in payload.get("contexts", []):
            inst.context_buffer.append(_as_float32(context))
        for reward in payload.get("rewards", []):
            inst.reward_buffer.append(float(reward))
        inst.article_stats = {
            str(article_id): {
                "count": float(stats.get("count", 0.0)),
                "mean_reward": float(stats.get("mean_reward", 0.0)),
            }
            for article_id, stats in payload.get("article_stats", {}).items()
        }
        inst.category_stats = {
            str(k): {
                "count": float(v.get("count", 0.0)),
                "mean_reward": float(v.get("mean_reward", 0.0)),
            }
            for k, v in payload.get("category_stats", {}).items()
        }
        inst.epsilon = float(payload.get("epsilon", 0.05))
        inst.total_updates = int(payload.get("total_updates", len(inst.reward_buffer)))
        return inst

    @classmethod
    def load_or_create(cls, path: str = "bandit.pkl", **kwargs) -> "NeuralContextualBandit":
        if os.path.exists(path):
            try:
                inst = cls.load(path)
                expected_dim = kwargs.get("context_dim")
                if expected_dim and inst.context_dim != int(expected_dim):
                    return cls(**kwargs)
                if "alpha" in kwargs:
                    inst.alpha = float(kwargs["alpha"])
                if "epsilon" in kwargs:
                    inst.epsilon = float(kwargs["epsilon"])
                return inst
            except Exception as exc:
                print(f"Bandit state load failed for {path}: {exc}. Recreating bandit state.")
        return cls(**kwargs)


# Preserve older imports used elsewhere in the project.
LinUCBBandit = NeuralContextualBandit
