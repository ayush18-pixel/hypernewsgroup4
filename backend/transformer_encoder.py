"""
transformer_encoder.py — 2-layer 4-head Transformer user history encoder.

Converts the user's last-20 article embedding sequence into a 384-dim
"user state vector" via CLS-token pooling.  This replaces the simple
weighted-mean in build_user_profile_vector() when >= 2 history items exist.

Architecture:
  - Learnable CLS token prepended to the sequence
  - Learned positional embeddings (position 0 = CLS)
  - 2-layer TransformerEncoder with Pre-LN (norm_first=True), 4 heads, FF=512
  - CLS output → LayerNorm → L2-normalise → 384-dim user state

The encoder starts with random weights (Xavier + zero bias) and is saved/
loaded via save_encoder() / load_encoder().  Even with random weights it
produces valid L2-normalised 384-dim vectors; attention still mixes input
embeddings in a position-aware way.  Quality improves if fine-tuning is added.

GPU: auto-detected via torch.cuda.is_available().  Falls back to CPU silently.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ─────────────────────────────────────────────────────────────────
_HISTORY_SEQ_LEN  = 20
_EMBED_DIM        = 384
_N_HEADS          = 4
_N_LAYERS         = 2
_FF_DIM           = 512
_DROPOUT          = 0.1
_MODEL_SAVE_PATH  = "models/transformer_encoder.pt"   # relative to project root

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────
class UserHistoryEncoder(nn.Module):
    """Compact Transformer encoder for user reading history."""

    def __init__(
        self,
        embed_dim: int = _EMBED_DIM,
        n_heads:   int = _N_HEADS,
        n_layers:  int = _N_LAYERS,
        ff_dim:    int = _FF_DIM,
        seq_len:   int = _HISTORY_SEQ_LEN,
        dropout:   float = _DROPOUT,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len   = seq_len

        # Learnable CLS token — prepended to every sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learned positional embeddings: position 0 = CLS, 1..seq_len = articles
        self.pos_embedding = nn.Embedding(seq_len + 1, embed_dim)

        # Pre-LN Transformer (norm_first=True) for stable training from random init
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        history_embeddings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            history_embeddings: (B, S, embed_dim), float32, L2-normalised
            padding_mask: (B, S+1) bool — True means "ignore this position".
                          If None, no masking applied.
        Returns:
            (B, embed_dim) — L2-normalised CLS-pooled user state
        """
        B, S, _ = history_embeddings.shape

        # Prepend CLS token: (B, S+1, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, history_embeddings], dim=1)

        # Positional embeddings
        positions = torch.arange(S + 1, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(positions)

        # Transformer
        if padding_mask is not None:
            x = self.transformer(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer(x)

        # CLS token output
        out = x[:, 0, :]                # (B, embed_dim)
        out = self.output_norm(out)
        out = F.normalize(out, dim=-1)  # L2-normalise
        return out


# ── Lazy module-level singleton ───────────────────────────────────────────────
_encoder: Optional[UserHistoryEncoder] = None


def _get_encoder() -> UserHistoryEncoder:
    global _encoder
    if _encoder is None:
        _encoder = UserHistoryEncoder().to(_DEVICE)
        _encoder.eval()
    return _encoder


# ── Public API ────────────────────────────────────────────────────────────────
def encode_user_history(
    news_ids: list,
    article_embeddings: np.ndarray,
    news_id_to_idx: dict,
    seq_len: int = _HISTORY_SEQ_LEN,
) -> Optional[np.ndarray]:
    """Encode user history into a 384-dim state vector.

    Args:
        news_ids:          Ordered list of article IDs (oldest first).
                           Typically recent_clicks[-20:] + reading_history[-20:]
        article_embeddings: Full embedding matrix (N, 384)
        news_id_to_idx:    {news_id: row_index} mapping into article_embeddings
        seq_len:           Max sequence length (last seq_len items used)

    Returns:
        (384,) float32 L2-normalised numpy array, or None if < 2 valid items found
        (None triggers the weighted-mean fallback in build_user_profile_vector)
    """
    # Take the last seq_len from the provided history
    ids_to_encode = list(news_ids)[-seq_len:]

    embeddings_list = []
    for nid in ids_to_encode:
        idx = news_id_to_idx.get(nid)
        if idx is None:
            continue
        embeddings_list.append(article_embeddings[idx].astype(np.float32))

    if len(embeddings_list) < 2:
        return None

    # Stack and L2-normalise each row
    seq = np.stack(embeddings_list, axis=0)                      # (S, 384)
    norms = np.linalg.norm(seq, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    seq = seq / norms

    x = torch.from_numpy(seq).unsqueeze(0).to(_DEVICE)          # (1, S, 384)

    enc = _get_encoder()
    with torch.no_grad():
        out = enc(x)                                             # (1, 384)

    return out.squeeze(0).cpu().numpy().astype(np.float32)       # (384,)


def save_encoder(path: str = _MODEL_SAVE_PATH):
    """Save encoder weights to disk."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    enc = _get_encoder()
    torch.save(enc.state_dict(), path)


def load_encoder(path: str = _MODEL_SAVE_PATH):
    """Load encoder weights from disk into the module singleton."""
    global _encoder
    enc = UserHistoryEncoder().to(_DEVICE)
    enc.load_state_dict(torch.load(path, map_location=_DEVICE, weights_only=True))
    enc.eval()
    _encoder = enc
