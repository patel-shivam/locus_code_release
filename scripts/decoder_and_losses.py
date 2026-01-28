# decoder_and_losses.py
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from encoder_attention_modules import mlp


class BilinearMLPDecoder(nn.Module):
    def __init__(
        self,
        z_dim: int,
        q_dim: int,
        hidden: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.U = nn.Linear(q_dim, z_dim, bias=True)
        self.mlp_head = mlp([3*z_dim, hidden, 1], dropout=dropout)

    def forward(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        uq = self.U(q)
        feats = torch.cat([z, uq, z * uq], dim=-1)
        return self.mlp_head(feats).squeeze(-1)


def select_anchors(
    Y: np.ndarray,
    strategy: str = "random",
    K: int = 2048,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    M, Q = Y.shape
    if K >= Q:
        return np.arange(Q)
    if strategy == "variance":
        p = Y.mean(axis=0)
        score = p * (1 - p)
        idx = np.argsort(-score)[:K]
    elif strategy == "entropy":
        p = Y.mean(axis=0).clip(1e-6, 1 - 1e-6)
        score = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        idx = np.argsort(-score)[:K]
    elif strategy == "random":
        idx = rng.choice(Q, size=K, replace=False)
    else:
        raise ValueError("Unknown strategy")
    return np.sort(idx)
