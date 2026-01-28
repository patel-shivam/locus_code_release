# encoder_attention_modules.py
# we motivate design and notation of latent bottleneck attention block and 
# learned-query attention aggregation block from Lee et. al. : https://arxiv.org/pdf/1810.00825


from __future__ import annotations

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================
# Model components (encoder side)
# ================================
def mlp(
    sizes: List[int],
    act=nn.ReLU,
    last_act: Optional[nn.Module] = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
    if last_act is not None:
        layers += [last_act()]
    return nn.Sequential(*layers)


def cosine_sim_matrix(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # compute in fp32 for stability, return fp32
    z32 = F.normalize(z.float(), dim=-1, eps=eps)
    return z32 @ z32.T


# ---- Minimal Set Transformer blocks ----
class rFF(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.net = mlp([d, hidden, d])

    def forward(self, x):
        return self.net(x)


class MAB(nn.Module):
    def __init__(
        self,
        d: int,
        h: int,
        dk: Optional[int] = None,
        dv: Optional[int] = None,
        rff_hidden: Optional[int] = None,
    ):
        super().__init__()
        dk = dk or d
        dv = dv or d
        self.h, self.d, self.dk, self.dv = h, d, dk, dv
        self.W_q, self.W_k, self.W_v = (
            nn.Linear(d, dk, bias=False),
            nn.Linear(d, dk, bias=False),
            nn.Linear(d, dv, bias=False),
        )
        self.ln0, self.ln1 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.fc_o = nn.Linear(dv, d, bias=False)
        self.rff = rFF(d, rff_hidden or d * 2)
        self.relu = nn.ReLU()


    def forward(
        self,
        Q,
        K,
        V,
        mask_q: Optional[torch.Tensor] = None,
        mask_k: Optional[torch.Tensor] = None,
    ):
        Q_, K_, V_ = self.W_q(Q), self.W_k(K), self.W_v(V)
        B, SQ, _ = Q_.shape

        def split_heads(t, d):
            return t.view(B, -1, self.h, d // self.h).transpose(1, 2)

        Qh, Kh, Vh = (
            split_heads(Q_, self.dk),
            split_heads(K_, self.dk),
            split_heads(V_, self.dv),
        )
        att = (Qh @ Kh.transpose(-2, -1)) / math.sqrt(self.dk / self.h)
        if mask_k is not None:
            att = att.masked_fill(
                ~mask_k[:, None, None, :].bool(), float("-inf")
            )
        A = att.softmax(dim=-1)

        H = A @ Vh
        H = H.transpose(1, 2).contiguous().view(B, SQ, self.dv)
        O = self.fc_o(H)
        O = self.ln0(O + Q)
        O = self.relu(O)
        O = self.ln1(O + self.rff(O))
        if mask_q is not None:
            O = O.masked_fill(~mask_q[..., None], 0.0)
        return O




class SAB(nn.Module):
    def __init__(self, d: int, h: int, rff_hidden: Optional[int] = None):
        super().__init__()
        self.mab = MAB(d, h, dk=d, dv=d, rff_hidden=rff_hidden)

    def forward(self, X, mask: Optional[torch.Tensor] = None):
        return self.mab(X, X, X, mask_q=mask, mask_k=mask)


class ISAB(nn.Module):
    """Induced Set Attention Block: O(A * m) instead of O(A^2)."""

    def __init__(self, d: int, h: int, m: int, rff_hidden: Optional[int] = None):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m, d))
        nn.init.xavier_uniform_(self.I)
        self.mab1 = MAB(d, h, dk=d, dv=d, rff_hidden=rff_hidden)  # I <- X
        self.mab2 = MAB(d, h, dk=d, dv=d, rff_hidden=rff_hidden)  # X <- I

    def forward(self, X, mask: Optional[torch.Tensor] = None):
        B = X.size(0)
        I = self.I.repeat(B, 1, 1)  # [B, m, d]
        H = self.mab1(I, X, X, mask_q=None, mask_k=mask)  # attend from I to X
        Y = self.mab2(X, H, H, mask_q=mask, mask_k=None)  # attend from X to I
        return Y


class PMA(nn.Module):
    def __init__(self, d: int, m: int, h: int, rff_hidden: Optional[int] = None):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, m, d))
        self.mab = MAB(d, h, dk=d, dv=d, rff_hidden=rff_hidden)

    def forward(self, X, mask: Optional[torch.Tensor] = None):
        B = X.size(0)
        S = self.S.repeat(B, 1, 1)
        return self.mab(S, X, X, mask_q=None, mask_k=mask)


class SetTransformerEncoder(nn.Module):
    def __init__(
        self,
        q_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        rff_hidden: Optional[int] = None,
        dropout: float = 0.0,
        encoder_kind: str = "isab",
        m_induce: int = 64,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert encoder_kind in ("sab", "isab")
        in_dim = q_dim + 1 + q_dim  # [q, y, q*y] 
        self.input = mlp([in_dim, d_model], dropout=dropout)
        if encoder_kind == "sab":
            self.layers = nn.ModuleList(
                [SAB(d_model, n_heads, rff_hidden=rff_hidden) for _ in range(n_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    ISAB(d_model, n_heads, m=m_induce, rff_hidden=rff_hidden)
                    for _ in range(n_layers)
                ]
            )
        self.pma = PMA(d_model, m=1, h=n_heads, rff_hidden=rff_hidden)
        self.out = mlp([d_model, d_model], dropout=dropout)
        self.encoder_kind = encoder_kind

    def forward(
        self,
        q_enc: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([q_enc, y, q_enc * (2 * y - 1)], dim=-1)  
        h = self.input(x)
        for layer in self.layers:
            h = layer(h, mask)
        return self.out(self.pma(h, mask).squeeze(1))  # [B, d_model]


