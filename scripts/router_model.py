# router_model.py
# we motivate design and notation of latent bottleneck attention block and 
# learned-query attention aggregation block from Lee et. al. : https://arxiv.org/pdf/1810.00825

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn

from decoder_and_losses import (
    BilinearMLPDecoder,
)
from encoder_attention_modules import SetTransformerEncoder


# ================================
# Full model
# ================================
@dataclass
class Config:
    q_dim: int
    z_dim: int = 128
    n_heads: int = 4
    n_layers: int = 3
    encoder_dropout: float = 0.0
    decoder_hidden: int = 256
    decoder_dropout: float = 0.0
    
    encoder_kind: str = "isab"  # "isab" (default) or "sab"
    m_induce: int = 64  # ISAB inducing points

    # ---- encoder anchor subsampling controls (training only) ----
    # If encoder_anchor_sizes is provided (non-empty), we randomly choose K
    # from that list (clipped to available anchors and encoder_max_anchors).
    encoder_anchor_sizes: Optional[List[int]] = None

    # Otherwise we sample K ~ Uniform[encoder_min_anchors, encoder_max_anchors]
    # (both inclusive), after clamping to [1, A_full].
    encoder_min_anchors: Optional[int] = None
    encoder_max_anchors: Optional[int] = None  # also acts as hard upper cap

    use_pos_weight: bool = False  # whether to apply class-imbalance pos_weight in BCE

    # ---- noise regularization flags & scales ----
    add_noise_encoder_q: bool = False  # noise on encoder input queries (anchors)
    add_noise_decoder_q: bool = False  # noise on decoder input queries (targets)
    add_noise_decoder_z: bool = False  # noise on model embeddings z before decoder

    noise_encoder_q_std: float = 0.0  # std of Gaussian noise for encoder q
    noise_decoder_q_std: float = 0.0  # std of Gaussian noise for decoder q
    noise_decoder_z_std: float = 0.0  # std of Gaussian noise for decoder z


class ModelEmbedRouter(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = SetTransformerEncoder(
            cfg.q_dim,
            d_model=cfg.z_dim,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.encoder_dropout,
            encoder_kind=cfg.encoder_kind,
            m_induce=cfg.m_induce,
        )
        self.decoder = BilinearMLPDecoder(
            cfg.z_dim,
            cfg.q_dim,
            hidden=cfg.decoder_hidden,
            dropout=cfg.decoder_dropout,
        )
        self.cfg = cfg

    @torch.no_grad()
    def embed_models(
        self,
        q_anchor: torch.Tensor,
        y_anchor: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        return self.encoder(q_anchor, y_anchor, mask)

    def _sample_anchor_size(
        self,
        A_full: int,
        device: torch.device,
    ) -> int:
        """
        Decide how many anchors K to keep for this forward pass (training only).
        Uses either:
          - encoder_anchor_sizes (list of candidate sizes), or
          - uniform range [encoder_min_anchors, encoder_max_anchors].
        All values are clipped to [1, A_full].
        """
        # Hard upper bound
        K_upper = A_full
        if self.cfg.encoder_max_anchors is not None and self.cfg.encoder_max_anchors > 0:
            K_upper = min(K_upper, self.cfg.encoder_max_anchors)
        K_upper = max(1, K_upper)

        # 1) Candidate size list has priority if provided
        if self.cfg.encoder_anchor_sizes is not None and len(self.cfg.encoder_anchor_sizes) > 0:
            valid_sizes = [int(k) for k in self.cfg.encoder_anchor_sizes if k > 0]
            if len(valid_sizes) == 0:
                return K_upper
            # Clip each size to [1, K_upper]
            valid_sizes = [max(1, min(K_upper, k)) for k in valid_sizes]
            # Random choice from this list (using torch RNG for reproducibility with seeds)
            idx = torch.randint(low=0, high=len(valid_sizes), size=(1,), device=device).item()
            return int(valid_sizes[idx])

        # 2) Else, use uniform range [encoder_min_anchors, encoder_max_anchors]
        if self.cfg.encoder_min_anchors is not None:
            K_lower = max(1, min(self.cfg.encoder_min_anchors, K_upper))
        else:
            K_lower = 1

        if K_lower >= K_upper:
            return K_upper

        # torch.randint high is exclusive; we want inclusive [K_lower, K_upper]
        K = torch.randint(low=K_lower, high=K_upper + 1, size=(1,), device=device).item()
        return int(K)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q_anchor = batch["q_anchor"]  # [B, A, Dq]
        y_anchor = batch["y_anchor"]  # [B, A, 1]
        mask = batch["mask_anchor"]  # [B, A]
        q_target = batch["q_target"]  # [B, T, Dq]

        # --- Optional stochastic subsampling of anchors for the encoder ---
        if self.training:
            B, A_full, Dq = q_anchor.shape

            # Decide anchor size K for this batch
            K = self._sample_anchor_size(A_full, q_anchor.device)

            if K < A_full:
                # Same random subset for all models in the batch for efficient gather
                idx = torch.randperm(A_full, device=q_anchor.device)[:K]  # [K]
                q_anchor = q_anchor[:, idx]  # [B, K, Dq]
                y_anchor = y_anchor[:, idx]  # [B, K, 1]
                mask = mask[:, idx]  # [B, K]

                # Update batch so contrastive_target_sim sees the same subset
                batch["q_anchor"] = q_anchor
                batch["y_anchor"] = y_anchor
                batch["mask_anchor"] = mask

        # --- Noise on encoder input queries (anchors) ---
        if (
            self.training
            and self.cfg.add_noise_encoder_q
            and self.cfg.noise_encoder_q_std > 0.0
        ):
            q_anchor = q_anchor + torch.randn_like(q_anchor) * self.cfg.noise_encoder_q_std

        # --- Encode model embeddings (clean z used for contrastive) ---
        z = self.encoder(q_anchor, y_anchor, mask)  # [B, Dz]

        # --- Noise on model embeddings *into* decoder only ---
        if (
            self.training
            and self.cfg.add_noise_decoder_z
            and self.cfg.noise_decoder_z_std > 0.0
        ):
            z_for_dec = z + torch.randn_like(z) * self.cfg.noise_decoder_z_std
        else:
            z_for_dec = z

        # --- Noise on decoder input queries (targets) ---
        if (
            self.training
            and self.cfg.add_noise_decoder_q
            and self.cfg.noise_decoder_q_std > 0.0
        ):
            q_target = q_target + torch.randn_like(q_target) * self.cfg.noise_decoder_q_std

        # Decoder: predict correctness for (z, q_target) pairs
        B, T, Dq = q_target.shape
        z_rep = z_for_dec.unsqueeze(1).expand(B, T, z_for_dec.size(-1)).reshape(
            B * T, -1
        )
        q_flat = q_target.reshape(B * T, Dq)
        logits = self.decoder(z_rep, q_flat).view(B, T)

        # Return clean z for contrastive loss, noisy path only affects logits
        return {"z": z, "logits": logits}




class RouterWrapper(nn.Module):
    """Wraps your loaded encoder/decoder to expose the same methods used in eval."""
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @torch.no_grad()
    def embed_models(self, q_anchor: torch.Tensor, y_anchor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # q_anchor: [B, A, Dq], y_anchor: [B, A, 1], mask: [B, A] (bool)
        self.encoder.eval()
        return self.encoder(q_anchor, y_anchor, mask)
