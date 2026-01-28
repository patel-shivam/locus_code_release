# train_utils.py
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from config import DTYPE, USE_AMP
from router_model import Config, ModelEmbedRouter
from utils import to_device_with_dtype


# ================================
# Train / eval
# ================================
def train_epoch(
    model: ModelEmbedRouter,
    loader,
    optimizer,
    device: str,
    cfg: Config,
    pos_weight: Optional[float] = None,
):
    model.train()
    use_pw = (
        cfg.use_pos_weight
        and (pos_weight is not None)
        and (pos_weight > 0.0)
    )
    pos_w_tensor = torch.tensor(pos_weight, device=device) if use_pw else None

    metrics = {"loss": 0.0, "bce": 0.0}
    n_batches = 0
    i = 0
    for batch in loader:
        i += 1
        print(f"  Training batch {i}/{len(loader)}", end="\r")
        batch = to_device_with_dtype(batch, device, DTYPE)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float32,
            enabled=USE_AMP,
        ):
            out = model(batch)
            # create BCE with correct dtype each step
            bce = nn.BCEWithLogitsLoss(
                pos_weight=(
                    pos_w_tensor.to(out["logits"].dtype)
                    if pos_w_tensor is not None
                    else None
                )
            )
            loss_bce = bce(out["logits"], batch["y_target"])

            loss = loss_bce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # safer scalar logging
        metrics["loss"] += loss.detach().item()
        metrics["bce"] += loss_bce.detach().item()
        
        n_batches += 1
    for k in metrics:
        metrics[k] /= max(1, n_batches)
    return metrics
