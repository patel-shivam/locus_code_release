# dataset_model_batches.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from decoder_and_losses import select_anchors


# ================================
# Dataset (anchors & targets from SAME pool)
# ================================
class ModelBatchDataset(torch.utils.data.Dataset):
    """
    For each model m:
      - Provide its anchor responses y_anchor[m, anchors_idx] with corresponding q_anchor
      - Sample num_target target queries.
        * If targets_from_anchors=True: sample from the anchor set (anchors_idx)
        * Else: sample from the full pool of Qenc_target
      - Return (q_target, y_target[m]) pairs to learn decoder
    """

    def __init__(
        self,
        Y_anchor: np.ndarray,
        Qenc_anchor: np.ndarray,
        anchors_idx: np.ndarray,
        Y_target: np.ndarray,
        Qenc_target: np.ndarray,
        num_target_per_model: int = 64,
        rng: Optional[np.random.Generator] = None,
        targets_from_anchors: bool = True,
    ):
        self.Ya = Y_anchor.astype(np.float32)  # [M, Qa]
        self.Qa = Qenc_anchor.astype(np.float32)  # [Qa, Dq]
        self.anchors_idx = anchors_idx  # [A]
        self.A = len(anchors_idx)
        self.M = self.Ya.shape[0]

        self.Yt = Y_target.astype(np.float32)  # [M, Qt]
        self.Qt = Qenc_target.astype(np.float32)  # [Qt, Dq]
        self.Qt_n = self.Qt.shape[0]

        # choose target pool
        self.targets_from_anchors = targets_from_anchors
        if targets_from_anchors:
            # Targets will be sampled from the same anchor set.
            # NOTE: anchors_idx must be valid indices for Qenc_target / Y_target
            self.target_pool = self.anchors_idx
        else:
            # Original behavior: any query in the target pool
            self.target_pool = np.arange(self.Qt_n)

        self.num_target = min(num_target_per_model, len(self.target_pool))
        self.rng = rng or np.random.default_rng(0)

    def __len__(self):
        return self.M

    def __getitem__(self, m: int) -> Dict[str, np.ndarray]:
        # Anchors are always taken from anchors_idx
        y_anchor = self.Ya[m, self.anchors_idx]  # [A]
        q_anchor = self.Qa[self.anchors_idx]  # [A, Dq]

        # Targets sampled from chosen pool
        t_idx = self.rng.choice(self.target_pool, size=self.num_target, replace=False)
        q_t = self.Qt[t_idx]  # [T, Dq]
        y_t = self.Yt[m, t_idx]  # [T]

        return {
            "model_id": m,
            "q_anchor": q_anchor,
            "y_anchor": y_anchor,
            "q_target": q_t,
            "y_target": y_t,
        }


def collate_batch(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    model_ids = torch.tensor(
        [b["model_id"] for b in batch], dtype=torch.long
    )
    q_anchor = torch.from_numpy(
        np.stack([b["q_anchor"] for b in batch], axis=0)
    )
    y_anchor = torch.from_numpy(
        np.stack([b["y_anchor"] for b in batch], axis=0)
    ).unsqueeze(-1)
    B, A, _ = q_anchor.shape
    mask_anchor = torch.ones(B, A, dtype=torch.bool)
    q_target = torch.from_numpy(
        np.stack([b["q_target"] for b in batch], axis=0)
    )
    y_target = torch.from_numpy(
        np.stack([b["y_target"] for b in batch], axis=0)
    )
    return {
        "model_id": model_ids,
        "q_anchor": q_anchor.float(),
        "y_anchor": y_anchor.float(),
        "mask_anchor": mask_anchor,
        "q_target": q_target.float(),
        "y_target": y_target.float(),
    }


def make_loaders(
    tr: Dict[str, np.ndarray],
    va: Dict[str, np.ndarray],
    te: Dict[str, np.ndarray],
    anchors_per_model: int = 1024,
    num_target_train: int = 64,
    num_target_valtest: int = 128,
    batch_size_train: int = 128,
    batch_size_eval: int = 128,
    rng_seed: int = 42,
):
    rng = np.random.default_rng(rng_seed)
    # orient to [M, Q]
    Y_tr = tr["Y"].T.astype(np.float32)  # [M, Q_tr]
    Y_va = va["Y"].T.astype(np.float32)  # [M, Q_val]
    Y_te = te["Y"].T.astype(np.float32)  # [M, Q_te]

    Qenc_tr = tr["X"].astype(np.float32)  # [Q_tr, Dq]
    Qenc_va = va["X"].astype(np.float32)  # [Q_val, Dq]
    Qenc_te = te["X"].astype(np.float32)  # [Q_te, Dq]

    anchors_idx = select_anchors(
        Y_tr,
        strategy="random",
        K=min(anchors_per_model, Qenc_tr.shape[0]),
        rng=rng,
    )

    # pos_weight from the *actual* train target pool (anchors included)
    p_pos = Y_tr.mean()
    pos_weight = float((1 - p_pos) / max(1e-6, p_pos))
    print(f"Train pos rate≈{p_pos:.4f} → pos_weight={pos_weight:.3f}")

    ds_train = ModelBatchDataset(
        Y_anchor=Y_tr,
        Qenc_anchor=Qenc_tr,
        anchors_idx=anchors_idx,
        Y_target=Y_tr,
        Qenc_target=Qenc_tr,
        num_target_per_model=num_target_train,
        rng=np.random.default_rng(rng_seed),
        targets_from_anchors=True,  # decoder targets from anchor set
    )
    ds_val = ModelBatchDataset(
        Y_anchor=Y_tr,
        Qenc_anchor=Qenc_tr,
        anchors_idx=anchors_idx,
        Y_target=Y_va,
        Qenc_target=Qenc_va,
        num_target_per_model=num_target_valtest,
        rng=np.random.default_rng(rng_seed + 1),
        targets_from_anchors=False,
    )
    ds_test = ModelBatchDataset(
        Y_anchor=Y_tr,
        Qenc_anchor=Qenc_tr,
        anchors_idx=anchors_idx,
        Y_target=Y_te,
        Qenc_target=Qenc_te,
        num_target_per_model=num_target_valtest,
        rng=np.random.default_rng(rng_seed + 2),
        targets_from_anchors=False,
    )

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        drop_last=False,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
        drop_last=False,
    )
    dl_test = torch.utils.data.DataLoader(
        ds_test,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
        drop_last=False,
    )

    # NOTE: For metrics we want [N, M] label matrices and per-query task labels,
    # so we keep the original va/te["Y"] and va/te["category"].
    aux = dict(
        anchors_idx=anchors_idx,
        Qenc_tr=Qenc_tr,
        Y_tr=Y_tr,  # [M, Q_tr] for embedding
        val_X=Qenc_va,  # [N_val, Dq]
        val_Y=va["Y"],  # [N_val, M]
        val_tasks=va["category"],  # [N_val]
        test_X=Qenc_te,  # [N_test, Dq]
        test_Y=te["Y"],  # [N_test, M]
        test_tasks=te["category"],  # [N_test]
    )
    return (dl_train, dl_val, dl_test), pos_weight, aux
