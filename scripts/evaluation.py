# evaluation.py
from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from config import DTYPE, USE_AMP
from router_model import ModelEmbedRouter, RouterWrapper


@torch.no_grad()
def embed_all_models(
    model: Union[ModelEmbedRouter, RouterWrapper],
    Qenc_tr: np.ndarray,
    Y_tr: np.ndarray,
    anchors_idx: np.ndarray,
    device: str,
    batch_models: int = 64,
) -> torch.Tensor:
    """
    Returns Z_all: [M, Dz] embeddings for all *selected* models using TRAIN anchors provided by anchors_idx.

    - Qenc_tr: query encodings used for anchors
    - Y_tr    : correctness matrix for the same queries; rows = selected models.
      M can be any number (e.g., 112, or a subset like 37).
    """
    model.eval()
    M = Y_tr.shape[0]
    A = len(anchors_idx)
    Dq = Qenc_tr.shape[1]
    Z = []
    q_anchor = torch.from_numpy(Qenc_tr[anchors_idx]).to(
        device=device, dtype=DTYPE
    )  # [A, Dq]
    mask1 = torch.ones(1, A, dtype=torch.bool, device=device)
    for start in range(0, M, batch_models):
        end = min(M, start + batch_models)
        y_a = (
            torch.from_numpy(Y_tr[start:end, anchors_idx])
            .to(device=device, dtype=DTYPE)
            .unsqueeze(-1)
        )  # [B,A,1]
        q_a = q_anchor.unsqueeze(0).expand(end - start, A, Dq)

        # <<< important: autocast matches training mixed precision >>>
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=USE_AMP,
        ):
            z = model.embed_models(q_a, y_a, mask1.expand(end - start, A))

        Z.append(z)
    return torch.cat(Z, dim=0)  # [M, Dz] (DTYPE on device)


@torch.no_grad()
def score_queries_against_models(
    # model: ModelEmbedRouter,
    model: Union[ModelEmbedRouter, RouterWrapper],
    Z_all: torch.Tensor,  # [M, Dz] on device
    Qenc: np.ndarray,  # [N, Dq] np
    device: str,
    q_chunk: int = 1024,
    m_chunk: int = 256,
) -> np.ndarray:
    """
    Returns scores/logits matrix S: [N, M] where S[n,m] = decoder(z_m, q_n).
    Chunked to avoid OOM.
    Not using Decoder.forward for efficiency, otherwise U.q will have to be recalculated for all models for a given query.
    """
    model.eval()
    N, Dq = Qenc.shape
    M, Dz = Z_all.shape
    S = np.empty((N, M), dtype=np.float32)
    U = model.decoder.U
    for i in range(0, N, q_chunk):
        qb = torch.from_numpy(Qenc[i : i + q_chunk]).to(
            device=device, dtype=DTYPE
        )  # [nq, Dq]
        nq = qb.size(0)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=USE_AMP,
        ):
            uq = U(qb)  # [nq, Dz]
            for j in range(0, M, m_chunk):
                zc = Z_all[j : j + m_chunk]  # [mc, Dz]
                mc = zc.size(0)
                z_exp = zc.unsqueeze(0).expand(nq, mc, Dz)
                q_exp = qb.unsqueeze(1).expand(nq, mc, Dq)
                uq_exp = uq.unsqueeze(1).expand(nq, mc, Dz)
                feats = torch.cat([z_exp, uq_exp, z_exp * uq_exp], dim=-1).reshape(nq*mc, -1)
                logits = (
                    model.decoder.mlp_head(feats)
                    .squeeze(-1)
                    .view(nq, mc)
                )
                S[i : i + nq, j : j + mc] = (
                    logits.detach().float().cpu().numpy()
                )
    return S  # logits


def _pair_metrics_from_probs(
    scores: np.ndarray,
    labels: np.ndarray,
    thr: float = 0.5,
) -> Dict[str, float]:
    """
    Compute metrics from *logits* (scores) so that BCE is on the same scale
    as nn.BCEWithLogitsLoss. We still use probabilities for accuracy,
    Brier score, AUC, etc.
    """
    # Treat `scores` as logits
    logits = scores.astype(np.float64)
    y = labels.astype(np.float64)

    # Stable sigmoid to get probabilities for metrics
    logits_clipped = np.clip(logits, -40.0, 40.0)
    probs = 1.0 / (1.0 + np.exp(-logits_clipped))

    # 0/1 predictions for accuracy
    pred01 = (probs >= thr).astype(np.uint8)
    correct = pred01 == labels

    # === BCE with logits (same functional form as BCEWithLogitsLoss) ===
    # loss(x, y) = max(x,0) - x*y + log1p(exp(-abs(x)))
    x = logits
    bce = float(
        np.mean(
            np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))
        )
    )

    # Brier score & AUC on probs
    brier = float(np.mean((probs - y) ** 2))
    try:
        auc = float(roc_auc_score(y.ravel(), probs.ravel()))
    except Exception:
        auc = float("nan")

    return dict(
        correctness_pred_acc=float(correct.mean()),
        bce_per_pair=bce,
        brier=brier,
        roc_auc=auc,
        mean_pred=float(probs.mean()),
        mean_label=float(y.mean()),
        num_queries=int(probs.shape[0]),
        num_pairs=int(probs.size),
        num_models=int(probs.shape[1]),
    )


def _routing_metrics_from_scores(
    scores: np.ndarray,
    labels01: np.ndarray,
    topk: Tuple[int, ...] = (1, 3, 5),
    ignore_zero_positive_queries: bool = False,
) -> Dict[str, float]:
    pos_counts = labels01.sum(axis=1)
    mask = (np.ones(len(scores), dtype=bool))
    s_mask, y_mask = scores[mask], labels01[mask]
    if s_mask.shape[0] == 0:
        return {
            "num_queries": 0,
            "avg_num_positive_models": float(
                pos_counts.mean() if len(pos_counts) > 0 else 0.0
            ),
            **{f"hit@{k}": float("nan") for k in topk},
        }
    order = np.argsort(-s_mask, axis=1)
    out = {
        "num_queries": int(s_mask.shape[0]),
        "avg_num_positive_models": float(pos_counts[mask].mean()),
    }
    rows = np.arange(s_mask.shape[0])[:, None]
    for K in topk:
        Kc = min(K, s_mask.shape[1])
        top_idx = order[:, :Kc]
        any_pos = y_mask[rows, top_idx].any(axis=1).astype(np.float32)
        out[f"hit@{K}"] = float(any_pos.mean())
    return out


def _merge_dataset_names(
    tasks: np.ndarray,
    merge_prefixes: Tuple[str, ...] = ("mmlu", "gpqa"),
) -> np.ndarray:
    def dataset_name(task_str: str) -> str:
        s = str(task_str).strip().lower()
        for pref in merge_prefixes:
            if s.startswith(pref):
                return pref.upper()
        return str(task_str)

    return np.array([dataset_name(t) for t in tasks], dtype=object)


@torch.no_grad()
def compute_val_metrics(
    model: Union[ModelEmbedRouter, RouterWrapper],
    Z_all: torch.Tensor,
    val_X: np.ndarray,  # [N_val, Dq]
    val_Y: np.ndarray,  # [N_val, M]
    val_tasks: np.ndarray,  # [N_val,] categories
    device: str,
    q_chunk: int = 1024,
    m_chunk: int = 256,
    threshold: float = 0.5,
    topk: Tuple[int, ...] = (1, 3, 5),
    ignore_zero_positive_queries: bool = False,
) -> Dict[str, object]:
    ignore_zero_positive_queries = False

    M_emb = Z_all.shape[0]
    if val_Y.shape[1] != M_emb:
        raise ValueError(
            f"compute_val_metrics: val_Y has {val_Y.shape[1]} models but Z_all "
            f"has embeddings for {M_emb} models. Make sure you're passing the "
            "same model subset to both training and evaluation."
        )

    scores = score_queries_against_models(
        model, Z_all, val_X, device,
        q_chunk=q_chunk, m_chunk=m_chunk
    )  # logits [N_val, M_emb]

    labels = val_Y.astype(np.uint8)

    out: Dict[str, object] = {}
    # BCE now computed from logits internally
    out["bp_overall"] = _pair_metrics_from_probs(
        scores,
        labels,
        thr=threshold,
    )
    out["rt_overall"] = _routing_metrics_from_scores(
        scores,
        (labels > 0.5),
        topk=topk,
        ignore_zero_positive_queries=ignore_zero_positive_queries,
    )

    gb = np.asarray(val_tasks)
    rows_task_bp, rows_task_rt = [], []
    for task in np.unique(gb):
        idx = gb == task
        if idx.sum() == 0:
            continue
        rows_task_bp.append(
            {
                "task": str(task),
                **_pair_metrics_from_probs(
                    scores[idx], labels[idx], thr=threshold
                ),
            }
        )
        rows_task_rt.append(
            {
                "task": str(task),
                **_routing_metrics_from_scores(
                    scores[idx],
                    (labels[idx] > 0.5),
                    topk=topk,
                    ignore_zero_positive_queries=ignore_zero_positive_queries,
                ),
            }
        )
    out["bp_by_task"] = (
        pd.DataFrame(rows_task_bp).sort_values(
            "num_pairs", ascending=False
        )
        if rows_task_bp
        else pd.DataFrame()
    )
    out["rt_by_task"] = (
        pd.DataFrame(rows_task_rt).sort_values(
            "num_queries", ascending=False
        )
        if rows_task_rt
        else pd.DataFrame()
    )

    ds_keys = _merge_dataset_names(gb)
    rows_ds_bp, rows_ds_rt = [], []
    for ds_name in np.unique(ds_keys):
        idx = ds_keys == ds_name
        if idx.sum() == 0:
            continue
        rows_ds_bp.append(
            {
                "dataset": str(ds_name),
                **_pair_metrics_from_probs(
                    scores[idx], labels[idx], thr=threshold
                ),
            }
        )
        rows_ds_rt.append(
            {
                "dataset": str(ds_name),
                **_routing_metrics_from_scores(
                    scores[idx],
                    (labels[idx] > 0.5),
                    topk=topk,
                    ignore_zero_positive_queries=ignore_zero_positive_queries,
                ),
            }
        )
    out["bp_by_dataset"] = (
        pd.DataFrame(rows_ds_bp).sort_values(
            "num_pairs", ascending=False
        )
        if rows_ds_bp
        else pd.DataFrame()
    )
    out["rt_by_dataset"] = (
        pd.DataFrame(rows_ds_rt).sort_values(
            "num_queries", ascending=False
        )
        if rows_ds_rt
        else pd.DataFrame()
    )


    probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -40.0, 40.0)))
    out["scores"] = scores  # logits
    out["probs"] = probs  # probs
    out["labels"] = labels
    return out
