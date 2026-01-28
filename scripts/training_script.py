# training_script.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
from config import DEVICE, EMB_COL, MODEL_ORDER_CSV, OUT_DIR, SEED
from data_embedllm import load_split
from dataset_model_batches import make_loaders
from evaluation import compute_val_metrics, embed_all_models
from router_model import Config, ModelEmbedRouter
from train_utils import train_epoch


def _to_jsonable(x: Any) -> Any:
    """Recursively convert numpy/pandas-ish values into JSON-serializable python types."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def _df_to_records(df: Any) -> list[dict]:
    """Convert a pandas DataFrame to list-of-dicts safely for JSON."""
    if df is None:
        return []
    if isinstance(df, pd.DataFrame):
        if df.empty:
            return []
        return [_to_jsonable(r) for r in df.to_dict(orient="records")]
    # best-effort fallback
    try:
        return _to_jsonable(df)
    except Exception:
        return []


def _summarize_eval(out: dict) -> dict:
    """
    Keep only what you asked for:
      - overall routing accuracy (hit@1/3/5)
      - overall correctness prediction (correctness_pred_acc, bce_per_pair, etc.)
      - dataset-wise routing (hit@1/3/5)
      - dataset-wise correctness prediction
    """
    bp_overall = out.get("bp_overall", {})
    rt_overall = out.get("rt_overall", {})
    bp_by_dataset = out.get("bp_by_dataset", None)
    rt_by_dataset = out.get("rt_by_dataset", None)

    return {
        "overall": {
            "routing": _to_jsonable(rt_overall),
            "correctness_prediction": _to_jsonable(bp_overall),
        },
        "by_dataset": {
            "routing": _df_to_records(rt_by_dataset),
            "correctness_prediction": _df_to_records(bp_by_dataset),
        },
    }


def _interp_linear(t: float) -> float:
    return float(max(0.0, min(1.0, t)))


def _interp_cosine(t: float) -> float:
    t = float(max(0.0, min(1.0, t)))
    return float(0.5 * (1.0 + np.cos(np.pi * t)))


def noise_schedule(
    epoch: int,
    *,
    start_epoch: int,
    end_epoch: int,
    start_std: float,
    end_std: float,
    schedule: Literal["linear", "cosine"] = "cosine",
) -> float:
    if end_epoch <= start_epoch:
        return float(end_std)

    if epoch <= start_epoch:
        return float(start_std)
    if epoch >= end_epoch:
        return float(end_std)

    t = (epoch - start_epoch) / float(end_epoch - start_epoch)
    if schedule == "linear":
        w = 1.0 - _interp_linear(t)
    elif schedule == "cosine":
        w = _interp_cosine(t)
    else:
        raise ValueError(f"Unknown schedule={schedule}")

    return float(end_std + (start_std - end_std) * w)


def _cpu_state_dict(sd: dict) -> dict:
    # Save checkpoints portable across devices
    return {k: v.detach().cpu() for k, v in sd.items()}


def _save_best_checkpoint(
    *,
    model: ModelEmbedRouter,
    cfg: Config,
    aux: dict,
    Dq: int,
    emb_col: str,
    out_dir: str,
    epoch: int,
    best_val_hit1: float,
    best_test_hit1: float,
    early_stopping_patience: int,
    model_idx_train: list[int] | None = None,
    model_idx_val: list[int] | None = None,
    model_idx_test: list[int] | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    enc_path = os.path.join(out_dir, "encoder.pt")
    dec_path = os.path.join(out_dir, "decoder.pt")
    meta_pt = os.path.join(out_dir, "meta.pt")
    meta_json = os.path.join(out_dir, "meta.json")

    torch.save(_cpu_state_dict(model.encoder.state_dict()), enc_path)
    torch.save(_cpu_state_dict(model.decoder.state_dict()), dec_path)

    cfg_dict = asdict(cfg) if is_dataclass(cfg) else dict(cfg.__dict__)
    meta = {
        "cfg": cfg_dict,
        "q_dim": int(Dq),
        "emb_col": emb_col,
        "encoder_kind": cfg_dict.get("encoder_kind", None),
        "anchors_idx": [int(x) for x in aux["anchors_idx"].tolist()],
        # keep as given by config; if config is code release, this is portable
        "model_order_csv": MODEL_ORDER_CSV,
        "pytorch_version": torch.__version__,
        # --- best/early-stop bookkeeping ---
        "best_epoch": int(epoch),
        "best_val_hit@1": float(best_val_hit1),
        "best_test_hit@1": float(best_test_hit1),
        "early_stopping_patience": int(early_stopping_patience),
        # --- model indices used for encoder/decoder training/eval splits ---
        "model_indices": {
            "train": [int(i) for i in model_idx_train] if model_idx_train is not None else None,
            "val": [int(i) for i in model_idx_val] if model_idx_val is not None else None,
            "test": [int(i) for i in model_idx_test] if model_idx_test is not None else None,
        },
        "n_model_indices": {
            "train": int(len(model_idx_train)) if model_idx_train is not None else None,
            "val": int(len(model_idx_val)) if model_idx_val is not None else None,
            "test": int(len(model_idx_test)) if model_idx_test is not None else None,
        },
    }

    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)
    torch.save(meta, meta_pt)

    print(
        f"[Best ckpt saved] epoch={epoch} val_hit@1={best_val_hit1:.6f} "
        f"test_hit@1={best_test_hit1:.6f}\n"
        f"  encoder→{enc_path}\n"
        f"  decoder→{dec_path}\n"
        f"  meta(json)→{meta_json}\n"
        f"  meta(pt)→{meta_pt}"
    )


def _parse_int_list(s: str) -> list[int]:
    """
    Parse comma-separated integers, e.g. "0,1,2" -> [0,1,2]
    Empty string -> []
    """
    s = (s or "").strip()
    if not s:
        return []
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train encoder-decoder model embedding router on EmbedLLM concise dataset."
    )

    # ---- Dataset selection knobs ----
    ap.add_argument(
        "--emb-col",
        type=str,
        default=EMB_COL,
        help="Embedding column in concise parquet (default: current EMB_COL).",
    )
    ap.add_argument(
        "--tasks-train",
        type=str,
        default="",
        help='Optional comma-separated dataset/task names for train split (e.g. "mmlu,gsm8k"). Default: none.',
    )
    ap.add_argument(
        "--tasks-val",
        type=str,
        default="",
        help='Optional comma-separated dataset/task names for val split. Default: none.',
    )
    ap.add_argument(
        "--tasks-test",
        type=str,
        default="",
        help='Optional comma-separated dataset/task names for test split. Default: none.',
    )
    ap.add_argument(
        "--model-idx-train",
        type=str,
        default="",
        help='Optional comma-separated 0-based model indices for training (e.g. "0,5,10"). Default: all models.',
    )
    ap.add_argument(
        "--model-idx-val",
        type=str,
        default="",
        help="Optional comma-separated 0-based model indices for val. Default: all models.",
    )
    ap.add_argument(
        "--model-idx-test",
        type=str,
        default="",
        help="Optional comma-separated 0-based model indices for test. Default: all models.",
    )

    # ---- Training / saving knobs ----
    ap.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=True,  # SAVE_MODEL = True
        help="Whether to save best encoder/decoder checkpoints.",
    )
    ap.add_argument(
        "--anchors-per-model",
        type=int,
        default=1024,  # ANCHORS_PER_MODEL = 1024
        help="Number of query evaluations per model used for training (anchors_per_model).",
    )
    ap.add_argument(
        "--num-target-valtest",
        type=int,
        default=256,  # num_target_valtest=256
        help="Number of target queries per model sampled for val/test batches (decoder targets).",
    )
    ap.add_argument(
        "--batch-size-train",
        type=int,
        default=128,  # batch_size_train=128
        help="Batch size (models per batch) for training.",
    )
    ap.add_argument(
        "--batch-size-eval",
        type=int,
        default=128,  # batch_size_eval=128
        help="Batch size (models per batch) for eval embedding/loader construction.",
    )
    ap.add_argument(
        "--rng-seed",
        type=int,
        default=SEED,  # rng_seed=SEED
        help="Random seed for numpy/torch (default: config.SEED).",
    )

    # ---- Noise annealing knobs ----
    ap.add_argument(
        "--noise-schedule",
        type=str,
        choices=["linear", "cosine"],
        default="cosine",  # NOISE_SCHEDULE = "cosine"
        help="Noise schedule type.",
    )
    ap.add_argument(
        "--noise-start-epoch",
        type=int,
        default=0,  # NOISE_START_EPOCH = 0
        help="Epoch to start noise annealing.",
    )
    ap.add_argument(
        "--noise-end-epoch",
        type=int,
        default=500,  # NOISE_END_EPOCH = 500
        help="Epoch to end noise annealing (noise stays at end value after this).",
    )
    ap.add_argument(
        "--enc-noise-start",
        type=float,
        default=0.10,  # ENC_NOISE_START = 0.10
        help="Encoder query noise std at start.",
    )
    ap.add_argument(
        "--enc-noise-end",
        type=float,
        default=0.05,  # ENC_NOISE_END = 0.05
        help="Encoder query noise std at end.",
    )
    ap.add_argument(
        "--decq-noise-start",
        type=float,
        default=0.10,  # DECQ_NOISE_START = 0.10
        help="Decoder query noise std at start.",
    )
    ap.add_argument(
        "--decq-noise-end",
        type=float,
        default=0.05,  # DECQ_NOISE_END = 0.05
        help="Decoder query noise std at end.",
    )

    # ---- Early stopping knobs ----
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=2500,  # MAX_EPOCHS = 2500
        help="Maximum training epochs.",
    )
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=500,  # EARLY_STOPPING_PATIENCE = 500
        help="Stop if val hit@1 doesn't improve for this many epochs.",
    )
    ap.add_argument(
        "--improvement-eps",
        type=float,
        default=0.0,  # IMPROVEMENT_EPS = 0.0
        help="Minimum delta required to count as improvement in val hit@1.",
    )

    # ---- Output directory knobs ----
    ap.add_argument(
        "--models-dir",
        type=str,
        default="",  # if empty, we build ./locus_encoder_decoder/models_{anchors}/
        help="Optional override for output directory. If not set, defaults to ./locus_encoder_decoder/models_{anchors_per_model}/",
    )

    # ---- Model config knobs (current defaults preserved) ----
    ap.add_argument("--z-dim", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--encoder-dropout", type=float, default=0.1)
    ap.add_argument("--decoder-hidden", type=int, default=64)
    ap.add_argument("--decoder-dropout", type=float, default=0.1)
    ap.add_argument("--encoder-kind", type=str, choices=["sab", "isab"], default="isab")
    ap.add_argument("--m-induce", type=int, default=64)

    # Anchor subsampling controls (kept as your defaults)
    ap.add_argument(
        "--encoder-anchor-sizes",
        type=str,
        default="",
        help='Optional comma-separated anchor sizes to sample each step (e.g. "256,512,1024"). Default: none.',
    )
    ap.add_argument("--encoder-min-anchors", type=int, default=1024)
    ap.add_argument("--encoder-max-anchors", type=int, default=None)

    # Loss/class imbalance / noise flags (current defaults preserved)
    ap.add_argument("--use-pos-weight", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--add-noise-encoder-q", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--add-noise-decoder-q", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--add-noise-decoder-z", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--noise-decoder-z-std", type=float, default=0.05)

    # Optimizer knobs (current defaults preserved)
    ap.add_argument("--enc-lr", type=float, default=2e-4)
    ap.add_argument("--dec-lr", type=float, default=8e-4)
    ap.add_argument("--enc-weight-decay", type=float, default=5e-2)
    ap.add_argument("--dec-weight-decay", type=float, default=5e-2)

    return ap


def main():
    args = build_argparser().parse_args()

    # ---- Parse list-like args ----
    TASKS_TRAIN = [t.strip() for t in args.tasks_train.split(",") if t.strip()] or None
    TASKS_VAL = [t.strip() for t in args.tasks_val.split(",") if t.strip()] or None
    TASKS_TEST = [t.strip() for t in args.tasks_test.split(",") if t.strip()] or None

    MODEL_IDX_TRAIN = _parse_int_list(args.model_idx_train) or None
    MODEL_IDX_VAL = _parse_int_list(args.model_idx_val) or None
    MODEL_IDX_TEST = _parse_int_list(args.model_idx_test) or None

    encoder_anchor_sizes = _parse_int_list(args.encoder_anchor_sizes) or None

    # If user didn't pass encoder_max_anchors, default to anchors_per_model (your previous behavior)
    encoder_max_anchors = (
        int(args.encoder_max_anchors)
        if args.encoder_max_anchors is not None
        else int(args.anchors_per_model)
    )

    # ---- Load data ----
    tr = load_split("train", emb_col=args.emb_col, tasks=TASKS_TRAIN, model_indices=MODEL_IDX_TRAIN)
    va = load_split("val", emb_col=args.emb_col, tasks=TASKS_VAL, model_indices=MODEL_IDX_VAL)
    te = load_split("test", emb_col=args.emb_col, tasks=TASKS_TEST, model_indices=MODEL_IDX_TEST)

    ANCHORS_PER_MODEL = int(args.anchors_per_model)

    (dl_train, dl_val, dl_test), pos_weight, aux = make_loaders(
        tr,
        va,
        te,
        anchors_per_model=ANCHORS_PER_MODEL,
        num_target_train=ANCHORS_PER_MODEL,
        num_target_valtest=int(args.num_target_valtest),
        batch_size_train=int(args.batch_size_train),
        batch_size_eval=int(args.batch_size_eval),
        rng_seed=int(args.rng_seed),
    )

    Dq = tr["X"].shape[1]

    # ---- Noise annealing knobs ----
    NOISE_SCHEDULE: Literal["linear", "cosine"] = args.noise_schedule  # default "cosine"
    NOISE_START_EPOCH = int(args.noise_start_epoch)
    NOISE_END_EPOCH = int(args.noise_end_epoch)

    ENC_NOISE_START = float(args.enc_noise_start)
    ENC_NOISE_END = float(args.enc_noise_end)
    DECQ_NOISE_START = float(args.decq_noise_start)
    DECQ_NOISE_END = float(args.decq_noise_end)

    # ---- Early stopping knobs ----
    MAX_EPOCHS = int(args.max_epochs)
    EARLY_STOPPING_PATIENCE = int(args.early_stopping_patience)
    IMPROVEMENT_EPS = float(args.improvement_eps)

    # ---- Output dir (make BEFORE opening csv) ----
    if args.models_dir.strip():
        MODELS_DIR = args.models_dir.strip()
    else:
        MODELS_DIR = f"./locus_encoder_decoder/models_{ANCHORS_PER_MODEL}/"
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ---- Model config ----
    cfg = Config(
        q_dim=Dq,
        z_dim=int(args.z_dim),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        encoder_dropout=float(args.encoder_dropout),
        decoder_hidden=int(args.decoder_hidden),
        decoder_dropout=float(args.decoder_dropout),
        encoder_kind=str(args.encoder_kind),
        m_induce=int(args.m_induce),
        encoder_anchor_sizes=encoder_anchor_sizes,
        encoder_min_anchors=int(args.encoder_min_anchors) if args.encoder_min_anchors is not None else None,
        encoder_max_anchors=encoder_max_anchors,
        use_pos_weight=bool(args.use_pos_weight),
        add_noise_encoder_q=bool(args.add_noise_encoder_q),
        add_noise_decoder_q=bool(args.add_noise_decoder_q),
        add_noise_decoder_z=bool(args.add_noise_decoder_z),
        noise_encoder_q_std=ENC_NOISE_START,
        noise_decoder_q_std=DECQ_NOISE_START,
        noise_decoder_z_std=float(args.noise_decoder_z_std),
    )

    model = ModelEmbedRouter(cfg).to(DEVICE)

    opt = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": float(args.enc_lr), "weight_decay": float(args.enc_weight_decay)},
            {"params": model.decoder.parameters(), "lr": float(args.dec_lr), "weight_decay": float(args.dec_weight_decay)},
        ]
    )

    csv_path = os.path.join(MODELS_DIR, f"val_metrics_train_{ANCHORS_PER_MODEL}.csv")
    with open(csv_path, "w", newline="") as f:
        f.write(
            "epoch,enc_noise,decq_noise,train_loss,train_bce,val_bce,test_bce,"
            "val_acc,test_acc,val_hit1,val_hit3,val_hit5,test_hit1,test_hit3,test_hit5\n"
        )

    anchors_idx = aux["anchors_idx"]
    Qenc_tr = aux["Qenc_tr"]
    Y_tr = aux["Y_tr"]
    val_X = aux["val_X"]
    val_Y = aux["val_Y"]
    val_tasks = aux["val_tasks"]
    test_X = aux["test_X"]
    test_Y = aux["test_Y"]
    test_tasks = aux["test_tasks"]

    # ---- best/early-stop trackers ----
    best_val_hit1 = -float("inf")
    best_test_hit1 = -float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    stopped_epoch = -1

    # ---- store best VAL/TEST eval summaries (JSON-safe) ----
    best_val_eval_summary = None
    best_test_eval_summary = None

    for epoch in range(MAX_EPOCHS):
        # ---- Update noise stds in-place ----
        cfg.noise_encoder_q_std = noise_schedule(
            epoch,
            start_epoch=NOISE_START_EPOCH,
            end_epoch=NOISE_END_EPOCH,
            start_std=ENC_NOISE_START,
            end_std=ENC_NOISE_END,
            schedule=NOISE_SCHEDULE,
        )
        cfg.noise_decoder_q_std = noise_schedule(
            epoch,
            start_epoch=NOISE_START_EPOCH,
            end_epoch=NOISE_END_EPOCH,
            start_std=DECQ_NOISE_START,
            end_std=DECQ_NOISE_END,
            schedule=NOISE_SCHEDULE,
        )

        if cfg.noise_encoder_q_std <= 0.0:
            cfg.add_noise_encoder_q = False
        if cfg.noise_decoder_q_std <= 0.0:
            cfg.add_noise_decoder_q = False

        enc_lr = float(opt.param_groups[0]["lr"])
        dec_lr = float(opt.param_groups[1]["lr"])

        print(
            f"[Noise/LR] epoch {epoch:04d} | "
            f"enc_q_std={cfg.noise_encoder_q_std:.4f}, dec_q_std={cfg.noise_decoder_q_std:.4f} | "
            f"enc_lr={enc_lr:.6g}, dec_lr={dec_lr:.6g}"
        )

        trm = train_epoch(model, dl_train, opt, DEVICE, cfg, pos_weight=pos_weight)

        Z_all = embed_all_models(model, Qenc_tr, Y_tr, anchors_idx, DEVICE, batch_models=64)

        # --- Z-all exploratory stats per epoch ---
        with torch.no_grad():
            Z_cpu = Z_all.detach().float().cpu()
            M_z, _ = Z_cpu.shape
            norms = Z_cpu.norm(dim=1)
            mean_norm = norms.mean().item()
            std_norm = norms.std(unbiased=False).item() if M_z > 1 else 0.0
            if M_z > 1:
                Z_norm = F.normalize(Z_cpu, dim=1, eps=1e-8)
                cos_sim = Z_norm @ Z_norm.T
                i, j = torch.triu_indices(M_z, M_z, offset=1)
                cos_dist = 1.0 - cos_sim[i, j]
                mean_cos_dist = cos_dist.mean().item()
                std_cos_dist = cos_dist.std(unbiased=False).item()
                min_cos_dist = cos_dist.min().item()
                max_cos_dist = cos_dist.max().item()
            else:
                mean_cos_dist = std_cos_dist = 0.0
                min_cos_dist = max_cos_dist = 0.0

            print(
                f"[Z stats] epoch {epoch:04d} | mean_norm={mean_norm:.4f}, std_norm={std_norm:.4f}, "
                f"mean_cos_dist={mean_cos_dist:.4f}, std_cos_dist={std_cos_dist:.4f}, "
                f"min_cos_dist={min_cos_dist:.4f}, max_cos_dist={max_cos_dist:.4f}"
            )

        val_out = compute_val_metrics(
            model, Z_all, val_X, val_Y, val_tasks, DEVICE,
            q_chunk=1024, m_chunk=256, threshold=0.5, topk=(1, 3, 5),
        )
        test_out = compute_val_metrics(
            model, Z_all, test_X, test_Y, test_tasks, DEVICE,
            q_chunk=1024, m_chunk=256, threshold=0.5, topk=(1, 3, 5),
        )

        bp_val = val_out["bp_overall"]
        rt_val = val_out["rt_overall"]
        bp_test = test_out["bp_overall"]
        rt_test = test_out["rt_overall"]

        print(
            f"Epoch {epoch:04d} | "
            f"train(bce={trm['bce']:.4f}) | "
            f"VAL acc={bp_val['correctness_pred_acc']:.4f} bce={bp_val['bce_per_pair']:.4f} | "
            f"TEST acc={bp_test['correctness_pred_acc']:.4f} bce={bp_test['bce_per_pair']:.4f} | "
            f"VAL hit@1,3,5=({rt_val['hit@1']:.4f},{rt_val['hit@3']:.4f},{rt_val['hit@5']:.4f}) | "
            f"TEST hit@1,3,5=({rt_test['hit@1']:.4f},{rt_test['hit@3']:.4f},{rt_test['hit@5']:.4f})"
        )

        with open(csv_path, "a") as f:
            f.write(
                f"{epoch},"
                f"{cfg.noise_encoder_q_std:.6f},{cfg.noise_decoder_q_std:.6f},"
                f"{trm['loss']:.6f},{trm['bce']:.6f},"
                f"{bp_val['bce_per_pair']:.6f},{bp_test['bce_per_pair']:.6f},"
                f"{bp_val['correctness_pred_acc']:.6f},{bp_test['correctness_pred_acc']:.6f},"
                f"{rt_val['hit@1']:.6f},{rt_val['hit@3']:.6f},{rt_val['hit@5']:.6f},"
                f"{rt_test['hit@1']:.6f},{rt_test['hit@3']:.6f},{rt_test['hit@5']:.6f}\n"
            )

        # ---- BEST CHECKPOINT + EARLY STOPPING (VAL hit@1) ----
        cur_val_hit1 = float(rt_val["hit@1"])
        cur_test_hit1 = float(rt_test["hit@1"])

        if cur_val_hit1 > (best_val_hit1 + IMPROVEMENT_EPS):
            prev = best_val_hit1
            best_val_hit1 = cur_val_hit1
            best_test_hit1 = cur_test_hit1
            best_epoch = epoch
            epochs_since_improve = 0

            # ---- capture best val/test metrics summaries ----
            best_val_eval_summary = _summarize_eval(val_out)
            best_test_eval_summary = _summarize_eval(test_out)

            print(
                f"[Best] epoch={epoch} improved val_hit@1: {prev:.6f} -> {best_val_hit1:.6f}"
            )
            if args.save_model:
                _save_best_checkpoint(
                    model=model,
                    cfg=cfg,
                    aux=aux,
                    Dq=Dq,
                    emb_col=args.emb_col,
                    out_dir=MODELS_DIR,
                    epoch=best_epoch,
                    best_val_hit1=best_val_hit1,
                    best_test_hit1=best_test_hit1,
                    early_stopping_patience=EARLY_STOPPING_PATIENCE,
                    model_idx_train=MODEL_IDX_TRAIN,
                    model_idx_val=MODEL_IDX_VAL,
                    model_idx_test=MODEL_IDX_TEST,
                )

        else:
            epochs_since_improve += 1
            if epochs_since_improve >= EARLY_STOPPING_PATIENCE:
                stopped_epoch = epoch
                print(
                    f"[EarlyStop] No VAL hit@1 improvement for {EARLY_STOPPING_PATIENCE} epochs. "
                    f"Best was epoch={best_epoch} with val_hit@1={best_val_hit1:.6f}. "
                    f"Stopping at epoch={stopped_epoch}."
                )
                break

    if stopped_epoch < 0:
        stopped_epoch = epoch  # finished MAX_EPOCHS without triggering early stop

    # ---- Final training summary now includes best VAL/TEST overall + dataset-wise metrics ----
    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "best_epoch": int(best_epoch),
                "best_val_hit@1": float(best_val_hit1),
                "best_test_hit@1": float(best_test_hit1),
                "stopped_epoch": int(stopped_epoch),
                "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
                "max_epochs": int(MAX_EPOCHS),
                "args": vars(args),  # store run configuration for reproducibility
                "best_eval": {
                    "val": best_val_eval_summary,
                    "test": best_test_eval_summary,
                },
            },
            f,
            indent=2,
        )
    print(f"[Summary] wrote {summary_path}")


if __name__ == "__main__":
    main()
