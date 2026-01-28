#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

HF_MODELS = {
    "paraphrase_albert_small_v2": "sentence-transformers/paraphrase-albert-small-v2",
    "all_minilm_l6_v2":           "sentence-transformers/all-MiniLM-L6-v2",
    "all_mpnet_base_v2":          "sentence-transformers/all-mpnet-base-v2",
    "all_distilroberta_v1":       "sentence-transformers/all-distilroberta-v1",
}

META_COLS = ["prompt_id", "prompt", "category_id", "category"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_model_order(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "model_order.csv")
    df = df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")]
    assert {"model_id", "model_name"}.issubset(df.columns), df.columns
    df = df.sort_values("model_id").reset_index(drop=True)
    return df


def read_split_csv(path: Path, model_map: dict[str, int]) -> pd.DataFrame:
    usecols = ["prompt_id", "model_id", "model_name", "label", "prompt", "category_id", "category"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    mi = df["model_id"]
    miss = mi.isna()
    if miss.any():
        fill = df.loc[miss, "model_name"].astype(str).map(model_map)
        mi = mi.copy()
        mi.loc[miss] = fill
    df = df.loc[~mi.isna()].copy()
    df["model_id"] = mi.loc[~mi.isna()].astype(int).to_numpy()

    df = df[["prompt_id", "prompt", "category_id", "category", "model_id", "label"]]
    df["label"] = df["label"].astype(np.uint8)

    # keep last occurrence per (prompt_id, model_id)
    df = df.sort_index()
    df = df.groupby(["prompt_id", "model_id"], as_index=False).tail(1)
    return df


def pivot_labels(df: pd.DataFrame, model_ids_ordered: list[int]) -> pd.DataFrame:
    meta = df.drop_duplicates("prompt_id")[META_COLS].set_index("prompt_id")
    wide = df.pivot(index="prompt_id", columns="model_id", values="label").reindex(columns=model_ids_ordered)
    wide.columns = [f"label_{int(c)}" for c in wide.columns]
    for mid in model_ids_ordered:
        col = f"label_{int(mid)}"
        if col not in wide.columns:
            wide[col] = np.nan
    out = meta.join(wide, how="left").reset_index()
    return out


def encode_embeddings(texts: list[str], model_name: str, batch_size: int = 256, device: str | None = None) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)


def add_embeddings(
    df: pd.DataFrame,
    encoders: dict[str, str],
    device: str | None = None,
    flat: bool = False,
    float_dtype=np.float32,
) -> pd.DataFrame:
    texts = df["prompt"].astype(str).tolist()
    for slug, hf_name in encoders.items():
        emb = encode_embeddings(texts, hf_name, batch_size=256, device=device).astype(float_dtype, copy=False)
        if not flat:
            df[f"emb_{slug}"] = list(emb)   # list column (parquet-friendly)
        else:
            for i in range(emb.shape[1]):
                df[f"emb_{slug}_{i}"] = emb[:, i]
    return df


def reorder_cols(df: pd.DataFrame, flat: bool = False) -> pd.DataFrame:
    # keep meta first, then embeddings, then labels, then any leftovers
    cols = df.columns.tolist()
    meta = [c for c in META_COLS if c in df.columns]

    emb = [c for c in cols if c.startswith("emb_")]  # works for both flat and list-col

    labels = sorted(
        [c for c in cols if c.startswith("label_")],
        key=lambda c: int(c.split("_")[1]),
    )
    left = [c for c in cols if c not in meta + emb + labels]
    return df[meta + emb + labels + left]


def main() -> None:
    repo_root = _repo_root()
    default_root = repo_root / "data" / "datasets" / "embedllm" / "raw"
    default_out = repo_root / "data" / "datasets" / "embedllm" / "concise"

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(default_root), help="EmbedLLM ROOT (contains train/val/test.csv)")
    ap.add_argument("--out-root", default=str(default_out), help="Destination directory for concise outputs")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--device", default=None, help="SentenceTransformer device: 'cuda' or 'cpu'")
    ap.add_argument("--flat-embeddings", action="store_true", help="Expand embedding vectors into many columns")
    ap.add_argument("--csv", action="store_true", help="Also write CSV (Parquet is default & preferred)")
    ap.add_argument("--skip-embeddings", action="store_true", help="Build pivot only, skip embedding computation")
    args = ap.parse_args()

    ROOT = Path(args.root).expanduser()
    OUT = Path(args.out_root).expanduser()
    split = args.split
    OUT.mkdir(parents=True, exist_ok=True)

    # Copy model_order.csv into OUT (once)
    src_model_order = ROOT / "model_order.csv"
    dst_model_order = OUT / "model_order.csv"
    if src_model_order.exists() and (not dst_model_order.exists()):
        pd.read_csv(src_model_order).to_csv(dst_model_order, index=False)

    model_order = load_model_order(ROOT)
    model_ids_ordered = model_order["model_id"].astype(int).tolist()
    name2id = dict(zip(model_order["model_name"].astype(str), model_order["model_id"].astype(int)))

    csv_path = ROOT / f"{split}.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    df = read_split_csv(csv_path, name2id)
    wide = pivot_labels(df, model_ids_ordered)

    if not args.skip_embeddings:
        print(f"[info] computing embeddings for {len(wide)} prompts ...")
        wide = add_embeddings(wide, HF_MODELS, device=args.device, flat=args.flat_embeddings)

    # >>> ensure embeddings come before labels <<<
    wide = reorder_cols(wide, flat=args.flat_embeddings)

    outbase = f"concise.{split}"
    pq_path = OUT / f"{outbase}.parquet"
    wide.to_parquet(pq_path, index=False)
    print(f"[ok] wrote {pq_path}")

    if args.csv:
        csv_out = OUT / f"{outbase}.csv"
        wide.to_csv(csv_out, index=False)
        print(f"[ok] wrote {csv_out} (note: embedding list columns get stringified)")


if __name__ == "__main__":
    main()
