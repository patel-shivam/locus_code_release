#!/usr/bin/env python3
"""
Download ONLY the small metadata/correctness CSVs from the EmbedLLM dataset.

We intentionally avoid downloading large .pth files (train_x.pth/train_y.pth/etc.).
Files fetched:
  - train.csv, val.csv, test.csv
  - model_order.csv
  - question_order.csv

Default output (relative to repo root):
  data/datasets/embedllm/raw/

Usage:
  python data/download_embedllm_metadata.py
  python data/download_embedllm_metadata.py --revision 70dee14af6101604ce1130cfcf2849daba4b6077
  python data/download_embedllm_metadata.py --out-dir data/datasets/embedllm/raw
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-id",
        default="RZ412/EmbedLLM",
        help="Hugging Face dataset repo id",
    )
    ap.add_argument(
        "--revision",
        default="main",
        help="Git revision/commit hash/tag to pin (recommended for reproducibility).",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Where to place downloaded files (defaults to data/datasets/embedllm/raw under repo).",
    )
    args = ap.parse_args()

    repo_root = _repo_root()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (repo_root / "data" / "datasets" / "embedllm" / "raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = [
        "train.csv",
        "val.csv",
        "test.csv",
        "model_order.csv",
        "question_order.csv",
    ]

    print(f"[info] repo_id   : {args.repo_id}")
    print(f"[info] revision  : {args.revision}")
    print(f"[info] out_dir   : {out_dir}")

    for fn in wanted:
        local_cached = hf_hub_download(
            repo_id=args.repo_id,
            filename=fn,
            repo_type="dataset",
            revision=args.revision,
        )
        dst = out_dir / fn
        shutil.copy2(local_cached, dst)
        print(f"[ok] {fn} -> {dst}")

    print("\n[done] Downloaded minimal EmbedLLM CSV metadata only.")


if __name__ == "__main__":
    main()
