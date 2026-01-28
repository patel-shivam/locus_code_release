#!/usr/bin/env python3
"""
Convenience wrapper to build concise.{train,val,test}.parquet with optional embeddings.

Typical workflow:
  1) python data/download_embedllm_metadata.py
  2) python data/build_concise_dataset.py --device cuda

This script uses the same default dirs as make_concise_split.py:
  raw    : data/datasets/embedllm/raw
  concise: data/datasets/embedllm/concise
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    repo_root = _repo_root()
    default_root = repo_root / "data" / "datasets" / "embedllm" / "raw"
    default_out = repo_root / "data" / "datasets" / "embedllm" / "concise"

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(default_root))
    ap.add_argument("--out-root", default=str(default_out))
    ap.add_argument("--device", default=None)
    ap.add_argument("--flat-embeddings", action="store_true")
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--skip-embeddings", action="store_true")
    args = ap.parse_args()

    script = repo_root / "data" / "make_concise_split.py"
    if not script.exists():
        print(f"ERROR: {script} not found", file=sys.stderr)
        raise SystemExit(1)

    for split in ["train", "val", "test"]:
        cmd = [
            sys.executable, str(script),
            "--root", args.root,
            "--out-root", args.out_root,
            "--split", split,
        ]
        if args.device is not None:
            cmd += ["--device", args.device]
        if args.flat_embeddings:
            cmd += ["--flat-embeddings"]
        if args.csv:
            cmd += ["--csv"]
        if args.skip_embeddings:
            cmd += ["--skip-embeddings"]

        print(f"\n[run] {' '.join(cmd)}")
        subprocess.check_call(cmd)

    print("\n[done] Built concise train/val/test.")


if __name__ == "__main__":
    main()
