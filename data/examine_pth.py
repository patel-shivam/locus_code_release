#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_embedllm_root() -> str:
    repo_root = _repo_root()
    return str(repo_root / "data" / "datasets" / "embedllm" / "raw")


def safe_torch_load(path, trust=False):
    # 1) try safest route
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        if not trust:
            print(
                "\n[SAFE-LOAD FAILURE]\n"
                f"  File: {path}\n"
                f"  Error: {e}\n\n"
                "This file was likely saved with regular torch.save (pickle protocol 4).\n"
                "For safety, I won't unpickle it unless you pass --i-trust-this-file.\n"
                "If you’re sure this file is from a trusted source, rerun with:\n"
                "  python data/examine_pth.py --i-trust-this-file\n",
                file=sys.stderr
            )
            sys.exit(2)
    # 2) trusted fallback (can execute arbitrary code if the file is malicious)
    print(f"[warning] Falling back to torch.load(weights_only=False) for {path}", file=sys.stderr)
    return torch.load(path, map_location="cpu")


def describe(name, obj):
    print(f"\n=== {name} ===")
    print(f"type: {type(obj)}")
    if isinstance(obj, np.ndarray):
        print(f"  numpy  shape={obj.shape}  dtype={obj.dtype}")
        return ("np", obj.shape)
    if torch.is_tensor(obj):
        print(f"  tensor shape={tuple(obj.shape)} dtype={obj.dtype}")
        return ("tensor", tuple(obj.shape))
    if isinstance(obj, dict):
        shapes = {}
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                shapes[k] = v.shape
                print(f"  {k:<24} numpy  {v.shape} {v.dtype}")
            elif torch.is_tensor(v):
                shapes[k] = tuple(v.shape)
                print(f"  {k:<24} tensor {tuple(v.shape)} {v.dtype}")
            else:
                print(f"  {k:<24} {type(v)}")
        return ("dict", shapes)
    if isinstance(obj, (list, tuple)):
        print(f"  len={len(obj)}")
        return ("sequence", len(obj))
    return ("other", None)


def line_count_csv(path):
    # count lines quickly; subtract header
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return max(n - 1, 0)


def infer_rows(kind_meta):
    kind, meta = kind_meta
    if kind in ("tensor", "np"):
        return meta[0] if len(meta) >= 1 else None
    if kind == "dict":
        best = None
        for shape in meta.values():
            if shape and len(shape) >= 1:
                best = max(best or 0, shape[0])
        return best
    if kind == "sequence":
        return meta
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=os.environ.get("EMBEDLLM_ROOT", _default_embedllm_root()),
        help="EmbedLLM directory (defaults to data/datasets/embedllm/raw).",
    )
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"], help="Which splits to inspect")
    ap.add_argument(
        "--i-trust-this-file",
        action="store_true",
        help="Allow fallback to pickle-based torch.load (unsafe for untrusted files).",
    )
    args = ap.parse_args()

    root = args.root
    print(f"[info] Using root: {root}")

    for split in args.splits:
        xpth = os.path.join(root, f"{split}_x.pth")
        ypth = os.path.join(root, f"{split}_y.pth")
        csvp = os.path.join(root, f"{split}.csv")

        Xinfo = Yinfo = None

        if os.path.exists(xpth):
            X = safe_torch_load(xpth, trust=args.i_trust_this_file)
            Xinfo = describe(f"{split}_x.pth", X)
            same = np.allclose(X, X[0:1, :, :], atol=1e-6)
            print("Embeddings identical across models?", bool(same))
        else:
            print(f"\n=== {split}_x.pth not found ===")

        if os.path.exists(ypth):
            y = safe_torch_load(ypth, trust=args.i_trust_this_file)
            Yinfo = describe(f"{split}_y.pth", y)
        else:
            print(f"\n=== {split}_y.pth not found ===")

        if os.path.exists(csvp):
            n_csv = line_count_csv(csvp)
            print(f"\n[{split}] CSV rows: {n_csv}")
            if Xinfo:
                nx = infer_rows(Xinfo)
                if nx is not None:
                    print(f"[{split}] inferred X rows: {nx}")
            if Yinfo:
                ny = infer_rows(Yinfo)
                if ny is not None:
                    print(f"[{split}] inferred y rows: {ny}")


if __name__ == "__main__":
    main()
