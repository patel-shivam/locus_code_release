# config.py
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List

import numpy as np
import torch

# ================================
# Config (edit params here)
# Paths are computed relative to repo root (portable; no hard-coded absolute paths)
# ================================

# repo_root/scripts/config.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[1]

# EmbedLLM concise dataset location (produced by data/ scripts)
# Expected files:
#   data/datasets/embedllm/concise/model_order.csv
#   data/datasets/embedllm/concise/concise.train.parquet
#   data/datasets/embedllm/concise/concise.val.parquet
#   data/datasets/embedllm/concise/concise.test.parquet
EMBEDLLM_CONCISE_REL = Path("data") / "datasets" / "embedllm" / "concise"
BASE = str(REPO_ROOT / EMBEDLLM_CONCISE_REL)

# Useful relative strings (avoid writing absolute paths into saved metadata)
MODEL_ORDER_CSV_REL = str(EMBEDLLM_CONCISE_REL / "model_order.csv")
MODEL_ORDER_CSV = str(REPO_ROOT / MODEL_ORDER_CSV_REL)

# Which embedding column to use from concise parquet
EMB_COL = "emb_all_mpnet_base_v2"
# default : "emb_all_mpnet_base_v2"
# 'emb_paraphrase_albert_small_v2', 'emb_all_minilm_l6_v2', 'emb_all_mpnet_base_v2', 'emb_all_distilroberta_v1'


SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {DEVICE}")

# Mixed precision flags (kept as your current defaults)
USE_BF16 = False
DTYPE = torch.float32
USE_AMP = False

# Outputs: place under repo root so it doesn't depend on current working directory
OUT_DIR_REL = Path("./locus")
OUT_DIR = str(REPO_ROOT / OUT_DIR_REL)



def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Initialize RNGs once on import
set_seed(SEED)
