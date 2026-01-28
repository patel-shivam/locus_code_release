# data_embedllm.py
from __future__ import annotations

import ast
import os
import re
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import BASE, EMB_COL

LABEL_PAT = re.compile(r"^label_(\d+)$")


def parse_embedding_series(series: pd.Series) -> np.ndarray:
    """Parse a pandas Series of embeddings (lists / arrays / strings) into
    a float32 numpy array of shape [N, D].
    """
    if series.dtype != object:
        return np.stack(series.values).astype(np.float32)
    parsed = []
    for v in series.values:
        if isinstance(v, (list, np.ndarray)):
            parsed.append(np.array(v, dtype=np.float32))
        else:
            parsed.append(np.array(ast.literal_eval(v), dtype=np.float32))
    return np.stack(parsed, axis=0).astype(np.float32)


def _merge_dataset_names(
    tasks: np.ndarray,
    merge_prefixes: Tuple[str, ...] = ("mmlu", "gpqa"),
) -> np.ndarray:
    """Map raw task strings to merged dataset names (all lowercase)."""
    def dataset_name(task_str: str) -> str:
        s = str(task_str).strip().lower()
        for pref in merge_prefixes:
            if s.startswith(pref):
                return pref
        return s

    return np.array([dataset_name(t) for t in tasks], dtype=object)


def _split_path(base: str, split: str) -> str:
    """
    code release default is:
      concise.train.parquet / concise.val.parquet / concise.test.parquet
    We also support an older fallback name:
      concise_train.parquet / concise_val.parquet / concise_test.parquet
    """
    p1 = os.path.join(base, f"concise.{split}.parquet")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(base, f"concise_{split}.parquet")
    return p2


def load_split(
    split: str,
    emb_col: str = EMB_COL,
    tasks: Optional[List[str]] = None,
    model_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Load a single EmbedLLM split with optional task and model subsampling."""
    path = _split_path(BASE, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find split parquet for '{split}'. Tried:\n"
            f"  - {os.path.join(BASE, f'concise.{split}.parquet')}\n"
            f"  - {os.path.join(BASE, f'concise_{split}.parquet')}\n"
            f"Make sure you ran the data/ scripts to create concise parquets."
        )

    df = pd.read_parquet(path)

    if emb_col not in df.columns:
        raise ValueError(
            f"{emb_col} not in columns. Available: "
            f"{[c for c in df.columns if c.startswith('emb_')]}"
        )

    # -----------------------------
    # 1) Optional task-based filter
    # -----------------------------
    if tasks:
        categories = df["category"].astype(str).to_numpy()
        merged_row_names = _merge_dataset_names(categories)

        tasks_arr = np.array(tasks, dtype=object)
        merged_allowed = _merge_dataset_names(tasks_arr)
        allowed_set = {str(t).lower() for t in merged_allowed}

        mask = np.isin(merged_row_names, list(allowed_set))
        n_total = len(df)
        n_kept = int(mask.sum())
        print(
            f"[load_split:{split}] task filter: keeping {n_kept}/{n_total} "
            f"examples for tasks={tasks}"
        )

        if n_kept == 0:
            print(
                f"[load_split:{split}] WARNING: no rows matched the requested "
                f"tasks {tasks}. Returning the unfiltered split."
            )
        else:
            df = df.iloc[mask].reset_index(drop=True)

    # -------------------------------------
    # 2) Build label column list (all M)
    # -------------------------------------
    all_label_cols = sorted(
        [c for c in df.columns if LABEL_PAT.match(c)],
        key=lambda c: int(LABEL_PAT.match(c).group(1)),
    )
    num_models_total = len(all_label_cols)

    # ----------------------------------------------
    # 3) Optional model subsampling (by indices)
    # ----------------------------------------------
    if model_indices is not None and len(model_indices) > 0:
        idx_unique: List[int] = []
        for i in model_indices:
            ii = int(i)
            if ii not in idx_unique:
                idx_unique.append(ii)

        valid_idx = [ii for ii in idx_unique if 0 <= ii < num_models_total]
        if len(valid_idx) == 0:
            raise ValueError(
                f"[load_split:{split}] model_indices {model_indices} produced "
                f"no valid indices within [0, {num_models_total-1}]."
            )

        label_cols = [all_label_cols[ii] for ii in valid_idx]
        print(
            f"[load_split:{split}] model filter: keeping "
            f"{len(label_cols)}/{num_models_total} models via indices={valid_idx}"
        )
    else:
        label_cols = all_label_cols

    # -------------------------------------
    # 4) Parse embeddings + labels
    # -------------------------------------
    X = parse_embedding_series(df[emb_col])  # [N, Dq]
    Y = df[label_cols].to_numpy().astype(np.uint8)  # [N, M_sel]

    out: Dict[str, Any] = {
        "X": X,
        "Y": Y,
        "prompt_id": df["prompt_id"].to_numpy(),
        "category": df["category"].to_numpy(),
        "df": df,
        "label_cols": label_cols,
    }
    return out
