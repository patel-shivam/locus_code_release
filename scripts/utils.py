# utils.py
from __future__ import annotations

from typing import Dict, Any

import torch


# ================================
# Utils
# ================================
def to_device_with_dtype(
    batch: Dict[str, Any],
    device: str,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """Move tensors to device; keep ints/bools in original dtype; cast floats to `dtype`."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if v.dtype in (
                torch.bool,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                out[k] = v.to(device, non_blocking=True)
            else:
                out[k] = v.to(device, dtype=dtype, non_blocking=True)
        else:
            out[k] = v
    return out
