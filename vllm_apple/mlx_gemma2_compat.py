"""Opt-in compatibility shim for a verified MLX-LM Gemma 2 implementation.

No upstream files are modified. The original attention computation is retained;
only the batch-aware shared-head mask gains the grouped-query axis.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from functools import wraps
from pathlib import Path
from typing import Any

GEMMA2_SOURCE_SHA256 = "64b0935b06fe2c4d5d4ed23a9cf62deb6218c55a88b9403a657afe9e2be8f251"


def grouped_query_mask(mask: Any, repeats: int) -> Any:
    if repeats > 1 and mask is not None and getattr(mask, "ndim", None) == 4:
        batch, heads, query, key = mask.shape
        if heads != 1:
            raise ValueError("Gemma 2 compatibility requires a shared-head attention mask")
        return mask.reshape(batch, 1, 1, query, key)
    return mask


def install_gemma2_batch_mask_fix() -> bool:
    """Apply once before model loading; reject unreviewed versions/source files."""
    if importlib.metadata.version("mlx-lm") != "0.32.0":
        raise ValueError("Gemma 2 batch mask fix requires reviewed MLX-LM 0.32.0")
    from mlx_lm.models import gemma2

    if hashlib.sha256(Path(gemma2.__file__).read_bytes()).hexdigest() != GEMMA2_SOURCE_SHA256:
        raise ValueError("Gemma 2 source changed; review the compatibility fix before use")
    original = gemma2.Attention.__call__
    if getattr(original, "_vllm_apple_gemma2_mask_fix", False):
        return False

    @wraps(original)
    def corrected(self: Any, x: Any, mask: Any = None, cache: Any = None) -> Any:
        return original(self, x, mask=grouped_query_mask(mask, self.repeats), cache=cache)

    corrected._vllm_apple_gemma2_mask_fix = True
    gemma2.Attention.__call__ = corrected
    return True


if __name__ == "__main__":
    # Explicit experimental entry point; use the upstream parser and lifecycle.
    # Never alter the installed package or implicitly promote managed serving.
    install_gemma2_batch_mask_fix()
    from mlx_lm.server import main

    main()
