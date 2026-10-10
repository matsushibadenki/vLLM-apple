"""Reuse immutable Gemma 2 RMSNorm weights, without capturing KV state.

The RMSNorm call follows MLX-LM (MIT), copyright Apple Inc.
P2 models are immutable after dtype conversion; reload to change weights.
"""
from __future__ import annotations

import hashlib
from functools import wraps
from pathlib import Path

from vllm_apple.mlx_gemma2_compat import GEMMA2_SOURCE_SHA256


def install_norm_weight_reuse(module, mx):
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != GEMMA2_SOURCE_SHA256:
        raise ValueError('RMSNorm reuse requires the reviewed Gemma 2 source')
    original = module.RMSNorm.__call__
    existing = getattr(original, '_vllm_p2_norm_reuse', None)
    if existing is not None:
        return existing
    metrics = dict(weights_built=0)

    @wraps(original)
    def call(self, x):
        # MLX Module keeps attributes in its dict, not in __dict__.
        weight = getattr(self, '_vllm_p2_norm_weight', None)
        if weight is None:
            weight = 1.0 + self.weight
            self._vllm_p2_norm_weight = weight
            metrics['weights_built'] += 1
        return mx.fast.rms_norm(x, weight, self.eps)

    call._vllm_p2_norm_reuse = metrics
    module.RMSNorm.__call__ = call
    return metrics
