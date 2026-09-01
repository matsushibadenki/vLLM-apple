from __future__ import annotations

import math
from collections.abc import Sequence


MAX_REFERENCE_WIDTH = 4096
MAX_REFERENCE_TOKENS = 8192


def _sigmoid(value: float) -> float:
    if value >= 0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


def _silu(value: float) -> float:
    return value * _sigmoid(value)


def _linear(values: Sequence[float], weights: Sequence[Sequence[float]]) -> list[float]:
    if any(len(row) != len(values) for row in weights):
        raise ValueError("reference linear weight shape is invalid")
    return [sum(float(weight) * float(value) for weight, value in zip(row, values)) for row in weights]


def qwen4_gated_residual_reference(
    hyper_input: Sequence[float],
    *,
    hc_count: int,
    hidden_size: int,
    norm_weight: Sequence[float],
    mix_down_weight: Sequence[Sequence[float]],
    mix_up_weight: Sequence[Sequence[float]],
    inject_weight: Sequence[Sequence[float]] | None,
    eps: float = 1e-6,
) -> dict[str, list[float] | None]:
    """Small float64 reference matching Transformers Qwen4-Exp gated residual."""
    width = hc_count * hidden_size
    if not 1 <= hc_count <= 64 or not 1 <= hidden_size <= MAX_REFERENCE_WIDTH:
        raise ValueError("reference gated residual dimensions are invalid")
    if len(hyper_input) != width or len(norm_weight) != width:
        raise ValueError("reference gated residual input shape is invalid")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("reference gated residual epsilon is invalid")
    normalized: list[float] = []
    for branch in range(hc_count):
        start = branch * hidden_size
        values = [float(value) for value in hyper_input[start : start + hidden_size]]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("reference gated residual input must be finite")
        scale = 1.0 / math.sqrt(sum(value * value for value in values) / hidden_size + eps)
        normalized.extend(
            value * scale * (1.0 + float(norm_weight[start + offset]))
            for offset, value in enumerate(values)
        )
    down = [_silu(value / hc_count) for value in _linear(normalized, mix_down_weight)]
    mix = [_sigmoid(value) for value in _linear(down, mix_up_weight)]
    if len(mix) != width:
        raise ValueError("reference gated residual mix projection shape is invalid")
    mixed = [
        sum(
            mix[branch * hidden_size + column]
            * normalized[branch * hidden_size + column]
            for branch in range(hc_count)
        )
        / hc_count
        for column in range(hidden_size)
    ]
    injection = None
    if inject_weight is not None:
        injection = [2.0 * _sigmoid(value / hc_count) for value in _linear(normalized, inject_weight)]
        if len(injection) != hc_count:
            raise ValueError("reference gated residual injection projection shape is invalid")
    return {
        "mixed_input": mixed,
        "hyper_input": list(map(float, hyper_input)) if inject_weight is not None else None,
        "injection_weights": injection,
    }


def qwen4_qsa_select_tokens_reference(
    query_heads: Sequence[Sequence[float]],
    raw_keys: Sequence[Sequence[float]],
    visible_token_indices: Sequence[int],
    *,
    compress_ratio: int,
    token_budget: int,
    eps: float = 1e-6,
) -> list[int]:
    """QSA selector fixture with normalized queries, zero-weight key RMSNorm, and identity RoPE."""
    if not query_heads or not raw_keys:
        raise ValueError("reference QSA requires query heads and keys")
    head_dim = len(query_heads[0])
    if not 1 <= head_dim <= MAX_REFERENCE_WIDTH:
        raise ValueError("reference QSA head dimension is invalid")
    if any(len(head) != head_dim for head in query_heads) or any(
        len(key) != head_dim for key in raw_keys
    ):
        raise ValueError("reference QSA vector shape is invalid")
    if not 1 <= compress_ratio <= MAX_REFERENCE_TOKENS or not 1 <= token_budget <= MAX_REFERENCE_TOKENS:
        raise ValueError("reference QSA budget is invalid")
    if token_budget < compress_ratio:
        raise ValueError("reference QSA budget must include at least one complete block")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("reference QSA epsilon is invalid")
    visible = [int(index) for index in visible_token_indices]
    if len(visible) > MAX_REFERENCE_TOKENS or visible != sorted(set(visible)):
        raise ValueError("reference QSA visible indices are invalid")
    if visible and (visible[0] < 0 or visible[-1] >= len(raw_keys)):
        raise ValueError("reference QSA visible index is outside the key range")
    complete = len(visible) // compress_ratio
    blocks = [
        visible[index * compress_ratio : (index + 1) * compress_ratio]
        for index in range(complete)
    ]
    scored: list[tuple[float, int]] = []
    for block_index, block in enumerate(blocks):
        pooled = [
            sum(float(raw_keys[token][column]) for token in block) / compress_ratio
            for column in range(head_dim)
        ]
        key_scale = 1.0 / math.sqrt(
            sum(value * value for value in pooled) / head_dim + eps
        )
        pooled = [value * key_scale for value in pooled]
        score = sum(
            max(0.0, sum(float(q) * key for q, key in zip(head, pooled)))
            for head in query_heads
        ) / math.sqrt(head_dim)
        scored.append((score, block_index))
    block_topk = token_budget // compress_ratio
    selected_blocks = sorted(scored, key=lambda item: (-item[0], item[1]))[:block_topk]
    selected = [token for _, block_index in selected_blocks for token in blocks[block_index]]
    selected.extend(visible[complete * compress_ratio :])
    return selected
