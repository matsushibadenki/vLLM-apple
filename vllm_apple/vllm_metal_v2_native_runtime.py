from __future__ import annotations

import hashlib
import json
import math
import time

from .vllm_metal_v2_adapter import (
    VLLM_METAL_V2_MEASUREMENT_ABI_VERSION,
    parse_v2_measurement_request,
)
from .vllm_metal_v2_tuning import V2PagedAttentionFamily

MAX_NATIVE_FIXTURE_BYTES = 256 * 1024 * 1024


def measure(request_json: str) -> str:
    """Run one forced-family native fixture inside the isolated helper process."""
    if not isinstance(request_json, str) or len(request_json.encode("utf-8")) > 16 * 1024:
        raise ValueError("native v2 request is invalid or oversized")
    payload = json.loads(request_json)
    shape, configuration = parse_v2_measurement_request(payload)
    if (
        shape.turboquant
        or shape.query_dtype not in {"float16", "bfloat16"}
        or shape.cache_dtype != shape.query_dtype
    ):
        raise ValueError(
            "native v2 fixture requires matching non-TurboQuant float16 or bfloat16"
        )
    if shape.sequences != 1:
        raise ValueError("native v2 fixture currently requires one sequence")
    blocks = math.ceil(shape.context_tokens / shape.block_size)
    cache_elements = blocks * shape.block_size * shape.kv_heads * shape.head_size * 2
    query_elements = shape.query_tokens * shape.query_heads * shape.head_size
    estimated_bytes = 2 * (cache_elements + query_elements * 2)
    if estimated_bytes > MAX_NATIVE_FIXTURE_BYTES:
        raise ValueError("native v2 fixture exceeds the 256 MiB allocation limit")

    import mlx.core as mx
    import numpy as np
    from vllm_metal.metal import get_ops

    ops = get_ops()
    if configuration.family is V2PagedAttentionFamily.NAX_PREFILL and not ops.nax_ready():
        raise RuntimeError("native v2 NAX family is unavailable on this device")

    query_data = np.zeros(
        (shape.query_tokens, shape.query_heads, shape.head_size), dtype=np.float16
    )
    key_data = np.zeros(
        (blocks, shape.block_size, shape.kv_heads, shape.head_size), dtype=np.float16
    )
    value_vector = (
        np.arange(shape.kv_heads * shape.head_size, dtype=np.float32)
        .reshape(shape.kv_heads, shape.head_size)
        % 31
    ) / 31.0
    value_data = np.broadcast_to(
        value_vector.astype(np.float16),
        (blocks, shape.block_size, shape.kv_heads, shape.head_size),
    ).copy()
    expected = np.empty_like(query_data)
    heads_per_kv = shape.query_heads // shape.kv_heads
    for head in range(shape.query_heads):
        expected[:, head, :] = value_vector[head // heads_per_kv]

    mlx_dtype = mx.float16 if shape.query_dtype == "float16" else mx.bfloat16
    query = mx.array(query_data).astype(mlx_dtype)
    key_cache = mx.array(key_data).astype(mlx_dtype)
    value_cache = mx.array(value_data).astype(mlx_dtype)
    block_tables = mx.array(np.arange(blocks, dtype=np.int32).reshape(1, blocks))
    seq_lens = mx.array(np.array([shape.context_tokens], dtype=np.int32))
    cu_seqlens_q = mx.array(np.array([0, shape.query_tokens], dtype=np.int32))

    def run_once():
        output = mx.zeros(query.shape, dtype=query.dtype)
        ops.paged_attention_primitive(
            query,
            key_cache,
            value_cache,
            shape.kv_heads,
            1.0 / math.sqrt(shape.head_size),
            0.0,
            block_tables,
            seq_lens,
            cu_seqlens_q,
            shape.block_size,
            shape.context_tokens,
            -1,
            output,
            window_seqlen_q=shape.window_seqlen_q,
            vllm_apple_family=configuration.family.value,
        )
        mx.eval(output)
        return output

    run_once()
    started = time.perf_counter_ns()
    output = run_once()
    latency = max(1, time.perf_counter_ns() - started)
    # MLX bfloat16 intentionally has no NumPy dtype/buffer representation.
    # Convert on-device to float32 before exposing the bounded result buffer.
    actual = np.asarray(output.astype(mx.float32))
    passed = bool(np.allclose(actual, expected, rtol=2e-2, atol=2e-2))
    digest = hashlib.sha256(expected.astype("<f4", copy=False).tobytes()).hexdigest()
    return json.dumps(
        {
            "abi_version": VLLM_METAL_V2_MEASUREMENT_ABI_VERSION,
            "passed": passed,
            "latency_nanoseconds": latency,
            "output_digest": digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
