from __future__ import annotations

import importlib.metadata
import json


def main() -> int:
    import mlx.core as mx

    eps = 1e-6
    hyper = mx.array([[3.0, 4.0, 0.0, 5.0]], dtype=mx.float32)
    branches = hyper.reshape(1, 2, 2)
    normalized = branches * mx.rsqrt(mx.mean(mx.square(branches), axis=-1, keepdims=True) + eps)
    normalized_flat = normalized.reshape(1, 4)
    down_weight = mx.zeros((1, 4), dtype=mx.float32)
    up_weight = mx.zeros((4, 1), dtype=mx.float32)
    inject_weight = mx.zeros((2, 4), dtype=mx.float32)
    down_input = (normalized_flat @ down_weight.T) / 2
    down = down_input * mx.sigmoid(down_input)
    mix = mx.sigmoid(down @ up_weight.T).reshape(1, 2, 2)
    mixed = mx.mean(mix * normalized, axis=1)
    injection = 2 * mx.sigmoid((normalized_flat @ inject_weight.T) / 2)

    query = mx.array([[0.0, 1.0]], dtype=mx.float32)
    keys = mx.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 4.0], [8.0, 0.0]],
        dtype=mx.float32,
    )
    pooled = mx.stack([mx.mean(keys[0:2], axis=0), mx.mean(keys[2:4], axis=0)])
    pooled = pooled * mx.rsqrt(mx.mean(mx.square(pooled), axis=-1, keepdims=True) + eps)
    scores = mx.sum(mx.maximum(query @ pooled.T, 0), axis=0) / (2**0.5)
    selected_block = int(mx.argmax(scores).item())
    selected = [selected_block * 2, selected_block * 2 + 1, 4]
    mx.eval(mixed, injection, scores)

    payload = {
        "schema_version": 1,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "gated_mixed_input": [float(value) for value in mixed[0].tolist()],
        "gated_injection_weights": [float(value) for value in injection[0].tolist()],
        "qsa_selected_tokens": selected,
        "fixture_tensor_bytes": 128,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
