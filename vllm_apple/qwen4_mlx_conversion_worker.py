from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from collections.abc import Iterable

from .qwen4_conversion_protocol import MAX_CONVERSION_REQUEST_BYTES
from .qwen4_conversion_worker import ConvertedTensorEvidence, execute_qwen4_conversion_request


MAX_CORRECTNESS_TENSOR_BYTES = 16 * 1024 * 1024
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


class Qwen4MLXCorrectnessConverter:
    """Bounded one-shot converter for correctness evidence, not runtime residency."""

    def convert(
        self,
        chunks: Iterable[bytes],
        *,
        source_dtype: str,
        target_dtype: str,
        output_shape: tuple[int, ...],
        reserved_bytes: int,
    ) -> ConvertedTensorEvidence:
        if source_dtype not in _DTYPE_BYTES or target_dtype not in _DTYPE_BYTES:
            raise ValueError("Qwen4 MLX correctness dtype is unsupported")
        raw = bytearray()
        for chunk in chunks:
            raw.extend(chunk)
            if len(raw) > MAX_CORRECTNESS_TENSOR_BYTES:
                raise ValueError("Qwen4 MLX correctness tensor exceeds the bounded limit")
        elements = 1
        for dimension in output_shape:
            elements *= dimension
        if len(raw) != elements * _DTYPE_BYTES[source_dtype]:
            raise ValueError("Qwen4 MLX correctness source bytes do not match the shape")
        output_bytes = elements * _DTYPE_BYTES[target_dtype]
        if output_bytes > MAX_CORRECTNESS_TENSOR_BYTES:
            raise ValueError("Qwen4 MLX correctness output exceeds the bounded limit")
        required_peak = len(raw) + elements * 4 + output_bytes * 2
        if required_peak > reserved_bytes:
            raise MemoryError("Qwen4 MLX correctness conversion exceeds its reservation")

        import mlx.core as mx
        import numpy as np

        if source_dtype == "BF16":
            words = np.frombuffer(raw, dtype=np.uint16)
            source = (words.astype(np.uint32) << 16).view(np.float32)
        elif source_dtype == "F16":
            source = np.frombuffer(raw, dtype=np.float16)
        else:
            source = np.frombuffer(raw, dtype=np.float32)
        source = source.reshape(output_shape)
        target = mx.array(source).astype(
            {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}[target_dtype]
        )
        mx.eval(target)
        bits_dtype = mx.uint16 if target_dtype in {"BF16", "F16"} else mx.uint32
        output = np.asarray(mx.view(target, bits_dtype))
        digest = hashlib.sha256(output.tobytes(order="C")).hexdigest()
        return ConvertedTensorEvidence(
            backend="mlx",
            backend_version=importlib.metadata.version("mlx"),
            output_shape=output_shape,
            output_bytes=output_bytes,
            output_digest=digest,
        )


def main() -> int:
    raw = sys.stdin.buffer.read(MAX_CONVERSION_REQUEST_BYTES + 1)
    if not 1 <= len(raw) <= MAX_CONVERSION_REQUEST_BYTES:
        return 2
    try:
        request = json.loads(raw)
        response = execute_qwen4_conversion_request(request, Qwen4MLXCorrectnessConverter())
    except (ImportError, MemoryError, OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return 2
    sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
