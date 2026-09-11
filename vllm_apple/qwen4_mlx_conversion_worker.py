from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
import struct
from dataclasses import replace
from collections.abc import Iterable

from .qwen4_conversion_protocol import MAX_CONVERSION_REQUEST_BYTES
from .qwen4_conversion_worker import ConvertedTensorEvidence, execute_qwen4_conversion_request
from .numeric_formats import ScaledInt8Tensor, _scale
from .numeric_precision import NumericPrecisionPolicy


MAX_CORRECTNESS_TENSOR_BYTES = 16 * 1024 * 1024
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


class Qwen4MLXCorrectnessConverter:
    """Bounded one-shot converter for correctness evidence, not runtime residency."""

    def convert_scaled_int8(
        self, tensor: ScaledInt8Tensor, *, target_dtype: str, reserved_bytes: int,
        precision_policy: NumericPrecisionPolicy | None = None,
    ) -> ConvertedTensorEvidence:
        """CPU reference dequantization into the existing MLX correctness path.

        Reservation accounts for explicit numeric buffers, not Python object/RSS
        overhead. This does not enable native INT8 compute or model loading.
        """
        if not isinstance(tensor, ScaledInt8Tensor):
            raise ValueError("expected validated scaled INT8 tensor")
        if precision_policy is not None and not isinstance(precision_policy, NumericPrecisionPolicy):
            raise ValueError("invalid precision policy")
        if target_dtype not in _DTYPE_BYTES:
            raise ValueError("Qwen4 MLX correctness dtype is unsupported")
        if type(reserved_bytes) is not int or reserved_bytes < 0:
            raise ValueError("invalid conversion reservation")
        tensor = replace(tensor)  # Recheck plan, metadata and target digest at the boundary.
        elements = len(tensor.payload)
        output_bytes = elements * _DTYPE_BYTES[target_dtype]
        bridge_bytes = len(tensor.payload) + len(tensor.block_scales) + elements * 4
        worker_bytes = elements * 8 + output_bytes * 2
        if bridge_bytes + worker_bytes > reserved_bytes:
            raise MemoryError("scaled INT8 bridge exceeds its reservation")
        # Decode one value at a time to avoid a full Python float tuple.
        raw = bytearray(elements * 4)
        geometry = tensor.geometry
        expected_digest = hashlib.sha256()
        for index, value in enumerate(tensor.payload):
            scale_index = index // 16 if geometry is None else geometry.scale_index(index)
            decoded = (value if value < 128 else value - 256) / 2
            decoded *= _scale(tensor.block_scales[scale_index])
            decoded *= tensor.global_scale
            if precision_policy is not None:
                expected_digest.update(precision_policy.checked_bytes(decoded, target_dtype))
            try:
                struct.pack_into("<f", raw, index * 4, decoded)
            except (OverflowError, struct.error) as error:
                raise ValueError("scaled INT8 value exceeds F32 bridge range") from error
        evidence = self.convert(
            (raw,), source_dtype="F32", target_dtype=target_dtype,
            output_shape=geometry.shape if geometry is not None else (elements,),
            reserved_bytes=reserved_bytes - bridge_bytes,
        )
        if precision_policy is not None and evidence.output_digest != expected_digest.hexdigest():
            raise ValueError("backend output differs from precision reference")
        return evidence

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
        if not np.isfinite(source).all():
            raise ValueError("Qwen4 MLX correctness source must be finite")
        target = mx.array(source).astype(
            {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}[target_dtype]
        )
        mx.eval(target)
        bits_dtype = mx.uint16 if target_dtype in {"BF16", "F16"} else mx.uint32
        output = np.asarray(mx.view(target, bits_dtype))
        exponent_mask = {"F16": 0x7C00, "BF16": 0x7F80, "F32": 0x7F800000}[target_dtype]
        if np.any((output & exponent_mask) == exponent_mask):
            raise ValueError("Qwen4 MLX correctness output is nonfinite")
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
