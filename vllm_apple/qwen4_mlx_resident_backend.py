"""Bounded MLX residency for validated scaled-INT8 compatibility tensors."""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
import struct
from dataclasses import dataclass, replace

from .numeric_formats import ScaledInt8Tensor, _scale
from .numeric_precision import PrecisionExecutionContract
from .qwen4_resident_store import ResidentBackendAllocation


@dataclass(slots=True)
class MLXResidentTensorResource:
    """Owns the sole backend reference; release makes the resource unusable."""

    array: object | None

    @property
    def released(self) -> bool:
        return self.array is None


class Qwen4MLXNumericResidentBackend:
    """Correctness-first MLX backend; native scaled-INT8 compute is not implied."""

    def load(self, *args: object, **kwargs: object) -> ResidentBackendAllocation:
        raise ValueError("this backend only accepts validated scaled INT8 tensors")

    def load_scaled_int8(
        self,
        tensor: ScaledInt8Tensor,
        *,
        target_dtype: str,
        execution_contract: PrecisionExecutionContract,
        reserved_bytes: int,
    ) -> ResidentBackendAllocation:
        if not isinstance(tensor, ScaledInt8Tensor):
            raise ValueError("MLX resident numeric tensor is invalid")
        tensor = replace(tensor)
        if (
            not isinstance(execution_contract, PrecisionExecutionContract)
            or execution_contract.tensor_digest != tensor.target_digest
            or execution_contract.target_dtype != target_dtype
        ):
            raise ValueError("MLX resident precision contract mismatch")
        if type(reserved_bytes) is not int or reserved_bytes < 0:
            raise ValueError("MLX resident reservation is invalid")
        dtype_bytes = {"F16": 2, "BF16": 2, "F32": 4}
        if target_dtype not in dtype_bytes:
            raise ValueError("MLX resident target dtype is unsupported")
        elements = len(tensor.payload)
        output_bytes = elements * dtype_bytes[target_dtype]
        required_bytes = (
            len(tensor.payload) + len(tensor.block_scales)
            + elements * 4 + output_bytes * 2
        )
        if required_bytes > reserved_bytes:
            raise MemoryError("MLX resident numeric conversion exceeds its reservation")

        import mlx.core as mx
        import numpy as np

        source = bytearray(elements * 4)
        expected_digest = hashlib.sha256()
        geometry = tensor.geometry
        for index, value in enumerate(tensor.payload):
            scale_index = index // 16 if geometry is None else geometry.scale_index(index)
            decoded = (value if value < 128 else value - 256) / 2
            decoded *= _scale(tensor.block_scales[scale_index]) * tensor.global_scale
            expected_digest.update(execution_contract.policy.checked_bytes(decoded, target_dtype))
            try:
                struct.pack_into("<f", source, index * 4, decoded)
            except (OverflowError, struct.error) as error:
                raise ValueError("scaled INT8 value exceeds F32 resident bridge range") from error

        shape = geometry.shape if geometry is not None else (elements,)
        array = mx.array(np.frombuffer(source, dtype=np.float32).reshape(shape)).astype(
            {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}[target_dtype]
        )
        mx.eval(array)
        bits_dtype = mx.uint16 if target_dtype in {"BF16", "F16"} else mx.uint32
        output = np.asarray(mx.view(array, bits_dtype))
        exponent_mask = {"F16": 0x7C00, "BF16": 0x7F80, "F32": 0x7F800000}[target_dtype]
        if np.any((output & exponent_mask) == exponent_mask):
            raise ValueError("MLX resident numeric output is nonfinite")
        output_digest = hashlib.sha256(output.tobytes(order="C")).hexdigest()
        if output_digest != expected_digest.hexdigest():
            raise ValueError("MLX resident output differs from precision contract")
        if not math.prod(shape) == elements or output.nbytes != output_bytes:
            raise ValueError("MLX resident output metadata mismatch")
        return ResidentBackendAllocation(
            resource=MLXResidentTensorResource(array),
            backend="mlx",
            backend_version=importlib.metadata.version("mlx"),
            output_shape=shape,
            output_bytes=output_bytes,
            output_digest=output_digest,
            precision_contract_id=execution_contract.contract_id,
            precision_policy_id=execution_contract.policy.policy_id,
            precision_checked=True,
        )

    def release(self, resource: object) -> None:
        if not isinstance(resource, MLXResidentTensorResource) or resource.released:
            raise ValueError("MLX resident resource is invalid or already released")
        resource.array = None
