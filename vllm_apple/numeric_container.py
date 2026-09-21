"""Strict metadata adapters separating containers, recipes and compute formats."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum

from .numeric_routing import NumericFormat


class NumericContainer(str, Enum):
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    MLX = "mlx"
    COREML = "coreml"


class QuantizationRecipe(str, Enum):
    NONE = "none"
    GPTQ = "gptq"
    AWQ = "awq"
    MLX_AFFINE = "mlx_affine"
    COREML_LINEAR = "coreml_linear"
    GGUF_Q4_0 = "gguf_q4_0"
    GGUF_Q4_K = "gguf_q4_k"
    GGUF_Q8_0 = "gguf_q8_0"


@dataclass(frozen=True, slots=True)
class NumericArtifactDescriptor:
    container: NumericContainer
    recipe: QuantizationRecipe
    storage_format: NumericFormat
    compute_format: NumericFormat
    bits: int
    group_size: int | None
    packing: str
    exporter: str

    def __post_init__(self) -> None:
        if (not isinstance(self.container, NumericContainer)
                or not isinstance(self.recipe, QuantizationRecipe)
                or not isinstance(self.storage_format, NumericFormat)
                or not isinstance(self.compute_format, NumericFormat)
                or self.bits not in {2, 4, 8, 16, 32}
                or (self.group_size is not None
                    and (type(self.group_size) is not int
                         or not 1 <= self.group_size <= 1 << 20))
                or not _identifier(self.packing) or not _identifier(self.exporter)):
            raise ValueError("invalid numeric artifact descriptor")

    @property
    def descriptor_id(self) -> str:
        value = asdict(self)
        for key in ("container", "recipe", "storage_format", "compute_format"):
            value[key] = value[key].value
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def inspect_numeric_artifact_metadata(
    container: NumericContainer, metadata: dict[str, object]
) -> NumericArtifactDescriptor:
    if not isinstance(container, NumericContainer) or not isinstance(metadata, dict):
        raise ValueError("invalid numeric artifact metadata")
    if container is NumericContainer.GGUF:
        return _gguf(metadata)
    method = metadata.get("quant_method")
    if method in {"gptq", "awq"}:
        return _gptq_awq(container, metadata, str(method))
    if container is NumericContainer.MLX and method == "affine":
        return _mlx(metadata)
    if container is NumericContainer.COREML and method == "linear":
        return _coreml(metadata)
    if method == "none":
        dtype = _format(metadata.get("dtype"))
        bits = {NumericFormat.FP16: 16, NumericFormat.BF16: 16,
                NumericFormat.FP32: 32}.get(dtype)
        if bits is None or set(metadata) != {"quant_method", "dtype", "exporter"}:
            raise ValueError("unsupported unquantized artifact metadata")
        return NumericArtifactDescriptor(
            container, QuantizationRecipe.NONE, dtype, dtype, bits, None,
            "native", _required_identifier(metadata, "exporter"),
        )
    raise ValueError("unsupported numeric artifact recipe")


def _gptq_awq(
    container: NumericContainer, metadata: dict[str, object], method: str
) -> NumericArtifactDescriptor:
    if container is not NumericContainer.SAFETENSORS or set(metadata) != {
        "quant_method", "bits", "group_size", "packing", "compute_dtype", "exporter"
    }:
        raise ValueError("invalid GPTQ/AWQ metadata")
    bits = metadata["bits"]
    group_size = metadata["group_size"]
    if type(bits) is not int or bits not in {2, 4, 8}:
        raise ValueError("unsupported GPTQ/AWQ bit width")
    storage = {2: NumericFormat.INT2, 4: NumericFormat.INT4, 8: NumericFormat.INT8}[bits]
    return NumericArtifactDescriptor(
        container, QuantizationRecipe(method), storage,
        _format(metadata["compute_dtype"]), bits, _group(group_size),
        _required_identifier(metadata, "packing"),
        _required_identifier(metadata, "exporter"),
    )


def _mlx(metadata: dict[str, object]) -> NumericArtifactDescriptor:
    if set(metadata) != {
        "quant_method", "bits", "group_size", "compute_dtype", "exporter"
    }:
        raise ValueError("invalid MLX affine metadata")
    bits = metadata["bits"]
    if type(bits) is not int or bits not in {2, 4, 8}:
        raise ValueError("unsupported MLX affine bit width")
    return NumericArtifactDescriptor(
        NumericContainer.MLX, QuantizationRecipe.MLX_AFFINE,
        {2: NumericFormat.INT2, 4: NumericFormat.INT4, 8: NumericFormat.INT8}[bits],
        _format(metadata["compute_dtype"]), bits, _group(metadata["group_size"]),
        "mlx_affine", _required_identifier(metadata, "exporter"),
    )


def _coreml(metadata: dict[str, object]) -> NumericArtifactDescriptor:
    if set(metadata) != {"quant_method", "bits", "compute_dtype", "exporter"}:
        raise ValueError("invalid Core ML linear metadata")
    bits = metadata["bits"]
    if type(bits) is not int or bits not in {4, 8}:
        raise ValueError("unsupported Core ML linear bit width")
    return NumericArtifactDescriptor(
        NumericContainer.COREML, QuantizationRecipe.COREML_LINEAR,
        {4: NumericFormat.INT4, 8: NumericFormat.INT8}[bits],
        _format(metadata["compute_dtype"]), bits, None, "coreml_linear",
        _required_identifier(metadata, "exporter"),
    )


def _gguf(metadata: dict[str, object]) -> NumericArtifactDescriptor:
    if set(metadata) != {"ggml_type", "compute_dtype", "exporter"}:
        raise ValueError("invalid GGUF metadata")
    mapping = {
        "Q4_0": (QuantizationRecipe.GGUF_Q4_0, NumericFormat.INT4, 32, "gguf_q4_0"),
        "Q4_K": (QuantizationRecipe.GGUF_Q4_K, NumericFormat.INT4, 256, "gguf_q4_k"),
        "Q8_0": (QuantizationRecipe.GGUF_Q8_0, NumericFormat.INT8, 32, "gguf_q8_0"),
    }
    ggml_type = metadata["ggml_type"]
    selected = mapping.get(ggml_type) if isinstance(ggml_type, str) else None
    if selected is None:
        raise ValueError("unsupported GGUF tensor type")
    recipe, storage, group, packing = selected
    return NumericArtifactDescriptor(
        NumericContainer.GGUF, recipe, storage, _format(metadata["compute_dtype"]),
        4 if storage is NumericFormat.INT4 else 8, group, packing,
        _required_identifier(metadata, "exporter"),
    )


def _format(value: object) -> NumericFormat:
    aliases = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}
    if isinstance(value, str):
        value = aliases.get(value, value)
    try:
        result = NumericFormat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("unsupported numeric compute format") from error
    if result not in {NumericFormat.FP16, NumericFormat.BF16, NumericFormat.FP32}:
        raise ValueError("unsupported numeric compute format")
    return result


def _group(value: object) -> int:
    if type(value) is not int or not 1 <= value <= 1 << 20:
        raise ValueError("invalid numeric quantization group size")
    return value


def _required_identifier(metadata: dict[str, object], key: str) -> str:
    value = metadata.get(key)
    if not _identifier(value):
        raise ValueError(f"invalid numeric artifact {key}")
    return value


def _identifier(value: object) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 128 and all(
        character.isalnum() or character in "._-" for character in value
    )
