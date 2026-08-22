from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable, Iterable, Protocol

from ..model import InspectedModel
from .types import OPTIMIZER_SCHEMA_VERSION


ADAPTER_API_VERSION = 1
MAX_REGISTERED_ADAPTERS = 32


@dataclass(frozen=True, slots=True)
class AdapterCapability:
    adapter_id: str
    adapter_api_version: int
    implementation_version: str | None
    available: bool
    compatible: bool
    executable: bool
    operations: tuple[str, ...]
    source_formats: tuple[str, ...]
    output_formats: tuple[str, ...]
    weight_bits: tuple[int, ...]
    dependency_versions: dict[str, str | None]
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.adapter_id or self.adapter_api_version != ADAPTER_API_VERSION:
            raise ValueError("invalid optimization adapter identity")
        required_collections = (self.operations, self.source_formats, self.output_formats)
        if any(
            not values or len(values) > 32 or any(not value for value in values)
            for values in required_collections
        ):
            raise ValueError("adapter capability collections must be bounded and non-empty")
        if len(self.issues) > 32 or any(not issue for issue in self.issues):
            raise ValueError("adapter issues must be bounded and non-empty")
        if not self.weight_bits or len(self.weight_bits) > 16:
            raise ValueError("adapter weight precision must be bounded and non-empty")
        if len(self.dependency_versions) > 16 or any(
            not name or (version is not None and not version)
            for name, version in self.dependency_versions.items()
        ):
            raise ValueError("invalid adapter dependency versions")
        if any(bits not in {2, 3, 4, 5, 6, 8, 16} for bits in self.weight_bits):
            raise ValueError("unsupported adapter weight precision")
        if self.executable and not (self.available and self.compatible):
            raise ValueError("an executable adapter must be available and compatible")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for field in ("operations", "source_formats", "output_formats", "weight_bits", "issues"):
            payload[field] = list(payload[field])
        return payload


@dataclass(frozen=True, slots=True)
class AdapterCapabilityReport:
    model_id: str
    source_format: str
    source_dtype: str
    adapters: tuple[AdapterCapability, ...]

    def __post_init__(self) -> None:
        if not self.model_id or not self.source_format or not self.source_dtype:
            raise ValueError("invalid adapter capability report source")
        if not self.adapters or len(self.adapters) > MAX_REGISTERED_ADAPTERS:
            raise ValueError("adapter capability report must be bounded and non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPTIMIZER_SCHEMA_VERSION,
            "model": {
                "model_id": self.model_id,
                "source_format": self.source_format,
                "source_dtype": self.source_dtype,
            },
            "adapters": [adapter.to_dict() for adapter in self.adapters],
        }


class OptimizationAdapter(Protocol):
    @property
    def adapter_id(self) -> str: ...

    def detect(
        self,
        model: InspectedModel,
        source_format: str,
        source_dtype: str,
    ) -> AdapterCapability: ...


class MLXOptimizationAdapter:
    adapter_id = "builtin.mlx-lm"
    _source_formats = ("safetensors", "pytorch-bin")
    _source_dtypes = ("float16", "bfloat16", "float32")

    def __init__(self, package_version: Callable[[str], str | None] | None = None) -> None:
        self._package_version = package_version or _installed_package_version

    def detect(
        self,
        model: InspectedModel,
        source_format: str,
        source_dtype: str,
    ) -> AdapterCapability:
        del model
        versions = {
            "mlx": self._package_version("mlx"),
            "mlx-lm": self._package_version("mlx-lm"),
        }
        issues: list[str] = []
        for package, version in versions.items():
            if version is None:
                issues.append(f"dependency_missing:{package}")
        if source_format not in self._source_formats:
            issues.append(f"source_format_unsupported:{source_format}")
        if source_dtype not in self._source_dtypes:
            issues.append(f"source_dtype_unsupported:{source_dtype}")
        # Execution is intentionally disabled until the isolated O1 worker and
        # atomic artifact promotion are implemented.
        issues.append("adapter_execution_not_implemented")
        return AdapterCapability(
            adapter_id=self.adapter_id,
            adapter_api_version=ADAPTER_API_VERSION,
            implementation_version=versions["mlx-lm"],
            available=all(version is not None for version in versions.values()),
            compatible=(
                source_format in self._source_formats and source_dtype in self._source_dtypes
            ),
            executable=False,
            operations=("quantize_weights", "export_model"),
            source_formats=self._source_formats,
            output_formats=("mlx",),
            weight_bits=(4, 8, 16),
            dependency_versions=versions,
            issues=tuple(issues),
        )


class AdapterRegistry:
    def __init__(self, adapters: Iterable[OptimizationAdapter]) -> None:
        registered = tuple(adapters)
        identifiers = [adapter.adapter_id for adapter in registered]
        if not registered or len(registered) > MAX_REGISTERED_ADAPTERS:
            raise ValueError("adapter registry must be bounded and non-empty")
        if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(
            identifiers
        ):
            raise ValueError("adapter identifiers must be non-empty and unique")
        self._adapters = registered

    def detect(self, model: InspectedModel) -> AdapterCapabilityReport:
        source_format = detect_source_format(model.path)
        source_dtype = detect_source_dtype(model)
        capabilities = tuple(
            adapter.detect(model, source_format, source_dtype)
            for adapter in sorted(self._adapters, key=lambda item: item.adapter_id)
        )
        return AdapterCapabilityReport(
            model_id=model.model_id,
            source_format=source_format,
            source_dtype=source_dtype,
            adapters=capabilities,
        )


def builtin_adapter_registry() -> AdapterRegistry:
    return AdapterRegistry((MLXOptimizationAdapter(),))


def detect_source_format(model_path: Path) -> str:
    formats: set[str] = set()
    suffix_formats = {
        ".safetensors": "safetensors",
        ".gguf": "gguf",
        ".bin": "pytorch-bin",
    }
    for candidate in model_path.rglob("*"):
        if candidate.is_file() and candidate.suffix in suffix_formats:
            formats.add(suffix_formats[candidate.suffix])
            if len(formats) == len(suffix_formats):
                break
    if not formats:
        return "unknown"
    return "+".join(sorted(formats))


def detect_source_dtype(model: InspectedModel) -> str:
    value = model.config.get("torch_dtype") or model.config.get("dtype")
    normalized = str(value).lower() if value is not None else "unknown"
    aliases = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
    return aliases.get(normalized, normalized)


def _installed_package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except (metadata.PackageNotFoundError, OSError, ValueError):
        return None
