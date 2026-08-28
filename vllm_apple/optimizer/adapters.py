from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Protocol

from ..model import InspectedModel
from .safety import validate_immutable_output_path
from .types import ArtifactManifest, OPTIMIZER_SCHEMA_VERSION

if TYPE_CHECKING:
    from .checkpoint import CheckpointStore
    from .worker import CancellationToken, IsolatedConversionWorker, WorkerResult


ADAPTER_API_VERSION = 1
MAX_REGISTERED_ADAPTERS = 32
MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
MLX_MIN_VERSION = (0, 26, 0)
MLX_MAX_VERSION_EXCLUSIVE = (0, 32, 0)
MLX_LM_MIN_VERSION = (0, 26, 0)
MLX_LM_MAX_VERSION_EXCLUSIVE = (0, 32, 0)


class AdapterUnavailableError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MLXExportInvocation:
    adapter_id: str
    implementation_version: str
    source_path: str
    output_path: str
    source_fingerprint: str
    target_weight_bits: int
    group_size: int
    estimated_output_bytes: int
    maximum_output_bytes: int
    command: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.adapter_id != "builtin.mlx-lm" or not self.implementation_version:
            raise ValueError("invalid MLX export adapter identity")
        if not Path(self.source_path).is_absolute() or not Path(self.output_path).is_absolute():
            raise ValueError("MLX export paths must be absolute")
        if len(self.source_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_fingerprint
        ):
            raise ValueError("invalid MLX export source fingerprint")
        if self.target_weight_bits not in {4, 8} or self.group_size not in {32, 64, 128}:
            raise ValueError("unsupported MLX affine quantization configuration")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (self.estimated_output_bytes, self.maximum_output_bytes)
        ):
            raise ValueError("MLX export byte budgets must be integers")
        if (
            self.estimated_output_bytes <= 0
            or self.maximum_output_bytes < self.estimated_output_bytes
        ):
            raise ValueError("MLX export byte budget is below the conservative estimate")
        if (
            not self.command
            or len(self.command) > 32
            or any(not isinstance(value, str) or not value for value in self.command)
        ):
            raise ValueError("invalid MLX export command")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPTIMIZER_SCHEMA_VERSION,
            "adapter_id": self.adapter_id,
            "implementation_version": self.implementation_version,
            "source_path": self.source_path,
            "output_path": self.output_path,
            "source_fingerprint": self.source_fingerprint,
            "target_weight_bits": self.target_weight_bits,
            "group_size": self.group_size,
            "estimated_output_bytes": self.estimated_output_bytes,
            "maximum_output_bytes": self.maximum_output_bytes,
            "command": list(self.command),
        }


@dataclass(frozen=True, slots=True)
class MLXExportReport:
    worker_result: "WorkerResult"
    artifact_manifest: ArtifactManifest
    manifest_path: str

    def __post_init__(self) -> None:
        if not Path(self.manifest_path).is_absolute():
            raise ValueError("artifact manifest path must be absolute")
        if self.worker_result.output_hash != self.artifact_manifest.output_hash:
            raise ValueError("worker result and artifact manifest hashes do not match")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPTIMIZER_SCHEMA_VERSION,
            "worker_result": self.worker_result.to_dict(),
            "artifact_manifest": self.artifact_manifest.to_dict(),
            "manifest_path": self.manifest_path,
        }


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
    _source_formats = ("safetensors",)
    _source_dtypes = ("float16", "bfloat16", "float32")

    def __init__(
        self,
        package_version: Callable[[str], str | None] | None = None,
        system_info: Callable[[], tuple[str, str]] | None = None,
        python_executable: str | None = None,
    ) -> None:
        self._package_version = package_version or _installed_package_version
        self._system_info = system_info or (lambda: (platform.system(), platform.machine()))
        self._python_executable = python_executable or sys.executable

    def detect(
        self,
        model: InspectedModel,
        source_format: str,
        source_dtype: str,
    ) -> AdapterCapability:
        versions = {
            "mlx": self._package_version("mlx"),
            "mlx-lm": self._package_version("mlx-lm"),
        }
        issues: list[str] = []
        for package, version in versions.items():
            if version is None:
                issues.append(f"dependency_missing:{package}")
        issues.extend(_mlx_version_issues(versions["mlx"], versions["mlx-lm"]))
        operating_system, architecture = self._system_info()
        if operating_system != "Darwin" or architecture not in {"arm64", "arm64e"}:
            issues.append(f"platform_unsupported:{operating_system}-{architecture}")
        executable_path = Path(self._python_executable).expanduser()
        if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
            issues.append("python_executable_unavailable")
        if source_format not in self._source_formats:
            issues.append(f"source_format_unsupported:{source_format}")
        if source_dtype not in self._source_dtypes:
            issues.append(f"source_dtype_unsupported:{source_dtype}")
        available = all(version is not None for version in versions.values())
        model_compatible = (
            source_format in self._source_formats and source_dtype in self._source_dtypes
        )
        execution_compatible = not issues
        return AdapterCapability(
            adapter_id=self.adapter_id,
            adapter_api_version=ADAPTER_API_VERSION,
            implementation_version=versions["mlx-lm"],
            available=available,
            compatible=model_compatible and execution_compatible,
            executable=available and model_compatible and execution_compatible,
            operations=("quantize_weights", "export_model"),
            source_formats=self._source_formats,
            output_formats=("mlx",),
            weight_bits=(4, 8),
            dependency_versions=versions,
            issues=tuple(issues),
        )

    def build_export_invocation(
        self,
        model: InspectedModel,
        output_path: Path,
        *,
        target_weight_bits: int,
        group_size: int,
        maximum_output_bytes: int,
        allow_existing_output: bool = False,
        require_executable: bool = True,
    ) -> MLXExportInvocation:
        source_format = detect_source_format(model.path)
        source_dtype = detect_source_dtype(model)
        capability = self.detect(model, source_format, source_dtype)
        if require_executable and (
            not capability.executable or capability.implementation_version is None
        ):
            detail = ",".join(capability.issues) or "adapter_not_executable"
            raise AdapterUnavailableError(detail)
        implementation_version = capability.implementation_version or "unavailable-dry-run"
        if (
            isinstance(target_weight_bits, bool)
            or isinstance(group_size, bool)
            or target_weight_bits not in {4, 8}
            or group_size not in {32, 64, 128}
        ):
            raise ValueError("unsupported MLX affine quantization configuration")
        if not isinstance(maximum_output_bytes, int) or isinstance(maximum_output_bytes, bool):
            raise ValueError("maximum output bytes must be an integer")
        safe_output = _validated_export_output(
            model.path,
            output_path,
            allow_existing=allow_existing_output,
        )
        estimated_output_bytes = math.ceil(
            model.memory_spec.weights_bytes * target_weight_bits / 16 * 1.10
        )
        if maximum_output_bytes < estimated_output_bytes:
            raise ValueError("maximum output bytes are below the conservative export estimate")
        source = model.path.expanduser().resolve(strict=True)
        command = (
            # Keep the virtual-environment launcher. Resolving this symlink can
            # select a base interpreter that cannot access the detected package.
            str(Path(self._python_executable).expanduser().absolute()),
            "-m",
            "mlx_lm",
            "convert",
            "--hf-path",
            str(source),
            "--mlx-path",
            "mlx-output",
            "--quantize",
            "--q-group-size",
            str(group_size),
            "--q-bits",
            str(target_weight_bits),
        )
        return MLXExportInvocation(
            adapter_id=self.adapter_id,
            implementation_version=implementation_version,
            source_path=str(source),
            output_path=str(safe_output),
            source_fingerprint=fingerprint_model_snapshot(model),
            target_weight_bits=target_weight_bits,
            group_size=group_size,
            estimated_output_bytes=estimated_output_bytes,
            maximum_output_bytes=maximum_output_bytes,
            command=command,
        )

    def execute_export(
        self,
        invocation: MLXExportInvocation,
        *,
        plan_id: str,
        worker: "IsolatedConversionWorker",
        checkpoint_store: "CheckpointStore",
        resume: bool = False,
        cancellation: "CancellationToken | None" = None,
        timeout_seconds: float | None = None,
    ) -> "WorkerResult":
        if invocation.adapter_id != self.adapter_id:
            raise ValueError("export invocation belongs to another adapter")
        return worker.run(
            plan_id=plan_id,
            source=Path(invocation.source_path),
            source_fingerprint=invocation.source_fingerprint,
            output=Path(invocation.output_path),
            command=invocation.command,
            maximum_output_bytes=invocation.maximum_output_bytes,
            checkpoint_store=checkpoint_store,
            resume=resume,
            cancellation=cancellation,
            timeout_seconds=timeout_seconds,
            produced_subdirectory="mlx-output",
        )

    def build_artifact_manifest(
        self,
        invocation: MLXExportInvocation,
        result: "WorkerResult",
        *,
        plan_id: str,
        license_name: str | None = None,
    ) -> ArtifactManifest:
        if (
            result.output_hash is None
            or result.output_path != invocation.output_path
            or result.plan_id != plan_id
        ):
            raise ValueError("completed MLX worker result does not match the export invocation")
        return ArtifactManifest(
            artifact_id=f"sha256-{result.output_hash[:16]}",
            created_at=datetime.now(timezone.utc).isoformat(),
            plan_id=plan_id,
            source_hash=invocation.source_fingerprint,
            output_hash=result.output_hash,
            output_bytes=result.output_bytes,
            elapsed_milliseconds=result.elapsed_milliseconds,
            peak_child_rss_bytes=result.peak_child_rss_bytes,
            transforms=(
                {
                    "type": "affine_quantization",
                    "backend": self.adapter_id,
                    "weight_bits": invocation.target_weight_bits,
                    "group_size": invocation.group_size,
                },
            ),
            tool_versions={"mlx-lm": invocation.implementation_version},
            calibration_fingerprint=None,
            evaluation={},
            license=license_name,
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
    configured = aliases.get(normalized, normalized)
    if configured != "unknown":
        return configured
    return _detect_safetensors_dtype(model.path)


def _detect_safetensors_dtype(model_path: Path) -> str:
    aliases = {"F16": "float16", "BF16": "bfloat16", "F32": "float32"}
    detected: set[str] = set()
    files = 0
    try:
        candidates = model_path.rglob("*.safetensors")
        for candidate in candidates:
            if not candidate.is_file():
                continue
            files += 1
            if files > 100_000:
                return "unknown"
            with candidate.open("rb", buffering=0) as handle:
                encoded_length = handle.read(8)
                if len(encoded_length) != 8:
                    return "unknown"
                header_length = int.from_bytes(encoded_length, "little")
                if not 2 <= header_length <= MAX_SAFETENSORS_HEADER_BYTES:
                    return "unknown"
                header = json.loads(handle.read(header_length))
            if not isinstance(header, dict):
                return "unknown"
            for name, metadata_value in header.items():
                if name == "__metadata__":
                    continue
                if not isinstance(metadata_value, dict):
                    return "unknown"
                dtype = metadata_value.get("dtype")
                if not isinstance(dtype, str):
                    return "unknown"
                detected.add(aliases.get(dtype, dtype.lower()))
                if len(detected) > 1:
                    return "+".join(sorted(detected))
    except (OSError, ValueError, json.JSONDecodeError):
        return "unknown"
    return next(iter(detected)) if detected else "unknown"


def fingerprint_model_snapshot(model: InspectedModel) -> str:
    files: list[Path] = []
    for candidate in model.path.rglob("*"):
        if not candidate.is_file():
            continue
        files.append(candidate)
        if len(files) > 100_000:
            raise ValueError("model snapshot exceeds the metadata file limit")
    if not files:
        raise ValueError("model snapshot is empty")
    digest = hashlib.sha256()
    header = {
        "model_id": model.model_id,
        "source_path": str(model.path.resolve(strict=True)),
        "config": model.config,
    }
    digest.update(
        json.dumps(header, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )
    for candidate in sorted(files, key=lambda path: str(path.relative_to(model.path))):
        relative = str(candidate.relative_to(model.path)).encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with candidate.open("rb", buffering=0) as handle:
            before = os.fstat(handle.fileno())
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_identity != after_identity:
            raise ValueError("model source changed while computing its fingerprint")
    return digest.hexdigest()


def _validated_export_output(source_path: Path, output_path: Path, *, allow_existing: bool) -> Path:
    if not allow_existing:
        return validate_immutable_output_path(source_path, output_path)
    source = source_path.expanduser().resolve(strict=True)
    output = output_path.expanduser().resolve(strict=False)
    if not output.exists():
        return validate_immutable_output_path(source, output)
    if output == Path(output.anchor) or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source and output paths must not overlap")
    info = output.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("resume output must be an owned real directory")
    return output


def persist_artifact_manifest(
    manifest: ArtifactManifest,
    path: Path,
    *,
    source_path: Path,
) -> tuple[Path, ArtifactManifest]:
    source = source_path.expanduser().resolve(strict=True)
    destination = path.expanduser().resolve(strict=False)
    try:
        overlaps = destination.is_relative_to(source) or source.is_relative_to(destination)
    except AttributeError:
        overlaps = source == destination or source in destination.parents or destination in source.parents
    if overlaps or destination == Path(destination.anchor):
        raise ValueError("source and artifact manifest paths must not overlap")
    if not destination.exists():
        destination = validate_immutable_output_path(source, destination)
    parent = destination.parent
    if not parent.is_dir():
        raise ValueError("artifact manifest parent must already exist")
    payload = json.dumps(
        manifest.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if destination.exists():
        info = destination.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
            or info.st_size > 1024 * 1024
        ):
            raise ValueError("existing artifact manifest is unsafe")
        existing = json.loads(destination.read_text(encoding="utf-8"))
        expected = manifest.to_dict()
        if not isinstance(existing, dict):
            raise ValueError("existing artifact manifest is malformed")
        expected["created_at"] = existing.get("created_at")
        if existing != expected:
            raise ValueError("existing artifact manifest does not match the artifact")
        return (
            destination.resolve(strict=True),
            replace(manifest, created_at=str(existing["created_at"])),
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        _fsync_directory(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination.resolve(strict=True), manifest


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mlx_version_issues(mlx_version: str | None, mlx_lm_version: str | None) -> list[str]:
    issues: list[str] = []
    parsed_mlx = _parse_release_version(mlx_version)
    parsed_mlx_lm = _parse_release_version(mlx_lm_version)
    if mlx_version is not None and parsed_mlx is None:
        issues.append(f"dependency_version_invalid:mlx:{mlx_version}")
    elif parsed_mlx is not None and not MLX_MIN_VERSION <= parsed_mlx < MLX_MAX_VERSION_EXCLUSIVE:
        issues.append(f"dependency_version_unsupported:mlx:{mlx_version}")
    if mlx_lm_version is not None and parsed_mlx_lm is None:
        issues.append(f"dependency_version_invalid:mlx-lm:{mlx_lm_version}")
    elif parsed_mlx_lm is not None and not (
        MLX_LM_MIN_VERSION <= parsed_mlx_lm < MLX_LM_MAX_VERSION_EXCLUSIVE
    ):
        issues.append(f"dependency_version_unsupported:mlx-lm:{mlx_lm_version}")
    if parsed_mlx is not None and parsed_mlx_lm is not None:
        if parsed_mlx[:2] < parsed_mlx_lm[:2]:
            issues.append(f"dependency_pair_incompatible:mlx-{mlx_version}:mlx-lm-{mlx_lm_version}")
    return issues


def _parse_release_version(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    parts = value.split(".")
    if len(parts) < 2 or len(parts) > 3 or any(not part.isdigit() for part in parts):
        return None
    numbers = tuple(int(part) for part in parts)
    return (numbers + (0, 0, 0))[:3]


def _installed_package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except (metadata.PackageNotFoundError, OSError, ValueError):
        return None
