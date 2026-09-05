from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .generative_qualification import GenerativeArtifactComponent


MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_FILES = 4096
IGNORED_DIRECTORIES = frozenset({".git", ".cache", "__pycache__"})
COMPONENT_ROLES = {
    "transformer": "denoiser",
    "text_encoder": "text_encoder",
    "tokenizer": "text_encoder",
    "vae": "vae",
    "scheduler": "vae",
}


class GenerativeArtifactInspectionError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        stat = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    if path.is_symlink() or not path.is_file() or stat.st_size > MAX_METADATA_BYTES:
        raise GenerativeArtifactInspectionError(f"unsafe or oversized metadata: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GenerativeArtifactInspectionError(f"invalid metadata: {path.name}") from error
    if not isinstance(value, dict):
        raise GenerativeArtifactInspectionError(f"metadata root must be an object: {path.name}")
    return value


def _read_model_card_metadata(path: Path) -> dict[str, str]:
    try:
        info = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return {}
    if path.is_symlink() or not path.is_file() or info.st_size > MAX_METADATA_BYTES:
        raise GenerativeArtifactInspectionError("unsafe or oversized metadata: README.md")
    try:
        with path.open(encoding="utf-8") as handle:
            if handle.readline().strip() != "---":
                return {}
            metadata: dict[str, str] = {}
            for line_number, line in enumerate(handle, start=1):
                if line_number > 128:
                    raise GenerativeArtifactInspectionError("model card front matter is oversized")
                if line.strip() == "---":
                    return metadata
                key, separator, value = line.partition(":")
                if separator and key in {"library_name", "license", "base_model"}:
                    metadata[key] = value.strip().strip('"\'')[:512]
    except (OSError, UnicodeError) as error:
        raise GenerativeArtifactInspectionError("invalid metadata: README.md") from error
    raise GenerativeArtifactInspectionError("unterminated model card front matter")


def _bounded_sizes(root: Path) -> tuple[dict[str, int], int]:
    sizes: dict[str, int] = {}
    file_count = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise GenerativeArtifactInspectionError(f"cannot scan artifact: {directory}") from error
        for entry in entries:
            if entry.name in IGNORED_DIRECTORIES:
                continue
            if entry.is_symlink():
                raise GenerativeArtifactInspectionError(f"artifact must not contain symlinks: {entry.name}")
            if entry.is_dir(follow_symlinks=False):
                stack.append(Path(entry.path))
                continue
            if not entry.is_file(follow_symlinks=False):
                continue
            file_count += 1
            if file_count > MAX_FILES:
                raise GenerativeArtifactInspectionError("artifact file count exceeds bounded limit")
            relative = Path(entry.path).relative_to(root)
            component = relative.parts[0] if len(relative.parts) > 1 else "root"
            sizes[component] = sizes.get(component, 0) + entry.stat(follow_symlinks=False).st_size
    return sizes, file_count


def _components(sizes: dict[str, int]) -> list[dict[str, object]]:
    totals: dict[str, int] = {"denoiser": 0, "text_encoder": 0, "vae": 0, "other": 0}
    names: dict[str, list[str]] = {role: [] for role in totals}
    for name, size in sorted(sizes.items()):
        role = COMPONENT_ROLES.get(name, "other")
        totals[role] += size
        names[role].append(name)
    return [
        {"name": "+".join(names[role]), "role": role, "artifact_bytes": totals[role]}
        for role in ("denoiser", "text_encoder", "vae", "other")
        if totals[role]
    ]


def inspect_generative_artifact(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink():
        raise GenerativeArtifactInspectionError("generative artifact must not be a symlink")
    root = source.resolve()
    if not root.is_dir():
        raise GenerativeArtifactInspectionError("generative artifact must be a local directory")

    sizes, file_count = _bounded_sizes(root)
    model_index = _read_json(root / "model_index.json")
    quantize_config = _read_json(root / "quantize_config.json")
    shard_index = _read_json(root / "model.safetensors.index.json")
    if shard_index is None:
        for component_name in ("transformer", "text_encoder", "vae"):
            shard_index = _read_json(root / component_name / "model.safetensors.index.json")
            if shard_index is not None:
                break
    shard_metadata = shard_index.get("metadata", {}) if shard_index else {}
    if not isinstance(shard_metadata, dict):
        shard_metadata = {}

    pipeline_class = model_index.get("_class_name") if model_index else None
    quantization: dict[str, object] = {}
    if quantize_config:
        config = quantize_config.get(
            "quantization_config", quantize_config.get("quantization", quantize_config)
        )
        if isinstance(config, dict):
            for key in ("bits", "group_size"):
                if isinstance(config.get(key), int):
                    quantization[key] = config[key]
    level = shard_metadata.get("quantization_level")
    if isinstance(level, (str, int)) and str(level).isdigit():
        quantization["bits"] = int(level)

    model_card = _read_model_card_metadata(root / "README.md")
    library_name = model_card.get("library_name")
    if library_name == "mlx-gen":
        artifact_format = "mlx-gen"
        backend_kind = "mlx-gen"
    elif isinstance(shard_metadata.get("mflux_version"), str):
        artifact_format = "mflux"
        backend_kind = "mflux"
    elif model_index and quantize_config:
        artifact_format = "mlx-diffusers-conversion"
        backend_kind = "mlx-diffusers"
    elif model_index:
        artifact_format = "diffusers"
        backend_kind = "diffusers"
    else:
        artifact_format = "unknown"
        backend_kind = "unknown"

    components = _components(sizes)
    total_bytes = sum(sizes.values())
    required_roles = {"denoiser", "text_encoder", "vae"}
    present_roles = {str(component["role"]) for component in components}
    issues = [f"missing_component_role:{role}" for role in sorted(required_roles - present_roles)]
    if backend_kind == "unknown":
        issues.append("unrecognized_artifact_format")

    return {
        "schema_version": 1,
        "path": str(root),
        "artifact_format": artifact_format,
        "backend_kind": backend_kind,
        "pipeline_class": pipeline_class if isinstance(pipeline_class, str) else None,
        "library_name": library_name,
        "license": model_card.get("license"),
        "base_model": model_card.get("base_model"),
        "quantization": quantization,
        "components": components,
        "artifact_bytes": total_bytes,
        "file_count": file_count,
        "weights_loaded": False,
        "metal_allocated": False,
        "memory_fit_evaluated": False,
        "issues": issues,
        "inspectable": not issues,
    }


def qualification_components_from_inspection(
    report: dict[str, object], estimated_resident_bytes: int
) -> tuple[GenerativeArtifactComponent, ...]:
    raw_components = report.get("components")
    artifact_bytes = report.get("artifact_bytes")
    if (
        not isinstance(raw_components, list)
        or not isinstance(artifact_bytes, int)
        or artifact_bytes <= 0
        or estimated_resident_bytes < len(raw_components)
    ):
        raise ValueError("artifact inspection cannot produce qualification components")
    components = []
    allocated = 0
    for index, item in enumerate(raw_components):
        if not isinstance(item, dict):
            raise ValueError("artifact inspection component is invalid")
        name, role, component_bytes = item.get("name"), item.get("role"), item.get("artifact_bytes")
        if not isinstance(name, str) or not isinstance(role, str) or not isinstance(component_bytes, int):
            raise ValueError("artifact inspection component is invalid")
        if index == len(raw_components) - 1:
            resident = estimated_resident_bytes - allocated
        else:
            resident = max(1, estimated_resident_bytes * component_bytes // artifact_bytes)
            allocated += resident
        components.append(GenerativeArtifactComponent(name, role, component_bytes, resident))
    return tuple(components)
