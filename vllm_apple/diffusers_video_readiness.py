from __future__ import annotations

from pathlib import Path

from .diffusers_generative_readiness import inspect_diffusers_generative_readiness
from .generative_artifact_inspection import inspect_generative_artifact


WAN_TI2V_CANDIDATE = "wan2.2-ti2v-5b"
WAN_PIPELINE_CLASS = "WanPipeline"
SUPPORTED_QUANTIZATION_BITS = frozenset({4, 8})


def assess_diffusers_video_readiness(
    *,
    executable: str,
    backend: dict[str, object],
    artifact: dict[str, object],
) -> dict[str, object]:
    issues: list[str] = []
    candidates = backend.get("candidates")
    candidate = candidates.get(WAN_TI2V_CANDIDATE) if isinstance(candidates, dict) else None
    required = candidate.get("required_pipeline_classes", []) if isinstance(candidate, dict) else []
    if WAN_PIPELINE_CLASS not in required or (
        isinstance(candidate, dict)
        and WAN_PIPELINE_CLASS in candidate.get("missing_pipeline_classes", [])
    ):
        issues.append("wan_pipeline_unavailable")
    if artifact.get("artifact_format") != "diffusers":
        issues.append(f"unsupported_artifact_format:{artifact.get('artifact_format')}")
    if artifact.get("pipeline_class") != WAN_PIPELINE_CLASS:
        issues.append("unexpected_pipeline_class")
    if not artifact.get("inspectable"):
        issues.append("artifact_not_inspectable")
    quantization = artifact.get("quantization")
    bits = quantization.get("bits") if isinstance(quantization, dict) else None
    if bits not in SUPPORTED_QUANTIZATION_BITS:
        issues.append("expected_4bit_or_8bit_quantization")
    return {
        "schema_version": 1,
        "backend": "diffusers",
        "candidate_id": WAN_TI2V_CANDIDATE,
        "executable": executable,
        "diffusers_version": backend.get("diffusers_version"),
        "artifact": artifact,
        "supported_quantization_bits": sorted(SUPPORTED_QUANTIZATION_BITS),
        "ready": not issues,
        "issues": issues,
        "imports_backend": False,
        "loads_weights": False,
        "allocates_model_or_metal": False,
    }


def inspect_diffusers_video_readiness(
    executable: str | Path, *, model: str | Path
) -> dict[str, object]:
    backend = inspect_diffusers_generative_readiness(executable)
    artifact = inspect_generative_artifact(model)
    return assess_diffusers_video_readiness(
        executable=str(Path(executable).expanduser().resolve()),
        backend=backend,
        artifact=artifact,
    )
