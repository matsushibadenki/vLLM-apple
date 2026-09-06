from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

from .artifact_admission import ArtifactAdmission, assess_artifact_admission_for_path
from .types import HardwareInfo


GENERATIVE_QUALIFICATION_SCHEMA_VERSION = 1
MAX_DIMENSION = 4096
MAX_FRAMES = 257
MAX_STEPS = 200
MAX_COMPONENTS = 16
MAX_COMPONENT_BYTES = 16_384 * 1024**3
STABILITY_PROMOTION_SAMPLE_COUNT = 4
COMPONENT_ROLES = frozenset({"denoiser", "text_encoder", "vae", "other"})


@dataclass(frozen=True, slots=True)
class GenerativeCandidate:
    candidate_id: str
    model: str
    modality: str
    tier: str
    modes: tuple[str, ...]
    initial_width: int
    initial_height: int
    initial_frames: int
    initial_steps: int
    requires_quantization: bool
    required_strategies: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["modes"] = list(self.modes)
        payload["required_strategies"] = list(self.required_strategies)
        return payload


_CANDIDATES = (
    GenerativeCandidate(
        "z-image-turbo-mlx-4bit",
        "mlx-community/Z-Image-Turbo-MLX-4bit",
        "image",
        "A",
        ("text-to-image",),
        512,
        512,
        1,
        9,
        True,
        ("mlx-native-quantization", "vae-tiling"),
    ),
    GenerativeCandidate(
        "flux2-klein-9b-base",
        "black-forest-labs/FLUX.2-klein-9B-base",
        "image",
        "A",
        ("text-to-image", "image-edit"),
        512,
        512,
        1,
        50,
        True,
        ("sequential-module-residency", "vae-tiling"),
    ),
    GenerativeCandidate(
        "qwen-image-2512",
        "Qwen/Qwen-Image-2512",
        "image",
        "B",
        ("text-to-image",),
        512,
        512,
        1,
        50,
        True,
        ("model-offload", "vae-tiling"),
    ),
    GenerativeCandidate(
        "flux2-dev",
        "black-forest-labs/FLUX.2-dev",
        "image",
        "C",
        ("text-to-image", "image-edit"),
        512,
        512,
        1,
        50,
        True,
        ("sequential-module-residency", "cpu-or-ssd-offload", "chunking"),
    ),
    GenerativeCandidate(
        "wan2.2-ti2v-5b",
        "Wan-AI/Wan2.2-TI2V-5B",
        "video",
        "A",
        ("text-to-video", "image-to-video"),
        640,
        360,
        33,
        20,
        True,
        ("sequential-module-residency", "vae-tiling"),
    ),
    GenerativeCandidate(
        "hunyuanvideo-1.5-8.3b",
        "tencent/HunyuanVideo-1.5",
        "video",
        "B",
        ("text-to-video", "image-to-video"),
        640,
        360,
        33,
        12,
        False,
        ("model-offload", "step-distilled"),
    ),
    GenerativeCandidate(
        "wan2.2-a14b-quantized",
        "Wan-AI/Wan2.2-A14B",
        "video",
        "C",
        ("text-to-video", "image-to-video"),
        640,
        360,
        33,
        20,
        True,
        ("dual-expert-staging", "cpu-or-ssd-offload", "chunking"),
    ),
)
GENERATIVE_CANDIDATES = {candidate.candidate_id: candidate for candidate in _CANDIDATES}


@dataclass(frozen=True, slots=True)
class GenerativeArtifactComponent:
    name: str
    role: str
    artifact_bytes: int
    estimated_resident_bytes: int

    def __post_init__(self) -> None:
        if (
            not self.name
            or self.name != self.name.strip()
            or len(self.name.encode("utf-8")) > 256
            or any(not character.isprintable() for character in self.name)
        ):
            raise ValueError("generative component name is invalid")
        if self.role not in COMPONENT_ROLES:
            raise ValueError(f"unsupported generative component role: {self.role}")
        values = (self.artifact_bytes, self.estimated_resident_bytes)
        if any(value <= 0 or value > MAX_COMPONENT_BYTES for value in values):
            raise ValueError("generative component byte count is outside the supported range")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenerativeQualificationPlan:
    schema_version: int
    candidate: GenerativeCandidate
    width: int
    height: int
    frames: int
    steps: int
    batch_size: int
    quantization: str
    components: tuple[GenerativeArtifactComponent, ...]
    component_artifact_bytes: int
    component_resident_bytes: int
    component_totals_verified: bool
    initial_profile: bool
    promotion_axis: str | None
    baseline_plan_sha256: str | None
    issues: tuple[str, ...]
    artifact_admission: ArtifactAdmission
    eligible: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate"] = self.candidate.to_dict()
        payload["components"] = [component.to_dict() for component in self.components]
        payload["issues"] = list(self.issues)
        payload["artifact_admission"] = self.artifact_admission.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class GenerativeBaselineEvidence:
    candidate_id: str
    plan_sha256: str
    sample_count: int
    width: int
    height: int
    frames: int
    memory_pressures: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.candidate_id
            or len(self.plan_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.plan_sha256)
            or not 1 <= self.sample_count <= 32
            or len(self.memory_pressures) != self.sample_count
            or min(self.width, self.height, self.frames) <= 0
        ):
            raise ValueError("generative baseline identity is invalid")
        if any(
            pressure not in {"normal", "warning", "critical", "unknown"}
            for pressure in self.memory_pressures
        ):
            raise ValueError("generative baseline memory pressure is invalid")


def list_generative_candidates() -> tuple[GenerativeCandidate, ...]:
    return _CANDIDATES


def generative_plan_sha256(plan: GenerativeQualificationPlan) -> str:
    encoded = json.dumps(
        plan.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generative_promotion_chain_sha256(
    stability_baseline: GenerativeBaselineEvidence,
    initial_baseline: GenerativeBaselineEvidence,
) -> str:
    encoded = json.dumps(
        {
            "initial_plan_sha256": initial_baseline.plan_sha256,
            "kind": "generative-resolution-promotion-chain",
            "schema_version": 1,
            "stability_plan_sha256": stability_baseline.plan_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_generative_component(value: str) -> GenerativeArtifactComponent:
    parts = value.split(":")
    if len(parts) != 4:
        raise ValueError("component must use name:role:artifact_bytes:resident_bytes")
    name, role, artifact_bytes, resident_bytes = parts
    try:
        return GenerativeArtifactComponent(name, role, int(artifact_bytes), int(resident_bytes))
    except ValueError as error:
        if "invalid literal" in str(error):
            raise ValueError("component byte counts must be integers") from error
        raise


def build_generative_qualification_plan(
    *,
    candidate_id: str,
    artifact_bytes: int,
    estimated_resident_bytes: int,
    hardware: HardwareInfo,
    target: Path,
    quantization: str,
    components: tuple[GenerativeArtifactComponent, ...],
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
    steps: int | None = None,
    batch_size: int = 1,
) -> GenerativeQualificationPlan:
    try:
        candidate = GENERATIVE_CANDIDATES[candidate_id]
    except KeyError as error:
        raise ValueError(f"unknown generative candidate: {candidate_id}") from error
    width = candidate.initial_width if width is None else width
    height = candidate.initial_height if height is None else height
    frames = candidate.initial_frames if frames is None else frames
    steps = candidate.initial_steps if steps is None else steps
    if not 1 <= width <= MAX_DIMENSION or not 1 <= height <= MAX_DIMENSION:
        raise ValueError("generation dimensions are outside the supported range")
    if not 1 <= frames <= MAX_FRAMES or not 1 <= steps <= MAX_STEPS:
        raise ValueError("generation frames or steps are outside the supported range")
    if not 1 <= batch_size <= 8:
        raise ValueError("generation batch size is outside the supported range")
    if quantization not in {"none", "int8", "fp8", "int4", "other"}:
        raise ValueError("unsupported quantization")
    if not 1 <= len(components) <= MAX_COMPONENTS:
        raise ValueError("between 1 and 16 generative components are required")
    names = tuple(component.name for component in components)
    if len(set(names)) != len(names):
        raise ValueError("generative component names must be unique")

    issues: list[str] = []
    if candidate.modality == "image" and frames != 1:
        issues.append("image_profile_requires_one_frame")
    if candidate.requires_quantization and quantization == "none":
        issues.append("candidate_requires_quantization_on_m4_32gb")
    initial_profile = (
        width <= candidate.initial_width
        and height <= candidate.initial_height
        and frames <= candidate.initial_frames
        and steps <= candidate.initial_steps
        and batch_size == 1
    )
    if not initial_profile:
        issues.append("initial_profile_limits_exceeded")

    roles = {component.role for component in components}
    missing_roles = {"denoiser", "text_encoder", "vae"} - roles
    for role in sorted(missing_roles):
        issues.append(f"missing_component_role:{role}")
    component_artifact_bytes = sum(component.artifact_bytes for component in components)
    component_resident_bytes = sum(
        component.estimated_resident_bytes for component in components
    )
    component_totals_verified = (
        component_artifact_bytes == artifact_bytes
        and component_resident_bytes == estimated_resident_bytes
    )
    if not component_totals_verified:
        issues.append("component_totals_mismatch")

    admission = assess_artifact_admission_for_path(
        model=candidate.model,
        artifact_bytes=artifact_bytes,
        estimated_resident_bytes=estimated_resident_bytes,
        hardware=hardware,
        target=target,
    )
    return GenerativeQualificationPlan(
        GENERATIVE_QUALIFICATION_SCHEMA_VERSION,
        candidate,
        width,
        height,
        frames,
        steps,
        batch_size,
        quantization,
        components,
        component_artifact_bytes,
        component_resident_bytes,
        component_totals_verified,
        initial_profile,
        None,
        None,
        tuple(issues),
        admission,
        not issues and admission.eligible,
    )


def promote_generative_resolution_plan(
    plan: GenerativeQualificationPlan,
    *,
    baseline_candidate_id: str,
    baseline_plan_sha256: str,
    baseline_sample_count: int,
    baseline_width: int,
    baseline_height: int,
    baseline_frames: int,
    baseline_memory_pressures: tuple[str, ...],
) -> GenerativeQualificationPlan:
    if plan.initial_profile:
        raise ValueError("initial generative profile does not require promotion")
    if (
        baseline_candidate_id != plan.candidate.candidate_id
        or len(baseline_plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in baseline_plan_sha256)
        or baseline_sample_count < 2
    ):
        raise ValueError("generative baseline identity is invalid")
    if len(baseline_memory_pressures) != baseline_sample_count or any(
        pressure != "normal" for pressure in baseline_memory_pressures
    ):
        raise ValueError("generative resolution promotion requires an all-normal baseline")
    if (
        baseline_width != plan.candidate.initial_width
        or baseline_height != plan.candidate.initial_height
        or baseline_frames != plan.candidate.initial_frames
        or plan.frames != baseline_frames
        or plan.batch_size != 1
        or plan.steps > plan.candidate.initial_steps
        or plan.width <= baseline_width
        or plan.height <= baseline_height
        or plan.width > baseline_width * 2
        or plan.height > baseline_height * 2
    ):
        raise ValueError("generative resolution promotion is not a bounded single-axis step")
    if plan.issues != ("initial_profile_limits_exceeded",):
        raise ValueError("generative plan has non-promotable issues")
    return replace(
        plan,
        promotion_axis="resolution",
        baseline_plan_sha256=baseline_plan_sha256,
        issues=(),
        eligible=plan.artifact_admission.eligible,
    )


def promote_generative_sample_count_plan(
    plan: GenerativeQualificationPlan,
    *,
    baseline_candidate_id: str,
    baseline_plan_sha256: str,
    baseline_sample_count: int,
    baseline_width: int,
    baseline_height: int,
    baseline_frames: int,
    baseline_memory_pressures: tuple[str, ...],
    target_sample_count: int,
) -> GenerativeQualificationPlan:
    """Promote a verified workload from two to four isolated stability samples."""
    if (
        baseline_candidate_id != plan.candidate.candidate_id
        or len(baseline_plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in baseline_plan_sha256)
    ):
        raise ValueError("generative stability baseline identity is invalid")
    if baseline_sample_count != 2 or target_sample_count != STABILITY_PROMOTION_SAMPLE_COUNT:
        raise ValueError("generative stability promotion must increase two samples to four")
    if len(baseline_memory_pressures) != baseline_sample_count or any(
        pressure != "normal" for pressure in baseline_memory_pressures
    ):
        raise ValueError("generative stability promotion requires an all-normal baseline")
    if (
        baseline_width != plan.width
        or baseline_height != plan.height
        or baseline_frames != plan.frames
        or plan.batch_size != 1
    ):
        raise ValueError("generative sample-count promotion changed the workload shape")
    if plan.issues != ("initial_profile_limits_exceeded",):
        raise ValueError("generative plan has non-promotable issues")
    return replace(
        plan,
        promotion_axis="sample_count_4",
        baseline_plan_sha256=baseline_plan_sha256,
        issues=(),
        eligible=plan.artifact_admission.eligible,
    )


def promote_generative_chained_resolution_plan(
    plan: GenerativeQualificationPlan,
    *,
    stability_baseline: GenerativeBaselineEvidence,
    initial_baseline: GenerativeBaselineEvidence,
) -> GenerativeQualificationPlan:
    """Promote resolution while binding the stable intermediate and initial reports."""
    if plan.initial_profile or plan.issues != ("initial_profile_limits_exceeded",):
        raise ValueError("generative plan has non-promotable issues")
    if any(
        evidence.candidate_id != plan.candidate.candidate_id
        for evidence in (stability_baseline, initial_baseline)
    ):
        raise ValueError("generative promotion chain candidate does not match")
    if (
        initial_baseline.width != plan.candidate.initial_width
        or initial_baseline.height != plan.candidate.initial_height
        or initial_baseline.frames != plan.candidate.initial_frames
        or initial_baseline.sample_count < 2
        or stability_baseline.sample_count != STABILITY_PROMOTION_SAMPLE_COUNT
    ):
        raise ValueError("generative promotion chain shape or sample count is invalid")
    if any(
        pressure != "normal"
        for evidence in (initial_baseline, stability_baseline)
        for pressure in evidence.memory_pressures
    ):
        raise ValueError("generative chained resolution promotion requires all-normal baselines")

    if (
        stability_baseline.frames != initial_baseline.frames
        or stability_baseline.width <= initial_baseline.width
        or stability_baseline.height <= initial_baseline.height
        or stability_baseline.width > initial_baseline.width * 2
        or stability_baseline.height > initial_baseline.height * 2
        or plan.frames != stability_baseline.frames
        or plan.batch_size != 1
        or plan.steps > plan.candidate.initial_steps
        or plan.width <= stability_baseline.width
        or plan.height <= stability_baseline.height
        or plan.width > stability_baseline.width * 2
        or plan.height > stability_baseline.height * 2
    ):
        raise ValueError("generative chained resolution promotion is not a bounded step")
    return replace(
        plan,
        promotion_axis="resolution",
        baseline_plan_sha256=generative_promotion_chain_sha256(
            stability_baseline, initial_baseline
        ),
        issues=(),
        eligible=plan.artifact_admission.eligible,
    )
