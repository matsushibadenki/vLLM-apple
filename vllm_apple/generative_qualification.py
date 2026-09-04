from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .artifact_admission import ArtifactAdmission, assess_artifact_admission_for_path
from .types import HardwareInfo


GENERATIVE_QUALIFICATION_SCHEMA_VERSION = 1
MAX_DIMENSION = 4096
MAX_FRAMES = 257
MAX_STEPS = 200
MAX_COMPONENTS = 16
MAX_COMPONENT_BYTES = 16_384 * 1024**3
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


def list_generative_candidates() -> tuple[GenerativeCandidate, ...]:
    return _CANDIDATES


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
        tuple(issues),
        admission,
        not issues and admission.eligible,
    )
