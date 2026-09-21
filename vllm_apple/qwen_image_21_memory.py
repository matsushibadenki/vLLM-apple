from __future__ import annotations

from .types import GIB

QWEN_IMAGE_21_REQUIRED_ROLES = frozenset({"denoiser", "text_encoder", "vae"})
QWEN_IMAGE_21_PHASE_MARGIN_BYTES = GIB
QWEN_IMAGE_21_MATERIALIZATION_NUMERATOR = 3
QWEN_IMAGE_21_MATERIALIZATION_DENOMINATOR = 2
QWEN_IMAGE_21_GROUP_OFFLOAD_EMPIRICAL_PEAK_BYTES = 17_954_488_320
QWEN_IMAGE_21_IMAGE_EDIT_EMPIRICAL_PEAK_BYTES = 19_381_600_256


def estimate_qwen_image_21_resident_bytes(
    artifact: dict[str, object], *, width: int, height: int, batch_size: int = 1,
    component_staged: bool = False,
    image_edit: bool = False,
) -> int:
    """Estimate Unified Memory residency for the Qwen-Image-2.1 worker.

    The default full-pipeline path retains the complete artifact-weight floor.
    A component-staged worker first loads only text_encoder+processor, releases it,
    then loads denoiser+VAE; that path admits against the larger phase plus shared
    artifact overhead instead of pretending that post-load release lowers load peak.
    """
    if not 1 <= width <= 4096 or not 1 <= height <= 4096:
        raise ValueError("image dimensions are outside the supported range")
    if batch_size != 1:
        raise ValueError("Qwen-Image-2.1 qualification requires batch size 1")
    raw_components = artifact.get("components")
    if not isinstance(raw_components, list):
        raise ValueError("artifact components are required for phase-aware estimation")

    role_bytes: dict[str, int] = {}
    for component in raw_components:
        if not isinstance(component, dict):
            raise ValueError("artifact component is invalid")
        role = component.get("role")
        size = component.get("artifact_bytes")
        if isinstance(role, str) and role in QWEN_IMAGE_21_REQUIRED_ROLES:
            if not isinstance(size, int) or size <= 0:
                raise ValueError("artifact component byte count must be positive")
            role_bytes[role] = role_bytes.get(role, 0) + size

    missing = QWEN_IMAGE_21_REQUIRED_ROLES - role_bytes.keys()
    if missing:
        raise ValueError(f"missing phase component roles: {','.join(sorted(missing))}")
    artifact_bytes = artifact.get("artifact_bytes")
    if not isinstance(artifact_bytes, int) or artifact_bytes <= 0:
        raise ValueError("artifact byte count is required for Unified Memory estimation")
    image_working_bytes = width * height * 4 * 4 * batch_size
    weight_floor = max(artifact_bytes, max(role_bytes.values()))
    if component_staged:
        classified = sum(role_bytes.values())
        if classified > artifact_bytes:
            raise ValueError("phase component bytes exceed the artifact")
        shared = artifact_bytes - classified
        text_phase = role_bytes["text_encoder"] + shared
        generation_phase = role_bytes["denoiser"] + role_bytes["vae"] + shared
        phase_floor = max(text_phase, generation_phase)
        weight_floor = (
            phase_floor * QWEN_IMAGE_21_MATERIALIZATION_NUMERATOR
            + QWEN_IMAGE_21_MATERIALIZATION_DENOMINATOR - 1
        ) // QWEN_IMAGE_21_MATERIALIZATION_DENOMINATOR
    estimate = weight_floor + QWEN_IMAGE_21_PHASE_MARGIN_BYTES + image_working_bytes
    if component_staged:
        estimate = max(
            estimate,
            QWEN_IMAGE_21_GROUP_OFFLOAD_EMPIRICAL_PEAK_BYTES
            + QWEN_IMAGE_21_PHASE_MARGIN_BYTES,
        )
    if image_edit:
        estimate = max(
            estimate,
            QWEN_IMAGE_21_IMAGE_EDIT_EMPIRICAL_PEAK_BYTES
            + QWEN_IMAGE_21_PHASE_MARGIN_BYTES,
        )
    return estimate
