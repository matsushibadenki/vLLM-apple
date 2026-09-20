from __future__ import annotations

from typing import Mapping

from .types import GIB


def estimate_mlx_gen_video_resident_bytes(
    artifact: Mapping[str, object], *, width: int, height: int, frames: int
) -> int:
    """Conservative phase-aware estimate for MLX-Gen Wan --low-ram execution."""
    if min(width, height, frames) <= 0:
        raise ValueError("video profile dimensions must be positive")
    raw_components = artifact.get("components")
    if not isinstance(raw_components, list):
        raise ValueError("video artifact components are unavailable")
    by_role: dict[str, int] = {}
    for component in raw_components:
        if not isinstance(component, dict):
            raise ValueError("video artifact component is invalid")
        role = component.get("role")
        size = component.get("artifact_bytes")
        if role in {"denoiser", "text_encoder", "vae"} and isinstance(size, int) and size > 0:
            by_role[role] = by_role.get(role, 0) + size
    if set(by_role) != {"denoiser", "text_encoder", "vae"}:
        raise ValueError("video artifact is missing a required residency component")
    # --low-ram releases the text encoder before denoising. VAE can overlap the
    # denoiser during transition, so retain it in the generation phase estimate.
    phase_bytes = max(by_role["text_encoder"], by_role["denoiser"] + by_role["vae"])
    # Artifact admission independently retains the larger of 1 GiB and 8% of
    # unified memory as an emergency reserve. Keep this estimate scoped to the
    # worker working set so the same safety budget is not counted twice.
    allocator_margin = GIB
    decoded_video_bytes = width * height * 4 * frames
    return phase_bytes + allocator_margin + decoded_video_bytes
