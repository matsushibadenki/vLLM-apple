from __future__ import annotations

from .types import GIB, HardwareInfo

QUANTIZABLE_ROLES = frozenset({"denoiser", "text_encoder"})


def build_qwen_image_21_residency_plan(
    artifact: dict[str, object], hardware: HardwareInfo, *, target_bits: int
) -> dict[str, object]:
    if target_bits not in {4, 8}:
        raise ValueError("Qwen-Image-2.1 target bits must be 4 or 8")
    if artifact.get("pipeline_class") != "QwenImage21Pipeline":
        raise ValueError("artifact pipeline must be QwenImage21Pipeline")
    raw_components = artifact.get("components")
    if not isinstance(raw_components, list):
        raise ValueError("artifact components are required")

    projected_components: list[dict[str, object]] = []
    projected_weight_bytes = 0
    roles: set[str] = set()
    for component in raw_components:
        if not isinstance(component, dict):
            raise ValueError("artifact component is invalid")
        name, role, size = (
            component.get("name"),
            component.get("role"),
            component.get("artifact_bytes"),
        )
        if not isinstance(name, str) or not isinstance(role, str) or not isinstance(size, int):
            raise ValueError("artifact component is invalid")
        if size <= 0:
            raise ValueError("artifact component byte count must be positive")
        roles.add(role)
        projected = (size * target_bits + 15) // 16 if role in QUANTIZABLE_ROLES else size
        projected_weight_bytes += projected
        projected_components.append(
            {
                "name": name,
                "role": role,
                "source_bytes": size,
                "projected_bytes": projected,
                "target_bits": target_bits if role in QUANTIZABLE_ROLES else 16,
            }
        )
    missing = {"denoiser", "text_encoder", "vae"} - roles
    if missing:
        raise ValueError(f"missing component roles: {','.join(sorted(missing))}")

    emergency_reserve = max(GIB, int(hardware.memory.total_bytes * 0.08))
    allocator_margin = GIB
    image_working_bytes = 512 * 512 * 4 * 4
    projected_resident_bytes = projected_weight_bytes + allocator_margin + image_working_bytes
    physical_safe_ceiling = max(0, hardware.memory.total_bytes - emergency_reserve)
    dynamic_safe_ceiling = max(0, hardware.memory.available_bytes - emergency_reserve)
    return {
        "schema_version": 1,
        "candidate_id": "qwen-image-2.1",
        "source_quantization": "bf16",
        "target_quantization": f"int{target_bits}",
        "strategy": "quantized-model-cpu-offload",
        "components": projected_components,
        "projected_weight_bytes": projected_weight_bytes,
        "projected_resident_bytes": projected_resident_bytes,
        "emergency_reserve_bytes": emergency_reserve,
        "physical_safe_ceiling_bytes": physical_safe_ceiling,
        "dynamic_safe_ceiling_bytes": dynamic_safe_ceiling,
        "fits_physical_memory": projected_resident_bytes <= physical_safe_ceiling,
        "fits_current_available_memory": projected_resident_bytes <= dynamic_safe_ceiling,
        "projection_assumptions": {
            "source_weight_bits": 16,
            "scale_and_metadata_overhead_included": False,
            "vae_and_other_components_remain_bits": 16,
        },
        "conversion_required": True,
        "post_conversion_measurement_required": True,
        "runtime_support_verified": False,
        "eligible_for_generation": False,
        "next_gate": "produce-and-inspect-quantized-artifact",
    }
