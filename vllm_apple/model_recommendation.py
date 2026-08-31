from __future__ import annotations

from dataclasses import dataclass

from .context import recommend_state_context
from .model import InspectedModel, ModelCapabilityError, ensure_model_backend_compatible
from .types import GIB, HardwareInfo

MODEL_RECOMMENDATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ModelRecommendation:
    model: str
    architecture: str
    modes: tuple[str, ...]
    required_features: tuple[str, ...]
    backend: str
    requested_modes: tuple[str, ...]
    backend_compatible: bool
    compatibility_issue: str | None
    state_memory: dict[str, object]
    context: dict[str, object]
    recommended_tier: str
    recommended_context_tokens: int
    estimated_resident_bytes: int
    memory_hard_ceiling_bytes: int
    fits_memory: bool
    runnable: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": MODEL_RECOMMENDATION_SCHEMA_VERSION,
            "model": self.model,
            "architecture": self.architecture,
            "modes": list(self.modes),
            "required_features": list(self.required_features),
            "backend": self.backend,
            "requested_modes": list(self.requested_modes),
            "backend_compatible": self.backend_compatible,
            "compatibility_issue": self.compatibility_issue,
            "state_memory": self.state_memory,
            "context": self.context,
            "recommended_tier": self.recommended_tier,
            "recommended_context_tokens": self.recommended_context_tokens,
            "estimated_resident_bytes": self.estimated_resident_bytes,
            "memory_hard_ceiling_bytes": self.memory_hard_ceiling_bytes,
            "fits_memory": self.fits_memory,
            "runnable": self.runnable,
        }


def build_model_recommendation(
    model: InspectedModel,
    hardware: HardwareInfo,
    *,
    backend: str,
    available_features: frozenset[str] = frozenset(),
    requested_modes: frozenset[str] = frozenset({"text"}),
) -> ModelRecommendation:
    if backend not in {"vllm_metal", "mlx_lm"}:
        raise ValueError("unsupported recommendation backend")
    if not model.model_id or len(model.model_id.encode("utf-8")) > 4_096:
        raise ValueError("model recommendation identifier is invalid")
    capability = model.architecture_capability
    if not capability.architecture or len(capability.architecture) > 128:
        raise ValueError("model recommendation architecture is invalid")
    if len(capability.required_features) > 64 or any(
        not feature or len(feature) > 128 for feature in capability.required_features
    ):
        raise ValueError("model recommendation features are invalid")
    if not requested_modes or len(requested_modes) > 4:
        raise ValueError("recommendation requires 1 to 4 modes")
    state = model.state_memory_spec or model.memory_spec.as_state_memory_spec()
    context = recommend_state_context(hardware.memory, state)
    tiers = {tier.name: tier for tier in context.tiers}
    selected = tiers.get("balanced") or tiers.get("safe")
    if selected is None:
        raise ValueError("context recommendation did not produce a usable tier")
    tokens = selected.max_tokens
    emergency_margin = max(GIB, int(hardware.memory.total_bytes * 0.08))
    hard_ceiling = max(0, hardware.memory.available_bytes - emergency_margin)
    resident = state.total_bytes(tokens)
    fits = resident <= hard_ceiling
    compatible = True
    issue = None
    try:
        ensure_model_backend_compatible(
            model,
            backend=backend,
            available_features=available_features,
            requested_modes=requested_modes,
        )
    except ModelCapabilityError as error:
        compatible = False
        issue = str(error)
    return ModelRecommendation(
        model=model.model_id,
        architecture=capability.architecture,
        modes=capability.modes,
        required_features=capability.required_features,
        backend=backend,
        requested_modes=tuple(sorted(requested_modes)),
        backend_compatible=compatible,
        compatibility_issue=issue,
        state_memory=state.to_dict(),
        context=context.to_dict(),
        recommended_tier=selected.name,
        recommended_context_tokens=tokens,
        estimated_resident_bytes=resident,
        memory_hard_ceiling_bytes=hard_ceiling,
        fits_memory=fits,
        runnable=compatible and fits and tokens > 0,
    )
