from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .architecture_evidence import verify_startup_evidence
from .architecture_registry import describe_architecture
from .context import recommend_state_context
from .model import InspectedModel, ModelCapabilityError, ensure_model_backend_compatible
from .types import GIB, HardwareInfo

MODEL_RECOMMENDATION_SCHEMA_VERSION = 2


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
    recognition: str
    structure_status: str
    config_sha256: str
    declared_features_match: bool
    eligible_for_validation: bool
    compatibility_basis: str = "metadata_and_caller_declarations"
    qualification: str = "unverified"

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
            "recognition": self.recognition,
            "structure_status": self.structure_status,
            "config_sha256": self.config_sha256,
            "declared_features_match": self.declared_features_match,
            "eligible_for_validation": self.eligible_for_validation,
            "compatibility_basis": self.compatibility_basis,
            "qualification": self.qualification,
        }


def build_model_recommendation(
    model: InspectedModel,
    hardware: HardwareInfo,
    *,
    backend: str,
    available_features: frozenset[str] = frozenset(),
    requested_modes: frozenset[str] = frozenset({"text"}),
    architecture_evidence: Path | None = None,
    backend_executable: Path | None = None,
    context_tokens: int | None = None,
    concurrency: int = 1,
) -> ModelRecommendation:
    if backend not in {"vllm_metal", "mlx_lm"}:
        raise ValueError("unsupported recommendation backend")
    if not model.model_id or len(model.model_id.encode("utf-8")) > 4_096:
        raise ValueError("model recommendation identifier is invalid")
    capability = model.architecture_capability
    descriptor = describe_architecture(model.config)
    required_features = tuple(sorted(
        set(capability.required_features) | set(descriptor["required_features"])
    ))
    if not capability.architecture or len(capability.architecture) > 128:
        raise ValueError("model recommendation architecture is invalid")
    if len(required_features) > 64 or any(
        not feature or len(feature) > 128 for feature in required_features
    ):
        raise ValueError("model recommendation features are invalid")
    if len(available_features) > 64 or any(
        not isinstance(feature, str) or not feature or len(feature) > 128
        for feature in available_features
    ):
        raise ValueError("backend feature declarations are invalid")
    if not requested_modes or len(requested_modes) > 4:
        raise ValueError("recommendation requires 1 to 4 modes")
    state = model.state_memory_spec or model.memory_spec.as_state_memory_spec()
    context = recommend_state_context(hardware.memory, state)
    tiers = {tier.name: tier for tier in context.tiers}
    selected = tiers.get("balanced") or tiers.get("safe")
    if selected is None:
        raise ValueError("context recommendation did not produce a usable tier")
    tokens = selected.max_tokens
    if context_tokens is not None:
        if type(context_tokens) is not int or not 1 <= context_tokens <= tokens:
            raise ValueError("requested context exceeds recommended memory/model limit")
        tokens = context_tokens
    emergency_margin = max(GIB, int(hardware.memory.total_bytes * 0.08))
    hard_ceiling = max(0, hardware.memory.available_bytes - emergency_margin)
    resident = state.total_bytes(tokens)
    fits = resident <= hard_ceiling
    declarations_match = True
    issue = None
    try:
        if descriptor["recognition"] != "recognized":
            raise ModelCapabilityError("architecture_unknown")
        if descriptor["structure_status"] != "described":
            raise ModelCapabilityError(str(descriptor["issues"][0]))
        missing = sorted(set(descriptor["required_features"]) - available_features)
        if missing:
            raise ModelCapabilityError("backend_missing_model_capabilities:" + ",".join(missing))
        ensure_model_backend_compatible(
            model,
            backend=backend,
            available_features=available_features,
            requested_modes=requested_modes,
        )
    except ModelCapabilityError as error:
        declarations_match = False
        issue = str(error)
    evidence_verified = False
    if architecture_evidence is not None:
        if backend_executable is None or context_tokens is None:
            raise ValueError("architecture evidence requires backend executable and explicit context")
        if requested_modes != frozenset({"text"}):
            raise ValueError("architecture evidence only covers text mode")
        verify_startup_evidence(
            architecture_evidence, model, backend_executable, backend, hardware,
            context_tokens=context_tokens, concurrency=concurrency,
        )
        evidence_verified = True
    return ModelRecommendation(
        model=model.model_id,
        architecture=capability.architecture,
        modes=capability.modes,
        required_features=required_features,
        backend=backend,
        requested_modes=tuple(sorted(requested_modes)),
        backend_compatible=evidence_verified,
        compatibility_issue=None if evidence_verified else issue or "backend_execution_unverified",
        state_memory=state.to_dict(),
        context=context.to_dict(),
        recommended_tier=selected.name,
        recommended_context_tokens=tokens,
        estimated_resident_bytes=resident,
        memory_hard_ceiling_bytes=hard_ceiling,
        fits_memory=fits,
        runnable=evidence_verified and fits and tokens > 0,
        recognition=str(descriptor["recognition"]),
        structure_status=str(descriptor["structure_status"]),
        config_sha256=str(descriptor["config_sha256"]),
        declared_features_match=declarations_match,
        eligible_for_validation=(declarations_match or evidence_verified) and fits and tokens > 0,
        compatibility_basis=(
            "identity_bound_text_smoke" if evidence_verified else "metadata_and_caller_declarations"
        ),
        qualification="text_smoke_30min" if evidence_verified else "unverified",
    )
