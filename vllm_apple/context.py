from __future__ import annotations

from dataclasses import dataclass

from .types import (
    ContextRecommendation,
    ContextTier,
    GIB,
    MemoryInfo,
    ModelMemorySpec,
    StateMemorySpec,
)


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    os_reserve_ratio: float = 0.15
    minimum_os_reserve_bytes: int = 3 * GIB
    safety_headroom_ratio: float = 0.08
    minimum_safety_headroom_bytes: int = GIB
    default_workspace_ratio: float = 0.05
    token_block_size: int = 256

    def __post_init__(self) -> None:
        ratios = (self.os_reserve_ratio, self.safety_headroom_ratio, self.default_workspace_ratio)
        if any(value < 0 or value >= 1 for value in ratios):
            raise ValueError("policy ratios must be in [0, 1)")
        if self.token_block_size <= 0:
            raise ValueError("token_block_size must be positive")


def _aligned_tokens(kv_bytes: int, bytes_per_token: int, block_size: int) -> int:
    raw = max(0, kv_bytes // bytes_per_token)
    return (raw // block_size) * block_size


def recommend_context(
    memory: MemoryInfo,
    model: ModelMemorySpec,
    policy: ContextPolicy = ContextPolicy(),
) -> ContextRecommendation:
    return recommend_state_context(memory, model.as_state_memory_spec(), policy)


def recommend_state_context(
    memory: MemoryInfo,
    model: StateMemorySpec,
    policy: ContextPolicy = ContextPolicy(),
) -> ContextRecommendation:
    os_reserve = max(
        policy.minimum_os_reserve_bytes, int(memory.total_bytes * policy.os_reserve_ratio)
    )
    safety = max(
        policy.minimum_safety_headroom_bytes,
        int(memory.total_bytes * policy.safety_headroom_ratio),
    )
    # Both limits matter: total capacity prevents overcommit after caches are reclaimed,
    # while currently available memory prevents destabilizing an already-busy machine.
    capacity_limit = max(0, memory.total_bytes - os_reserve - safety)
    availability_limit = max(0, memory.available_bytes - safety)
    allocatable = min(capacity_limit, availability_limit)
    workspace = model.scratch_workspace_bytes or int(
        model.weights_bytes * policy.default_workspace_ratio
    )
    fixed = (
        model.weights_bytes
        + model.recurrent_state_bytes
        + model.prefix_state_bytes
        + model.expert_working_set_bytes
        + model.ngram_working_set_bytes
        + model.mtp_working_set_bytes
        + workspace
    )
    state_capacity = max(0, allocatable - fixed)

    tier_ratios = (("safe", 0.70), ("balanced", 0.85), ("aggressive", 0.95))
    tiers: list[ContextTier] = []
    limited_by_model = False
    for name, ratio in tier_ratios:
        budget = int(state_capacity * ratio)
        tokens = _maximum_tokens(model, budget, policy.token_block_size)
        if model.model_max_context is not None and tokens == (
            model.model_max_context // policy.token_block_size
        ) * policy.token_block_size:
            limited_by_model = True
        tiers.append(ContextTier(name=name, max_tokens=tokens, kv_budget_bytes=budget))

    if state_capacity == 0:
        limiting_factor = "insufficient_memory"
    elif limited_by_model:
        limiting_factor = "model_max_context"
    elif availability_limit < capacity_limit:
        limiting_factor = "current_memory_availability"
    else:
        limiting_factor = "physical_memory"

    return ContextRecommendation(
        model_id=model.model_id,
        allocatable_bytes=allocatable,
        os_reserve_bytes=os_reserve,
        safety_headroom_bytes=safety,
        workspace_bytes=workspace,
        tiers=tuple(tiers),
        limiting_factor=limiting_factor,
    )


def _maximum_tokens(model: StateMemorySpec, budget: int, block_size: int) -> int:
    per_token = model.kv_bytes_per_token + model.attention_window_bytes_per_token
    if per_token == 0:
        assert model.model_max_context is not None
        return (model.model_max_context // block_size) * block_size
    if model.model_max_context is not None:
        upper = model.model_max_context // block_size
    elif model.kv_bytes_per_token > 0:
        upper = budget // model.kv_bytes_per_token // block_size
    else:
        # A pure sliding-window state is bounded only if the architecture declares
        # its model context limit (enforced by StateMemorySpec).
        assert model.model_max_context is not None
        upper = model.model_max_context // block_size
    fixed_state = model.recurrent_state_bytes + model.prefix_state_bytes
    low = 0
    high = max(0, upper)
    while low < high:
        middle = (low + high + 1) // 2
        tokens = middle * block_size
        if model.state_bytes(tokens) - fixed_state <= budget:
            low = middle
        else:
            high = middle - 1
    return low * block_size
