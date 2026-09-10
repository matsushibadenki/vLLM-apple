from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class WeightBlockResidencyFeasibility:
    incremental_block_load: bool
    safe_block_release: bool
    compiled_graph_rebind: bool
    stable_weight_keys: bool
    blockers: tuple[str, ...]
    eligible: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["blockers"] = list(self.blockers)
        return payload


def assess_weight_block_residency(
    *,
    incremental_block_load: bool,
    safe_block_release: bool,
    compiled_graph_rebind: bool,
    stable_weight_keys: bool,
) -> WeightBlockResidencyFeasibility:
    blockers = []
    if not incremental_block_load:
        blockers.append("incremental_block_loader_missing")
    if not safe_block_release:
        blockers.append("block_release_barrier_missing")
    if not compiled_graph_rebind:
        blockers.append("compiled_graph_rebind_missing")
    if not stable_weight_keys:
        blockers.append("stable_weight_key_contract_missing")
    return WeightBlockResidencyFeasibility(
        incremental_block_load,
        safe_block_release,
        compiled_graph_rebind,
        stable_weight_keys,
        tuple(blockers),
        not blockers,
    )


def current_mlx_gen_weight_residency_feasibility() -> WeightBlockResidencyFeasibility:
    """MLX-Gen 0.33.1 eagerly applies weights and compiled graphs capture them."""

    return assess_weight_block_residency(
        incremental_block_load=False,
        safe_block_release=False,
        compiled_graph_rebind=False,
        stable_weight_keys=True,
    )
