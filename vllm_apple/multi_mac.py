"""Deterministic, fail-closed planning contracts for a bounded Mac compute fabric."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum

MAX_FABRIC_NODES = 64
MAX_FABRIC_LINKS = 2016
MAX_FABRIC_STAGES = 1024
MAX_STATE_SHARDS = 4096


class FabricTransport(str, Enum):
    THUNDERBOLT = "thunderbolt"
    ETHERNET = "ethernet"


class StageKind(str, Enum):
    PIPELINE = "pipeline"
    MODALITY = "modality"


@dataclass(frozen=True, slots=True)
class FabricNode:
    node_id: str
    memory_capacity_bytes: int
    modalities: tuple[str, ...]
    healthy: bool = True

    def __post_init__(self) -> None:
        _identifier(self.node_id, "node")
        if not 0 < self.memory_capacity_bytes <= 2**63 - 1:
            raise ValueError("invalid fabric node memory capacity")
        if (
            not self.modalities
            or len(self.modalities) > 16
            or len(set(self.modalities)) != len(self.modalities)
        ):
            raise ValueError("invalid fabric node modalities")
        for modality in self.modalities:
            _identifier(modality, "modality")


@dataclass(frozen=True, slots=True)
class FabricLink:
    source: str
    destination: str
    transport: FabricTransport
    bandwidth_bytes_per_second: int
    latency_microseconds: int
    mtu_bytes: int
    authenticated: bool
    measurement_id: str

    def __post_init__(self) -> None:
        _identifier(self.source, "link source")
        _identifier(self.destination, "link destination")
        _identifier(self.measurement_id, "measurement")
        if self.source == self.destination or not isinstance(self.transport, FabricTransport):
            raise ValueError("invalid fabric link identity")
        if (
            not 0 < self.bandwidth_bytes_per_second <= 2**63 - 1
            or not 0 < self.latency_microseconds <= 60_000_000
            or not 576 <= self.mtu_bytes <= 1_048_576
            or not self.authenticated
        ):
            raise ValueError("fabric link is not qualified")


@dataclass(frozen=True, slots=True)
class FabricStage:
    stage_id: str
    kind: StageKind
    required_modality: str
    resident_bytes: int
    output_bytes: int
    dependencies: tuple[str, ...] = ()
    checkpointable: bool = True

    def __post_init__(self) -> None:
        _identifier(self.stage_id, "stage")
        _identifier(self.required_modality, "stage modality")
        if not isinstance(self.kind, StageKind):
            raise ValueError("invalid fabric stage kind")
        if (
            not 0 < self.resident_bytes <= 2**63 - 1
            or not 0 <= self.output_bytes <= 2**63 - 1
            or len(self.dependencies) > MAX_FABRIC_STAGES
            or len(set(self.dependencies)) != len(self.dependencies)
            or self.stage_id in self.dependencies
        ):
            raise ValueError("invalid fabric stage resources or dependencies")
        for dependency in self.dependencies:
            _identifier(dependency, "stage dependency")


@dataclass(frozen=True, slots=True)
class DistributedStateShard:
    shard_id: str
    digest: str
    size_bytes: int
    replica_nodes: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.shard_id, "state shard")
        if (
            len(self.digest) != 64
            or any(character not in "0123456789abcdef" for character in self.digest)
            or not 0 < self.size_bytes <= 2**63 - 1
            or not self.replica_nodes
            or len(self.replica_nodes) > MAX_FABRIC_NODES
            or len(set(self.replica_nodes)) != len(self.replica_nodes)
        ):
            raise ValueError("invalid distributed state shard")
        for node_id in self.replica_nodes:
            _identifier(node_id, "state replica node")


@dataclass(frozen=True, slots=True)
class StagePlacement:
    stage_id: str
    node_id: str
    reserved_bytes: int


@dataclass(frozen=True, slots=True)
class FabricTransfer:
    source_stage: str
    destination_stage: str
    source_node: str
    destination_node: str
    transport: FabricTransport
    bytes: int
    estimated_microseconds: int
    measurement_id: str


@dataclass(frozen=True, slots=True)
class MultiMacExecutionPlan:
    plan_id: str
    placements: tuple[StagePlacement, ...]
    transfers: tuple[FabricTransfer, ...]
    state_sources: tuple[tuple[str, str], ...]
    failed_nodes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "plan_id": self.plan_id,
            "placements": [asdict(value) for value in self.placements],
            "transfers": [
                {**asdict(value), "transport": value.transport.value}
                for value in self.transfers
            ],
            "state_sources": [
                {"shard_id": shard, "node_id": node}
                for shard, node in self.state_sources
            ],
            "failed_nodes": list(self.failed_nodes),
        }


def build_multi_mac_plan(
    nodes: tuple[FabricNode, ...],
    links: tuple[FabricLink, ...],
    stages: tuple[FabricStage, ...],
    state_shards: tuple[DistributedStateShard, ...] = (),
    *,
    previous_plan: MultiMacExecutionPlan | None = None,
) -> MultiMacExecutionPlan:
    if not 1 <= len(nodes) <= MAX_FABRIC_NODES or len({n.node_id for n in nodes}) != len(nodes):
        raise ValueError("invalid fabric node set")
    if not 0 <= len(links) <= MAX_FABRIC_LINKS:
        raise ValueError("invalid fabric link set")
    if not 1 <= len(stages) <= MAX_FABRIC_STAGES:
        raise ValueError("invalid fabric stage set")
    if len({stage.stage_id for stage in stages}) != len(stages):
        raise ValueError("duplicate fabric stage")
    if len(state_shards) > MAX_STATE_SHARDS:
        raise ValueError("too many distributed state shards")
    node_by_id = {node.node_id: node for node in nodes}
    healthy = {node.node_id: node for node in nodes if node.healthy}
    if not healthy:
        raise RuntimeError("no healthy fabric nodes")
    link_by_pair: dict[tuple[str, str], FabricLink] = {}
    for link in links:
        if link.source not in node_by_id or link.destination not in node_by_id:
            raise ValueError("fabric link references an unknown node")
        for pair in ((link.source, link.destination), (link.destination, link.source)):
            current = link_by_pair.get(pair)
            if current is None or _link_cost(link, 1) < _link_cost(current, 1):
                link_by_pair[pair] = link
    ordered = _topological_stages(stages)
    failed_nodes = tuple(sorted(node_id for node_id in node_by_id if node_id not in healthy))
    if previous_plan is not None and failed_nodes:
        failed_stage_ids = {
            placement.stage_id
            for placement in previous_plan.placements
            if placement.node_id in failed_nodes
        }
        stage_by_id = {stage.stage_id: stage for stage in stages}
        if any(not stage_by_id[stage_id].checkpointable for stage_id in failed_stage_ids):
            raise RuntimeError("failed node owned a non-checkpointable stage")
    state_sources = []
    for shard in sorted(state_shards, key=lambda value: value.shard_id):
        if any(node not in node_by_id for node in shard.replica_nodes):
            raise ValueError("state shard references an unknown node")
        available = sorted(node for node in shard.replica_nodes if node in healthy)
        if not available:
            raise RuntimeError("distributed state has no healthy replica")
        state_sources.append((shard.shard_id, available[0]))
    remaining = {node_id: node.memory_capacity_bytes for node_id, node in healthy.items()}
    placement_by_stage: dict[str, StagePlacement] = {}
    placements = []
    transfers = []
    for stage in ordered:
        candidates = []
        for node_id, node in healthy.items():
            if stage.required_modality not in node.modalities or remaining[node_id] < stage.resident_bytes:
                continue
            transfer_cost = 0
            feasible = True
            for dependency_id in stage.dependencies:
                dependency = placement_by_stage[dependency_id]
                if dependency.node_id == node_id:
                    continue
                link = link_by_pair.get((dependency.node_id, node_id))
                if link is None:
                    feasible = False
                    break
                dependency_stage = next(value for value in ordered if value.stage_id == dependency_id)
                transfer_cost += _link_cost(link, dependency_stage.output_bytes)
            if feasible:
                used = node.memory_capacity_bytes - remaining[node_id]
                candidates.append((transfer_cost, used, node_id))
        if not candidates:
            raise RuntimeError(f"no qualified node can host stage {stage.stage_id}")
        _, _, node_id = min(candidates)
        remaining[node_id] -= stage.resident_bytes
        placement = StagePlacement(stage.stage_id, node_id, stage.resident_bytes)
        placements.append(placement)
        placement_by_stage[stage.stage_id] = placement
        for dependency_id in stage.dependencies:
            dependency = placement_by_stage[dependency_id]
            if dependency.node_id == node_id:
                continue
            link = link_by_pair[(dependency.node_id, node_id)]
            dependency_stage = next(value for value in ordered if value.stage_id == dependency_id)
            transfers.append(FabricTransfer(
                dependency_id, stage.stage_id, dependency.node_id, node_id,
                link.transport, dependency_stage.output_bytes,
                _link_cost(link, dependency_stage.output_bytes), link.measurement_id,
            ))
    identity = {
        "nodes": [asdict(value) for value in sorted(nodes, key=lambda value: value.node_id)],
        "links": [
            {**asdict(value), "transport": value.transport.value}
            for value in sorted(links, key=lambda value: (value.source, value.destination, value.transport.value))
        ],
        "stages": [
            {**asdict(value), "kind": value.kind.value}
            for value in ordered
        ],
        "placements": [asdict(value) for value in placements],
        "state_sources": state_sources,
        "failed_nodes": failed_nodes,
    }
    plan_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return MultiMacExecutionPlan(
        plan_id, tuple(placements), tuple(transfers), tuple(state_sources), failed_nodes
    )


def _topological_stages(stages: tuple[FabricStage, ...]) -> tuple[FabricStage, ...]:
    by_id = {stage.stage_id: stage for stage in stages}
    if any(dependency not in by_id for stage in stages for dependency in stage.dependencies):
        raise ValueError("fabric stage references an unknown dependency")
    pending = set(by_id)
    ordered = []
    while pending:
        ready = sorted(
            stage_id
            for stage_id in pending
            if all(dependency not in pending for dependency in by_id[stage_id].dependencies)
        )
        if not ready:
            raise ValueError("fabric stage graph contains a cycle")
        for stage_id in ready:
            pending.remove(stage_id)
            ordered.append(by_id[stage_id])
    return tuple(ordered)


def _link_cost(link: FabricLink, payload_bytes: int) -> int:
    transfer = (payload_bytes * 1_000_000 + link.bandwidth_bytes_per_second - 1) // (
        link.bandwidth_bytes_per_second
    )
    return link.latency_microseconds + transfer


def _identifier(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 128
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise ValueError(f"invalid fabric {label} identifier")
