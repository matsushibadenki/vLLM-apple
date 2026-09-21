"""Bounded dependency-aware dispatch through the production backend registry."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .backend_engine import BackendEngineRegistry, BackendEngineRequest, BackendEngineResult
from .execution import WorkloadPhase
from .inference_request import InferenceRequestContext

MAX_OPERATOR_GRAPH_NODES = 64
MAX_OPERATOR_GRAPH_EDGES = 256
MAX_OPERATOR_SYNC_NANOSECONDS = 60_000_000_000

_ALLOWED_PHASE_PREDECESSORS: dict[WorkloadPhase, frozenset[WorkloadPhase]] = {
    WorkloadPhase.PREFILL: frozenset({
        WorkloadPhase.VISION_ENCODER,
        WorkloadPhase.AUDIO_ENCODER,
        WorkloadPhase.EMBEDDING,
        WorkloadPhase.AUXILIARY,
    }),
    WorkloadPhase.DECODE: frozenset({
        WorkloadPhase.PREFILL,
        WorkloadPhase.DECODE,
        WorkloadPhase.VERIFY,
        WorkloadPhase.AUXILIARY,
    }),
    WorkloadPhase.SAMPLING: frozenset({
        WorkloadPhase.PREFILL,
        WorkloadPhase.DECODE,
        WorkloadPhase.VERIFY,
        WorkloadPhase.AUXILIARY,
    }),
    WorkloadPhase.VERIFY: frozenset({
        WorkloadPhase.DRAFT,
        WorkloadPhase.DECODE,
        WorkloadPhase.AUXILIARY,
    }),
}


@dataclass(frozen=True, slots=True)
class OperatorGraphNode:
    node_id: str
    request: BackendEngineRequest
    dependencies: tuple[str, ...] = ()
    synchronization_nanoseconds: int = 0

    def __post_init__(self) -> None:
        if (not self.node_id or len(self.node_id) > 128
                or len(set(self.dependencies)) != len(self.dependencies)
                or self.node_id in self.dependencies
                or not 0 <= self.synchronization_nanoseconds
                <= MAX_OPERATOR_SYNC_NANOSECONDS):
            raise ValueError("invalid operator graph node")


@dataclass(frozen=True, slots=True)
class OperatorGraphResult:
    values: dict[str, dict[str, Any]]
    backend_results: tuple[tuple[str, BackendEngineResult[dict[str, Any]]], ...]
    synchronization_nanoseconds: int


class OperatorGraphDispatcher:
    def __init__(self, registry: BackendEngineRegistry[dict[str, Any]]) -> None:
        self._registry = registry

    def execute(
        self,
        nodes: tuple[OperatorGraphNode, ...],
        context: InferenceRequestContext,
        *,
        maximum_synchronization_nanoseconds: int,
    ) -> OperatorGraphResult:
        ordered = _topological_order(nodes)
        _validate_phase_dependencies(nodes)
        synchronization = sum(node.synchronization_nanoseconds for node in ordered)
        if (not 0 <= maximum_synchronization_nanoseconds
                <= MAX_OPERATOR_SYNC_NANOSECONDS
                or synchronization > maximum_synchronization_nanoseconds):
            raise ValueError("operator graph synchronization budget exceeded")
        values: dict[str, dict[str, Any]] = {}
        results = []
        for node in ordered:
            context.raise_if_cancelled()
            if not isinstance(node.request.payload, dict):
                raise ValueError("operator graph payload must be an object")
            dependency_values = {
                dependency: values[dependency] for dependency in node.dependencies
            }
            payload = dict(node.request.payload)
            if "_dependencies" in payload:
                raise ValueError("operator graph payload reserves _dependencies")
            payload["_dependencies"] = dependency_values
            result = self._registry.execute(
                replace(node.request, payload=payload), context
            )
            values[node.node_id] = result.value
            results.append((node.node_id, result))
        return OperatorGraphResult(values, tuple(results), synchronization)


def _topological_order(nodes: tuple[OperatorGraphNode, ...]) -> tuple[OperatorGraphNode, ...]:
    if (not nodes or len(nodes) > MAX_OPERATOR_GRAPH_NODES
            or len({node.node_id for node in nodes}) != len(nodes)
            or sum(len(node.dependencies) for node in nodes) > MAX_OPERATOR_GRAPH_EDGES):
        raise ValueError("invalid operator graph")
    by_id = {node.node_id: node for node in nodes}
    if any(dependency not in by_id for node in nodes for dependency in node.dependencies):
        raise ValueError("operator graph dependency is missing")
    remaining = {node.node_id: set(node.dependencies) for node in nodes}
    ordered = []
    while remaining:
        ready = sorted(identifier for identifier, dependencies in remaining.items() if not dependencies)
        if not ready:
            raise ValueError("operator graph contains a cycle")
        for identifier in ready:
            ordered.append(by_id[identifier])
            del remaining[identifier]
        for dependencies in remaining.values():
            dependencies.difference_update(ready)
    return tuple(ordered)


def _validate_phase_dependencies(nodes: tuple[OperatorGraphNode, ...]) -> None:
    """Reject semantically reversed phase edges before any backend executes."""
    by_id = {node.node_id: node for node in nodes}
    for node in nodes:
        allowed = _ALLOWED_PHASE_PREDECESSORS.get(node.request.phase)
        if allowed is None:
            continue
        for dependency in node.dependencies:
            dependency_phase = by_id[dependency].request.phase
            if dependency_phase not in allowed:
                raise ValueError(
                    "operator graph phase dependency is invalid: "
                    f"{dependency_phase.value}->{node.request.phase.value}"
                )
