"""Capability-gated symbolic fusion pass for measured MLX kernel groups."""
from __future__ import annotations

from dataclasses import dataclass

from .execution import ExecutionBackend
from .kernel_probe import KernelCapabilityRegistry


@dataclass(frozen=True, slots=True)
class FusionRule:
    pattern: tuple[str, ...]
    fused_operator: str

    def __post_init__(self) -> None:
        if (
            len(self.pattern) < 2
            or len(self.pattern) > 8
            or any(not item or len(item) > 128 for item in self.pattern)
            or not self.fused_operator
            or len(self.fused_operator) > 128
        ):
            raise ValueError("invalid graph fusion rule")


DEFAULT_MLX_FUSION_RULES = (
    FusionRule(("dequant_q4", "matmul", "silu"), "fused_q4_silu"),
    FusionRule(("rms_norm", "rope"), "fused_rmsnorm_rope"),
    FusionRule(("moe_route", "expert_gemm"), "fused_moe"),
)


@dataclass(frozen=True, slots=True)
class GraphFusionResult:
    operators: tuple[str, ...]
    applied: tuple[str, ...]


class CapabilityGatedGraphFusionPass:
    """Apply only non-overlapping rules with passing environment-bound evidence."""

    def __init__(
        self,
        registry: KernelCapabilityRegistry,
        rules: tuple[FusionRule, ...] = DEFAULT_MLX_FUSION_RULES,
    ) -> None:
        if (
            not isinstance(registry, KernelCapabilityRegistry)
            or not isinstance(rules, tuple)
            or not 1 <= len(rules) <= 64
            or any(not isinstance(rule, FusionRule) for rule in rules)
            or len({rule.pattern for rule in rules}) != len(rules)
        ):
            raise ValueError("invalid graph fusion pass configuration")
        self._registry = registry
        self._rules = tuple(sorted(rules, key=lambda rule: -len(rule.pattern)))

    def apply(self, operators: tuple[str, ...]) -> GraphFusionResult:
        if (
            not isinstance(operators, tuple)
            or len(operators) > 4096
            or any(not isinstance(item, str) or not item for item in operators)
        ):
            raise ValueError("invalid operator graph")
        output: list[str] = []
        applied: list[str] = []
        index = 0
        while index < len(operators):
            selected = next(
                (
                    rule for rule in self._rules
                    if operators[index:index + len(rule.pattern)] == rule.pattern
                    and self._registry.is_usable(
                        ExecutionBackend.NATIVE_MLX, rule.fused_operator
                    )
                ),
                None,
            )
            if selected is None:
                output.append(operators[index])
                index += 1
                continue
            output.append(selected.fused_operator)
            applied.append(selected.fused_operator)
            index += len(selected.pattern)
        return GraphFusionResult(tuple(output), tuple(applied))
