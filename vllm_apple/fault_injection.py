"""Bounded deterministic fault injection for lifecycle qualification."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum


MAX_FAULT_RULES = 32


class FaultPoint(str, Enum):
    BACKEND_EXECUTE = "backend_execute"
    BACKEND_STOP = "backend_stop"


class FaultAction(str, Enum):
    RETRYABLE = "retryable"
    FATAL = "fatal"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class FaultRule:
    point: FaultPoint
    action: FaultAction
    trigger_on_hit: int = 1
    repeat: bool = False

    def __post_init__(self) -> None:
        if (not isinstance(self.point, FaultPoint)
                or not isinstance(self.action, FaultAction)
                or type(self.trigger_on_hit) is not int
                or not 1 <= self.trigger_on_hit <= 1_000_000
                or type(self.repeat) is not bool):
            raise ValueError("invalid fault injection rule")


class InjectedFault(RuntimeError):
    def __init__(self, point: FaultPoint, action: FaultAction) -> None:
        super().__init__(f"injected_{point.value}_{action.value}")
        self.point = point
        self.action = action


class DeterministicFaultInjector:
    """Thread-safe one-shot rules with fixed-cardinality non-secret telemetry."""

    def __init__(self, rules: tuple[FaultRule, ...]) -> None:
        if (len(rules) > MAX_FAULT_RULES
                or len({rule.point for rule in rules}) != len(rules)):
            raise ValueError("fault injection rules must be bounded and unique")
        self._rules = {rule.point: rule for rule in rules}
        self._hits = {point: 0 for point in FaultPoint}
        self._injections = {point: 0 for point in FaultPoint}
        self._lock = threading.Lock()

    def hit(self, point: FaultPoint) -> None:
        if not isinstance(point, FaultPoint):
            raise TypeError("fault point must use the declared enum")
        with self._lock:
            self._hits[point] += 1
            rule = self._rules.get(point)
            if rule is None:
                return
            should_inject = (
                self._hits[point] >= rule.trigger_on_hit
                if rule.repeat
                else self._hits[point] == rule.trigger_on_hit
            )
            if not should_inject:
                return
            self._injections[point] += 1
            action = rule.action
        raise InjectedFault(point, action)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "schema_version": 1,
                "armed_points": sorted(point.value for point in self._rules),
                "hits": {point.value: self._hits[point] for point in FaultPoint},
                "injections": {
                    point.value: self._injections[point] for point in FaultPoint
                },
            }
