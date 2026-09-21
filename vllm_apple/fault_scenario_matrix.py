"""Bounded cross-component fault qualification matrix."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum

MAX_FAULT_SCENARIOS = 64


class PlatformFaultPoint(str, Enum):
    PROFILE_PERSISTENCE = "profile_persistence"
    SCHEDULER_ADMISSION = "scheduler_admission"
    WORKER_CRASH = "worker_crash"
    CLIENT_DISCONNECT = "client_disconnect"


@dataclass(frozen=True, slots=True)
class PlatformFaultScenario:
    scenario_id: str
    point: PlatformFaultPoint
    action: str
    expected_recovery: str

    def __post_init__(self) -> None:
        if (not _identifier(self.scenario_id) or not isinstance(self.point, PlatformFaultPoint)
                or not _identifier(self.action) or not _identifier(self.expected_recovery)):
            raise ValueError("invalid platform fault scenario")


@dataclass(frozen=True, slots=True)
class PlatformFaultObservation:
    recovery: str
    service_ready: bool
    active_reservations: int
    temporary_files: int
    stored_prompts: int
    stored_outputs: int
    last_known_good_restored: bool

    def __post_init__(self) -> None:
        if (not _identifier(self.recovery)
                or any(type(value) is not int or value < 0 for value in (
                    self.active_reservations, self.temporary_files,
                    self.stored_prompts, self.stored_outputs,
                ))
                or type(self.service_ready) is not bool
                or type(self.last_known_good_restored) is not bool):
            raise ValueError("invalid platform fault observation")


@dataclass(frozen=True, slots=True)
class PlatformFaultScenarioResult:
    scenario_id: str
    point: str
    passed: bool
    reason: str
    observation: PlatformFaultObservation | None


@dataclass(frozen=True, slots=True)
class PlatformFaultMatrixReport:
    report_id: str
    results: tuple[PlatformFaultScenarioResult, ...]
    passed: bool


class PlatformFaultScenarioMatrix:
    def __init__(self, scenarios: tuple[PlatformFaultScenario, ...]) -> None:
        required = set(PlatformFaultPoint)
        if (not scenarios or len(scenarios) > MAX_FAULT_SCENARIOS
                or len({item.scenario_id for item in scenarios}) != len(scenarios)
                or {item.point for item in scenarios} != required):
            raise ValueError("platform fault matrix must cover every required point")
        self._scenarios = scenarios

    def run(
        self,
        handlers: dict[
            PlatformFaultPoint,
            Callable[[PlatformFaultScenario], PlatformFaultObservation],
        ],
    ) -> PlatformFaultMatrixReport:
        if set(handlers) != set(PlatformFaultPoint) or any(
            not callable(handler) for handler in handlers.values()
        ):
            raise ValueError("platform fault handlers are incomplete")
        results = []
        for scenario in self._scenarios:
            try:
                observation = handlers[scenario.point](scenario)
                if not isinstance(observation, PlatformFaultObservation):
                    raise ValueError("handler returned an invalid observation")
                reason = _failure_reason(scenario, observation)
            except Exception as error:
                results.append(PlatformFaultScenarioResult(
                    scenario.scenario_id, scenario.point.value, False,
                    f"handler_{type(error).__name__}", None,
                ))
                continue
            results.append(PlatformFaultScenarioResult(
                scenario.scenario_id, scenario.point.value,
                reason is None, reason or "passed", observation,
            ))
        payload = [{
            **asdict(result),
            "observation": (
                asdict(result.observation) if result.observation is not None else None
            ),
        } for result in results]
        report_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return PlatformFaultMatrixReport(
            report_id, tuple(results), all(result.passed for result in results)
        )


def default_platform_fault_scenarios() -> tuple[PlatformFaultScenario, ...]:
    return (
        PlatformFaultScenario(
            "profile-write-interrupted", PlatformFaultPoint.PROFILE_PERSISTENCE,
            "interrupt_before_atomic_replace", "last_known_good",
        ),
        PlatformFaultScenario(
            "scheduler-capacity-rejected", PlatformFaultPoint.SCHEDULER_ADMISSION,
            "reject_after_queue_before_reservation", "structured_rejection",
        ),
        PlatformFaultScenario(
            "worker-crash-after-reservation", PlatformFaultPoint.WORKER_CRASH,
            "terminate_worker_after_reservation", "worker_restarted",
        ),
        PlatformFaultScenario(
            "client-disconnect-mid-stream", PlatformFaultPoint.CLIENT_DISCONNECT,
            "disconnect_after_first_chunk", "request_cancelled",
        ),
    )


def _failure_reason(
    scenario: PlatformFaultScenario, observation: PlatformFaultObservation
) -> str | None:
    if observation.recovery != scenario.expected_recovery:
        return "unexpected_recovery"
    if not observation.service_ready:
        return "service_not_ready"
    if observation.active_reservations:
        return "reservation_leak"
    if observation.temporary_files:
        return "temporary_file_leak"
    if observation.stored_prompts or observation.stored_outputs:
        return "sensitive_output_persisted"
    if (scenario.point is PlatformFaultPoint.PROFILE_PERSISTENCE
            and not observation.last_known_good_restored):
        return "last_known_good_not_restored"
    return None


def _identifier(value: object) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 128 and all(
        character.isalnum() or character in "._-" for character in value
    )
