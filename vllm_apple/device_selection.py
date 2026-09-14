"""Conservative promotion gate for measured heterogeneous device placement."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .device_benchmark import DeviceBenchmarkReport
from .execution import ExecutionBackend


@dataclass(frozen=True, slots=True)
class DevicePlacementCandidate:
    backend: ExecutionBackend
    report_id: str
    effective_latency_nanoseconds: int
    eligible: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            len(self.report_id) != 24
            or self.effective_latency_nanoseconds <= 0
            or self.eligible == bool(self.rejection_reasons)
        ):
            raise ValueError("invalid device placement candidate")


@dataclass(frozen=True, slots=True)
class DevicePlacementDecision:
    selected: ExecutionBackend
    baseline: ExecutionBackend
    improvement_ratio: float
    candidates: tuple[DevicePlacementCandidate, ...]

    def __post_init__(self) -> None:
        if (
            not 0 <= self.improvement_ratio < 1
            or not self.candidates
            or self.selected not in tuple(
                value.backend for value in self.candidates if value.eligible
            )
            or self.baseline not in tuple(value.backend for value in self.candidates)
        ):
            raise ValueError("invalid device placement decision")


def select_measured_device_backend(
    reports: tuple[DeviceBenchmarkReport, ...],
    *,
    baseline_backend: ExecutionBackend = ExecutionBackend.CPU,
    minimum_samples: int = 3,
    minimum_improvement_ratio: float = 0.05,
    cold_load_amortization_runs: int = 1,
    require_peak_memory: bool = False,
) -> DevicePlacementDecision:
    """Promote only a correctness-equivalent end-to-end measured backend."""
    if (
        not 1 <= len(reports) <= len(ExecutionBackend)
        or type(minimum_samples) is not int
        or not 1 <= minimum_samples <= 64
        or not math.isfinite(minimum_improvement_ratio)
        or not 0 <= minimum_improvement_ratio < 1
        or type(cold_load_amortization_runs) is not int
        or not 1 <= cold_load_amortization_runs <= 1_000_000
    ):
        raise ValueError("invalid device placement selection bounds")
    identities = {
        (
            report.hardware_fingerprint,
            report.environment_fingerprint,
            report.config.operator,
            report.config.phase,
            report.config.precision,
            report.config.dimensions,
            report.config.batch_size,
        )
        for report in reports
    }
    if len(identities) != 1 or len({report.config.backend for report in reports}) != len(reports):
        raise ValueError("device placement reports are not directly comparable")
    baseline = next(
        (report for report in reports if report.config.backend is baseline_backend),
        None,
    )
    if baseline is None or len(baseline.measurements) < minimum_samples:
        raise ValueError("device placement baseline evidence is unavailable")
    baseline_payload = baseline.to_dict()
    baseline_latency = _effective_latency(baseline, cold_load_amortization_runs)
    baseline_digest = baseline_payload["output_digest"]
    baseline_peak = baseline_payload["peak_memory_bytes"]
    candidates: list[DevicePlacementCandidate] = []
    eligible_reports: list[tuple[int, DeviceBenchmarkReport]] = []
    for report in reports:
        payload = report.to_dict()
        latency = _effective_latency(report, cold_load_amortization_runs)
        reasons: list[str] = []
        if len(report.measurements) < minimum_samples:
            reasons.append("insufficient_samples")
        if payload["output_digest"] != baseline_digest:
            reasons.append("output_mismatch")
        peak = payload["peak_memory_bytes"]
        if require_peak_memory and peak is None:
            reasons.append("peak_memory_unknown")
        elif baseline_peak is not None and peak is not None and peak > baseline_peak:
            reasons.append("peak_memory_regression")
        if report.config.backend is not baseline_backend and (
            latency > baseline_latency * (1 - minimum_improvement_ratio)
        ):
            reasons.append("insufficient_latency_improvement")
        eligible = not reasons
        candidates.append(DevicePlacementCandidate(
            report.config.backend,
            report.report_id,
            latency,
            eligible,
            tuple(reasons),
        ))
        if eligible:
            eligible_reports.append((latency, report))
    if not any(report.config.backend is baseline_backend for _, report in eligible_reports):
        raise RuntimeError("device placement baseline failed its correctness gate")
    _, selected = min(
        eligible_reports,
        key=lambda value: (value[0], value[1].config.backend.value),
    )
    selected_latency = _effective_latency(selected, cold_load_amortization_runs)
    return DevicePlacementDecision(
        selected.config.backend,
        baseline_backend,
        max(0.0, 1 - selected_latency / baseline_latency),
        tuple(candidates),
    )


def _effective_latency(report: DeviceBenchmarkReport, amortization_runs: int) -> int:
    median = report.to_dict()["median_total_nanoseconds"]
    load = report.cold_load_nanoseconds or 0
    return int(median) + math.ceil(load / amortization_runs)
