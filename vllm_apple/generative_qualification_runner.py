from __future__ import annotations

from pathlib import Path
import time
from typing import Callable

from .generative_collector import collect_generative_sample
from .generative_evaluation import (
    GenerativeEvaluationReport,
    GenerativeEvaluationProvenance,
    evaluate_generative_qualification,
    save_generative_evaluation_report,
)
from .generative_qualification import GenerativeQualificationPlan
from .generative_subprocess_adapter import SubprocessGenerativeTelemetryAdapter
from .generative_worker_protocol import (
    build_generative_worker_request,
    save_private_generative_request,
)
from .hardware import detect_hardware


def wait_for_memory_pressure_recovery(
    *,
    pressure_probe: Callable[[], str],
    timeout_seconds: float,
    poll_seconds: float,
    stable_observations: int = 2,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if timeout_seconds <= 0 or poll_seconds <= 0 or not 1 <= stable_observations <= 16:
        raise ValueError("memory recovery bounds are invalid")
    deadline = monotonic() + timeout_seconds
    consecutive_normal = 0
    while True:
        pressure = pressure_probe()
        if pressure not in {"normal", "warning", "critical", "unknown"}:
            raise ValueError("memory pressure probe returned an invalid state")
        consecutive_normal = consecutive_normal + 1 if pressure == "normal" else 0
        if consecutive_normal >= stable_observations:
            return
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise RuntimeError("memory pressure did not recover before the bounded timeout")
        sleep(min(poll_seconds, remaining))


def run_generative_qualification(
    plan: GenerativeQualificationPlan,
    *,
    workspace_root: str | Path,
    model_root: str | Path,
    private_root: str | Path,
    report_path: str | Path,
    prompt: str,
    sample_count: int,
    worker_command: tuple[str, ...],
    provenance: GenerativeEvaluationProvenance,
    timeout_seconds: float = 3600.0,
    adapter_factory: Callable[..., SubprocessGenerativeTelemetryAdapter] = (
        SubprocessGenerativeTelemetryAdapter
    ),
    pressure_probe: Callable[[], str] = lambda: detect_hardware().memory.pressure.value,
    recovery_timeout_seconds: float = 300.0,
    recovery_poll_seconds: float = 5.0,
) -> GenerativeEvaluationReport:
    if not plan.eligible:
        raise ValueError("cannot run an ineligible generative qualification plan")
    if not 2 <= sample_count <= 32:
        raise ValueError("memory-stability qualification requires between 2 and 32 samples")
    workspace = Path(workspace_root).expanduser().resolve(strict=True)
    private = Path(private_root).expanduser().resolve()
    if private == workspace or not private.is_relative_to(workspace):
        raise ValueError("generative private root must be inside the workspace")
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.chmod(0o700)
    output = private / "outputs"
    output.mkdir(mode=0o700, exist_ok=True)
    output.chmod(0o700)

    samples = []
    for sample_index in range(sample_count):
        if sample_index:
            wait_for_memory_pressure_recovery(
                pressure_probe=pressure_probe,
                timeout_seconds=recovery_timeout_seconds,
                poll_seconds=recovery_poll_seconds,
            )
        request = build_generative_worker_request(
            plan,
            workspace_root=workspace,
            model_root=model_root,
            output_root=output,
            mode="text-to-image",
            prompt=prompt,
            seed=42 + sample_index,
            sample_index=sample_index,
        )
        request_path = private / f"request-{sample_index}.json"
        save_private_generative_request(request, request_path)
        command = (*worker_command, "--request", str(request_path), "--workspace-root", str(workspace))
        try:
            adapter = adapter_factory(
                command,
                timeout_seconds=timeout_seconds,
                cwd=workspace,
            )
            samples.append(
                collect_generative_sample(
                    plan,
                    sample_index=sample_index,
                    events=adapter.events(),
                )
            )
        finally:
            request_path.unlink(missing_ok=True)

    report = evaluate_generative_qualification(plan, tuple(samples), provenance)
    save_generative_evaluation_report(report, Path(report_path))
    return report
