from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

from .generative_collector import collect_generative_sample
from .generative_evaluation import (
    MAX_DURATION_MS,
    MEMORY_PRESSURES,
    THERMAL_STATES,
    GenerativeEvaluationProvenance,
    GenerativeEvaluationReport,
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


def resolve_generative_qualification_mode(
    plan: GenerativeQualificationPlan, mode: str | None = None
) -> str:
    """Select a source-free qualification mode without guessing required inputs."""
    if mode is not None:
        if mode not in plan.candidate.modes:
            raise ValueError("generative mode is not supported by the candidate")
        return mode
    preferred = "text-to-video" if plan.candidate.modality == "video" else "text-to-image"
    if preferred not in plan.candidate.modes:
        raise ValueError("candidate has no default source-free qualification mode")
    return preferred


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
    mode: str | None = None,
    input_image_path: str | Path | None = None,
    timeout_seconds: float = 3600.0,
    adapter_factory: Callable[..., SubprocessGenerativeTelemetryAdapter] = (
        SubprocessGenerativeTelemetryAdapter
    ),
    pressure_probe: Callable[[], str] = lambda: detect_hardware().memory.pressure.value,
    recovery_timeout_seconds: float = 300.0,
    recovery_poll_seconds: float = 5.0,
    disk_offload: bool = False,
    phase_encoder_command: tuple[str, ...] | None = None,
) -> GenerativeEvaluationReport:
    if not plan.eligible:
        raise ValueError("cannot run an ineligible generative qualification plan")
    if not 2 <= sample_count <= 32:
        raise ValueError("memory-stability qualification requires between 2 and 32 samples")
    qualification_mode = resolve_generative_qualification_mode(plan, mode)
    if phase_encoder_command is not None and (
        plan.candidate.candidate_id != "qwen-image-2.1"
        or qualification_mode != "text-to-image"
        or disk_offload
    ):
        raise ValueError("two-phase qualification requires Qwen-Image text-to-image")
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
            mode=qualification_mode,
            prompt=prompt,
            seed=42 + sample_index,
            sample_index=sample_index,
            input_image_path=input_image_path,
            disk_offload=disk_offload,
        )
        request_path = private / f"request-{sample_index}.json"
        handoff = private / f"handoff-{sample_index}"
        sample_started = time.monotonic()
        encoder_telemetry = None
        try:
            phase_arguments: tuple[str, ...] = ()
            if phase_encoder_command is not None:
                handoff.mkdir(mode=0o700)
                handoff.chmod(0o700)
                encoder_request = private / f"encoder-request-{sample_index}.json"
                save_private_generative_request(request, encoder_request)
                encoder_command = (
                    *phase_encoder_command, "--request", str(encoder_request),
                    "--workspace-root", str(workspace), "--handoff-root", str(handoff),
                )
                with tempfile.TemporaryFile(mode="w+b", dir=private) as diagnostic, tempfile.TemporaryFile(
                    mode="w+b", dir=private
                ) as phase_output:
                    process = subprocess.Popen(
                        encoder_command, cwd=workspace, stdin=subprocess.DEVNULL,
                        stdout=phase_output, stderr=diagnostic,
                        start_new_session=True,
                    )
                    try:
                        try:
                            exit_code = process.wait(timeout=timeout_seconds)
                        except subprocess.TimeoutExpired as error:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait(timeout=2)
                            raise RuntimeError("Qwen-Image encoder phase timed out") from error
                    finally:
                        encoder_request.unlink(missing_ok=True)
                    diagnostic.seek(0, os.SEEK_END)
                    diagnostic.seek(max(0, diagnostic.tell() - 4096))
                    diagnostic_tail = diagnostic.read(4096)
                    phase_output.seek(0, os.SEEK_END)
                    phase_size = phase_output.tell()
                    phase_output.seek(0)
                    phase_bytes = phase_output.read(4097) if phase_size <= 4096 else b""
                if exit_code != 0:
                    from .generative_subprocess_adapter import (
                        SubprocessGenerativeTelemetryAdapter,
                    )

                    detail = SubprocessGenerativeTelemetryAdapter._diagnostic(
                        diagnostic_tail
                    )
                    message = detail[1] if detail and detail[1] else "no bounded diagnostic"
                    raise RuntimeError(f"Qwen-Image encoder phase failed: {message}")
                try:
                    encoder_telemetry = json.loads(phase_bytes)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise RuntimeError("Qwen-Image encoder telemetry is invalid") from error
                if (
                    not isinstance(encoder_telemetry, dict)
                    or set(encoder_telemetry) != {
                        "schema_version", "phase", "elapsed_ms", "peak_rss_bytes",
                        "memory_pressure", "thermal_state",
                    }
                    or encoder_telemetry["schema_version"] != 1
                    or encoder_telemetry["phase"] != "text_encoder"
                    or not isinstance(encoder_telemetry["elapsed_ms"], (float, int))
                    or not math.isfinite(encoder_telemetry["elapsed_ms"])
                    or not 0 < encoder_telemetry["elapsed_ms"] <= MAX_DURATION_MS
                    or type(encoder_telemetry["peak_rss_bytes"]) is not int
                    or not 0 < encoder_telemetry["peak_rss_bytes"] <= 16_384 * 1024**3
                    or encoder_telemetry["memory_pressure"] not in MEMORY_PRESSURES
                    or encoder_telemetry["thermal_state"] not in THERMAL_STATES
                ):
                    raise RuntimeError("Qwen-Image encoder telemetry schema is invalid")
                if encoder_telemetry["peak_rss_bytes"] > plan.artifact_admission.memory_hard_ceiling_bytes:
                    raise RuntimeError("Qwen-Image encoder exceeded memory hard ceiling")
                if not (handoff / "prompt-embeddings.json").is_file():
                    raise RuntimeError("Qwen-Image encoder phase produced no handoff")
                wait_for_memory_pressure_recovery(
                    pressure_probe=pressure_probe,
                    timeout_seconds=recovery_timeout_seconds,
                    poll_seconds=recovery_poll_seconds,
                )
                phase_arguments = (
                    "--phase-handoff", str(handoff / "prompt-embeddings.json")
                )
            save_private_generative_request(request, request_path)
            command = (
                *worker_command, "--request", str(request_path),
                "--workspace-root", str(workspace), *phase_arguments,
            )
            adapter = adapter_factory(
                command,
                timeout_seconds=timeout_seconds,
                cwd=workspace,
            )
            sample = collect_generative_sample(
                plan, sample_index=sample_index, events=adapter.events(),
            )
            if encoder_telemetry is not None:
                delay_ms = (time.monotonic() - sample_started) * 1000.0 - sample.wall_time_ms
                delay_ms = max(0.0, delay_ms)
                memory_rank = {"normal": 0, "warning": 1, "unknown": 2, "critical": 3}
                thermal_rank = {"nominal": 0, "fair": 1, "unknown": 2, "serious": 3, "critical": 4}
                sample = replace(
                    sample,
                    wall_time_ms=sample.wall_time_ms + delay_ms,
                    first_output_ms=sample.first_output_ms + delay_ms,
                    peak_rss_bytes=max(sample.peak_rss_bytes, encoder_telemetry["peak_rss_bytes"]),
                    memory_pressure=max(
                        (sample.memory_pressure, encoder_telemetry["memory_pressure"]),
                        key=memory_rank.__getitem__,
                    ),
                    thermal_state=max(
                        (sample.thermal_state, encoder_telemetry["thermal_state"]),
                        key=thermal_rank.__getitem__,
                    ),
                )
            samples.append(sample)
        finally:
            request_path.unlink(missing_ok=True)
            if phase_encoder_command is not None:
                shutil.rmtree(handoff, ignore_errors=False)

    report = evaluate_generative_qualification(plan, tuple(samples), provenance)
    save_generative_evaluation_report(report, Path(report_path))
    return report
