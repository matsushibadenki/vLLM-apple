from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from .backend import BackendConfig, BackendProcess
from .backend_memory import KVCacheCapacityResolver, MLXMemoryMetricsAdapter, VLLMMemoryMetricsAdapter
from .context_reevaluation import ContextCapacityReevaluator
from .hardware import default_application_support, detect_hardware, detect_memory
from .model import (
    DEFAULT_UNINSPECTED_CONTEXT,
    ModelCapabilityError,
    ModelInspectionError,
    assess_model_memory_fit,
    ensure_model_backend_compatible,
    inspect_model,
)
from .phase_probe import PhaseProbeConfig, run_phase_probe
from .promotion_probe import PromotionProbeConfig, run_serving_promotion_probe
from .quality_smoke import run_serving_quality_smoke
from .vllm_metal_v2_tuning import build_v2_hardware_fingerprint
from .soak import SoakConfig, run_soak


@dataclass(frozen=True, slots=True)
class QualificationConfig:
    model: str
    executable: Path
    port: int = 8001
    max_model_len: int | None = None
    startup_timeout_seconds: float = 600
    duration_seconds: float = 1800
    warmup_seconds: float = 30
    concurrency: int = 4
    request_timeout_seconds: float = 300
    max_rss_growth_bytes: int = 256 * 1024 * 1024
    require_30_minute_window: bool = True
    vllm_version: str | None = None
    allow_context_reduction: bool = False
    backend_kind: str = "vllm_metal"
    architecture_features: tuple[str, ...] = ()
    phase_samples: int = 0
    phase_output_tokens: int = 32
    quality_smoke: bool = False
    requested_modes: tuple[str, ...] = ("text",)
    backend_versions: dict[str, str | None] | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        for value in (
            self.startup_timeout_seconds,
            self.duration_seconds,
            self.request_timeout_seconds,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("qualification durations must be positive and finite")
        if not math.isfinite(self.warmup_seconds) or self.warmup_seconds < 0:
            raise ValueError("warmup must be non-negative and finite")
        if not 1 <= self.concurrency <= 256:
            raise ValueError("concurrency must be between 1 and 256")
        if not 0 <= self.phase_samples <= 1_000:
            raise ValueError("phase samples must be between 0 and 1000")
        if not 1 <= self.phase_output_tokens <= 4096:
            raise ValueError("phase output tokens must be between 1 and 4096")
        if self.max_rss_growth_bytes < 0:
            raise ValueError("RSS growth limit cannot be negative")
        if self.backend_kind not in {"vllm_metal", "mlx_lm"}:
            raise ValueError("unsupported qualification backend")
        if (
            not self.requested_modes
            or len(set(self.requested_modes)) != len(self.requested_modes)
            or any(mode not in {"text", "vision", "mtp", "yarn"} for mode in self.requested_modes)
        ):
            raise ValueError("qualification modes are invalid")
        if self.require_30_minute_window and self.duration_seconds < 1800:
            raise ValueError("real-model certification requires at least 1800 seconds")


def qualify_model(
    config: QualificationConfig,
    *,
    process_factory: Callable[[BackendConfig], BackendProcess] = BackendProcess,
) -> dict[str, object]:
    qualification_hardware = None
    model_memory_fit: dict[str, object] | None = None
    inspected = inspect_model(config.model)
    ensure_model_backend_compatible(
        inspected,
        backend=config.backend_kind,
        available_features=frozenset(config.architecture_features),
        requested_modes=frozenset(config.requested_modes),
    )
    qualification_hardware = detect_hardware()
    context_tokens = (
        config.max_model_len
        or inspected.architecture_capability.native_context_tokens
        or inspected.memory_spec.model_max_context
        or DEFAULT_UNINSPECTED_CONTEXT
    )
    memory_model = inspected
    if "mtp" not in config.requested_modes and inspected.state_memory_spec is not None:
        memory_model = replace(
            inspected,
            state_memory_spec=replace(
                inspected.state_memory_spec,
                mtp_working_set_bytes=0,
            ),
        )
    fit = assess_model_memory_fit(
        memory_model, qualification_hardware, context_tokens=context_tokens
    )
    model_memory_fit = {
        "artifact_bytes": fit.artifact_bytes,
        "estimated_resident_bytes": fit.estimated_resident_bytes,
        "hard_ceiling_bytes": fit.hard_ceiling_bytes,
        "context_tokens": fit.context_tokens,
        "fits": fit.fits,
    }
    if not fit.fits:
        raise ModelCapabilityError("model_memory_hard_ceiling_exceeded")
    backend = process_factory(
        BackendConfig(
            model=config.model,
            executable=config.executable,
            port=config.port,
            max_model_len=config.max_model_len,
            startup_timeout=config.startup_timeout_seconds,
            backend_kind=config.backend_kind,
        )
    )
    load_started = time.monotonic()
    load_seconds = 0.0
    soak: dict[str, object] | None = None
    promotion: dict[str, object] | None = None
    phase_profile: dict[str, object] | None = None
    quality_smoke: dict[str, object] | None = None
    shutdown_clean = False
    context = _pending_context_report(config)
    api_model = "default_model" if config.backend_kind == "mlx_lm" else config.model
    try:
        backend.start()
        load_seconds = time.monotonic() - load_started
        pid = backend.pid
        if pid is None:
            raise RuntimeError("backend started without a process ID")
        context = evaluate_qualification_context(config, backend.base_url)
        if context["passed"]:
            promotion = run_serving_promotion_probe(
                PromotionProbeConfig(
                    base_url=backend.base_url,
                    model=api_model,
                    timeout_seconds=config.request_timeout_seconds,
                    supports_seeded_sampling=config.backend_kind != "mlx_lm",
                )
            )
            if not promotion["passed"]:
                raise RuntimeError("backend sampling/streaming promotion probe failed")
            if config.phase_samples:
                hardware_fingerprint = build_v2_hardware_fingerprint(
                    qualification_hardware or detect_hardware()
                )
                phase_profile = run_phase_probe(
                    PhaseProbeConfig(
                        base_url=backend.base_url,
                        model=api_model,
                        hardware_fingerprint=hardware_fingerprint,
                        backend=config.backend_kind,
                        samples=config.phase_samples,
                        maximum_output_tokens=config.phase_output_tokens,
                        timeout_seconds=config.request_timeout_seconds,
                        target_pid=pid,
                    )
                )
            if config.quality_smoke:
                hardware_fingerprint = build_v2_hardware_fingerprint(
                    qualification_hardware or detect_hardware()
                )
                quality_smoke = run_serving_quality_smoke(
                    PhaseProbeConfig(
                        base_url=backend.base_url,
                        model=api_model,
                        hardware_fingerprint=hardware_fingerprint,
                        backend=config.backend_kind,
                        samples=1,
                        maximum_output_tokens=8,
                        timeout_seconds=config.request_timeout_seconds,
                        target_pid=pid,
                    )
                )
                if not quality_smoke["passed"]:
                    raise RuntimeError("backend multilingual quality smoke failed")
            soak = run_soak(
                SoakConfig(
                    base_url=backend.base_url,
                    duration_seconds=config.duration_seconds,
                    warmup_seconds=config.warmup_seconds,
                    concurrency=config.concurrency,
                    request_timeout_seconds=config.request_timeout_seconds,
                    mode="chat-mixed",
                    model=api_model,
                    target_pid=pid,
                    max_rss_growth_bytes=config.max_rss_growth_bytes,
                    require_30_minute_window=config.require_30_minute_window,
                )
            )
    finally:
        backend.stop()
        shutdown_clean = not backend.running
    return {
        "schema_version": 1,
        "model": config.model,
        "backend": config.backend_kind,
        "backend_versions": config.backend_versions,
        "requested_modes": list(config.requested_modes),
        "load_seconds": round(load_seconds, 3),
        "shutdown_clean": shutdown_clean,
        "promotion_probe": promotion,
        "phase_profile": phase_profile,
        "model_memory_fit": model_memory_fit,
        "quality_smoke": quality_smoke,
        "soak": soak,
        "context_reevaluation": context,
        "passed": bool(
            promotion
            and promotion["passed"]
            and soak
            and soak["passed"]
            and shutdown_clean
            and context["passed"]
        ),
    }


def _pending_context_report(config: QualificationConfig) -> dict[str, object]:
    return {
        "enabled": False,
        "status": "unavailable",
        "configured_context_tokens": config.max_model_len,
        "effective_context_tokens": config.max_model_len,
        "capacity_context_tokens": None,
        "kv_capacity_bytes": None,
        "kv_bytes_per_token": None,
        "weights_bytes": None,
        "source": None,
        "reevaluations": 0,
        "passed": True,
    }


def evaluate_qualification_context(
    config: QualificationConfig, base_url: str
) -> dict[str, object]:
    report = _pending_context_report(config)
    try:
        inspected = inspect_model(config.model)
        configured = config.max_model_len or inspected.memory_spec.model_max_context
        if configured is not None and config.backend_kind == "mlx_lm":
            sample = MLXMemoryMetricsAdapter(
                base_url, timeout=min(5.0, config.request_timeout_seconds)
            ).sample()
            memory = detect_memory()
            reserve = max(2 * 1024**3, memory.total_bytes // 10)
            available_for_growth = max(0, memory.available_bytes - reserve)
            capacity = (sample.kv_used_bytes or 0) + available_for_growth
            reevaluator = ContextCapacityReevaluator(
                configured,
                inspected.memory_spec.kv_bytes_per_token,
                inspected.memory_spec.weights_bytes,
            )
            reevaluator.update(capacity, source="mlx-allocator-os-available-v1")
            snapshot = reevaluator.snapshot().to_dict()
            return {
                **snapshot,
                "passed": snapshot["status"] != "reduced" or config.allow_context_reduction,
            }
        if configured is None or config.vllm_version is None:
            return report
        resolver = KVCacheCapacityResolver(
            config.vllm_version,
            inspected.memory_spec.kv_bytes_per_token,
            str(inspected.config.get("model_type") or ""),
        )
        sample = VLLMMemoryMetricsAdapter(
            base_url, timeout=min(5.0, config.request_timeout_seconds), kv_capacity_resolver=resolver
        ).sample()
        if sample.kv_capacity_bytes is None:
            return report
        reevaluator = ContextCapacityReevaluator(
            configured,
            inspected.memory_spec.kv_bytes_per_token,
            inspected.memory_spec.weights_bytes,
        )
        reevaluator.update(
            sample.kv_capacity_bytes, source="vllm-prometheus-cache-config-v1"
        )
        snapshot = reevaluator.snapshot().to_dict()
        return {
            **snapshot,
            "passed": snapshot["status"] != "reduced" or config.allow_context_reduction,
        }
    except (ModelInspectionError, OSError, RuntimeError, ValueError):
        return report


def save_qualification_report(report: dict[str, object], path: Path) -> Path:
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not parent_existed:
        path.parent.chmod(0o700)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def default_qualification_report_path(
    model: str, *, application_support: Path | None = None
) -> Path:
    """Return a collision-resistant, filesystem-safe path for a qualification run."""
    root = application_support or default_application_support()
    fingerprint = hashlib.sha256(model.encode("utf-8")).hexdigest()[:16]
    return root / "qualification-reports" / f"{time.time_ns()}-{fingerprint}.json"
