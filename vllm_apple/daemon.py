from __future__ import annotations

import argparse
import shutil
import signal
import threading
import time
from dataclasses import replace
from pathlib import Path

from .api import create_server, create_unix_server
from .auth import load_or_create_token_file
from .backend import (
    BackendProcess,
    OpenAIProxyEngine,
    make_backend_config,
    supports_kernel_tuning_middleware,
)
from .backend_memory import (
    IOGPUMemoryAdapter,
    KVCacheCapacityResolver,
    MemoryMetricsMonitor,
    VLLMMemoryMetricsAdapter,
)
from .compat import inspect_backend
from .context import recommend_context
from .execution import AppleChipProfile
from .execution_profile import detect_apple_chip_profile
from .hardware import default_application_support, detect_hardware
from .kernel_probe import build_environment_fingerprint
from .kernel_profile import build_model_kernel_shape_profile
from .kv_calibration import (
    calibration_report_directory,
    discover_latest_kv_calibration,
)
from .metal_tuning import (
    MetalTuningReport,
    discover_metal_tuning_report,
    load_metal_tuning_report,
)
from .memory_pressure import MemoryPressureMonitor
from .model import (
    DEFAULT_UNINSPECTED_CONTEXT,
    InspectedModel,
    ModelCapabilityError,
    ModelInspectionError,
    inspect_model,
)
from .model_integrity import verify_model_integrity
from .profile import build_profile
from .runtime_probe import (
    RuntimeEnvironmentVersions,
    RuntimeProbeCoordinator,
    discover_runtime_versions,
)
from .runtime_errors import classify_runtime_failure, persist_crash_diagnostic
from .service import RuntimeService
from .types import RuntimeState
from .vllm_metal_integration import inspect_vllm_metal_integration
from .vllm_metal_v2_adapter import V2MeasurementAdapterError, VLLMMetalV2MeasurementAdapter
from .vllm_metal_v2_observation import default_v2_observation_path, load_v2_observations
from .vllm_metal_v2_orchestration import NativeV2ObservationMonitor
from .vllm_metal_v2_preference import default_native_v2_preference_path
from .vllm_metal_v2_tuning import (
    build_v2_hardware_fingerprint,
    inspect_v2_tuning_quarantine,
    quarantine_v2_tuning_profile,
    restore_quarantined_v2_profile,
    save_v2_tuning_profile,
    tune_v2_observed_shapes,
    VLLMMetalV2TuningProfile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-appled")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-concurrent-requests", type=int, default=32)
    parser.add_argument("model", nargs="?")
    parser.add_argument("--backend-executable")
    parser.add_argument("--backend-port", type=int, default=8001)
    parser.add_argument("--backend-startup-timeout", type=float, default=600.0)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--model-integrity-manifest", type=Path)
    parser.add_argument("--skip-backend-check", action="store_true")
    parser.add_argument("--socket-path")
    parser.add_argument("--session-token")
    parser.add_argument("--session-token-file")
    parser.add_argument("--shutdown-grace-period", type=float, default=30.0)
    parser.add_argument("--skip-runtime-probes", action="store_true")
    tuning = parser.add_mutually_exclusive_group()
    tuning.add_argument("--metal-tuning-report", type=Path)
    tuning.add_argument("--disable-metal-tuning", action="store_true")
    parser.add_argument("--disable-kv-calibration", action="store_true")
    parser.add_argument("--kv-calibration-root", type=Path)
    parser.add_argument("--vllm-metal-source-root", type=Path)
    parser.add_argument("--vllm-metal-v2-helper", type=Path)
    parser.add_argument("--disable-native-v2-idle-tuning", action="store_true")
    parser.add_argument("--native-v2-preference-path", type=Path)
    return parser


def apply_startup_kv_calibration(
    model: InspectedModel,
    hardware_fingerprint: str,
    *,
    root: Path | None = None,
) -> tuple[InspectedModel, dict[str, int | float | str | bool | None]]:
    provenance: dict[str, int | float | str | bool | None] = {
        "enabled": True,
        "status": "not_found",
        "backend": "vllm_metal",
        "evaluation_id": None,
        "calibrated_bytes_per_token": None,
        "maximum_observed_context": None,
        "sample_count": None,
        "safety_margin_ratio": None,
    }
    directory = calibration_report_directory(
        model.model_id,
        hardware_fingerprint,
        "vllm_metal",
        application_support=root,
    )
    if not directory.exists():
        return model, provenance
    try:
        calibration, _ = discover_latest_kv_calibration(
            expected_model_id=model.model_id,
            expected_hardware_fingerprint=hardware_fingerprint,
            expected_backend="vllm_metal",
            application_support=root,
        )
    except ValueError:
        provenance["status"] = "invalid"
        return model, provenance
    calibrated = replace(model, memory_spec=calibration.apply(model.memory_spec))
    provenance.update(
        {
            "status": "applied",
            "evaluation_id": calibration.evaluation_id,
            "calibrated_bytes_per_token": calibration.calibrated_bytes_per_token,
            "maximum_observed_context": calibration.maximum_observed_context,
            "sample_count": calibration.sample_count,
            "safety_margin_ratio": calibration.safety_margin_ratio,
        }
    )
    return calibrated, provenance


def install_startup_metal_tuning(
    service: RuntimeService,
    model: InspectedModel,
    chip: AppleChipProfile,
    versions: RuntimeEnvironmentVersions,
    *,
    report_path: Path | None = None,
    tuning_root: Path | None = None,
) -> MetalTuningReport | None:
    environment_fingerprint = build_environment_fingerprint(
        platform=chip.platform,
        os_version=chip.os_version,
        toolchain_version=versions.toolchain_version,
        mlx_version=versions.mlx_version,
        backend_version=versions.backend_version,
    )
    shape_profile = build_model_kernel_shape_profile(
        model, context_tiers=(128, 1024), block_tokens=16
    )
    if report_path is not None:
        report = load_metal_tuning_report(
            report_path,
            profile_id=shape_profile.profile_id,
            hardware_fingerprint=chip.hardware_fingerprint,
            environment_fingerprint=environment_fingerprint,
        )
    else:
        report = discover_metal_tuning_report(
            profile_id=shape_profile.profile_id,
            hardware_fingerprint=chip.hardware_fingerprint,
            environment_fingerprint=environment_fingerprint,
            root=tuning_root,
        )
    if report is None:
        service.events.publish("runtime.metal_tuning.startup", {"status": "not_found"})
        return None
    applied = service.install_metal_tuning(report)
    service.events.publish(
        "runtime.metal_tuning.startup",
        {
            "status": "applied" if applied else "deferred",
            "tuning_id": report.tuning_id,
            "source": "explicit" if report_path is not None else "automatic",
        },
    )
    return report


def start_observed_native_v2_tuning(
    service: RuntimeService,
    backend: BackendProcess,
    *,
    source_root: Path | None,
    helper: Path | None,
    samples: int = 3,
) -> bool:
    """Tune saved production shapes and recycle EngineCore under one idle lease."""
    candidate = shutil.which("vllm-apple-v2-measure") if helper is None else None
    resolved_helper = helper or (Path(candidate) if candidate else None)
    if source_root is None or resolved_helper is None:
        service.events.publish(
            "runtime.native_v2_tuning",
            {"status": "disabled", "reason": "source_or_helper_unavailable"},
        )
        return False
    try:
        inspection = inspect_vllm_metal_integration(source_root)
        if not inspection.native_v2_detected:
            raise ValueError("native v2 topology unavailable")
        hardware_fingerprint = build_v2_hardware_fingerprint(service.profile.hardware)
        quarantine_count, latest_quarantined = inspect_v2_tuning_quarantine(
            hardware_fingerprint=hardware_fingerprint,
            source_fingerprint=inspection.source_fingerprint,
        )
        service.native_v2_tuning.update_quarantine(
            quarantine_count, latest_quarantined
        )
        observation = default_v2_observation_path(
            hardware_fingerprint, inspection.source_fingerprint
        )
        if not observation.exists():
            service.events.publish(
                "runtime.native_v2_tuning",
                {"status": "not_found", "reason": "no_observations"},
            )
            return False
        adapter = VLLMMetalV2MeasurementAdapter(resolved_helper)
        if not adapter.capability()["compatible"]:
            raise ValueError("native v2 measurement capability unavailable")
    except (OSError, ValueError):
        service.events.publish(
            "runtime.native_v2_tuning",
            {"status": "rejected", "reason": "validation_failed"},
        )
        return False

    def tune():
        shapes = load_v2_observations(
            observation,
            hardware_fingerprint=hardware_fingerprint,
            source_fingerprint=inspection.source_fingerprint,
        )
        return tune_v2_observed_shapes(
            shapes,
            adapter.measure,
            hardware_fingerprint=hardware_fingerprint,
            source_fingerprint=inspection.source_fingerprint,
            samples=samples,
        )

    def apply(profile) -> None:
        destination = save_v2_tuning_profile(profile)
        _apply_native_v2_profile(
            service,
            backend,
            profile,
            destination,
            hardware_fingerprint=hardware_fingerprint,
            source_fingerprint=inspection.source_fingerprint,
        )

    return service.start_native_v2_idle_tuning(tune, apply)


def _apply_native_v2_profile(
    service: RuntimeService,
    backend: BackendProcess,
    profile: VLLMMetalV2TuningProfile,
    destination: Path,
    *,
    hardware_fingerprint: str,
    source_fingerprint: str,
) -> None:
    """Recycle through readiness, quarantining and rolling back on failure."""
    service.set_state(RuntimeState.LOADING_MODEL)
    try:
        backend.restart()
    except Exception as apply_error:
        try:
            quarantine_v2_tuning_profile(destination)
        except (OSError, ValueError) as quarantine_error:
            service.set_failure(quarantine_error)
            raise RuntimeError("native v2 profile quarantine failed") from apply_error
        service.events.publish(
            "runtime.native_v2_tuning",
            {"status": "quarantined", "profile_id": profile.profile_id},
        )
        quarantine_count, latest_quarantined = inspect_v2_tuning_quarantine(
            hardware_fingerprint=hardware_fingerprint,
            source_fingerprint=source_fingerprint,
        )
        service.native_v2_tuning.update_quarantine(
            quarantine_count, latest_quarantined
        )
        try:
            backend.restart()
        except Exception as rollback_error:
            service.set_failure(rollback_error)
            raise RuntimeError("native v2 profile rollback failed") from apply_error
        service.set_state(RuntimeState.READY)
        service.events.publish(
            "runtime.native_v2_tuning",
            {"status": "rolled_back", "profile_id": profile.profile_id},
        )
        raise RuntimeError("native v2 profile was rolled back") from apply_error
    service.set_state(RuntimeState.READY)


def configure_native_v2_restore(
    service: RuntimeService,
    backend: BackendProcess,
    *,
    source_root: Path | None,
    helper: Path | None,
    samples: int = 3,
) -> bool:
    """Register an explicit, remeasurement-gated restore transaction."""
    candidate = shutil.which("vllm-apple-v2-measure") if helper is None else None
    resolved_helper = helper or (Path(candidate) if candidate else None)
    if source_root is None or resolved_helper is None:
        return False
    try:
        inspection = inspect_vllm_metal_integration(source_root)
        if not inspection.native_v2_detected:
            raise ValueError("native v2 topology unavailable")
        hardware_fingerprint = build_v2_hardware_fingerprint(service.profile.hardware)
        adapter = VLLMMetalV2MeasurementAdapter(resolved_helper)
        if not adapter.capability()["compatible"]:
            raise ValueError("native v2 measurement capability unavailable")
    except (OSError, ValueError, V2MeasurementAdapterError):
        return False

    def start(profile_id: str) -> bool:
        destination: list[Path] = []

        def restore():
            profile, path = restore_quarantined_v2_profile(
                profile_id,
                adapter.measure,
                hardware_fingerprint=hardware_fingerprint,
                source_fingerprint=inspection.source_fingerprint,
                samples=samples,
            )
            destination.append(path)
            return profile

        def apply(profile) -> None:
            if len(destination) != 1:
                raise RuntimeError("native v2 restored profile path unavailable")
            _apply_native_v2_profile(
                service,
                backend,
                profile,
                destination[0],
                hardware_fingerprint=hardware_fingerprint,
                source_fingerprint=inspection.source_fingerprint,
            )
            quarantine_count, latest_quarantined = inspect_v2_tuning_quarantine(
                hardware_fingerprint=hardware_fingerprint,
                source_fingerprint=inspection.source_fingerprint,
            )
            service.native_v2_tuning.update_quarantine(
                quarantine_count, latest_quarantined
            )

        return service.start_native_v2_idle_tuning(restore, apply)

    service.configure_native_v2_restore(start)
    return True


def build_native_v2_observation_monitor(
    service: RuntimeService,
    backend: BackendProcess,
    *,
    source_root: Path | None,
    helper: Path | None,
    prime_existing: bool,
) -> NativeV2ObservationMonitor | None:
    candidate = shutil.which("vllm-apple-v2-measure") if helper is None else None
    resolved_helper = helper or (Path(candidate) if candidate else None)
    if source_root is None or resolved_helper is None:
        return None
    try:
        inspection = inspect_vllm_metal_integration(source_root)
        hardware_fingerprint = build_v2_hardware_fingerprint(service.profile.hardware)
        observation = default_v2_observation_path(
            hardware_fingerprint, inspection.source_fingerprint
        )
    except (OSError, ValueError):
        return None
    return NativeV2ObservationMonitor(
        observation,
        lambda: start_observed_native_v2_tuning(
            service,
            backend,
            source_root=source_root,
            helper=resolved_helper,
        ),
        prime_existing=prime_existing,
    )


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    max_concurrent_requests: int = 32,
    model: str | None = None,
    backend_executable: str | None = None,
    backend_port: int = 8001,
    backend_startup_timeout: float = 600.0,
    max_model_len: int | None = None,
    model_integrity_manifest: Path | None = None,
    require_compatible_backend: bool = True,
    socket_path: str | None = None,
    session_token: str | None = None,
    session_token_file: str | None = None,
    shutdown_grace_period: float = 30.0,
    enable_runtime_probes: bool = True,
    enable_metal_tuning: bool = True,
    metal_tuning_report: Path | None = None,
    enable_kv_calibration: bool = True,
    kv_calibration_root: Path | None = None,
    enable_native_v2_idle_tuning: bool = True,
    vllm_metal_source_root: Path | None = None,
    vllm_metal_v2_helper: Path | None = None,
    native_v2_preference_path: Path | None = None,
) -> None:
    if shutdown_grace_period < 0:
        raise ValueError("shutdown grace period cannot be negative")
    if model is not None and port == backend_port and host in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("control and inference backend ports must differ")
    if session_token is not None and session_token_file is not None:
        raise ValueError("session_token and session_token_file are mutually exclusive")
    if not enable_metal_tuning and metal_tuning_report is not None:
        raise ValueError("metal_tuning_report and disabled Metal tuning are mutually exclusive")
    if session_token_file is not None:
        session_token = load_or_create_token_file(Path(session_token_file))

    backend: BackendProcess | None = None
    launch_thread: threading.Thread | None = None
    memory_monitor: MemoryMetricsMonitor | None = None
    if model is not None:
        if model_integrity_manifest is not None:
            verify_model_integrity(Path(model), model_integrity_manifest)
        hardware = detect_hardware()
        recommendation = None
        calibration_provenance = {
            "enabled": enable_kv_calibration,
            "status": "not_found" if enable_kv_calibration else "disabled",
            "backend": "vllm_metal",
            "evaluation_id": None,
            "calibrated_bytes_per_token": None,
            "maximum_observed_context": None,
            "sample_count": None,
            "safety_margin_ratio": None,
        }
        inspected: InspectedModel | None = None
        try:
            inspected = inspect_model(model, backend="vllm_metal")
            if max_model_len is None:
                if enable_kv_calibration:
                    chip_identity = detect_apple_chip_profile(hardware)
                    inspected, calibration_provenance = apply_startup_kv_calibration(
                        inspected,
                        chip_identity.hardware_fingerprint,
                        root=kv_calibration_root,
                    )
                recommendation = recommend_context(hardware.memory, inspected.memory_spec)
                balanced = next(tier for tier in recommendation.tiers if tier.name == "balanced")
                if balanced.max_tokens <= 0:
                    raise RuntimeError("model does not fit in the current safe memory budget")
                max_model_len = balanced.max_tokens
        except ModelCapabilityError:
            raise
        except ModelInspectionError:
            if max_model_len is None:
                max_model_len = DEFAULT_UNINSPECTED_CONTEXT
        resolved_config = make_backend_config(
            model=model,
            executable=backend_executable,
            port=backend_port,
            max_model_len=max_model_len,
            startup_timeout=backend_startup_timeout,
        )
        backend_tuning_enabled = enable_metal_tuning and supports_kernel_tuning_middleware(
            resolved_config.executable
        )
        config = make_backend_config(
            model=model,
            executable=str(resolved_config.executable),
            port=backend_port,
            max_model_len=max_model_len,
            startup_timeout=backend_startup_timeout,
            enable_kernel_tuning_middleware=backend_tuning_enabled,
        )
        compatibility = inspect_backend(config.executable)
        if require_compatible_backend and not compatibility.compatible:
            issues = ", ".join(compatibility.issues)
            raise RuntimeError(f"incompatible vLLM-Metal environment: {issues}")
        backend = BackendProcess(config)
        profile = build_profile(hardware, recommendation)
        proxy_engine = OpenAIProxyEngine(backend.base_url, backend)
        service = RuntimeService(
            proxy_engine,
            profile=profile,
            model_memory_spec=inspected.memory_spec if inspected is not None else None,
            state_memory_spec=inspected.state_memory_spec if inspected is not None else None,
            configured_context_tokens=max_model_len,
            kv_calibration=calibration_provenance,
        )
        if inspected is not None:
            service.record_memory_budget_component(
                "weights",
                inspected.memory_spec.weights_bytes,
                source="model_weight_files",
            )
        kv_capacity_resolver = None
        if inspected is not None and compatibility.vllm_version is not None:
            try:
                kv_capacity_resolver = KVCacheCapacityResolver(
                    compatibility.vllm_version,
                    inspected.memory_spec.kv_bytes_per_token,
                    str(inspected.config.get("model_type") or ""),
                )
            except ValueError:
                pass
        memory_monitor = MemoryMetricsMonitor(
            VLLMMemoryMetricsAdapter(
                backend.base_url,
                kv_capacity_resolver=kv_capacity_resolver,
            ),
            service,
            iogpu_adapter=IOGPUMemoryAdapter(),
        )
        service.set_state(RuntimeState.LOADING_MODEL)
        chip = None
        versions = None
        if (
            backend_tuning_enabled
            or enable_native_v2_idle_tuning
            or (enable_runtime_probes and require_compatible_backend)
        ):
            chip = detect_apple_chip_profile(hardware, compatibility)
            versions = discover_runtime_versions(
                compatibility.vllm_metal_version or compatibility.vllm_version
            )
        if backend_tuning_enabled and inspected is not None:
            assert chip is not None and versions is not None
            try:
                install_startup_metal_tuning(
                    service,
                    inspected,
                    chip,
                    versions,
                    report_path=metal_tuning_report,
                )
            except (OSError, ValueError, ModelInspectionError):
                service.events.publish(
                    "runtime.metal_tuning.startup",
                    {"status": "rejected", "reason": "validation_failed"},
                )
        elif backend_tuning_enabled:
            service.events.publish(
                "runtime.metal_tuning.startup",
                {"status": "not_found", "reason": "model_inspection_unavailable"},
            )
        elif enable_metal_tuning:
            service.events.publish(
                "runtime.metal_tuning.startup",
                {"status": "disabled", "reason": "backend_middleware_unsupported"},
            )
        if enable_runtime_probes and require_compatible_backend:
            assert chip is not None and versions is not None
            try:
                probe_report = RuntimeProbeCoordinator(
                    chip,
                    toolchain_version=versions.toolchain_version,
                    mlx_version=versions.mlx_version,
                    backend_version=versions.backend_version,
                    cache_root=default_application_support() / "profiles" / "kernel",
                ).probe_and_install(service, samples=1)
            except Exception:
                service.events.publish(
                    "runtime.kernel_probe", {"status": "failed", "reason": "coordinator_error"}
                )
            else:
                service.events.publish(
                    "runtime.kernel_probe",
                    {
                        "status": "completed",
                        "passed": sum(result.passed for result in probe_report.results),
                        "quarantined": sum(result.quarantined for result in probe_report.results),
                        "cache_status": probe_report.cache_status,
                    },
                )
    else:
        service = RuntimeService()
    enable_native_v2_idle_tuning = service.configure_native_v2_preference(
        native_v2_preference_path or default_native_v2_preference_path(),
        override_enabled=False if not enable_native_v2_idle_tuning else None,
    )
    server = create_server(
        host,
        port,
        service,
        max_concurrent_requests=max_concurrent_requests,
        session_token=session_token,
    )
    unix_server = None
    unix_thread = None
    if socket_path is not None:
        socket_parent = Path(socket_path).expanduser().parent
        socket_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        unix_server = create_unix_server(
            str(Path(socket_path).expanduser()),
            service,
            max_concurrent_requests=max_concurrent_requests,
            session_token=session_token,
        )
        unix_thread = threading.Thread(
            target=unix_server.serve_forever,
            kwargs={"poll_interval": 0.25},
            daemon=True,
            name="vllm-apple-uds-server",
        )
        unix_thread.start()

    # Optional OS telemetry must never delay control-plane readiness. Some
    # virtualized macOS runners expose libdispatch's symbol but stall while
    # registering the source, so registration is isolated from both servers.
    pressure_monitor_lock = threading.Lock()
    pressure_monitor_holder: list[MemoryPressureMonitor] = []
    pressure_monitor_shutdown = threading.Event()

    def launch_pressure_monitor() -> None:
        try:
            monitor = MemoryPressureMonitor(service.apply_memory_pressure)
            if pressure_monitor_shutdown.is_set():
                return
            monitor.start()
        except (OSError, RuntimeError, ValueError):
            service.events.publish(
                "memory.pressure_monitor", {"status": "unavailable", "fallback": "vm_stat"}
            )
            return
        with pressure_monitor_lock:
            if pressure_monitor_shutdown.is_set():
                monitor.stop()
                return
            pressure_monitor_holder.append(monitor)
        service.events.publish("memory.pressure_monitor", {"status": "active"})

    pressure_monitor_thread = threading.Thread(
        target=launch_pressure_monitor,
        daemon=True,
        name="vllm-apple-pressure-monitor",
    )
    pressure_monitor_thread.start()

    native_v2_monitor_lock = threading.Lock()
    native_v2_monitor_holder: list[NativeV2ObservationMonitor] = []

    if backend is not None:

        def launch_backend() -> None:
            try:
                backend.start()
            except Exception as error:
                failure = classify_runtime_failure(error)
                service.set_failure(failure)
                try:
                    diagnostic = persist_crash_diagnostic(
                        failure,
                        backend.recent_logs(),
                    )
                except (OSError, ValueError):
                    service.events.publish(
                        "runtime.crash_diagnostic",
                        {"status": "failed", "code": failure.code.value},
                    )
                else:
                    service.events.publish(
                        "runtime.crash_diagnostic",
                        {
                            "status": "persisted",
                            "code": failure.code.value,
                            "diagnostic_id": diagnostic.stem,
                        },
                    )
            else:
                if service.state == RuntimeState.LOADING_MODEL:
                    service.set_state(RuntimeState.READY)
                    assert memory_monitor is not None
                    memory_monitor.start()
                    configure_native_v2_restore(
                        service,
                        backend,
                        source_root=vllm_metal_source_root,
                        helper=vllm_metal_v2_helper,
                    )
                    started = False
                    if enable_native_v2_idle_tuning:
                        started = start_observed_native_v2_tuning(
                            service,
                            backend,
                            source_root=vllm_metal_source_root,
                            helper=vllm_metal_v2_helper,
                        )
                    monitor = build_native_v2_observation_monitor(
                        service,
                        backend,
                        source_root=vllm_metal_source_root,
                        helper=vllm_metal_v2_helper,
                        prime_existing=started,
                    )
                    if monitor is not None:
                        with native_v2_monitor_lock:
                            native_v2_monitor_holder.append(monitor)
                            monitor.start()

        launch_thread = threading.Thread(
            target=launch_backend,
            daemon=True,
            name="vllm-apple-backend-launch",
        )
        launch_thread.start()

    def stop(_signum: int, _frame: object) -> None:
        service.set_state(RuntimeState.STOPPING)
        server.begin_draining()
        if unix_server is not None:
            unix_server.begin_draining()
        threading.Thread(target=server.shutdown, daemon=True).start()
        if unix_server is not None:
            threading.Thread(target=unix_server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        pressure_monitor_shutdown.set()
        with pressure_monitor_lock:
            for pressure_monitor in pressure_monitor_holder:
                pressure_monitor.stop()
        with native_v2_monitor_lock:
            for native_v2_monitor in native_v2_monitor_holder:
                native_v2_monitor.stop()
        server.server_close()
        drain_deadline = time.monotonic() + shutdown_grace_period
        server.wait_for_drain(max(0.0, drain_deadline - time.monotonic()))
        if unix_server is not None:
            unix_server.shutdown()
            unix_server.server_close()
            unix_server.wait_for_drain(max(0.0, drain_deadline - time.monotonic()))
        if unix_thread is not None:
            unix_thread.join(timeout=2.0)
        if backend is not None:
            if memory_monitor is not None:
                memory_monitor.stop()
            backend.stop()
        if launch_thread is not None:
            launch_thread.join(timeout=2.0)
        service.set_state(RuntimeState.STOPPED)


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    serve(
        host=arguments.host,
        port=arguments.port,
        max_concurrent_requests=arguments.max_concurrent_requests,
        model=arguments.model,
        backend_executable=arguments.backend_executable,
        backend_port=arguments.backend_port,
        backend_startup_timeout=arguments.backend_startup_timeout,
        max_model_len=arguments.max_model_len,
        model_integrity_manifest=arguments.model_integrity_manifest,
        require_compatible_backend=not arguments.skip_backend_check,
        socket_path=arguments.socket_path,
        session_token=arguments.session_token,
        session_token_file=arguments.session_token_file,
        shutdown_grace_period=arguments.shutdown_grace_period,
        enable_runtime_probes=not arguments.skip_runtime_probes,
        enable_metal_tuning=not arguments.disable_metal_tuning,
        metal_tuning_report=arguments.metal_tuning_report,
        enable_kv_calibration=not arguments.disable_kv_calibration,
        kv_calibration_root=arguments.kv_calibration_root,
        enable_native_v2_idle_tuning=not arguments.disable_native_v2_idle_tuning,
        vllm_metal_source_root=arguments.vllm_metal_source_root,
        vllm_metal_v2_helper=arguments.vllm_metal_v2_helper,
        native_v2_preference_path=arguments.native_v2_preference_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
