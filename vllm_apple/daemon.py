from __future__ import annotations

import argparse
import signal
import threading
import time
from pathlib import Path

from .api import create_server, create_unix_server
from .auth import load_or_create_token_file
from .backend import (
    BackendProcess,
    OpenAIProxyEngine,
    make_backend_config,
    supports_kernel_tuning_middleware,
)
from .compat import inspect_backend
from .context import recommend_context
from .execution import AppleChipProfile
from .execution_profile import detect_apple_chip_profile
from .hardware import default_application_support, detect_hardware
from .kernel_probe import build_environment_fingerprint
from .kernel_profile import build_model_kernel_shape_profile
from .metal_tuning import (
    MetalTuningReport,
    discover_metal_tuning_report,
    load_metal_tuning_report,
)
from .model import (
    DEFAULT_UNINSPECTED_CONTEXT,
    InspectedModel,
    ModelInspectionError,
    inspect_model,
)
from .profile import build_profile
from .runtime_probe import (
    RuntimeEnvironmentVersions,
    RuntimeProbeCoordinator,
    discover_runtime_versions,
)
from .service import RuntimeService
from .types import RuntimeState


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
    parser.add_argument("--skip-backend-check", action="store_true")
    parser.add_argument("--socket-path")
    parser.add_argument("--session-token")
    parser.add_argument("--session-token-file")
    parser.add_argument("--shutdown-grace-period", type=float, default=30.0)
    parser.add_argument("--skip-runtime-probes", action="store_true")
    tuning = parser.add_mutually_exclusive_group()
    tuning.add_argument("--metal-tuning-report", type=Path)
    tuning.add_argument("--disable-metal-tuning", action="store_true")
    return parser


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


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    max_concurrent_requests: int = 32,
    model: str | None = None,
    backend_executable: str | None = None,
    backend_port: int = 8001,
    backend_startup_timeout: float = 600.0,
    max_model_len: int | None = None,
    require_compatible_backend: bool = True,
    socket_path: str | None = None,
    session_token: str | None = None,
    session_token_file: str | None = None,
    shutdown_grace_period: float = 30.0,
    enable_runtime_probes: bool = True,
    enable_metal_tuning: bool = True,
    metal_tuning_report: Path | None = None,
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
    if model is not None:
        hardware = detect_hardware()
        recommendation = None
        inspected: InspectedModel | None = None
        if max_model_len is None:
            try:
                inspected = inspect_model(model)
                recommendation = recommend_context(hardware.memory, inspected.memory_spec)
                balanced = next(tier for tier in recommendation.tiers if tier.name == "balanced")
                if balanced.max_tokens <= 0:
                    raise RuntimeError("model does not fit in the current safe memory budget")
                max_model_len = balanced.max_tokens
            except ModelInspectionError:
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
        service = RuntimeService(OpenAIProxyEngine(backend.base_url, backend), profile=profile)
        service.set_state(RuntimeState.LOADING_MODEL)
        chip = None
        versions = None
        if backend_tuning_enabled or (enable_runtime_probes and require_compatible_backend):
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

    if backend is not None:

        def launch_backend() -> None:
            try:
                backend.start()
            except Exception as error:
                service.set_failure(str(error))
            else:
                service.set_state(RuntimeState.READY)

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
        require_compatible_backend=not arguments.skip_backend_check,
        socket_path=arguments.socket_path,
        session_token=arguments.session_token,
        session_token_file=arguments.session_token_file,
        shutdown_grace_period=arguments.shutdown_grace_period,
        enable_runtime_probes=not arguments.skip_runtime_probes,
        enable_metal_tuning=not arguments.disable_metal_tuning,
        metal_tuning_report=arguments.metal_tuning_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
