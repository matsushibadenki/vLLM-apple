from __future__ import annotations

import argparse
import signal
import threading

from .api import create_server
from .backend import BackendProcess, OpenAIProxyEngine, make_backend_config
from .compat import inspect_backend
from .context import recommend_context
from .hardware import detect_hardware
from .model import DEFAULT_UNINSPECTED_CONTEXT, ModelInspectionError, inspect_model
from .profile import build_profile
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
    return parser


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
) -> None:
    if model is not None and port == backend_port and host in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("control and inference backend ports must differ")

    backend: BackendProcess | None = None
    launch_thread: threading.Thread | None = None
    if model is not None:
        hardware = detect_hardware()
        recommendation = None
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
        config = make_backend_config(
            model=model,
            executable=backend_executable,
            port=backend_port,
            max_model_len=max_model_len,
            startup_timeout=backend_startup_timeout,
        )
        if require_compatible_backend:
            compatibility = inspect_backend(config.executable)
            if not compatibility.compatible:
                issues = ", ".join(compatibility.issues)
                raise RuntimeError(f"incompatible vLLM-Metal environment: {issues}")
        backend = BackendProcess(config)
        profile = build_profile(hardware, recommendation)
        service = RuntimeService(OpenAIProxyEngine(backend.base_url, backend), profile=profile)
        service.set_state(RuntimeState.LOADING_MODEL)
    else:
        service = RuntimeService()
    server = create_server(host, port, service, max_concurrent_requests=max_concurrent_requests)

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
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
