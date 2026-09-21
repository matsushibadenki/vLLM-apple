"""Production composition root for managed inference backend engines."""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .backend_engine import (
    BackendEngineDescriptor,
    BackendEngineFailure,
    BackendEngineRegistry,
    BackendEngineRequest,
)
from .execution import ExecutionBackend, WorkloadPhase
from .inference_request import (
    InferenceEngineBusy,
    InferenceRequestCancelled,
    InferenceRequestContext,
)


class ManagedInferenceBackendEngine:
    """Adapt a production chat engine to the versioned backend registry contract."""

    def __init__(
        self,
        descriptor: BackendEngineDescriptor,
        factory: Callable[[], object],
    ) -> None:
        if not callable(factory):
            raise ValueError("backend engine factory must be callable")
        self.descriptor = descriptor
        self._factory = factory
        self._engine: object | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        engine = self._engine
        return engine is not None and getattr(engine, "ready", False) is True

    def start(self) -> None:
        with self._lock:
            if self._engine is not None:
                if getattr(self._engine, "ready", False) is not True:
                    raise BackendEngineFailure("backend_not_ready", retryable=True)
                return
            try:
                engine = self._factory()
            except Exception as error:
                raise BackendEngineFailure(
                    "backend_start_failed", retryable=True
                ) from error
            if getattr(engine, "ready", False) is not True:
                close = getattr(engine, "close", None)
                if callable(close):
                    close()
                raise BackendEngineFailure("backend_not_ready", retryable=True)
            self._engine = engine

    def execute(
        self,
        request: BackendEngineRequest,
        context: InferenceRequestContext,
    ) -> dict[str, Any]:
        engine = self._engine
        if engine is None or getattr(engine, "ready", False) is not True:
            raise BackendEngineFailure("backend_not_ready", retryable=True)
        if not isinstance(request.payload, dict):
            raise BackendEngineFailure("invalid_backend_payload", retryable=False)
        execute = getattr(engine, "chat_completions_with_request_context", None)
        if not callable(execute):
            raise BackendEngineFailure("backend_context_api_missing", retryable=False)
        try:
            result = execute(request.payload, None, context)
        except InferenceEngineBusy:
            # Capacity is a request-level condition, not backend failure. Preserve
            # it so the HTTP boundary can return a bounded 503 response.
            raise
        except InferenceRequestCancelled:
            raise
        except BackendEngineFailure:
            raise
        except Exception as error:
            raise BackendEngineFailure("backend_execution_failed", retryable=True) from error
        if not isinstance(result, dict):
            raise BackendEngineFailure("invalid_backend_result", retryable=False)
        return result

    def stop(self) -> None:
        with self._lock:
            engine, self._engine = self._engine, None
        if engine is None:
            return
        close = getattr(engine, "close", None)
        if not callable(close) or close() is not True:
            raise RuntimeError("backend engine did not stop cleanly")

    def models(self) -> list[dict[str, Any]]:
        engine = self._engine
        models = getattr(engine, "models", None)
        if engine is None or not callable(models):
            return []
        result = models()
        if not isinstance(result, list) or any(not isinstance(item, dict) for item in result):
            raise RuntimeError("backend engine model list is invalid")
        return [dict(item) for item in result]

    def diagnostics(self) -> dict[str, Any]:
        engine = self._engine
        diagnostics = getattr(engine, "diagnostics", None)
        if engine is None or not callable(diagnostics):
            raise RuntimeError("backend engine diagnostics are unavailable")
        result = diagnostics()
        if not isinstance(result, dict):
            raise RuntimeError("backend engine diagnostics are invalid")
        return dict(result)


@dataclass(frozen=True, slots=True)
class BackendEngineRegistration:
    descriptor: BackendEngineDescriptor
    factory: Callable[[], object]


class ProductionBackendComposition:
    """Own startup, registry publication, and reverse-order shutdown atomically."""

    def __init__(self, registrations: tuple[BackendEngineRegistration, ...]) -> None:
        if not registrations:
            raise ValueError("production backend composition is empty")
        self._engines = tuple(
            ManagedInferenceBackendEngine(item.descriptor, item.factory)
            for item in registrations
        )
        self._registry: BackendEngineRegistry[dict[str, Any]] | None = None

    @property
    def registry(self) -> BackendEngineRegistry[dict[str, Any]]:
        if self._registry is None:
            raise RuntimeError("production backend composition is not started")
        return self._registry

    def start(self) -> BackendEngineRegistry[dict[str, Any]]:
        if self._registry is not None:
            return self._registry
        started: list[ManagedInferenceBackendEngine] = []
        try:
            for engine in self._engines:
                engine.start()
                started.append(engine)
            self._registry = BackendEngineRegistry(self._engines)
            return self._registry
        except BaseException:
            for engine in reversed(started):
                try:
                    engine.stop()
                except Exception:
                    pass
            raise

    def close(self) -> bool:
        if self._registry is None:
            return True
        registry, self._registry = self._registry, None
        registry.stop_all()
        return True

    def models(self) -> list[dict[str, Any]]:
        if self._registry is None:
            return []
        models: list[dict[str, Any]] = []
        seen: set[str] = set()
        for engine in self._engines:
            for model in engine.models():
                identity = model.get("id")
                if isinstance(identity, str) and identity not in seen:
                    seen.add(identity)
                    models.append(model)
        return models

    def diagnostics(self, backend: ExecutionBackend) -> dict[str, Any]:
        if self._registry is None:
            raise RuntimeError("production backend composition is not started")
        for engine in self._engines:
            if engine.descriptor.backend is backend:
                return engine.diagnostics()
        raise ValueError("backend engine is not registered")


class BackendRegistryInferenceEngine:
    """Expose a production backend registry through RuntimeService's chat ABI."""

    def __init__(
        self,
        composition: ProductionBackendComposition,
        *,
        model_architecture: str,
        precision: str,
        candidates: tuple[ExecutionBackend, ...],
    ) -> None:
        if (not model_architecture or not precision or not candidates
                or len(set(candidates)) != len(candidates)):
            raise ValueError("invalid backend registry inference route")
        self._composition = composition
        self._architecture = model_architecture
        self._precision = precision
        self._candidates = candidates
        self._registry = composition.start()

    @property
    def ready(self) -> bool:
        return self._registry is not None

    def models(self) -> list[dict[str, Any]]:
        return self._composition.models()

    def chat_completions_with_request_context(
        self,
        request: dict[str, Any],
        _kernel_context: object | None,
        request_context: InferenceRequestContext,
    ) -> dict[str, Any]:
        routed = BackendEngineRequest(
            "chat.completions",
            WorkloadPhase.PREFILL,
            self._precision,
            self._architecture,
            self._candidates,
            request,
        )
        return self._registry.execute(routed, request_context).value

    def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("registry requests require an inference request context")

    def open_chat_stream(self, request: dict[str, Any]):
        raise RuntimeError("registry streaming is not enabled")

    def close(self) -> bool:
        if self._registry is None:
            return True
        self._registry = None
        return self._composition.close()

    def diagnostics(self) -> dict[str, Any]:
        if len(self._candidates) != 1:
            raise RuntimeError("backend diagnostics require an explicit backend")
        result = dict(self._composition.diagnostics(self._candidates[0]))
        if "routing_telemetry" in result:
            raise RuntimeError("backend diagnostics reserve routing_telemetry")
        result["routing_telemetry"] = self._registry.telemetry_snapshot()
        return result
