import threading
import time
import unittest

from vllm_apple.backend_composition import (
    BackendRegistryInferenceEngine,
    BackendEngineRegistration,
    ManagedInferenceBackendEngine,
    ProductionBackendComposition,
)
from vllm_apple.backend_engine import (
    BackendEngineDescriptor,
    BackendEngineFailure,
    BackendEngineRequest,
)
from vllm_apple.execution import ExecutionBackend, WorkloadPhase
from vllm_apple.inference_request import InferenceRequestContext


class Engine:
    def __init__(self, name, calls, *, ready=True, result=None):
        self.name = name
        self.ready = ready
        self.result = {"engine": name} if result is None else result
        self.calls = calls

    def chat_completions_with_request_context(self, request, kernel, context):
        self.calls.append((self.name, request, kernel, context.request_id))
        return self.result

    def close(self):
        self.calls.append(("close", self.name))
        self.ready = False
        return True

    def diagnostics(self):
        return {"engine": self.name}


def descriptor(backend):
    return BackendEngineDescriptor(
        backend, "production-1", ("qwen",), ("fp16",),
        (WorkloadPhase.PREFILL,), ("chat.completions",), "subprocess",
    )


def request(candidates, payload=None):
    return BackendEngineRequest(
        "chat.completions", WorkloadPhase.PREFILL, "fp16", "qwen",
        candidates, payload,
    )


def context():
    return InferenceRequestContext(
        "request-1", deadline=time.monotonic() + 10,
        cancellation=threading.Event(),
    )


class BackendCompositionTests(unittest.TestCase):
    def test_composition_starts_executes_and_stops_in_reverse_order(self):
        calls = []
        composition = ProductionBackendComposition((
            BackendEngineRegistration(
                descriptor(ExecutionBackend.NATIVE_MLX),
                lambda: Engine("mlx", calls),
            ),
            BackendEngineRegistration(
                descriptor(ExecutionBackend.CPU),
                lambda: Engine("cpu", calls),
            ),
        ))
        registry = composition.start()
        result = registry.execute(
            request((ExecutionBackend.NATIVE_MLX,), {"messages": []}), context()
        )
        self.assertEqual(result.value, {"engine": "mlx"})
        self.assertEqual(calls[0][2], None)
        self.assertTrue(composition.close())
        self.assertEqual(calls[-2:], [("close", "cpu"), ("close", "mlx")])

    def test_runtime_service_facade_routes_through_registry(self):
        calls = []
        composition = ProductionBackendComposition((
            BackendEngineRegistration(
                descriptor(ExecutionBackend.NATIVE_MLX),
                lambda: Engine("mlx", calls),
            ),
        ))
        facade = BackendRegistryInferenceEngine(
            composition,
            model_architecture="qwen",
            precision="fp16",
            candidates=(ExecutionBackend.NATIVE_MLX,),
        )
        result = facade.chat_completions_with_request_context(
            {"messages": []}, None, context()
        )
        self.assertEqual(result, {"engine": "mlx"})
        diagnostics = facade.diagnostics()
        self.assertEqual(diagnostics["engine"], "mlx")
        self.assertEqual(
            diagnostics["routing_telemetry"]["attempts"][0]["status"],
            "succeeded",
        )
        self.assertTrue(facade.close())
        self.assertTrue(facade.close())

    def test_start_failure_rolls_back_previously_started_engine(self):
        calls = []
        composition = ProductionBackendComposition((
            BackendEngineRegistration(
                descriptor(ExecutionBackend.NATIVE_MLX),
                lambda: Engine("mlx", calls),
            ),
            BackendEngineRegistration(
                descriptor(ExecutionBackend.CPU),
                lambda: Engine("cpu", calls, ready=False),
            ),
        ))
        with self.assertRaisesRegex(BackendEngineFailure, "backend_not_ready"):
            composition.start()
        self.assertIn(("close", "mlx"), calls)

    def test_invalid_payload_and_result_fail_closed(self):
        calls = []
        adapter = ManagedInferenceBackendEngine(
            descriptor(ExecutionBackend.CPU), lambda: Engine("cpu", calls, result=[])
        )
        adapter.start()
        with self.assertRaisesRegex(BackendEngineFailure, "invalid_backend_payload"):
            adapter.execute(request((ExecutionBackend.CPU,)), context())
        with self.assertRaisesRegex(BackendEngineFailure, "invalid_backend_result"):
            adapter.execute(request((ExecutionBackend.CPU,), {"messages": []}), context())
        adapter.stop()


if __name__ == "__main__":
    unittest.main()
