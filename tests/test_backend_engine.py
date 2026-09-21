import unittest
import threading
import time
from dataclasses import replace
from unittest.mock import Mock

from vllm_apple.backend_engine import (
    BackendEngineDescriptor,
    BackendEngineFailure,
    BackendEngineRegistry,
    BackendEngineRequest,
)
from vllm_apple.execution import ExecutionBackend, WorkloadPhase
from vllm_apple.inference_request import InferenceRequestContext


class Engine:
    def __init__(self, backend, *, ready=True, result=None, failure=None):
        self.descriptor = BackendEngineDescriptor(
            backend, "test-1", ("qwen",), ("fp16",),
            (WorkloadPhase.PREFILL,), ("matmul",), "subprocess",
        )
        self.ready = ready
        self.result = result
        self.failure = failure
        self.stopped = False

    def start(self):
        self.ready = True

    def execute(self, request, context):
        if self.failure is not None:
            raise self.failure
        return self.result

    def stop(self):
        self.stopped = True


class BackendEngineTests(unittest.TestCase):
    def request(self, candidates):
        return BackendEngineRequest(
            "matmul", WorkloadPhase.PREFILL, "fp16", "qwen", candidates)

    def context(self):
        return InferenceRequestContext(
            "request-1", deadline=time.monotonic() + 60, cancellation=threading.Event())

    def test_explicit_retryable_fallback_chain(self):
        metal = Engine(
            ExecutionBackend.VLLM_METAL,
            failure=BackendEngineFailure("kernel_failed", retryable=True),
        )
        cpu = Engine(ExecutionBackend.CPU, result="ok")
        result = BackendEngineRegistry((metal, cpu)).execute(
            self.request((ExecutionBackend.VLLM_METAL, ExecutionBackend.CPU)),
            self.context(),
        )
        self.assertEqual(result.value, "ok")
        self.assertEqual(result.backend, ExecutionBackend.CPU)
        self.assertEqual([attempt.status for attempt in result.attempts],
                         ["failed", "succeeded"])

    def test_unready_and_unsupported_engines_are_rejected(self):
        mlx = Engine(ExecutionBackend.NATIVE_MLX, ready=False)
        cpu = Engine(ExecutionBackend.CPU, result=1)
        cpu.descriptor = replace(cpu.descriptor, precisions=("fp32",))
        registry = BackendEngineRegistry((mlx, cpu))
        with self.assertRaisesRegex(BackendEngineFailure, "fallback_exhausted"):
            registry.execute(
                self.request((ExecutionBackend.NATIVE_MLX, ExecutionBackend.CPU)),
                self.context(),
            )

    def test_duplicate_backend_and_invalid_metadata_fail_closed(self):
        first = Engine(ExecutionBackend.CPU)
        with self.assertRaises(ValueError):
            BackendEngineRegistry((first, Engine(ExecutionBackend.CPU)))
        with self.assertRaises(ValueError):
            replace(first.descriptor, isolation="shared_global")

    def test_nonretryable_failure_and_context_deadline_do_not_fallback(self):
        metal = Engine(
            ExecutionBackend.NATIVE_METAL,
            failure=BackendEngineFailure("corrupt_output", retryable=False),
        )
        cpu = Engine(ExecutionBackend.CPU, result="must-not-run")
        registry = BackendEngineRegistry((metal, cpu))
        with self.assertRaisesRegex(BackendEngineFailure, "corrupt_output"):
            registry.execute(
                self.request((ExecutionBackend.NATIVE_METAL, ExecutionBackend.CPU)),
                self.context(),
            )

    def test_stop_all_reverses_registration_order(self):
        calls = []
        first = Engine(ExecutionBackend.VLLM_METAL)
        second = Engine(ExecutionBackend.CPU)
        first.stop = Mock(side_effect=lambda: calls.append("metal"))
        second.stop = Mock(side_effect=lambda: calls.append("cpu"))
        BackendEngineRegistry((first, second)).stop_all()
        self.assertEqual(calls, ["cpu", "metal"])


if __name__ == "__main__":
    unittest.main()
