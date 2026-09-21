import threading
import time
import unittest

from vllm_apple.backend_engine import BackendEngineDescriptor, BackendEngineRegistry, BackendEngineRequest
from vllm_apple.execution import ExecutionBackend, WorkloadPhase
from vllm_apple.inference_request import InferenceRequestContext
from vllm_apple.operator_graph_dispatch import OperatorGraphDispatcher, OperatorGraphNode


class Engine:
    descriptor = BackendEngineDescriptor(
        ExecutionBackend.CPU, "test", ("graph",), ("fp32",),
        (WorkloadPhase.AUXILIARY,), ("op",), "in_process",
    )
    ready = True

    def start(self): pass

    def stop(self): pass

    def execute(self, request, context):
        return {"value": request.payload["value"], "inputs": request.payload["_dependencies"]}


def request(value, phase=WorkloadPhase.AUXILIARY):
    return BackendEngineRequest(
        "op", phase, "fp32", "graph",
        (ExecutionBackend.CPU,), {"value": value},
    )


def context():
    return InferenceRequestContext(
        "graph-1", deadline=time.monotonic() + 10, cancellation=threading.Event()
    )


class OperatorGraphDispatchTests(unittest.TestCase):
    def test_topological_dispatch_injects_only_declared_dependencies(self):
        dispatcher = OperatorGraphDispatcher(BackendEngineRegistry((Engine(),)))
        result = dispatcher.execute((
            OperatorGraphNode("decode", request(3), ("encode",), 7),
            OperatorGraphNode("encode", request(2), (), 5),
        ), context(), maximum_synchronization_nanoseconds=12)
        self.assertEqual(tuple(result.values), ("encode", "decode"))
        self.assertEqual(result.values["decode"]["inputs"], {"encode": result.values["encode"]})
        self.assertEqual(result.synchronization_nanoseconds, 12)

    def test_cycle_missing_dependency_and_sync_overrun_fail_before_execution(self):
        dispatcher = OperatorGraphDispatcher(BackendEngineRegistry((Engine(),)))
        with self.assertRaisesRegex(ValueError, "cycle"):
            dispatcher.execute((
                OperatorGraphNode("a", request(1), ("b",)),
                OperatorGraphNode("b", request(2), ("a",)),
            ), context(), maximum_synchronization_nanoseconds=10)
        with self.assertRaisesRegex(ValueError, "missing"):
            dispatcher.execute(
                (OperatorGraphNode("a", request(1), ("unknown",)),),
                context(), maximum_synchronization_nanoseconds=10,
            )
        with self.assertRaisesRegex(ValueError, "budget"):
            dispatcher.execute(
                (OperatorGraphNode("a", request(1), (), 11),),
                context(), maximum_synchronization_nanoseconds=10,
            )

    def test_reserved_dependency_payload_is_rejected(self):
        dispatcher = OperatorGraphDispatcher(BackendEngineRegistry((Engine(),)))
        bad = request(1)
        bad = BackendEngineRequest(
            bad.operator, bad.phase, bad.precision, bad.model_architecture,
            bad.candidates, {"value": 1, "_dependencies": {}},
        )
        with self.assertRaisesRegex(ValueError, "reserves"):
            dispatcher.execute(
                (OperatorGraphNode("a", bad),), context(),
                maximum_synchronization_nanoseconds=0,
            )

    def test_reversed_phase_dependency_fails_before_execution(self):
        dispatcher = OperatorGraphDispatcher(BackendEngineRegistry((Engine(),)))
        with self.assertRaisesRegex(ValueError, "decode->prefill"):
            dispatcher.execute((
                OperatorGraphNode("decode", request(1, WorkloadPhase.DECODE)),
                OperatorGraphNode(
                    "prefill", request(2, WorkloadPhase.PREFILL), ("decode",)
                ),
            ), context(), maximum_synchronization_nanoseconds=0)

    def test_phase_vocabulary_covers_heterogeneous_pipeline(self):
        self.assertEqual(
            {phase.value for phase in WorkloadPhase},
            {
                "prefill", "decode", "sampling", "vision_encoder",
                "audio_encoder", "embedding", "classifier", "draft",
                "verify", "auxiliary",
            },
        )


if __name__ == "__main__":
    unittest.main()
