import threading
import time
import unittest

from vllm_apple.device_capability import (
    ComputeDevice,
    DeviceCapability,
    DeviceCapabilityRegistry,
)
from vllm_apple.device_pipeline import (
    ANEAuxiliaryWorkload,
    AsyncEncoderLLMPipeline,
    DevicePipelineExecutor,
    DevicePipelineStage,
    require_ane_auxiliary_route,
)
from vllm_apple.device_resources import (
    BandwidthContentionEvidence,
    DeviceResourceCapacityError,
    UnifiedDeviceResourceLedger,
)
from vllm_apple.execution import ExecutionBackend, WorkloadPhase


class DevicePipelineTests(unittest.TestCase):
    def setUp(self):
        self.ledger = UnifiedDeviceResourceLedger(
            unified_memory_bytes=100,
            cpu_threads=2,
            gpu_command_queues=1,
            ane_tasks=1,
            bandwidth_slots=3,
            contention_profile_id="profile",
        )
        self.executor = DevicePipelineExecutor(self.ledger)

    def qualify(self, first, second):
        self.assertTrue(self.ledger.install_contention_evidence(
            BandwidthContentionEvidence(
                "profile", first, second, 100, 90, 3, True,
            )
        ))

    def test_unqualified_pipeline_fails_without_running_or_reserving(self):
        called = []
        stages = (
            DevicePipelineStage("cpu", ExecutionBackend.CPU, 10, lambda: called.append("cpu")),
            DevicePipelineStage(
                "gpu", ExecutionBackend.NATIVE_MLX, 10, lambda: called.append("gpu")
            ),
        )
        with self.assertRaisesRegex(
            DeviceResourceCapacityError, "bandwidth_contention_unqualified"
        ):
            self.executor.execute(stages)
        self.assertEqual(called, [])
        self.assertEqual(self.ledger.snapshot()["active_reservations"], 0)

    def test_qualified_stages_run_concurrently_and_release_atomically(self):
        self.qualify(ExecutionBackend.CPU, ExecutionBackend.NATIVE_MLX)
        barrier = threading.Barrier(2, timeout=1)

        def operation(value):
            barrier.wait()
            time.sleep(0.01)
            return value

        result = self.executor.execute((
            DevicePipelineStage(
                "cpu", ExecutionBackend.CPU, 10, lambda: operation("cpu")
            ),
            DevicePipelineStage(
                "gpu", ExecutionBackend.NATIVE_MLX, 10, lambda: operation("gpu")
            ),
        ))
        self.assertEqual(result.outputs, ("cpu", "gpu"))
        self.assertEqual(
            result.backends, (ExecutionBackend.CPU, ExecutionBackend.NATIVE_MLX)
        )
        self.assertGreater(result.elapsed_nanoseconds, 0)
        self.assertEqual(self.ledger.snapshot()["active_reservations"], 0)

    def test_three_device_pipeline_requires_every_pair(self):
        self.qualify(ExecutionBackend.CPU, ExecutionBackend.NATIVE_MLX)
        self.qualify(ExecutionBackend.CPU, ExecutionBackend.COREML_DRAFT)
        stages = (
            DevicePipelineStage("cpu", ExecutionBackend.CPU, 10, lambda: 1),
            DevicePipelineStage("gpu", ExecutionBackend.NATIVE_MLX, 10, lambda: 2),
            DevicePipelineStage("ane", ExecutionBackend.COREML_DRAFT, 10, lambda: 3),
        )
        with self.assertRaises(DeviceResourceCapacityError):
            self.executor.execute(stages)
        self.qualify(ExecutionBackend.NATIVE_MLX, ExecutionBackend.COREML_DRAFT)
        self.assertEqual(self.executor.execute(stages).outputs, (1, 2, 3))

    def test_failure_releases_entire_pipeline_group(self):
        self.qualify(ExecutionBackend.CPU, ExecutionBackend.NATIVE_MLX)

        def fail():
            raise RuntimeError("stage failed")

        with self.assertRaisesRegex(RuntimeError, "stage failed"):
            self.executor.execute((
                DevicePipelineStage("cpu", ExecutionBackend.CPU, 10, fail),
                DevicePipelineStage("gpu", ExecutionBackend.NATIVE_MLX, 10, lambda: 2),
            ))
        self.assertEqual(self.ledger.snapshot()["active_reservations"], 0)

    def ane_route(self, workload=ANEAuxiliaryWorkload.VISION_ENCODER):
        operator = f"{workload.value}@fixture"
        registry = DeviceCapabilityRegistry("m4-test", "environment-test")
        registry.record(DeviceCapability(
            ExecutionBackend.COREML_DRAFT,
            ComputeDevice.ANE,
            "coreml-test",
            "m4-test",
            "environment-test",
            (operator,),
            (WorkloadPhase.AUXILIARY,),
            ("fp16",),
            "available",
            "probe_passed",
            ("a" * 24,),
        ))
        route = require_ane_auxiliary_route(
            registry, workload=workload, operator=operator, precision="fp16"
        )
        return registry, route

    def test_all_auxiliary_workloads_require_exact_probe_passing_ane_evidence(self):
        for workload in ANEAuxiliaryWorkload:
            with self.subTest(workload=workload):
                _, route = self.ane_route(workload)
                self.assertEqual(route.workload, workload)
                self.assertEqual(route.operator, f"{workload.value}@fixture")
                self.assertEqual(len(route.capability_id), 24)

        empty = DeviceCapabilityRegistry("m4-test", "environment-test")
        with self.assertRaisesRegex(RuntimeError, "fallback chain is exhausted"):
            require_ane_auxiliary_route(
                empty,
                workload=ANEAuxiliaryWorkload.AUDIO_ENCODER,
                operator="audio_encoder@unprobed",
                precision="fp16",
            )
        with self.assertRaisesRegex(ValueError, "does not match workload"):
            require_ane_auxiliary_route(
                empty,
                workload=ANEAuxiliaryWorkload.VISION_ENCODER,
                operator="audio_encoder@fixture",
                precision="fp16",
            )

    def test_async_ane_encoder_output_is_handed_to_gpu_without_unproven_overlap(self):
        registry, route = self.ane_route()
        order = []

        def encode():
            order.append(("encode", self.ledger.snapshot()["used"].copy()))
            return (1.0, 2.0)

        def consume(value):
            order.append(("consume", self.ledger.snapshot()["used"].copy()))
            return tuple(item * 2 for item in value)

        with AsyncEncoderLLMPipeline(self.ledger, registry) as pipeline:
            future = pipeline.submit(
                route,
                encoder_memory_bytes=20,
                llm_backend=ExecutionBackend.NATIVE_MLX,
                llm_memory_bytes=30,
                encode=encode,
                consume=consume,
            )
            result = future.result(timeout=2)
        self.assertEqual(result.output, (2.0, 4.0))
        self.assertEqual(result.encoder_backend, ExecutionBackend.COREML_DRAFT)
        self.assertEqual(result.llm_backend, ExecutionBackend.NATIVE_MLX)
        self.assertEqual(result.capability_id, route.capability_id)
        self.assertEqual([item[0] for item in order], ["encode", "consume"])
        self.assertEqual(order[0][1]["ane_tasks"], 1)
        self.assertEqual(order[0][1]["gpu_command_queues"], 0)
        self.assertEqual(order[1][1]["ane_tasks"], 0)
        self.assertEqual(order[1][1]["gpu_command_queues"], 1)
        self.assertEqual(self.ledger.snapshot()["active_reservations"], 0)

    def test_async_pipeline_releases_ane_reservation_when_encoder_fails(self):
        def fail():
            raise RuntimeError("encoder failed")

        registry, route = self.ane_route(ANEAuxiliaryWorkload.CLASSIFIER)
        with AsyncEncoderLLMPipeline(self.ledger, registry) as pipeline:
            future = pipeline.submit(
                route,
                encoder_memory_bytes=20,
                llm_backend=ExecutionBackend.VLLM_METAL,
                llm_memory_bytes=30,
                encode=fail,
                consume=lambda value: value,
            )
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                future.result(timeout=2)
        self.assertEqual(self.ledger.snapshot()["active_reservations"], 0)

    def test_async_pipeline_rejects_cpu_as_llm_stage(self):
        registry, route = self.ane_route(ANEAuxiliaryWorkload.EMBEDDING)
        with AsyncEncoderLLMPipeline(self.ledger, registry) as pipeline:
            with self.assertRaisesRegex(ValueError, "invalid encoder LLM"):
                pipeline.submit(
                    route,
                    encoder_memory_bytes=1,
                    llm_backend=ExecutionBackend.CPU,
                    llm_memory_bytes=1,
                    encode=lambda: 1,
                    consume=lambda value: value,
                )

    def test_async_pipeline_rejects_stale_or_fabricated_route(self):
        registry, route = self.ane_route()
        stale = type(route)(
            route.workload, route.operator, route.precision, "f" * 24
        )
        with AsyncEncoderLLMPipeline(self.ledger, registry) as pipeline:
            with self.assertRaisesRegex(RuntimeError, "evidence is stale"):
                pipeline.submit(
                    stale,
                    encoder_memory_bytes=1,
                    llm_backend=ExecutionBackend.NATIVE_MLX,
                    llm_memory_bytes=1,
                    encode=lambda: 1,
                    consume=lambda value: value,
                )


if __name__ == "__main__":
    unittest.main()
