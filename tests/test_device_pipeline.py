import threading
import time
import unittest

from vllm_apple.device_pipeline import DevicePipelineExecutor, DevicePipelineStage
from vllm_apple.device_resources import (
    BandwidthContentionEvidence,
    DeviceResourceCapacityError,
    UnifiedDeviceResourceLedger,
)
from vllm_apple.execution import ExecutionBackend


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


if __name__ == "__main__":
    unittest.main()
