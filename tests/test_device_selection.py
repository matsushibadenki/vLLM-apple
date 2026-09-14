import unittest

from vllm_apple.device_benchmark import (
    DeviceBenchmarkConfig,
    DeviceBenchmarkMeasurement,
    DeviceBenchmarkReport,
)
from vllm_apple.device_selection import select_measured_device_backend
from vllm_apple.execution import ExecutionBackend, WorkloadPhase


def report(backend, latency, *, digest="a" * 64, load=None, peak=100):
    config = DeviceBenchmarkConfig(
        "vector_add", backend, WorkloadPhase.AUXILIARY, "fp32", (256,), 1, 3
    )
    measurements = tuple(
        DeviceBenchmarkMeasurement(
            latency, latency, 0, 0, 256, peak, None, digest
        )
        for _ in range(3)
    )
    return DeviceBenchmarkReport(
        "m4-test", "environment-test", backend.value.encode().hex().ljust(24, "0")[:24],
        config, measurements, load,
    )


class DeviceSelectionTests(unittest.TestCase):
    def test_selects_accelerator_only_for_material_end_to_end_improvement(self):
        decision = select_measured_device_backend((
            report(ExecutionBackend.CPU, 100),
            report(ExecutionBackend.NATIVE_METAL, 60, load=10),
        ))
        self.assertEqual(decision.selected, ExecutionBackend.NATIVE_METAL)
        self.assertAlmostEqual(decision.improvement_ratio, 0.3)

    def test_cold_load_can_keep_cpu_as_winner(self):
        decision = select_measured_device_backend((
            report(ExecutionBackend.CPU, 100),
            report(ExecutionBackend.COREML_DRAFT, 40, load=100),
        ))
        self.assertEqual(decision.selected, ExecutionBackend.CPU)
        rejected = next(
            value for value in decision.candidates
            if value.backend is ExecutionBackend.COREML_DRAFT
        )
        self.assertIn("insufficient_latency_improvement", rejected.rejection_reasons)

    def test_rejects_output_or_memory_regression(self):
        decision = select_measured_device_backend((
            report(ExecutionBackend.CPU, 100),
            report(ExecutionBackend.NATIVE_MLX, 40, digest="b" * 64, peak=200),
        ))
        self.assertEqual(decision.selected, ExecutionBackend.CPU)
        rejected = decision.candidates[1]
        self.assertEqual(
            set(rejected.rejection_reasons),
            {"output_mismatch", "peak_memory_regression"},
        )

    def test_requires_directly_comparable_identity(self):
        other = report(ExecutionBackend.NATIVE_METAL, 40)
        mismatched = DeviceBenchmarkReport(
            other.hardware_fingerprint,
            other.environment_fingerprint,
            other.capability_id,
            DeviceBenchmarkConfig(
                "vector_add", ExecutionBackend.NATIVE_METAL,
                WorkloadPhase.AUXILIARY, "fp32", (512,), 1, 3,
            ),
            other.measurements,
            None,
        )
        with self.assertRaisesRegex(ValueError, "not directly comparable"):
            select_measured_device_backend((report(ExecutionBackend.CPU, 100), mismatched))


if __name__ == "__main__":
    unittest.main()
