import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from vllm_apple.device_benchmark import (
    BoundedCPUReferenceBenchmarkAdapter,
    CoreMLFixedGraphBenchmarkAdapter,
    DeviceBenchmarkConfig,
    DeviceBenchmarkMeasurement,
    DeviceBenchmarkSuite,
    NativeCPUBenchmarkAdapter,
    NativeKernelBenchmarkAdapter,
    load_device_benchmark,
    representative_device_benchmark_configs,
    run_device_benchmark_suite,
    run_device_microbenchmark,
    save_device_benchmark,
)
from vllm_apple.device_capability import (
    ComputeDevice,
    DeviceCapability,
    DeviceCapabilityRegistry,
)
from vllm_apple.execution import ExecutionBackend, WorkloadPhase
from vllm_apple.coreml_backend import (
    CoreMLFixedGraphBackend,
    CoreMLFixedGraphResource,
    CoreMLFixedGraphResult,
)
from unittest.mock import Mock
from vllm_apple.kernel_probe import KernelMeasurement


class DeviceBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.registry = DeviceCapabilityRegistry("m4-test", "environment-test")
        self.capability = DeviceCapability(
            ExecutionBackend.CPU,
            ComputeDevice.CPU,
            "python-test",
            "m4-test",
            "environment-test",
            ("matmul_8x8",),
            (WorkloadPhase.PREFILL,),
            ("fp32",),
            "available",
            "probe_passed",
            ("a" * 24,),
        )
        self.registry.record(self.capability)
        self.config = DeviceBenchmarkConfig(
            "matmul_8x8",
            ExecutionBackend.CPU,
            WorkloadPhase.PREFILL,
            "fp32",
            (8, 8, 8),
            1,
            samples=3,
        )

    @staticmethod
    def measurement(index):
        return DeviceBenchmarkMeasurement(
            100 + index,
            70 + index,
            10,
            5,
            512,
            4096 + index,
            2.5,
            "b" * 64,
        )

    def test_builds_profile_bound_aggregate(self):
        report = run_device_microbenchmark(
            self.config,
            self.registry,
            self.measurement,
            cold_load=lambda: 20,
        )
        payload = report.to_dict()
        self.assertEqual(payload["median_total_nanoseconds"], 101)
        self.assertEqual(payload["median_execution_nanoseconds"], 71)
        self.assertEqual(payload["peak_memory_bytes"], 4098)
        self.assertEqual(payload["energy_microjoules"], 7.5)
        self.assertEqual(payload["cold_load_nanoseconds"], 20)
        self.assertEqual(report.capability_id, self.capability.capability_id)
        self.assertEqual(len(report.report_id), 24)

    def test_unknown_or_wrong_phase_capability_fails_closed(self):
        config = DeviceBenchmarkConfig(
            "matmul_8x8",
            ExecutionBackend.CPU,
            WorkloadPhase.DECODE,
            "fp32",
            (8, 8, 8),
            1,
        )
        with self.assertRaisesRegex(RuntimeError, "fallback chain is exhausted"):
            run_device_microbenchmark(config, self.registry, self.measurement)

    def test_rejects_inconsistent_output(self):
        def operation(index):
            value = self.measurement(index)
            if index == 2:
                return DeviceBenchmarkMeasurement(
                    value.total_nanoseconds,
                    value.execution_nanoseconds,
                    value.conversion_nanoseconds,
                    value.synchronization_nanoseconds,
                    value.work_items,
                    value.peak_memory_bytes,
                    value.energy_microjoules,
                    "c" * 64,
                )
            return value

        with self.assertRaisesRegex(ValueError, "output changed"):
            run_device_microbenchmark(self.config, self.registry, operation)

    def test_unknown_energy_or_memory_remains_unknown(self):
        def operation(_index):
            return DeviceBenchmarkMeasurement(
                100, 80, 0, 0, 1, None, None, "b" * 64
            )

        payload = run_device_microbenchmark(
            self.config, self.registry, operation
        ).to_dict()
        self.assertIsNone(payload["peak_memory_bytes"])
        self.assertIsNone(payload["energy_microjoules"])

    def test_component_time_cannot_exceed_total(self):
        with self.assertRaisesRegex(ValueError, "invalid device benchmark measurement"):
            DeviceBenchmarkMeasurement(10, 8, 2, 1, 1, None, None, "b" * 64)

    def test_private_atomic_report_round_trip_and_tamper_rejection(self):
        report = run_device_microbenchmark(
            self.config, self.registry, self.measurement, cold_load=lambda: 20
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            path = save_device_benchmark(report, root / "benchmark.json")
            loaded = load_device_benchmark(
                path,
                hardware_fingerprint="m4-test",
                environment_fingerprint="environment-test",
                capability_id=self.capability.capability_id,
            )
            self.assertEqual(loaded.report_id, report.report_id)
            payload = json.loads(path.read_text())
            payload["median_total_nanoseconds"] += 1
            path.write_text(json.dumps(payload))
            path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "derived values"):
                load_device_benchmark(
                    path,
                    hardware_fingerprint="m4-test",
                    environment_fingerprint="environment-test",
                    capability_id=self.capability.capability_id,
                )

    def test_native_cpu_adapter_runs_bounded_matmul(self):
        config = DeviceBenchmarkConfig(
            "matmul",
            ExecutionBackend.CPU,
            WorkloadPhase.PREFILL,
            "fp32",
            (8, 8, 8),
            2,
            samples=2,
        )
        registry = DeviceCapabilityRegistry("m4-test", "environment-test")
        registry.record(DeviceCapability(
            ExecutionBackend.CPU,
            ComputeDevice.CPU,
            "python-test",
            "m4-test",
            "environment-test",
            ("matmul",),
            (WorkloadPhase.PREFILL,),
            ("fp32",),
            "available",
            "probe_passed",
            ("d" * 24,),
        ))
        adapter = NativeCPUBenchmarkAdapter()
        report = run_device_microbenchmark(config, registry, adapter.operation(config))
        payload = report.to_dict()
        self.assertGreater(payload["throughput_work_items_per_second"], 0)
        self.assertIsNone(payload["peak_memory_bytes"])
        self.assertEqual(payload["sample_count"], 2)

    def test_coreml_adapter_preserves_end_to_end_and_device_time(self):
        resource = CoreMLFixedGraphResource(
            "e" * 32, "coreml_fixed_graph@" + "a" * 16, "f" * 24, "a" * 64
        )
        backend = Mock(spec=CoreMLFixedGraphBackend)
        backend.execute.return_value = CoreMLFixedGraphResult(
            (2.0, 4.0), 50, ExecutionBackend.COREML_DRAFT, "f" * 24
        )
        adapter = CoreMLFixedGraphBenchmarkAdapter(backend, resource)
        config = DeviceBenchmarkConfig(
            resource.operator,
            ExecutionBackend.COREML_DRAFT,
            WorkloadPhase.AUXILIARY,
            "fp32",
            (2,),
            1,
        )
        measurement = adapter.measure(config, 0, input_values=(1.0, 2.0))
        self.assertGreaterEqual(measurement.total_nanoseconds, 1)
        self.assertLessEqual(measurement.execution_nanoseconds, measurement.total_nanoseconds)
        self.assertEqual(measurement.work_items, 2)
        expected_digest = hashlib.sha256(b"[2.0,4.0]").hexdigest()
        self.assertEqual(measurement.output_digest, expected_digest)

    def test_bounded_cpu_reference_matches_accelerator_workload_identity(self):
        operator = "coreml_fixed_graph@" + "a" * 16
        adapter = BoundedCPUReferenceBenchmarkAdapter(
            operator,
            lambda values: tuple(value * 2 for value in values),
            (1.0, -2.0, 0.5, 4.0),
        )
        config = DeviceBenchmarkConfig(
            operator, ExecutionBackend.CPU, WorkloadPhase.AUXILIARY,
            "fp32", (4,), 1, samples=3,
        )
        first = adapter.operation(config)(0)
        second = adapter.operation(config)(1)
        self.assertEqual(first.output_digest, second.output_digest)
        self.assertEqual(first.work_items, 4)

        wrong_shape = DeviceBenchmarkConfig(
            operator, ExecutionBackend.CPU, WorkloadPhase.AUXILIARY,
            "fp32", (8,), 1,
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            adapter.operation(wrong_shape)

    def test_coreml_adapter_binds_tolerant_output_to_reference_digest(self):
        resource = CoreMLFixedGraphResource(
            "e" * 32, "coreml_fixed_graph@" + "a" * 16, "f" * 24, "a" * 64
        )
        backend = Mock(spec=CoreMLFixedGraphBackend)
        backend.execute.return_value = CoreMLFixedGraphResult(
            (2.0001, 3.9999), 10, ExecutionBackend.COREML_DRAFT, "f" * 24
        )
        adapter = CoreMLFixedGraphBenchmarkAdapter(backend, resource)
        config = DeviceBenchmarkConfig(
            resource.operator, ExecutionBackend.COREML_DRAFT,
            WorkloadPhase.AUXILIARY, "fp32", (2,), 1,
        )
        measurement = adapter.measure(
            config, 0, input_values=(1.0, 2.0), expected_values=(2.0, 4.0),
            maximum_absolute_error=0.001,
        )
        self.assertEqual(
            measurement.output_digest,
            hashlib.sha256(b"[2.0,4.0]").hexdigest(),
        )
        with self.assertRaisesRegex(ValueError, "output mismatch"):
            adapter.measure(
                config, 0, input_values=(1.0, 2.0), expected_values=(2.0, 4.0),
                maximum_absolute_error=0.00001,
            )

    def test_native_kernel_adapter_captures_kernel_and_end_to_end_time(self):
        native = Mock()
        native.measure_operator.return_value = KernelMeasurement("a" * 64, 10)
        adapter = NativeKernelBenchmarkAdapter(native, ExecutionBackend.NATIVE_MLX)
        config = DeviceBenchmarkConfig(
            "matmul",
            ExecutionBackend.NATIVE_MLX,
            WorkloadPhase.PREFILL,
            "fp32",
            (8, 8, 8),
            2,
        )
        measurement = adapter.operation(config)(0)
        native.measure_operator.assert_called_once_with("matmul")
        self.assertGreaterEqual(measurement.total_nanoseconds, 1)
        self.assertLessEqual(measurement.execution_nanoseconds, measurement.total_nanoseconds)
        self.assertEqual(measurement.work_items, 1024)

    def test_native_kernel_adapter_rejects_backend_mismatch(self):
        adapter = NativeKernelBenchmarkAdapter(Mock(measure_operator=Mock()), ExecutionBackend.NATIVE_METAL)
        config = DeviceBenchmarkConfig(
            "vector_add", ExecutionBackend.NATIVE_MLX, WorkloadPhase.AUXILIARY,
            "fp32", (256,), 1,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            adapter.operation(config)

    def test_representative_suite_uses_only_available_known_fp32_capabilities(self):
        registry = DeviceCapabilityRegistry("m4-test", "environment-test")
        registry.record(DeviceCapability(
            ExecutionBackend.CPU, ComputeDevice.CPU, "python-test", "m4-test",
            "environment-test", ("matmul",), (WorkloadPhase.PREFILL,), ("fp32",),
            "available", "probe_passed", ("1" * 24,),
        ))
        registry.record(DeviceCapability(
            ExecutionBackend.NATIVE_MLX, ComputeDevice.GPU, "mlx-test", "m4-test",
            "environment-test", ("unknown_kernel",), (WorkloadPhase.PREFILL,),
            ("fp32",), "available", "probe_passed", ("2" * 24,),
        ))
        registry.record(DeviceCapability(
            ExecutionBackend.NATIVE_METAL, ComputeDevice.GPU, "metal-test", "m4-test",
            "environment-test", ("vector_add",), (WorkloadPhase.AUXILIARY,),
            ("fp16",), "available", "probe_passed", ("3" * 24,),
        ))
        configs = representative_device_benchmark_configs(registry, samples=2)
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].dimensions, (8, 8, 8))
        self.assertEqual(configs[0].backend, ExecutionBackend.CPU)

    def test_runs_representative_suite_with_explicit_operations(self):
        config = DeviceBenchmarkConfig(
            "matmul_8x8", ExecutionBackend.CPU, WorkloadPhase.PREFILL, "fp32",
            (8, 8, 8), 1, samples=3,
        )
        suite = run_device_benchmark_suite(
            self.registry,
            (config,),
            {(ExecutionBackend.CPU, "matmul_8x8"): self.measurement},
        )
        self.assertIsInstance(suite, DeviceBenchmarkSuite)
        self.assertEqual(len(suite.reports), 1)
        self.assertEqual(len(suite.suite_id), 24)

    def test_suite_rejects_missing_operation(self):
        config = DeviceBenchmarkConfig(
            "matmul_8x8", ExecutionBackend.CPU, WorkloadPhase.PREFILL, "fp32",
            (8, 8, 8), 1, samples=3,
        )
        with self.assertRaisesRegex(ValueError, "operation is missing"):
            run_device_benchmark_suite(self.registry, (config,), {})

    def test_suite_records_explicit_cold_load(self):
        config = DeviceBenchmarkConfig(
            "matmul_8x8", ExecutionBackend.CPU, WorkloadPhase.PREFILL, "fp32",
            (8, 8, 8), 1, samples=3,
        )
        suite = run_device_benchmark_suite(
            self.registry,
            (config,),
            {(ExecutionBackend.CPU, "matmul_8x8"): self.measurement},
            cold_loads={(ExecutionBackend.CPU, "matmul_8x8"): lambda: 25},
        )
        self.assertEqual(suite.reports[0].cold_load_nanoseconds, 25)


if __name__ == "__main__":
    unittest.main()
