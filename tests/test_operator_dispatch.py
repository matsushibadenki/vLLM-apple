import hashlib
import json
import subprocess
import unittest
from unittest.mock import patch

from tests.test_scheduler import hardware
from vllm_apple.execution import ExecutionBackend
from vllm_apple.kernel_probe import (
    KernelCapabilityRegistry,
    KernelMeasurement,
    KernelProbeConfig,
    run_kernel_probe,
)
from vllm_apple.kernel_profile import ModelKernelShapeProfile, PagedAttentionShape
from vllm_apple.metal_probe import (
    MetalThreadConfiguration,
    NativeMetalProbeAdapter,
    _model_paged_attention_program,
)
from vllm_apple.mlx_probe import NativeMLXProbeAdapter, build_mlx_probe_registry
from vllm_apple.operator_dispatch import OperatorDispatcher, OperatorDispatchRequest
from vllm_apple.scheduler import BasicScheduler, ScheduleRequest
from vllm_apple.types import Backend


def passing_result(backend: ExecutionBackend, operator: str):
    digest = hashlib.sha256(b"same").hexdigest()
    return run_kernel_probe(
        KernelProbeConfig("hardware", "environment", backend, operator),
        lambda: KernelMeasurement(digest, 100),
        lambda: KernelMeasurement(digest, 100),
    )


class OperatorDispatchTests(unittest.TestCase):
    def test_metal_probe_is_isolated_and_validates_fixed_output(self) -> None:
        values = [float(value + value) for value in range(64)]
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {"output": values, "latency_nanoseconds": 200}
            ).encode(),
            stderr=b"",
        )
        adapter = NativeMetalProbeAdapter()
        with patch(
            "vllm_apple.metal_probe.subprocess.run", return_value=completed
        ) as run:
            result = adapter.probe_vector_add(
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                samples=1,
                maximum_slowdown_ratio=10_000,
            )
        self.assertTrue(result.passed)
        command = run.call_args.args[0]
        self.assertEqual(command[0], "/usr/bin/swift")
        self.assertIn("-e", command)

        bad = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b'{"output":[],"latency_nanoseconds":1}', stderr=b""
        )
        with patch("vllm_apple.metal_probe.subprocess.run", return_value=bad):
            failed = adapter.probe_vector_add(
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                samples=1,
            )
        self.assertTrue(failed.quarantined)
        self.assertEqual(failed.reason, "probe_error")

    def test_metal_paged_attention_is_an_independent_capability(self) -> None:
        adapter = NativeMetalProbeAdapter()
        with patch.object(
            NativeMetalProbeAdapter,
            "_candidate_paged_attention",
            side_effect=adapter._baseline_paged_attention,
        ):
            passed = adapter.probe_paged_attention(
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                samples=1,
            )
        self.assertTrue(passed.passed)
        self.assertEqual(passed.operator, "paged_attention")

        with patch.object(
            NativeMetalProbeAdapter,
            "_candidate_paged_attention",
            side_effect=RuntimeError("native failure"),
        ):
            failed = adapter.probe_paged_attention(
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                samples=1,
            )
        self.assertTrue(failed.quarantined)
        self.assertEqual(failed.reason, "probe_error")

    def test_metal_consumes_bounded_model_shape_profile_independently(self) -> None:
        shapes = tuple(
            PagedAttentionShape(context, 1, 4, 2, 8, 4, (context + 3) // 4, context * 32)
            for context in (8, 16, 32)
        )
        profile = ModelKernelShapeProfile(1, "a" * 24, "model", "test", 2, 2, shapes)
        adapter = NativeMetalProbeAdapter()
        with patch.object(
            NativeMetalProbeAdapter,
            "_candidate_model_shape",
            side_effect=lambda shape: adapter._baseline_model_shape(shape),
        ):
            results = adapter.probe_shape_profile(
                profile,
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                samples=1,
                maximum_shapes=2,
            )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.passed for result in results))
        self.assertNotEqual(results[0].operator, results[1].operator)

    def test_model_shape_probe_rejects_large_representative_allocation(self) -> None:
        shape = PagedAttentionShape(
            262144, 1, 32, 8, 128, 16, 16384, 1024 * 1024 * 1024
        )
        profile = ModelKernelShapeProfile(
            1, "b" * 24, "model", "test", 32, 2, (shape,)
        )
        with self.assertRaises(ValueError):
            NativeMetalProbeAdapter().probe_shape_profile(
                profile,
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
            )

    def test_model_shape_program_has_bounded_three_stage_dispatch(self) -> None:
        shape = PagedAttentionShape(1024, 1, 32, 8, 128, 16, 64, 4194304)
        program = _model_paged_attention_program(shape)
        self.assertIn('name: "attention_scores"', program)
        self.assertIn('name: "attention_softmax"', program)
        self.assertIn('name: "attention_output"', program)
        self.assertEqual(program.count("encoder.memoryBarrier(scope: .buffers)"), 2)
        self.assertIn("length: 1024 * MemoryLayout<Float>.stride", program)
        self.assertIn("threadgroup float scratch[256]", program)
        self.assertIn("encoder.dispatchThreadgroups", program)
        self.assertIn("threadgroup_barrier", program)

    def test_model_shape_tuner_selects_fastest_correct_candidate(self) -> None:
        shape = PagedAttentionShape(1024, 1, 32, 8, 128, 16, 64, 4194304)
        adapter = NativeMetalProbeAdapter()
        baseline = adapter._baseline_model_shape(shape)
        latencies = {32: 400, 64: 300, 128: 100, 256: 200}

        def candidate(_, configuration):
            return KernelMeasurement(
                baseline.output_digest,
                latencies[configuration.score_width],
                baseline.numeric_values,
            )

        with patch.object(
            NativeMetalProbeAdapter, "_candidate_model_shape", side_effect=candidate
        ):
            decision = adapter.tune_model_shape(
                shape,
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                samples=1,
            )
        self.assertEqual(len(decision.candidates), 4)
        self.assertEqual(decision.winner, MetalThreadConfiguration(128, 128, 128))
        self.assertTrue(all(result.passed for _, result in decision.candidates))

    def test_mlx_suite_registers_operators_independently(self) -> None:
        adapter = NativeMLXProbeAdapter()
        baselines = {
            "vector_add": adapter._baseline_vector_add,
            "matmul": adapter._baseline_matmul,
            "kv_copy": adapter._baseline_kv_copy,
            "attention": adapter._baseline_attention,
            "paged_attention": adapter._baseline_paged_attention,
            "mla": adapter._baseline_mla,
        }
        with patch.object(
            NativeMLXProbeAdapter,
            "_candidate",
            side_effect=lambda operator: baselines[operator](),
        ):
            registry = build_mlx_probe_registry(
                adapter,
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                samples=1,
            )
        self.assertEqual(len(registry.snapshot()), 6)
        for operator in baselines:
            self.assertTrue(registry.is_usable(ExecutionBackend.NATIVE_MLX, operator))

        def partially_broken(operator: str):
            if operator == "matmul":
                raise RuntimeError("matmul failed")
            return baselines[operator]()

        with patch.object(
            NativeMLXProbeAdapter, "_candidate", side_effect=partially_broken
        ):
            partial = build_mlx_probe_registry(
                adapter,
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                samples=1,
            )
        self.assertTrue(partial.is_usable(ExecutionBackend.NATIVE_MLX, "vector_add"))
        self.assertFalse(partial.is_usable(ExecutionBackend.NATIVE_MLX, "matmul"))
        self.assertTrue(partial.is_usable(ExecutionBackend.NATIVE_MLX, "kv_copy"))

    def test_quarantined_metal_falls_back_to_probed_mlx(self) -> None:
        registry = KernelCapabilityRegistry("hardware", "environment")
        wrong = hashlib.sha256(b"wrong").hexdigest()
        right = hashlib.sha256(b"right").hexdigest()
        registry.record(
            run_kernel_probe(
                KernelProbeConfig(
                    "hardware",
                    "environment",
                    ExecutionBackend.NATIVE_METAL,
                    "paged_attention",
                ),
                lambda: KernelMeasurement(right, 100),
                lambda: KernelMeasurement(wrong, 50),
            )
        )
        registry.record(passing_result(ExecutionBackend.NATIVE_MLX, "paged_attention"))
        decision = OperatorDispatcher(registry).dispatch(
            OperatorDispatchRequest(
                "paged_attention",
                (
                    ExecutionBackend.NATIVE_METAL,
                    ExecutionBackend.NATIVE_MLX,
                    ExecutionBackend.CPU,
                ),
            )
        )
        self.assertEqual(decision.selected, ExecutionBackend.NATIVE_MLX)
        self.assertEqual(decision.quarantined, (ExecutionBackend.NATIVE_METAL,))

    def test_unprobed_accelerator_fails_closed_to_cpu_in_scheduler(self) -> None:
        registry = KernelCapabilityRegistry("hardware", "environment")
        scheduler = BasicScheduler(
            hardware(), 500, operator_dispatcher=OperatorDispatcher(registry)
        )
        self.assertEqual(
            scheduler.choose_backend(ScheduleRequest("paged_attention", 10)), Backend.CPU
        )
        registry.record(passing_result(ExecutionBackend.NATIVE_METAL, "paged_attention"))
        self.assertEqual(
            scheduler.choose_backend(ScheduleRequest("paged_attention", 10)), Backend.METAL
        )

    def test_mlx_probe_is_isolated_and_native_crash_becomes_quarantine(self) -> None:
        adapter = NativeMLXProbeAdapter()
        values = [float(value + value) for value in range(256)]
        digest = hashlib.sha256(
            json.dumps(values, separators=(",", ":")).encode()
        ).hexdigest()
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {"output_digest": digest, "latency_nanoseconds": 100}
            ).encode(),
            stderr=b"",
        )
        with patch("vllm_apple.mlx_probe.subprocess.run", return_value=completed) as run:
            passed = adapter.probe_vector_add(
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                samples=1,
                maximum_slowdown_ratio=10_000,
            )
        self.assertTrue(passed.passed)
        self.assertIn("-c", run.call_args.args[0])

        with patch(
            "vllm_apple.mlx_probe.subprocess.run",
            side_effect=subprocess.CalledProcessError(-6, ["python"]),
        ):
            failed = adapter.probe_vector_add(
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                samples=1,
            )
        self.assertTrue(failed.quarantined)
        self.assertEqual(failed.reason, "probe_error")


if __name__ == "__main__":
    unittest.main()
