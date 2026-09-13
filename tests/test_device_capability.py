import unittest

from vllm_apple.device_capability import (
    ComputeDevice,
    DeviceCapability,
    DeviceCapabilityRegistry,
    DeviceEligibilityRequest,
    device_capability_from_probe,
)
from vllm_apple.execution import ExecutionBackend, WorkloadPhase
from vllm_apple.kernel_probe import KernelProbeConfig, KernelMeasurement, run_kernel_probe


class DeviceCapabilityRegistryTests(unittest.TestCase):
    def capability(self, backend, device, **changes):
        values = {
            "backend": backend,
            "device": device,
            "backend_version": "1",
            "hardware_fingerprint": "m4-test",
            "environment_fingerprint": "macos-test",
            "operators": ("matmul",),
            "phases": (WorkloadPhase.PREFILL, WorkloadPhase.DECODE),
            "precisions": ("fp16",),
            "status": "available",
            "reason": "probe_passed",
            "evidence_ids": ("a" * 24,),
        }
        values.update(changes)
        return DeviceCapability(**values)

    def test_selects_only_exact_operator_phase_and_precision_matches(self):
        registry = DeviceCapabilityRegistry("m4-test", "macos-test")
        registry.record(self.capability(
            ExecutionBackend.NATIVE_MLX, ComputeDevice.GPU,
            phases=(WorkloadPhase.PREFILL,),
        ))
        registry.record(self.capability(ExecutionBackend.CPU, ComputeDevice.CPU))
        decision = registry.decide(DeviceEligibilityRequest(
            "matmul", WorkloadPhase.DECODE, "fp16",
            (ExecutionBackend.NATIVE_MLX, ExecutionBackend.CPU),
        ))
        self.assertEqual(decision.selected, ExecutionBackend.CPU)
        self.assertEqual(
            decision.rejected,
            ((ExecutionBackend.NATIVE_MLX, "phase_ineligible"),),
        )

    def test_ane_is_explicit_and_unknown_backend_fails_closed(self):
        registry = DeviceCapabilityRegistry("m4-test", "macos-test")
        registry.record(self.capability(
            ExecutionBackend.COREML_DRAFT,
            ComputeDevice.ANE,
            operators=("draft",),
            phases=(WorkloadPhase.AUXILIARY,),
        ))
        decision = registry.decide(DeviceEligibilityRequest(
            "draft", WorkloadPhase.AUXILIARY, "fp16",
            (ExecutionBackend.COREML_DRAFT, ExecutionBackend.CPU),
        ))
        self.assertEqual(decision.selected, ExecutionBackend.COREML_DRAFT)
        self.assertEqual(decision.rejected, ((ExecutionBackend.CPU, "unprobed"),))

    def test_quarantine_is_sticky_for_same_environment(self):
        registry = DeviceCapabilityRegistry("m4-test", "macos-test")
        registry.record(self.capability(
            ExecutionBackend.NATIVE_METAL,
            ComputeDevice.GPU,
            status="quarantined",
            reason="correctness_mismatch",
        ))
        with self.assertRaisesRegex(ValueError, "sticky"):
            registry.record(self.capability(
                ExecutionBackend.NATIVE_METAL, ComputeDevice.GPU
            ))

    def test_backend_device_mismatch_and_profile_replay_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "do not match"):
            self.capability(ExecutionBackend.CPU, ComputeDevice.GPU)
        registry = DeviceCapabilityRegistry("another-chip", "macos-test")
        with self.assertRaisesRegex(ValueError, "profile"):
            registry.record(self.capability(ExecutionBackend.CPU, ComputeDevice.CPU))

    def test_promotes_existing_kernel_probe_evidence(self):
        def measurement():
            return KernelMeasurement("a" * 64, 10)

        result = run_kernel_probe(
            KernelProbeConfig(
                "m4-test", "macos-test", ExecutionBackend.NATIVE_MLX, "matmul",
                samples=1,
            ),
            measurement,
            measurement,
        )
        capability = device_capability_from_probe(
            result,
            device=ComputeDevice.GPU,
            backend_version="mlx-1",
            phases=(WorkloadPhase.PREFILL,),
            precisions=("fp16",),
        )
        self.assertEqual(capability.evidence_ids, (result.probe_id,))
        self.assertEqual(capability.status, "available")


if __name__ == "__main__":
    unittest.main()
