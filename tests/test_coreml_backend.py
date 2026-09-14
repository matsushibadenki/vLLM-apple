import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_apple.ane_probe import CoreMLANEModelProbeConfig, CoreMLPrediction
from vllm_apple.coreml_backend import CoreMLFixedGraphBackend
from vllm_apple.device_capability import (
    ComputeDevice,
    DeviceCapability,
    DeviceCapabilityRegistry,
)
from vllm_apple.execution import ExecutionBackend, WorkloadPhase


class CoreMLFixedGraphBackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.model = root / "fixture.mlmodelc"
        self.model.mkdir()
        (self.model / "model.bin").write_bytes(b"fixture")
        self.manifest = root / "integrity.json"
        self.manifest.write_text("{}")
        self.digest = "a" * 64
        self.config = CoreMLANEModelProbeConfig(
            self.model,
            self.manifest,
            self.digest,
            "input",
            "output",
            (1.0, 2.0),
            (2.0, 4.0),
            100,
        )
        self.capability = DeviceCapability(
            ExecutionBackend.COREML_DRAFT,
            ComputeDevice.ANE,
            "coreml-test",
            "m4-test",
            "environment-test",
            ("coreml_fixed_graph@" + "a" * 16,),
            (WorkloadPhase.AUXILIARY,),
            ("fp32",),
            "available",
            "probe_passed",
            ("b" * 24,),
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_probe_bound_resource_load_execute_dispatch_and_unload(self):
        backend = CoreMLFixedGraphBackend()
        evidence = {"root_sha256": self.digest}
        registry = DeviceCapabilityRegistry("m4-test", "environment-test")
        registry.record(self.capability)
        with patch(
            "vllm_apple.coreml_backend.verify_model_integrity",
            return_value=evidence,
        ):
            resource = backend.load(self.config, self.capability)
            backend.require_auxiliary_dispatch(registry, resource)
            with patch(
                "vllm_apple.coreml_backend.run_coreml_prediction",
                return_value=CoreMLPrediction((2.0, 4.0), 50),
            ):
                result = backend.execute(resource, (1.0, 2.0))
            backend.unload(resource)
        self.assertEqual(result.values, (2.0, 4.0))
        self.assertEqual(result.backend, ExecutionBackend.COREML_DRAFT)
        self.assertEqual(result.capability_id, self.capability.capability_id)
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            backend.execute(resource, (1.0, 2.0))

    def test_rejects_capability_for_another_model(self):
        config = CoreMLANEModelProbeConfig(
            self.model,
            self.manifest,
            "c" * 64,
            "input",
            "output",
            (1.0, 2.0),
            (2.0, 4.0),
            100,
        )
        with self.assertRaisesRegex(ValueError, "not execution eligible"):
            CoreMLFixedGraphBackend().load(config, self.capability)

    def test_rejects_non_auxiliary_or_unavailable_capability(self):
        capability = DeviceCapability(
            ExecutionBackend.COREML_DRAFT,
            ComputeDevice.ANE,
            "coreml-test",
            "m4-test",
            "environment-test",
            ("coreml_fixed_graph@" + "a" * 16,),
            (WorkloadPhase.DECODE,),
            ("fp32",),
            "available",
            "probe_passed",
            ("b" * 24,),
        )
        with self.assertRaisesRegex(ValueError, "not execution eligible"):
            CoreMLFixedGraphBackend().load(self.config, capability)

    def test_dispatch_rejects_resource_from_stale_capability_evidence(self):
        backend = CoreMLFixedGraphBackend()
        with patch(
            "vllm_apple.coreml_backend.verify_model_integrity",
            return_value={"root_sha256": self.digest},
        ):
            resource = backend.load(self.config, self.capability)
        replacement = DeviceCapability(
            ExecutionBackend.COREML_DRAFT,
            ComputeDevice.ANE,
            "coreml-test",
            "m4-test",
            "environment-test",
            self.capability.operators,
            (WorkloadPhase.AUXILIARY,),
            ("fp32",),
            "available",
            "probe_passed",
            ("c" * 24,),
        )
        registry = DeviceCapabilityRegistry("m4-test", "environment-test")
        registry.record(replacement)
        with self.assertRaisesRegex(RuntimeError, "evidence is stale"):
            backend.require_auxiliary_dispatch(registry, resource)

    def test_detects_integrity_change_during_execution(self):
        backend = CoreMLFixedGraphBackend()
        evidence = {"root_sha256": self.digest}
        with patch(
            "vllm_apple.coreml_backend.verify_model_integrity",
            return_value=evidence,
        ):
            resource = backend.load(self.config, self.capability)
        with (
            patch(
                "vllm_apple.coreml_backend.verify_model_integrity",
                side_effect=[evidence, {"root_sha256": self.digest, "changed": True}],
            ),
            patch(
                "vllm_apple.coreml_backend.run_coreml_prediction",
                return_value=CoreMLPrediction((2.0, 4.0), 50),
            ),
            self.assertRaisesRegex(ValueError, "changed during execution"),
        ):
            backend.execute(resource, (1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
