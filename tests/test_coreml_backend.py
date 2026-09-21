import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from vllm_apple.ane_probe import CoreMLANEModelProbeConfig, CoreMLPrediction
from vllm_apple.coreml_backend import CoreMLFixedGraphBackend
from vllm_apple.coreml_worker_cache import CoreMLWorkerCache
from vllm_apple.device_capability import (
    ComputeDevice,
    DeviceCapability,
    DeviceCapabilityRegistry,
)
from vllm_apple.execution import ExecutionBackend, WorkloadPhase
from vllm_apple.operator_dispatch import BackendExecutionError


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
        self.worker = Mock()
        self.worker.predict.return_value = CoreMLPrediction((2.0, 4.0), 50)
        self.backend = CoreMLFixedGraphBackend(
            worker_factory=lambda _config: self.worker
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_probe_bound_resource_load_execute_dispatch_and_unload(self):
        backend = self.backend
        evidence = {"root_sha256": self.digest}
        registry = DeviceCapabilityRegistry("m4-test", "environment-test")
        registry.record(self.capability)
        with patch(
            "vllm_apple.coreml_backend.verify_model_integrity",
            return_value=evidence,
        ):
            resource = backend.load(self.config, self.capability)
            backend.require_auxiliary_dispatch(registry, resource)
            result = backend.execute(resource, (1.0, 2.0))
            backend.unload(resource)
        self.assertEqual(result.values, (2.0, 4.0))
        self.assertEqual(result.backend, ExecutionBackend.COREML_DRAFT)
        self.assertEqual(result.capability_id, self.capability.capability_id)
        self.worker.predict.assert_called_once_with((1.0, 2.0))
        self.worker.close.assert_called_once_with()
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
        backend = self.backend
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
        backend = self.backend
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
            self.assertRaisesRegex(ValueError, "changed during execution"),
        ):
            backend.execute(resource, (1.0, 2.0))

    def test_reuses_one_worker_for_multiple_predictions(self):
        evidence = {"root_sha256": self.digest}
        with patch(
            "vllm_apple.coreml_backend.verify_model_integrity",
            return_value=evidence,
        ):
            resource = self.backend.load(self.config, self.capability)
            self.backend.execute(resource, (1.0, 2.0))
            self.backend.execute(resource, (1.0, 2.0))
            self.backend.unload(resource)
        self.assertEqual(self.worker.predict.call_count, 2)
        self.worker.close.assert_called_once_with()

    def test_worker_timeout_becomes_retryable_fallback_error(self):
        evidence = {"root_sha256": self.digest}
        self.worker.predict.side_effect = TimeoutError("private detail")
        with patch(
            "vllm_apple.coreml_backend.verify_model_integrity",
            return_value=evidence,
        ):
            resource = self.backend.load(self.config, self.capability)
            with self.assertRaises(BackendExecutionError) as raised:
                self.backend.execute(resource, (1.0, 2.0))
            self.backend.unload(resource)
        self.assertEqual(raised.exception.error_code, "coreml_worker_timeout")
        self.assertTrue(raised.exception.retryable)

    def test_identity_cache_reuses_worker_across_resource_lifetimes(self):
        created = []

        def factory(_config):
            worker = Mock()
            worker.predict.return_value = CoreMLPrediction((2.0, 4.0), 50)
            created.append(worker)
            return worker

        cache = CoreMLWorkerCache(2, worker_factory=factory)
        backend = CoreMLFixedGraphBackend(
            worker_cache=cache,
            hardware_fingerprint="m4-test",
            os_version="macos-test",
        )
        evidence = {"root_sha256": self.digest}
        with patch(
            "vllm_apple.coreml_backend.verify_model_integrity",
            return_value=evidence,
        ):
            first = backend.load(self.config, self.capability)
            backend.execute(first, (1.0, 2.0))
            backend.unload(first)
            second = backend.load(self.config, self.capability)
            backend.execute(second, (1.0, 2.0))
            backend.close()
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].predict.call_count, 2)
        created[0].close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
