import hashlib
import tempfile
import unittest
from pathlib import Path

from vllm_apple.kernel_probe import KernelMeasurement
from vllm_apple.model_integrity import (
    build_model_integrity_manifest,
    save_model_integrity_manifest,
)
from vllm_apple.qwen3_vl_ane import Qwen3VLVisionANEAdapterSpec
from vllm_apple.qwen3_vl_coreml import (
    Qwen3VLCoreMLConversionManifest,
    load_qwen3_vl_coreml_conversion,
    qualify_qwen3_vl_coreml_conversion,
    save_qwen3_vl_coreml_conversion,
)


class Qwen3VLCoreMLTests(unittest.TestCase):
    def source(self, fingerprint="a" * 64, revision="b" * 40):
        return Qwen3VLVisionANEAdapterSpec(
            revision, fingerprint, f"vision_encoder@{fingerprint[:16]}",
            2, 16, 32, 4, 16, 2, 2, 24, (0,), 20, 100,
            "affine-int4", False,
        )

    def artifact(self, root: Path, source=None):
        source = source or self.source()
        model = root / "vision.mlmodelc"
        model.mkdir()
        (model / "model.bin").write_bytes(b"compiled-coreml")
        integrity = root / "integrity.json"
        identity = build_model_integrity_manifest(model)
        save_model_integrity_manifest(identity, integrity)
        conversion = Qwen3VLCoreMLConversionManifest(
            source.artifact_fingerprint,
            source.model_revision,
            identity["root_sha256"],
            "pixel_values",
            "image_embeddings",
            (1, 3, 2, 16, 16),
            (1, 4, 24),
            "fp16",
            "coremltools-8.1",
        )
        path = save_qwen3_vl_coreml_conversion(conversion, root / "conversion.json")
        return source, conversion, path, model, integrity

    def measurement(self, values, latency):
        digest = hashlib.sha256(repr(values).encode()).hexdigest()
        return KernelMeasurement(digest, latency, values)

    def test_conversion_manifest_is_source_and_integrity_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            source, conversion, path, model, integrity = self.artifact(Path(directory))
            loaded = load_qwen3_vl_coreml_conversion(
                path,
                source=source,
                coreml_model_path=model,
                integrity_manifest_path=integrity,
            )
            self.assertEqual(loaded, conversion)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(len(loaded.conversion_id), 24)

    def test_conversion_replay_for_another_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, path, model, integrity = self.artifact(Path(directory))
            other = self.source(fingerprint="c" * 64)
            with self.assertRaisesRegex(ValueError, "source does not match"):
                load_qwen3_vl_coreml_conversion(
                    path,
                    source=other,
                    coreml_model_path=model,
                    integrity_manifest_path=integrity,
                )
            self.assertNotEqual(source.artifact_fingerprint, other.artifact_fingerprint)

    def test_compiled_tree_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, path, model, integrity = self.artifact(Path(directory))
            (model / "model.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "integrity"):
                load_qwen3_vl_coreml_conversion(
                    path,
                    source=source,
                    coreml_model_path=model,
                    integrity_manifest_path=integrity,
                )

    def test_numeric_and_performance_pass_promotes_ane_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            source, conversion, _, _, _ = self.artifact(Path(directory))
            def baseline():
                return self.measurement((0.1, 0.2), 200)

            def candidate():
                return self.measurement((0.1001, 0.1999), 100)
            result, capability = qualify_qwen3_vl_coreml_conversion(
                source,
                conversion,
                hardware_fingerprint="m4-test",
                environment_fingerprint="macos-test",
                baseline=baseline,
                candidate=candidate,
                maximum_absolute_error=0.001,
            )
        self.assertTrue(result.passed)
        self.assertIsNotNone(capability)
        self.assertEqual(capability.operators, (source.operator,))
        self.assertEqual(capability.evidence_ids, (result.probe_id,))

    def test_numeric_mismatch_or_slowdown_never_promotes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, conversion, _, _, _ = self.artifact(Path(directory))
            def reference():
                return self.measurement((0.1, 0.2), 100)
            mismatch, mismatch_capability = qualify_qwen3_vl_coreml_conversion(
                source,
                conversion,
                hardware_fingerprint="m4-test",
                environment_fingerprint="macos-test",
                baseline=reference,
                candidate=lambda: self.measurement((1.0, 2.0), 50),
            )
            slow, slow_capability = qualify_qwen3_vl_coreml_conversion(
                source,
                conversion,
                hardware_fingerprint="m4-test",
                environment_fingerprint="macos-test",
                baseline=reference,
                candidate=lambda: self.measurement((0.1, 0.2), 101),
            )
        self.assertEqual(mismatch.reason, "correctness_mismatch")
        self.assertIsNone(mismatch_capability)
        self.assertEqual(slow.reason, "performance_regression")
        self.assertIsNone(slow_capability)


if __name__ == "__main__":
    unittest.main()
