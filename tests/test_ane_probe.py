import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_apple.ane_probe import (
    CoreMLANEModelProbe,
    CoreMLANEModelProbeConfig,
    CoreMLANESurfaceProbe,
)


class CoreMLANESurfaceProbeTests(unittest.TestCase):
    def test_non_apple_platform_is_normal_unavailable_state(self):
        result = CoreMLANESurfaceProbe().probe(
            platform_name="Linux", architecture="x86_64"
        )
        self.assertEqual(result.reason, "non_apple_platform")
        self.assertFalse(result.execution_qualified)

    def test_public_compute_unit_surface_does_not_claim_execution(self):
        completed = subprocess.CompletedProcess(
            [], 0,
            json.dumps({
                "coreml_framework": True,
                "cpu_and_neural_engine": True,
            }).encode(),
            b"",
        )
        with patch("vllm_apple.ane_probe.subprocess.run", return_value=completed):
            result = CoreMLANESurfaceProbe().probe(
                platform_name="Darwin", architecture="arm64"
            )
        self.assertEqual(result.reason, "surface_available_model_probe_required")
        self.assertFalse(result.execution_qualified)
        self.assertEqual(len(result.evidence_id), 24)

    def test_malformed_or_failed_probe_is_fail_closed(self):
        completed = subprocess.CompletedProcess([], 0, b"{}", b"")
        with patch("vllm_apple.ane_probe.subprocess.run", return_value=completed):
            result = CoreMLANESurfaceProbe().probe(
                platform_name="Darwin", architecture="arm64"
            )
        self.assertEqual(result.reason, "probe_error")

    def test_digest_bound_model_probe_passes_correct_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "fixture.mlmodelc"
            model.mkdir()
            (model / "model.bin").write_bytes(b"fixture")
            manifest = root / "integrity.json"
            manifest.write_text("{}")
            config = CoreMLANEModelProbeConfig(
                model,
                manifest,
                "a" * 64,
                "input",
                "output",
                (1.0, 2.0),
                (2.0, 4.0),
                100,
            )
            completed = subprocess.CompletedProcess(
                [], 0,
                json.dumps({
                    "latency_nanoseconds": 50,
                    "output_values": [2.0, 4.0],
                }).encode(),
                b"",
            )
            evidence = {"root_sha256": "a" * 64}
            with (
                patch("vllm_apple.ane_probe.verify_model_integrity", return_value=evidence),
                patch("vllm_apple.ane_probe.subprocess.run", return_value=completed),
            ):
                result = CoreMLANEModelProbe().probe(
                    config,
                    hardware_fingerprint="m4-test",
                    environment_fingerprint="coreml-test",
                )
            self.assertTrue(result.passed)
            self.assertEqual(result.operator, "coreml_fixed_graph@" + "a" * 16)

    def test_model_probe_quarantines_numerical_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "fixture.mlmodelc"
            model.mkdir()
            (model / "model.bin").write_bytes(b"fixture")
            manifest = root / "integrity.json"
            manifest.write_text("{}")
            config = CoreMLANEModelProbeConfig(
                model, manifest, "a" * 64, "input", "output", (1.0,), (2.0,), 100
            )
            completed = subprocess.CompletedProcess(
                [], 0,
                json.dumps({
                    "latency_nanoseconds": 50,
                    "output_values": [3.0],
                }).encode(),
                b"",
            )
            with (
                patch(
                    "vllm_apple.ane_probe.verify_model_integrity",
                    return_value={"root_sha256": "a" * 64},
                ),
                patch("vllm_apple.ane_probe.subprocess.run", return_value=completed),
            ):
                result = CoreMLANEModelProbe().probe(
                    config,
                    hardware_fingerprint="m4-test",
                    environment_fingerprint="coreml-test",
                )
            self.assertTrue(result.quarantined)
            self.assertEqual(result.reason, "correctness_mismatch")


if __name__ == "__main__":
    unittest.main()
