import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.qualification_bundle import (
    QualificationBundleError,
    build_qualification_bundle,
    save_qualification_bundle,
    sign_qualification_bundle,
    verify_qualification_bundle,
    verify_signed_qualification_bundle,
)


class QualificationBundleTests(unittest.TestCase):
    def reports(self, root: Path, *, admission: bool = True) -> None:
        qualification = {
            "schema_version": 1,
            "model": "Qwen/Qwen3.8-Flash-Next",
            "backend": "vllm_metal",
            "backend_versions": {
                "vllm": "0.28.0",
                "vllm_metal": "0.3.0",
                "transformers": "5.15.0",
            },
            "requested_modes": ["text"],
            "shutdown_clean": True,
            "promotion_probe": {"passed": True},
            "phase_profile": {"sample_count": 3},
            "quality_smoke": {
                "passed": True,
                "stores_generated_text": False,
                "checks": {
                    "english": True,
                    "japanese": True,
                    "simplified_chinese": True,
                },
            },
            "model_memory_fit": {
                "artifact_bytes": 100,
                "estimated_resident_bytes": 120,
                "hard_ceiling_bytes": 200,
                "fits": True,
            },
            "soak": {"passed": True},
            "context_reevaluation": {"passed": True},
            "passed": True,
        }
        preflight = {
            "schema_version": 1,
            "eligible": True,
            "backend_kind": "vllm_metal",
        }
        (root / "qualification.json").write_text(json.dumps(qualification))
        (root / "preflight.json").write_text(json.dumps(preflight))
        if admission:
            (root / "artifact-admission.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model": qualification["model"],
                        "artifact_bytes": 100,
                        "estimated_resident_bytes": 120,
                        "eligible": True,
                    }
                )
            )

    def test_builds_schema_valid_deterministic_bundle_without_model_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.reports(root)
            first = build_qualification_bundle(root)
            second = build_qualification_bundle(root)
            output = save_qualification_bundle(first, root / "bundle.json")
            self.assertEqual(first, second)
            self.assertNotIn("Qwen", output.read_text())
            schema = json.loads(
                Path("schemas/runtime/qualification-bundle-v1.schema.json").read_text()
            )
            validate_instance(first, schema)
            self.assertEqual(verify_qualification_bundle(root, output), first)

            qualification = json.loads((root / "qualification.json").read_text())
            qualification["load_seconds"] = 999
            (root / "qualification.json").write_text(json.dumps(qualification))
            with self.assertRaisesRegex(QualificationBundleError, "verification failed"):
                verify_qualification_bundle(root, output)

    def test_rejects_failed_or_mismatched_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.reports(root)
            admission = json.loads((root / "artifact-admission.json").read_text())
            admission["estimated_resident_bytes"] = 121
            (root / "artifact-admission.json").write_text(json.dumps(admission))
            with self.assertRaisesRegex(QualificationBundleError, "admission"):
                build_qualification_bundle(root)

    def test_rejects_symlinked_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.reports(root, admission=False)
            target = root / "real.json"
            (root / "qualification.json").replace(target)
            (root / "qualification.json").symlink_to(target)
            with self.assertRaisesRegex(QualificationBundleError, "qualification evidence"):
                build_qualification_bundle(root)

    def test_cms_signed_bundle_requires_trusted_signer_and_unchanged_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.reports(root)
            bundle = root / "bundle.json"
            save_qualification_bundle(build_qualification_bundle(root), bundle)
            key = root / "signer.key"
            certificate = root / "signer.pem"
            subprocess.run(
                [
                    "/usr/bin/openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", str(key), "-out", str(certificate), "-days", "1",
                    "-subj", "/CN=Qualification Bundle Test Signer",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.chmod(key, 0o600)
            fingerprint = subprocess.run(
                [
                    "/usr/bin/openssl", "x509", "-in", str(certificate), "-noout",
                    "-fingerprint", "-sha256",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip().split("=", 1)[1].replace(":", "")
            signature = root / "bundle.cms"
            sign_qualification_bundle(root, bundle, certificate, key, signature)
            verified = verify_signed_qualification_bundle(
                root, bundle, signature, certificate, fingerprint
            )
            self.assertTrue(verified["passed"])
            with self.assertRaisesRegex(QualificationBundleError, "identity"):
                verify_signed_qualification_bundle(
                    root, bundle, signature, certificate, "0" * 64
                )


if __name__ == "__main__":
    unittest.main()
