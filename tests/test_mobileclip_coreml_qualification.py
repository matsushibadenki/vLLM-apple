from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.qualify_mobileclip_coreml import _artifact_digest, _validated_payload


def _payload() -> dict[str, object]:
    samples = [
        {
            "sample_index": index,
            "latency_nanoseconds": 100 + index,
            "output_count": 512,
            "output_sha256": f"{index + 1:064x}",
            "l2_norm": 1.0 + index,
        }
        for index in range(3)
    ]
    return {
        "schema_version": 1,
        "scope": "mobileclip_s0_coreml_embedding_qualification",
        "compute_units": "cpu_and_neural_engine",
        "image_input": {"name": "image", "shape": [256, 256, 3]},
        "text_input": {"name": "text", "shape": [1, 77], "dtype": "int32"},
        "output": {"name": "final_emb_1", "shape": [1, 512], "dtype": "float32"},
        "image_samples": samples,
        "text_samples": [dict(sample) for sample in samples],
        "sample_count": 3,
        "passed": True,
    }


class MobileCLIPCoreMLQualificationTests(unittest.TestCase):
    def test_artifact_digest_binds_both_packages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (
                ("mobileclip_s0_image.mlpackage", b"image"),
                ("mobileclip_s0_text.mlpackage", b"text"),
            ):
                package = root / name
                package.mkdir()
                (package / "Manifest.json").write_bytes(value)
            first = _artifact_digest(root)
            self.assertEqual(first[1:], (2, 9))
            (root / "mobileclip_s0_text.mlpackage" / "Manifest.json").write_bytes(b"changed")
            self.assertNotEqual(_artifact_digest(root)[0], first[0])

    def test_artifact_digest_rejects_missing_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing"):
                _artifact_digest(Path(directory))

    def test_payload_validation_rejects_duplicate_embedding(self) -> None:
        payload = _payload()
        self.assertEqual(_validated_payload(json.dumps(payload).encode())["passed"], True)
        image_samples = payload["image_samples"]
        assert isinstance(image_samples, list)
        image_samples[1]["output_sha256"] = image_samples[0]["output_sha256"]
        with self.assertRaisesRegex(RuntimeError, "input-distinct"):
            _validated_payload(json.dumps(payload).encode())


if __name__ == "__main__":
    unittest.main()
