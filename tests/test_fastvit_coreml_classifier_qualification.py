from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.qualify_fastvit_coreml_classifier import _artifact_digest, _validated_payload


class FastViTCoreMLClassifierQualificationTests(unittest.TestCase):
    def test_artifact_digest_is_content_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights").mkdir()
            (root / "model.mlmodel").write_bytes(b"model")
            (root / "weights" / "weight.bin").write_bytes(b"weight")
            first = _artifact_digest(root)
            self.assertEqual(first[1:], (2, 11))
            (root / "weights" / "weight.bin").write_bytes(b"changed")
            self.assertNotEqual(_artifact_digest(root)[0], first[0])

    def test_validates_input_distinct_distributions(self) -> None:
        samples = [{
            "sample_index": index,
            "latency_nanoseconds": 100,
            "class_count": 1000,
            "unique_label_count": 999,
            "top_label": f"label-{index}",
            "top_probability": 0.2,
            "probability_sum": 1.0,
            "probability_sha256": f"{index + 1:064x}",
        } for index in range(3)]
        payload = {
            "schema_version": 1,
            "scope": "fastvit_t8_coreml_classifier_qualification",
            "compute_units": "cpu_and_neural_engine",
            "input": {"name": "image", "shape": [256, 256, 3]},
            "outputs": {
                "label": "classLabel",
                "probabilities": "classLabel_probs",
                "probability_array": "classLabelProbs",
                "class_count": 1000,
                "unique_label_count": 999,
            },
            "sample_count": 3,
            "samples": samples,
            "passed": True,
        }
        self.assertTrue(_validated_payload(json.dumps(payload).encode())["passed"])
        samples[1]["probability_sha256"] = samples[0]["probability_sha256"]
        with self.assertRaisesRegex(RuntimeError, "input-distinct"):
            _validated_payload(json.dumps(payload).encode())


if __name__ == "__main__":
    unittest.main()
