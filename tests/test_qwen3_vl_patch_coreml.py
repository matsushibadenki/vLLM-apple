import json
import tempfile
import unittest
from pathlib import Path

from vllm_apple.qwen3_vl_ane import Qwen3VLVisionANEAdapterSpec
from vllm_apple.qwen3_vl_patch_coreml import (
    _fixed_position_embedding,
    qualify_qwen3_vl_patch_coreml,
)

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "Qwen3-VL patch Core ML tests require NumPy")
class Qwen3VLPatchCoreMLTests(unittest.TestCase):
    def test_fixed_position_embedding_matches_merge_order(self):
        source = Qwen3VLVisionANEAdapterSpec(
            "b" * 40, "a" * 64, "vision_encoder@" + "a" * 16,
            1, 2, 4, 1, 1, 1, 2, 2, (0,), 1, 1, "bf16", False,
        )
        table = np.asarray(
            [[0, 10], [1, 11], [2, 12], [3, 13]], dtype=np.float16
        )
        result = _fixed_position_embedding(table, (1, 2, 2), source, np)
        self.assertEqual(result.shape, (4, 2))
        self.assertEqual(result.tolist(), table.tolist())

    def test_qualification_rejects_traversal_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "report.json").write_text(json.dumps({
                "schema_version": 1,
                "graph_id": "a" * 64,
                "conversion_plan_id": "b" * 64,
                "partition": "patch-position-v1",
                "grid_thw": [1, 2, 2],
                "input_name": "pixel_values",
                "input_shape": [4, 3],
                "output_name": "patch_hidden_states",
                "output_shape": [4, 2],
                "compute_precision": "fp16",
                "maximum_scaled_error": 0.002,
                "expected_sparse_input_sha256": "c" * 64,
                "sparse_input_reference": "reference.fp16",
                "compiled_model": "../escape.mlmodelc",
            }))
            with self.assertRaisesRegex(ValueError, "report is invalid"):
                qualify_qwen3_vl_patch_coreml(root)


if __name__ == "__main__":
    unittest.main()
