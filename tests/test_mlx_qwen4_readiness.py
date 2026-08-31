import tempfile
import unittest
from pathlib import Path

from vllm_apple.mlx_qwen4_readiness import inspect_mlx_qwen4_sources


class MLXQwen4ReadinessTests(unittest.TestCase):
    def test_reports_reusable_components_without_claiming_architecture_support(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gated_delta.py").write_text(
                "def gated_delta_update(): pass\n", encoding="utf-8"
            )
            (root / "qwen3_next.py").write_text(
                "class Qwen3NextSparseMoeBlock: pass\n", encoding="utf-8"
            )
            (root / "longcat_flash_ngram.py").write_text(
                "class NgramEmbedding: pass\n", encoding="utf-8"
            )
            result = inspect_mlx_qwen4_sources(
                root, version="0.31.3", executable="/venv/bin/mlx_lm.server"
            )
        self.assertFalse(result["ready"])
        self.assertFalse(result["architecture_registered"])
        self.assertEqual(
            result["reusable_components"],
            ["gated_deltanet", "mixture_of_experts", "ngram_embedding"],
        )
        self.assertEqual(
            result["missing_components"],
            ["gated_residual", "native_long_context", "qwen_sparse_attention"],
        )
        self.assertFalse(result["imports_backend"])
        self.assertFalse(result["allocates_model_or_metal"])

    def test_requires_complete_qwen4_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gated_delta.py").write_text(
                "def gated_delta_update(): pass\n", encoding="utf-8"
            )
            (root / "qwen3_next.py").write_text(
                "class Qwen3NextSparseMoeBlock: pass\n", encoding="utf-8"
            )
            (root / "longcat_flash_ngram.py").write_text(
                "class NgramEmbedding: pass\n", encoding="utf-8"
            )
            (root / "qwen4_exp.py").write_text(
                "class QwenSparseAttention: pass\n"
                "class GatedResidual: pass\n"
                "class Model: pass\n",
                encoding="utf-8",
            )
            result = inspect_mlx_qwen4_sources(
                root, version="1.0.0", executable="/venv/bin/mlx_lm.server"
            )
        self.assertTrue(result["ready"])
        self.assertTrue(result["architecture_registered"])


if __name__ == "__main__":
    unittest.main()
