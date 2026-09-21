import unittest

from vllm_apple.numeric_container import (
    NumericContainer,
    QuantizationRecipe,
    inspect_numeric_artifact_metadata,
)
from vllm_apple.numeric_routing import NumericFormat


class NumericContainerTests(unittest.TestCase):
    def test_gptq_awq_mlx_coreml_and_gguf_are_normalized(self):
        cases = (
            (NumericContainer.SAFETENSORS, {
                "quant_method": "gptq", "bits": 4, "group_size": 128,
                "packing": "marlin", "compute_dtype": "fp16", "exporter": "tool.v1",
            }, QuantizationRecipe.GPTQ, NumericFormat.INT4),
            (NumericContainer.SAFETENSORS, {
                "quant_method": "awq", "bits": 8, "group_size": 64,
                "packing": "gemm", "compute_dtype": "bf16", "exporter": "tool.v1",
            }, QuantizationRecipe.AWQ, NumericFormat.INT8),
            (NumericContainer.MLX, {
                "quant_method": "affine", "bits": 4, "group_size": 64,
                "compute_dtype": "fp16", "exporter": "mlx.v1",
            }, QuantizationRecipe.MLX_AFFINE, NumericFormat.INT4),
            (NumericContainer.COREML, {
                "quant_method": "linear", "bits": 8,
                "compute_dtype": "fp16", "exporter": "coremltools.v1",
            }, QuantizationRecipe.COREML_LINEAR, NumericFormat.INT8),
            (NumericContainer.GGUF, {
                "ggml_type": "Q4_K", "compute_dtype": "fp32", "exporter": "llama.cpp.v1",
            }, QuantizationRecipe.GGUF_Q4_K, NumericFormat.INT4),
        )
        identifiers = set()
        for container, metadata, recipe, storage in cases:
            descriptor = inspect_numeric_artifact_metadata(container, metadata)
            self.assertEqual((descriptor.recipe, descriptor.storage_format), (recipe, storage))
            identifiers.add(descriptor.descriptor_id)
        self.assertEqual(len(identifiers), len(cases))

    def test_container_recipe_and_compute_format_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "GPTQ/AWQ"):
            inspect_numeric_artifact_metadata(NumericContainer.MLX, {
                "quant_method": "gptq", "bits": 4, "group_size": 128,
                "packing": "x", "compute_dtype": "fp16", "exporter": "x",
            })
        with self.assertRaisesRegex(ValueError, "GGUF"):
            inspect_numeric_artifact_metadata(NumericContainer.GGUF, {
                "ggml_type": "UNKNOWN", "compute_dtype": "fp16", "exporter": "x",
            })
        with self.assertRaisesRegex(ValueError, "compute"):
            inspect_numeric_artifact_metadata(NumericContainer.COREML, {
                "quant_method": "linear", "bits": 8,
                "compute_dtype": "int8", "exporter": "x",
            })


if __name__ == "__main__":
    unittest.main()
