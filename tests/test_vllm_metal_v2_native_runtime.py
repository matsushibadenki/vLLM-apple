import json
import unittest

from vllm_apple.vllm_metal_v2_native_runtime import measure


class VLLMMetalV2NativeRuntimeTests(unittest.TestCase):
    def test_rejects_unsupported_fixture_before_importing_mlx(self) -> None:
        payload = {
            "abi_version": 1,
            "operation": "measure_paged_attention_v2",
            "shape": {
                "context_tokens": 4096,
                "query_tokens": 1,
                "sequences": 1,
                "query_heads": 32,
                "kv_heads": 8,
                "head_size": 128,
                "block_size": 16,
                "gpu_cores": 16,
                "query_dtype": "float16",
                "cache_dtype": "int8",
                "turboquant": False,
                "window_seqlen_q": 1,
            },
            "configuration": {
                "family": "per_token",
                "threads": 256,
                "tile_query": 0,
                "tile_kv": 0,
                "partition_size": 0,
            },
        }
        with self.assertRaisesRegex(ValueError, "requires non-TurboQuant float16"):
            measure(json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
