import io
import json
import unittest
from unittest.mock import patch

from vllm_apple.backend_memory import MLXMemoryMetricsAdapter
from vllm_apple.mlx_server import bounded_cache_nbytes, tokenize_chat_request


class FakeArray:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes


class FakeCache:
    def __init__(self, state: object) -> None:
        self.state = state


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        return list(range(len(messages[0]["content"]) + 2))


class FakeProvider:
    def load(self, model, draft_model_path):
        assert model == "default_model"
        assert draft_model_path == "default_model"
        return object(), FakeTokenizer()


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class MLXServerTelemetryTests(unittest.TestCase):
    def test_tokenize_contract_returns_only_a_bounded_count(self) -> None:
        count = tokenize_chat_request(
            FakeProvider(),
            {
                "model": "default_model",
                "messages": [{"role": "user", "content": "hello"}],
                "add_generation_prompt": True,
            },
        )
        self.assertEqual(count, 7)
        with self.assertRaisesRegex(ValueError, "preloaded"):
            tokenize_chat_request(
                FakeProvider(),
                {"model": "other", "messages": [{"role": "user", "content": "x"}]},
            )

    def test_cache_traversal_counts_distinct_arrays_and_is_bounded(self) -> None:
        shared = FakeArray(128)
        total, complete = bounded_cache_nbytes([FakeCache((shared, shared)), FakeArray(64)])
        self.assertEqual(total, 192)
        self.assertTrue(complete)
        _, bounded = bounded_cache_nbytes([FakeArray(1), FakeArray(2)], maximum_nodes=1)
        self.assertFalse(bounded)

    def test_metrics_adapter_validates_wrapper_payload(self) -> None:
        payload = json.dumps(
            {
                "schema_version": 1,
                "active_bytes": 100,
                "cache_bytes": 20,
                "peak_bytes": 150,
                "kv_cache_bytes": 16,
                "kv_cache_tokens": 2,
                "traversal_complete": True,
            }
        ).encode()
        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            sample = MLXMemoryMetricsAdapter("http://127.0.0.1:8001").sample()
        self.assertEqual(sample.allocator_current_bytes, 120)
        self.assertEqual(sample.allocator_peak_bytes, 150)
        self.assertEqual(sample.kv_used_bytes, 16)

    def test_incomplete_cache_traversal_fails_closed(self) -> None:
        payload = json.dumps(
            {
                "active_bytes": 1,
                "cache_bytes": 1,
                "peak_bytes": 1,
                "kv_cache_bytes": 1,
                "traversal_complete": False,
            }
        ).encode()
        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                MLXMemoryMetricsAdapter("http://localhost:8001").sample()


if __name__ == "__main__":
    unittest.main()
