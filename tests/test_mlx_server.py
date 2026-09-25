import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vllm_apple.backend_memory import MLXMemoryMetricsAdapter
from vllm_apple.mlx_server import bounded_cache_nbytes, tokenize_chat_request


class FakeArray:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes


class FakeCache:
    def __init__(self, state: object) -> None:
        self.state = state


class LazyArray:
    def __init__(self, shape, item_size=2):
        self.shape = shape
        self.dtype = SimpleNamespace(size=item_size)

    @property
    def nbytes(self):
        raise AssertionError("lazy array nbytes must not be accessed")


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

    def test_lazy_quantized_and_nested_states_use_metadata(self) -> None:
        packed = LazyArray((2, 3, 4), 4)
        scales = LazyArray((2, 3))
        state = FakeCache({"quantized": (packed, scales, None),
                           "nested": [packed, LazyArray(()), LazyArray((0, 4))]})
        self.assertEqual(bounded_cache_nbytes(state), (110, True))

    def test_cycles_and_exact_budget_are_supported(self) -> None:
        state = [FakeArray(8)]
        state.append(state)
        self.assertEqual(bounded_cache_nbytes(state, maximum_nodes=3), (8, True))
        self.assertEqual(bounded_cache_nbytes(state, maximum_nodes=2), (8, False))

    def test_wide_containers_do_not_expand_before_budget_check(self) -> None:
        class WideList(list):
            def __iter__(self):
                for _ in range(1_000_000):
                    self.visits += 1
                    if self.visits > 5:
                        raise AssertionError("traversal eagerly expanded children")
                    yield None

        state = WideList()
        state.visits = 0
        self.assertEqual(bounded_cache_nbytes(state, maximum_nodes=5), (0, False))
        self.assertEqual(state.visits, 5)

    def test_repeated_aliases_consume_traversal_budget(self) -> None:
        shared = FakeArray(8)
        self.assertEqual(
            bounded_cache_nbytes([shared] * 100, maximum_nodes=5), (8, False)
        )

    def test_invalid_metadata_falls_back_to_nbytes(self) -> None:
        for shape, size in [((True,), 2), ((-1,), 2), ((2,), True), ((2,), None)]:
            array = SimpleNamespace(shape=shape, dtype=SimpleNamespace(size=size), nbytes=8)
            self.assertEqual(bounded_cache_nbytes(array), (8, True))

    def test_invalid_traversal_budgets_are_rejected(self) -> None:
        for budget in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                bounded_cache_nbytes([], maximum_nodes=budget)

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
