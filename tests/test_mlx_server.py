import io
import json
import socket
import unittest
from email.message import Message
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm_apple.backend_memory import MLXMemoryMetricsAdapter
from vllm_apple.generation_admission import (
    GenerationAdmission,
    GenerationAdmissionError,
    peer_disconnected,
)
from vllm_apple.mlx_server import (
    GenerationBodyError,
    bounded_cache_nbytes,
    prompt_cache_metrics,
    read_generation_body,
    tokenize_chat_request,
)


class GenerationBodyTests(unittest.TestCase):
    def test_closed_peer_with_unread_body_is_cancelled_before_generation(self):
        receiver, sender = socket.socketpair()
        try:
            sender.sendall(b'{}')
            sender.close()
            self.assertFalse(peer_disconnected(receiver))
            handler = self.handler()
            handler.connection = receiver
            with receiver.makefile('rb') as stream:
                handler.rfile = stream
                gate = GenerationAdmission(1, 1)
                with gate.admit():
                    with self.assertRaisesRegex(GenerationAdmissionError, 'cancelled'):
                        with gate.admit(lambda: peer_disconnected(receiver),
                                        lambda: read_generation_body(handler)):
                            self.fail('disconnected request generated')
                with gate.admit():
                    pass
        finally:
            sender.close()
            receiver.close()

    def handler(self, body=b'{}', length='2'):
        headers = Message()
        headers['Content-Length'] = length
        connection = Mock()
        connection.gettimeout.return_value = None
        return SimpleNamespace(headers=headers, connection=connection, rfile=io.BytesIO(body))

    def test_body_replay_and_timeout_restoration(self):
        handler = self.handler()
        self.assertEqual(read_generation_body(handler), b'{}')
        self.assertIsNone(handler.connection.settimeout.call_args.args[0])
        handler = self.handler(b'{')
        with self.assertRaisesRegex(GenerationBodyError, 'incomplete'):
            read_generation_body(handler)
        self.assertIsNone(handler.connection.settimeout.call_args.args[0])

    def test_framing_size_and_deadline(self):
        for length in ('invalid', '-1', '8388609'):
            with self.subTest(length=length), self.assertRaises(GenerationBodyError):
                read_generation_body(self.handler(length=length))
        handler = self.handler()
        handler.headers['Content-Length'] = '2'
        with self.assertRaisesRegex(GenerationBodyError, 'framing'):
            read_generation_body(handler)
        handler = self.handler()
        with patch('vllm_apple.mlx_server.time.monotonic', side_effect=[0, 11]):
            with self.assertRaisesRegex(GenerationBodyError, 'timeout'):
                read_generation_body(handler)


class FakeArray:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes


class PromptCacheMetricsTests(unittest.TestCase):
    def test_legacy_and_lru_sources_are_distinct(self):
        legacy = prompt_cache_metrics(SimpleNamespace(cache=[FakeArray(24)], tokens=[1, 2]))
        self.assertEqual(legacy['kv_cache_bytes'], 24)
        self.assertTrue(legacy['traversal_complete'])
        modern = prompt_cache_metrics(SimpleNamespace(nbytes=48))
        self.assertEqual(modern['kv_cache_bytes'], 48)
        self.assertIsNone(modern['kv_cache_tokens'])
        self.assertFalse(modern['traversal_complete'])
        self.assertEqual(modern['kv_measurement_source'], 'backend_lru_accounting')
        with self.assertRaises(ValueError):
            prompt_cache_metrics(SimpleNamespace(nbytes=True))


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
