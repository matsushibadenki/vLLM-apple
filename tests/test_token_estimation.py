import json
import threading
import unittest
from unittest.mock import patch

from vllm_apple.backend import OpenAIProxyEngine
from vllm_apple.memory_admission import MemoryPressureAdmissionError
from vllm_apple.memory_telemetry import UnifiedMemoryTelemetry
from vllm_apple.service import RuntimeService
from vllm_apple.types import GIB, MemoryPressure


class ChunkedResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0
        self.read_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, size: int) -> bytes:
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        self.read_bytes += len(chunk)
        return chunk


class TokenEstimationTests(unittest.TestCase):
    def test_backend_reads_count_without_retaining_token_array(self) -> None:
        response = ChunkedResponse(b'{"count":321,"tokens":[' + b"1," * 100_000)
        observed_request = None

        def open_request(request, timeout):
            nonlocal observed_request
            observed_request = request
            self.assertEqual(timeout, 5.0)
            return response

        engine = OpenAIProxyEngine("http://127.0.0.1:1")
        request = {
            "model": "model",
            "messages": [{"role": "user", "content": "private prompt"}],
            "tools": [],
            "temperature": 0.7,
        }
        with patch("urllib.request.urlopen", side_effect=open_request):
            self.assertEqual(engine.estimate_prompt_tokens(request), 321)
        self.assertLessEqual(response.read_bytes, 4096)
        sent = json.loads(observed_request.data)
        self.assertNotIn("temperature", sent)
        self.assertEqual(sent["messages"], request["messages"])

    def test_backend_reuses_count_for_identical_canonical_request(self) -> None:
        calls = 0

        def open_request(_request, timeout):
            nonlocal calls
            self.assertEqual(timeout, 5.0)
            calls += 1
            return ChunkedResponse(b'{"count":17,"tokens":[1,2,3]}')

        engine = OpenAIProxyEngine("http://127.0.0.1:1")
        request = {
            "model": "model",
            "messages": [{"role": "user", "content": "private prompt"}],
            "temperature": 0.1,
        }
        with patch("urllib.request.urlopen", side_effect=open_request):
            self.assertEqual(engine.estimate_prompt_tokens(request), 17)
            request["temperature"] = 0.9
            self.assertEqual(engine.estimate_prompt_tokens(request), 17)

        self.assertEqual(calls, 1)
        snapshot = engine.tokenizer_snapshot()
        self.assertEqual(snapshot["measured"], 1)
        self.assertEqual(snapshot["cache_entries"], 1)
        self.assertEqual(snapshot["cache_hits"], 1)
        self.assertEqual(snapshot["cache_misses"], 1)

    def test_backend_does_not_reuse_count_for_different_tools(self) -> None:
        calls = 0

        def open_request(_request, timeout):
            nonlocal calls
            self.assertEqual(timeout, 5.0)
            calls += 1
            return ChunkedResponse(b'{"count":9}')

        engine = OpenAIProxyEngine("http://127.0.0.1:1")
        base = {"model": "model", "messages": [{"role": "user", "content": "hello"}]}
        with patch("urllib.request.urlopen", side_effect=open_request):
            self.assertEqual(engine.estimate_prompt_tokens({**base, "tools": []}), 9)
            self.assertEqual(
                engine.estimate_prompt_tokens({**base, "tools": [{"type": "function"}]}),
                9,
            )
        self.assertEqual(calls, 2)

    def test_backend_coalesces_concurrent_identical_requests(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        calls = 0

        def open_request(_request, timeout):
            nonlocal calls
            self.assertEqual(timeout, 5.0)
            calls += 1
            entered.set()
            self.assertTrue(release.wait(1.0))
            return ChunkedResponse(b'{"count":31}')

        engine = OpenAIProxyEngine("http://127.0.0.1:1")
        request = {"model": "model", "messages": [{"role": "user", "content": "same"}]}
        results: list[int | None] = []
        with patch("urllib.request.urlopen", side_effect=open_request):
            leader = threading.Thread(
                target=lambda: results.append(engine.estimate_prompt_tokens(request))
            )
            follower = threading.Thread(
                target=lambda: results.append(engine.estimate_prompt_tokens(request))
            )
            leader.start()
            self.assertTrue(entered.wait(1.0))
            follower.start()
            release.set()
            leader.join(timeout=1.0)
            follower.join(timeout=1.0)

        self.assertEqual(sorted(results), [31, 31])
        self.assertEqual(calls, 1)
        snapshot = engine.tokenizer_snapshot()
        self.assertEqual(snapshot["single_flight_active"], 0)
        self.assertEqual(snapshot["single_flight_leaders"], 1)
        self.assertEqual(snapshot["single_flight_followers"], 1)

    def test_service_combines_measured_prompt_and_output_context(self) -> None:
        class Engine:
            ready = True

            def estimate_prompt_tokens(self, request):
                return 3_000

            def tokenizer_snapshot(self):
                return {
                    "measured": 1,
                    "failures": 0,
                    "cache_capacity": 256,
                    "cache_entries": 1,
                    "cache_hits": 2,
                    "cache_misses": 1,
                    "cache_evictions": 0,
                    "cache_expirations": 0,
                    "single_flight_capacity": 64,
                    "single_flight_active": 0,
                    "single_flight_leaders": 1,
                    "single_flight_followers": 2,
                    "single_flight_bypasses": 0,
                    "single_flight_timeouts": 0,
                }

        telemetry = UnifiedMemoryTelemetry(32 * GIB, 28 * GIB, "test", MemoryPressure.NORMAL)
        service = RuntimeService(  # type: ignore[arg-type]
            engine=Engine(),
            memory_telemetry=telemetry,
        )
        service.apply_memory_pressure(MemoryPressure.CRITICAL)
        service.apply_memory_pressure(MemoryPressure.NORMAL)
        schedule = service.chat_schedule_request(
            {"messages": [], "max_completion_tokens": 2_000}
        )
        self.assertEqual(schedule.estimated_context_tokens, 5_000)
        with self.assertRaisesRegex(MemoryPressureAdmissionError, "recovery_context_limited"):
            service.admit_schedule(schedule)
        snapshot = service.token_estimation_snapshot()
        self.assertEqual(snapshot["source"], "backend_tokenizer")
        self.assertEqual(snapshot["last_prompt_tokens"], 3_000)
        self.assertEqual(snapshot["cache_capacity"], 256)
        self.assertEqual(snapshot["cache_hits"], 2)
        self.assertEqual(snapshot["single_flight_capacity"], 64)
        self.assertEqual(snapshot["single_flight_followers"], 2)

    def test_missing_tokenizer_falls_back_to_completion_only(self) -> None:
        service = RuntimeService()
        schedule = service.chat_schedule_request({"messages": [], "max_tokens": 12})
        self.assertEqual(schedule.estimated_context_tokens, 12)
        self.assertEqual(service.token_estimation_snapshot()["fallbacks"], 1)


if __name__ == "__main__":
    unittest.main()
