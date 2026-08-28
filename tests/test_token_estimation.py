import json
import unittest
from unittest.mock import patch

from vllm_apple.backend import OpenAIProxyEngine
from vllm_apple.memory_admission import MemoryPressureAdmissionError
from vllm_apple.service import RuntimeService
from vllm_apple.types import MemoryPressure


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

    def test_service_combines_measured_prompt_and_output_context(self) -> None:
        class Engine:
            ready = True

            def estimate_prompt_tokens(self, request):
                return 3_000

            def tokenizer_snapshot(self):
                return {"measured": 1, "failures": 0}

        service = RuntimeService(engine=Engine())  # type: ignore[arg-type]
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

    def test_missing_tokenizer_falls_back_to_completion_only(self) -> None:
        service = RuntimeService()
        schedule = service.chat_schedule_request({"messages": [], "max_tokens": 12})
        self.assertEqual(schedule.estimated_context_tokens, 12)
        self.assertEqual(service.token_estimation_snapshot()["fallbacks"], 1)


if __name__ == "__main__":
    unittest.main()
