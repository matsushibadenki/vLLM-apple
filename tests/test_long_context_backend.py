import json
import re
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests.schema_validator import validate_instance
from tests.test_schemas import load_schema
from vllm_apple.long_context import LongContextEvaluationError, LongContextEvaluator
from vllm_apple.backend_memory import BackendMemorySample
from vllm_apple.long_context_backend import MLXLongContextAdapter, VLLMLongContextAdapter
from vllm_apple.phase_probe import PhaseProbeConfig


def token_count(prompt: str) -> int:
    return 20 + prompt.count("The archival record contains ordinary neutral context.")


class TokenizerAndStreamHandler(BaseHTTPRequestHandler):
    tokenize_calls = 0
    stream_calls = 0

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        if self.path == "/tokenize":
            type(self).tokenize_calls += 1
            prompt = request["messages"][0]["content"]
            body = json.dumps(
                {"count": token_count(prompt), "max_model_len": 32768, "tokens": []}
            ).encode("utf-8")
            self._send("application/json", body)
            return
        if self.path == "/v1/chat/completions":
            type(self).stream_calls += 1
            prompt = request["messages"][0]["content"]
            expected = re.search(r"NEEDLE-\d+-A7C9", prompt).group(0)
            events = [
                {"choices": [{"delta": {"content": expected[:8]}}]},
                {"choices": [{"delta": {"content": expected[8:]}}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": token_count(prompt),
                        "completion_tokens": 2,
                    },
                },
            ]
            body = b"".join(
                b"data: " + json.dumps(event).encode("utf-8") + b"\n\n"
                for event in events
            ) + b"data: [DONE]\n\n"
            self._send("text/event-stream", body)
            return
        self.send_error(404)

    def _send(self, content_type: str, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class LongContextBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        TokenizerAndStreamHandler.tokenize_calls = 0
        TokenizerAndStreamHandler.stream_calls = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), TokenizerAndStreamHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_tokenizer_aligned_live_retrieval_stages(self) -> None:
        config = PhaseProbeConfig(
            base_url=f"http://127.0.0.1:{self.server.server_port}",
            model="test-model",
            hardware_fingerprint="test-hardware",
            samples=1,
            maximum_output_tokens=8,
        )
        adapter = VLLMLongContextAdapter(config, state_bytes_per_token=128)
        report = LongContextEvaluator(
            model_id="test-model",
            hardware_fingerprint="test-hardware",
            memory_ceiling_bytes=1024**3,
        ).evaluate((1024, 4096, 16384), adapter.measure)
        self.assertTrue(report["passed"])
        self.assertEqual(
            [stage["actual_prompt_tokens"] for stage in report["stages"]],
            [1024, 4096, 16384],
        )
        self.assertEqual(TokenizerAndStreamHandler.stream_calls, 3)
        self.assertLessEqual(TokenizerAndStreamHandler.tokenize_calls, 12)
        self.assertEqual(report["stages"][2]["state_bytes"], 16384 * 128)
        validate_instance(
            report, load_schema("runtime/long-context-evaluation-v1.schema.json")
        )

    def test_mlx_adapter_uses_observed_kv_bytes(self) -> None:
        class MemoryAdapter:
            def sample(self):
                return BackendMemorySample(None, None, kv_used_bytes=987654)

        config = PhaseProbeConfig(
            base_url=f"http://127.0.0.1:{self.server.server_port}",
            model="models/example",
            hardware_fingerprint="test-hardware",
            samples=1,
            maximum_output_tokens=8,
        )
        adapter = MLXLongContextAdapter(
            config,
            state_bytes_per_token=128,
            memory_adapter=MemoryAdapter(),
        )
        observation = adapter.measure(1024)
        self.assertEqual(observation.state_bytes, 987654)
        self.assertEqual(observation.actual_prompt_tokens, 1024)

    def test_model_context_limit_fails_before_backend_request(self) -> None:
        config = PhaseProbeConfig(
            base_url=f"http://127.0.0.1:{self.server.server_port}",
            model="test-model",
            hardware_fingerprint="test-hardware",
            maximum_output_tokens=32,
        )
        adapter = VLLMLongContextAdapter(
            config,
            state_bytes_per_token=128,
            model_max_context=8192,
        )
        with self.assertRaisesRegex(
            LongContextEvaluationError,
            "generation reserve exceeds the model context limit",
        ) as caught:
            adapter.measure(8192)
        self.assertEqual(caught.exception.code, "model_context_limit_exceeded")
        self.assertEqual(TokenizerAndStreamHandler.tokenize_calls, 0)
        self.assertEqual(TokenizerAndStreamHandler.stream_calls, 0)


if __name__ == "__main__":
    unittest.main()
