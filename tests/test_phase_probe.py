import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests.schema_validator import validate_instance
from tests.test_schemas import load_schema
from vllm_apple.phase_probe import PhaseProbeConfig, PhaseProbeError, run_phase_probe


class UsageStreamHandler(BaseHTTPRequestHandler):
    include_usage = True

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        if request.get("stream_options") != {"include_usage": True}:
            self.send_error(400)
            return
        events = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "hello"}}]},
        ]
        if self.include_usage:
            events.append(
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 2},
                }
            )
        body = b"".join(
            b"data: " + json.dumps(event).encode("utf-8") + b"\n\n" for event in events
        ) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MissingUsageStreamHandler(UsageStreamHandler):
    include_usage = False


class PhaseProbeTests(unittest.TestCase):
    def _server(self, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self._stop_server, server, thread)
        return server

    @staticmethod
    def _stop_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    def test_real_stream_timing_usage_and_rss_produce_valid_profile(self) -> None:
        server = self._server(UsageStreamHandler)
        result = run_phase_probe(
            PhaseProbeConfig(
                base_url=f"http://127.0.0.1:{server.server_port}",
                model="test-model",
                hardware_fingerprint="test-hardware",
                samples=2,
                target_pid=os.getpid(),
            )
        )
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["prefill"]["prompt_tokens"], 18)
        self.assertEqual(result["decode"]["output_tokens"], 4)
        self.assertGreater(result["peak_memory_bytes"], 0)
        validate_instance(
            result, load_schema("runtime/execution-phase-profile-v1.schema.json")
        )

    def test_missing_usage_is_not_replaced_by_an_estimate(self) -> None:
        server = self._server(MissingUsageStreamHandler)
        with self.assertRaises(PhaseProbeError) as raised:
            run_phase_probe(
                PhaseProbeConfig(
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    model="test-model",
                    hardware_fingerprint="test-hardware",
                    samples=1,
                )
            )
        self.assertEqual(raised.exception.code, "usage_missing")

    def test_remote_endpoint_requires_explicit_permission(self) -> None:
        with self.assertRaises(ValueError):
            PhaseProbeConfig(
                base_url="https://example.com",
                model="test-model",
                hardware_fingerprint="test-hardware",
            )


if __name__ == "__main__":
    unittest.main()
