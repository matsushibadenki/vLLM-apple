import os
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vllm_apple.api import create_server
from vllm_apple.service import RuntimeService
from vllm_apple.soak import BoundedMetrics, SoakConfig, run_soak


class SoakRunnerTests(unittest.TestCase):
    def test_metrics_storage_stays_bounded(self) -> None:
        metrics = BoundedMetrics()
        for index in range(100_000):
            metrics.record(index % 12_000, f"error-{index}" if index % 10 == 0 else None)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requests"], 100_000)
        self.assertEqual(snapshot["storage"]["latency_buckets"], 12)
        self.assertLessEqual(snapshot["storage"]["error_keys"], 17)

    def test_health_soak_reports_throughput_and_rss(self) -> None:
        server = create_server("127.0.0.1", 0, RuntimeService())
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            result = run_soak(
                SoakConfig(
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    duration_seconds=0.2,
                    warmup_seconds=0,
                    concurrency=2,
                    request_timeout_seconds=2,
                    target_pid=os.getpid(),
                    max_rss_growth_bytes=64 * 1024 * 1024,
                )
            )
            self.assertTrue(result["passed"])
            self.assertGreater(result["requests"], 0)
            self.assertGreater(result["requests_per_second"], 0)
            self.assertIsNotNone(result["rss"]["baseline_bytes"])
            self.assertIsNotNone(result["rss"]["peak_growth_bytes"])
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

    def test_remote_target_requires_explicit_opt_in(self) -> None:
        with self.assertRaises(ValueError):
            SoakConfig(base_url="https://example.com")

    def test_mixed_chat_validates_non_streaming_and_streaming_responses(self) -> None:
        counts = {"plain": 0, "stream": 0}
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                streaming = request.get("stream") is True
                with lock:
                    counts["stream" if streaming else "plain"] += 1
                if streaming:
                    body = (
                        'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                        "data: [DONE]\n\n"
                    ).encode()
                    content_type = "text/event-stream"
                else:
                    body = b'{"choices":[{"message":{"content":"ok"}}]}'
                    content_type = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = run_soak(
                SoakConfig(
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    duration_seconds=0.2,
                    warmup_seconds=0,
                    concurrency=2,
                    request_timeout_seconds=2,
                    mode="chat-mixed",
                    model="test-model",
                )
            )
            self.assertTrue(result["passed"])
            self.assertGreater(counts["plain"], 0)
            self.assertGreater(counts["stream"], 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_certification_requires_full_window_pid_and_rss_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1800"):
            SoakConfig(duration_seconds=60, require_30_minute_window=True)
        with self.assertRaisesRegex(ValueError, "requires PID"):
            SoakConfig(
                duration_seconds=1800,
                require_30_minute_window=True,
            )


if __name__ == "__main__":
    unittest.main()
