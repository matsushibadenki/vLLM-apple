import json
import os
import sys
import tempfile
import textwrap
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vllm_apple.api import create_server
from vllm_apple.backend import (
    BackendConfig,
    BackendConfigurationError,
    BackendProcess,
    OpenAIProxyEngine,
)
from vllm_apple.compat import inspect_backend
from vllm_apple.service import RuntimeService


class FakeVLLMHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._send({"object": "list", "data": [{"id": "test-model", "object": "model"}]})
        elif self.path == "/health":
            self._send({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        if request.get("stream"):
            body = (
                b'data: {"id":"chunk-1","choices":[{"delta":{"content":"hello"}}]}\n\n'
                b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        else:
            self._send(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )


class BackendConfigTests(unittest.TestCase):
    def test_command_is_explicit_and_managed_options_cannot_be_overridden(self) -> None:
        config = BackendConfig(
            model="mlx-community/test",
            executable=Path("/tmp/vllm"),
            port=9001,
            max_model_len=8192,
        )
        self.assertEqual(
            config.command(),
            [
                "/tmp/vllm",
                "serve",
                "mlx-community/test",
                "--host",
                "127.0.0.1",
                "--port",
                "9001",
                "--max-model-len",
                "8192",
            ],
        )
        with self.assertRaises(BackendConfigurationError):
            BackendConfig("model", Path("/tmp/vllm"), host="0.0.0.0")
        with self.assertRaises(BackendConfigurationError):
            BackendConfig("model", Path("/tmp/vllm"), extra_arguments=("--port", "9999"))

    def test_doctor_reports_missing_executable_without_raising(self) -> None:
        report = inspect_backend("/definitely/missing/vllm")
        self.assertFalse(report.compatible)
        self.assertIn("vllm_executable_not_found", report.issues)

    def test_managed_process_reaches_readiness_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "fake-vllm"
            executable.write_text(
                f"#!{sys.executable}\n"
                + textwrap.dedent(
                    """
                    import argparse
                    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

                    parser = argparse.ArgumentParser()
                    parser.add_argument("command")
                    parser.add_argument("model")
                    parser.add_argument("--host")
                    parser.add_argument("--port", type=int)
                    arguments = parser.parse_args()

                    class Handler(BaseHTTPRequestHandler):
                        def log_message(self, format, *args):
                            return
                        def do_GET(self):
                            body = b'{"status":"ok"}'
                            self.send_response(200)
                            self.send_header("Content-Length", str(len(body)))
                            self.end_headers()
                            self.wfile.write(body)

                    print("fake backend ready", flush=True)
                    ThreadingHTTPServer((arguments.host, arguments.port), Handler).serve_forever()
                    """
                ),
                encoding="utf-8",
            )
            os.chmod(executable, 0o700)
            probe = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLMHandler)
            port = probe.server_port
            probe.server_close()
            process = BackendProcess(
                BackendConfig("test-model", executable, port=port, startup_timeout=5)
            )
            try:
                process.start()
                self.assertTrue(process.ready)
                self.assertTrue(any("fake backend ready" in line for line in process.recent_logs()))
            finally:
                process.stop()
            self.assertFalse(process.running)


class ProxyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLMHandler)
        cls.upstream_thread = threading.Thread(target=cls.upstream.serve_forever, daemon=True)
        cls.upstream_thread.start()
        upstream_url = f"http://127.0.0.1:{cls.upstream.server_port}"
        cls.engine = OpenAIProxyEngine(upstream_url)
        cls.control = create_server("127.0.0.1", 0, RuntimeService(cls.engine))
        cls.control_thread = threading.Thread(target=cls.control.serve_forever, daemon=True)
        cls.control_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.control.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.control.shutdown()
        cls.control.server_close()
        cls.control_thread.join(timeout=2)
        cls.upstream.shutdown()
        cls.upstream.server_close()
        cls.upstream_thread.join(timeout=2)

    def test_models_and_non_streaming_chat_are_proxied(self) -> None:
        with urllib.request.urlopen(self.base_url + "/v1/models", timeout=2) as response:
            models = json.load(response)
        self.assertEqual(models["data"][0]["id"], "test-model")

        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps({"model": "test-model", "messages": []}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.load(response)
        self.assertEqual(payload["choices"][0]["message"]["content"], "hello")

    def test_sse_stream_is_relayed_without_json_buffering(self) -> None:
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps({"model": "test-model", "messages": [], "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            self.assertEqual(response.headers.get_content_type(), "text/event-stream")
            body = response.read().decode()
        self.assertIn('"content":"hello"', body)
        self.assertIn("data: [DONE]", body)


if __name__ == "__main__":
    unittest.main()
