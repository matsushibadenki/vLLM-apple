import json
import os
import socket
import sys
import tempfile
import textwrap
import threading
import unittest
import urllib.request
from unittest.mock import Mock
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vllm_apple.api import create_server
from vllm_apple.backend import (
    VLLM_APPLE_TUNING_MIDDLEWARE,
    BackendConfig,
    BackendConfigurationError,
    BackendProcess,
    BackendStartupError,
    OpenAIProxyEngine,
    supports_kernel_tuning_middleware,
)
from vllm_apple.compat import inspect_backend
from vllm_apple.cli import build_parser, main
from vllm_apple.kernel_context import (
    KERNEL_TUNING_ACCEPTED_HEADER,
    KERNEL_TUNING_CONTEXT_HEADER,
    KERNEL_TUNING_ID_HEADER,
    InferenceKernelContext,
    PagedAttentionKernelSelection,
)
from vllm_apple.kernel_profile import PagedAttentionShape
from vllm_apple.metal_probe import MetalThreadConfiguration
from vllm_apple.service import RuntimeService


class FakeVLLMHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    last_tuning_headers: dict[str, str | None] = {}

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        tuning_id = self.headers.get(KERNEL_TUNING_ID_HEADER)
        if tuning_id is not None:
            self.send_header(KERNEL_TUNING_ACCEPTED_HEADER, tuning_id)
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
        type(self).last_tuning_headers = {
            "id": self.headers.get(KERNEL_TUNING_ID_HEADER),
            "context": self.headers.get(KERNEL_TUNING_CONTEXT_HEADER),
        }
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
    def test_top_level_serve_forwards_mlx_backend_choice(self) -> None:
        arguments = build_parser().parse_args([
            "serve", "models/gemma", "--backend-kind", "mlx_lm",
            "--backend-executable", "/path/to/mlx_lm.server",
        ])
        self.assertEqual(arguments.backend_kind, "mlx_lm")
        with patch("vllm_apple.cli.serve") as serve:
            self.assertEqual(main([
                "serve", "models/gemma", "--backend-kind", "mlx_lm",
                "--backend-executable", "/path/to/mlx_lm.server",
            ]), 0)
        self.assertEqual(serve.call_args.kwargs["backend_kind"], "mlx_lm")

    def test_managed_mlx_models_advertise_loaded_alias_not_unrelated_cache(self) -> None:
        process = Mock()
        process.config.backend_kind = "mlx_lm"
        process.ready = True
        engine = OpenAIProxyEngine("http://127.0.0.1:1", process)
        self.assertEqual(engine.models(), [{"id": "default_model", "object": "model"}])
        process.ready = False
        with self.assertRaisesRegex(Exception, "managed backend is not ready"):
            engine.models()

    def test_factory_preserves_explicit_mlx_backend_kind(self) -> None:
        from vllm_apple.backend import make_backend_config

        config = make_backend_config(
            "/models/gemma",
            "/bin/echo",
            8123,
            None,
            30,
            backend_kind="mlx_lm",
        )
        self.assertEqual(config.backend_kind, "mlx_lm")
        self.assertEqual(config.command()[1:3], ["--model", "/models/gemma"])

    def test_mlx_server_command_uses_its_native_cli_contract(self) -> None:
        config = BackendConfig(
            model="/models/gemma",
            executable=Path("/tmp/mlx_lm.server"),
            port=8123,
            max_model_len=8192,
            backend_kind="mlx_lm",
        )
        self.assertEqual(
            config.command(),
            [
                "/tmp/mlx_lm.server",
                "--model",
                "/models/gemma",
                "--host",
                "127.0.0.1",
                "--port",
                "8123",
                "--log-level",
                "WARNING",
            ],
        )

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

    def test_tuning_enabled_command_registers_in_process_asgi_middleware(self) -> None:
        config = BackendConfig(
            "model",
            Path("/tmp/vllm"),
            enable_kernel_tuning_middleware=True,
        )
        command = config.command()
        self.assertIn("--disable-frontend-multiprocessing", command)
        middleware_index = command.index("--middleware")
        self.assertEqual(command[middleware_index + 1], VLLM_APPLE_TUNING_MIDDLEWARE)
        with self.assertRaises(BackendConfigurationError):
            BackendConfig(
                "model",
                Path("/tmp/vllm"),
                extra_arguments=("--middleware", "untrusted.middleware"),
            )

    def test_middleware_capability_probe_requires_both_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            supported = Path(directory) / "supported"
            supported.write_text(
                f"#!{sys.executable}\n"
                "print('--middleware VALUE --disable-frontend-multiprocessing')\n"
            )
            unsupported = Path(directory) / "unsupported"
            unsupported.write_text(f"#!{sys.executable}\nprint('--middleware VALUE')\n")
            os.chmod(supported, 0o700)
            os.chmod(unsupported, 0o700)
            self.assertTrue(supports_kernel_tuning_middleware(supported))
            self.assertFalse(supports_kernel_tuning_middleware(unsupported))

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
                process.restart()
                self.assertTrue(process.ready)
                self.assertIsNotNone(process.pid)
            finally:
                process.stop()
            self.assertFalse(process.running)

    def test_managed_process_allows_port_after_server_active_close(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            address = listener.getsockname()
            with socket.create_connection(address, timeout=2) as client:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(2)
                    # The server closes first, leaving its port in TIME_WAIT.
                    connection.shutdown(socket.SHUT_WR)
                    self.assertEqual(client.recv(1), b"")
                    client.shutdown(socket.SHUT_WR)
                    self.assertEqual(connection.recv(1), b"")
        process = BackendProcess(
            BackendConfig("test-model", Path("/bin/false"), port=address[1])
        )
        with patch("vllm_apple.backend.subprocess.Popen", side_effect=OSError("spawn sentinel")) as spawn:
            with self.assertRaisesRegex(BackendStartupError, "unable to start backend"):
                process.start()
            spawn.assert_called_once()
        self.assertFalse(process.running)

    def test_managed_process_rejects_occupied_backend_port_before_spawn(self) -> None:
        listener = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLMHandler)
        process = BackendProcess(
            BackendConfig("test-model", Path("/bin/false"), port=listener.server_port)
        )
        try:
            with self.assertRaises(BackendStartupError) as raised:
                process.start()
            self.assertEqual(raised.exception.code, "backend_port_in_use")
            self.assertFalse(process.running)
        finally:
            listener.server_close()


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

    def test_tuning_context_is_forwarded_without_mutating_openai_body(self) -> None:
        shape = PagedAttentionShape(1024, 1, 32, 8, 128, 16, 64, 4194304)
        context = InferenceKernelContext(
            "a" * 24,
            "b" * 24,
            (PagedAttentionKernelSelection(shape, MetalThreadConfiguration(128, 256, 64)),),
        )
        payload = self.engine.chat_completions_with_context(
            {"model": "test-model", "messages": []}, context
        )
        self.assertEqual(payload["choices"][0]["message"]["content"], "hello")
        self.assertEqual(FakeVLLMHandler.last_tuning_headers["id"], "a" * 24)
        forwarded = json.loads(FakeVLLMHandler.last_tuning_headers["context"] or "{}")
        self.assertEqual(forwarded, context.to_dict())
        self.assertLessEqual(len(FakeVLLMHandler.last_tuning_headers["context"] or ""), 4096)
        self.assertEqual(self.engine.tuning_ack_snapshot()["acknowledged"], 1)


if __name__ == "__main__":
    unittest.main()
