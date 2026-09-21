import io
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from contextlib import nullcontext

from vllm_apple.api import create_server
from vllm_apple.inference_request import (
    InferenceRequestCancelled,
    InferenceRequestContext,
)
from vllm_apple.service import RuntimeService


class Signal:
    def __init__(self, value=False):
        self.value = value

    def is_set(self):
        return self.value


class ManagedEngine:
    ready = True

    def __init__(self):
        self.context = None
        self.close_count = 0

    def models(self):
        return []

    def chat_completions_with_request_context(self, request, kernel, context):
        context.raise_if_cancelled()
        self.context = context
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    def open_chat_stream(self, request):
        return nullcontext(io.BytesIO(b"data: [DONE]\n\n"))

    def close(self):
        self.close_count += 1
        return True


class ExpiringEngine(ManagedEngine):
    def chat_completions_with_request_context(self, request, kernel, context):
        time.sleep(0.03)
        context.raise_if_cancelled()


class InferenceRequestTests(unittest.TestCase):
    def test_service_closes_managed_engine_exactly_once(self):
        engine = ManagedEngine()
        service = RuntimeService(engine=engine)
        self.assertTrue(service.close())
        self.assertTrue(service.close())
        self.assertEqual(engine.close_count, 1)

    def test_context_distinguishes_live_cancelled_and_expired(self):
        signal = Signal()
        context = InferenceRequestContext("request-123", time.monotonic() + 1, signal)
        context.raise_if_cancelled()
        self.assertGreater(context.remaining_seconds, 0)
        signal.value = True
        with self.assertRaisesRegex(InferenceRequestCancelled, "disconnected"):
            context.raise_if_cancelled()

        expired = InferenceRequestContext("request-456", time.monotonic() - 1, Signal())
        with self.assertRaisesRegex(InferenceRequestCancelled, "timed out"):
            expired.raise_if_cancelled()

    def test_http_chat_passes_bounded_request_context_to_engine(self):
        engine = ManagedEngine()
        server = create_server("127.0.0.1", 0, RuntimeService(engine=engine))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                data=json.dumps({"model": "fixture", "messages": []}).encode(),
                headers={"Content-Type": "application/json", "X-Request-ID": "managed-12345678"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.load(response)
            self.assertEqual(payload["choices"][0]["message"]["content"], "ok")
            self.assertEqual(engine.context.request_id, "managed-12345678")
            self.assertGreater(engine.context.remaining_seconds, 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_http_chat_timeout_releases_request_slot(self):
        server = create_server(
            "127.0.0.1", 0, RuntimeService(engine=ExpiringEngine()),
            socket_timeout=0.01,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                data=json.dumps({"model": "fixture", "messages": []}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request, timeout=2)
            self.assertEqual(raised.exception.code, 408)
            self.assertEqual(json.load(raised.exception)["error"]["code"], "request_cancelled")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(server.request_metrics()["active_requests"], 0)


if __name__ == "__main__":
    unittest.main()
