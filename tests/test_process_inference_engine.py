import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

from vllm_apple.api import create_server
from vllm_apple.inference_request import (
    InferenceEngineBusy,
    InferenceRequestCancelled,
    InferenceRequestContext,
)
from vllm_apple.process_inference_engine import MainThreadSubprocessInferenceEngine
from vllm_apple.service import RuntimeService


class ProcessInferenceEngineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.marker = root / "closed.txt"
        self.config = root / "config.json"
        self.config.write_text(json.dumps({"marker": str(self.marker)}))
        self.config.chmod(0o600)

    def tearDown(self):
        self.directory.cleanup()

    def engine(self, maximum_pending_requests=8):
        return MainThreadSubprocessInferenceEngine(
            "tests.process_engine_fixture:create_delegate",
            python_executable=Path(sys.executable),
            config_path=self.config,
            maximum_pending_requests=maximum_pending_requests,
            startup_timeout_seconds=5,
        )

    def context(self, seconds=5, cancellation=None):
        return InferenceRequestContext(
            "parent-request",
            time.monotonic() + seconds,
            cancellation or threading.Event(),
        )

    def test_child_owns_delegate_on_main_thread_and_preserves_order(self):
        engine = self.engine()
        try:
            first = engine.chat_completions_with_request_context(
                {"value": "a"}, None, self.context()
            )
            second = engine.chat_completions_with_request_context(
                {"value": "b"}, None, self.context()
            )
            self.assertNotEqual(first["pid"], os.getpid())
            self.assertTrue(first["main_thread"])
            self.assertEqual((first["sequence"], second["sequence"]), (1, 2))
            self.assertEqual(engine.models()[0]["id"], "process-fixture")
            diagnostics = engine.diagnostics()
            self.assertEqual(diagnostics["sequence"], 2)
            self.assertTrue(diagnostics["main_thread"])
        finally:
            self.assertTrue(engine.close())
        pid, main = self.marker.read_text().split(":")
        self.assertEqual(int(pid), first["pid"])
        self.assertEqual(main, "True")

    def test_parent_queue_saturation_is_bounded(self):
        engine = self.engine(maximum_pending_requests=1)
        outcome = []
        thread = threading.Thread(target=lambda: outcome.append(
            engine.chat_completions_with_request_context(
                {"delay": 0.3}, None, self.context()
            )
        ))
        thread.start()
        time.sleep(0.05)
        try:
            with self.assertRaises(InferenceEngineBusy):
                engine.chat_completions_with_request_context(
                    {"value": "rejected"}, None, self.context()
                )
            thread.join(timeout=2)
            self.assertEqual(len(outcome), 1)
        finally:
            engine.close()

    def test_active_cancellation_reaches_child_safe_point(self):
        engine = self.engine()
        cancellation = threading.Event()
        context = self.context(cancellation=cancellation)
        raised = []

        def invoke():
            try:
                engine.chat_completions_with_request_context(
                    {"delay": 2}, None, context
                )
            except BaseException as error:
                raised.append(error)

        thread = threading.Thread(target=invoke)
        thread.start()
        time.sleep(0.1)
        cancellation.set()
        thread.join(timeout=2)
        try:
            self.assertEqual(len(raised), 1)
            self.assertIsInstance(raised[0], InferenceRequestCancelled)
            recovered = engine.chat_completions_with_request_context(
                {"value": "after-cancel"}, None, self.context()
            )
            self.assertEqual(recovered["value"], "after-cancel")
        finally:
            engine.close()

    def test_real_http_path_reaches_child_main_thread(self):
        engine = self.engine()
        service = RuntimeService(engine=engine)
        server = create_server("127.0.0.1", 0, service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = {
                "model": "process-fixture",
                "messages": [{"role": "user", "content": "hello"}],
                "value": "through-http",
            }
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                result = json.load(response)
            self.assertTrue(result["main_thread"])
            self.assertEqual(result["value"], "through-http")
            self.assertNotEqual(result["pid"], os.getpid())
        finally:
            server.shutdown()
            server.server_close()
            service.close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
