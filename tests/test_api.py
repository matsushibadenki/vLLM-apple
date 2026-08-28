import json
import threading
import unittest
import urllib.error
import urllib.request

from vllm_apple.api import create_server
from vllm_apple.service import RuntimeService


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = create_server("127.0.0.1", 0, RuntimeService())
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def get_json(self, path: str) -> dict:
        with urllib.request.urlopen(self.base_url + path, timeout=2) as response:
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            return json.load(response)

    def test_health_and_versioned_hardware(self) -> None:
        health = self.get_json("/health")
        self.assertTrue(health["control_ready"])
        self.assertFalse(health["inference_ready"])
        hardware = self.get_json("/v1/hardware")
        self.assertEqual(hardware["api_version"], "v1")
        self.assertIn("total_bytes", hardware["hardware"]["memory"])
        runtime = self.get_json("/v1/runtime")
        self.assertIsNone(runtime["last_error"])

    def test_openai_models_shape(self) -> None:
        self.assertEqual(self.get_json("/v1/models"), {"object": "list", "data": []})

    def test_chat_reports_backend_unavailable(self) -> None:
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps({"model": "none", "messages": []}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=2)
        self.assertEqual(raised.exception.code, 503)
        payload = json.load(raised.exception)
        self.assertEqual(payload["error"]["code"], "backend_unavailable")


class AuthenticatedAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.token = "s" * 32
        cls.service = RuntimeService()
        cls.server = create_server("127.0.0.1", 0, cls.service, session_token=cls.token)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_missing_token_is_rejected_and_valid_token_is_accepted(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(self.base_url + "/health", timeout=2)
        self.assertEqual(raised.exception.code, 401)
        request = urllib.request.Request(
            self.base_url + "/health",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            self.assertTrue(json.load(response)["control_ready"])

    def test_authenticated_runtime_event_stream(self) -> None:
        request = urllib.request.Request(
            self.base_url + "/v1/events",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        response = urllib.request.urlopen(request, timeout=2)
        try:
            identifier = response.readline().decode()
            event_type = response.readline().decode()
            data = response.readline().decode()
            self.assertTrue(identifier.startswith("id: "))
            self.assertEqual(event_type.strip(), "event: runtime.state")
            payload = json.loads(data.removeprefix("data: "))
            self.assertEqual(payload["type"], "runtime.state")
        finally:
            response.close()
            self.service.events.publish("test.release", {})


if __name__ == "__main__":
    unittest.main()
