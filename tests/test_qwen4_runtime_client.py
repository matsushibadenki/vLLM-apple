import json
import tempfile
import threading
import unittest
from pathlib import Path

from tests import test_qwen4_runtime_protocol as protocol_tests
from vllm_apple.qwen4_runtime_client import Qwen4RuntimeClient
from vllm_apple.qwen4_runtime_protocol import Qwen4RuntimeCommandService
from vllm_apple.qwen4_runtime_transport import Qwen4RuntimeUnixServer


class Qwen4RuntimeClientTests(unittest.TestCase):
    def test_status_and_shutdown_bind_private_session_and_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            session = root / "session.json"
            session.write_text(json.dumps({"schema_version": 1, "session_id": "a" * 32}))
            session.chmod(0o600)
            service = Qwen4RuntimeCommandService("a" * 32, protocol_tests.FakeStore())
            server = Qwen4RuntimeUnixServer(root / "runtime.sock", service)
            server.start()
            thread = threading.Thread(target=server.serve_until_shutdown)
            thread.start()
            client = Qwen4RuntimeClient(server.socket_path, session)
            try:
                self.assertTrue(client.status(sequence=1)["passed"])
                self.assertTrue(client.shutdown(sequence=2)["passed"])
            finally:
                thread.join(timeout=2)
                server.close()
            self.assertFalse(thread.is_alive())

    def test_rejects_unsafe_session_before_connecting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "session.json"
            session.write_text('{"schema_version":1,"session_id":"' + "a" * 32 + '"}')
            session.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                Qwen4RuntimeClient(root / "missing.sock", session).status(sequence=1)


if __name__ == "__main__":
    unittest.main()
