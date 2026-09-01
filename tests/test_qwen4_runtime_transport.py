import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import test_qwen4_runtime_protocol as protocol_tests
from vllm_apple.qwen4_runtime_protocol import Qwen4RuntimeCommandService
from vllm_apple.qwen4_runtime_transport import (
    Qwen4RuntimeUnixServer,
    receive_qwen4_runtime_frame,
    send_qwen4_runtime_frame,
)


class Qwen4RuntimeTransportTests(unittest.TestCase):
    def service(self):
        return Qwen4RuntimeCommandService("a" * 32, protocol_tests.FakeStore())

    def test_serves_ordered_frames_over_current_user_socket(self) -> None:
        server_socket, client_socket = socket.socketpair()
        server = Qwen4RuntimeUnixServer("/tmp/not-bound.sock", self.service())
        thread = threading.Thread(target=server.serve_connection, args=(server_socket,))
        thread.start()
        try:
            send_qwen4_runtime_frame(
                client_socket,
                {
                    "abi_version": 1,
                    "session_id": "a" * 32,
                    "sequence": 1,
                    "request_id": "1" * 32,
                    "operation": "status",
                },
            )
            response = receive_qwen4_runtime_frame(client_socket)
            self.assertTrue(response["passed"])
            send_qwen4_runtime_frame(
                client_socket,
                {
                    "abi_version": 1,
                    "session_id": "a" * 32,
                    "sequence": 2,
                    "request_id": "2" * 32,
                    "operation": "shutdown",
                },
            )
            self.assertTrue(receive_qwen4_runtime_frame(client_socket)["result"]["shutdown"])
        finally:
            client_socket.close()
            thread.join(timeout=2)
            server_socket.close()
        self.assertFalse(thread.is_alive())

    def test_binds_configured_path_and_applies_private_socket_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            path = root / "runtime.sock"
            server = Qwen4RuntimeUnixServer(path, self.service())
            calls = []

            class Listener:
                def bind(self, value):
                    calls.append(("bind", value))
                    path.touch()

                def listen(self, value):
                    calls.append(("listen", value))

                def close(self):
                    calls.append(("close", None))

            with (
                patch("vllm_apple.qwen4_runtime_transport.socket.socket", return_value=Listener()),
                patch("vllm_apple.qwen4_runtime_transport.os.chmod") as chmod,
                patch("vllm_apple.qwen4_runtime_transport.stat.S_ISSOCK", return_value=True),
            ):
                server.start()
                chmod.assert_called_once_with(server.socket_path, 0o600)
                server.close()
            self.assertIn(("bind", str(server.socket_path)), calls)
            self.assertIn(("listen", 1), calls)

    def test_rejects_oversized_frame_before_json_allocation(self) -> None:
        server_socket, client_socket = socket.socketpair()
        try:
            client_socket.sendall((16 * 1024 + 1).to_bytes(4, "big"))
            with self.assertRaisesRegex(ValueError, "bounded"):
                receive_qwen4_runtime_frame(server_socket)
        finally:
            client_socket.close()
            server_socket.close()


if __name__ == "__main__":
    unittest.main()
