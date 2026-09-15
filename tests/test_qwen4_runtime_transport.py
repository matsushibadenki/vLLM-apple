import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import test_qwen4_runtime_protocol as protocol_tests
from vllm_apple.qwen4_runtime_protocol import (
    Qwen4RuntimeCommandService,
    build_qwen4_numeric_streaming_runtime_request,
)
from vllm_apple.qwen4_runtime_transport import (
    Qwen4RuntimeUnixServer,
    _SocketCancellationSignal,
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

                def settimeout(self, value):
                    calls.append(("settimeout", value))

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
            self.assertIn(("listen", 8), calls)

    def test_rejects_oversized_frame_before_json_allocation(self) -> None:
        server_socket, client_socket = socket.socketpair()
        try:
            client_socket.sendall((16 * 1024 + 1).to_bytes(4, "big"))
            with self.assertRaisesRegex(ValueError, "bounded"):
                receive_qwen4_runtime_frame(server_socket)
        finally:
            client_socket.close()
            server_socket.close()

    def test_socket_cancellation_signal_detects_disconnect_without_consuming_data(self) -> None:
        server_socket, client_socket = socket.socketpair()
        try:
            signal = _SocketCancellationSignal(server_socket)
            self.assertFalse(signal.is_set())
            client_socket.sendall(b"next")
            self.assertFalse(signal.is_set())
            self.assertEqual(server_socket.recv(4), b"next")
            client_socket.close()
            self.assertTrue(signal.is_set())
        finally:
            server_socket.close()

    def test_cancel_consume_shutdown_socket_race_is_bounded(self) -> None:
        _, _, reader = protocol_tests.Qwen4RuntimeProtocolTests.numeric_fixture()
        store = protocol_tests.FakeStore()
        started = threading.Event()

        def blocked_load(*args, cancellation=None, **kwargs):
            started.set()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not cancellation.is_set():
                time.sleep(0.001)
            if not cancellation.is_set():
                raise RuntimeError("test cancellation deadline exceeded")
            raise ValueError("cancelled")

        store.load_scaled_int8_streaming = blocked_load
        service = Qwen4RuntimeCommandService("a" * 32, store, reader)
        server = Qwen4RuntimeUnixServer("/tmp/not-bound.sock", service)
        pairs = [socket.socketpair() for _ in range(3)]
        workers = [
            threading.Thread(target=server.serve_connection, args=(server_socket,))
            for server_socket, _ in pairs
        ]
        for worker in workers:
            worker.start()
        try:
            load = build_qwen4_numeric_streaming_runtime_request(
                session_id="a" * 32, sequence=1, request_id="1" * 32,
                artifact_name="weight.json", artifact_digest="b" * 64,
                target_dtype="F16", tile_bytes=1, buffer_count=1,
            )
            send_qwen4_runtime_frame(pairs[0][1], load)
            self.assertTrue(started.wait(timeout=1))
            send_qwen4_runtime_frame(pairs[1][1], {
                "abi_version": 1, "session_id": "a" * 32, "sequence": 99,
                "request_id": "2" * 32, "operation": "cancel",
                "target_request_id": "1" * 32,
            })
            self.assertTrue(receive_qwen4_runtime_frame(pairs[1][1])["result"]["cancelled"])
            send_qwen4_runtime_frame(pairs[2][1], {
                "abi_version": 1, "session_id": "a" * 32, "sequence": 2,
                "request_id": "3" * 32, "operation": "shutdown",
            })
            self.assertFalse(receive_qwen4_runtime_frame(pairs[0][1])["passed"])
            self.assertTrue(
                receive_qwen4_runtime_frame(pairs[2][1])["result"]["shutdown"]
            )
        finally:
            for server_socket, client_socket in pairs:
                client_socket.close()
                server_socket.close()
            for worker in workers:
                worker.join(timeout=2)
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        diagnostics = service.numeric_diagnostics_snapshot()
        self.assertEqual(diagnostics["active_requests"], 0)
        self.assertEqual(diagnostics["cancel_hits"], 1)
        self.assertEqual(diagnostics["artifacts_consumed"], 0)


if __name__ == "__main__":
    unittest.main()
