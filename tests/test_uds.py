import os
import socket
import stat
import tempfile
import threading
import unittest
from pathlib import Path

from vllm_apple.api import create_unix_server
from vllm_apple.service import RuntimeService


class UnixDomainSocketTests(unittest.TestCase):
    def test_private_socket_serves_authenticated_control_api_and_is_removed(self) -> None:
        token = "t" * 32
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sock"
            server = create_unix_server(str(path), RuntimeService(), session_token=token)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                self.assertTrue(stat.S_ISSOCK(os.lstat(path).st_mode))
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.settimeout(2)
                client.connect(str(path))
                request = (
                    "GET /health HTTP/1.1\r\n"
                    "Host: localhost\r\n"
                    f"Authorization: Bearer {token}\r\n"
                    "Connection: close\r\n\r\n"
                )
                client.sendall(request.encode("ascii"))
                chunks = []
                while True:
                    chunk = client.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                client.close()
                response = b"".join(chunks)
                self.assertIn(b"HTTP/1.1 200 OK", response)
                self.assertIn(b'"control_ready":true', response)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
