import os
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def request_over_uds(socket_path: Path, token: str) -> bytes:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(1)
    try:
        client.connect(str(socket_path))
        request = (
            "GET /health HTTP/1.1\r\n"
            "Host: localhost\r\n"
            f"Authorization: Bearer {token}\r\n"
            "Connection: close\r\n\r\n"
        )
        client.sendall(request.encode("ascii"))
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        client.close()


def wait_for_health(socket_path: Path, token_path: Path, timeout: float = 5) -> str:
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            token = token_path.read_text(encoding="utf-8").strip()
            if len(token) >= 32:
                response = request_over_uds(socket_path, token)
                if b"HTTP/1.1 200 OK" in response and b'"control_ready":true' in response:
                    return token
        except OSError as error:
            last_error = error
        time.sleep(0.02)
    raise AssertionError(f"daemon did not become healthy: {last_error}")


def launch_daemon(socket_path: Path, token_path: Path) -> subprocess.Popen[bytes]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm_apple.daemon",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--socket-path",
            str(socket_path),
            "--session-token-file",
            str(token_path),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


class DaemonLifecycleIntegrationTests(unittest.TestCase):
    def test_crash_then_relaunch_reuses_token_and_replaces_stale_socket(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vla-daemon-", dir="/tmp") as directory:
            root = Path(directory)
            socket_path = root / "runtime.sock"
            token_path = root / "session.token"
            first = launch_daemon(socket_path, token_path)
            second: subprocess.Popen[bytes] | None = None
            try:
                token = wait_for_health(socket_path, token_path)
                self.assertEqual(stat.S_IMODE(token_path.stat().st_mode), 0o600)
                self.assertIn(
                    b"HTTP/1.1 401 Unauthorized",
                    request_over_uds(socket_path, "x" * 32),
                )

                first.kill()
                self.assertLess(first.wait(timeout=3), 0)
                self.assertTrue(socket_path.exists())
                self.assertTrue(stat.S_ISSOCK(os.lstat(socket_path).st_mode))

                second = launch_daemon(socket_path, token_path)
                restarted_token = wait_for_health(socket_path, token_path)
                self.assertEqual(restarted_token, token)
                self.assertEqual(stat.S_IMODE(socket_path.stat().st_mode), 0o600)
                self.assertIn(b"HTTP/1.1 200 OK", request_over_uds(socket_path, token))

                second.terminate()
                self.assertEqual(second.wait(timeout=5), 0)
                self.assertFalse(socket_path.exists())
                self.assertEqual(stat.S_IMODE(token_path.stat().st_mode), 0o600)
                self.assertIsNotNone(first.poll())
                self.assertIsNotNone(second.poll())
            finally:
                stop_process(first)
                if second is not None:
                    stop_process(second)


if __name__ == "__main__":
    unittest.main()
