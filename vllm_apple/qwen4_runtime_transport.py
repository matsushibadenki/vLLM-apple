from __future__ import annotations

import json
import math
import os
import socket
import stat
import struct
from pathlib import Path

from .qwen4_runtime_protocol import MAX_RUNTIME_MESSAGE_BYTES, Qwen4RuntimeCommandService


MAX_COMMANDS_PER_CONNECTION = 1024


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Qwen4 runtime frame contains a duplicate JSON key")
        result[key] = value
    return result


def _read_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise EOFError("Qwen4 runtime connection closed during a frame")
        chunks.extend(chunk)
    return bytes(chunks)


def receive_qwen4_runtime_frame(connection: socket.socket) -> object:
    length = struct.unpack("!I", _read_exact(connection, 4))[0]
    if not 1 <= length <= MAX_RUNTIME_MESSAGE_BYTES:
        raise ValueError("Qwen4 runtime frame length is outside the bounded limit")
    encoded = _read_exact(connection, length)
    try:
        return json.loads(encoded, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Qwen4 runtime frame is not valid JSON") from error


def send_qwen4_runtime_frame(connection: socket.socket, payload: object) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if not 1 <= len(encoded) <= MAX_RUNTIME_MESSAGE_BYTES:
        raise ValueError("Qwen4 runtime response frame is outside the bounded limit")
    connection.sendall(struct.pack("!I", len(encoded)) + encoded)


def _peer_uid(connection: socket.socket) -> int:
    getpeereid = getattr(connection, "getpeereid", None)
    if getpeereid is not None:
        uid, _ = getpeereid()
        return uid
    if hasattr(socket, "LOCAL_PEERCRED"):
        credentials = connection.getsockopt(0, socket.LOCAL_PEERCRED, 128)
        if len(credentials) < 8:
            raise RuntimeError("Qwen4 runtime Darwin peer credentials are truncated")
        return struct.unpack_from("=I", credentials, 4)[0]
    if hasattr(socket, "SO_PEERCRED"):
        credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        _, uid, _ = struct.unpack("3i", credentials)
        return uid
    raise RuntimeError("Qwen4 runtime peer credentials are unavailable")


class Qwen4RuntimeUnixServer:
    def __init__(
        self,
        socket_path: str | Path,
        service: Qwen4RuntimeCommandService,
        *,
        connection_timeout_seconds: float = 30.0,
    ) -> None:
        if (
            not math.isfinite(connection_timeout_seconds)
            or connection_timeout_seconds <= 0
            or connection_timeout_seconds > 300
        ):
            raise ValueError("Qwen4 runtime connection timeout is invalid")
        self.socket_path = Path(socket_path).expanduser().resolve(strict=False)
        self.service = service
        self.connection_timeout_seconds = connection_timeout_seconds
        self._listener: socket.socket | None = None
        self._socket_identity: tuple[int, int] | None = None

    def start(self) -> None:
        if self._listener is not None:
            raise RuntimeError("Qwen4 runtime Unix server is already started")
        parent = self.socket_path.parent
        parent_info = parent.lstat()
        if (
            not stat.S_ISDIR(parent_info.st_mode)
            or parent_info.st_uid != os.getuid()
            or stat.S_IMODE(parent_info.st_mode) & 0o077
            or self.socket_path.exists()
            or self.socket_path.is_symlink()
        ):
            raise ValueError("Qwen4 runtime socket path is unsafe")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            bound_info = self.socket_path.lstat()
            if not stat.S_ISSOCK(bound_info.st_mode) or bound_info.st_uid != os.getuid():
                raise ValueError("Qwen4 runtime bound socket identity is unsafe")
            self._socket_identity = (bound_info.st_dev, bound_info.st_ino)
            os.chmod(self.socket_path, 0o600)
            listener.listen(1)
        except BaseException:
            listener.close()
            self._unlink_owned_socket()
            raise
        self._listener = listener

    def serve_once(self) -> None:
        if self._listener is None:
            raise RuntimeError("Qwen4 runtime Unix server is not started")
        connection, _ = self._listener.accept()
        try:
            self.serve_connection(connection)
        finally:
            connection.close()

    def serve_until_shutdown(self) -> None:
        while not self.service.closed:
            self.serve_once()

    def serve_connection(self, connection: socket.socket) -> None:
        if _peer_uid(connection) != os.getuid():
            raise PermissionError("Qwen4 runtime peer belongs to another user")
        connection.settimeout(self.connection_timeout_seconds)
        for _ in range(MAX_COMMANDS_PER_CONNECTION):
            try:
                request = receive_qwen4_runtime_frame(connection)
            except EOFError:
                return
            response = self.service.handle(request)
            send_qwen4_runtime_frame(connection, response)
            if request.get("operation") == "shutdown" and response.get("passed") is True:
                return
        raise ValueError("Qwen4 runtime connection command limit reached")

    def close(self) -> None:
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        self._unlink_owned_socket()

    def _unlink_owned_socket(self) -> None:
        try:
            info = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if (
            self._socket_identity is not None
            and stat.S_ISSOCK(info.st_mode)
            and info.st_uid == os.getuid()
            and (info.st_dev, info.st_ino) == self._socket_identity
        ):
            self.socket_path.unlink()
            self._socket_identity = None

    def __enter__(self) -> Qwen4RuntimeUnixServer:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
