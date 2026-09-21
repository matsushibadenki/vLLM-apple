from __future__ import annotations

import json
import math
import os
import select
import socket
import stat
import struct
import threading
from pathlib import Path

from .qwen4_runtime_protocol import MAX_RUNTIME_MESSAGE_BYTES, Qwen4RuntimeCommandService

MAX_COMMANDS_PER_CONNECTION = 1024


class _SocketCancellationSignal:
    """Non-consuming disconnect probe evaluated only at backend safe points."""

    def __init__(self, connection: socket.socket) -> None:
        self.connection = connection

    def is_set(self) -> bool:
        try:
            readable, _, exceptional = select.select(
                [self.connection], [], [self.connection], 0)
            if exceptional:
                return True
            if not readable:
                return False
            flags = socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0)
            return self.connection.recv(1, flags) == b""
        except BlockingIOError:
            return False
        except OSError:
            return True


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
        self._connections: set[socket.socket] = set()
        self._busy_connections: set[socket.socket] = set()
        self._connections_lock = threading.Lock()

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
            listener.listen(8)
            listener.settimeout(0.25)
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
        if self._listener is None:
            raise RuntimeError("Qwen4 runtime Unix server is not started")
        workers: set[threading.Thread] = set()
        while not self.service.closed:
            workers = {worker for worker in workers if worker.is_alive()}
            try:
                connection, _ = self._listener.accept()
            except TimeoutError:
                continue

            def serve(accepted: socket.socket = connection) -> None:
                try:
                    self.serve_connection(accepted)
                finally:
                    accepted.close()

            worker = threading.Thread(target=serve, daemon=False)
            workers.add(worker)
            worker.start()
        for worker in workers:
            worker.join(timeout=self.connection_timeout_seconds)

    def serve_connection(self, connection: socket.socket) -> None:
        with self._connections_lock:
            self._connections.add(connection)
        try:
            if _peer_uid(connection) != os.getuid():
                raise PermissionError("Qwen4 runtime peer belongs to another user")
            connection.settimeout(self.connection_timeout_seconds)
            for _ in range(MAX_COMMANDS_PER_CONNECTION):
                if self.service.closed:
                    return
                try:
                    request = receive_qwen4_runtime_frame(connection)
                except (EOFError, OSError):
                    return
                with self._connections_lock:
                    self._busy_connections.add(connection)
                try:
                    response = self.service.handle(
                        request, cancellation=_SocketCancellationSignal(connection))
                    try:
                        send_qwen4_runtime_frame(connection, response)
                    except OSError:
                        return
                finally:
                    with self._connections_lock:
                        self._busy_connections.discard(connection)
                if request.get("operation") == "shutdown" and response.get("passed") is True:
                    self._interrupt_connections(excluding=connection)
                    return
            raise ValueError("Qwen4 runtime connection command limit reached")
        finally:
            with self._connections_lock:
                self._connections.discard(connection)
                self._busy_connections.discard(connection)

    def close(self) -> None:
        self._interrupt_connections()
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        self._unlink_owned_socket()

    def _interrupt_connections(self, *, excluding: socket.socket | None = None) -> None:
        with self._connections_lock:
            connections = tuple(
                connection for connection in self._connections
                if connection is not excluding and connection not in self._busy_connections
            )
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

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
