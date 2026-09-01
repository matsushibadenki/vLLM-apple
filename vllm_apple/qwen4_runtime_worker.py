from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from .qwen4_component_loader import Qwen4MemoryAdmission
from .qwen4_resident_store import Qwen4ResidentBackend, Qwen4ResidentStore
from .qwen4_runtime_protocol import (
    Qwen4RuntimeCommandService,
    create_qwen4_runtime_session_id,
)
from .qwen4_runtime_transport import Qwen4RuntimeUnixServer
from .qwen4_tensor_reader import Qwen4TensorReader


class Qwen4RuntimeWorker:
    def __init__(
        self,
        *,
        stage_root: str | Path,
        socket_path: str | Path,
        session_file: str | Path,
        maximum_artifact_bytes: int,
        memory_capacity_bytes: int,
        backend: Qwen4ResidentBackend,
        requested_modes: tuple[str, ...] = ("text",),
        component_limits: dict[str, int] | None = None,
    ) -> None:
        self.session_id = create_qwen4_runtime_session_id()
        self.session_file = Path(session_file).expanduser().resolve(strict=False)
        self._session_identity: tuple[int, int] | None = None
        self._session_descriptor: int | None = None
        reader = Qwen4TensorReader(
            stage_root,
            maximum_artifact_bytes=maximum_artifact_bytes,
            requested_modes=requested_modes,
        )
        admission = Qwen4MemoryAdmission(
            memory_capacity_bytes,
            component_limits=component_limits,
        )
        self.store = Qwen4ResidentStore(reader, admission, backend)
        self.service = Qwen4RuntimeCommandService(self.session_id, self.store)
        self.server = Qwen4RuntimeUnixServer(socket_path, self.service)

    def start(self) -> None:
        self._write_session_file()
        try:
            self.server.start()
        except BaseException:
            self._remove_session_file()
            raise

    def serve_until_shutdown(self) -> None:
        self.server.serve_until_shutdown()

    def close(self) -> dict[str, object]:
        self.server.close()
        snapshot = self.store.shutdown()
        self._remove_session_file()
        return snapshot

    def _write_session_file(self) -> None:
        parent = self.session_file.parent
        parent_info = parent.lstat()
        if (
            not stat.S_ISDIR(parent_info.st_mode)
            or parent_info.st_uid != os.getuid()
            or stat.S_IMODE(parent_info.st_mode) & 0o077
            or self.session_file.exists()
            or self.session_file.is_symlink()
        ):
            raise ValueError("Qwen4 runtime session credential path is unsafe")
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.session_file.name}.", dir=parent)
        try:
            os.fchmod(descriptor, 0o600)
            temporary_info = os.fstat(descriptor)
            self._session_identity = (temporary_info.st_dev, temporary_info.st_ino)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {"schema_version": 1, "session_id": self.session_id},
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, self.session_file, follow_symlinks=False)
            os.unlink(temporary)
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            session_descriptor = os.open(self.session_file, flags)
            info = os.fstat(session_descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077
                or (info.st_dev, info.st_ino) != self._session_identity
            ):
                os.close(session_descriptor)
                raise ValueError("Qwen4 runtime session credential identity changed during publish")
            self._session_descriptor = session_descriptor
            # Keep the published inode open until cleanup. Linux may otherwise
            # immediately reuse its inode number for a replacement path.
            directory = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            self._remove_session_file()
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def _remove_session_file(self) -> None:
        try:
            info = self.session_file.lstat()
        except FileNotFoundError:
            info = None
        try:
            descriptor_info = (
                os.fstat(self._session_descriptor)
                if self._session_descriptor is not None
                else None
            )
            identity = (
                (descriptor_info.st_dev, descriptor_info.st_ino)
                if descriptor_info is not None
                else self._session_identity
            )
            if (
                info is not None
                and identity is not None
                and stat.S_ISREG(info.st_mode)
                and info.st_uid == os.getuid()
                and (info.st_dev, info.st_ino) == identity
            ):
                self.session_file.unlink()
        finally:
            if self._session_descriptor is not None:
                os.close(self._session_descriptor)
                self._session_descriptor = None
            self._session_identity = None

    def __enter__(self) -> Qwen4RuntimeWorker:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
