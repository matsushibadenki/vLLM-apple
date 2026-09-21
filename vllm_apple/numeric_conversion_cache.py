"""Bounded, signed and single-flight cache for numeric conversions."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import shutil
import stat
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_CONVERSION_CACHE_ENTRIES = 64
MAX_CONVERSION_CACHE_BYTES = 64 * 1024**3
MAX_CONVERSION_OUTPUT_BYTES = 32 * 1024**3


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True)
class NumericConversionCacheIdentity:
    source_sha256: str
    scale_sha256: str
    layout_sha256: str
    kernel_sha256: str
    environment_sha256: str

    def __post_init__(self) -> None:
        if any(not _digest(value) for value in asdict(self).values()):
            raise ValueError("invalid numeric conversion cache identity")

    @property
    def cache_id(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True, slots=True)
class NumericConversionCacheEntry:
    cache_id: str
    output_path: Path
    output_bytes: int
    output_sha256: str


class NumericConversionCache:
    def __init__(
        self,
        root: Path,
        *,
        signing_key: bytes,
        maximum_entries: int = MAX_CONVERSION_CACHE_ENTRIES,
        maximum_bytes: int = MAX_CONVERSION_CACHE_BYTES,
    ) -> None:
        if not isinstance(signing_key, bytes) or len(signing_key) < 32:
            raise ValueError("numeric conversion cache key must contain at least 32 bytes")
        if (not 1 <= maximum_entries <= MAX_CONVERSION_CACHE_ENTRIES
                or not 1 <= maximum_bytes <= MAX_CONVERSION_CACHE_BYTES):
            raise ValueError("invalid numeric conversion cache bounds")
        self.root = root.expanduser().resolve(strict=False)
        self._key = signing_key
        self._maximum_entries = maximum_entries
        self._maximum_bytes = maximum_bytes
        self._lock = threading.RLock()
        self._flights: dict[str, threading.Event] = {}
        for path in (self.root, self.root / "entries", self.root / "quarantine"):
            _private_directory(path)

    def get_or_create(
        self,
        identity: NumericConversionCacheIdentity,
        converter: Callable[[Path], None],
    ) -> NumericConversionCacheEntry:
        if not callable(converter):
            raise ValueError("numeric conversion cache converter is invalid")
        cache_id = identity.cache_id
        while True:
            with self._lock:
                entry_path = self.root / "entries" / cache_id
                if entry_path.exists():
                    return self.load(identity)
                event = self._flights.get(cache_id)
                if event is None:
                    event = threading.Event()
                    self._flights[cache_id] = event
                    owner = True
                else:
                    owner = False
            if owner:
                break
            event.wait()
        try:
            return self._publish(identity, converter)
        finally:
            with self._lock:
                self._flights.pop(cache_id).set()

    def load(self, identity: NumericConversionCacheIdentity) -> NumericConversionCacheEntry:
        entry = self.root / "entries" / identity.cache_id
        try:
            manifest = _read_manifest(entry / "manifest.json")
            signature = manifest.pop("signature")
            expected = hmac.new(self._key, _canonical(manifest), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError("numeric conversion cache signature mismatch")
            if (manifest["schema_version"] != 1
                    or manifest["identity"] != asdict(identity)
                    or manifest["cache_id"] != identity.cache_id):
                raise ValueError("numeric conversion cache identity mismatch")
            output = entry / "output.bin"
            size, digest = _file_identity(output)
            if size != manifest["output_bytes"] or digest != manifest["output_sha256"]:
                raise ValueError("numeric conversion cache output mismatch")
            os.utime(output, None, follow_symlinks=False)
            return NumericConversionCacheEntry(identity.cache_id, output, size, digest)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            if entry.exists():
                self._quarantine(entry, "verification_failed")
            raise ValueError("numeric conversion cache entry was quarantined") from error

    def revoke(self, identity: NumericConversionCacheIdentity, reason: str) -> Path:
        if not reason or len(reason) > 128 or any(ord(character) < 0x20 for character in reason):
            raise ValueError("invalid numeric conversion cache revocation")
        with self._lock:
            entry = self.root / "entries" / identity.cache_id
            if not entry.is_dir():
                raise ValueError("numeric conversion cache entry is unavailable")
            return self._quarantine(entry, reason)

    def _publish(
        self,
        identity: NumericConversionCacheIdentity,
        converter: Callable[[Path], None],
    ) -> NumericConversionCacheEntry:
        temporary = Path(tempfile.mkdtemp(prefix=".convert-", dir=self.root / "entries"))
        os.chmod(temporary, 0o700)
        try:
            output = temporary / "output.bin"
            converter(output)
            size, digest = _file_identity(output)
            manifest = {
                "schema_version": 1,
                "cache_id": identity.cache_id,
                "identity": asdict(identity),
                "output_bytes": size,
                "output_sha256": digest,
            }
            manifest["signature"] = hmac.new(
                self._key, _canonical(manifest), hashlib.sha256
            ).hexdigest()
            _write_private(temporary / "manifest.json", _canonical(manifest) + b"\n")
            _fsync_directory(temporary)
            with self._lock:
                self._make_room(size)
                destination = self.root / "entries" / identity.cache_id
                os.replace(temporary, destination)
                _fsync_directory(destination.parent)
            return NumericConversionCacheEntry(
                identity.cache_id, destination / "output.bin", size, digest
            )
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _make_room(self, incoming_bytes: int) -> None:
        entries = []
        total = 0
        for path in (self.root / "entries").iterdir():
            if path.name.startswith(".convert-"):
                continue
            try:
                attributes = (path / "output.bin").stat()
            except OSError:
                self._quarantine(path, "inventory_failed")
                continue
            total += attributes.st_size
            entries.append((attributes.st_mtime_ns, path, attributes.st_size))
        while (len(entries) >= self._maximum_entries
               or total + incoming_bytes > self._maximum_bytes):
            if not entries:
                raise ValueError("numeric conversion exceeds cache capacity")
            _, victim, size = min(entries, key=lambda item: (item[0], item[1].name))
            entries = [item for item in entries if item[1] != victim]
            total -= size
            self._quarantine(victim, "bounded_lru_eviction")

    def _quarantine(self, entry: Path, reason: str) -> Path:
        destination = self.root / "quarantine" / f"{entry.name}-{secrets.token_hex(8)}"
        os.replace(entry, destination)
        _write_private(destination / "revocation.json", _canonical({
            "schema_version": 1, "cache_id": entry.name, "reason": reason,
        }) + b"\n")
        _fsync_directory(destination.parent)
        return destination


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    attributes = path.lstat()
    if (not stat.S_ISDIR(attributes.st_mode) or attributes.st_uid != os.getuid()
            or attributes.st_mode & 0o077):
        raise ValueError("numeric conversion cache directory must be private")


def _file_identity(path: Path) -> tuple[int, str]:
    attributes = path.lstat()
    if (not stat.S_ISREG(attributes.st_mode) or attributes.st_uid != os.getuid()
            or attributes.st_mode & 0o077
            or not 1 <= attributes.st_size <= MAX_CONVERSION_OUTPUT_BYTES):
        raise ValueError("numeric conversion cache output is unsafe")
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return attributes.st_size, digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, object]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        attributes = os.fstat(descriptor)
        if attributes.st_size > 64 * 1024:
            raise ValueError("numeric conversion cache manifest is too large")
        value = json.loads(os.read(descriptor, 64 * 1024 + 1))
    finally:
        os.close(descriptor)
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "cache_id", "identity", "output_bytes",
        "output_sha256", "signature",
    }:
        raise ValueError("numeric conversion cache manifest is invalid")
    return value


def _write_private(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
