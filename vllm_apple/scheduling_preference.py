"""Private, bounded persistence for the runtime scheduling preference."""
from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from .hardware import default_application_support

SCHEMA_VERSION = 1
MAX_BYTES = 4096
PREFERENCES = frozenset({"automatic", "low_power", "high_performance"})


def default_scheduling_preference_path() -> Path:
    return default_application_support() / "settings" / "scheduling-preference.json"


def _private_parent(path: Path) -> None:
    attributes = path.parent.lstat()
    if not stat.S_ISDIR(attributes.st_mode) or attributes.st_uid != os.getuid() or (
        attributes.st_mode & 0o077
    ):
        raise ValueError("scheduling preference directory must be private")


def _private_file(attributes: os.stat_result) -> bool:
    return (
        stat.S_ISREG(attributes.st_mode)
        and attributes.st_uid == os.getuid()
        and not attributes.st_mode & 0o077
        and 1 <= attributes.st_size <= MAX_BYTES
    )


def load_scheduling_preference(path: Path) -> str:
    _private_parent(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        attributes = os.fstat(descriptor)
        if not _private_file(attributes):
            raise ValueError("scheduling preference must be a bounded private regular file")
        encoded = os.read(descriptor, MAX_BYTES + 1)
    finally:
        os.close(descriptor)
    payload = json.loads(encoded.decode("utf-8"))
    if (
        type(payload) is not dict
        or set(payload) != {"schema_version", "preference"}
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != SCHEMA_VERSION
        or type(payload["preference"]) is not str
        or payload["preference"] not in PREFERENCES
    ):
        raise ValueError("invalid scheduling preference")
    return payload["preference"]


def save_scheduling_preference(preference: str, path: Path) -> Path:
    if type(preference) is not str or preference not in PREFERENCES:
        raise ValueError("invalid scheduling preference")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _private_parent(path)
    try:
        existing = path.lstat()
    except FileNotFoundError:
        pass
    else:
        if not _private_file(existing):
            raise ValueError("existing scheduling preference must be private")
    encoded = (
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "preference": preference},
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=".scheduling-preference.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path
