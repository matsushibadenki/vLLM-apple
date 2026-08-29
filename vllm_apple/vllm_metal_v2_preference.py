from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from .hardware import default_application_support

PREFERENCE_SCHEMA_VERSION = 1
MAX_PREFERENCE_BYTES = 4096


def default_native_v2_preference_path() -> Path:
    return default_application_support() / "settings" / "native-v2-tuning.json"


def load_native_v2_preference(path: Path) -> bool:
    attributes = path.lstat()
    if (
        not stat.S_ISREG(attributes.st_mode)
        or attributes.st_uid != os.getuid()
        or attributes.st_mode & 0o077
        or not 1 <= attributes.st_size <= MAX_PREFERENCE_BYTES
    ):
        raise ValueError("native v2 preference must be a bounded private regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "enabled"}
        or payload["schema_version"] != PREFERENCE_SCHEMA_VERSION
        or not isinstance(payload["enabled"], bool)
    ):
        raise ValueError("invalid native v2 preference")
    return payload["enabled"]


def save_native_v2_preference(enabled: bool, path: Path) -> Path:
    if not isinstance(enabled, bool):
        raise ValueError("native v2 preference must be boolean")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = path.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise ValueError("native v2 preference directory must be private")
    encoded = (
        json.dumps(
            {"schema_version": PREFERENCE_SCHEMA_VERSION, "enabled": enabled},
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode()
    descriptor, temporary = tempfile.mkstemp(prefix=".native-v2-tuning.", dir=path.parent)
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
