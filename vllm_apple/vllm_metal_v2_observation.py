from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import asdict
from pathlib import Path

from .hardware import default_application_support
from .vllm_metal_v2_tuning import V2PagedAttentionShape

OBSERVATION_SCHEMA_VERSION = 1
MAX_OBSERVED_V2_SHAPES = 16
MAX_OBSERVATION_BYTES = 64 * 1024


def default_v2_observation_path(hardware_fingerprint: str, source_fingerprint: str) -> Path:
    for value in (hardware_fingerprint, source_fingerprint):
        if not value or value in {".", ".."} or Path(value).name != value:
            raise ValueError("invalid native v2 observation identity")
    return (
        default_application_support()
        / "profiles"
        / "vllm-metal-v2-observations"
        / hardware_fingerprint
        / source_fingerprint
        / "shapes.json"
    )


def load_v2_observations(
    path: Path, *, hardware_fingerprint: str, source_fingerprint: str
) -> tuple[V2PagedAttentionShape, ...]:
    attributes = path.lstat()
    if (
        not stat.S_ISREG(attributes.st_mode)
        or attributes.st_uid != os.getuid()
        or attributes.st_mode & 0o077
        or not 1 <= attributes.st_size <= MAX_OBSERVATION_BYTES
    ):
        raise ValueError("native v2 observations must be a bounded private regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "hardware_fingerprint",
        "source_fingerprint",
        "shapes",
    }:
        raise ValueError("invalid native v2 observation fields")
    if (
        payload["schema_version"] != OBSERVATION_SCHEMA_VERSION
        or payload["hardware_fingerprint"] != hardware_fingerprint
        or payload["source_fingerprint"] != source_fingerprint
        or not isinstance(payload["shapes"], list)
        or len(payload["shapes"]) > MAX_OBSERVED_V2_SHAPES
    ):
        raise ValueError("invalid native v2 observation identity or shape count")
    try:
        shapes = tuple(V2PagedAttentionShape(**item) for item in payload["shapes"])
    except (TypeError, ValueError) as error:
        raise ValueError("invalid native v2 observed shape") from error
    if len(set(shapes)) != len(shapes):
        raise ValueError("duplicate native v2 observed shape")
    return shapes


def record_v2_observed_shape(
    shape: V2PagedAttentionShape,
    *,
    hardware_fingerprint: str,
    source_fingerprint: str,
) -> Path:
    destination = default_v2_observation_path(hardware_fingerprint, source_fingerprint)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = destination.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise ValueError("native v2 observation directory must be private")
    try:
        current = list(
            load_v2_observations(
                destination,
                hardware_fingerprint=hardware_fingerprint,
                source_fingerprint=source_fingerprint,
            )
        )
    except FileNotFoundError:
        current = []
    if shape in current or len(current) >= MAX_OBSERVED_V2_SHAPES:
        return destination
    current.append(shape)
    payload = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "hardware_fingerprint": hardware_fingerprint,
        "source_fingerprint": source_fingerprint,
        "shapes": [asdict(item) for item in current],
    }
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    if len(encoded) > MAX_OBSERVATION_BYTES:
        raise ValueError("native v2 observations exceeded 64 KiB")
    descriptor, temporary = tempfile.mkstemp(prefix=".shapes.", dir=destination.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination
