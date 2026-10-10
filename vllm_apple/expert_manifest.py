"""Bounded expert artifact identity; hashes detect drift, not publisher trust."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .expert_residency import ExpertKey
from .expert_timing import measure


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


@dataclass(frozen=True)
class ExpertManifest:
    model_sha256: str
    quantization_bits: int | None
    group_size: int
    entries: Mapping[ExpertKey, tuple[int, str]]

    @classmethod
    def load(cls, path: Path, *, expected_model_sha256: str) -> "ExpertManifest":
        if not _sha(expected_model_sha256):
            raise ValueError("invalid expected model identity")
        path = Path(path)
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 4 * 1024 * 1024:
            raise ValueError("invalid expert manifest file")
        raw = path.read_bytes()
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("expert manifest exceeds limit")
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate manifest field")
                result[key] = value
            return result
        payload = json.loads(raw, object_pairs_hook=unique_pairs)
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version", "model_sha256", "quantization_bits", "group_size", "experts"
        }:
            raise ValueError("invalid expert manifest schema")
        bits, group = payload["quantization_bits"], payload["group_size"]
        rows = payload["experts"]
        if (type(payload["schema_version"]) is not int or payload["schema_version"] != 1
                or payload["model_sha256"] != expected_model_sha256
                or (bits is not None and (type(bits) is not int or bits not in (4, 8)))
                or type(group) is not int or group not in (32, 64, 128)
                or not isinstance(rows, list) or not 1 <= len(rows) <= 65_536):
            raise ValueError("expert manifest identity or format mismatch")
        entries = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"layer", "expert", "size_bytes", "sha256"}:
                raise ValueError("invalid expert manifest entry")
            key = ExpertKey(row["layer"], row["expert"])
            if (key in entries or type(row["size_bytes"]) is not int
                    or not 0 < row["size_bytes"] <= 1 << 40 or not _sha(row["sha256"])):
                raise ValueError("invalid or duplicate expert identity")
            entries[key] = (row["size_bytes"], row["sha256"])
        return cls(expected_model_sha256, bits, group, MappingProxyType(entries))

    def verify_file(self, key: ExpertKey, path: Path, *, timings=None) -> None:
        expected = self.entries.get(key)
        if expected is None or path.stat().st_size != expected[0]:
            raise ValueError("expert missing from manifest or size mismatch")
        digest = hashlib.sha256()
        total = 0
        with measure(timings, "checksum_read"):
            stream = path.open("rb")
        with stream:
            while True:
                with measure(timings, "checksum_read"):
                    chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > expected[0]:
                    raise ValueError("expert changed during verification")
                with measure(timings, "checksum_hash"):
                    digest.update(chunk)
        if total != expected[0] or digest.hexdigest() != expected[1]:
            raise ValueError("expert checksum mismatch")
