"""Private, bounded NVFP4 artifact transport for the local runtime worker."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import stat
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .numeric_formats import (
    NumericFormatDescriptor,
    ScaledInt8Tensor,
    TensorGeometry,
    convert_nvfp4_to_int8,
)
from .numeric_precision import NumericPrecisionPolicy, PrecisionExecutionContract


MAX_NUMERIC_ARTIFACT_BYTES = 128 * 1024


def _digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _name(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and not value.startswith(".")
        and value.endswith(".json")
        and all(character.isascii() and (character.isalnum() or character in "-_.")
                for character in value)
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("numeric artifact contains a duplicate JSON key")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class LoadedNumericArtifact:
    tensor: ScaledInt8Tensor
    execution_contract: PrecisionExecutionContract
    artifact_digest: str
    artifact_name: str | None = None
    file_identity: tuple[int, int, int] | None = None
    quarantined: bool = False


def encode_nvfp4_numeric_artifact(
    descriptor: NumericFormatDescriptor,
    packed: bytes,
    scales: bytes,
    global_scale: float,
    *,
    geometry: TensorGeometry | None,
    target_dtype: str,
    precision_policy: NumericPrecisionPolicy,
) -> tuple[bytes, LoadedNumericArtifact]:
    if not isinstance(precision_policy, NumericPrecisionPolicy):
        raise ValueError("numeric artifact precision policy is invalid")
    tensor = convert_nvfp4_to_int8(
        descriptor, packed, scales, global_scale, geometry=geometry)
    contract = PrecisionExecutionContract(
        tensor.target_digest, target_dtype, precision_policy)
    payload = {
        "schema_version": 1,
        "kind": "nvfp4_scaled_int8_v1",
        "source_descriptor": asdict(descriptor),
        "geometry": None if geometry is None else asdict(geometry),
        "packed_source_base64": base64.b64encode(packed).decode("ascii"),
        "block_scales_base64": base64.b64encode(scales).decode("ascii"),
        "global_scale_hex": float(global_scale).hex(),
        "source_digest": tensor.source_digest,
        "target_digest": tensor.target_digest,
        "execution_contract": contract.to_dict(),
        "stores_tensor_values": True,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if len(encoded) > MAX_NUMERIC_ARTIFACT_BYTES:
        raise ValueError("numeric artifact exceeds the bounded limit")
    digest = hashlib.sha256(encoded).hexdigest()
    return encoded, LoadedNumericArtifact(tensor, contract, digest)


def write_nvfp4_numeric_artifact(
    path: str | Path,
    descriptor: NumericFormatDescriptor,
    packed: bytes,
    scales: bytes,
    global_scale: float,
    *,
    geometry: TensorGeometry | None,
    target_dtype: str,
    precision_policy: NumericPrecisionPolicy,
) -> LoadedNumericArtifact:
    destination = Path(path).expanduser()
    if not _name(destination.name) or destination.exists() or destination.is_symlink():
        raise ValueError("numeric artifact destination is invalid")
    parent = destination.parent.resolve(strict=True)
    destination = parent / destination.name
    info = parent.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077):
        raise ValueError("numeric artifact directory must be private and current-user owned")
    encoded, loaded = encode_nvfp4_numeric_artifact(
        descriptor, packed, scales, global_scale, geometry=geometry,
        target_dtype=target_dtype, precision_policy=precision_policy)
    descriptor_fd, temporary = tempfile.mkstemp(prefix=".numeric-", dir=parent)
    try:
        os.fchmod(descriptor_fd, 0o600)
        with os.fdopen(descriptor_fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    info = destination.lstat()
    return replace(
        loaded, artifact_name=destination.name,
        file_identity=(info.st_dev, info.st_ino, info.st_size))


class NumericArtifactReader:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        info = self.root.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077):
            raise ValueError("numeric artifact root must be private and current-user owned")

    def read(self, artifact_name: str, expected_digest: str) -> LoadedNumericArtifact:
        if not _name(artifact_name) or not _digest(expected_digest):
            raise ValueError("numeric artifact request identity is invalid")
        encoded, identity, digest = self._read_file(self.root / artifact_name)
        if digest != expected_digest:
            raise ValueError("numeric artifact digest mismatch")
        return replace(
            _decode_numeric_artifact(encoded, digest), artifact_name=artifact_name,
            file_identity=identity)

    def claim(self, artifact_name: str, expected_digest: str) -> LoadedNumericArtifact:
        """Move a safe artifact out of the inbox before attempting backend work."""
        if not _name(artifact_name) or not _digest(expected_digest):
            raise ValueError("numeric artifact request identity is invalid")
        source = self.root / artifact_name
        encoded, identity, digest = self._read_file(source)
        if digest != expected_digest:
            self._move_to_quarantine(source, identity, digest)
            raise ValueError("numeric artifact digest mismatch; artifact quarantined")
        try:
            loaded = _decode_numeric_artifact(encoded, digest)
        except ValueError:
            self._move_to_quarantine(source, identity, digest)
            raise
        destination = self._move_to_quarantine(source, identity, digest)
        return replace(
            loaded, artifact_name=destination.name, file_identity=identity,
            quarantined=True)

    def consume(self, loaded: LoadedNumericArtifact) -> None:
        if (not isinstance(loaded, LoadedNumericArtifact) or not loaded.quarantined
                or not _name(loaded.artifact_name) or loaded.file_identity is None):
            raise ValueError("numeric artifact consume lease is invalid")
        quarantine = self._quarantine_directory(create=False)
        path = quarantine / loaded.artifact_name
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode)
                or (info.st_dev, info.st_ino, info.st_size) != loaded.file_identity):
            raise ValueError("numeric artifact consume identity changed")
        path.unlink()
        descriptor_fd = os.open(quarantine, os.O_RDONLY)
        try:
            os.fsync(descriptor_fd)
        finally:
            os.close(descriptor_fd)

    def _read_file(self, path: Path) -> tuple[bytes, tuple[int, int, int], str]:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor_fd = os.open(path, flags)
        try:
            before = os.fstat(descriptor_fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                    or stat.S_IMODE(before.st_mode) & 0o077
                    or not 1 <= before.st_size <= MAX_NUMERIC_ARTIFACT_BYTES):
                raise ValueError("numeric artifact file is unsafe")
            chunks = bytearray()
            while len(chunks) <= MAX_NUMERIC_ARTIFACT_BYTES:
                chunk = os.read(
                    descriptor_fd, min(64 * 1024, MAX_NUMERIC_ARTIFACT_BYTES + 1 - len(chunks)))
                if not chunk:
                    break
                chunks.extend(chunk)
            encoded = bytes(chunks)
            after = os.fstat(descriptor_fd)
            if (len(encoded) != before.st_size or (before.st_dev, before.st_ino, before.st_size)
                    != (after.st_dev, after.st_ino, after.st_size)):
                raise ValueError("numeric artifact changed while reading")
        finally:
            os.close(descriptor_fd)
        digest = hashlib.sha256(encoded).hexdigest()
        return encoded, (before.st_dev, before.st_ino, before.st_size), digest

    def _quarantine_directory(self, *, create: bool) -> Path:
        path = self.root / "quarantine"
        if create:
            try:
                path.mkdir(mode=0o700)
            except FileExistsError:
                pass
        info = path.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077):
            raise ValueError("numeric artifact quarantine is unsafe")
        return path

    def _move_to_quarantine(
        self, source: Path, identity: tuple[int, int, int], digest: str,
    ) -> Path:
        info = source.lstat()
        if (not stat.S_ISREG(info.st_mode)
                or (info.st_dev, info.st_ino, info.st_size) != identity):
            raise ValueError("numeric artifact identity changed before quarantine")
        quarantine = self._quarantine_directory(create=True)
        destination = quarantine / f"{digest}-{secrets.token_hex(8)}.json"
        os.link(source, destination, follow_symlinks=False)
        source.unlink()
        for directory in (self.root, quarantine):
            descriptor_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor_fd)
            finally:
                os.close(descriptor_fd)
        return destination


def _decode_numeric_artifact(encoded: bytes, artifact_digest: str) -> LoadedNumericArtifact:
    try:
        payload = json.loads(encoded, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("numeric artifact is not valid JSON") from error
    expected = {
        "schema_version", "kind", "source_descriptor", "geometry",
        "packed_source_base64", "block_scales_base64", "global_scale_hex",
        "source_digest", "target_digest", "execution_contract", "stores_tensor_values",
    }
    if (not isinstance(payload, dict) or set(payload) != expected
            or payload["schema_version"] != 1
            or payload["kind"] != "nvfp4_scaled_int8_v1"
            or payload["stores_tensor_values"] is not True
            or not _digest(payload["source_digest"])
            or not _digest(payload["target_digest"])):
        raise ValueError("numeric artifact schema is invalid")
    try:
        descriptor_payload = payload["source_descriptor"]
        if not isinstance(descriptor_payload, dict) or set(descriptor_payload) != {
            "encoding", "elements", "block_size", "packing", "scale_encoding",
            "layout", "value_multiplier", "schema_version"
        }:
            raise ValueError("invalid source descriptor fields")
        descriptor = NumericFormatDescriptor(**descriptor_payload)
        geometry_payload = payload["geometry"]
        if geometry_payload is None:
            geometry = None
        else:
            if not isinstance(geometry_payload, dict) or set(geometry_payload) != {
                "shape", "scale_axis", "block_size", "schema_version"
            }:
                raise ValueError("invalid geometry fields")
            geometry = TensorGeometry(
                tuple(geometry_payload["shape"]), geometry_payload["scale_axis"],
                geometry_payload["block_size"], geometry_payload["schema_version"])
        packed = base64.b64decode(payload["packed_source_base64"], validate=True)
        scales = base64.b64decode(payload["block_scales_base64"], validate=True)
        global_scale = float.fromhex(payload["global_scale_hex"])
        tensor = convert_nvfp4_to_int8(
            descriptor, packed, scales, global_scale, geometry=geometry)
        contract = PrecisionExecutionContract.from_dict(payload["execution_contract"])
    except (TypeError, ValueError) as error:
        raise ValueError("numeric artifact content is invalid") from error
    if (tensor.source_digest != payload["source_digest"]
            or tensor.target_digest != payload["target_digest"]
            or contract.tensor_digest != tensor.target_digest):
        raise ValueError("numeric artifact content binding mismatch")
    return LoadedNumericArtifact(tensor, contract, artifact_digest)
