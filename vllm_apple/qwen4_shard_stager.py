from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

from .qwen4_conversion_plan import build_qwen4_conversion_plan
from .qwen4_weight_map import _bounded_index


COPY_CHUNK_BYTES = 8 * 1024 * 1024
MAX_STAGE_FILES = 512
CHECKPOINT_NAME = ".qwen4-stage-checkpoint.json"
MANIFEST_NAME = "qwen4-stage-manifest.json"
MAX_MANIFEST_BYTES = 1024 * 1024


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_regular(path: Path, *, maximum_bytes: int) -> os.stat_result:
    info = path.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or not 1 <= info.st_size <= maximum_bytes
    ):
        raise ValueError("Qwen4 stage source shard is outside the bounded file policy")
    return info


def _hash_file(path: Path, *, maximum_bytes: int) -> tuple[int, str]:
    info = _safe_regular(path, maximum_bytes=maximum_bytes)
    digest = hashlib.sha256()
    read_bytes = 0
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
            read_bytes += len(chunk)
            if read_bytes > maximum_bytes:
                raise ValueError("Qwen4 stage file exceeded its byte limit")
    after = path.stat()
    if read_bytes != info.st_size or (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Qwen4 stage file changed while reading")
    return read_bytes, digest.hexdigest()


def _copy_atomic(source: Path, destination: Path, *, maximum_bytes: int) -> tuple[int, str]:
    source_info = _safe_regular(source, maximum_bytes=maximum_bytes)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    digest = hashlib.sha256()
    copied = 0
    try:
        os.fchmod(descriptor, 0o600)
        with source.open("rb", buffering=0) as reader, os.fdopen(descriptor, "wb", buffering=0) as writer:
            while chunk := reader.read(COPY_CHUNK_BYTES):
                copied += len(chunk)
                if copied > maximum_bytes:
                    raise ValueError("Qwen4 stage output exceeded its byte limit")
                digest.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        after = source.stat()
        if copied != source_info.st_size or (
            source_info.st_dev,
            source_info.st_ino,
            source_info.st_size,
            source_info.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("Qwen4 stage source changed while copying")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return copied, digest.hexdigest()


def _load_checkpoint(path: Path, *, plan_id: str) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    info = _safe_regular(path, maximum_bytes=1024 * 1024)
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("Qwen4 stage checkpoint is not private")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Qwen4 stage checkpoint is invalid") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("plan_id") != plan_id
        or not isinstance(payload.get("completed"), dict)
    ):
        raise ValueError("Qwen4 stage checkpoint does not match the conversion plan")
    return payload["completed"]


def _load_manifest(path: Path) -> dict[str, object]:
    info = _safe_regular(path, maximum_bytes=MAX_MANIFEST_BYTES)
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("Qwen4 stage manifest is not private")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Qwen4 stage manifest is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("Qwen4 stage manifest is invalid")
    return payload


def verify_qwen4_stage(
    root: str | Path,
    *,
    requested_modes: tuple[str, ...] = ("text",),
    maximum_artifact_bytes: int,
) -> dict[str, object]:
    if (
        not isinstance(maximum_artifact_bytes, int)
        or isinstance(maximum_artifact_bytes, bool)
        or maximum_artifact_bytes <= 0
    ):
        raise ValueError("Qwen4 stage verification byte ceiling is invalid")
    stage_root = Path(root).expanduser().resolve(strict=True)
    root_info = stage_root.lstat()
    if (
        stat.S_ISLNK(root_info.st_mode)
        or not stat.S_ISDIR(root_info.st_mode)
        or root_info.st_uid != os.getuid()
        or stat.S_IMODE(root_info.st_mode) & 0o077
    ):
        raise ValueError("Qwen4 stage directory is unsafe")
    manifest = _load_manifest(stage_root / MANIFEST_NAME)
    index_path = stage_root / "model.safetensors.index.json"
    plan = build_qwen4_conversion_plan(stage_root, index_path, requested_modes=requested_modes)
    weight_map, _, _ = _bounded_index(index_path)
    shard_names = sorted(set(weight_map.values()))
    evidence = manifest.get("shards")
    expected_keys = {
        "schema_version",
        "completed",
        "plan_id",
        "config_fingerprint",
        "index_sha256",
        "requested_modes",
        "shard_count",
        "output_bytes",
        "copied_shards",
        "reused_shards",
        "copy_chunk_bytes",
        "peak_open_source_shards",
        "peak_open_destination_shards",
        "preserves_source_tensor_names",
        "shards",
    }
    if set(manifest) != expected_keys or not isinstance(evidence, dict):
        raise ValueError("Qwen4 stage manifest schema is invalid")
    if (
        manifest["schema_version"] != 1
        or manifest["completed"] is not True
        or manifest["plan_id"] != plan["plan_id"]
        or manifest["config_fingerprint"] != plan["config_fingerprint"]
        or manifest["index_sha256"] != plan["index_sha256"]
        or manifest["requested_modes"] != list(requested_modes)
        or manifest["shard_count"] != len(shard_names)
        or manifest["copy_chunk_bytes"] != COPY_CHUNK_BYTES
        or manifest["peak_open_source_shards"] != 1
        or manifest["peak_open_destination_shards"] != 1
        or manifest["preserves_source_tensor_names"] is not True
        or set(evidence) != set(shard_names)
        or not isinstance(manifest["copied_shards"], int)
        or isinstance(manifest["copied_shards"], bool)
        or not isinstance(manifest["reused_shards"], int)
        or isinstance(manifest["reused_shards"], bool)
        or manifest["copied_shards"] < 0
        or manifest["reused_shards"] < 0
        or manifest["copied_shards"] + manifest["reused_shards"] != len(shard_names)
    ):
        raise ValueError("Qwen4 stage manifest does not match the artifact plan")
    expected_files = set(shard_names) | {
        "config.json",
        "model.safetensors.index.json",
        MANIFEST_NAME,
    }
    if {entry.name for entry in stage_root.iterdir()} != expected_files:
        raise ValueError("Qwen4 stage contains an unexpected or missing file")
    total_bytes = 0
    for shard_name in shard_names:
        item = evidence[shard_name]
        if (
            not isinstance(item, dict)
            or set(item) != {"bytes", "sha256"}
            or not isinstance(item["bytes"], int)
            or isinstance(item["bytes"], bool)
            or item["bytes"] <= 0
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in item["sha256"])
        ):
            raise ValueError("Qwen4 stage shard evidence is invalid")
        size, digest = _hash_file(stage_root / shard_name, maximum_bytes=maximum_artifact_bytes)
        if item != {"bytes": size, "sha256": digest}:
            raise ValueError("Qwen4 stage shard digest verification failed")
        total_bytes += size
        if total_bytes > maximum_artifact_bytes:
            raise ValueError("Qwen4 staged artifact exceeded its byte limit")
    for metadata_name in ("config.json", "model.safetensors.index.json"):
        size, _ = _hash_file(
            stage_root / metadata_name,
            maximum_bytes=min(maximum_artifact_bytes, 4 * 1024 * 1024),
        )
        total_bytes += size
    if total_bytes != manifest["output_bytes"] or total_bytes > maximum_artifact_bytes:
        raise ValueError("Qwen4 stage manifest output size is invalid")
    return {
        "schema_version": 1,
        "verified": True,
        "plan_id": plan["plan_id"],
        "config_fingerprint": plan["config_fingerprint"],
        "index_sha256": plan["index_sha256"],
        "requested_modes": list(requested_modes),
        "shard_count": len(shard_names),
        "output_bytes": total_bytes,
    }


def stage_qwen4_shards(
    source: str | Path,
    output: str | Path,
    *,
    maximum_output_bytes: int,
    requested_modes: tuple[str, ...] = ("text",),
    resume: bool = False,
) -> dict[str, object]:
    if (
        not isinstance(maximum_output_bytes, int)
        or isinstance(maximum_output_bytes, bool)
        or maximum_output_bytes <= 0
    ):
        raise ValueError("Qwen4 stage maximum output bytes are invalid")
    source_root = Path(source).expanduser().resolve(strict=True)
    output_root = Path(output).expanduser().resolve(strict=False)
    if output_root == source_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("Qwen4 stage source and output must be disjoint")
    if output_root.exists():
        info = output_root.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise ValueError("Qwen4 stage output directory is unsafe")
        if not resume and any(output_root.iterdir()):
            raise ValueError("Qwen4 stage output already contains files; explicit resume is required")
    else:
        output_root.mkdir(mode=0o700, parents=False)
    index_path = source_root / "model.safetensors.index.json"
    plan = build_qwen4_conversion_plan(
        source_root, index_path, requested_modes=requested_modes
    )
    weight_map, _, _ = _bounded_index(index_path)
    shard_names = sorted(set(weight_map.values()))
    if len(shard_names) > MAX_STAGE_FILES:
        raise ValueError("Qwen4 stage shard count exceeds the bounded limit")
    checkpoint_path = output_root / CHECKPOINT_NAME
    completed = _load_checkpoint(checkpoint_path, plan_id=plan["plan_id"]) if resume else {}
    if not set(completed).issubset(shard_names):
        raise ValueError("Qwen4 stage checkpoint contains an unknown shard")
    copied_shards = 0
    reused_shards = 0
    total_bytes = 0
    for shard_name in shard_names:
        destination = output_root / shard_name
        evidence = completed.get(shard_name)
        if evidence is not None:
            if not isinstance(evidence, dict) or set(evidence) != {"bytes", "sha256"}:
                raise ValueError("Qwen4 stage checkpoint shard evidence is invalid")
            size, digest = _hash_file(destination, maximum_bytes=maximum_output_bytes)
            if evidence != {"bytes": size, "sha256": digest}:
                raise ValueError("Qwen4 staged shard does not match its checkpoint")
            reused_shards += 1
        else:
            if destination.exists():
                raise ValueError("Qwen4 staged shard exists without checkpoint evidence")
            remaining_bytes = maximum_output_bytes - total_bytes
            if remaining_bytes <= 0:
                raise ValueError("Qwen4 staged artifact exceeded its byte limit")
            size, digest = _copy_atomic(
                source_root / shard_name,
                destination,
                maximum_bytes=remaining_bytes,
            )
            completed[shard_name] = {"bytes": size, "sha256": digest}
            copied_shards += 1
            _atomic_json(
                checkpoint_path,
                {"schema_version": 1, "plan_id": plan["plan_id"], "completed": completed},
            )
        total_bytes += size
        if total_bytes > maximum_output_bytes:
            raise ValueError("Qwen4 staged artifact exceeded its byte limit")
    for metadata_name in ("config.json", "model.safetensors.index.json"):
        destination = output_root / metadata_name
        if destination.exists():
            if not resume:
                raise ValueError("Qwen4 staged metadata already exists")
            source_size, source_digest = _hash_file(
                source_root / metadata_name,
                maximum_bytes=min(maximum_output_bytes, 4 * 1024 * 1024),
            )
            size, destination_digest = _hash_file(
                destination,
                maximum_bytes=min(maximum_output_bytes, 4 * 1024 * 1024),
            )
            if size != source_size or destination_digest != source_digest:
                raise ValueError("Qwen4 staged metadata does not match the source")
        else:
            remaining_bytes = maximum_output_bytes - total_bytes
            if remaining_bytes <= 0:
                raise ValueError("Qwen4 staged artifact exceeded its byte limit")
            size, _ = _copy_atomic(
                source_root / metadata_name,
                destination,
                maximum_bytes=min(remaining_bytes, 4 * 1024 * 1024),
            )
        total_bytes += size
        if total_bytes > maximum_output_bytes:
            raise ValueError("Qwen4 staged artifact exceeded its byte limit")
    manifest = {
        "schema_version": 1,
        "completed": True,
        "plan_id": plan["plan_id"],
        "config_fingerprint": plan["config_fingerprint"],
        "index_sha256": plan["index_sha256"],
        "requested_modes": list(requested_modes),
        "shard_count": len(shard_names),
        "output_bytes": total_bytes,
        "copied_shards": copied_shards,
        "reused_shards": reused_shards,
        "copy_chunk_bytes": COPY_CHUNK_BYTES,
        "peak_open_source_shards": 1,
        "peak_open_destination_shards": 1,
        "preserves_source_tensor_names": True,
        "shards": dict(sorted(completed.items())),
    }
    _atomic_json(output_root / MANIFEST_NAME, manifest)
    try:
        checkpoint_path.unlink()
    except FileNotFoundError:
        pass
    _fsync_directory(output_root)
    return manifest
