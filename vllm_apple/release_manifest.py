"""Bounded provenance manifests for notarized Mac release archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO


SCHEMA_VERSION = 1
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_ENTRY_COUNT = 4096
MAX_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
CHUNK_BYTES = 8 * 1024 * 1024
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
EXPECTED_COMPONENTS = (
    "VLLMAppleChat.app/Contents/MacOS/VLLMAppleChat",
    "VLLMAppleChat.app/Contents/MacOS/vllm-appled",
)
INFO_PLIST = "VLLMAppleChat.app/Contents/Info.plist"


class ReleaseManifestError(ValueError):
    """Raised when release evidence is unsafe or does not match."""


def _regular_file(path: Path, *, maximum_bytes: int, description: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ReleaseManifestError(f"{description} must be a regular file, not a symlink")
    if path.stat().st_size > maximum_bytes:
        raise ReleaseManifestError(f"{description} exceeds the size limit")


def _sha256_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(CHUNK_BYTES):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _sha256_file(path: Path) -> tuple[str, int]:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _validated_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members = archive.infolist()
    if len(members) > MAX_ENTRY_COUNT:
        raise ReleaseManifestError("release archive has too many entries")
    if sum(member.file_size for member in members) > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise ReleaseManifestError("release archive expands beyond the size limit")
    result: dict[str, zipfile.ZipInfo] = {}
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or "" in path.parts:
            raise ReleaseManifestError("release archive contains an unsafe path")
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ReleaseManifestError("release archive contains a symlink")
        if member.filename in result:
            raise ReleaseManifestError("release archive contains a duplicate path")
        result[member.filename] = member
    return result


def _read_bounded_member(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo, *, maximum_bytes: int
) -> bytes:
    if member.file_size > maximum_bytes:
        raise ReleaseManifestError("release metadata exceeds the size limit")
    with archive.open(member) as stream:
        content = stream.read(maximum_bytes + 1)
    if len(content) > maximum_bytes:
        raise ReleaseManifestError("release metadata exceeds the size limit")
    return content


def build_release_manifest(
    archive_path: Path,
    notary_report_path: Path,
    *,
    source_commit: str,
    build_run_id: str,
    runner_image: str,
) -> dict[str, object]:
    _regular_file(archive_path, maximum_bytes=MAX_ARCHIVE_BYTES, description="release archive")
    _regular_file(
        notary_report_path, maximum_bytes=MAX_METADATA_BYTES, description="notary report"
    )
    if not isinstance(source_commit, str) or not COMMIT_PATTERN.fullmatch(source_commit):
        raise ReleaseManifestError("source commit must be 40 lowercase hexadecimal characters")
    if not build_run_id or len(build_run_id) > 128:
        raise ReleaseManifestError("build run ID must contain 1 to 128 characters")
    if not runner_image or len(runner_image) > 128:
        raise ReleaseManifestError("runner image must contain 1 to 128 characters")

    try:
        notary_report = json.loads(notary_report_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseManifestError("notary report is not valid JSON") from error
    if not isinstance(notary_report, dict) or notary_report.get("status") != "Accepted":
        raise ReleaseManifestError("notary report does not contain an Accepted result")
    submission_id = notary_report.get("id")
    if not isinstance(submission_id, str) or not submission_id or len(submission_id) > 128:
        raise ReleaseManifestError("notary report does not contain a bounded submission ID")

    archive_sha256, archive_size = _sha256_file(archive_path)
    report_sha256, report_size = _sha256_file(notary_report_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = _validated_members(archive)
            missing = [name for name in (*EXPECTED_COMPONENTS, INFO_PLIST) if name not in members]
            if missing:
                raise ReleaseManifestError(f"release archive is missing {missing[0]}")
            components = []
            for name in EXPECTED_COMPONENTS:
                with archive.open(members[name]) as stream:
                    sha256, size = _sha256_stream(stream)
                components.append({"path": name, "sha256": sha256, "size_bytes": size})
            info = plistlib.loads(
                _read_bounded_member(archive, members[INFO_PLIST], maximum_bytes=MAX_METADATA_BYTES)
            )
    except (zipfile.BadZipFile, plistlib.InvalidFileException) as error:
        raise ReleaseManifestError("release archive metadata is invalid") from error

    if not isinstance(info, dict):
        raise ReleaseManifestError("release Info.plist must be a dictionary")
    required_info = {
        "bundle_identifier": info.get("CFBundleIdentifier"),
        "bundle_version": info.get("CFBundleVersion"),
        "short_version": info.get("CFBundleShortVersionString"),
        "minimum_system_version": info.get("LSMinimumSystemVersion"),
    }
    if any(not isinstance(value, str) or not value for value in required_info.values()):
        raise ReleaseManifestError("release Info.plist is missing required version metadata")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"commit": source_commit},
        "build": {"run_id": build_run_id, "runner_image": runner_image},
        "notarization": {
            "id": submission_id,
            "status": "Accepted",
            "report_sha256": report_sha256,
            "report_size_bytes": report_size,
        },
        "artifact": {
            "filename": archive_path.name,
            "sha256": archive_sha256,
            "size_bytes": archive_size,
            **required_info,
            "components": components,
        },
    }


def save_release_manifest(manifest: dict[str, object], output_path: Path) -> None:
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(output_path, flags, 0o600)
    except OSError as error:
        raise ReleaseManifestError("release manifest output must not already exist") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def verify_release_manifest(
    archive_path: Path, notary_report_path: Path, manifest_path: Path
) -> dict[str, object]:
    _regular_file(manifest_path, maximum_bytes=MAX_METADATA_BYTES, description="release manifest")
    try:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseManifestError("release manifest is not valid JSON") from error
    if not isinstance(expected, dict):
        raise ReleaseManifestError("release manifest must be a JSON object")
    source = expected.get("source")
    build = expected.get("build")
    if not isinstance(source, dict) or not isinstance(build, dict):
        raise ReleaseManifestError("release manifest provenance is missing")
    actual = build_release_manifest(
        archive_path,
        notary_report_path,
        source_commit=source.get("commit", ""),
        build_run_id=build.get("run_id", ""),
        runner_image=build.get("runner_image", ""),
    )
    actual["created_at"] = expected.get("created_at")
    if actual != expected:
        raise ReleaseManifestError("release artifact does not match its manifest")
    return expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vllm-apple-release-manifest")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("archive", type=Path)
    create.add_argument("--notary-report", required=True, type=Path)
    create.add_argument("--source-commit", required=True)
    create.add_argument("--build-run-id", required=True)
    create.add_argument("--runner-image", required=True)
    create.add_argument("--output", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("archive", type=Path)
    verify.add_argument("--notary-report", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "create":
            manifest = build_release_manifest(
                arguments.archive,
                arguments.notary_report,
                source_commit=arguments.source_commit,
                build_run_id=arguments.build_run_id,
                runner_image=arguments.runner_image,
            )
            save_release_manifest(manifest, arguments.output)
        else:
            verify_release_manifest(arguments.archive, arguments.notary_report, arguments.manifest)
    except ReleaseManifestError as error:
        parser.error(str(error))
    return 0
