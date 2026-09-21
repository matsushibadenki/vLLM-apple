#!/usr/bin/env python3
"""Record load-free Qwen Image text-encoder layer staging evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_apple.mflux_qwen_streaming_plan import (  # noqa: E402
    inspect_mflux_qwen_text_encoder_staging,
)
from vllm_apple.qualification import save_qualification_report  # noqa: E402


def _deployable_tree_identity(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256(b"vllm-apple-mflux-deployable-tree-v1\0")
    files: list[Path] = []
    for directory, names, filenames in os.walk(root):
        names[:] = [name for name in names if name not in {".git", ".cache", "__pycache__"}]
        for filename in filenames:
            files.append(Path(directory, filename))
    if not 1 <= len(files) <= 100_000:
        raise ValueError("deployable model file count is invalid")
    total = 0
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("deployable model contains a non-regular file")
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(info.st_size.to_bytes(8, "big"))
        with path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
        after = path.stat()
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ValueError("deployable model changed while hashing")
        total += info.st_size
    return digest.hexdigest(), len(files), total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    model_root = arguments.model.expanduser().resolve(strict=True)
    plan = inspect_mflux_qwen_text_encoder_staging(model_root)
    root_digest, file_count, artifact_bytes = _deployable_tree_identity(model_root)
    payload = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_text_encoder_load_free_staging_inventory",
        "candidate_id": "qwen-image-2512",
        "artifact_root_sha256": root_digest,
        "artifact_file_count": file_count,
        "artifact_bytes": artifact_bytes,
        "artifact_identity_scope": "deployable_files_excluding_git_cache_pycache",
        "staging": plan.to_dict(),
        "weights_loaded": False,
        "runtime_peak_qualified": False,
        "passed": True,
    }
    save_qualification_report(payload, arguments.report)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
