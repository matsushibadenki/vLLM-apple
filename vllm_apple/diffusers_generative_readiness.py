from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path


MAX_PROBE_OUTPUT_BYTES = 16 * 1024
MAX_SOURCE_FILE_BYTES = 2 * 1024 * 1024
MAX_SOURCE_FILES = 4096
MAX_TOTAL_SOURCE_BYTES = 32 * 1024 * 1024

_REQUIRED_PIPELINES = {
    "flux2-klein-9b-base": {"Flux2KleinPipeline"},
    "qwen-image-2512": {"QwenImagePipeline"},
    "flux2-dev": {"Flux2Pipeline"},
    "wan2.2-ti2v-5b": {"WanPipeline", "WanImageToVideoPipeline"},
    "hunyuanvideo-1.5-8.3b": {
        "HunyuanVideo15Pipeline",
        "HunyuanVideo15ImageToVideoPipeline",
    },
    "wan2.2-a14b-quantized": {"WanPipeline", "WanImageToVideoPipeline"},
}


def _source_symbols(root: Path) -> tuple[frozenset[str], int, int]:
    symbols: set[str] = set()
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*.py")):
        file_count += 1
        if file_count > MAX_SOURCE_FILES:
            raise ValueError("diffusers source file limit exceeded")
        if path.is_symlink() or not path.is_file():
            continue
        size = path.stat().st_size
        if not 1 <= size <= MAX_SOURCE_FILE_BYTES:
            continue
        total_bytes += size
        if total_bytes > MAX_TOTAL_SOURCE_BYTES:
            raise ValueError("diffusers source byte limit exceeded")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        symbols.update(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        )
    return frozenset(symbols), file_count, total_bytes


def inspect_diffusers_generative_sources(
    source_root: str | Path, *, version: str, executable: str
) -> dict[str, object]:
    root = Path(source_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("diffusers source root must be a directory")
    symbols, file_count, total_bytes = _source_symbols(root)
    candidates: dict[str, dict[str, object]] = {}
    for candidate_id, required in _REQUIRED_PIPELINES.items():
        missing = sorted(required - symbols)
        candidates[candidate_id] = {
            "ready": not missing,
            "required_pipeline_classes": sorted(required),
            "missing_pipeline_classes": missing,
        }
    ready_candidates = sorted(
        candidate_id for candidate_id, evidence in candidates.items() if evidence["ready"]
    )
    return {
        "schema_version": 1,
        "backend": "diffusers",
        "executable": executable,
        "diffusers_version": version,
        "source_root": str(root),
        "source_files_examined": file_count,
        "source_bytes_examined": total_bytes,
        "ready": len(ready_candidates) == len(_REQUIRED_PIPELINES),
        "ready_candidates": ready_candidates,
        "candidates": candidates,
        "imports_backend": False,
        "allocates_model_or_metal": False,
    }


def inspect_diffusers_generative_readiness(executable: str | Path) -> dict[str, object]:
    path = Path(executable).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("diffusers Python executable is not executable")
    script = (
        "import importlib.metadata as m,json;"
        "d=m.distribution('diffusers');"
        "print(json.dumps({'version':d.version,'source_root':"
        "str(d.locate_file('diffusers'))}))"
    )
    try:
        result = subprocess.run(
            [str(path), "-c", script],
            capture_output=True,
            check=True,
            text=True,
            timeout=5.0,
        )
        raw = result.stdout.strip()
        if not 1 <= len(raw.encode("utf-8")) <= MAX_PROBE_OUTPUT_BYTES:
            raise ValueError("diffusers metadata probe output is outside the bounded limit")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or set(payload) != {"version", "source_root"}:
            raise ValueError("diffusers metadata probe output has an invalid schema")
        version = payload["version"]
        source_root = payload["source_root"]
        if not isinstance(version, str) or not isinstance(source_root, str):
            raise ValueError("diffusers metadata probe values are invalid")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise ValueError("diffusers metadata probe failed") from error
    return inspect_diffusers_generative_sources(
        source_root,
        version=version,
        executable=str(path.resolve()),
    )
