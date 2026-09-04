from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path

from .generative_artifact_inspection import inspect_generative_artifact


MAX_PROBE_OUTPUT_BYTES = 16 * 1024
MAX_SOURCE_FILE_BYTES = 2 * 1024 * 1024
MAX_SOURCE_FILES = 2048
MAX_TOTAL_SOURCE_BYTES = 32 * 1024 * 1024
_REQUIRED_CLASSES = {
    "z-image-turbo-mlx-4bit": {"ZImageTurbo", "ModelConfig"},
    "qwen-image-2512": {"QwenImage", "ModelConfig"},
}


def _source_symbols(root: Path) -> tuple[frozenset[str], int, int]:
    symbols: set[str] = set()
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*.py")):
        file_count += 1
        if file_count > MAX_SOURCE_FILES:
            raise ValueError("MFLUX source file limit exceeded")
        if path.is_symlink() or not path.is_file():
            continue
        size = path.stat().st_size
        if not 1 <= size <= MAX_SOURCE_FILE_BYTES:
            continue
        total_bytes += size
        if total_bytes > MAX_TOTAL_SOURCE_BYTES:
            raise ValueError("MFLUX source byte limit exceeded")
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


def inspect_mflux_generative_sources(
    source_root: str | Path,
    *,
    version: str,
    executable: str,
    model: str | Path | None = None,
) -> dict[str, object]:
    root = Path(source_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("MFLUX source root must be a directory")
    symbols, file_count, total_bytes = _source_symbols(root)
    artifact = inspect_generative_artifact(model) if model is not None else None
    artifact_format = artifact["artifact_format"] if artifact else None
    candidates: dict[str, dict[str, object]] = {}
    for candidate_id, required in _REQUIRED_CLASSES.items():
        missing = sorted(required - symbols)
        format_compatible = artifact is None or artifact_format == "mflux"
        issues = [f"missing_backend_class:{name}" for name in missing]
        if not format_compatible:
            issues.append(f"unsupported_artifact_format:{artifact_format}")
        candidates[candidate_id] = {
            "ready": not issues,
            "required_classes": sorted(required),
            "missing_classes": missing,
            "artifact_format_compatible": format_compatible,
            "issues": issues,
        }
    ready_candidates = sorted(key for key, value in candidates.items() if value["ready"])
    return {
        "schema_version": 1,
        "backend": "mflux",
        "executable": executable,
        "mflux_version": version,
        "source_root": str(root),
        "source_files_examined": file_count,
        "source_bytes_examined": total_bytes,
        "artifact": artifact,
        "ready": bool(ready_candidates),
        "ready_candidates": ready_candidates,
        "candidates": candidates,
        "imports_backend": False,
        "allocates_model_or_metal": False,
    }


def inspect_mflux_generative_readiness(
    executable: str | Path, *, model: str | Path | None = None
) -> dict[str, object]:
    path = Path(executable).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("MFLUX Python executable is not executable")
    script = (
        "import importlib.metadata as m,json;"
        "d=m.distribution('mflux');"
        "print(json.dumps({'version':d.version,'source_root':str(d.locate_file('mflux'))}))"
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
            raise ValueError("MFLUX metadata probe output is outside the bounded limit")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or set(payload) != {"version", "source_root"}:
            raise ValueError("MFLUX metadata probe output has an invalid schema")
        version, source_root = payload["version"], payload["source_root"]
        if not isinstance(version, str) or not isinstance(source_root, str):
            raise ValueError("MFLUX metadata probe values are invalid")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise ValueError("MFLUX metadata probe failed") from error
    return inspect_mflux_generative_sources(
        source_root,
        version=version,
        executable=str(path.resolve()),
        model=model,
    )
