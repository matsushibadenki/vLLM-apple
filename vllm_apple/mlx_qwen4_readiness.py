from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path


MAX_PROBE_OUTPUT_BYTES = 16 * 1024
MAX_SOURCE_BYTES = 1024 * 1024

_COMPONENTS = {
    "gated_deltanet": ("gated_delta.py", {"gated_delta_update"}),
    "mixture_of_experts": ("qwen3_next.py", {"Qwen3NextSparseMoeBlock"}),
    "ngram_embedding": ("longcat_flash_ngram.py", {"NgramEmbedding"}),
    "qwen_sparse_attention": ("qwen4_exp.py", {"QwenSparseAttention"}),
    "gated_residual": ("qwen4_exp.py", {"GatedResidual"}),
    "native_long_context": ("qwen4_exp.py", {"Model"}),
}


def _symbols(path: Path) -> frozenset[str]:
    try:
        attributes = path.stat()
    except OSError:
        return frozenset()
    if (
        not path.is_file()
        or path.is_symlink()
        or attributes.st_uid != os.getuid()
        or not 1 <= attributes.st_size <= MAX_SOURCE_BYTES
    ):
        return frozenset()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return frozenset()
    return frozenset(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    )


def inspect_mlx_qwen4_sources(
    models_root: str | Path, *, version: str, executable: str
) -> dict[str, object]:
    root = Path(models_root).expanduser().resolve()
    components: dict[str, dict[str, object]] = {}
    for feature, (filename, required_symbols) in _COMPONENTS.items():
        path = root / filename
        found = _symbols(path)
        components[feature] = {
            "available": required_symbols <= found,
            "source": filename,
            "required_symbols": sorted(required_symbols),
        }
    architecture_file = root / "qwen4_exp.py"
    architecture_symbols = _symbols(architecture_file)
    architecture_registered = "Model" in architecture_symbols
    reusable = sorted(
        name for name, evidence in components.items() if evidence["available"]
    )
    missing = sorted(
        name for name, evidence in components.items() if not evidence["available"]
    )
    return {
        "schema_version": 1,
        "backend": "mlx_lm",
        "executable": executable,
        "mlx_lm_version": version,
        "target_architecture": "qwen4_exp",
        "architecture_registered": architecture_registered,
        "ready": architecture_registered and not missing,
        "reusable_components": reusable,
        "missing_components": missing,
        "components": components,
        "imports_backend": False,
        "allocates_model_or_metal": False,
    }


def inspect_mlx_qwen4_readiness(executable: str | Path) -> dict[str, object]:
    path = Path(executable).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("mlx_lm server is not executable")
    python = path.parent / "python"
    if not python.is_file():
        python = Path(sys.executable)
    script = (
        "import importlib.metadata as m,json;"
        "d=m.distribution('mlx-lm');"
        "print(json.dumps({'version':d.version,'models_root':"
        "str(d.locate_file('mlx_lm/models'))}))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", script],
            capture_output=True,
            check=True,
            text=True,
            timeout=5.0,
        )
        raw = result.stdout.strip()
        if not 1 <= len(raw.encode("utf-8")) <= MAX_PROBE_OUTPUT_BYTES:
            raise ValueError("mlx_lm metadata probe output is outside the bounded limit")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or set(payload) != {"version", "models_root"}:
            raise ValueError("mlx_lm metadata probe output has an invalid schema")
        version = payload["version"]
        models_root = payload["models_root"]
        if not isinstance(version, str) or not isinstance(models_root, str):
            raise ValueError("mlx_lm metadata probe values are invalid")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise ValueError("mlx_lm metadata probe failed") from error
    return inspect_mlx_qwen4_sources(models_root, version=version, executable=str(path.resolve()))
