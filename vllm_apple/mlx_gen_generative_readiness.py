from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .generative_artifact_inspection import inspect_generative_artifact


MINIMUM_MLX_GEN_VERSION = (0, 18, 2)
MAX_PROBE_OUTPUT_BYTES = 16 * 1024


def _version_tuple(value: str) -> tuple[int, int, int] | None:
    core = value.split("+", 1)[0].split("-", 1)[0]
    parts = core.split(".")
    if len(parts) < 3 or any(not part.isdigit() for part in parts[:3]):
        return None
    return tuple(int(part) for part in parts[:3])


def assess_mlx_gen_generative_readiness(
    *,
    executable: str,
    version: str,
    cli_registered: bool,
    model: str | Path,
) -> dict[str, object]:
    artifact = inspect_generative_artifact(model)
    parsed_version = _version_tuple(version)
    issues: list[str] = []
    if parsed_version is None or parsed_version < MINIMUM_MLX_GEN_VERSION:
        issues.append("mlx_gen_version_below_0.18.2")
    if not cli_registered:
        issues.append("mlxgen_console_script_missing")
    if artifact["artifact_format"] != "mlx-gen":
        issues.append(f"unsupported_artifact_format:{artifact['artifact_format']}")
    if artifact.get("base_model") != "black-forest-labs/FLUX.2-klein-base-9B":
        issues.append("unexpected_base_model")
    if artifact.get("quantization", {}).get("bits") != 4:
        issues.append("expected_4bit_quantization")
    return {
        "schema_version": 1,
        "backend": "mlx-gen",
        "executable": executable,
        "mlx_gen_version": version,
        "minimum_version": "0.18.2",
        "cli_registered": cli_registered,
        "artifact": artifact,
        "ready": not issues,
        "issues": issues,
        "imports_backend": False,
        "allocates_model_or_metal": False,
    }


def inspect_mlx_gen_generative_readiness(
    executable: str | Path, *, model: str | Path
) -> dict[str, object]:
    path = Path(executable).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("MLX-Gen Python executable is not executable")
    script = (
        "import importlib.metadata as m,json;"
        "d=m.distribution('mlx-gen');"
        "e=any(x.group=='console_scripts' and x.name=='mlxgen' for x in d.entry_points);"
        "print(json.dumps({'version':d.version,'cli_registered':e}))"
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
            raise ValueError("MLX-Gen metadata probe output is outside the bounded limit")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or set(payload) != {"version", "cli_registered"}:
            raise ValueError("MLX-Gen metadata probe output has an invalid schema")
        if not isinstance(payload["version"], str) or not isinstance(
            payload["cli_registered"], bool
        ):
            raise ValueError("MLX-Gen metadata probe values are invalid")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise ValueError("MLX-Gen metadata probe failed") from error
    return assess_mlx_gen_generative_readiness(
        executable=str(path.resolve()),
        version=payload["version"],
        cli_registered=payload["cli_registered"],
        model=model,
    )
