from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .generative_artifact_inspection import inspect_generative_artifact


MINIMUM_MLX_GEN_VIDEO_VERSION = (0, 33, 1)
MAX_PROBE_OUTPUT_BYTES = 16 * 1024


def _version_tuple(value: str) -> tuple[int, int, int] | None:
    parts = value.split("+", 1)[0].split("-", 1)[0].split(".")
    if len(parts) < 3 or any(not part.isdigit() for part in parts[:3]):
        return None
    return tuple(int(part) for part in parts[:3])


def assess_mlx_gen_video_readiness(
    *, executable: str, version: str, cli_registered: bool, model: str | Path
) -> dict[str, object]:
    artifact = inspect_generative_artifact(model)
    issues: list[str] = []
    parsed = _version_tuple(version)
    if parsed is None or parsed < MINIMUM_MLX_GEN_VIDEO_VERSION:
        issues.append("mlx_gen_version_below_0.33.1")
    if not cli_registered:
        issues.append("mlxgen_console_script_missing")
    if artifact.get("artifact_format") != "mlx-gen":
        issues.append(f"unsupported_artifact_format:{artifact.get('artifact_format')}")
    if artifact.get("pipeline_class") != "WanPipeline":
        issues.append("unexpected_pipeline_class")
    if artifact.get("base_model") != "Wan-AI/Wan2.2-TI2V-5B-Diffusers":
        issues.append("unexpected_base_model")
    if artifact.get("quantization", {}).get("bits") != 8:
        issues.append("expected_mixed_q8_bf16_quantization")
    if not artifact.get("inspectable"):
        issues.append("artifact_not_inspectable")
    return {
        "schema_version": 1,
        "backend": "mlx-gen",
        "candidate_id": "wan2.2-ti2v-5b",
        "executable": executable,
        "mlx_gen_version": version,
        "minimum_version": "0.33.1",
        "cli_registered": cli_registered,
        "artifact": artifact,
        "ready": not issues,
        "issues": issues,
        "imports_backend": False,
        "loads_weights": False,
        "allocates_model_or_metal": False,
    }


def inspect_mlx_gen_video_readiness(
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
            raise ValueError("MLX-Gen video metadata probe output is outside the bounded limit")
        payload = json.loads(raw)
        if (
            not isinstance(payload, dict)
            or set(payload) != {"version", "cli_registered"}
            or not isinstance(payload["version"], str)
            or not isinstance(payload["cli_registered"], bool)
        ):
            raise ValueError("MLX-Gen video metadata probe output has an invalid schema")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise ValueError("MLX-Gen video metadata probe failed") from error
    return assess_mlx_gen_video_readiness(
        executable=str(path.resolve()),
        version=payload["version"],
        cli_registered=payload["cli_registered"],
        model=model,
    )
