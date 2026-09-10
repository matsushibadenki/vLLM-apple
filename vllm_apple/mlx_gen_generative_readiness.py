from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

from .generative_artifact_inspection import inspect_generative_artifact
from .generative_weight_residency import current_mlx_gen_weight_residency_feasibility


MINIMUM_MLX_GEN_VERSION = (0, 18, 2)
MINIMUM_Z_IMAGE_VERSION = (0, 33, 1)
MAX_PROBE_OUTPUT_BYTES = 16 * 1024
FLUX2_LOW_CACHE_LIMIT_GB = 0.25


def select_mlx_gen_qualification_candidate(
    candidate_id: str,
    cache_limit_gb: float | None,
    *,
    blockwise_residency: bool = False,
    attention_query_chunk_size: int | None = None,
    mlp_sequence_chunk_size: int | None = None,
) -> str:
    if mlp_sequence_chunk_size is not None:
        if (
            mlp_sequence_chunk_size != 512
            or cache_limit_gb is not None
            or blockwise_residency
            or attention_query_chunk_size is not None
            or candidate_id != "flux2-klein-9b-base"
        ):
            raise ValueError(
                "formal MLX-Gen MLP chunk qualification supports only FLUX.2 Klein "
                "with an exclusive --mlp-sequence-chunk-size 512 profile"
            )
        return "flux2-klein-9b-base-mlp-chunked"
    if attention_query_chunk_size is not None:
        if (
            attention_query_chunk_size != 512
            or cache_limit_gb is not None
            or blockwise_residency
            or candidate_id != "flux2-klein-9b-base"
        ):
            raise ValueError(
                "formal MLX-Gen attention chunk qualification supports only FLUX.2 Klein "
                "with an exclusive --attention-query-chunk-size 512 profile"
            )
        return "flux2-klein-9b-base-attention-chunked"
    if blockwise_residency:
        if cache_limit_gb is not None or candidate_id != "flux2-klein-9b-base":
            raise ValueError(
                "formal MLX-Gen blockwise qualification supports only FLUX.2 Klein "
                "and cannot be combined with the low-cache profile"
            )
        return "flux2-klein-9b-base-blockwise"
    if cache_limit_gb is None:
        return candidate_id
    if (
        candidate_id != "flux2-klein-9b-base"
        or not math.isfinite(cache_limit_gb)
        or cache_limit_gb != FLUX2_LOW_CACHE_LIMIT_GB
    ):
        raise ValueError(
            "formal MLX-Gen low-cache qualification supports only "
            "FLUX.2 Klein with --mlx-cache-limit-gb 0.25"
        )
    return "flux2-klein-9b-base-low-cache"


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
    z_image_cli_registered: bool = False,
) -> dict[str, object]:
    artifact = inspect_generative_artifact(model)
    parsed_version = _version_tuple(version)
    is_z_image = (
        artifact.get("pipeline_class") == "ZImagePipeline"
        or artifact.get("base_model") == "Tongyi-MAI/Z-Image-Turbo"
    )
    candidate_id = "z-image-turbo-mlx-4bit" if is_z_image else "flux2-klein-9b-base"
    minimum_version = MINIMUM_Z_IMAGE_VERSION if is_z_image else MINIMUM_MLX_GEN_VERSION
    issues: list[str] = []
    if parsed_version is None or parsed_version < minimum_version:
        issues.append(
            "mlx_gen_version_below_0.33.1"
            if is_z_image
            else "mlx_gen_version_below_0.18.2"
        )
    if not cli_registered:
        issues.append("mlxgen_console_script_missing")
    if is_z_image:
        if not z_image_cli_registered:
            issues.append("z_image_turbo_console_script_missing")
        if artifact["artifact_format"] != "mlx-gen":
            issues.append(f"unsupported_artifact_format:{artifact['artifact_format']}")
        if artifact.get("base_model") != "Tongyi-MAI/Z-Image-Turbo":
            issues.append("unexpected_base_model")
        # Native MLX-Gen packages are selected by their model-card base_model and
        # do not contain a Diffusers model_index.json. If one is present, keep
        # validating it so a mismatched conversion cannot pass as a native package.
        if artifact.get("pipeline_class") not in {None, "ZImagePipeline"}:
            issues.append("unexpected_pipeline_class")
    else:
        if artifact["artifact_format"] != "mlx-gen":
            issues.append(f"unsupported_artifact_format:{artifact['artifact_format']}")
        if artifact.get("base_model") != "black-forest-labs/FLUX.2-klein-base-9B":
            issues.append("unexpected_base_model")
    if artifact.get("quantization", {}).get("bits") != 4:
        issues.append("expected_4bit_quantization")
    return {
        "schema_version": 1,
        "backend": "mlx-gen",
        "candidate_id": candidate_id,
        "executable": executable,
        "mlx_gen_version": version,
        "minimum_version": ".".join(str(part) for part in minimum_version),
        "cli_registered": cli_registered,
        "z_image_cli_registered": z_image_cli_registered,
        "artifact": artifact,
        "ready": not issues,
        "issues": issues,
        "imports_backend": False,
        "allocates_model_or_metal": False,
        "weight_block_residency": current_mlx_gen_weight_residency_feasibility().to_dict(),
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
        "z=any(x.group=='console_scripts' and x.name=='mflux-generate-z-image-turbo' "
        "for x in d.entry_points);"
        "print(json.dumps({'version':d.version,'cli_registered':e,'z_image_cli_registered':z}))"
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
        if not isinstance(payload, dict) or set(payload) != {
            "version",
            "cli_registered",
            "z_image_cli_registered",
        }:
            raise ValueError("MLX-Gen metadata probe output has an invalid schema")
        if not isinstance(payload["version"], str) or any(
            not isinstance(payload[key], bool)
            for key in ("cli_registered", "z_image_cli_registered")
        ):
            raise ValueError("MLX-Gen metadata probe values are invalid")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise ValueError("MLX-Gen metadata probe failed") from error
    return assess_mlx_gen_generative_readiness(
        executable=str(path.resolve()),
        version=payload["version"],
        cli_registered=payload["cli_registered"],
        z_image_cli_registered=payload["z_image_cli_registered"],
        model=model,
    )
