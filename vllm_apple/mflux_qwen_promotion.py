"""Strict evidence gate for staged Qwen image resolution promotion."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

MAX_REPORT_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class MFluxQwenPromotion:
    baseline_size: int
    target_size: int
    minimum_available_bytes: int
    report_sha256: str


def load_mflux_qwen_promotion(
    path: Path,
    *,
    artifact_root_sha256: str,
    target_size: int,
) -> MFluxQwenPromotion:
    """Validate an all-normal staged 20-step report before one-axis promotion."""
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise ValueError("Qwen promotion report must not be a symlink")
    report_path = unresolved.resolve(strict=True)
    info = report_path.stat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or not 1 <= info.st_size <= MAX_REPORT_BYTES
    ):
        raise ValueError("Qwen promotion report must be a bounded owner file")
    raw = report_path.read_bytes()
    after = report_path.stat()
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("Qwen promotion report changed while reading")
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("Qwen promotion report is invalid JSON") from error
    transformer = report.get("transformer") if isinstance(report, dict) else None
    steps = transformer.get("step_reports") if isinstance(transformer, dict) else None
    baseline_size = report.get("width") if isinstance(report, dict) else None
    if (
        report.get("schema_version") != 1
        or report.get("scope") != "mflux_qwen_image_staged_real_prompt_streamed_denoising"
        or report.get("candidate_id") != "qwen-image-2512"
        or report.get("artifact_root_sha256") != artifact_root_sha256
        or report.get("passed") is not True
        or report.get("uses_disjoint_encoder_processes") is not True
        or report.get("uses_real_prompt") is not True
        or report.get("uses_true_cfg") is not True
        or report.get("guidance") != 4.0
        or report.get("denoise_steps") != 20
        or report.get("height") != baseline_size
        or type(baseline_size) is not int
        or baseline_size < 128
        or report.get("child_consumed_handoff_cleanup_verified") is not True
        or report.get("private_handoff_cleanup_verified") is not True
        or report.get("final_memory_pressure") != "normal"
        or report.get("final_thermal_state") not in {"nominal", "fair"}
        or not isinstance(transformer, dict)
        or transformer.get("passed") is not True
        or transformer.get("width") != baseline_size
        or transformer.get("height") != baseline_size
        or transformer.get("steps") != 20
        or transformer.get("uses_real_prompt") is not True
        or transformer.get("uses_true_cfg") is not True
        or transformer.get("final_memory_pressure") != "normal"
        or not isinstance(steps, list)
        or len(steps) != 20
        or any(
            not isinstance(step, dict)
            or step.get("completed_blocks") != 120
            or step.get("latent_finite") is not True
            or step.get("memory_pressure") != "normal"
            or step.get("thermal_state") not in {"nominal", "fair"}
            for step in steps
        )
        or type(target_size) is not int
        or target_size % 16
        or not baseline_size < target_size <= baseline_size * 2
    ):
        raise ValueError("Qwen promotion report does not satisfy the strict contract")
    peak_mlx = transformer.get("peak_mlx_bytes")
    peak_rss = transformer.get("peak_process_rss_bytes")
    encoder_rss = report.get("encoder_peak_process_rss_bytes")
    if any(type(value) is not int or value <= 0 for value in (peak_mlx, peak_rss, encoder_rss)):
        raise ValueError("Qwen promotion report lacks bounded memory evidence")
    minimum = max(10_000_000_000, peak_mlx * 4, peak_rss * 4, encoder_rss * 2)
    if minimum > 20_000_000_000:
        raise ValueError("Qwen promotion evidence requires an unsafe memory ceiling")
    return MFluxQwenPromotion(
        baseline_size=baseline_size,
        target_size=target_size,
        minimum_available_bytes=minimum,
        report_sha256=hashlib.sha256(raw).hexdigest(),
    )
