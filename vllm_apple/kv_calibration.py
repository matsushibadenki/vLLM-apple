from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .hardware import default_application_support
from .types import ModelMemorySpec

MAX_CALIBRATION_REPORT_BYTES = 1024 * 1024
MINIMUM_CALIBRATION_STAGES = 3
MINIMUM_CALIBRATED_CONTEXT = 4096
MAX_CALIBRATION_REPORTS = 256


@dataclass(frozen=True, slots=True)
class KVCalibration:
    model_id: str
    backend: str
    observed_bytes_per_token: int
    calibrated_bytes_per_token: int
    maximum_observed_context: int
    sample_count: int
    safety_margin_ratio: float
    evaluation_id: str

    def apply(self, model: ModelMemorySpec) -> ModelMemorySpec:
        if model.model_id != self.model_id:
            raise ValueError("calibration model identity does not match")
        maximum = self.maximum_observed_context
        if model.model_max_context is not None:
            maximum = min(maximum, model.model_max_context)
        return replace(
            model,
            kv_bytes_per_token=self.calibrated_bytes_per_token,
            model_max_context=maximum,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "observed_bytes_per_token": self.observed_bytes_per_token,
            "calibrated_bytes_per_token": self.calibrated_bytes_per_token,
            "maximum_observed_context": self.maximum_observed_context,
            "sample_count": self.sample_count,
            "safety_margin_ratio": self.safety_margin_ratio,
            "evaluation_id": self.evaluation_id,
        }


def load_kv_calibration(
    path: Path,
    *,
    expected_model_id: str,
    expected_hardware_fingerprint: str | None = None,
    expected_backend: str | None = None,
    safety_margin_ratio: float = 0.25,
) -> KVCalibration:
    if not 0 <= safety_margin_ratio <= 1:
        raise ValueError("calibration safety margin must be in [0, 1]")
    try:
        attributes = path.lstat()
        if (
            not stat.S_ISREG(attributes.st_mode)
            or attributes.st_uid != os.getuid()
            or attributes.st_mode & 0o077
        ):
            raise ValueError(
                "long-context report must be a private current-user regular file"
            )
        size = attributes.st_size
        if not 0 < size <= MAX_CALIBRATION_REPORT_BYTES:
            raise ValueError("long-context report size is invalid")
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("long-context report is not readable JSON") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("long-context report schema is unsupported")
    if payload.get("model_id") != expected_model_id:
        raise ValueError("calibration model identity does not match")
    backend = payload.get("backend")
    if not isinstance(backend, str) or not backend:
        raise ValueError("calibration backend identity is invalid")
    if expected_backend is not None and backend != expected_backend:
        raise ValueError("calibration backend identity does not match")
    hardware_fingerprint = payload.get("hardware_fingerprint")
    if not isinstance(hardware_fingerprint, str) or not hardware_fingerprint:
        raise ValueError("calibration hardware fingerprint is invalid")
    if (
        expected_hardware_fingerprint is not None
        and hardware_fingerprint != expected_hardware_fingerprint
    ):
        raise ValueError("calibration hardware fingerprint does not match")
    evaluation_id = payload.get("evaluation_id")
    stages = payload.get("stages")
    if not isinstance(evaluation_id, str) or len(evaluation_id) != 24:
        raise ValueError("long-context evaluation identity is invalid")
    if not isinstance(stages, list) or len(stages) > 16:
        raise ValueError("long-context calibration stages are invalid")

    observations: list[tuple[int, int, int]] = []
    for stage in stages:
        if not isinstance(stage, dict) or stage.get("status") != "passed":
            continue
        tokens = stage.get("actual_prompt_tokens")
        target = stage.get("target_tokens")
        state_bytes = stage.get("state_bytes")
        retrieval = stage.get("retrieval_score")
        if (
            not isinstance(tokens, int)
            or isinstance(tokens, bool)
            or tokens <= 0
            or not isinstance(target, int)
            or isinstance(target, bool)
            or target <= 0
            or not isinstance(state_bytes, int)
            or isinstance(state_bytes, bool)
            or state_bytes <= 0
            or retrieval != 1.0
        ):
            raise ValueError("passed calibration stage is incomplete")
        observations.append((target, tokens, state_bytes))
    if len(observations) < MINIMUM_CALIBRATION_STAGES:
        raise ValueError("at least three passed calibration stages are required")
    if any(
        next_target <= target or next_tokens <= tokens or next_bytes < state_bytes
        for (target, tokens, state_bytes), (next_target, next_tokens, next_bytes) in zip(
            observations, observations[1:]
        )
    ):
        raise ValueError("calibration observations must increase monotonically")
    maximum_target, maximum_context, _ = observations[-1]
    if maximum_target < MINIMUM_CALIBRATED_CONTEXT:
        raise ValueError("calibration has not reached 4096 tokens")
    observed = max(
        math.ceil(state_bytes / tokens) for _, tokens, state_bytes in observations
    )
    calibrated = math.ceil(observed * (1 + safety_margin_ratio))
    return KVCalibration(
        model_id=expected_model_id,
        backend=backend,
        observed_bytes_per_token=observed,
        calibrated_bytes_per_token=calibrated,
        maximum_observed_context=maximum_context,
        sample_count=len(observations),
        safety_margin_ratio=safety_margin_ratio,
        evaluation_id=evaluation_id,
    )


def calibration_report_directory(
    model_id: str,
    hardware_fingerprint: str,
    backend: str,
    *,
    application_support: Path | None = None,
) -> Path:
    if not model_id or not hardware_fingerprint or not backend:
        raise ValueError("calibration identity must not be empty")
    root = application_support or default_application_support()
    hardware_key = hashlib.sha256(hardware_fingerprint.encode()).hexdigest()[:16]
    model_key = hashlib.sha256(model_id.encode()).hexdigest()[:16]
    backend_key = hashlib.sha256(backend.encode()).hexdigest()[:16]
    return root / "calibration-reports" / hardware_key / backend_key / model_key


def default_calibration_report_path(
    model_id: str,
    hardware_fingerprint: str,
    backend: str,
    evaluation_id: str,
    *,
    application_support: Path | None = None,
    timestamp_ns: int | None = None,
) -> Path:
    if len(evaluation_id) != 24:
        raise ValueError("long-context evaluation identity is invalid")
    timestamp = time.time_ns() if timestamp_ns is None else timestamp_ns
    if timestamp <= 0:
        raise ValueError("calibration timestamp must be positive")
    return calibration_report_directory(
        model_id,
        hardware_fingerprint,
        backend,
        application_support=application_support,
    ) / f"{timestamp}-{evaluation_id}.json"


def discover_latest_kv_calibration(
    *,
    expected_model_id: str,
    expected_hardware_fingerprint: str,
    expected_backend: str,
    application_support: Path | None = None,
    safety_margin_ratio: float = 0.25,
) -> tuple[KVCalibration, Path]:
    directory = calibration_report_directory(
        expected_model_id,
        expected_hardware_fingerprint,
        expected_backend,
        application_support=application_support,
    )
    try:
        parent = directory.lstat()
    except FileNotFoundError as error:
        raise ValueError("no calibration reports were found") from error
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or parent.st_mode & 0o077
    ):
        raise ValueError("calibration report directory must be private")
    candidates: list[tuple[int, Path]] = []
    with os.scandir(directory) as entries:
        for index, entry in enumerate(entries):
            if index >= MAX_CALIBRATION_REPORTS:
                raise ValueError("calibration report directory contains too many entries")
            if not entry.name.endswith(".json"):
                continue
            try:
                attributes = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISREG(attributes.st_mode):
                candidates.append((attributes.st_mtime_ns, Path(entry.path)))
    for _, path in sorted(candidates, reverse=True):
        try:
            calibration = load_kv_calibration(
                path,
                expected_model_id=expected_model_id,
                expected_hardware_fingerprint=expected_hardware_fingerprint,
                expected_backend=expected_backend,
                safety_margin_ratio=safety_margin_ratio,
            )
        except ValueError:
            continue
        return calibration, path
    raise ValueError("no compatible calibration report was found")
