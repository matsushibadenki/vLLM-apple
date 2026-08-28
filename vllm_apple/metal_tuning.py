from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .hardware import default_application_support
from .kernel_probe import parse_kernel_probe_result
from .kernel_profile import ModelKernelShapeProfile, PagedAttentionShape
from .metal_probe import (
    MetalShapeTuningDecision,
    MetalThreadConfiguration,
    NativeMetalProbeAdapter,
)

METAL_TUNING_SCHEMA_VERSION = 1
MAX_METAL_TUNING_BYTES = 512 * 1024
MAX_METAL_TUNING_CANDIDATES = 64


@dataclass(frozen=True, slots=True)
class MetalTuningReport:
    schema_version: int
    tuning_id: str
    profile_id: str
    model_id: str
    hardware_fingerprint: str
    environment_fingerprint: str
    created_at_unix_seconds: int
    samples_per_candidate: int
    tie_ratio: float
    decisions: tuple[MetalShapeTuningDecision, ...]

    def __post_init__(self) -> None:
        if self.schema_version != METAL_TUNING_SCHEMA_VERSION:
            raise ValueError("unsupported Metal tuning schema")
        if len(self.tuning_id) != 24 or any(
            character not in "0123456789abcdef" for character in self.tuning_id
        ):
            raise ValueError("invalid Metal tuning ID")
        if len(self.profile_id) != 24 or not self.model_id:
            raise ValueError("invalid Metal tuning model identity")
        if not self.hardware_fingerprint or not self.environment_fingerprint:
            raise ValueError("Metal tuning fingerprints cannot be empty")
        if self.created_at_unix_seconds <= 0 or not 1 <= self.samples_per_candidate <= 32:
            raise ValueError("invalid Metal tuning sampling metadata")
        if self.tie_ratio != 1.02 or not 1 <= len(self.decisions) <= 16:
            raise ValueError("invalid Metal tuning policy")
        if self.tuning_id != _tuning_id(
            self.profile_id,
            self.hardware_fingerprint,
            self.environment_fingerprint,
            self.samples_per_candidate,
            self.decisions,
        ):
            raise ValueError("Metal tuning ID does not match decisions")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tuning_id": self.tuning_id,
            "profile_id": self.profile_id,
            "model_id": self.model_id,
            "hardware_fingerprint": self.hardware_fingerprint,
            "environment_fingerprint": self.environment_fingerprint,
            "created_at_unix_seconds": self.created_at_unix_seconds,
            "samples_per_candidate": self.samples_per_candidate,
            "tie_ratio": self.tie_ratio,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


def tune_metal_shape_profile(
    profile: ModelKernelShapeProfile,
    adapter: NativeMetalProbeAdapter,
    *,
    hardware_fingerprint: str,
    environment_fingerprint: str,
    samples: int = 3,
    maximum_shapes: int = 4,
    clock: Callable[[], float] = time.time,
) -> MetalTuningReport:
    if not 1 <= maximum_shapes <= 16:
        raise ValueError("maximum_shapes must be between 1 and 16")
    decisions = tuple(
        adapter.tune_model_shape(
            shape,
            hardware_fingerprint=hardware_fingerprint,
            environment_fingerprint=environment_fingerprint,
            samples=samples,
        )
        for shape in profile.shapes[:maximum_shapes]
    )
    created_at = int(clock())
    tuning_id = _tuning_id(
        profile.profile_id,
        hardware_fingerprint,
        environment_fingerprint,
        samples,
        decisions,
    )
    return MetalTuningReport(
        METAL_TUNING_SCHEMA_VERSION,
        tuning_id,
        profile.profile_id,
        profile.model_id,
        hardware_fingerprint,
        environment_fingerprint,
        created_at,
        samples,
        1.02,
        decisions,
    )


def default_metal_tuning_path(report: MetalTuningReport) -> Path:
    return (
        default_application_support()
        / "profiles"
        / "metal-tuning"
        / report.hardware_fingerprint
        / report.environment_fingerprint
        / f"{report.profile_id}-{report.tuning_id}.json"
    )


def discover_metal_tuning_report(
    *,
    profile_id: str,
    hardware_fingerprint: str,
    environment_fingerprint: str,
    root: Path | None = None,
) -> MetalTuningReport | None:
    """Load the newest strictly compatible report from a bounded private directory."""
    for value in (hardware_fingerprint, environment_fingerprint):
        if not 1 <= len(value) <= 128 or value in {".", ".."} or Path(value).name != value:
            raise ValueError("invalid Metal tuning fingerprint path component")
    directory = (
        (root if root is not None else default_application_support() / "profiles" / "metal-tuning")
        / hardware_fingerprint
        / environment_fingerprint
    )
    try:
        attributes = directory.lstat()
    except FileNotFoundError:
        return None
    if (
        not stat.S_ISDIR(attributes.st_mode)
        or attributes.st_uid != os.getuid()
        or attributes.st_mode & 0o077
    ):
        raise ValueError("Metal tuning discovery directory must be private")

    prefix = f"{profile_id}-"
    candidates: list[Path] = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.name.startswith(prefix) or not entry.name.endswith(".json"):
                continue
            tuning_id = entry.name[len(prefix) : -5]
            if len(tuning_id) != 24 or any(
                character not in "0123456789abcdef" for character in tuning_id
            ):
                continue
            candidates.append(Path(entry.path))
            if len(candidates) > MAX_METAL_TUNING_CANDIDATES:
                raise ValueError("Metal tuning candidate count exceeded 64")

    valid: list[MetalTuningReport] = []
    for path in sorted(candidates):
        try:
            valid.append(
                load_metal_tuning_report(
                    path,
                    profile_id=profile_id,
                    hardware_fingerprint=hardware_fingerprint,
                    environment_fingerprint=environment_fingerprint,
                )
            )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if not valid:
        return None
    return max(valid, key=lambda report: (report.created_at_unix_seconds, report.tuning_id))


def save_metal_tuning_report(report: MetalTuningReport, path: Path | None = None) -> Path:
    destination = path or default_metal_tuning_path(report)
    encoded = (
        json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_METAL_TUNING_BYTES:
        raise ValueError("Metal tuning report exceeded 512 KiB")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = destination.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise ValueError("Metal tuning directory must be private")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def load_metal_tuning_report(
    path: Path,
    *,
    profile_id: str,
    hardware_fingerprint: str,
    environment_fingerprint: str,
) -> MetalTuningReport:
    attributes = path.lstat()
    if (
        not stat.S_ISREG(attributes.st_mode)
        or attributes.st_uid != os.getuid()
        or attributes.st_mode & 0o077
    ):
        raise ValueError("Metal tuning report must be a private current-user regular file")
    if attributes.st_size > MAX_METAL_TUNING_BYTES:
        raise ValueError("Metal tuning report exceeded 512 KiB")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "schema_version",
        "tuning_id",
        "profile_id",
        "model_id",
        "hardware_fingerprint",
        "environment_fingerprint",
        "created_at_unix_seconds",
        "samples_per_candidate",
        "tie_ratio",
        "decisions",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("invalid Metal tuning report fields")
    if (
        payload["profile_id"] != profile_id
        or payload["hardware_fingerprint"] != hardware_fingerprint
        or payload["environment_fingerprint"] != environment_fingerprint
    ):
        raise ValueError("Metal tuning report identity mismatch")
    values = payload["decisions"]
    if not isinstance(values, list) or not 1 <= len(values) <= 16:
        raise ValueError("invalid Metal tuning decision count")
    try:
        return MetalTuningReport(
            payload["schema_version"],
            payload["tuning_id"],
            payload["profile_id"],
            payload["model_id"],
            payload["hardware_fingerprint"],
            payload["environment_fingerprint"],
            payload["created_at_unix_seconds"],
            payload["samples_per_candidate"],
            payload["tie_ratio"],
            tuple(_parse_decision(value) for value in values),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid Metal tuning report values") from error


def _parse_decision(value: object) -> MetalShapeTuningDecision:
    if not isinstance(value, dict) or set(value) != {"shape", "winner", "candidates"}:
        raise ValueError("invalid Metal tuning decision fields")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 4:
        raise ValueError("invalid Metal tuning candidates")
    parsed = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {"configuration", "result"}:
            raise ValueError("invalid Metal tuning candidate fields")
        parsed.append(
            (
                MetalThreadConfiguration(**candidate["configuration"]),
                parse_kernel_probe_result(candidate["result"]),
            )
        )
    return MetalShapeTuningDecision(
        PagedAttentionShape(**value["shape"]),
        MetalThreadConfiguration(**value["winner"]),
        tuple(parsed),
    )


def _tuning_id(
    profile_id: str,
    hardware_fingerprint: str,
    environment_fingerprint: str,
    samples: int,
    decisions: tuple[MetalShapeTuningDecision, ...],
) -> str:
    identity = {
        "schema_version": METAL_TUNING_SCHEMA_VERSION,
        "profile_id": profile_id,
        "hardware_fingerprint": hardware_fingerprint,
        "environment_fingerprint": environment_fingerprint,
        "samples_per_candidate": samples,
        "winners": [
            {
                "shape": decision.to_dict()["shape"],
                "configuration": decision.winner.to_dict(),
            }
            for decision in decisions
        ],
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
