"""Locally trusted, identity-bound text qualification evidence for startup."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from pathlib import Path

from .architecture_registry import describe_architecture
from .backend_fingerprint import fingerprint_backend
from .model import InspectedModel
from .model_integrity import build_model_integrity_manifest
from .qualification_bundle import _read_report
from .types import HardwareInfo

MAX_AGE_SECONDS = 7 * 24 * 3600


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def capture_identity(model: InspectedModel, executable: Path, backend: str,
                     hardware: HardwareInfo) -> dict[str, object]:
    descriptor = describe_architecture(model.config)
    if descriptor["structure_status"] != "described":
        raise ValueError("architecture evidence requires a described model")
    if backend not in {"mlx_lm", "vllm_metal"}:
        raise ValueError("unsupported evidence backend")
    return {
        "config_sha256": descriptor["config_sha256"],
        "model_root_sha256": build_model_integrity_manifest(model.path)["root_sha256"],
        "backend": backend,
        "backend_sha256": fingerprint_backend(executable),
        "environment_sha256": _digest({
            key: value for key, value in os.environ.items()
            if key.startswith(("MLX_", "VLLM_", "METAL_", "DYLD_", "OMP_", "VECLIB_",
                               "TOKENIZERS_", "PYTORCH_"))
        }),
        "hardware_sha256": _digest({
            "soc": hardware.soc, "os": hardware.os_version,
            "architecture": hardware.architecture, "platform": hardware.platform,
            "memory_bytes": hardware.memory.total_bytes, "gpu_cores": hardware.gpu_core_count,
        }),
        # Daemon/wrapper behavior matters as well as the installed engine.
        "runtime_sha256": _digest({
            str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.rglob("*.py"))
        }),
    }


def _positive(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"invalid architecture evidence {name}")
    return value


def _validate_identity(identity: object) -> None:
    keys = {"config_sha256", "model_root_sha256", "backend_sha256",
            "environment_sha256", "hardware_sha256", "runtime_sha256"}
    if not isinstance(identity, dict) or set(identity) != keys | {"backend"}:
        raise ValueError("invalid architecture evidence identity")
    if not isinstance(identity["backend"], str) or identity["backend"] not in {"mlx_lm", "vllm_metal"}:
        raise ValueError("invalid architecture evidence backend")
    for key in keys:
        value = identity[key]
        if not isinstance(value, str) or len(value) != 64 or any(
            c not in "0123456789abcdef" for c in value
        ):
            raise ValueError("invalid architecture evidence identity digest")


def _passed_checks(report: dict[str, object]) -> None:
    if report.get("passed") is not True or report.get("shutdown_clean") is not True:
        raise ValueError("architecture qualification did not pass cleanly")
    if report.get("requested_modes") != ["text"]:
        raise ValueError("architecture evidence must be text-only")
    for name in ("promotion_probe", "soak", "context_reevaluation"):
        item = report.get(name)
        if not isinstance(item, dict) or item.get("passed") is not True:
            raise ValueError(f"architecture evidence missing {name}")
    checks = report["promotion_probe"].get("checks")
    expected = {"greedy_repeat_equal", "sampled_repeat_equal", "sampled_stream_equal",
                "stream_completed"}
    if not isinstance(checks, dict) or set(checks) != expected or any(
        value is not True for value in checks.values()
    ):
        raise ValueError("architecture evidence requires passing promotion checks")
    soak = report["soak"]
    elapsed = soak.get("elapsed_seconds")
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 1800
            or soak.get("process_alive") is not True or soak.get("stability_window_met") is not True
            or type(soak.get("failures")) is not int or soak["failures"] != 0):
        raise ValueError("architecture evidence requires a clean 30-minute soak")
    _positive(soak.get("successes"), "successes", 100_000_000)
    if type(soak.get("requests")) is not int or soak["requests"] != soak["successes"]:
        raise ValueError("architecture evidence request counts differ")
    rss = soak.get("rss")
    if not isinstance(rss, dict):
        raise ValueError("architecture evidence requires measured RSS")
    for key in ("peak_growth_bytes", "limit_bytes"):
        if type(rss.get(key)) is not int or rss[key] < 0:
            raise ValueError("architecture evidence requires measured RSS")
    if rss["peak_growth_bytes"] > rss["limit_bytes"]:
        raise ValueError("architecture evidence memory regression")
    quality = report.get("quality_smoke")
    if not isinstance(quality, dict) or quality.get("passed") is not True or quality.get(
        "checks"
    ) != {"english": True, "japanese": True, "simplified_chinese": True}:
        raise ValueError("architecture evidence requires multilingual quality")
    if any(value is not True for value in quality["checks"].values()):
        raise ValueError("architecture evidence quality checks must be booleans")
    phase = report.get("phase_profile")
    if not isinstance(phase, dict):
        raise ValueError("architecture evidence requires phase samples")
    if _positive(phase.get("sample_count"), "phase samples", 1000) < 3:
        raise ValueError("architecture evidence requires three phase samples")


def bind_evidence(report: dict[str, object], identity: dict[str, object], *,
                  context_tokens: int, concurrency: int, now: float | None = None) -> dict[str, object]:
    _validate_identity(identity)
    _passed_checks(report)
    context_tokens = _positive(context_tokens, "context", 16_777_216)
    concurrency = _positive(concurrency, "concurrency", 256)
    if report.get("backend") != identity.get("backend"):
        raise ValueError("architecture evidence backend differs")
    if report.get("qualification_limits") != {
        "context_tokens": context_tokens, "concurrency": concurrency,
    }:
        raise ValueError("architecture evidence limits differ from qualification")
    memory = report.get("model_memory_fit")
    context = report["context_reevaluation"]
    if (not isinstance(memory, dict) or memory.get("fits") is not True
            or memory.get("context_tokens") != context_tokens
            or context.get("effective_context_tokens") != context_tokens):
        raise ValueError("architecture evidence context differs from measured configuration")
    binding = {
        "schema_version": 1, "scope": "text_smoke_30min",
        "created_at": time.time() if now is None else now,
        "identity": identity, "context_tokens": context_tokens, "concurrency": concurrency,
        "report_sha256": _digest(report),
    }
    return {**report, "architecture_evidence": binding}


def validate_evidence(report: dict[str, object], identity: dict[str, object], *,
                      context_tokens: int, concurrency: int,
                      now: float | None = None) -> dict[str, object]:
    _validate_identity(identity)
    _passed_checks(report)
    binding = report.get("architecture_evidence")
    if not isinstance(binding, dict) or set(binding) != {
        "schema_version", "scope", "created_at", "identity", "context_tokens",
        "concurrency", "report_sha256",
    }:
        raise ValueError("architecture evidence binding missing or invalid")
    if type(binding["schema_version"]) is not int or binding["schema_version"] != 1 or binding[
        "scope"
    ] != "text_smoke_30min":
        raise ValueError("architecture evidence schema or scope unsupported")
    created = binding["created_at"]
    if type(created) not in (int, float) or not 0 <= created <= 1e12:
        raise ValueError("architecture evidence timestamp invalid")
    age = (time.time() if now is None else now) - created if type(created) in (int, float) else -1
    if not math.isfinite(age) or not 0 <= age <= MAX_AGE_SECONDS:
        raise ValueError("architecture evidence expired or dated in the future")
    if binding["identity"] != identity or report.get("backend") != identity.get("backend"):
        raise ValueError("architecture evidence identity mismatch")
    body = {key: value for key, value in report.items() if key != "architecture_evidence"}
    if binding["report_sha256"] != _digest(body):
        raise ValueError("architecture evidence report digest mismatch")
    allowed_context = _positive(binding["context_tokens"], "context", 16_777_216)
    allowed_concurrency = _positive(binding["concurrency"], "concurrency", 256)
    bind_evidence(body, identity, context_tokens=allowed_context, concurrency=allowed_concurrency,
                  now=created)
    if (_positive(context_tokens, "requested context", 16_777_216) > allowed_context
            or _positive(concurrency, "requested concurrency", 256) > allowed_concurrency):
        raise ValueError("architecture evidence workload exceeds qualified bounds")
    return binding


def verify_startup_evidence(path: Path, model: InspectedModel, executable: Path,
                            backend: str, hardware: HardwareInfo, *,
                            context_tokens: int, concurrency: int) -> dict[str, object]:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077):
        raise ValueError("architecture evidence must be an owner-only regular file")
    report, _ = _read_report(path)
    binding = report.get("architecture_evidence")
    candidate = binding.get("identity") if isinstance(binding, dict) else None
    validate_evidence(report, candidate, context_tokens=context_tokens, concurrency=concurrency)
    identity = capture_identity(model, executable, backend, hardware)
    return validate_evidence(report, identity, context_tokens=context_tokens, concurrency=concurrency)
