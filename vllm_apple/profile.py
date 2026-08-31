from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .hardware import default_application_support, detect_hardware
from .types import (
    ContextRecommendation,
    ContextTier,
    HardwareInfo,
    MemoryInfo,
    MemoryPressure,
    RuntimeProfile,
)
from .version import PROFILE_VERSION, __version__

MAXIMUM_PROFILE_BYTES = 1024 * 1024
PROFILE_CACHE_KEY_LENGTH = 24
MAXIMUM_CACHED_PROFILES = 128
MAXIMUM_CACHE_SCAN_ENTRIES = 512


def build_profile(
    hardware: HardwareInfo | None = None,
    context: ContextRecommendation | None = None,
) -> RuntimeProfile:
    detected = hardware or detect_hardware()
    capabilities = ["control-api", "memory-aware-context", "basic-scheduler"]
    if detected.is_apple_silicon:
        capabilities.extend(("apple-silicon", "unified-memory"))
    return RuntimeProfile(
        profile_version=PROFILE_VERSION,
        runtime_version=__version__,
        created_at=datetime.now(timezone.utc).isoformat(),
        hardware=detected,
        context=context,
        capabilities=tuple(capabilities),
    )


def default_profile_path(profile: RuntimeProfile) -> Path:
    safe_soc = "".join(c if c.isalnum() else "-" for c in profile.hardware.soc).strip("-")
    name = f"{safe_soc or 'unknown'}-{profile.hardware.memory.total_bytes}.json"
    return default_application_support() / "profiles" / name


def profile_cache_key(hardware: HardwareInfo, model_id: str) -> str:
    bounded_model = _bounded_string(model_id, "model ID")
    identity = {
        "architecture": hardware.architecture,
        "gpu_core_count": hardware.gpu_core_count,
        "is_apple_silicon": hardware.is_apple_silicon,
        "model_id": bounded_model,
        "os_version": hardware.os_version,
        "platform": hardware.platform,
        "soc": hardware.soc,
        "total_memory_bytes": hardware.memory.total_bytes,
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:PROFILE_CACHE_KEY_LENGTH]


def default_profile_cache_directory() -> Path:
    return default_application_support() / "profiles" / "model-cache-v1"


def save_cached_profile(profile: RuntimeProfile, directory: Path | None = None) -> Path:
    if profile.context is None:
        raise ValueError("model-specific profile cache requires context")
    cache_directory = (directory or default_profile_cache_directory()).expanduser()
    cache_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not cache_directory.is_dir() or cache_directory.is_symlink():
        raise ValueError("unsafe runtime profile cache directory")
    cache_directory.chmod(0o700)
    key = profile_cache_key(profile.hardware, profile.context.model_id)
    saved = save_profile(profile, cache_directory / f"{key}.json")
    _prune_profile_cache(cache_directory)
    return saved


def _prune_profile_cache(directory: Path) -> None:
    eligible: list[tuple[int, Path]] = []
    try:
        iterator = os.scandir(directory)
    except OSError:
        return
    with iterator:
        for index, entry in enumerate(iterator):
            if index >= MAXIMUM_CACHE_SCAN_ENTRIES:
                break
            stem, suffix = os.path.splitext(entry.name)
            if (
                suffix != ".json"
                or len(stem) != PROFILE_CACHE_KEY_LENGTH
                or any(character not in "0123456789abcdef" for character in stem)
            ):
                continue
            try:
                details = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISREG(details.st_mode) and details.st_uid == os.getuid():
                eligible.append((details.st_mtime_ns, Path(entry.path)))
    eligible.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    for _, candidate in eligible[MAXIMUM_CACHED_PROFILES:]:
        try:
            candidate.unlink()
        except OSError:
            pass


def load_cached_profile(
    hardware: HardwareInfo,
    model_id: str,
    directory: Path | None = None,
) -> RuntimeProfile | None:
    cache_directory = (directory or default_profile_cache_directory()).expanduser()
    key = profile_cache_key(hardware, model_id)
    candidate = cache_directory / f"{key}.json"
    if not candidate.exists():
        return None
    profile = load_profile(candidate)
    if (
        profile.context is None
        or profile.context.model_id != model_id
        or profile_cache_key(profile.hardware, profile.context.model_id) != key
    ):
        raise ValueError("runtime profile cache identity mismatch")
    return profile


def save_profile(profile: RuntimeProfile, path: Path | None = None) -> Path:
    destination = path or default_profile_path(profile)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(profile.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
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


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"invalid runtime profile {name}")
    return value


def _bounded_string(value: object, name: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"invalid runtime profile {name}")
    return value


def _migrate_profile(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("runtime profile must be an object")
    version = payload.get("profile_version")
    if version == PROFILE_VERSION:
        return payload
    if version == 0:
        legacy_fields = {"profile_version", "runtime_version", "created_at", "hardware", "context"}
        if set(payload) != legacy_fields:
            raise ValueError("invalid legacy runtime profile fields")
        return {**payload, "profile_version": PROFILE_VERSION, "capabilities": [], "metadata": {}}
    raise ValueError("unsupported runtime profile version")


def load_profile(path: Path, *, maximum_bytes: int = MAXIMUM_PROFILE_BYTES) -> RuntimeProfile:
    candidate = path.expanduser()
    try:
        details = candidate.lstat()
    except OSError as error:
        raise ValueError("runtime profile is unavailable") from error
    if (
        maximum_bytes <= 0
        or not stat.S_ISREG(details.st_mode)
        or details.st_uid != os.getuid()
        or details.st_size <= 0
        or details.st_size > maximum_bytes
        or stat.S_IMODE(details.st_mode) & 0o077
    ):
        raise ValueError("unsafe runtime profile file")
    try:
        raw = candidate.read_bytes()
        if len(raw) > maximum_bytes:
            raise ValueError("runtime profile exceeds size limit")
        payload = _migrate_profile(json.loads(raw))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid runtime profile JSON") from error

    required = {
        "profile_version",
        "runtime_version",
        "created_at",
        "hardware",
        "context",
        "capabilities",
        "metadata",
    }
    if set(payload) != required:
        raise ValueError("invalid runtime profile fields")
    hardware_payload = payload["hardware"]
    hardware_fields = {
        "platform",
        "architecture",
        "soc",
        "physical_cpu_count",
        "logical_cpu_count",
        "gpu_core_count",
        "memory",
        "is_apple_silicon",
        "os_version",
    }
    if not isinstance(hardware_payload, dict) or set(hardware_payload) != hardware_fields:
        raise ValueError("invalid runtime profile hardware")
    memory_payload = hardware_payload["memory"]
    memory_fields = {
        "total_bytes",
        "available_bytes",
        "process_resident_bytes",
        "pressure",
        "source",
    }
    if not isinstance(memory_payload, dict) or set(memory_payload) != memory_fields:
        raise ValueError("invalid runtime profile memory")
    total_bytes = _positive_integer(memory_payload["total_bytes"], "total memory")
    available_bytes = memory_payload["available_bytes"]
    resident_bytes = memory_payload["process_resident_bytes"]
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (available_bytes, resident_bytes)
    ):
        raise ValueError("invalid runtime profile memory counters")
    memory = MemoryInfo(
        total_bytes=total_bytes,
        available_bytes=available_bytes,
        process_resident_bytes=resident_bytes,
        pressure=MemoryPressure(memory_payload["pressure"]),
        source=_bounded_string(memory_payload["source"], "memory source", maximum=256),
    )
    gpu_cores = hardware_payload["gpu_core_count"]
    if gpu_cores is not None:
        gpu_cores = _positive_integer(gpu_cores, "GPU core count")
    apple_silicon = hardware_payload["is_apple_silicon"]
    if not isinstance(apple_silicon, bool):
        raise ValueError("invalid runtime profile Apple Silicon flag")
    hardware = HardwareInfo(
        platform=_bounded_string(hardware_payload["platform"], "platform", maximum=128),
        architecture=_bounded_string(hardware_payload["architecture"], "architecture", maximum=128),
        soc=_bounded_string(hardware_payload["soc"], "SoC", maximum=256),
        physical_cpu_count=_positive_integer(hardware_payload["physical_cpu_count"], "CPU count"),
        logical_cpu_count=_positive_integer(hardware_payload["logical_cpu_count"], "CPU count"),
        gpu_core_count=gpu_cores,
        memory=memory,
        is_apple_silicon=apple_silicon,
        os_version=_bounded_string(hardware_payload["os_version"], "OS version", maximum=128),
    )
    context_payload = payload["context"]
    context = None
    if context_payload is not None:
        context_fields = {
            "model_id",
            "allocatable_bytes",
            "os_reserve_bytes",
            "safety_headroom_bytes",
            "workspace_bytes",
            "tiers",
            "limiting_factor",
        }
        if not isinstance(context_payload, dict) or set(context_payload) != context_fields:
            raise ValueError("invalid runtime profile context")
        tiers_payload = context_payload["tiers"]
        if not isinstance(tiers_payload, list) or len(tiers_payload) > 16:
            raise ValueError("invalid runtime profile context tiers")
        tiers: list[ContextTier] = []
        for tier in tiers_payload:
            if not isinstance(tier, dict) or set(tier) != {"name", "max_tokens", "kv_budget_bytes"}:
                raise ValueError("invalid runtime profile context tier")
            tiers.append(
                ContextTier(
                    _bounded_string(tier["name"], "tier name", maximum=64),
                    _positive_integer(tier["max_tokens"], "tier tokens"),
                    _positive_integer(tier["kv_budget_bytes"], "tier KV budget"),
                )
            )
        counters = tuple(
            context_payload[name]
            for name in ("allocatable_bytes", "os_reserve_bytes", "safety_headroom_bytes", "workspace_bytes")
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counters):
            raise ValueError("invalid runtime profile context counters")
        limiting_factor = context_payload["limiting_factor"]
        if limiting_factor is not None:
            limiting_factor = _bounded_string(limiting_factor, "limiting factor", maximum=256)
        context = ContextRecommendation(
            model_id=_bounded_string(context_payload["model_id"], "model ID"),
            allocatable_bytes=counters[0],
            os_reserve_bytes=counters[1],
            safety_headroom_bytes=counters[2],
            workspace_bytes=counters[3],
            tiers=tuple(tiers),
            limiting_factor=limiting_factor,
        )
    capabilities_payload = payload["capabilities"]
    metadata = payload["metadata"]
    if (
        not isinstance(capabilities_payload, list)
        or len(capabilities_payload) > 64
        or any(not isinstance(item, str) or not item or len(item) > 128 for item in capabilities_payload)
        or not isinstance(metadata, dict)
    ):
        raise ValueError("invalid runtime profile extensions")
    return RuntimeProfile(
        profile_version=PROFILE_VERSION,
        runtime_version=_bounded_string(payload["runtime_version"], "runtime version", maximum=128),
        created_at=_bounded_string(payload["created_at"], "created timestamp", maximum=128),
        hardware=hardware,
        context=context,
        capabilities=tuple(capabilities_payload),
        metadata=metadata,
    )
