from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .hardware import default_application_support, detect_hardware
from .types import ContextRecommendation, HardwareInfo, RuntimeProfile
from .version import PROFILE_VERSION, __version__


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

