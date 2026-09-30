"""Bounded observations outside the benchmark measurement window."""
from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone

from .hardware import _parse_power_mode, _pmset, detect_thermal_state


def _process_age_seconds(pid: int | None) -> int | None:
    if pid is None:
        return None
    if type(pid) is not int or pid <= 0:
        raise ValueError("target PID must be a positive integer")
    try:
        result = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "etime="],
            capture_output=True, text=True, check=True, timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    if len(value) > 32:
        return None
    match = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d{1,2}):(\d{2})", value)
    if match is None:
        return None
    days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
    if minutes >= 60 or seconds >= 60 or (days and hours >= 24):
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def observe_benchmark_context(pid: int | None = None) -> dict[str, object]:
    source = _pmset(["-g", "ps"])
    settings = _pmset(["-g", "custom"])
    source_match = re.search(r"Now drawing from '(AC Power|Battery Power)'", source or "")
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "thermal_state": detect_thermal_state().value,
        "power_source": source_match.group(1) if source_match else "unknown",
        "power_mode": _parse_power_mode(source or "", settings or "").value,
        "target_process_age_seconds": _process_age_seconds(pid),
    }
