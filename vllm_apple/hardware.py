from __future__ import annotations

import os
import platform
import re
import resource
import subprocess
from pathlib import Path

from .types import HardwareInfo, MemoryInfo, MemoryPressure


def _sysctl(name: str) -> str | None:
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            capture_output=True,
            check=True,
            text=True,
            timeout=1.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _positive_int(value: str | None) -> int | None:
    try:
        parsed = int(value or "")
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _total_memory() -> tuple[int, str]:
    sysctl_value = _positive_int(_sysctl("hw.memsize"))
    if sysctl_value:
        return sysctl_value, "sysctl"
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size, "sysconf"
    except (ValueError, OSError):
        pass
    raise RuntimeError("unable to determine physical memory")


def _vm_stat_available(total_bytes: int) -> int | None:
    try:
        result = subprocess.run(
            ["/usr/bin/vm_stat"], capture_output=True, check=True, text=True, timeout=1.0
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    page_match = re.search(r"page size of (\d+) bytes", result.stdout)
    if not page_match:
        return None
    page_size = int(page_match.group(1))
    counters: dict[str, int] = {}
    for line in result.stdout.splitlines():
        match = re.match(r"([^:]+):\s+(\d+)\.?", line)
        if match:
            counters[match.group(1)] = int(match.group(2))
    available_pages = sum(
        counters.get(key, 0)
        for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    )
    if available_pages <= 0:
        return None
    return min(total_bytes, available_pages * page_size)


def _process_resident_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if platform.system() == "Darwin" else usage * 1024)


def _pressure(total_bytes: int, available_bytes: int) -> MemoryPressure:
    ratio = available_bytes / total_bytes
    if ratio < 0.08:
        return MemoryPressure.CRITICAL
    if ratio < 0.18:
        return MemoryPressure.WARNING
    return MemoryPressure.NORMAL


def detect_memory() -> MemoryInfo:
    total, source = _total_memory()
    available = _vm_stat_available(total)
    if available is None:
        # Never assume an otherwise unobservable machine is completely idle.
        # A conservative fallback is slower but avoids planning directly into
        # swap or jetsam-like pressure when vm_stat is temporarily unavailable.
        available = total // 2
        source += "+available-conservative-estimate"
    return MemoryInfo(
        total_bytes=total,
        available_bytes=available,
        process_resident_bytes=_process_resident_bytes(),
        pressure=_pressure(total, available),
        source=source,
    )


def detect_hardware() -> HardwareInfo:
    machine = platform.machine().lower()
    system = platform.system()
    physical = _positive_int(_sysctl("hw.physicalcpu")) or (os.cpu_count() or 1)
    logical = _positive_int(_sysctl("hw.logicalcpu")) or (os.cpu_count() or physical)
    soc = _sysctl("machdep.cpu.brand_string") or platform.processor() or "Unknown Apple SoC"
    gpu_cores = _positive_int(_sysctl("hw.perflevel0.gpu_count"))
    return HardwareInfo(
        platform=system,
        architecture=machine,
        soc=soc,
        physical_cpu_count=physical,
        logical_cpu_count=logical,
        gpu_core_count=gpu_cores,
        memory=detect_memory(),
        is_apple_silicon=system == "Darwin" and machine == "arm64",
        os_version=platform.mac_ver()[0] if system == "Darwin" else platform.release(),
    )


def default_application_support() -> Path:
    return Path.home() / "Library" / "Application Support" / "vllm-apple"
