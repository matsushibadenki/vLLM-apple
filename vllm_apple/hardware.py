from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import re
import resource
import subprocess
from pathlib import Path

from .types import HardwareInfo, MemoryInfo, MemoryPressure, PowerMode, ThermalState


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


def _ioreg_gpu_core_count() -> int | None:
    """Read Apple GPU cores when the model-specific sysctl is unavailable."""
    try:
        result = subprocess.run(
            ["/usr/sbin/ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"],
            capture_output=True,
            check=True,
            text=True,
            timeout=1.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    match = re.search(r'"gpu-core-count"\s*=\s*(\d+)', result.stdout)
    if match:
        return _positive_int(match.group(1))
    data_match = re.search(r'"gpu-core-count"\s*=\s*<([0-9a-fA-F]{8})>', result.stdout)
    if not data_match:
        return None
    raw = bytes.fromhex(data_match.group(1))
    value = int.from_bytes(raw, byteorder="little", signed=False)
    return value if 0 < value <= 512 else None


def _system_profiler_chip() -> str | None:
    """Read the Apple chip name only when the fast sysctl probe is unavailable."""
    try:
        result = subprocess.run(
            ["/usr/sbin/system_profiler", "SPHardwareDataType", "-detailLevel", "mini"],
            capture_output=True,
            check=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if len(result.stdout.encode("utf-8")) > 64 * 1024:
        return None
    match = re.search(r"^\s*(?:Chip|チップ|芯片):\s*(.{1,128})$", result.stdout, re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip()
    return value if value.startswith("Apple ") else None


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


def _ns_process_info_integer(selector_name: bytes) -> int | None:
    if platform.system() != "Darwin":
        return None
    objc_path = ctypes.util.find_library("objc")
    foundation_path = ctypes.util.find_library("Foundation")
    if objc_path is None or foundation_path is None:
        return None
    try:
        ctypes.CDLL(foundation_path)
        objc = ctypes.CDLL(objc_path)
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        message = objc.objc_msgSend
        message.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        message.restype = ctypes.c_void_p
        process_info = message(
            objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo")
        )
        message.restype = ctypes.c_long
        return int(message(process_info, objc.sel_registerName(selector_name)))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def detect_thermal_state() -> ThermalState:
    value = _ns_process_info_integer(b"thermalState")
    return {
        0: ThermalState.NOMINAL,
        1: ThermalState.FAIR,
        2: ThermalState.SERIOUS,
        3: ThermalState.CRITICAL,
    }.get(value, ThermalState.UNKNOWN)


def _pmset(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["/usr/bin/pmset", *arguments],
            capture_output=True,
            check=True,
            text=True,
            timeout=1.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if len(result.stdout.encode("utf-8")) > 64 * 1024:
        return None
    return result.stdout


def _parse_power_mode(power_source: str, custom_settings: str) -> PowerMode:
    source_match = re.search(r"Now drawing from '([^']+)'", power_source)
    if source_match is None:
        return PowerMode.UNKNOWN
    target = source_match.group(1)
    sections: dict[str, dict[str, int]] = {}
    current: str | None = None
    for line in custom_settings.splitlines():
        header = re.match(r"^([^:\n]{1,64}):\s*$", line)
        if header:
            current = header.group(1).strip()
            sections[current] = {}
            continue
        setting = re.match(r"^\s+([a-z][a-z0-9]*)\s+(-?\d+)\s*$", line)
        if current is not None and setting:
            sections[current][setting.group(1)] = int(setting.group(2))
    settings = sections.get(target)
    if settings is None:
        return PowerMode.UNKNOWN
    if settings.get("lowpowermode") == 1 or settings.get("powermode") == 1:
        return PowerMode.LOW_POWER
    if settings.get("powermode") == 2:
        return PowerMode.HIGH_POWER
    if settings.get("lowpowermode") == 0 or settings.get("powermode") == 0:
        return PowerMode.AUTOMATIC
    return PowerMode.UNKNOWN


def detect_power_mode() -> PowerMode:
    if platform.system() != "Darwin":
        return PowerMode.UNKNOWN
    power_source = _pmset(["-g", "ps"])
    custom_settings = _pmset(["-g", "custom"])
    if power_source is None or custom_settings is None:
        return PowerMode.UNKNOWN
    return _parse_power_mode(power_source, custom_settings)


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
    apple_silicon = system == "Darwin" and machine == "arm64"
    soc = _sysctl("machdep.cpu.brand_string") or platform.processor()
    if apple_silicon and (not soc or soc.strip().lower() in {"arm", "arm64", "unknown"}):
        soc = _system_profiler_chip()
    soc = soc or "Unknown Apple SoC"
    gpu_cores = _positive_int(_sysctl("hw.perflevel0.gpu_count"))
    if gpu_cores is None and apple_silicon:
        gpu_cores = _ioreg_gpu_core_count()
    return HardwareInfo(
        platform=system,
        architecture=machine,
        soc=soc,
        physical_cpu_count=physical,
        logical_cpu_count=logical,
        gpu_core_count=gpu_cores,
        memory=detect_memory(),
        is_apple_silicon=apple_silicon,
        os_version=platform.mac_ver()[0] if system == "Darwin" else platform.release(),
        thermal_state=detect_thermal_state(),
        power_mode=detect_power_mode(),
    )


def default_application_support() -> Path:
    return Path.home() / "Library" / "Application Support" / "vllm-apple"
