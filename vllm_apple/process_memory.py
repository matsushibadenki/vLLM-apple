"""Current RSS in bytes, without spawning a process for each macOS sample."""
from __future__ import annotations

import ctypes
import errno
import os
import subprocess
import sys
from functools import lru_cache


class _TaskInfo(ctypes.Structure):
    # Public Darwin proc_taskinfo ABI: six uint64 values, twelve int32 values.
    # Only resident_size is consumed. This is RSS, not physical footprint/peak.
    _fields_ = [('virtual_size', ctypes.c_uint64), ('resident_size', ctypes.c_uint64),
                ('_times', ctypes.c_uint64 * 4), ('_counters', ctypes.c_int32 * 12)]


@lru_cache(maxsize=1)
def _task_info_function():
    library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    function = library.proc_pidinfo
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
    function.restype = ctypes.c_int
    return library, function  # Keep the library alive alongside the function.


def _ps_resident_bytes(pid: int) -> int:
    result = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(pid)],
                            capture_output=True, check=True, text=True, timeout=2)
    return int(result.stdout.strip()) * 1024


def resident_bytes(pid: int) -> int:
    if type(pid) is not int or not 0 < pid <= 2**31-1:
        raise ValueError('pid must be a positive signed 32-bit integer')
    if sys.platform != 'darwin':
        return _ps_resident_bytes(pid)
    try:
        _, function = _task_info_function()
    except (OSError, AttributeError):
        # Preserve the existing portable path when the OS API cannot be loaded.
        return _ps_resident_bytes(pid)
    info = _TaskInfo()  # Per-call storage: concurrent samplers never share a buffer.
    size = ctypes.sizeof(info)
    ctypes.set_errno(0)
    received = function(pid, 4, 0, ctypes.byref(info), size)  # PROC_PIDTASKINFO
    if received != size:
        code = ctypes.get_errno() or errno.EIO
        raise OSError(code, os.strerror(code))
    return int(info.resident_size)
