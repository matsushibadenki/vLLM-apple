"""Cooperative cross-process exclusion for local GPU measurement runners.

The default flock file is shared with Kapitsa. This reserves participating
measurement runners, not the GPU device or processes which ignore the lock.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

LOCK_FD_ENV = "VLLM_APPLE_BENCH_GPU_LOCK_FD"
DEFAULT_LOCK_PATH = "/tmp/kapitsa-gpu.lock"


class BenchmarkGpuBusy(RuntimeError):
    """Another cooperating runner owns the benchmark lock."""


class BenchmarkGpuLock:
    def __init__(self) -> None:
        self.path = Path(os.environ.get("BENCH_GPU_LOCK") or DEFAULT_LOCK_PATH).expanduser().absolute()
        self._fd: int | None = None
        self._inherited = False

    def __enter__(self) -> BenchmarkGpuLock:
        import fcntl

        if self._fd is not None:
            raise RuntimeError("benchmark lock is already entered")
        inherited = os.environ.get(LOCK_FD_ENV)
        if inherited is None:
            fd = os.open(self.path, os.O_RDONLY | os.O_CREAT | os.O_NOFOLLOW, 0o644)
        else:
            if not inherited.isascii() or not inherited.isdecimal() or int(inherited) < 3:
                raise ValueError("invalid inherited benchmark lock descriptor")
            fd = os.dup(int(inherited))
        try:
            opened, current = os.fstat(fd), self.path.stat()
            if (not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)):
                raise ValueError("benchmark lock descriptor does not match the lock file")
            try:
                # The same open file description can be rechecked without
                # waiting on the parent. A marker alone never bypasses flock.
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise BenchmarkGpuBusy(f"GPU benchmark already running: {self.path}") from error
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        self._inherited = inherited is not None
        return self

    def child_options(self, env: dict[str, str] | None = None) -> dict[str, object]:
        """Pass only to participating child runners, never as a boolean bypass."""
        if self._fd is None:
            raise RuntimeError("benchmark lock is not held")
        child_env = dict(os.environ if env is None else env)
        child_env.update(BENCH_GPU_LOCK=str(self.path), **{LOCK_FD_ENV: str(self._fd)})
        return {"env": child_env, "pass_fds": (self._fd,)}

    def snapshot(self) -> dict[str, object]:
        if self._fd is None:
            raise RuntimeError("benchmark lock is not held")
        info = os.fstat(self._fd)
        return {"path": str(self.path), "inherited": self._inherited,
                "device": info.st_dev, "inode": info.st_ino,
                "scope": "cooperating benchmark runners only; not GPU device exclusivity"}

    def __exit__(self, *exc: object) -> None:
        if self._fd is not None:
            # Do not LOCK_UN: nested runners share this open file description.
            # The lock survives until every explicitly inherited fd is closed.
            os.close(self._fd)
            self._fd = None
