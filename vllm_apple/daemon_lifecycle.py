"""Safe per-user launchd lifecycle for the local vLLM-Apple daemon."""
from __future__ import annotations

import os
import plistlib
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DEFAULT_DAEMON_LABEL = "io.vllm-apple.daemon"
MAX_DAEMON_PLIST_BYTES = 128 * 1024
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{0,127}\Z")


@dataclass(frozen=True, slots=True)
class DaemonLifecycleResult:
    action: str
    label: str
    plist_path: str
    running: bool


def default_daemon_plist(label: str = DEFAULT_DAEMON_LABEL) -> Path:
    _validate_label(label)
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def install_daemon(
    model: str | None,
    *,
    label: str = DEFAULT_DAEMON_LABEL,
    plist_path: Path | None = None,
    serve_arguments: Sequence[str] = (),
    force: bool = False,
) -> DaemonLifecycleResult:
    """Atomically install, but do not load, one current-user LaunchAgent."""
    _require_darwin()
    _validate_label(label)
    path = (plist_path or default_daemon_plist(label)).expanduser().absolute()
    if path.name != f"{label}.plist" or path.suffix != ".plist":
        raise ValueError("daemon plist name must match its launchd label")
    if model is not None and (not model or "\x00" in model):
        raise ValueError("daemon model argument is invalid")
    arguments = tuple(serve_arguments)
    if len(arguments) > 128 or any(
        not isinstance(value, str) or not value or "\x00" in value
        or len(value.encode("utf-8")) > 16_384
        for value in arguments
    ):
        raise ValueError("daemon serve arguments are invalid")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError("daemon plist already exists; use --force to replace it")
    support = Path.home() / "Library" / "Application Support" / "vLLM-Apple" / "daemon"
    support.mkdir(parents=True, exist_ok=True, mode=0o700)
    support_info = support.stat()
    if (not support.is_dir() or support.is_symlink() or support_info.st_uid != os.getuid()
            or stat.S_IMODE(support_info.st_mode) & 0o077):
        raise ValueError("daemon Application Support directory must be owner-only")
    has_socket = "--socket-path" in arguments
    has_token = "--session-token-file" in arguments
    if has_socket != has_token:
        raise ValueError("daemon socket and session token file must be configured together")
    if not has_socket:
        arguments += (
            "--socket-path", str(support / "runtime.sock"),
            "--session-token-file", str(support / "session.token"),
        )
    command = [sys.executable, "-m", "vllm_apple.cli", "serve"]
    if model is not None:
        command.append(model)
    command.extend(arguments)
    payload = plistlib.dumps({
        "Label": label,
        "ProgramArguments": command,
        "RunAtLoad": False,
        "KeepAlive": False,
        "ProcessType": "Interactive",
        "StandardOutPath": str(support / "daemon.stdout.log"),
        "StandardErrorPath": str(support / "daemon.stderr.log"),
    }, sort_keys=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{label}.", dir=parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or path.is_symlink():
                raise ValueError("existing daemon plist is unsafe")
        os.replace(temporary_path, path)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise
    return DaemonLifecycleResult("install", label, str(path), False)


def start_daemon(*, label: str = DEFAULT_DAEMON_LABEL,
                 plist_path: Path | None = None) -> DaemonLifecycleResult:
    _require_darwin()
    path = _validate_installed_plist(label, plist_path)
    _launchctl(("bootstrap", _domain(), str(path)))
    return DaemonLifecycleResult("start", label, str(path), True)


def stop_daemon(*, label: str = DEFAULT_DAEMON_LABEL,
                plist_path: Path | None = None) -> DaemonLifecycleResult:
    _require_darwin()
    path = _validate_installed_plist(label, plist_path)
    _launchctl(("bootout", f"{_domain()}/{label}"))
    return DaemonLifecycleResult("stop", label, str(path), False)


def daemon_status(*, label: str = DEFAULT_DAEMON_LABEL,
                  plist_path: Path | None = None) -> DaemonLifecycleResult:
    _require_darwin()
    path = _validate_installed_plist(label, plist_path)
    result = subprocess.run(
        ("/bin/launchctl", "print", f"{_domain()}/{label}"),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=10, check=False,
    )
    return DaemonLifecycleResult("status", label, str(path), result.returncode == 0)


def _validate_installed_plist(label: str, plist_path: Path | None) -> Path:
    _validate_label(label)
    path = (plist_path or default_daemon_plist(label)).expanduser().absolute()
    if path.name != f"{label}.plist" or path.is_symlink():
        raise ValueError("daemon plist name or type is invalid")
    info = path.stat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
            or not 1 <= info.st_size <= MAX_DAEMON_PLIST_BYTES):
        raise ValueError("daemon plist must be an owner-only regular file")
    payload = plistlib.loads(path.read_bytes())
    arguments = payload.get("ProgramArguments") if isinstance(payload, dict) else None
    if (not isinstance(payload, dict) or payload.get("Label") != label
            or not isinstance(arguments, list)
            or arguments[:4] != [sys.executable, "-m", "vllm_apple.cli", "serve"]
            or any(not isinstance(value, str) or not value or "\x00" in value
                   for value in arguments)):
        raise ValueError("daemon plist identity does not match")
    return path


def _launchctl(arguments: Sequence[str]) -> None:
    try:
        subprocess.run(
            ("/bin/launchctl", *arguments), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10, check=True,
        )
    except (subprocess.SubprocessError, OSError) as error:
        raise RuntimeError(f"launchctl {arguments[0]} failed") from error


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _validate_label(label: str) -> None:
    if not isinstance(label, str) or _LABEL.fullmatch(label) is None:
        raise ValueError("invalid launchd label")


def _require_darwin() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("daemon lifecycle commands require macOS")
