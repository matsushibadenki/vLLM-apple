"""Fingerprint a Python backend's installed files without importing its GPU runtime."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_FILES = 100_000
MAX_BYTES = 32 * 1024**3


def _environment_inventory(prefix: Path) -> set[Path]:
    """Inventory a brew-managed isolated venv when RECORD was deliberately removed."""
    prefix = prefix.resolve(strict=True)
    config = prefix / "pyvenv.cfg"
    if not config.is_file() or not any(
        line.strip().lower() == "include-system-site-packages = false"
        for line in config.read_text().splitlines()
    ):
        raise ValueError("Homebrew inventory requires an isolated virtual environment")
    paths: set[Path] = set()
    entries = 0
    for root, directories, files in os.walk(prefix, followlinks=False):
        entries += len(directories) + len(files)
        if entries > MAX_FILES:
            raise ValueError("backend file inventory exceeds limit")
        directories[:] = [name for name in directories if name != "__pycache__"]
        for name in directories:
            if (Path(root) / name).is_symlink():
                raise ValueError("backend directory symlinks cannot be inventoried")
        for name in files:
            path = Path(root) / name
            if path.suffix in {".pyc", ".pyo"}:
                continue
            resolved = path.resolve(strict=True)
            # The interpreter is a normal external venv symlink; other files must
            # remain inside the environment so its inventory stays complete.
            if not resolved.is_relative_to(prefix) and not (
                path.parent == prefix / "bin" and name.startswith("python")
                and resolved == Path(sys.executable).resolve(strict=True)
            ):
                raise ValueError("backend inventory contains an external symlink")
            if not resolved.is_file():
                raise ValueError("backend inventory contains a nonregular file")
            paths.add(path)
    return paths


def _check_search_path(path: Path) -> None:
    if path.suffix == ".pth" and any(
        line.strip() and not line.startswith(("#", "import ", "import\t"))
        for line in path.read_text().splitlines()
    ):
        raise ValueError("external backend search paths are not supported")


def installed_fingerprint(executable: Path) -> str:
    """Worker: hash interpreter, launcher and distribution-listed non-bytecode files.

    Editable installs are rejected: their external sources are not covered by a
    wheel's file inventory. This binds local trusted files, not remote provenance.
    """
    import importlib.metadata as metadata

    paths = {executable.resolve(strict=True), Path(sys.executable).resolve(strict=True)}
    versions = []
    environment_paths = None
    for distribution in metadata.distributions():
        direct = distribution.read_text("direct_url.json")
        if direct and json.loads(direct).get("dir_info", {}).get("editable") is True:
            raise ValueError("editable backend environments cannot be fingerprinted")
        files = distribution.files
        if not files:
            if (distribution.read_text("INSTALLER") or "").strip() != "brew":
                raise ValueError("backend distribution has no file inventory")
            if environment_paths is None:
                environment_paths = _environment_inventory(Path(sys.prefix))
            if not Path(distribution.locate_file("")).resolve().is_relative_to(
                Path(sys.prefix).resolve()
            ):
                raise ValueError("Homebrew distribution is outside its environment")
            paths.update(environment_paths)
            files = ()
        versions.append((distribution.metadata["Name"], distribution.version))
        for item in files:
            if str(item).endswith((".pyc", ".pyo")) or "__pycache__" in item.parts:
                continue
            path = Path(distribution.locate_file(item))
            _check_search_path(path)
            paths.add(path.resolve(strict=True))
            if len(paths) > MAX_FILES:
                raise ValueError("backend file inventory exceeds limit")
    digest = hashlib.sha256(json.dumps(sorted(versions), separators=(",", ":")).encode())
    if len(paths) > MAX_FILES:
        raise ValueError("backend file inventory exceeds limit")
    total = 0
    for path in sorted(paths):
        _check_search_path(path)
        # Hash the logical path and resolved target, including symlink identity.
        digest.update(str(path).encode() + b"\0")
        path = path.resolve(strict=True)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("backend file is not regular")
            total += before.st_size
            if total > MAX_BYTES:
                raise ValueError("backend files exceed byte limit")
            digest.update(str(path).encode() + b"\0" + str(before.st_size).encode() + b"\0")
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(8 * 1024**2, remaining))
                if not chunk:
                    raise ValueError("backend file changed")
                digest.update(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns
            ):
                raise ValueError("backend file changed")
        finally:
            os.close(descriptor)
    if environment_paths is not None and environment_paths != _environment_inventory(Path(sys.prefix)):
        raise ValueError("backend inventory changed during fingerprinting")
    return digest.hexdigest()


def fingerprint_backend(executable: Path) -> str:
    path = executable.expanduser().absolute()
    python = path.parent / "python"
    if not python.is_file() or not path.is_file():
        raise ValueError("backend launcher requires its adjacent Python interpreter")
    with path.open("rb") as launcher:
        first_line = launcher.readline(4096).decode("utf-8", errors="replace").strip()
    if not first_line.startswith("#!/"):
        raise ValueError("backend launcher must have an explicit Python shebang")
    interpreter = Path(first_line[2:])
    if interpreter.parent.resolve() != python.parent.resolve() or not interpreter.samefile(python):
        raise ValueError("backend launcher interpreter differs from its environment")
    # -I excludes caller PYTHONPATH/user-site from the measurement. Such runtime
    # overrides are rejected rather than silently omitted from the identity.
    if any(os.environ.get(key) for key in ("PYTHONPATH", "PYTHONHOME")):
        raise ValueError("backend Python environment overrides are not supported")
    with tempfile.TemporaryFile() as output:
        try:
            result = subprocess.run(
                [str(python), "-I", str(Path(__file__).resolve()), str(path)],
                stdout=output, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                timeout=120, check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError("backend fingerprint probe failed") from error
        output.seek(0)
        raw = output.read(257)
    value = raw.decode("ascii", errors="replace").strip()
    if result.returncode or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("backend fingerprint probe failed")
    return value


if __name__ == "__main__":
    print(installed_fingerprint(Path(sys.argv[1])))
