from __future__ import annotations

import hmac
import os
import secrets
import stat
import tempfile
from pathlib import Path


class SessionTokenError(RuntimeError):
    pass


class SessionAuthenticator:
    def __init__(self, token: str | None = None) -> None:
        if token is not None and len(token) < 32:
            raise SessionTokenError("session token must contain at least 32 characters")
        self._token = token

    @property
    def required(self) -> bool:
        return self._token is not None

    def authorize(self, authorization_header: str | None) -> bool:
        if self._token is None:
            return True
        if not authorization_header or not authorization_header.startswith("Bearer "):
            return False
        candidate = authorization_header[7:]
        return hmac.compare_digest(candidate.encode("utf-8"), self._token.encode("utf-8"))


def generate_session_token() -> str:
    return secrets.token_urlsafe(32)


def load_or_create_token_file(path: Path) -> str:
    destination = path.expanduser()
    if destination.exists():
        mode = stat.S_IMODE(destination.stat().st_mode)
        if mode & 0o077:
            raise SessionTokenError("session token file must not be accessible by group or others")
        token = destination.read_text(encoding="utf-8").strip()
        if len(token) < 32:
            raise SessionTokenError("session token file contains an invalid token")
        return token

    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    token = generate_session_token()
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return token

