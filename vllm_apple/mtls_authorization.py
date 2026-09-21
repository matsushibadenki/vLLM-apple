"""Strict reloadable mTLS subject/SAN authorization policy."""
from __future__ import annotations

import json
import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path

MAX_POLICY_BYTES = 64 * 1024
MAX_IDENTITIES = 256
_SAN_KINDS = {"DNS", "email", "URI", "IP Address"}


def _bounded_strings(value: object, label: str) -> tuple[str, ...]:
    if (not isinstance(value, list) or len(value) > MAX_IDENTITIES
            or any(not isinstance(item, str) or not 1 <= len(item) <= 512
                   or any(ord(character) < 0x20 for character in item)
                   for item in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"invalid mTLS {label}")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class ClientCertificatePolicy:
    allowed_subjects: tuple[str, ...]
    allowed_sans: tuple[str, ...]
    revoked_serial_numbers: tuple[str, ...]
    policy_version: int = 1

    def __post_init__(self) -> None:
        for values, label in (
            (self.allowed_subjects, "allowed subjects"),
            (self.allowed_sans, "allowed SANs"),
            (self.revoked_serial_numbers, "revoked serial numbers"),
        ):
            if (not isinstance(values, tuple) or len(values) > MAX_IDENTITIES
                    or len(set(values)) != len(values)
                    or any(not isinstance(value, str) or not 1 <= len(value) <= 512
                           for value in values)):
                raise ValueError(f"invalid mTLS {label}")
        if self.policy_version != 1 or not (
            self.allowed_subjects or self.allowed_sans
        ):
            raise ValueError("mTLS policy requires at least one allowed identity")

    @classmethod
    def from_dict(cls, payload: object) -> "ClientCertificatePolicy":
        if not isinstance(payload, dict) or set(payload) != {
            "policy_version", "allowed_subjects", "allowed_sans",
            "revoked_serial_numbers",
        }:
            raise ValueError("invalid mTLS policy schema")
        if payload["policy_version"] != 1:
            raise ValueError("unsupported mTLS policy version")
        return cls(
            _bounded_strings(payload["allowed_subjects"], "allowed subjects"),
            _bounded_strings(payload["allowed_sans"], "allowed SANs"),
            tuple(value.upper() for value in _bounded_strings(
                payload["revoked_serial_numbers"], "revoked serial numbers"
            )),
        )

    def authorize(self, certificate: object) -> bool:
        if not isinstance(certificate, dict):
            return False
        serial = certificate.get("serialNumber")
        if isinstance(serial, str) and serial.upper() in self.revoked_serial_numbers:
            return False
        subject = _certificate_subject(certificate.get("subject"))
        if subject is not None and subject in self.allowed_subjects:
            return True
        sans = certificate.get("subjectAltName")
        if not isinstance(sans, tuple) or len(sans) > MAX_IDENTITIES:
            return False
        identities = {
            f"{kind}:{value}"
            for item in sans
            if (isinstance(item, tuple) and len(item) == 2)
            for kind, value in (item,)
            if kind in _SAN_KINDS and isinstance(value, str) and len(value) <= 512
        }
        return bool(identities.intersection(self.allowed_sans))


def _certificate_subject(value: object) -> str | None:
    if not isinstance(value, tuple) or len(value) > 32:
        return None
    components: list[str] = []
    for rdn in value:
        if not isinstance(rdn, tuple) or len(rdn) > 8:
            return None
        for pair in rdn:
            if (not isinstance(pair, tuple) or len(pair) != 2
                    or not all(isinstance(item, str) for item in pair)
                    or any(not 1 <= len(item) <= 256 for item in pair)):
                return None
            key, item_value = pair
            components.append(f"{key}={item_value}")
    return ",".join(components) if components else None


class ClientCertificatePolicyStore:
    """Reload an atomically replaced policy; changed invalid state denies access."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().absolute()
        self._lock = threading.Lock()
        self._identity: tuple[int, int, int] | None = None
        self._policy: ClientCertificatePolicy | None = None
        self._invalid = False
        self.reload()

    def reload(self) -> None:
        with self._lock:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    info = os.fstat(descriptor)
                    if (not stat.S_ISREG(info.st_mode)
                            or info.st_uid != os.getuid()
                            or stat.S_IMODE(info.st_mode) & 0o022
                            or not 0 < info.st_size <= MAX_POLICY_BYTES):
                        raise ValueError("unsafe mTLS policy file")
                    identity = (info.st_ino, info.st_mtime_ns, info.st_size)
                    if identity == self._identity and not self._invalid:
                        return
                    content = bytearray()
                    while len(content) <= MAX_POLICY_BYTES:
                        chunk = os.read(
                            descriptor,
                            min(8192, MAX_POLICY_BYTES + 1 - len(content)),
                        )
                        if not chunk:
                            break
                        content.extend(chunk)
                    final_info = os.fstat(descriptor)
                    final_identity = (
                        final_info.st_ino, final_info.st_mtime_ns, final_info.st_size
                    )
                    if len(content) != info.st_size or final_identity != identity:
                        raise ValueError("mTLS policy changed while reading")
                finally:
                    os.close(descriptor)
                payload = json.loads(content)
                policy = ClientCertificatePolicy.from_dict(payload)
            except (OSError, ValueError, json.JSONDecodeError):
                self._policy = None
                self._invalid = True
                self._identity = None
                return
            self._policy = policy
            self._identity = identity
            self._invalid = False

    def authorize(self, certificate: object) -> bool:
        self.reload()
        with self._lock:
            return self._policy is not None and self._policy.authorize(certificate)

    def snapshot(self) -> dict[str, object]:
        self.reload()
        with self._lock:
            return {
                "configured": True,
                "valid": self._policy is not None,
                "policy_version": (
                    self._policy.policy_version if self._policy is not None else None
                ),
                "allowed_subject_count": (
                    len(self._policy.allowed_subjects) if self._policy else 0
                ),
                "allowed_san_count": len(self._policy.allowed_sans) if self._policy else 0,
                "revoked_serial_count": (
                    len(self._policy.revoked_serial_numbers) if self._policy else 0
                ),
            }
