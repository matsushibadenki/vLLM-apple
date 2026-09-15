"""Profile-bound CPU/GPU/ANE shared-bandwidth qualification."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .device_resources import BandwidthContentionEvidence, UnifiedDeviceResourceLedger
from .execution import ExecutionBackend
from .hardware import default_application_support


CONTENTION_PROFILE_SCHEMA_VERSION = 1
MAX_CONTENTION_PAIRS = 16
MAX_CONTENTION_SAMPLES = 64
MAX_CONTENTION_PROFILE_BYTES = 128 * 1024


def default_contention_profile_path(
    profile_id: str, *, application_support: Path | None = None
) -> Path:
    if not profile_id or len(profile_id) != 64 or any(
        character not in "0123456789abcdef" for character in profile_id
    ):
        raise ValueError("invalid contention profile path identity")
    return default_contention_profile_paths(
        profile_id, application_support=application_support
    )[0]


def default_contention_profile_paths(
    profile_id: str, *, application_support: Path | None = None
) -> tuple[Path, Path]:
    if not profile_id or len(profile_id) != 64 or any(
        character not in "0123456789abcdef" for character in profile_id
    ):
        raise ValueError("invalid contention profile path identity")
    root = (application_support or default_application_support()) / "profiles" / "device-contention"
    return root / f"{profile_id}.json", root / f"{profile_id}.last-known-good.json"


@dataclass(frozen=True, slots=True)
class ContentionBenchmarkConfig:
    profile_id: str
    first_backend: ExecutionBackend
    second_backend: ExecutionBackend
    samples: int = 5

    def __post_init__(self) -> None:
        if (
            not self.profile_id or len(self.profile_id) > 128
            or self.first_backend is self.second_backend
            or type(self.samples) is not int
            or not 3 <= self.samples <= MAX_CONTENTION_SAMPLES
        ):
            raise ValueError("invalid contention benchmark configuration")


@dataclass(frozen=True, slots=True)
class ContentionProfile:
    profile_id: str
    evidence: tuple[BandwidthContentionEvidence, ...]
    schema_version: int = CONTENTION_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        pairs = {
            frozenset((item.first_backend, item.second_backend)) for item in self.evidence
        }
        if (
            self.schema_version != CONTENTION_PROFILE_SCHEMA_VERSION
            or not self.profile_id or len(self.profile_id) > 128
            or not 1 <= len(self.evidence) <= MAX_CONTENTION_PAIRS
            or len(pairs) != len(self.evidence)
            or any(item.profile_id != self.profile_id for item in self.evidence)
            or any(not item.qualified for item in self.evidence)
        ):
            raise ValueError("invalid contention profile")

    @property
    def profile_digest(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict(include_digest=False))).hexdigest()

    def to_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        entries = []
        for item in self.evidence:
            value = asdict(item)
            value["first_backend"] = item.first_backend.value
            value["second_backend"] = item.second_backend.value
            entries.append(value)
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "evidence": entries,
        }
        if include_digest:
            payload["profile_digest"] = self.profile_digest
        return payload


def run_contention_benchmark(
    config: ContentionBenchmarkConfig,
    first_operation: Callable[[], str],
    second_operation: Callable[[], str],
) -> BandwidthContentionEvidence:
    sequential: list[int] = []
    parallel: list[int] = []
    expected: tuple[str, str] | None = None
    outputs_match = True
    for _ in range(config.samples):
        started = time.perf_counter_ns()
        sequential_output = (first_operation(), second_operation())
        sequential.append(max(1, time.perf_counter_ns() - started))
        outputs: list[str | None] = [None, None]
        errors: list[BaseException] = []
        barrier = threading.Barrier(3)

        def invoke(index: int, operation: Callable[[], str]) -> None:
            try:
                barrier.wait(timeout=5)
                outputs[index] = operation()
            except BaseException as error:
                errors.append(error)

        workers = (
            threading.Thread(target=invoke, args=(0, first_operation)),
            threading.Thread(target=invoke, args=(1, second_operation)),
        )
        for worker in workers:
            worker.start()
        barrier.wait(timeout=5)
        started = time.perf_counter_ns()
        for worker in workers:
            worker.join(timeout=300)
        parallel.append(max(1, time.perf_counter_ns() - started))
        if errors or any(worker.is_alive() for worker in workers):
            raise RuntimeError("contention benchmark operation failed or timed out")
        parallel_output = (outputs[0], outputs[1])
        if any(not _digest(value) for value in (*sequential_output, *parallel_output)):
            raise ValueError("contention benchmark output digest is invalid")
        expected = sequential_output if expected is None else expected
        outputs_match &= sequential_output == parallel_output == expected
    sequential.sort()
    parallel.sort()
    middle = len(sequential) // 2
    return BandwidthContentionEvidence(
        config.profile_id, config.first_backend, config.second_backend,
        sequential[middle], parallel[middle], config.samples, outputs_match,
    )


def install_contention_profile(
    ledger: UnifiedDeviceResourceLedger, profile: ContentionProfile
) -> int:
    ledger.replace_contention_evidence(profile.evidence)
    return len(profile.evidence)


def save_contention_profile(profile: ContentionProfile, path: Path) -> Path:
    destination = path.expanduser()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.parent.chmod(0o700)
    _validate_private_directory(destination.parent)
    encoded = _canonical(profile.to_dict()) + b"\n"
    if len(encoded) > MAX_CONTENTION_PROFILE_BYTES:
        raise ValueError("contention profile exceeds bounded size")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return destination


def load_contention_profile(path: Path, *, profile_id: str) -> ContentionProfile:
    _validate_private_directory(path.parent)
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
        or info.st_mode & 0o077 or not 0 < info.st_size <= MAX_CONTENTION_PROFILE_BYTES
    ):
        raise ValueError("contention profile must be a bounded private regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "profile_id", "evidence", "profile_digest",
    } or payload.get("profile_id") != profile_id or not isinstance(payload.get("evidence"), list):
        raise ValueError("invalid contention profile fields")
    try:
        evidence = tuple(BandwidthContentionEvidence(
            item["profile_id"], ExecutionBackend(item["first_backend"]),
            ExecutionBackend(item["second_backend"]),
            item["sequential_latency_nanoseconds"], item["parallel_latency_nanoseconds"],
            item["sample_count"], item["outputs_match"],
        ) for item in payload["evidence"] if isinstance(item, dict) and set(item) == {
            "profile_id", "first_backend", "second_backend",
            "sequential_latency_nanoseconds", "parallel_latency_nanoseconds",
            "sample_count", "outputs_match",
        })
        profile = ContentionProfile(payload["profile_id"], evidence, payload["schema_version"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid contention profile values") from error
    if len(evidence) != len(payload["evidence"]) or payload["profile_digest"] != profile.profile_digest:
        raise ValueError("contention profile digest mismatch")
    return profile


def load_contention_profile_with_fallback(
    current: Path, last_known_good: Path, *, profile_id: str
) -> tuple[ContentionProfile, str]:
    try:
        return load_contention_profile(current, profile_id=profile_id), "current"
    except (OSError, ValueError, json.JSONDecodeError):
        return load_contention_profile(
            last_known_good, profile_id=profile_id
        ), "last_known_good"


def promote_contention_profile(
    profile: ContentionProfile, current: Path, last_known_good: Path
) -> Path:
    if current == last_known_good:
        raise ValueError("contention current and rollback paths must differ")
    if current.exists():
        try:
            previous = load_contention_profile(current, profile_id=profile.profile_id)
        except (OSError, ValueError, json.JSONDecodeError):
            previous = None
        if previous is not None and previous.profile_digest != profile.profile_digest:
            save_contention_profile(previous, last_known_good)
    return save_contention_profile(profile, current)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate_private_directory(path: Path) -> None:
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("contention profile directory must be private")
