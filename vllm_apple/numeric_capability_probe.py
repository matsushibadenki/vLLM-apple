"""Environment-bound numeric capability probes with explicit reference fallback."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from .execution import ExecutionBackend
from .numeric_routing import NumericFormat, NumericTensorRole

MAX_NUMERIC_PROBE_BYTES = 16 * 1024 * 1024
MAX_NUMERIC_PROBE_RECORDS = 512


def _text(value: object) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 256 and all(
        ord(character) >= 0x20 for character in value
    )


@dataclass(frozen=True, slots=True)
class NumericProbeIdentity:
    chip: str
    os_build: str
    toolchain: str
    backend: ExecutionBackend
    backend_version: str
    operator: str
    shape: tuple[int, ...]
    source_format: NumericFormat
    compute_format: NumericFormat
    tensor_role: NumericTensorRole
    recipe_id: str

    def __post_init__(self) -> None:
        if (any(not _text(value) for value in (
                self.chip, self.os_build, self.toolchain, self.backend_version,
                self.operator, self.recipe_id,
        )) or not isinstance(self.backend, ExecutionBackend)
                or not isinstance(self.source_format, NumericFormat)
                or not isinstance(self.compute_format, NumericFormat)
                or not isinstance(self.tensor_role, NumericTensorRole)
                or not 1 <= len(self.shape) <= 8
                or any(type(value) is not int or not 1 <= value <= 1 << 30
                       for value in self.shape)):
            raise ValueError("invalid numeric probe identity")

    @property
    def probe_id(self) -> str:
        payload = asdict(self)
        payload["backend"] = self.backend.value
        payload["source_format"] = self.source_format.value
        payload["compute_format"] = self.compute_format.value
        payload["tensor_role"] = self.tensor_role.value
        return hashlib.sha256(_canonical(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class NumericProbeResult:
    probe_id: str
    passed: bool
    reason: str
    reference_sha256: str
    candidate_sha256: str | None
    latency_nanoseconds: int


@dataclass(frozen=True, slots=True)
class NumericFallbackResult:
    output: bytes
    backend: str
    fallback_reason: str | None


class NumericCapabilityProbeRegistry:
    def __init__(self, supported_recipe_ids: tuple[str, ...]) -> None:
        if (not supported_recipe_ids or len(supported_recipe_ids) > 256
                or len(set(supported_recipe_ids)) != len(supported_recipe_ids)
                or any(not _text(value) for value in supported_recipe_ids)):
            raise ValueError("invalid numeric probe recipe inventory")
        self._recipes = frozenset(supported_recipe_ids)
        self._results: dict[str, NumericProbeResult] = {}
        self._quarantine: dict[str, str] = {}
        self._lock = threading.Lock()

    def probe(
        self,
        identity: NumericProbeIdentity,
        probe_input: bytes,
        reference: Callable[[bytes], bytes],
        candidate: Callable[[bytes], bytes],
    ) -> NumericProbeResult:
        if identity.recipe_id not in self._recipes:
            raise ValueError("unsupported numeric recipe")
        if (not isinstance(probe_input, bytes)
                or not 1 <= len(probe_input) <= MAX_NUMERIC_PROBE_BYTES
                or not callable(reference) or not callable(candidate)):
            raise ValueError("invalid numeric probe request")
        reference_output = _bounded_output(reference(probe_input))
        reference_digest = hashlib.sha256(reference_output).hexdigest()
        started = time.perf_counter_ns()
        try:
            candidate_output = _bounded_output(candidate(probe_input))
            candidate_digest = hashlib.sha256(candidate_output).hexdigest()
            passed = candidate_digest == reference_digest
            reason = "probe_passed" if passed else "output_mismatch"
        except Exception as error:
            candidate_digest = None
            passed = False
            reason = f"candidate_{type(error).__name__}"
        latency = max(1, time.perf_counter_ns() - started)
        result = NumericProbeResult(
            identity.probe_id, passed, reason, reference_digest,
            candidate_digest, latency,
        )
        with self._lock:
            if identity.probe_id not in self._results and len(self._results) >= MAX_NUMERIC_PROBE_RECORDS:
                raise ValueError("numeric probe registry is full")
            self._results[identity.probe_id] = result
            if passed:
                self._quarantine.pop(identity.probe_id, None)
            else:
                self._quarantine[identity.probe_id] = reason
        return result

    def eligible(self, identity: NumericProbeIdentity) -> bool:
        with self._lock:
            result = self._results.get(identity.probe_id)
            return (
                identity.recipe_id in self._recipes
                and result is not None
                and result.passed
                and identity.probe_id not in self._quarantine
            )

    def execute(
        self,
        identity: NumericProbeIdentity,
        value: bytes,
        candidate: Callable[[bytes], bytes],
        reference: Callable[[bytes], bytes],
    ) -> NumericFallbackResult:
        if identity.recipe_id not in self._recipes:
            raise ValueError("unsupported numeric recipe")
        if not isinstance(value, bytes) or not 1 <= len(value) <= MAX_NUMERIC_PROBE_BYTES:
            raise ValueError("invalid numeric execution input")
        if not self.eligible(identity):
            return NumericFallbackResult(
                _bounded_output(reference(value)), "cpu_reference", "probe_unavailable"
            )
        try:
            output = _bounded_output(candidate(value))
        except Exception as error:
            reason = f"runtime_{type(error).__name__}"
            with self._lock:
                self._quarantine[identity.probe_id] = reason
            return NumericFallbackResult(
                _bounded_output(reference(value)), "cpu_reference", reason
            )
        return NumericFallbackResult(output, identity.backend.value, None)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "schema_version": 1,
                "supported_recipes": sorted(self._recipes),
                "probes": tuple(asdict(self._results[key]) for key in sorted(self._results)),
                "quarantine": dict(sorted(self._quarantine.items())),
            }


def _bounded_output(value: object) -> bytes:
    if not isinstance(value, bytes) or not 1 <= len(value) <= MAX_NUMERIC_PROBE_BYTES:
        raise ValueError("numeric probe output is invalid")
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
