"""Correctness-neutral heterogeneous draft and GPU verification execution."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

from .backend_engine import BackendEngineRegistry, BackendEngineRequest
from .device_resources import DeviceResourceRequest, UnifiedDeviceResourceLedger
from .execution import ExecutionBackend, WorkloadPhase
from .inference_request import InferenceRequestContext

MAX_DRAFT_TOKENS = 64
MAX_OUTPUT_TOKENS = 65_536
MAX_SPECULATIVE_ROUNDS = 4_096
MAX_SPECULATIVE_PROFILE_BYTES = 16 * 1024
SPECULATIVE_PROFILE_SCHEMA_VERSION = 1
_DRAFT_BACKENDS = frozenset({ExecutionBackend.CPU, ExecutionBackend.COREML_DRAFT})
_VERIFY_BACKENDS = frozenset({
    ExecutionBackend.VLLM_METAL,
    ExecutionBackend.NATIVE_MLX,
    ExecutionBackend.NATIVE_METAL,
})


@dataclass(frozen=True, slots=True)
class SpeculativeExecutionProfile:
    model_hash: str
    precision: str
    draft_backend: ExecutionBackend
    verify_backend: ExecutionBackend
    sample_count: int
    baseline_latency_nanoseconds: int
    speculative_latency_nanoseconds: int
    outputs_match: bool
    profile_id: str

    def __post_init__(self) -> None:
        if (
            len(self.model_hash) != 64
            or not self.precision
            or self.draft_backend not in _DRAFT_BACKENDS
            or self.verify_backend not in _VERIFY_BACKENDS
            or not 3 <= self.sample_count <= 1024
            or self.baseline_latency_nanoseconds <= 0
            or self.speculative_latency_nanoseconds <= 0
            or len(self.profile_id) != 64
            or self.profile_id != _profile_id(self)
        ):
            raise ValueError("invalid speculative execution profile")

    @property
    def qualified(self) -> bool:
        return (
            self.outputs_match
            and self.speculative_latency_nanoseconds * 100
            <= self.baseline_latency_nanoseconds * 95
        )

    @classmethod
    def create(
        cls,
        *,
        model_hash: str,
        precision: str,
        draft_backend: ExecutionBackend,
        verify_backend: ExecutionBackend,
        sample_count: int,
        baseline_latency_nanoseconds: int,
        speculative_latency_nanoseconds: int,
        outputs_match: bool,
    ) -> "SpeculativeExecutionProfile":
        values = {
            "model_hash": model_hash,
            "precision": precision,
            "draft_backend": draft_backend,
            "verify_backend": verify_backend,
            "sample_count": sample_count,
            "baseline_latency_nanoseconds": baseline_latency_nanoseconds,
            "speculative_latency_nanoseconds": speculative_latency_nanoseconds,
            "outputs_match": outputs_match,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "profile_id", "")
        return cls(**values, profile_id=_profile_id(provisional))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SPECULATIVE_PROFILE_SCHEMA_VERSION,
            "model_hash": self.model_hash,
            "precision": self.precision,
            "draft_backend": self.draft_backend.value,
            "verify_backend": self.verify_backend.value,
            "sample_count": self.sample_count,
            "baseline_latency_nanoseconds": self.baseline_latency_nanoseconds,
            "speculative_latency_nanoseconds": self.speculative_latency_nanoseconds,
            "outputs_match": self.outputs_match,
            "profile_id": self.profile_id,
        }


def save_speculative_profile(profile: SpeculativeExecutionProfile, path: Path) -> Path:
    """Persist only a qualified profile as an owner-only atomic artifact."""
    if not isinstance(profile, SpeculativeExecutionProfile) or not profile.qualified:
        raise ValueError("only a qualified speculative profile may be persisted")
    destination = path.expanduser()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.parent.chmod(0o700)
    _validate_private_directory(destination.parent)
    encoded = json.dumps(
        profile.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"
    if len(encoded) > MAX_SPECULATIVE_PROFILE_BYTES:
        raise ValueError("speculative profile exceeds bounded size")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
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


def load_speculative_profile(
    path: Path, *, model_hash: str, precision: str
) -> SpeculativeExecutionProfile:
    """Load a qualified profile only for its exact model and precision identity."""
    _validate_private_directory(path.parent)
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
        or not 0 < info.st_size <= MAX_SPECULATIVE_PROFILE_BYTES
    ):
        raise ValueError("speculative profile must be a bounded private regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "schema_version", "model_hash", "precision", "draft_backend", "verify_backend",
        "sample_count", "baseline_latency_nanoseconds", "speculative_latency_nanoseconds",
        "outputs_match", "profile_id",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != fields
        or payload.get("schema_version") != SPECULATIVE_PROFILE_SCHEMA_VERSION
        or payload.get("model_hash") != model_hash
        or payload.get("precision") != precision
    ):
        raise ValueError("speculative profile identity mismatch")
    try:
        profile = SpeculativeExecutionProfile(
            model_hash=payload["model_hash"],
            precision=payload["precision"],
            draft_backend=ExecutionBackend(payload["draft_backend"]),
            verify_backend=ExecutionBackend(payload["verify_backend"]),
            sample_count=payload["sample_count"],
            baseline_latency_nanoseconds=payload["baseline_latency_nanoseconds"],
            speculative_latency_nanoseconds=payload["speculative_latency_nanoseconds"],
            outputs_match=payload["outputs_match"],
            profile_id=payload["profile_id"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid speculative profile values") from error
    if not profile.qualified:
        raise ValueError("persisted speculative profile is not qualified")
    return profile


def _validate_private_directory(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("speculative profile directory must be owner-only")


class DraftExecutor(Protocol):
    def __call__(
        self, prefix: tuple[int, ...], maximum_tokens: int, context: InferenceRequestContext
    ) -> tuple[int, ...]: ...


class VerifyExecutor(Protocol):
    def __call__(
        self, prefix: tuple[int, ...], proposed: tuple[int, ...], context: InferenceRequestContext
    ) -> tuple[int, ...]: ...


class BackendRegistrySpeculativeAdapter:
    """Binds speculative phases to the production backend registry contract."""

    def __init__(
        self,
        registry: BackendEngineRegistry[tuple[int, ...]],
        profile: SpeculativeExecutionProfile,
        *,
        model_architecture: str,
    ) -> None:
        if (
            not isinstance(registry, BackendEngineRegistry)
            or not isinstance(profile, SpeculativeExecutionProfile)
            or not profile.qualified
            or not isinstance(model_architecture, str)
            or not 1 <= len(model_architecture) <= 128
        ):
            raise ValueError("invalid speculative backend registry adapter")
        self._registry = registry
        self.profile = profile
        self.model_architecture = model_architecture

    def draft(
        self,
        prefix: tuple[int, ...],
        maximum_tokens: int,
        context: InferenceRequestContext,
    ) -> tuple[int, ...]:
        request = BackendEngineRequest(
            "speculative.draft",
            WorkloadPhase.DRAFT,
            self.profile.precision,
            self.model_architecture,
            (self.profile.draft_backend,),
            {"prefix_token_ids": prefix, "maximum_tokens": maximum_tokens},
        )
        result = self._registry.execute(request, context)
        if result.backend is not self.profile.draft_backend:
            raise RuntimeError("speculative draft backend identity mismatch")
        return _validated_tokens(result.value, maximum_tokens, "draft")

    def verify(
        self,
        prefix: tuple[int, ...],
        proposed: tuple[int, ...],
        context: InferenceRequestContext,
    ) -> tuple[int, ...]:
        request = BackendEngineRequest(
            "speculative.verify",
            WorkloadPhase.VERIFY,
            self.profile.precision,
            self.model_architecture,
            (self.profile.verify_backend,),
            {"prefix_token_ids": prefix, "proposed_token_ids": proposed},
        )
        result = self._registry.execute(request, context)
        if result.backend is not self.profile.verify_backend:
            raise RuntimeError("speculative verify backend identity mismatch")
        return _validated_tokens(result.value, len(proposed), "verify")


@dataclass(frozen=True, slots=True)
class SpeculativeExecutionResult:
    token_ids: tuple[int, ...]
    drafted_tokens: int
    accepted_draft_tokens: int
    verifier_corrections: int
    rounds: int
    elapsed_nanoseconds: int
    output_fingerprint: str
    profile_id: str

    def __post_init__(self) -> None:
        if (
            not self.token_ids
            or len(self.token_ids) > MAX_OUTPUT_TOKENS
            or any(type(token) is not int or token < 0 for token in self.token_ids)
            or not 1 <= self.rounds <= MAX_SPECULATIVE_ROUNDS
            or self.drafted_tokens < len(self.token_ids)
            or not 0 <= self.accepted_draft_tokens <= self.drafted_tokens
            or not 0 <= self.verifier_corrections <= self.rounds
            or self.elapsed_nanoseconds <= 0
            or self.output_fingerprint != _tokens_hash(self.token_ids)
            or len(self.profile_id) != 64
        ):
            raise ValueError("invalid speculative execution result")


class HeterogeneousSpeculativeExecutor:
    def __init__(
        self,
        ledger: UnifiedDeviceResourceLedger,
        profile: SpeculativeExecutionProfile,
        *,
        draft_memory_bytes: int,
        verify_memory_bytes: int,
        maximum_draft_tokens: int = 8,
    ) -> None:
        if (
            not isinstance(ledger, UnifiedDeviceResourceLedger)
            or not isinstance(profile, SpeculativeExecutionProfile)
            or not profile.qualified
            or type(draft_memory_bytes) is not int
            or type(verify_memory_bytes) is not int
            or draft_memory_bytes < 0
            or verify_memory_bytes < 0
            or type(maximum_draft_tokens) is not int
            or not 1 <= maximum_draft_tokens <= MAX_DRAFT_TOKENS
        ):
            raise ValueError("invalid or unqualified speculative executor configuration")
        self._ledger = ledger
        self.profile = profile
        self.draft_memory_bytes = draft_memory_bytes
        self.verify_memory_bytes = verify_memory_bytes
        self.maximum_draft_tokens = maximum_draft_tokens

    def execute(
        self,
        context: InferenceRequestContext,
        *,
        maximum_output_tokens: int,
        draft: DraftExecutor,
        verify: VerifyExecutor,
        publish: Callable[[tuple[int, ...]], None] | None = None,
    ) -> SpeculativeExecutionResult:
        if (
            not isinstance(context, InferenceRequestContext)
            or type(maximum_output_tokens) is not int
            or not 1 <= maximum_output_tokens <= MAX_OUTPUT_TOKENS
            or not callable(draft)
            or not callable(verify)
            or (publish is not None and not callable(publish))
        ):
            raise ValueError("invalid speculative execution request")
        started = time.monotonic_ns()
        output: list[int] = []
        drafted = accepted = corrections = rounds = 0
        while len(output) < maximum_output_tokens:
            context.raise_if_cancelled()
            if rounds >= MAX_SPECULATIVE_ROUNDS:
                raise RuntimeError("speculative execution round limit exceeded")
            count = min(self.maximum_draft_tokens, maximum_output_tokens - len(output))
            proposal = self._run_stage(
                self.profile.draft_backend,
                self.draft_memory_bytes,
                lambda: draft(tuple(output), count, context),
            )
            proposal = _validated_tokens(proposal, count, "draft")
            if not proposal:
                raise RuntimeError("draft backend returned no token")
            drafted += len(proposal)
            context.raise_if_cancelled()
            authoritative = self._run_stage(
                self.profile.verify_backend,
                self.verify_memory_bytes,
                lambda: verify(tuple(output), proposal, context),
            )
            authoritative = _validated_tokens(authoritative, len(proposal), "verify")
            if not authoritative:
                raise RuntimeError("verify backend returned no token")
            matching = 0
            for left, right in zip(proposal, authoritative):
                if left != right:
                    break
                matching += 1
            accepted += matching
            if matching < len(authoritative):
                committed = authoritative[: matching + 1]
                corrections += 1
            else:
                committed = authoritative
            if len(output) + len(committed) > maximum_output_tokens:
                committed = committed[: maximum_output_tokens - len(output)]
            output.extend(committed)
            rounds += 1
            context.raise_if_cancelled()
            if publish is not None:
                publish(committed)
        tokens = tuple(output)
        return SpeculativeExecutionResult(
            token_ids=tokens,
            drafted_tokens=drafted,
            accepted_draft_tokens=accepted,
            verifier_corrections=corrections,
            rounds=rounds,
            elapsed_nanoseconds=max(1, time.monotonic_ns() - started),
            output_fingerprint=_tokens_hash(tokens),
            profile_id=self.profile.profile_id,
        )

    def _run_stage(self, backend: ExecutionBackend, memory_bytes: int, operation):
        reservation = self._ledger.reserve(DeviceResourceRequest.for_backend(backend, memory_bytes))
        try:
            return operation()
        finally:
            self._ledger.release(reservation.reservation_id)


def _validated_tokens(tokens: object, maximum: int, label: str) -> tuple[int, ...]:
    if (
        not isinstance(tokens, tuple)
        or len(tokens) > maximum
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise ValueError(f"{label} backend returned invalid tokens")
    return tokens


def _tokens_hash(tokens: tuple[int, ...]) -> str:
    digest = hashlib.sha256(b"vllm-apple-speculative-tokens-v1\0")
    for token in tokens:
        digest.update(token.to_bytes(8, "big"))
    return digest.hexdigest()


def _profile_id(profile: SpeculativeExecutionProfile) -> str:
    payload = asdict(profile)
    payload.pop("profile_id", None)
    payload["draft_backend"] = profile.draft_backend.value
    payload["verify_backend"] = profile.verify_backend.value
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b"vllm-apple-speculative-profile-v1\0" + encoded).hexdigest()
