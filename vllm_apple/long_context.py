from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


LONG_CONTEXT_SCHEMA_VERSION = 1
MAX_CONTEXT_STAGES = 16
MAX_CONTEXT_TOKENS = 1_000_000
MAX_LONG_CONTEXT_REPORT_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class LongContextObservation:
    target_tokens: int
    actual_prompt_tokens: int
    retrieval_score: float
    ttft_ms: float
    tpot_ms: float
    tokens_per_second: float
    load_peak_rss_bytes: int
    steady_state_rss_bytes: int
    state_bytes: int

    def __post_init__(self) -> None:
        numeric = (self.retrieval_score, self.ttft_ms, self.tpot_ms, self.tokens_per_second)
        byte_values = (
            self.load_peak_rss_bytes,
            self.steady_state_rss_bytes,
            self.state_bytes,
        )
        if not 1 <= self.target_tokens <= MAX_CONTEXT_TOKENS:
            raise ValueError("target_tokens is outside the supported range")
        if self.actual_prompt_tokens <= 0:
            raise ValueError("actual_prompt_tokens must be positive")
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise ValueError("long-context metrics must be finite and non-negative")
        if self.retrieval_score > 1:
            raise ValueError("retrieval_score must be at most one")
        if any(value < 0 for value in byte_values):
            raise ValueError("long-context memory metrics must not be negative")


@dataclass(frozen=True, slots=True)
class LongContextStageResult:
    target_tokens: int
    actual_prompt_tokens: int | None
    status: str
    retrieval_score: float | None
    ttft_ms: float | None
    tpot_ms: float | None
    tokens_per_second: float | None
    load_peak_rss_bytes: int | None
    steady_state_rss_bytes: int | None
    state_bytes: int | None
    error_code: str | None

    def __post_init__(self) -> None:
        if self.status not in {"passed", "failed", "skipped"}:
            raise ValueError("invalid long-context stage status")
        if self.status == "passed" and self.error_code is not None:
            raise ValueError("passed stage cannot contain an error")
        if self.status != "passed" and not self.error_code:
            raise ValueError("failed or skipped stage requires an error code")


class LongContextEvaluationError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


class LongContextEvaluator:
    """Sequential fail-fast evaluator that stores measurements, never prompt content."""

    def __init__(
        self,
        *,
        model_id: str,
        hardware_fingerprint: str,
        memory_ceiling_bytes: int,
        backend: str = "unknown",
        minimum_retrieval_score: float = 1.0,
        token_tolerance_ratio: float = 0.05,
    ) -> None:
        if not model_id or not hardware_fingerprint or not backend:
            raise ValueError("evaluation identity must not be empty")
        if memory_ceiling_bytes <= 0:
            raise ValueError("memory_ceiling_bytes must be positive")
        if not 0 <= minimum_retrieval_score <= 1:
            raise ValueError("minimum_retrieval_score must be in [0, 1]")
        if not 0 <= token_tolerance_ratio <= 0.25:
            raise ValueError("token_tolerance_ratio must be in [0, 0.25]")
        self.model_id = model_id
        self.hardware_fingerprint = hardware_fingerprint
        self.memory_ceiling_bytes = memory_ceiling_bytes
        self.backend = backend
        self.minimum_retrieval_score = minimum_retrieval_score
        self.token_tolerance_ratio = token_tolerance_ratio

    def evaluate(
        self,
        stages: Sequence[int],
        measure: Callable[[int], LongContextObservation],
    ) -> dict[str, Any]:
        targets = tuple(stages)
        self._validate_stages(targets)
        results: list[LongContextStageResult] = []
        halted = False
        for target in targets:
            if halted:
                results.append(self._empty_result(target, "prior_stage_failed", "skipped"))
                continue
            try:
                observation = measure(target)
                if observation.target_tokens != target:
                    raise LongContextEvaluationError(
                        "stage_mismatch", "adapter returned a different target stage"
                    )
                error = self._observation_error(observation)
                if error is not None:
                    results.append(self._result(observation, "failed", error))
                    halted = True
                else:
                    results.append(self._result(observation, "passed", None))
            except LongContextEvaluationError as error:
                results.append(self._empty_result(target, error.code, "failed"))
                halted = True
        payload = {
            "schema_version": LONG_CONTEXT_SCHEMA_VERSION,
            "model_id": self.model_id,
            "hardware_fingerprint": self.hardware_fingerprint,
            "backend": self.backend,
            "memory_ceiling_bytes": self.memory_ceiling_bytes,
            "minimum_retrieval_score": self.minimum_retrieval_score,
            "token_tolerance_ratio": self.token_tolerance_ratio,
            "passed": all(result.status == "passed" for result in results),
            "stages": [asdict(result) for result in results],
            "storage": {"raw_prompt_count": 0, "raw_output_count": 0},
        }
        payload["evaluation_id"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        return payload

    @staticmethod
    def _validate_stages(stages: tuple[int, ...]) -> None:
        if not stages or len(stages) > MAX_CONTEXT_STAGES:
            raise ValueError("context stages must contain between 1 and 16 entries")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= MAX_CONTEXT_TOKENS
            for value in stages
        ):
            raise ValueError("context stage is invalid")
        if any(left >= right for left, right in zip(stages, stages[1:])):
            raise ValueError("context stages must be strictly increasing")

    def _observation_error(self, observation: LongContextObservation) -> str | None:
        tolerance = max(64, int(observation.target_tokens * self.token_tolerance_ratio))
        if abs(observation.actual_prompt_tokens - observation.target_tokens) > tolerance:
            return "prompt_token_mismatch"
        if observation.retrieval_score < self.minimum_retrieval_score:
            return "retrieval_quality_failed"
        peak = max(observation.load_peak_rss_bytes, observation.steady_state_rss_bytes)
        if peak > self.memory_ceiling_bytes:
            return "memory_ceiling_exceeded"
        return None

    @staticmethod
    def _result(
        observation: LongContextObservation, status: str, error_code: str | None
    ) -> LongContextStageResult:
        return LongContextStageResult(
            **asdict(observation), status=status, error_code=error_code
        )

    @staticmethod
    def _empty_result(target: int, code: str, status: str) -> LongContextStageResult:
        return LongContextStageResult(
            target_tokens=target,
            actual_prompt_tokens=None,
            status=status,
            retrieval_score=None,
            ttft_ms=None,
            tpot_ms=None,
            tokens_per_second=None,
            load_peak_rss_bytes=None,
            steady_state_rss_bytes=None,
            state_bytes=None,
            error_code=code,
        )


def save_long_context_report(report: dict[str, Any], path: Path) -> Path:
    encoded = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    if not 0 < len(encoded) <= MAX_LONG_CONTEXT_REPORT_BYTES:
        raise ValueError("long-context report exceeded 1 MiB")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or parent.st_mode & 0o077
    ):
        raise ValueError("long-context report directory must be private")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path
