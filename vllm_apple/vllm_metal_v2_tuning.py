from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import statistics
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from .hardware import default_application_support
from .kernel_profile import ModelKernelShapeProfile

VLLM_METAL_V2_TUNING_SCHEMA_VERSION = 1
VLLM_METAL_V2_TIE_RATIO = 1.02
VLLM_METAL_V2_PARTITION_SIZE = 512
MAX_V2_SHAPES = 16
MAX_V2_SAMPLES = 9
MAX_V2_PROFILE_BYTES = 512 * 1024


class V2PagedAttentionFamily(str, Enum):
    NAX_PREFILL = "nax_prefill"
    TILED_PREFILL = "tiled_prefill"
    PER_TOKEN = "per_token"
    SPLIT_KV = "split_kv"


@dataclass(frozen=True, slots=True)
class V2PagedAttentionShape:
    context_tokens: int
    query_tokens: int
    sequences: int
    query_heads: int
    kv_heads: int
    head_size: int
    block_size: int
    gpu_cores: int
    query_dtype: str = "float16"
    cache_dtype: str = "float16"
    turboquant: bool = False
    window_seqlen_q: int = 1

    def __post_init__(self) -> None:
        values = (
            self.context_tokens,
            self.query_tokens,
            self.sequences,
            self.query_heads,
            self.kv_heads,
            self.head_size,
            self.block_size,
            self.gpu_cores,
            self.window_seqlen_q,
        )
        if any(isinstance(value, bool) or value <= 0 for value in values):
            raise ValueError("vLLM-Metal v2 shape values must be positive integers")
        if self.query_tokens < self.sequences or self.query_heads % self.kv_heads:
            raise ValueError("invalid vLLM-Metal v2 query/KV shape")
        if self.query_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("unsupported vLLM-Metal query dtype")
        if self.cache_dtype not in {"float16", "bfloat16", "float32", "int8"}:
            raise ValueError("unsupported vLLM-Metal cache dtype")

    @property
    def has_prefill(self) -> bool:
        return self.query_tokens > self.sequences

    @property
    def window_batch(self) -> bool:
        return self.has_prefill and self.window_seqlen_q > 1 and self.head_size <= 256


@dataclass(frozen=True, slots=True)
class V2DispatchConfiguration:
    family: V2PagedAttentionFamily
    threads: int
    tile_query: int = 0
    tile_kv: int = 0
    partition_size: int = 0

    def __post_init__(self) -> None:
        if self.threads not in {32, 64, 128, 256}:
            raise ValueError("vLLM-Metal v2 threads must be 32, 64, 128, or 256")
        if any(value < 0 for value in (self.tile_query, self.tile_kv, self.partition_size)):
            raise ValueError("vLLM-Metal v2 dispatch dimensions cannot be negative")
        if self.family is V2PagedAttentionFamily.NAX_PREFILL:
            if (self.threads, self.tile_query, self.tile_kv, self.partition_size) != (
                128,
                64,
                0,
                0,
            ):
                raise ValueError("invalid NAX dispatch configuration")
        elif self.family is V2PagedAttentionFamily.TILED_PREFILL:
            if not self.tile_query or not self.tile_kv or self.partition_size:
                raise ValueError("invalid tiled dispatch configuration")
        elif self.family is V2PagedAttentionFamily.PER_TOKEN:
            if (self.threads, self.tile_query, self.tile_kv, self.partition_size) != (
                256,
                0,
                0,
                0,
            ):
                raise ValueError("invalid per-token dispatch configuration")
        elif self.family is V2PagedAttentionFamily.SPLIT_KV and (
            self.threads != 256
            or self.tile_query
            or self.tile_kv
            or self.partition_size != VLLM_METAL_V2_PARTITION_SIZE
        ):
            raise ValueError("invalid split-KV dispatch configuration")


@dataclass(frozen=True, slots=True)
class V2CandidateResult:
    configuration: V2DispatchConfiguration
    passed: bool
    median_latency_nanoseconds: int
    sample_latencies_nanoseconds: tuple[int, ...]
    output_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise ValueError("native v2 correctness result must be boolean")
        if not 1 <= len(self.sample_latencies_nanoseconds) <= MAX_V2_SAMPLES:
            raise ValueError("invalid native v2 sample count")
        if any(value <= 0 for value in self.sample_latencies_nanoseconds):
            raise ValueError("native v2 latency samples must be positive")
        expected = int(statistics.median(self.sample_latencies_nanoseconds))
        if self.median_latency_nanoseconds != expected:
            raise ValueError("native v2 median does not match samples")
        if len(self.output_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.output_digest
        ):
            raise ValueError("native v2 output digest must be SHA-256")

    def to_dict(self) -> dict[str, object]:
        return {
            "configuration": {
                **asdict(self.configuration),
                "family": self.configuration.family.value,
            },
            "passed": self.passed,
            "median_latency_nanoseconds": self.median_latency_nanoseconds,
            "sample_latencies_nanoseconds": list(self.sample_latencies_nanoseconds),
            "output_digest": self.output_digest,
        }


@dataclass(frozen=True, slots=True)
class V2ShapeTuningDecision:
    shape: V2PagedAttentionShape
    winner: V2DispatchConfiguration
    candidates: tuple[V2CandidateResult, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.candidates) <= 4:
            raise ValueError("native v2 tuning requires 1 to 4 candidates")
        _validate_candidate_set(self.shape, self.candidates)
        if self.winner != select_v2_winner(self.candidates):
            raise ValueError("native v2 winner does not match candidate results")

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": asdict(self.shape),
            "winner": {
                **asdict(self.winner),
                "family": self.winner.family.value,
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class VLLMMetalV2TuningProfile:
    schema_version: int
    profile_id: str
    hardware_fingerprint: str
    source_fingerprint: str
    decisions: tuple[V2ShapeTuningDecision, ...]

    def __post_init__(self) -> None:
        if self.schema_version != VLLM_METAL_V2_TUNING_SCHEMA_VERSION:
            raise ValueError("unsupported native v2 tuning schema")
        for value in (self.profile_id, self.hardware_fingerprint, self.source_fingerprint):
            if not value or len(value) > 128:
                raise ValueError("invalid native v2 tuning identity")
        if not 1 <= len(self.decisions) <= MAX_V2_SHAPES:
            raise ValueError("native v2 tuning profile requires 1 to 16 shapes")
        if self.profile_id != _profile_id(
            self.hardware_fingerprint, self.source_fingerprint, self.decisions
        ):
            raise ValueError("native v2 tuning profile ID mismatch")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "hardware_fingerprint": self.hardware_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "tie_ratio": VLLM_METAL_V2_TIE_RATIO,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


Measurement = tuple[bool, int, str]


def candidate_configurations(
    shape: V2PagedAttentionShape, *, nax_available: bool = True
) -> tuple[V2DispatchConfiguration, ...]:
    if shape.has_prefill and not shape.window_batch:
        candidates: list[V2DispatchConfiguration] = []
        if (
            nax_available
            and not shape.turboquant
            and shape.query_dtype in {"float16", "bfloat16"}
            and shape.head_size in {64, 96, 128, 256, 512}
            and shape.block_size in {8, 16, 32}
        ):
            candidates.append(V2DispatchConfiguration(V2PagedAttentionFamily.NAX_PREFILL, 128, 64))
        tiled = _tiled_configuration(shape)
        if tiled is not None:
            candidates.append(tiled)
        candidates.append(V2DispatchConfiguration(V2PagedAttentionFamily.PER_TOKEN, 256))
        return tuple(candidates)

    candidates = [V2DispatchConfiguration(V2PagedAttentionFamily.PER_TOKEN, 256)]
    partitions = math.ceil(shape.context_tokens / VLLM_METAL_V2_PARTITION_SIZE)
    gate_grid = shape.query_heads * shape.query_tokens
    if not shape.has_prefill and partitions >= 2 and gate_grid < shape.gpu_cores * 8:
        candidates.append(
            V2DispatchConfiguration(
                V2PagedAttentionFamily.SPLIT_KV,
                256,
                partition_size=VLLM_METAL_V2_PARTITION_SIZE,
            )
        )
    return tuple(candidates)


def tune_v2_shape(
    shape: V2PagedAttentionShape,
    measure: Callable[[V2PagedAttentionShape, V2DispatchConfiguration], Measurement],
    *,
    samples: int = 3,
    nax_available: bool = True,
) -> V2ShapeTuningDecision:
    if not 1 <= samples <= MAX_V2_SAMPLES:
        raise ValueError("native v2 samples must be between 1 and 9")
    results: list[V2CandidateResult] = []
    for configuration in candidate_configurations(shape, nax_available=nax_available):
        measurements = tuple(measure(shape, configuration) for _ in range(samples))
        passed = all(item[0] for item in measurements)
        latencies = tuple(item[1] for item in measurements)
        digests = {item[2] for item in measurements}
        if len(digests) != 1:
            passed = False
        results.append(
            V2CandidateResult(
                configuration,
                passed,
                int(statistics.median(latencies)),
                latencies,
                next(iter(digests)) if len(digests) == 1 else "0" * 64,
            )
        )
    # The per-token path is the least-specialized native-v2 implementation and
    # acts as the cross-family output reference. A helper must additionally
    # compare each result with its CPU/MLX reference through `passed`.
    reference = next(
        (
            result.output_digest
            for result in results
            if result.configuration.family is V2PagedAttentionFamily.PER_TOKEN
            and result.passed
        ),
        None,
    )
    if reference is not None:
        results = [
            result
            if not result.passed or result.output_digest == reference
            else V2CandidateResult(
                result.configuration,
                False,
                result.median_latency_nanoseconds,
                result.sample_latencies_nanoseconds,
                result.output_digest,
            )
            for result in results
        ]
    candidates = tuple(results)
    return V2ShapeTuningDecision(shape, select_v2_winner(candidates), candidates)


def build_v2_tuning_profile(
    decisions: tuple[V2ShapeTuningDecision, ...],
    *,
    hardware_fingerprint: str,
    source_fingerprint: str,
) -> VLLMMetalV2TuningProfile:
    return VLLMMetalV2TuningProfile(
        VLLM_METAL_V2_TUNING_SCHEMA_VERSION,
        _profile_id(hardware_fingerprint, source_fingerprint, decisions),
        hardware_fingerprint,
        source_fingerprint,
        decisions,
    )


def tune_v2_model_profile(
    model_profile: ModelKernelShapeProfile,
    measure: Callable[[V2PagedAttentionShape, V2DispatchConfiguration], Measurement],
    *,
    hardware_fingerprint: str,
    source_fingerprint: str,
    gpu_cores: int,
    samples: int = 3,
    maximum_shapes: int = MAX_V2_SHAPES,
    prefill_query_tokens: int = 128,
    nax_available: bool = True,
) -> VLLMMetalV2TuningProfile:
    if isinstance(gpu_cores, bool) or gpu_cores <= 0:
        raise ValueError("native v2 tuning requires a positive GPU core count")
    if not 1 <= maximum_shapes <= MAX_V2_SHAPES:
        raise ValueError("native v2 maximum shapes must be between 1 and 16")
    if isinstance(prefill_query_tokens, bool) or prefill_query_tokens <= 0:
        raise ValueError("native v2 prefill query tokens must be positive")
    cache_dtype = {1: "int8", 2: "float16", 4: "float32"}[model_profile.kv_dtype_bytes]
    shapes: list[V2PagedAttentionShape] = []
    for source_shape in model_profile.shapes:
        common = {
            "context_tokens": source_shape.context_tokens,
            "sequences": source_shape.batch_size,
            "query_heads": source_shape.query_heads,
            "kv_heads": source_shape.kv_heads,
            "head_size": source_shape.head_dimension,
            "block_size": source_shape.block_tokens,
            "gpu_cores": gpu_cores,
            "query_dtype": "float16",
            "cache_dtype": cache_dtype,
        }
        shapes.append(V2PagedAttentionShape(query_tokens=source_shape.batch_size, **common))
        prefill = min(source_shape.context_tokens, prefill_query_tokens)
        if prefill > source_shape.batch_size:
            shapes.append(V2PagedAttentionShape(query_tokens=prefill, **common))
        if len(shapes) >= maximum_shapes:
            break
    decisions = tuple(
        tune_v2_shape(
            shape,
            measure,
            samples=samples,
            nax_available=nax_available,
        )
        for shape in shapes[:maximum_shapes]
    )
    return build_v2_tuning_profile(
        decisions,
        hardware_fingerprint=hardware_fingerprint,
        source_fingerprint=source_fingerprint,
    )


def default_v2_tuning_profile_path(profile: VLLMMetalV2TuningProfile) -> Path:
    for value in (profile.hardware_fingerprint, profile.source_fingerprint):
        if value in {".", ".."} or Path(value).name != value:
            raise ValueError("invalid native v2 fingerprint path component")
    return (
        default_application_support()
        / "profiles"
        / "vllm-metal-v2"
        / profile.hardware_fingerprint
        / profile.source_fingerprint
        / f"{profile.profile_id}.json"
    )


def save_v2_tuning_profile(profile: VLLMMetalV2TuningProfile, path: Path | None = None) -> Path:
    destination = path or default_v2_tuning_profile_path(profile)
    encoded = (
        json.dumps(profile.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_V2_PROFILE_BYTES:
        raise ValueError("native v2 tuning profile exceeded 512 KiB")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = destination.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise ValueError("native v2 tuning directory must be private")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def load_v2_tuning_profile(
    path: Path,
    *,
    hardware_fingerprint: str,
    source_fingerprint: str,
) -> VLLMMetalV2TuningProfile:
    attributes = path.lstat()
    if (
        not stat.S_ISREG(attributes.st_mode)
        or attributes.st_uid != os.getuid()
        or attributes.st_mode & 0o077
    ):
        raise ValueError("native v2 profile must be a private current-user regular file")
    if not 1 <= attributes.st_size <= MAX_V2_PROFILE_BYTES:
        raise ValueError("native v2 tuning profile size is invalid")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "profile_id",
        "hardware_fingerprint",
        "source_fingerprint",
        "tie_ratio",
        "decisions",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("invalid native v2 tuning profile fields")
    if (
        payload["hardware_fingerprint"] != hardware_fingerprint
        or payload["source_fingerprint"] != source_fingerprint
        or payload["tie_ratio"] != VLLM_METAL_V2_TIE_RATIO
    ):
        raise ValueError("native v2 tuning profile identity mismatch")
    raw_decisions = payload["decisions"]
    if not isinstance(raw_decisions, list) or not 1 <= len(raw_decisions) <= MAX_V2_SHAPES:
        raise ValueError("invalid native v2 tuning decision count")
    try:
        decisions = tuple(_parse_decision(value) for value in raw_decisions)
        return VLLMMetalV2TuningProfile(
            payload["schema_version"],
            payload["profile_id"],
            payload["hardware_fingerprint"],
            payload["source_fingerprint"],
            decisions,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid native v2 tuning profile values") from error


def select_v2_winner(
    candidates: tuple[V2CandidateResult, ...],
) -> V2DispatchConfiguration:
    passing = [candidate for candidate in candidates if candidate.passed]
    if not passing:
        raise ValueError("native v2 tuning has no correctness-passing candidate")
    fastest = min(candidate.median_latency_nanoseconds for candidate in passing)
    equivalent = [
        candidate
        for candidate in passing
        if candidate.median_latency_nanoseconds <= fastest * VLLM_METAL_V2_TIE_RATIO
    ]
    family_rank = {
        V2PagedAttentionFamily.PER_TOKEN: 0,
        V2PagedAttentionFamily.NAX_PREFILL: 1,
        V2PagedAttentionFamily.TILED_PREFILL: 2,
        V2PagedAttentionFamily.SPLIT_KV: 3,
    }
    return min(
        equivalent,
        key=lambda item: (
            family_rank[item.configuration.family],
            item.configuration.threads,
            item.configuration.tile_query,
            item.configuration.tile_kv,
        ),
    ).configuration


def _tiled_configuration(
    shape: V2PagedAttentionShape,
) -> V2DispatchConfiguration | None:
    if shape.turboquant or shape.query_dtype == "float32" or shape.query_dtype != shape.cache_dtype:
        return None
    mapping = {
        64: (32, 32, 128),
        96: (32, 32, 128),
        128: (32, 32, 128),
        256: (16, 16, 64),
        512: (8, 8, 32),
    }
    values = mapping.get(shape.head_size)
    if values is None:
        return None
    tile_query, tile_kv, threads = values
    return V2DispatchConfiguration(
        V2PagedAttentionFamily.TILED_PREFILL,
        threads,
        tile_query,
        tile_kv,
    )


def _parse_configuration(value: object) -> V2DispatchConfiguration:
    expected = {"family", "threads", "tile_query", "tile_kv", "partition_size"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("invalid native v2 dispatch fields")
    return V2DispatchConfiguration(
        V2PagedAttentionFamily(value["family"]),
        value["threads"],
        value["tile_query"],
        value["tile_kv"],
        value["partition_size"],
    )


def _parse_decision(value: object) -> V2ShapeTuningDecision:
    if not isinstance(value, dict) or set(value) != {"shape", "winner", "candidates"}:
        raise ValueError("invalid native v2 decision fields")
    shape_value = value["shape"]
    raw_candidates = value["candidates"]
    if not isinstance(shape_value, dict) or not isinstance(raw_candidates, list):
        raise ValueError("invalid native v2 decision values")
    candidates: list[V2CandidateResult] = []
    for candidate in raw_candidates:
        expected = {
            "configuration",
            "passed",
            "median_latency_nanoseconds",
            "sample_latencies_nanoseconds",
            "output_digest",
        }
        if not isinstance(candidate, dict) or set(candidate) != expected:
            raise ValueError("invalid native v2 candidate fields")
        samples = candidate["sample_latencies_nanoseconds"]
        if not isinstance(samples, list):
            raise ValueError("invalid native v2 latency samples")
        candidates.append(
            V2CandidateResult(
                _parse_configuration(candidate["configuration"]),
                candidate["passed"],
                candidate["median_latency_nanoseconds"],
                tuple(samples),
                candidate["output_digest"],
            )
        )
    return V2ShapeTuningDecision(
        V2PagedAttentionShape(**shape_value),
        _parse_configuration(value["winner"]),
        tuple(candidates),
    )


def _validate_candidate_set(
    shape: V2PagedAttentionShape, candidates: tuple[V2CandidateResult, ...]
) -> None:
    expected = set(candidate_configurations(shape))
    actual = [candidate.configuration for candidate in candidates]
    if len(actual) != len(set(actual)) or not set(actual).issubset(expected):
        raise ValueError("native v2 candidate is duplicated or ineligible")


def _profile_id(
    hardware_fingerprint: str,
    source_fingerprint: str,
    decisions: tuple[V2ShapeTuningDecision, ...],
) -> str:
    identity = {
        "schema_version": VLLM_METAL_V2_TUNING_SCHEMA_VERSION,
        "hardware_fingerprint": hardware_fingerprint,
        "source_fingerprint": source_fingerprint,
        "decisions": [decision.to_dict() for decision in decisions],
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
