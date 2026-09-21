"""Evidence-bound prefill/decode CPU, Metal and Unified Memory profile."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PhaseResourceEntry:
    phase: str
    cpu_operator: str
    metal_operator: str
    cpu_median_nanoseconds: int
    metal_median_nanoseconds: int
    metal_speedup: float
    attention_median_nanoseconds: int
    unified_memory_bandwidth_bytes_per_second: float
    recommended_backend: str

    def __post_init__(self) -> None:
        if (self.phase not in {"prefill", "decode"}
                or not self.cpu_operator or not self.metal_operator
                or any(type(value) is not int or value <= 0 for value in (
                    self.cpu_median_nanoseconds, self.metal_median_nanoseconds,
                    self.attention_median_nanoseconds,
                ))
                or not math.isfinite(self.metal_speedup) or self.metal_speedup <= 0
                or not math.isfinite(self.unified_memory_bandwidth_bytes_per_second)
                or self.unified_memory_bandwidth_bytes_per_second <= 0
                or self.recommended_backend not in {"cpu", "native_metal"}):
            raise ValueError("invalid phase resource entry")


@dataclass(frozen=True, slots=True)
class PhaseResourceProfile:
    hardware_fingerprint: str
    cpu_evidence_sha256: tuple[str, str]
    metal_evidence_sha256: str
    phases: tuple[PhaseResourceEntry, PhaseResourceEntry]
    schema_version: int = 1

    def __post_init__(self) -> None:
        digests = (*self.cpu_evidence_sha256, self.metal_evidence_sha256)
        if (self.schema_version != 1 or len(self.hardware_fingerprint) != 24
                or any(len(value) != 64 or any(character not in "0123456789abcdef"
                                                for character in value) for value in digests)
                or tuple(item.phase for item in self.phases) != ("prefill", "decode")):
            raise ValueError("invalid phase resource profile")

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "hardware_fingerprint": self.hardware_fingerprint,
            "cpu_evidence_sha256": list(self.cpu_evidence_sha256),
            "metal_evidence_sha256": self.metal_evidence_sha256,
            "phases": [asdict(item) for item in self.phases],
        }
        payload["profile_id"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return payload


def build_phase_resource_profile(
    cpu_prefill_path: Path, cpu_decode_path: Path, metal_path: Path
) -> PhaseResourceProfile:
    cpu_prefill, prefill_digest = _load(cpu_prefill_path)
    cpu_decode, decode_digest = _load(cpu_decode_path)
    metal, metal_digest = _load(metal_path)
    hardware = cpu_prefill.get("hardware_fingerprint")
    if (not isinstance(hardware, str) or len(hardware) != 24
            or cpu_decode.get("hardware_fingerprint") != hardware
            or metal.get("passed") is not True or metal.get("device") != "Apple M4"):
        raise ValueError("phase resource evidence identity mismatch")
    reports = metal.get("reports")
    if not isinstance(reports, list):
        raise ValueError("phase resource Metal reports missing")
    by_operator = {item.get("operator"): item for item in reports if isinstance(item, dict)}
    required = {"gpu_gemm", "gpu_gemv", "attention", "unified_memory_copy"}
    if not required.issubset(by_operator):
        raise ValueError("phase resource Metal evidence incomplete")
    attention = _positive_int(by_operator["attention"], "median_nanoseconds")
    bandwidth = _positive_number(
        by_operator["unified_memory_copy"], "bandwidth_bytes_per_second"
    )
    entries = []
    for phase, cpu, metal_operator, expected_cpu_operator in (
        ("prefill", cpu_prefill, "gpu_gemm", "matmul"),
        ("decode", cpu_decode, "gpu_gemv", "gemv"),
    ):
        config = cpu.get("config")
        if (not isinstance(config, dict) or config.get("phase") != phase
                or config.get("operator") != expected_cpu_operator
                or config.get("samples") != 7 or cpu.get("sample_count") != 7):
            raise ValueError("phase resource CPU evidence mismatch")
        cpu_ns = _positive_int(cpu, "median_total_nanoseconds")
        metal_ns = _positive_int(by_operator[metal_operator], "median_nanoseconds")
        speedup = cpu_ns / metal_ns
        entries.append(PhaseResourceEntry(
            phase, expected_cpu_operator, metal_operator, cpu_ns, metal_ns, speedup,
            attention, bandwidth, "native_metal" if speedup >= 1.05 else "cpu",
        ))
    return PhaseResourceProfile(
        hardware, (prefill_digest, decode_digest), metal_digest, tuple(entries)  # type: ignore[arg-type]
    )


def _load(path: Path) -> tuple[dict[str, object], str]:
    data = path.read_bytes()
    if not 1 <= len(data) <= 256 * 1024:
        raise ValueError("phase resource evidence exceeds bound")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("phase resource evidence must be an object")
    return payload, hashlib.sha256(data).hexdigest()


def _positive_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError("phase resource evidence integer is invalid")
    return value


def _positive_number(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0):
        raise ValueError("phase resource evidence number is invalid")
    return float(value)
