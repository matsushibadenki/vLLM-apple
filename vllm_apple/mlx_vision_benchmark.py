"""Bounded MLX image latency, throughput, and allocator-memory benchmark."""
from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VisionBenchmarkMeasurement:
    batch_size: int
    latency_nanoseconds: int
    peak_increment_bytes: int
    output_digest: str


@dataclass(frozen=True, slots=True)
class VisionBatchBenchmark:
    batch_size: int
    samples: int
    median_latency_nanoseconds: int
    images_per_second: float
    maximum_peak_increment_bytes: int
    maximum_memory_per_image_bytes: int
    deterministic: bool


@dataclass(frozen=True, slots=True)
class VisionBenchmarkReport:
    schema_version: int
    operator: str
    environment_fingerprint: str
    measurements: tuple[VisionBatchBenchmark, ...]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "operator": self.operator,
            "environment_fingerprint": self.environment_fingerprint,
            "measurements": [
                {
                    "batch_size": item.batch_size,
                    "samples": item.samples,
                    "median_latency_nanoseconds": item.median_latency_nanoseconds,
                    "images_per_second": item.images_per_second,
                    "maximum_peak_increment_bytes": item.maximum_peak_increment_bytes,
                    "maximum_memory_per_image_bytes": item.maximum_memory_per_image_bytes,
                    "deterministic": item.deterministic,
                }
                for item in self.measurements
            ],
            "passed": self.passed,
        }


_PROGRAM = """
import hashlib,json,time
import mlx.core as mx
batch={batch_size}
source_height,source_width,target,patch,channels,projection=256,256,224,16,3,64
pixels=mx.array([(item*17+row*7+column*3+channel*41)%256 for item in range(batch) for row in range(source_height) for column in range(source_width) for channel in range(channels)],dtype=mx.uint8).reshape((batch,source_height,source_width,channels))
row_index=mx.array([index*source_height//target for index in range(target)])
column_index=mx.array([index*source_width//target for index in range(target)])
mean=mx.array([0.48145466,0.4578275,0.40821073])
std=mx.array([0.26862954,0.26130258,0.27577711])
weights=mx.array([float((row*13+column*5)%37-18)/37 for row in range(patch*patch*channels) for column in range(projection)]).reshape((patch*patch*channels,projection))
mx.eval(pixels,row_index,column_index,mean,std,weights)
mx.reset_peak_memory()
active=mx.get_active_memory()
started=time.perf_counter_ns()
resized=mx.take(mx.take(pixels,row_index,axis=1),column_index,axis=2)
normalized=(resized.astype(mx.float32)/255.0-mean)/std
patches=normalized.reshape((batch,target//patch,patch,target//patch,patch,channels)).transpose((0,1,3,2,4,5)).reshape((batch,target//patch*target//patch,patch*patch*channels))
result=mx.matmul(patches,weights)
mx.eval(result)
elapsed=time.perf_counter_ns()-started
peak=max(0,mx.get_peak_memory()-active)
values=[round(float(value),5) for value in result.reshape((-1,))[:1024].tolist()]
digest=hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()
print(json.dumps({{'batch_size':batch,'latency_nanoseconds':elapsed,'peak_increment_bytes':peak,'output_digest':digest}},separators=(',',':')))
"""


@dataclass(frozen=True, slots=True)
class MLXVisionBenchmarkAdapter:
    python_executable: Path = Path(sys.executable)
    timeout_seconds: float = 15
    maximum_output_bytes: int = 16 * 1024

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("benchmark timeout must be positive and finite")
        if not 1024 <= self.maximum_output_bytes <= 256 * 1024:
            raise ValueError("benchmark output bound must be between 1 and 256 KiB")

    def benchmark(
        self,
        *,
        environment_fingerprint: str,
        batch_sizes: tuple[int, ...] = (1, 2, 4),
        samples: int = 3,
    ) -> VisionBenchmarkReport:
        if (
            not environment_fingerprint
            or not 1 <= len(batch_sizes) <= 8
            or len(set(batch_sizes)) != len(batch_sizes)
            or any(type(size) is not int or not 1 <= size <= 16 for size in batch_sizes)
            or not 1 <= samples <= 16
        ):
            raise ValueError("invalid MLX vision benchmark configuration")
        results = []
        for batch_size in batch_sizes:
            measured = [self._measure(batch_size) for _ in range(samples)]
            latency = int(statistics.median(item.latency_nanoseconds for item in measured))
            peak = max(item.peak_increment_bytes for item in measured)
            deterministic = len({item.output_digest for item in measured}) == 1
            results.append(VisionBatchBenchmark(
                batch_size=batch_size,
                samples=samples,
                median_latency_nanoseconds=latency,
                images_per_second=round(batch_size * 1_000_000_000 / latency, 6),
                maximum_peak_increment_bytes=peak,
                maximum_memory_per_image_bytes=(peak + batch_size - 1) // batch_size,
                deterministic=deterministic,
            ))
        return VisionBenchmarkReport(
            schema_version=1,
            operator="fused_vision_preprocess_projection",
            environment_fingerprint=environment_fingerprint,
            measurements=tuple(results),
            passed=all(item.deterministic for item in results),
        )

    def _measure(self, batch_size: int) -> VisionBenchmarkMeasurement:
        if type(batch_size) is not int or not 1 <= batch_size <= 16:
            raise ValueError("vision benchmark batch size must be between 1 and 16")
        completed = subprocess.run(
            [str(self.python_executable), "-c", _PROGRAM.format(batch_size=batch_size)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            timeout=self.timeout_seconds,
        )
        if len(completed.stdout) > self.maximum_output_bytes:
            raise RuntimeError("MLX vision benchmark output exceeded its bound")
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict) or set(payload) != {
            "batch_size", "latency_nanoseconds", "peak_increment_bytes", "output_digest"
        }:
            raise RuntimeError("MLX vision benchmark returned invalid output")
        return VisionBenchmarkMeasurement(**payload)
