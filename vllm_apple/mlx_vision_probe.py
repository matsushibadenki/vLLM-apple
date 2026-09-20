"""Isolated MLX correctness and performance probe for vision preprocessing fusion."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .execution import ExecutionBackend
from .kernel_probe import (
    KernelMeasurement,
    KernelProbeConfig,
    KernelProbeResult,
    run_kernel_probe,
)


_RESULT = """
flat=[round(float(value),6) for value in result.reshape((-1,)).tolist()]
values=flat[:4096]
elapsed=time.perf_counter_ns()-started
digest=hashlib.sha256(json.dumps(flat,separators=(',',':')).encode()).hexdigest()
print(json.dumps({'output_digest':digest,'latency_nanoseconds':elapsed,'numeric_values':values},separators=(',',':')))
"""

_SETUP = """
import hashlib,json,time
import mlx.core as mx
source_height,source_width,target,patch,channels,projection=256,256,224,16,3,64
pixels=mx.array([(row*7+column*3+channel*41)%256 for row in range(source_height) for column in range(source_width) for channel in range(channels)],dtype=mx.uint8).reshape((source_height,source_width,channels))
row_index=mx.array([index*source_height//target for index in range(target)])
column_index=mx.array([index*source_width//target for index in range(target)])
mean=mx.array([0.48145466,0.4578275,0.40821073])
std=mx.array([0.26862954,0.26130258,0.27577711])
weights=mx.array([float((row*13+column*5)%37-18)/37 for row in range(patch*patch*channels) for column in range(projection)]).reshape((patch*patch*channels,projection))
"""

_PROGRAMS = {
    "staged_vision_preprocess_projection": _SETUP + """
started=time.perf_counter_ns()
resized=mx.take(mx.take(pixels,row_index,axis=0),column_index,axis=1)
mx.eval(resized)
normalized=(resized.astype(mx.float32)/255.0-mean)/std
mx.eval(normalized)
patches=normalized.reshape((target//patch,patch,target//patch,patch,channels)).transpose((0,2,1,3,4)).reshape((-1,patch*patch*channels))
mx.eval(patches)
result=mx.matmul(patches,weights)
mx.eval(result)
""" + _RESULT,
    "fused_vision_preprocess_projection": _SETUP + """
started=time.perf_counter_ns()
resized=mx.take(mx.take(pixels,row_index,axis=0),column_index,axis=1)
normalized=(resized.astype(mx.float32)/255.0-mean)/std
patches=normalized.reshape((target//patch,patch,target//patch,patch,channels)).transpose((0,2,1,3,4)).reshape((-1,patch*patch*channels))
result=mx.matmul(patches,weights)
mx.eval(result)
""" + _RESULT,
}


@dataclass(frozen=True, slots=True)
class MLXVisionFusionProbeAdapter:
    python_executable: Path = Path(sys.executable)
    timeout_seconds: float = 15
    maximum_output_bytes: int = 512 * 1024

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("probe timeout must be positive and finite")
        if not 1024 <= self.maximum_output_bytes <= 2 * 1024 * 1024:
            raise ValueError("probe output bound must be between 1 KiB and 2 MiB")

    def probe_fusion(
        self,
        *,
        hardware_fingerprint: str,
        environment_fingerprint: str,
        samples: int = 3,
        maximum_slowdown_ratio: float = 1.25,
    ) -> KernelProbeResult:
        return run_kernel_probe(
            KernelProbeConfig(
                hardware_fingerprint,
                environment_fingerprint,
                ExecutionBackend.NATIVE_MLX,
                "fused_vision_preprocess_projection",
                samples,
                maximum_slowdown_ratio,
                1e-5,
            ),
            lambda: self._measure("staged_vision_preprocess_projection"),
            lambda: self._measure("fused_vision_preprocess_projection"),
        )

    def _measure(self, operator: str) -> KernelMeasurement:
        if operator not in _PROGRAMS:
            raise ValueError("unsupported MLX vision fusion probe operator")
        completed = subprocess.run(
            [str(self.python_executable), "-c", _PROGRAMS[operator]],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            timeout=self.timeout_seconds,
        )
        if len(completed.stdout) > self.maximum_output_bytes:
            raise RuntimeError("MLX vision fusion probe output exceeded its bound")
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict) or set(payload) != {
            "output_digest", "latency_nanoseconds", "numeric_values"
        }:
            raise RuntimeError("MLX vision fusion probe returned invalid output")
        return KernelMeasurement(
            payload["output_digest"],
            payload["latency_nanoseconds"],
            tuple(payload["numeric_values"]),
        )
