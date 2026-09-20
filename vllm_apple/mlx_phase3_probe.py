"""Isolated MLX probes for Phase 3 fusion and routed expert candidates."""
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


_COMMON_RESULT = """
values=[round(float(value),6) for value in result.reshape((-1,)).tolist()]
elapsed=time.perf_counter_ns()-started
digest=hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()
print(json.dumps({'output_digest':digest,'latency_nanoseconds':elapsed,'numeric_values':values},separators=(',',':')))
"""

_PROGRAMS = {
    "staged_q4_silu": """
import hashlib,json,time
import mlx.core as mx
rows,columns=16,64
w=mx.array([float((r*7+c*3)%31-15)/31 for r in range(rows) for c in range(columns)]).reshape((rows,columns))
x=mx.array([float((c*5)%17-8)/17 for c in range(columns)]).reshape((1,columns))
packed,scales,biases=mx.quantize(w,group_size=32,bits=4,mode='affine')
started=time.perf_counter_ns()
linear=mx.quantized_matmul(x,packed,scales,biases,transpose=True,group_size=32,bits=4,mode='affine')
mx.eval(linear)
result=linear*mx.sigmoid(linear)
mx.eval(result)
""" + _COMMON_RESULT,
    "fused_q4_silu": """
import hashlib,json,time
import mlx.core as mx
rows,columns=16,64
w=mx.array([float((r*7+c*3)%31-15)/31 for r in range(rows) for c in range(columns)]).reshape((rows,columns))
x=mx.array([float((c*5)%17-8)/17 for c in range(columns)]).reshape((1,columns))
packed,scales,biases=mx.quantize(w,group_size=32,bits=4,mode='affine')
started=time.perf_counter_ns()
linear=mx.quantized_matmul(x,packed,scales,biases,transpose=True,group_size=32,bits=4,mode='affine')
result=linear*mx.sigmoid(linear)
mx.eval(result)
""" + _COMMON_RESULT,
    "staged_rmsnorm_rope": """
import hashlib,json,time
import mlx.core as mx
x=mx.array([float((b*29+t*7+d*3)%37-18)/37 for b in range(2) for t in range(16) for d in range(64)]).reshape((2,1,16,64))
weight=mx.array([1+float(d%7)/64 for d in range(64)])
started=time.perf_counter_ns()
normalized=mx.fast.rms_norm(x,weight,1e-5)
mx.eval(normalized)
result=mx.fast.rope(normalized,64,traditional=False,base=10000.0,scale=1.0,offset=0)
mx.eval(result)
""" + _COMMON_RESULT,
    "fused_rmsnorm_rope": """
import hashlib,json,time
import mlx.core as mx
x=mx.array([float((b*29+t*7+d*3)%37-18)/37 for b in range(2) for t in range(16) for d in range(64)]).reshape((2,1,16,64))
weight=mx.array([1+float(d%7)/64 for d in range(64)])
started=time.perf_counter_ns()
normalized=mx.fast.rms_norm(x,weight,1e-5)
result=mx.fast.rope(normalized,64,traditional=False,base=10000.0,scale=1.0,offset=0)
mx.eval(result)
""" + _COMMON_RESULT,
    "staged_moe": """
import hashlib,json,time
import mlx.core as mx
tokens,hidden,intermediate,experts,topk=8,16,32,4,2
x=mx.array([float((t*11+d*5)%29-14)/29 for t in range(tokens) for d in range(hidden)]).reshape((tokens,hidden))
router=mx.array([float((d*7+e*3)%23-11)/23 for d in range(hidden) for e in range(experts)]).reshape((hidden,experts))
up=mx.array([float((e*13+i*5+d*3)%31-15)/31 for e in range(experts) for i in range(intermediate) for d in range(hidden)]).reshape((experts,intermediate,hidden))
gate=mx.array([float((e*17+i*7+d)%37-18)/37 for e in range(experts) for i in range(intermediate) for d in range(hidden)]).reshape((experts,intermediate,hidden))
down=mx.array([float((e*19+d*3+i*5)%41-20)/41 for e in range(experts) for d in range(hidden) for i in range(intermediate)]).reshape((experts,hidden,intermediate))
started=time.perf_counter_ns()
logits=mx.matmul(x,router)
indices=mx.argpartition(logits,kth=experts-topk,axis=-1)[:,-topk:]
selected=mx.take_along_axis(logits,indices,axis=-1)
weights=mx.softmax(selected,axis=-1)
mx.eval(indices,weights)
rows=[]
for token,row in enumerate(indices.tolist()):
    combined=mx.zeros((hidden,))
    for slot,expert in enumerate(row):
        u=mx.matmul(x[token],mx.transpose(up[expert]))
        g=mx.matmul(x[token],mx.transpose(gate[expert]))
        h=(g*mx.sigmoid(g))*u
        combined=combined+mx.matmul(h,mx.transpose(down[expert]))*weights[token,slot]
    rows.append(combined)
result=mx.stack(rows)
mx.eval(result)
""" + _COMMON_RESULT,
    "fused_moe": """
import hashlib,json,time
import mlx.core as mx
tokens,hidden,intermediate,experts,topk=8,16,32,4,2
x=mx.array([float((t*11+d*5)%29-14)/29 for t in range(tokens) for d in range(hidden)]).reshape((tokens,hidden))
router=mx.array([float((d*7+e*3)%23-11)/23 for d in range(hidden) for e in range(experts)]).reshape((hidden,experts))
up=mx.array([float((e*13+i*5+d*3)%31-15)/31 for e in range(experts) for i in range(intermediate) for d in range(hidden)]).reshape((experts,intermediate,hidden))
gate=mx.array([float((e*17+i*7+d)%37-18)/37 for e in range(experts) for i in range(intermediate) for d in range(hidden)]).reshape((experts,intermediate,hidden))
down=mx.array([float((e*19+d*3+i*5)%41-20)/41 for e in range(experts) for d in range(hidden) for i in range(intermediate)]).reshape((experts,hidden,intermediate))
started=time.perf_counter_ns()
logits=mx.matmul(x,router)
indices=mx.argpartition(logits,kth=experts-topk,axis=-1)[:,-topk:]
weights=mx.softmax(mx.take_along_axis(logits,indices,axis=-1),axis=-1)
selected_up=up[indices]
selected_gate=gate[indices]
selected_down=down[indices]
inputs=x[:,None,None,:]
u=mx.matmul(inputs,mx.swapaxes(selected_up,-1,-2)).squeeze(-2)
g=mx.matmul(inputs,mx.swapaxes(selected_gate,-1,-2)).squeeze(-2)
h=(g*mx.sigmoid(g))*u
outputs=mx.matmul(h[:,:,None,:],mx.swapaxes(selected_down,-1,-2)).squeeze(-2)
result=mx.sum(outputs*weights[:,:,None],axis=1)
mx.eval(result)
""" + _COMMON_RESULT,
}


@dataclass(frozen=True, slots=True)
class MLXPhase3ProbeAdapter:
    python_executable: Path = Path(sys.executable)
    timeout_seconds: float = 10
    maximum_output_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("probe timeout must be positive and finite")
        if not 1024 <= self.maximum_output_bytes <= 256 * 1024:
            raise ValueError("probe output bound must be between 1 and 256 KiB")

    def probe_fusion_suite(
        self,
        *,
        hardware_fingerprint: str,
        environment_fingerprint: str,
        samples: int = 3,
        maximum_slowdown_ratio: float = 1.25,
    ) -> tuple[KernelProbeResult, ...]:
        pairs = (
            ("fused_q4_silu", "staged_q4_silu", 1e-5),
            ("fused_rmsnorm_rope", "staged_rmsnorm_rope", 1e-5),
            ("fused_moe", "staged_moe", 2e-4),
        )
        return tuple(
            run_kernel_probe(
                KernelProbeConfig(
                    hardware_fingerprint,
                    environment_fingerprint,
                    ExecutionBackend.NATIVE_MLX,
                    candidate,
                    samples,
                    maximum_slowdown_ratio,
                    tolerance,
                ),
                lambda baseline=baseline: self._measure(baseline),
                lambda candidate=candidate: self._measure(candidate),
            )
            for candidate, baseline, tolerance in pairs
        )

    def _measure(self, operator: str) -> KernelMeasurement:
        if operator not in _PROGRAMS:
            raise ValueError("unsupported Phase 3 MLX probe operator")
        completed = subprocess.run(
            [str(self.python_executable), "-c", _PROGRAMS[operator]],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            timeout=self.timeout_seconds,
        )
        if len(completed.stdout) > self.maximum_output_bytes:
            raise RuntimeError("Phase 3 MLX probe output exceeded its bound")
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict) or set(payload) != {
            "output_digest", "latency_nanoseconds", "numeric_values"
        }:
            raise RuntimeError("Phase 3 MLX probe returned invalid output")
        return KernelMeasurement(
            payload["output_digest"],
            payload["latency_nanoseconds"],
            tuple(payload["numeric_values"]),
        )
