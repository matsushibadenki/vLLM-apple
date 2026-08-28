from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .execution import ExecutionBackend
from .kernel_probe import (
    KernelCapabilityRegistry,
    KernelMeasurement,
    KernelProbeConfig,
    KernelProbeResult,
    run_kernel_probe,
)


_VECTOR_VALUES = tuple(float(value) for value in range(256))
_MATRIX_SIZE = 16
_MATRIX_VALUES = tuple(float(value % 7) for value in range(_MATRIX_SIZE**2))
_KV_VALUES = tuple(float(value % 31) for value in range(512))
_MLX_PROGRAMS = {
    "vector_add": """
import hashlib,json,time
import mlx.core as mx
values=[float(value) for value in range(256)]
started=time.perf_counter_ns()
result=mx.array(values)+mx.array(values)
mx.eval(result)
output=[float(value) for value in result.tolist()]
elapsed=time.perf_counter_ns()-started
digest=hashlib.sha256(json.dumps(output,separators=(',',':')).encode()).hexdigest()
print(json.dumps({'output_digest':digest,'latency_nanoseconds':elapsed},separators=(',',':')))
""".strip(),
    "matmul": """
import hashlib,json,time
import mlx.core as mx
n=16
values=[float(value%7) for value in range(n*n)]
matrix=mx.array(values).reshape((n,n))
started=time.perf_counter_ns()
result=mx.matmul(matrix,matrix)
mx.eval(result)
output=[[float(value) for value in row] for row in result.tolist()]
elapsed=time.perf_counter_ns()-started
digest=hashlib.sha256(json.dumps(output,separators=(',',':')).encode()).hexdigest()
print(json.dumps({'output_digest':digest,'latency_nanoseconds':elapsed},separators=(',',':')))
""".strip(),
    "kv_copy": """
import hashlib,json,time
import mlx.core as mx
values=[float(value%31) for value in range(512)]
source=mx.array(values)
started=time.perf_counter_ns()
result=source[64:192]
mx.eval(result)
output=[float(value) for value in result.tolist()]
elapsed=time.perf_counter_ns()-started
digest=hashlib.sha256(json.dumps(output,separators=(',',':')).encode()).hexdigest()
print(json.dumps({'output_digest':digest,'latency_nanoseconds':elapsed},separators=(',',':')))
""".strip(),
    "attention": """
import hashlib,json,math,time
import mlx.core as mx
outputs=[]
started=time.perf_counter_ns()
def attend(q,k,v,causal=False):
    dimension=q.shape[-1]
    scores=mx.matmul(q,mx.transpose(k))/math.sqrt(dimension)
    if causal:
        rows=q.shape[0]
        columns=k.shape[0]
        mask=mx.array([[column>row for column in range(columns)] for row in range(rows)])
        scores=mx.where(mask,mx.array(-1e9),scores)
    result=mx.matmul(mx.softmax(scores,axis=-1),v)
    mx.eval(result)
    return [[round(float(value),5) for value in row] for row in result.tolist()]
def tensors(sequence,dimension,offset=0):
    q=mx.array([float((row*3+column+offset)%11)/11 for row in range(sequence) for column in range(dimension)]).reshape((sequence,dimension))
    k=mx.array([float((row*5+column+offset)%13)/13 for row in range(sequence) for column in range(dimension)]).reshape((sequence,dimension))
    v=mx.array([float((row*7+column+offset)%17)/17 for row in range(sequence) for column in range(dimension)]).reshape((sequence,dimension))
    return q,k,v
for sequence in (8,32):
    dimension=8
    outputs.append(attend(*tensors(sequence,dimension)))
outputs.append(attend(*tensors(16,8),causal=True))
_,decode_k,decode_v=tensors(32,8)
decode_q=tensors(1,8)[0]
outputs.append(attend(decode_q,decode_k,decode_v))
gqa=[]
for query_head in range(4):
    kv_head=query_head//2
    query=tensors(16,4,query_head)[0]
    _,key,value=tensors(16,4,kv_head)
    gqa.append(attend(query,key,value,causal=True))
outputs.append(gqa)
elapsed=time.perf_counter_ns()-started
digest=hashlib.sha256(json.dumps(outputs,separators=(',',':')).encode()).hexdigest()
print(json.dumps({'output_digest':digest,'latency_nanoseconds':elapsed},separators=(',',':')))
""".strip(),
    "paged_attention": """
import hashlib,json,math,time
import mlx.core as mx
dimension=8
query=mx.array([float((column*3)%11)/11 for column in range(dimension)]).reshape((1,dimension))
outputs=[]
started=time.perf_counter_ns()
for context,page_size,page_count in ((14,4,8),(256,16,16),(1024,16,64)):
    pages=mx.array([float((page*19+token*7+column)%23)/23 for page in range(page_count) for token in range(page_size) for column in range(dimension)]).reshape((page_count,page_size,dimension))
    block_table=([2,0,5,1,6,3,7,4] if context==14 else [(index*5+2)%page_count for index in range(page_count)])
    key=mx.concatenate([pages[index] for index in block_table],axis=0)[:context]
    value=key*0.5+0.125
    scores=mx.matmul(query,mx.transpose(key))/math.sqrt(dimension)
    result=mx.matmul(mx.softmax(scores,axis=-1),value)
    mx.eval(result)
    outputs.append([[round(float(value),5) for value in row] for row in result.tolist()])
elapsed=time.perf_counter_ns()-started
digest=hashlib.sha256(json.dumps(outputs,separators=(',',':')).encode()).hexdigest()
print(json.dumps({'output_digest':digest,'latency_nanoseconds':elapsed},separators=(',',':')))
""".strip(),
    "mla": """
import hashlib,json,math,time
import mlx.core as mx
sequence=16
latent_dimension=4
head_dimension=8
latent=mx.array([float((row*5+column)%17)/17 for row in range(sequence) for column in range(latent_dimension)]).reshape((sequence,latent_dimension))
key_projection=mx.array([float((row*3+column)%13)/13 for row in range(latent_dimension) for column in range(head_dimension)]).reshape((latent_dimension,head_dimension))
value_projection=mx.array([float((row*7+column)%19)/19 for row in range(latent_dimension) for column in range(head_dimension)]).reshape((latent_dimension,head_dimension))
query=mx.array([float((column*11)%23)/23 for column in range(head_dimension)]).reshape((1,head_dimension))
started=time.perf_counter_ns()
key=mx.matmul(latent,key_projection)
value=mx.matmul(latent,value_projection)
scores=mx.matmul(query,mx.transpose(key))/math.sqrt(head_dimension)
result=mx.matmul(mx.softmax(scores,axis=-1),value)
mx.eval(result)
output=[[round(float(value),5) for value in row] for row in result.tolist()]
elapsed=time.perf_counter_ns()-started
digest=hashlib.sha256(json.dumps(output,separators=(',',':')).encode()).hexdigest()
print(json.dumps({'output_digest':digest,'latency_nanoseconds':elapsed},separators=(',',':')))
""".strip(),
}


@dataclass(frozen=True, slots=True)
class NativeMLXProbeAdapter:
    python_executable: Path = Path(sys.executable)
    timeout_seconds: float = 10
    maximum_output_bytes: int = 4096

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("probe timeout must be positive and finite")
        if not 128 <= self.maximum_output_bytes <= 64 * 1024:
            raise ValueError("probe output limit must be between 128 bytes and 64 KiB")

    def probe_vector_add(
        self,
        *,
        hardware_fingerprint: str,
        environment_fingerprint: str,
        samples: int = 3,
        maximum_slowdown_ratio: float = 20,
    ) -> KernelProbeResult:
        return self._probe(
            "vector_add",
            self._baseline_vector_add,
            hardware_fingerprint,
            environment_fingerprint,
            samples,
            maximum_slowdown_ratio,
        )

    def probe_suite(
        self,
        *,
        hardware_fingerprint: str,
        environment_fingerprint: str,
        samples: int = 3,
        maximum_slowdown_ratio: float = 20,
    ) -> tuple[KernelProbeResult, ...]:
        baselines = {
            "vector_add": self._baseline_vector_add,
            "matmul": self._baseline_matmul,
            "kv_copy": self._baseline_kv_copy,
            "attention": self._baseline_attention,
            "paged_attention": self._baseline_paged_attention,
            "mla": self._baseline_mla,
        }
        return tuple(
            self._probe(
                operator,
                baseline,
                hardware_fingerprint,
                environment_fingerprint,
                samples,
                maximum_slowdown_ratio,
            )
            for operator, baseline in baselines.items()
        )

    def _probe(
        self,
        operator: str,
        baseline: Callable[[], KernelMeasurement],
        hardware_fingerprint: str,
        environment_fingerprint: str,
        samples: int,
        maximum_slowdown_ratio: float,
    ) -> KernelProbeResult:
        config = KernelProbeConfig(
            hardware_fingerprint=hardware_fingerprint,
            environment_fingerprint=environment_fingerprint,
            backend=ExecutionBackend.NATIVE_MLX,
            operator=operator,
            samples=samples,
            maximum_slowdown_ratio=maximum_slowdown_ratio,
        )
        return run_kernel_probe(
            config,
            baseline,
            lambda: self._candidate(operator),
        )

    @staticmethod
    def _baseline_vector_add() -> KernelMeasurement:
        import time

        started = time.perf_counter_ns()
        output = [value + value for value in _VECTOR_VALUES]
        elapsed = max(1, time.perf_counter_ns() - started)
        digest = hashlib.sha256(
            json.dumps(output, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return KernelMeasurement(digest, elapsed)

    @staticmethod
    def _baseline_matmul() -> KernelMeasurement:
        import time

        matrix = [
            _MATRIX_VALUES[index : index + _MATRIX_SIZE]
            for index in range(0, len(_MATRIX_VALUES), _MATRIX_SIZE)
        ]
        started = time.perf_counter_ns()
        output = [
            [
                float(
                    sum(matrix[row][inner] * matrix[inner][column] for inner in range(_MATRIX_SIZE))
                )
                for column in range(_MATRIX_SIZE)
            ]
            for row in range(_MATRIX_SIZE)
        ]
        elapsed = max(1, time.perf_counter_ns() - started)
        digest = hashlib.sha256(
            json.dumps(output, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return KernelMeasurement(digest, elapsed)

    @staticmethod
    def _baseline_kv_copy() -> KernelMeasurement:
        import time

        started = time.perf_counter_ns()
        output = list(_KV_VALUES[64:192])
        elapsed = max(1, time.perf_counter_ns() - started)
        digest = hashlib.sha256(
            json.dumps(output, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return KernelMeasurement(digest, elapsed)

    @staticmethod
    def _baseline_attention() -> KernelMeasurement:
        import time

        outputs: list[object] = []
        started = time.perf_counter_ns()
        for sequence in (8, 32):
            outputs.append(_reference_attention(*_attention_tensors(sequence, 8)))
        outputs.append(
            _reference_attention(*_attention_tensors(16, 8), causal=True)
        )
        _, decode_key, decode_value = _attention_tensors(32, 8)
        decode_query, _, _ = _attention_tensors(1, 8)
        outputs.append(_reference_attention(decode_query, decode_key, decode_value))
        gqa = []
        for query_head in range(4):
            query, _, _ = _attention_tensors(16, 4, query_head)
            _, key, value = _attention_tensors(16, 4, query_head // 2)
            gqa.append(_reference_attention(query, key, value, causal=True))
        outputs.append(gqa)
        elapsed = max(1, time.perf_counter_ns() - started)
        digest = hashlib.sha256(
            json.dumps(outputs, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return KernelMeasurement(digest, elapsed)

    @staticmethod
    def _baseline_paged_attention() -> KernelMeasurement:
        import time

        dimension = 8
        query = [[float((column * 3) % 11) / 11 for column in range(dimension)]]
        started = time.perf_counter_ns()
        outputs = []
        for context, page_size, page_count in (
            (14, 4, 8),
            (256, 16, 16),
            (1024, 16, 64),
        ):
            pages = [
                [
                    [
                        float((page * 19 + token * 7 + column) % 23) / 23
                        for column in range(dimension)
                    ]
                    for token in range(page_size)
                ]
                for page in range(page_count)
            ]
            block_table = (
                [2, 0, 5, 1, 6, 3, 7, 4]
                if context == 14
                else [(index * 5 + 2) % page_count for index in range(page_count)]
            )
            key = [token for page in block_table for token in pages[page]][:context]
            value = [[item * 0.5 + 0.125 for item in token] for token in key]
            outputs.append(_reference_attention(query, key, value))
        elapsed = max(1, time.perf_counter_ns() - started)
        return KernelMeasurement(_digest_value(outputs), elapsed)

    @staticmethod
    def _baseline_mla() -> KernelMeasurement:
        import time

        sequence = 16
        latent_dimension = 4
        head_dimension = 8
        latent = [
            [
                float((row * 5 + column) % 17) / 17
                for column in range(latent_dimension)
            ]
            for row in range(sequence)
        ]
        key_projection = [
            [
                float((row * 3 + column) % 13) / 13
                for column in range(head_dimension)
            ]
            for row in range(latent_dimension)
        ]
        value_projection = [
            [
                float((row * 7 + column) % 19) / 19
                for column in range(head_dimension)
            ]
            for row in range(latent_dimension)
        ]
        query = [
            [float((column * 11) % 23) / 23 for column in range(head_dimension)]
        ]
        started = time.perf_counter_ns()
        key = _reference_matmul(latent, key_projection)
        value = _reference_matmul(latent, value_projection)
        output = _reference_attention(query, key, value)
        elapsed = max(1, time.perf_counter_ns() - started)
        return KernelMeasurement(_digest_value(output), elapsed)

    def _candidate(self, operator: str) -> KernelMeasurement:
        completed = subprocess.run(
            [str(self.python_executable), "-c", _MLX_PROGRAMS[operator]],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            timeout=self.timeout_seconds,
        )
        if len(completed.stdout) > self.maximum_output_bytes:
            raise RuntimeError("MLX probe output exceeded its bound")
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict) or set(payload) != {
            "output_digest",
            "latency_nanoseconds",
        }:
            raise RuntimeError("MLX probe returned invalid fields")
        return KernelMeasurement(
            output_digest=payload["output_digest"],
            latency_nanoseconds=payload["latency_nanoseconds"],
        )


def build_mlx_probe_registry(
    adapter: NativeMLXProbeAdapter,
    *,
    hardware_fingerprint: str,
    environment_fingerprint: str,
    samples: int = 3,
) -> KernelCapabilityRegistry:
    registry = KernelCapabilityRegistry(hardware_fingerprint, environment_fingerprint)
    for result in adapter.probe_suite(
        hardware_fingerprint=hardware_fingerprint,
        environment_fingerprint=environment_fingerprint,
        samples=samples,
    ):
        registry.record(result)
    return registry


def _attention_tensors(
    sequence: int, dimension: int, offset: int = 0
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    query = [
        [float((row * 3 + column + offset) % 11) / 11 for column in range(dimension)]
        for row in range(sequence)
    ]
    key = [
        [float((row * 5 + column + offset) % 13) / 13 for column in range(dimension)]
        for row in range(sequence)
    ]
    value = [
        [float((row * 7 + column + offset) % 17) / 17 for column in range(dimension)]
        for row in range(sequence)
    ]
    return query, key, value


def _reference_attention(
    query: list[list[float]],
    key: list[list[float]],
    value: list[list[float]],
    *,
    causal: bool = False,
) -> list[list[float]]:
    dimension = len(query[0])
    scores = [
        [
            (
                -1e9
                if causal and column > row
                else sum(
                    query[row][inner] * key[column][inner]
                    for inner in range(dimension)
                )
                / math.sqrt(dimension)
            )
            for column in range(len(key))
        ]
        for row in range(len(query))
    ]
    weights = []
    for row in scores:
        maximum = max(row)
        exponentials = [math.exp(item - maximum) for item in row]
        total = sum(exponentials)
        weights.append([item / total for item in exponentials])
    return [
        [
            round(
                sum(
                    weights[row][token] * value[token][column]
                    for token in range(len(key))
                ),
                5,
            )
            for column in range(dimension)
        ]
        for row in range(len(query))
    ]


def _reference_matmul(
    left: list[list[float]], right: list[list[float]]
) -> list[list[float]]:
    return [
        [
            sum(left[row][inner] * right[inner][column] for inner in range(len(right)))
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def _digest_value(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
