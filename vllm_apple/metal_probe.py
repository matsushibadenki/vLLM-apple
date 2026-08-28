from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .execution import ExecutionBackend
from .kernel_probe import (
    KernelMeasurement,
    KernelProbeConfig,
    KernelProbeResult,
    run_kernel_probe,
)
from .kernel_profile import ModelKernelShapeProfile, PagedAttentionShape

_METAL_VECTOR_COUNT = 64
_MAX_MODEL_SHAPE_PROBE_BYTES = 64 * 1024 * 1024
_THREAD_WIDTHS = (32, 64, 128, 256)
_TUNING_TIE_RATIO = 1.02


@dataclass(frozen=True, slots=True)
class MetalThreadConfiguration:
    score_width: int
    softmax_width: int
    output_width: int

    def __post_init__(self) -> None:
        if any(
            value not in _THREAD_WIDTHS
            for value in (self.score_width, self.softmax_width, self.output_width)
        ):
            raise ValueError("Metal thread widths must be 32, 64, 128, or 256")

    def to_dict(self) -> dict[str, int]:
        return {
            "score_width": self.score_width,
            "softmax_width": self.softmax_width,
            "output_width": self.output_width,
        }


@dataclass(frozen=True, slots=True)
class MetalShapeTuningDecision:
    shape: PagedAttentionShape
    winner: MetalThreadConfiguration
    candidates: tuple[tuple[MetalThreadConfiguration, KernelProbeResult], ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.candidates) <= 4:
            raise ValueError("Metal tuning must contain 1 to 4 candidates")
        if self.winner not in tuple(configuration for configuration, _ in self.candidates):
            raise ValueError("Metal tuning winner must be a measured candidate")
        if self.winner != _select_tuning_winner(self.candidates):
            raise ValueError("Metal tuning winner does not match measured candidates")

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": asdict(self.shape),
            "winner": self.winner.to_dict(),
            "candidates": [
                {"configuration": configuration.to_dict(), "result": result.to_dict()}
                for configuration, result in self.candidates
            ],
        }
_METAL_VECTOR_ADD_PROGRAM = r'''
import Foundation
import Metal

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(message.utf8))
    exit(2)
}

guard let device = MTLCreateSystemDefaultDevice() else { fail("metal_device_unavailable") }
let source = """
#include <metal_stdlib>
using namespace metal;
kernel void vector_add(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < 64) { output[index] = a[index] + b[index]; }
}
"""
let library: MTLLibrary
do { library = try device.makeLibrary(source: source, options: nil) }
catch { fail("metal_compile_failed") }
guard let function = library.makeFunction(name: "vector_add") else { fail("function_missing") }
let pipeline: MTLComputePipelineState
do { pipeline = try device.makeComputePipelineState(function: function) }
catch { fail("pipeline_failed") }
guard let queue = device.makeCommandQueue() else { fail("queue_unavailable") }
let input = (0..<64).map { Float($0) }
let byteCount = input.count * MemoryLayout<Float>.stride
guard let a = device.makeBuffer(bytes: input, length: byteCount),
      let b = device.makeBuffer(bytes: input, length: byteCount),
      let output = device.makeBuffer(length: byteCount),
      let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder() else { fail("allocation_failed") }
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(a, offset: 0, index: 0)
encoder.setBuffer(b, offset: 0, index: 1)
encoder.setBuffer(output, offset: 0, index: 2)
let width = min(pipeline.maxTotalThreadsPerThreadgroup, 64)
let started = DispatchTime.now().uptimeNanoseconds
encoder.dispatchThreads(
    MTLSize(width: 64, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
)
encoder.endEncoding()
command.commit()
command.waitUntilCompleted()
let elapsed = max(1, DispatchTime.now().uptimeNanoseconds - started)
if command.status != .completed { fail("command_failed") }
let pointer = output.contents().bindMemory(to: Float.self, capacity: 64)
let values = (0..<64).map { Double(pointer[$0]) }
let payload: [String: Any] = ["output": values, "latency_nanoseconds": elapsed]
guard let data = try? JSONSerialization.data(withJSONObject: payload),
      let text = String(data: data, encoding: .utf8) else { fail("encoding_failed") }
print(text)
'''.strip()
_METAL_PAGED_ATTENTION_PROGRAM = r'''
import Foundation
import Metal

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(message.utf8))
    exit(2)
}
guard let device = MTLCreateSystemDefaultDevice() else { fail("metal_device_unavailable") }
let source = """
#include <metal_stdlib>
using namespace metal;
kernel void paged_attention(
    device const float *pages [[buffer(0)]],
    device const uint *blocks [[buffer(1)]],
    device const float *query [[buffer(2)]],
    device float *output [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index != 0) { return; }
    float scores[14];
    float maximum = -INFINITY;
    for (uint token = 0; token < 14; ++token) {
        uint physical = blocks[token / 4] * 4 + token % 4;
        float score = 0.0f;
        for (uint column = 0; column < 8; ++column) {
            score += query[column] * pages[physical * 8 + column];
        }
        scores[token] = score * 0.3535533905932738f;
        maximum = max(maximum, scores[token]);
    }
    float denominator = 0.0f;
    for (uint token = 0; token < 14; ++token) {
        scores[token] = exp(scores[token] - maximum);
        denominator += scores[token];
    }
    for (uint column = 0; column < 8; ++column) {
        float value = 0.0f;
        for (uint token = 0; token < 14; ++token) {
            uint physical = blocks[token / 4] * 4 + token % 4;
            float source = pages[physical * 8 + column] * 0.5f + 0.125f;
            value += (scores[token] / denominator) * source;
        }
        output[column] = value;
    }
}
"""
let library: MTLLibrary
do { library = try device.makeLibrary(source: source, options: nil) }
catch { fail("metal_compile_failed") }
guard let function = library.makeFunction(name: "paged_attention") else { fail("function_missing") }
let pipeline: MTLComputePipelineState
do { pipeline = try device.makeComputePipelineState(function: function) }
catch { fail("pipeline_failed") }
guard let queue = device.makeCommandQueue() else { fail("queue_unavailable") }
let pages = (0..<(8*4*8)).map { index -> Float in
    let page = index / 32
    let token = (index / 8) % 4
    let column = index % 8
    return Float((page * 19 + token * 7 + column) % 23) / 23.0
}
let blocks: [UInt32] = [2, 0, 5, 1]
let query = (0..<8).map { Float(($0 * 3) % 11) / 11.0 }
func makeBuffer<T>(_ values: [T]) -> MTLBuffer? {
    values.withUnsafeBytes { bytes in
        guard let address = bytes.baseAddress else { return nil }
        return device.makeBuffer(bytes: address, length: bytes.count)
    }
}
guard let pageBuffer = makeBuffer(pages),
      let blockBuffer = makeBuffer(blocks),
      let queryBuffer = makeBuffer(query),
      let output = device.makeBuffer(length: 8 * MemoryLayout<Float>.stride),
      let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder() else { fail("allocation_failed") }
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(pageBuffer, offset: 0, index: 0)
encoder.setBuffer(blockBuffer, offset: 0, index: 1)
encoder.setBuffer(queryBuffer, offset: 0, index: 2)
encoder.setBuffer(output, offset: 0, index: 3)
let started = DispatchTime.now().uptimeNanoseconds
encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
encoder.endEncoding()
command.commit()
command.waitUntilCompleted()
let elapsed = max(1, DispatchTime.now().uptimeNanoseconds - started)
if command.status != .completed { fail("command_failed") }
let pointer = output.contents().bindMemory(to: Float.self, capacity: 8)
let values = (0..<8).map { Double(pointer[$0]) }
let payload: [String: Any] = ["output": values, "latency_nanoseconds": elapsed]
guard let data = try? JSONSerialization.data(withJSONObject: payload),
      let text = String(data: data, encoding: .utf8) else { fail("encoding_failed") }
print(text)
'''.strip()


@dataclass(frozen=True, slots=True)
class NativeMetalProbeAdapter:
    swift_executable: Path = Path("/usr/bin/swift")
    timeout_seconds: float = 30
    maximum_output_bytes: int = 4096

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("probe timeout must be positive and finite")
        if not 512 <= self.maximum_output_bytes <= 64 * 1024:
            raise ValueError("probe output limit must be between 512 bytes and 64 KiB")

    def probe_vector_add(
        self,
        *,
        hardware_fingerprint: str,
        environment_fingerprint: str,
        samples: int = 3,
        maximum_slowdown_ratio: float = 20,
    ) -> KernelProbeResult:
        config = KernelProbeConfig(
            hardware_fingerprint=hardware_fingerprint,
            environment_fingerprint=environment_fingerprint,
            backend=ExecutionBackend.NATIVE_METAL,
            operator="vector_add",
            samples=samples,
            maximum_slowdown_ratio=maximum_slowdown_ratio,
        )
        return run_kernel_probe(config, self._baseline, self._candidate)

    def probe_paged_attention(
        self,
        *,
        hardware_fingerprint: str,
        environment_fingerprint: str,
        samples: int = 3,
        maximum_slowdown_ratio: float = 20,
    ) -> KernelProbeResult:
        config = KernelProbeConfig(
            hardware_fingerprint=hardware_fingerprint,
            environment_fingerprint=environment_fingerprint,
            backend=ExecutionBackend.NATIVE_METAL,
            operator="paged_attention",
            samples=samples,
            maximum_slowdown_ratio=maximum_slowdown_ratio,
        )
        return run_kernel_probe(
            config, self._baseline_paged_attention, self._candidate_paged_attention
        )

    def probe_suite(
        self,
        *,
        hardware_fingerprint: str,
        environment_fingerprint: str,
        samples: int = 3,
    ) -> tuple[KernelProbeResult, ...]:
        return (
            self.probe_vector_add(
                hardware_fingerprint=hardware_fingerprint,
                environment_fingerprint=environment_fingerprint,
                samples=samples,
            ),
            self.probe_paged_attention(
                hardware_fingerprint=hardware_fingerprint,
                environment_fingerprint=environment_fingerprint,
                samples=samples,
            ),
        )

    def probe_shape_profile(
        self,
        profile: ModelKernelShapeProfile,
        *,
        hardware_fingerprint: str,
        environment_fingerprint: str,
        samples: int = 1,
        maximum_shapes: int = 4,
    ) -> tuple[KernelProbeResult, ...]:
        if not 1 <= maximum_shapes <= 16:
            raise ValueError("maximum_shapes must be between 1 and 16")
        return tuple(
            self._probe_model_shape(
                shape,
                hardware_fingerprint=hardware_fingerprint,
                environment_fingerprint=environment_fingerprint,
                samples=samples,
            )
            for shape in profile.shapes[:maximum_shapes]
        )

    def _probe_model_shape(
        self,
        shape: PagedAttentionShape,
        *,
        hardware_fingerprint: str,
        environment_fingerprint: str,
        samples: int,
    ) -> KernelProbeResult:
        allocation_bytes = (
            shape.blocks_per_sequence
            * shape.block_tokens
            * shape.head_dimension
            * 4
            + shape.context_tokens * 4
            + shape.head_dimension * 8
            + shape.blocks_per_sequence * 4
        )
        if allocation_bytes > _MAX_MODEL_SHAPE_PROBE_BYTES:
            raise ValueError("representative Metal shape exceeds the 64 MiB probe bound")
        operator = (
            f"paged_attention:c{shape.context_tokens}:d{shape.head_dimension}:"
            f"q{shape.query_heads}:kv{shape.kv_heads}:b{shape.block_tokens}"
        )
        config = KernelProbeConfig(
            hardware_fingerprint=hardware_fingerprint,
            environment_fingerprint=environment_fingerprint,
            backend=ExecutionBackend.NATIVE_METAL,
            operator=operator,
            samples=samples,
            maximum_slowdown_ratio=20,
            maximum_absolute_error=1e-5,
        )
        return run_kernel_probe(
            config,
            lambda: self._baseline_model_shape(shape),
            lambda: self._candidate_model_shape(shape),
        )

    def tune_model_shape(
        self,
        shape: PagedAttentionShape,
        *,
        hardware_fingerprint: str,
        environment_fingerprint: str,
        samples: int = 3,
    ) -> MetalShapeTuningDecision:
        measured: list[tuple[MetalThreadConfiguration, KernelProbeResult]] = []
        for configuration in _thread_configurations(shape):
            operator = (
                f"paged_attention_tune:c{shape.context_tokens}:d{shape.head_dimension}:"
                f"s{configuration.score_width}:m{configuration.softmax_width}:"
                f"o{configuration.output_width}"
            )
            result = run_kernel_probe(
                KernelProbeConfig(
                    hardware_fingerprint=hardware_fingerprint,
                    environment_fingerprint=environment_fingerprint,
                    backend=ExecutionBackend.NATIVE_METAL,
                    operator=operator,
                    samples=samples,
                    maximum_slowdown_ratio=20,
                    maximum_absolute_error=1e-5,
                ),
                lambda: self._baseline_model_shape(shape),
                lambda configuration=configuration: self._candidate_model_shape(
                    shape, configuration
                ),
            )
            measured.append((configuration, result))
        winner = _select_tuning_winner(tuple(measured))
        return MetalShapeTuningDecision(shape, winner, tuple(measured))

    @staticmethod
    def _baseline() -> KernelMeasurement:
        import time

        started = time.perf_counter_ns()
        output = [float(value + value) for value in range(_METAL_VECTOR_COUNT)]
        elapsed = max(1, time.perf_counter_ns() - started)
        return KernelMeasurement(_digest(output), elapsed)

    @staticmethod
    def _baseline_paged_attention() -> KernelMeasurement:
        import time

        dimension = 8
        pages = [
            [
                [
                    float((page * 19 + token * 7 + column) % 23) / 23
                    for column in range(dimension)
                ]
                for token in range(4)
            ]
            for page in range(8)
        ]
        key = [token for page in (2, 0, 5, 1) for token in pages[page]][:14]
        query = [float((column * 3) % 11) / 11 for column in range(dimension)]
        started = time.perf_counter_ns()
        scores = [
            sum(query[column] * token[column] for column in range(dimension))
            / math.sqrt(dimension)
            for token in key
        ]
        maximum = max(scores)
        exponentials = [math.exp(score - maximum) for score in scores]
        total = sum(exponentials)
        output = [
            round(
                sum(
                    exponentials[token]
                    / total
                    * (key[token][column] * 0.5 + 0.125)
                    for token in range(14)
                ),
                5,
            )
            for column in range(dimension)
        ]
        elapsed = max(1, time.perf_counter_ns() - started)
        return KernelMeasurement(_digest(output), elapsed)

    def _candidate(self) -> KernelMeasurement:
        return self._run_program(_METAL_VECTOR_ADD_PROGRAM, _METAL_VECTOR_COUNT)

    def _candidate_paged_attention(self) -> KernelMeasurement:
        measurement = self._run_program(_METAL_PAGED_ATTENTION_PROGRAM, 8)
        return KernelMeasurement(measurement.output_digest, measurement.latency_nanoseconds)

    @staticmethod
    def _baseline_model_shape(shape: PagedAttentionShape) -> KernelMeasurement:
        import time

        started = time.perf_counter_ns()
        dimension = shape.head_dimension
        query = [float((column * 3) % 11) / 11 for column in range(dimension)]

        def key_value(token: int, column: int) -> float:
            logical_block, offset = divmod(token, shape.block_tokens)
            physical_block = shape.blocks_per_sequence - 1 - logical_block
            return float((physical_block * 19 + offset * 7 + column) % 23) / 23

        scores = [
            sum(query[column] * key_value(token, column) for column in range(dimension))
            / math.sqrt(dimension)
            for token in range(shape.context_tokens)
        ]
        maximum = max(scores)
        weights = [math.exp(score - maximum) for score in scores]
        denominator = sum(weights)
        numeric_output = [
            sum(
                weights[token]
                / denominator
                * (key_value(token, column) * 0.5 + 0.125)
                for token in range(shape.context_tokens)
            )
            for column in range(dimension)
        ]
        output = [round(value, 4) for value in numeric_output]
        return KernelMeasurement(
            _digest(output),
            max(1, time.perf_counter_ns() - started),
            tuple(numeric_output),
        )

    def _candidate_model_shape(
        self,
        shape: PagedAttentionShape,
        configuration: MetalThreadConfiguration | None = None,
    ) -> KernelMeasurement:
        return self._run_program(
            _model_paged_attention_program(shape, configuration),
            shape.head_dimension,
            decimal_places=4,
        )

    def _run_program(
        self, program: str, output_count: int, *, decimal_places: int = 5
    ) -> KernelMeasurement:
        if not 3 <= decimal_places <= 7:
            raise ValueError("probe decimal places must be between 3 and 7")
        with tempfile.TemporaryDirectory(prefix="vllm-apple-metal-probe-") as cache:
            environment = os.environ.copy()
            environment["CLANG_MODULE_CACHE_PATH"] = cache
            environment["SWIFT_MODULECACHE_PATH"] = cache
            completed = subprocess.run(
                [
                    str(self.swift_executable),
                    "-module-cache-path",
                    cache,
                    "-e",
                    program,
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=True,
                timeout=self.timeout_seconds,
                env=environment,
            )
        if len(completed.stdout) > self.maximum_output_bytes:
            raise RuntimeError("Metal probe output exceeded its bound")
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict) or set(payload) != {
            "output",
            "latency_nanoseconds",
        }:
            raise RuntimeError("Metal probe returned invalid fields")
        output = payload["output"]
        if (
            not isinstance(output, list)
            or len(output) != output_count
            or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in output
            )
        ):
            raise RuntimeError("Metal probe returned invalid output")
        return KernelMeasurement(
            _digest([round(float(value), decimal_places) for value in output]),
            payload["latency_nanoseconds"],
            tuple(float(value) for value in output),
        )


def _digest(values: list[float]) -> str:
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _model_paged_attention_program(
    shape: PagedAttentionShape,
    configuration: MetalThreadConfiguration | None = None,
) -> str:
    selected = configuration or MetalThreadConfiguration(256, 256, 128)
    template = r'''
import Foundation
import Metal
func fail(_ message: String) -> Never { FileHandle.standardError.write(Data(message.utf8)); exit(2) }
guard let device = MTLCreateSystemDefaultDevice() else { fail("metal_device_unavailable") }
let source = """
#include <metal_stdlib>
using namespace metal;
kernel void attention_scores(device const float *pages [[buffer(0)]], device const uint *blocks [[buffer(1)]], device const float *query [[buffer(2)]], device float *scores [[buffer(3)]], uint token [[thread_position_in_grid]]) {
    if (token >= __CONTEXT__) return;
    uint physical = blocks[token / __BLOCK__] * __BLOCK__ + token % __BLOCK__;
    float score = 0.0f;
    for (uint column = 0; column < __DIMENSION__; ++column) score += query[column] * pages[physical * __DIMENSION__ + column];
    scores[token] = score * __SCALE__f;
}
kernel void attention_softmax(device float *scores [[buffer(0)]], uint lane [[thread_index_in_threadgroup]]) {
    constexpr uint width = __SOFTMAX_WIDTH__;
    threadgroup float scratch[256];
    float localMaximum = -INFINITY;
    for (uint token = lane; token < __CONTEXT__; token += width) localMaximum = max(localMaximum, scores[token]);
    scratch[lane] = localMaximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = width / 2; offset > 0; offset /= 2) {
        if (lane < offset) scratch[lane] = max(scratch[lane], scratch[lane + offset]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = scratch[0];
    float localSum = 0.0f;
    for (uint token = lane; token < __CONTEXT__; token += width) {
        scores[token] = exp(scores[token] - maximum);
        localSum += scores[token];
    }
    scratch[lane] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = width / 2; offset > 0; offset /= 2) {
        if (lane < offset) scratch[lane] += scratch[lane + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denominator = scratch[0];
    for (uint token = lane; token < __CONTEXT__; token += width) scores[token] /= denominator;
}
kernel void attention_output(device const float *pages [[buffer(0)]], device const uint *blocks [[buffer(1)]], device const float *scores [[buffer(2)]], device float *output [[buffer(3)]], uint column [[thread_position_in_grid]]) {
    if (column >= __DIMENSION__) return;
    float value = 0.0f;
    for (uint token = 0; token < __CONTEXT__; ++token) {
        uint physical = blocks[token / __BLOCK__] * __BLOCK__ + token % __BLOCK__;
        value += scores[token] * (pages[physical * __DIMENSION__ + column] * 0.5f + 0.125f);
    }
    output[column] = value;
}
"""
let library = try! device.makeLibrary(source: source, options: nil)
let scorePipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "attention_scores")!)
let softmaxPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "attention_softmax")!)
let outputPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "attention_output")!)
let queue = device.makeCommandQueue()!
let pages = (0..<(__BLOCKS__ * __BLOCK__ * __DIMENSION__)).map { index -> Float in
    let page = index / (__BLOCK__ * __DIMENSION__); let token = (index / __DIMENSION__) % __BLOCK__; let column = index % __DIMENSION__
    return Float((page * 19 + token * 7 + column) % 23) / 23.0
}
let blocks = (0..<__BLOCKS__).reversed().map { UInt32($0) }
let query = (0..<__DIMENSION__).map { Float(($0 * 3) % 11) / 11.0 }
func buffer<T>(_ values: [T]) -> MTLBuffer { values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)! } }
let pageBuffer = buffer(pages), blockBuffer = buffer(blocks), queryBuffer = buffer(query)
let scores = device.makeBuffer(length: __CONTEXT__ * MemoryLayout<Float>.stride)!
let output = device.makeBuffer(length: __DIMENSION__ * MemoryLayout<Float>.stride)!
let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
let started = DispatchTime.now().uptimeNanoseconds
encoder.setComputePipelineState(scorePipeline); encoder.setBuffer(pageBuffer, offset: 0, index: 0); encoder.setBuffer(blockBuffer, offset: 0, index: 1); encoder.setBuffer(queryBuffer, offset: 0, index: 2); encoder.setBuffer(scores, offset: 0, index: 3)
if scorePipeline.maxTotalThreadsPerThreadgroup < __SCORE_WIDTH__ { fail("score_threadgroup_unsupported") }
encoder.dispatchThreads(MTLSize(width: __CONTEXT__, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: __SCORE_WIDTH__, height: 1, depth: 1)); encoder.memoryBarrier(scope: .buffers)
encoder.setComputePipelineState(softmaxPipeline); encoder.setBuffer(scores, offset: 0, index: 0)
if softmaxPipeline.maxTotalThreadsPerThreadgroup < __SOFTMAX_WIDTH__ { fail("softmax_threadgroup_unsupported") }
encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: __SOFTMAX_WIDTH__, height: 1, depth: 1)); encoder.memoryBarrier(scope: .buffers)
encoder.setComputePipelineState(outputPipeline); encoder.setBuffer(pageBuffer, offset: 0, index: 0); encoder.setBuffer(blockBuffer, offset: 0, index: 1); encoder.setBuffer(scores, offset: 0, index: 2); encoder.setBuffer(output, offset: 0, index: 3)
if outputPipeline.maxTotalThreadsPerThreadgroup < __OUTPUT_WIDTH__ { fail("output_threadgroup_unsupported") }
encoder.dispatchThreads(MTLSize(width: __DIMENSION__, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: __OUTPUT_WIDTH__, height: 1, depth: 1)); encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
if command.status != .completed { fail("command_failed") }
let elapsed = max(1, DispatchTime.now().uptimeNanoseconds - started), pointer = output.contents().bindMemory(to: Float.self, capacity: __DIMENSION__)
let payload: [String: Any] = ["output": (0..<__DIMENSION__).map { Double(pointer[$0]) }, "latency_nanoseconds": elapsed]
print(String(data: try! JSONSerialization.data(withJSONObject: payload), encoding: .utf8)!)
'''
    replacements = {
        "__CONTEXT__": str(shape.context_tokens),
        "__BLOCK__": str(shape.block_tokens),
        "__BLOCKS__": str(shape.blocks_per_sequence),
        "__DIMENSION__": str(shape.head_dimension),
        "__SCALE__": format(1 / math.sqrt(shape.head_dimension), ".17g"),
        "__SCORE_WIDTH__": str(selected.score_width),
        "__SOFTMAX_WIDTH__": str(selected.softmax_width),
        "__OUTPUT_WIDTH__": str(selected.output_width),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    return template.strip()


def _thread_configurations(
    shape: PagedAttentionShape,
) -> tuple[MetalThreadConfiguration, ...]:
    maximum_output = max(width for width in _THREAD_WIDTHS if width <= max(32, shape.head_dimension))
    widths = tuple(width for width in _THREAD_WIDTHS if width <= max(32, shape.context_tokens))
    return tuple(
        MetalThreadConfiguration(width, width, min(width, maximum_output))
        for width in widths[-4:]
    )


def _select_tuning_winner(
    candidates: tuple[tuple[MetalThreadConfiguration, KernelProbeResult], ...],
) -> MetalThreadConfiguration:
    passing = [
        item
        for item in candidates
        if item[1].passed and item[1].candidate_latency_nanoseconds is not None
    ]
    if not passing:
        raise ValueError("no correct Metal thread configuration was found")
    fastest = min(result.candidate_latency_nanoseconds for _, result in passing)
    tied = [
        item
        for item in passing
        if item[1].candidate_latency_nanoseconds <= fastest * _TUNING_TIE_RATIO
    ]
    return min(
        tied,
        key=lambda item: (
            item[0].score_width + item[0].softmax_width + item[0].output_width,
            item[0].score_width,
            item[0].softmax_width,
            item[0].output_width,
        ),
    )[0]
