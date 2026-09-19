"""Qualify direct Core ML handoff across Qwen3-VL vision tower segments."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .qwen3_vl_deepstack_coreml import (
    DEEPSTACK_MAXIMUM_SCALED_ERROR,
    MAIN_MAXIMUM_SCALED_ERROR,
)


_PIPELINE_PROGRAM = r'''
import CoreML
import CryptoKit
import Darwin
import Foundation

func error(_ array: MLMultiArray, _ path: String) throws -> [String: Any] {
    let reference = try Data(contentsOf: URL(fileURLWithPath: path))
    guard array.dataType == .float16,
          reference.count == array.count * MemoryLayout<UInt16>.size else {
        throw NSError(domain: "vllm-apple", code: 1)
    }
    let actual = array.dataPointer.assumingMemoryBound(to: UInt16.self)
    let data = Data(bytes: array.dataPointer, count: reference.count)
    var maximumAbsoluteError = 0.0
    var maximumScaledError = 0.0
    reference.withUnsafeBytes { raw in
        let expected = raw.bindMemory(to: UInt16.self)
        for index in 0..<array.count {
            let lhs = Double(Float(Float16(bitPattern: actual[index])))
            let rhs = Double(Float(Float16(bitPattern: expected[index])))
            let difference = abs(lhs - rhs)
            maximumAbsoluteError = max(maximumAbsoluteError, difference)
            maximumScaledError = max(maximumScaledError, difference / max(1.0, abs(rhs)))
        }
    }
    return [
        "output_count": array.count,
        "maximum_absolute_error": maximumAbsoluteError,
        "maximum_scaled_error": maximumScaledError,
        "output_sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
    ]
}

func peakRSS() -> Int64 {
    var usage = rusage()
    guard getrusage(RUSAGE_SELF, &usage) == 0 else { return -1 }
    return Int64(usage.ru_maxrss)
}

let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuAndNeuralEngine
var models: [MLModel] = []
for index in 0..<4 {
    models.append(try MLModel(
        contentsOf: URL(fileURLWithPath: CommandLine.arguments[1 + index * 3]),
        configuration: configuration
    ))
}
let input = try MLMultiArray(shape: [256, 1024], dataType: .float16)
for row in 0..<256 {
    for column in 0..<1024 {
        input[row * 1024 + column] = NSNumber(value: Float((column % 31) - 15) / 16.0)
    }
}
let repetitions = CommandLine.arguments.count > 13
    ? Int(CommandLine.arguments[13])! : 1
var runs: [[String: Any]] = []
for repetition in 0..<repetitions {
    var current = input
    var stages: [[String: Any]] = []
    let pipelineStarted = DispatchTime.now().uptimeNanoseconds
    for index in 0..<4 {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: current)
        ])
        let started = DispatchTime.now().uptimeNanoseconds
        let result = try models[index].prediction(from: provider)
        let elapsed = DispatchTime.now().uptimeNanoseconds - started
        guard let main = result.featureValue(for: "tower_hidden_states")?.multiArrayValue else {
            throw NSError(domain: "vllm-apple", code: 2)
        }
        let mergedName = index == 3 ? "final_hidden_states" : "deepstack_hidden_states"
        guard let merged = result.featureValue(for: mergedName)?.multiArrayValue else {
            throw NSError(domain: "vllm-apple", code: 3)
        }
        let mainReference = CommandLine.arguments[2 + index * 3]
        let mergedReference = CommandLine.arguments[3 + index * 3]
        stages.append([
            "index": index,
            "latency_nanoseconds": elapsed,
            "main": try error(main, mainReference),
            "merged": try error(merged, mergedReference),
        ])
        current = main
    }
    let total = DispatchTime.now().uptimeNanoseconds - pipelineStarted
    runs.append([
        "index": repetition,
        "peak_rss_bytes": peakRSS(),
        "stages": stages,
        "total_latency_nanoseconds": total,
    ])
}
let payload: [String: Any] = [
    "compute_units": "cpu_and_neural_engine",
    "runs": runs,
]
let encoded = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
print(String(data: encoded, encoding: .utf8)!)
'''.strip()


def qualify_qwen3_vl_segment_pipeline_coreml(
    package_roots: tuple[Path, Path, Path, Path],
    *,
    repetitions: int = 1,
) -> dict[str, object]:
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError("Qwen3-VL pipeline repetition count is invalid")
    expected = (
        "tower-blocks-0-5-deepstack-0-v1",
        "tower-blocks-6-11-deepstack-1-v1",
        "tower-blocks-12-17-deepstack-2-v1",
        "tower-blocks-18-23-final-v1",
    )
    arguments = ["/usr/bin/swift", "-e", _PIPELINE_PROGRAM]
    graph_id = None
    for root_value, partition in zip(package_roots, expected, strict=True):
        root = root_value.expanduser().resolve(strict=True)
        report = json.loads((root / "report.json").read_bytes())
        if report.get("partition") != partition:
            raise ValueError("Qwen3-VL pipeline partition order is invalid")
        if graph_id is None:
            graph_id = report.get("graph_id")
        elif report.get("graph_id") != graph_id:
            raise ValueError("Qwen3-VL pipeline graph provenance changed")
        arguments.extend(
            (
                str((root / report["compiled_model"]).resolve(strict=True)),
                str((root / report["main_reference_file"]).resolve(strict=True)),
                str((root / report["deepstack_reference_file"]).resolve(strict=True)),
            )
        )
    arguments.append(str(repetitions))
    completed = subprocess.run(
        arguments,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=900,
        check=False,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    )
    if completed.returncode != 0 or len(completed.stdout) > 16_384:
        raise RuntimeError("Qwen3-VL Core ML pipeline failed: " + completed.stderr[-2048:])
    result = json.loads(completed.stdout)
    runs = result.get("runs")
    if (
        not isinstance(runs, list)
        or len(runs) != repetitions
    ):
        raise RuntimeError("Qwen3-VL Core ML pipeline result is invalid")
    expected_digests = None
    for run in runs:
        stages = run.get("stages")
        if (
            not isinstance(stages, list)
            or len(stages) != 4
            or type(run.get("peak_rss_bytes")) is not int
            or run["peak_rss_bytes"] <= 0
        ):
            raise RuntimeError("Qwen3-VL Core ML pipeline run is invalid")
        for stage in stages:
            if (
                stage.get("main", {}).get("output_count") != 256 * 1024
                or stage.get("merged", {}).get("output_count") != 64 * 2048
                or stage.get("merged", {}).get(
                    "maximum_scaled_error", float("inf")
                )
                > DEEPSTACK_MAXIMUM_SCALED_ERROR
            ):
                raise RuntimeError("Qwen3-VL Core ML pipeline output mismatch")
        digests = tuple(stage["merged"]["output_sha256"] for stage in stages)
        if expected_digests is None:
            expected_digests = digests
        elif digests != expected_digests:
            raise RuntimeError("Qwen3-VL Core ML pipeline output is unstable")
    stages = runs[-1]["stages"]
    inference_latencies = [
        sum(stage["latency_nanoseconds"] for stage in run["stages"])
        for run in runs
    ]
    maximum_intermediate_error = max(
        stage["main"]["maximum_scaled_error"] for stage in stages
    )
    return {
        **result,
        "graph_id": graph_id,
        "inference_latency_nanoseconds": inference_latencies[-1],
        "inference_latency_samples_nanoseconds": inference_latencies,
        "maximum_peak_rss_bytes": max(run["peak_rss_bytes"] for run in runs),
        "maximum_intermediate_scaled_error": maximum_intermediate_error,
        "intermediate_diagnostic_limit": MAIN_MAXIMUM_SCALED_ERROR,
        "passed": True,
    }
