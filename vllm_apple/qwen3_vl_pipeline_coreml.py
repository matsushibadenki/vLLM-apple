"""Qualify direct Core ML handoff across Qwen3-VL vision tower segments."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import tempfile
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
    guard (array.dataType == .float16 || array.dataType == .float32),
          reference.count == array.count * MemoryLayout<UInt16>.size else {
        throw NSError(domain: "vllm-apple", code: 1)
    }
    let data = Data(
        bytes: array.dataPointer,
        count: array.count * (array.dataType == .float16 ? 2 : 4)
    )
    var maximumAbsoluteError = 0.0
    var maximumScaledError = 0.0
    var nonfiniteCount = 0
    reference.withUnsafeBytes { raw in
        let expected = raw.bindMemory(to: UInt16.self)
        for index in 0..<array.count {
            let lhs: Double
            if array.dataType == .float16 {
                let actual = array.dataPointer.assumingMemoryBound(to: UInt16.self)
                lhs = Double(Float(Float16(bitPattern: actual[index])))
            } else {
                let actual = array.dataPointer.assumingMemoryBound(to: Float.self)
                lhs = Double(actual[index])
            }
            let rhs = Double(Float(Float16(bitPattern: expected[index])))
            let difference = abs(lhs - rhs)
            if !lhs.isFinite { nonfiniteCount += 1 }
            maximumAbsoluteError = max(maximumAbsoluteError, difference)
            maximumScaledError = max(maximumScaledError, difference / max(1.0, abs(rhs)))
        }
    }
    return [
        "output_count": array.count,
        "maximum_absolute_error": maximumAbsoluteError,
        "maximum_scaled_error": maximumScaledError,
        "nonfinite_count": nonfiniteCount,
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
let outputDirectory = CommandLine.arguments.count > 14 && CommandLine.arguments[14] != "-"
    ? CommandLine.arguments[14] : nil
var inputs: [MLMultiArray] = []
if CommandLine.arguments.count > 16 {
    let patchModel = try MLModel(
        contentsOf: URL(fileURLWithPath: CommandLine.arguments[15]),
        configuration: configuration
    )
    for path in CommandLine.arguments[16...] {
        let pixelData = try Data(contentsOf: URL(fileURLWithPath: path))
        guard pixelData.count == 256 * 1536 * MemoryLayout<UInt16>.size else {
            throw NSError(domain: "vllm-apple", code: 4)
        }
        let pixels = try MLMultiArray(shape: [256, 1536], dataType: .float16)
        pixelData.withUnsafeBytes { raw in
            pixels.dataPointer.copyMemory(from: raw.baseAddress!, byteCount: pixelData.count)
        }
        let patchProvider = try MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: pixels)
        ])
        let patchResult = try patchModel.prediction(from: patchProvider)
        guard let patchOutput = patchResult.featureValue(
            for: "patch_hidden_states"
        )?.multiArrayValue else {
            throw NSError(domain: "vllm-apple", code: 5)
        }
        inputs.append(patchOutput)
    }
} else {
    let input = try MLMultiArray(shape: [256, 1024], dataType: .float16)
    for row in 0..<256 {
        for column in 0..<1024 {
            input[row * 1024 + column] = NSNumber(
                value: Float((column % 31) - 15) / 16.0
            )
        }
    }
    inputs.append(input)
}
let repetitions = CommandLine.arguments.count > 13
    ? Int(CommandLine.arguments[13])! : 1
var runs: [[String: Any]] = []
for (requestIndex, input) in inputs.enumerated() {
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
        if repetition == repetitions - 1, let baseDirectory = outputDirectory {
            let directory: String
            if inputs.count == 1 {
                directory = baseDirectory
            } else {
                directory = URL(fileURLWithPath: baseDirectory).appendingPathComponent(
                    "request_\(requestIndex)"
                ).path
                if index == 0 {
                    try FileManager.default.createDirectory(
                        atPath: directory,
                        withIntermediateDirectories: false,
                        attributes: [.posixPermissions: 0o700]
                    )
                }
            }
            let name = index == 3 ? "final.fp16" : "deepstack_\(index).fp16"
            let url = URL(fileURLWithPath: directory).appendingPathComponent(name)
            let data = Data(
                bytes: merged.dataPointer,
                count: merged.count * MemoryLayout<UInt16>.size
            )
            try data.write(to: url, options: [.atomic])
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: url.path
            )
        }
        current = main
    }
    let total = DispatchTime.now().uptimeNanoseconds - pipelineStarted
    runs.append([
        "request_index": requestIndex,
        "index": repetition,
        "peak_rss_bytes": peakRSS(),
        "stages": stages,
        "total_latency_nanoseconds": total,
    ])
  }
}
let payload: [String: Any] = [
    "compute_units": "cpu_and_neural_engine",
    "model_load_count": CommandLine.arguments.count > 16 ? 5 : 4,
    "request_count": inputs.count,
    "runs": runs,
]
let encoded = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
print(String(data: encoded, encoding: .utf8)!)
'''.strip()


def qualify_qwen3_vl_segment_pipeline_coreml(
    package_roots: tuple[Path, Path, Path, Path],
    *,
    repetitions: int = 1,
    patch_package_root: Path | None = None,
    pixel_values_file: Path | None = None,
    pixel_values_files: tuple[Path, ...] | None = None,
    _transport_directory: Path | None = None,
) -> dict[str, object]:
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError("Qwen3-VL pipeline repetition count is invalid")
    expected = (
        {
            "tower-blocks-0-5-deepstack-0-v1",
            "tower-blocks-0-5-deepstack-0-fp32-prefix-1-v1",
            "tower-blocks-0-5-deepstack-0-fp32-v1",
        },
        {
            "tower-blocks-6-11-deepstack-1-v1",
            "tower-blocks-6-11-deepstack-1-fp32-prefix-1-v1",
            "tower-blocks-6-11-deepstack-1-fp32-v1",
        },
        {
            "tower-blocks-12-17-deepstack-2-v1",
            "tower-blocks-12-17-deepstack-2-fp32-prefix-1-v1",
            "tower-blocks-12-17-deepstack-2-fp32-v1",
        },
        {
            "tower-blocks-18-23-final-v1",
            "tower-blocks-18-23-final-fp32-prefix-1-v1",
            "tower-blocks-18-23-final-fp32-v1",
        },
    )
    arguments = ["/usr/bin/swift", "-e", _PIPELINE_PROGRAM]
    graph_id = None
    previous_main_precision = None
    for root_value, partitions in zip(package_roots, expected, strict=True):
        root = root_value.expanduser().resolve(strict=True)
        report = json.loads((root / "report.json").read_bytes())
        if report.get("partition") not in partitions:
            raise ValueError("Qwen3-VL pipeline partition order is invalid")
        if graph_id is None:
            graph_id = report.get("graph_id")
        elif report.get("graph_id") != graph_id:
            raise ValueError("Qwen3-VL pipeline graph provenance changed")
        input_precision = report.get("input_precision", "fp16")
        main_output_precision = report.get("main_output_precision", "fp16")
        if (
            input_precision not in {"fp16", "fp32"}
            or main_output_precision not in {"fp16", "fp32"}
            or (previous_main_precision is None and input_precision != "fp16")
            or (
                previous_main_precision is not None
                and input_precision != previous_main_precision
            )
        ):
            raise ValueError("Qwen3-VL pipeline precision handoff is invalid")
        previous_main_precision = main_output_precision
        arguments.extend(
            (
                str((root / report["compiled_model"]).resolve(strict=True)),
                str((root / report["main_reference_file"]).resolve(strict=True)),
                str((root / report["deepstack_reference_file"]).resolve(strict=True)),
            )
        )
    arguments.append(str(repetitions))
    arguments.append(str(_transport_directory) if _transport_directory is not None else "-")
    if pixel_values_file is not None and pixel_values_files is not None:
        raise ValueError("Qwen3-VL patch pipeline accepts one pixel input mode")
    pixel_inputs = (
        pixel_values_files
        if pixel_values_files is not None
        else (() if pixel_values_file is None else (pixel_values_file,))
    )
    if (patch_package_root is None) != (not pixel_inputs):
        raise ValueError("Qwen3-VL patch model and pixel input must be supplied together")
    if patch_package_root is not None and pixel_inputs:
        if not 1 <= len(pixel_inputs) <= 8:
            raise ValueError("Qwen3-VL patch pipeline request count is invalid")
        patch_root = patch_package_root.expanduser().resolve(strict=True)
        patch_report = json.loads((patch_root / "report.json").read_bytes())
        if (
            patch_report.get("partition")
            not in {"patch-position-v1", "patch-position-fp32-v1"}
            or patch_report.get("graph_id") != graph_id
        ):
            raise ValueError("Qwen3-VL patch pipeline identity is invalid")
        resolved_pixels = tuple(
            pixels.expanduser().resolve(strict=True) for pixels in pixel_inputs
        )
        if any(pixels.stat().st_size != 256 * 1536 * 2 for pixels in resolved_pixels):
            raise ValueError("Qwen3-VL patch pipeline identity is invalid")
        arguments.append(
            str((patch_root / patch_report["compiled_model"]).resolve(strict=True))
        )
        arguments.extend(str(pixels) for pixels in resolved_pixels)
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
        or len(runs) != repetitions * max(1, len(pixel_inputs))
        or result.get("request_count") != max(1, len(pixel_inputs))
        or result.get("model_load_count") != (5 if pixel_inputs else 4)
    ):
        raise RuntimeError("Qwen3-VL Core ML pipeline result is invalid")
    expected_digests: dict[int, tuple[str, ...]] = {}
    reference_qualification = patch_package_root is None
    for run in runs:
        stages = run.get("stages")
        request_index = run.get("request_index", 0)
        if (
            not isinstance(stages, list)
            or len(stages) != 4
            or type(run.get("peak_rss_bytes")) is not int
            or run["peak_rss_bytes"] <= 0
            or type(request_index) is not int
            or not 0 <= request_index < max(1, len(pixel_inputs))
        ):
            raise RuntimeError("Qwen3-VL Core ML pipeline run is invalid")
        for stage in stages:
            if (
                stage.get("main", {}).get("output_count") != 256 * 1024
                or stage.get("merged", {}).get("output_count") != 64 * 2048
                or stage.get("main", {}).get("nonfinite_count") != 0
                or stage.get("merged", {}).get("nonfinite_count") != 0
                or (
                    reference_qualification
                    and stage.get("merged", {}).get(
                        "maximum_scaled_error", float("inf")
                    )
                    > DEEPSTACK_MAXIMUM_SCALED_ERROR
                )
            ):
                raise RuntimeError("Qwen3-VL Core ML pipeline output mismatch")
        digests = tuple(stage["merged"]["output_sha256"] for stage in stages)
        previous_digests = expected_digests.setdefault(request_index, digests)
        if digests != previous_digests:
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
        "reference_qualified": reference_qualification,
        "passed": True,
    }


def export_qwen3_vl_segment_pipeline_coreml(
    package_roots: tuple[Path, Path, Path, Path],
    destination: Path,
) -> dict[str, object]:
    output = destination.expanduser().resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Qwen3-VL transport destination must be new")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    try:
        qualification = qualify_qwen3_vl_segment_pipeline_coreml(
            package_roots,
            _transport_directory=temporary,
        )
        names = ("final.fp16", "deepstack_0.fp16", "deepstack_1.fp16", "deepstack_2.fp16")
        records = []
        expected_bytes = 64 * 2048 * 2
        for name in names:
            path = temporary / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_bytes:
                raise RuntimeError("Qwen3-VL transport payload is invalid")
            records.append(
                {
                    "name": name.removesuffix(".fp16"),
                    "file": name,
                    "shape": [64, 2048],
                    "dtype": "float16",
                    "bytes": expected_bytes,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        manifest = {
            "schema_version": 1,
            "graph_id": qualification["graph_id"],
            "grid_thw": [1, 16, 16],
            "records": records,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.chmod(manifest_path, 0o600)
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
