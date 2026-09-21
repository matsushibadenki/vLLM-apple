"""Inspect Core ML's preferred devices for compiled Qwen3-VL artifacts."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

_COMPUTE_PLAN_PROGRAM = r'''
import CoreML
import Foundation

func deviceName(_ device: MLComputeDevice) -> String {
    switch device {
    case .cpu: return "cpu"
    case .gpu: return "gpu"
    case .neuralEngine: return "neural_engine"
    @unknown default: return "unknown"
    }
}

func inspectBlock(
    _ block: MLModelStructure.Program.Block,
    _ plan: MLComputePlan,
    _ counts: inout [String: Int],
    _ supportedCounts: inout [String: Int],
    _ cost: inout Double
) {
    for operation in block.operations {
        if let usage = plan.deviceUsage(for: operation) {
            counts[deviceName(usage.preferred), default: 0] += 1
            for device in usage.supported {
                supportedCounts[deviceName(device), default: 0] += 1
            }
        } else {
            counts["unavailable", default: 0] += 1
        }
        if let estimate = plan.estimatedCost(of: operation) {
            cost += estimate.weight
        }
        for nested in operation.blocks {
            inspectBlock(nested, plan, &counts, &supportedCounts, &cost)
        }
    }
}

let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuAndNeuralEngine
var artifacts: [[String: Any]] = []
for path in CommandLine.arguments.dropFirst() {
    let url = URL(fileURLWithPath: path)
    let plan = try await MLComputePlan.load(contentsOf: url, configuration: configuration)
    var preferred: [String: Int] = [:]
    var supported: [String: Int] = [:]
    var estimatedCost = 0.0
    switch plan.modelStructure {
    case .program(let program):
        for function in program.functions.values {
            inspectBlock(function.block, plan, &preferred, &supported, &estimatedCost)
        }
    default:
        throw NSError(domain: "vllm-apple", code: 1)
    }
    artifacts.append([
        "name": url.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent,
        "preferred_operation_counts": preferred,
        "supported_operation_counts": supported,
        "estimated_cost_weight": estimatedCost,
    ])
}
let payload: [String: Any] = [
    "compute_units": "cpu_and_neural_engine",
    "available_devices": MLComputeDevice.allComputeDevices.map(deviceName).sorted(),
    "artifacts": artifacts,
]
let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)
'''.strip()


def inspect_qwen3_vl_coreml_compute_plans(
    compiled_models: tuple[Path, ...],
) -> dict[str, object]:
    if not 1 <= len(compiled_models) <= 8:
        raise ValueError("Core ML compute plan model count is invalid")
    resolved = tuple(model.expanduser().resolve(strict=True) for model in compiled_models)
    if any(not model.is_dir() or model.suffix != ".mlmodelc" for model in resolved):
        raise ValueError("Core ML compute plan requires compiled model directories")
    completed = subprocess.run(
        ["/usr/bin/swift", "-e", _COMPUTE_PLAN_PROGRAM, *map(str, resolved)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=900,
        check=False,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    )
    if completed.returncode != 0 or len(completed.stdout) > 65_536:
        raise RuntimeError("Core ML compute plan inspection failed: " + completed.stderr[-2048:])
    report = json.loads(completed.stdout)
    artifacts = report.get("artifacts")
    if (
        report.get("compute_units") != "cpu_and_neural_engine"
        or not isinstance(artifacts, list)
        or len(artifacts) != len(resolved)
        or report.get("available_devices") != ["cpu", "gpu", "neural_engine"]
    ):
        raise RuntimeError("Core ML compute plan report is invalid")
    for artifact in artifacts:
        preferred = artifact.get("preferred_operation_counts")
        supported = artifact.get("supported_operation_counts")
        if (
            not isinstance(preferred, dict)
            or not isinstance(supported, dict)
            or not preferred
            or any(key not in {"cpu", "gpu", "neural_engine", "unavailable"} for key in preferred)
            or any(type(value) is not int or value < 0 for value in preferred.values())
            or any(key not in {"cpu", "gpu", "neural_engine"} for key in supported)
            or any(type(value) is not int or value < 0 for value in supported.values())
        ):
            raise RuntimeError("Core ML compute plan operation usage is invalid")
    return report
