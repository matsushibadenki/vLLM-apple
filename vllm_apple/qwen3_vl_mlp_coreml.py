"""Build and qualify a fixed-shape Qwen3-VL vision MLP residual partition."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .qwen3_vl_ane import Qwen3VLVisionANEAdapterSpec
from .qwen3_vl_conversion_plan import Qwen3VLCoreMLConversionPlan
from .qwen3_vl_graph_spec import build_qwen3_vl_coreml_graph_spec

MLP_MAXIMUM_SCALED_ERROR = 1e-2

_PREDICTION_PROGRAM = r'''
import CoreML
import CryptoKit
import Foundation

let modelPath = CommandLine.arguments[1]
let rows = Int(CommandLine.arguments[2])!
let columns = Int(CommandLine.arguments[3])!
let reference = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[4]))
let outputName = CommandLine.arguments.count > 5
    ? CommandLine.arguments[5] : "mlp_hidden_states"
let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuAndNeuralEngine
let model = try MLModel(contentsOf: URL(fileURLWithPath: modelPath), configuration: configuration)
let inputDataType: MLMultiArrayDataType = (
    CommandLine.arguments.count > 7 && CommandLine.arguments[7] == "fp32"
) ? .float32 : .float16
let input = try MLMultiArray(
    shape: [NSNumber(value: rows), NSNumber(value: columns)], dataType: inputDataType
)
if CommandLine.arguments.count > 6 {
    let inputData = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[6]))
    guard inputData.count == input.count * MemoryLayout<UInt16>.size else {
        throw NSError(domain: "vllm-apple", code: 3)
    }
    if inputDataType == .float16 {
        inputData.withUnsafeBytes { raw in
            input.dataPointer.copyMemory(from: raw.baseAddress!, byteCount: inputData.count)
        }
    } else {
        inputData.withUnsafeBytes { raw in
            let words = raw.bindMemory(to: UInt16.self)
            let values = input.dataPointer.assumingMemoryBound(to: Float.self)
            for index in 0..<input.count {
                values[index] = Float(Float16(bitPattern: words[index]))
            }
        }
    }
} else {
    for row in 0..<rows {
        for column in 0..<columns {
            input[row * columns + column] = NSNumber(value: Float((column % 31) - 15) / 16.0)
        }
    }
}
let provider = try MLDictionaryFeatureProvider(dictionary: [
    "hidden_states": MLFeatureValue(multiArray: input)
])
let started = DispatchTime.now().uptimeNanoseconds
let result = try model.prediction(from: provider)
let elapsed = DispatchTime.now().uptimeNanoseconds - started
guard let output = result.featureValue(for: outputName)?.multiArrayValue,
      output.dataType == .float16 || output.dataType == .float32 else {
    throw NSError(domain: "vllm-apple", code: 1)
}
let data = Data(
    bytes: output.dataPointer,
    count: output.count * (output.dataType == .float16 ? 2 : 4)
)
guard reference.count == output.count * MemoryLayout<UInt16>.size else {
    throw NSError(domain: "vllm-apple", code: 2)
}
var maximumAbsoluteError = 0.0
var maximumScaledError = 0.0
reference.withUnsafeBytes { raw in
    let expectedWords = raw.bindMemory(to: UInt16.self)
    for index in 0..<output.count {
        let actual: Double
        if output.dataType == .float16 {
            let actualWords = output.dataPointer.assumingMemoryBound(to: UInt16.self)
            actual = Double(Float(Float16(bitPattern: actualWords[index])))
        } else {
            let actualValues = output.dataPointer.assumingMemoryBound(to: Float.self)
            actual = Double(actualValues[index])
        }
        let expected = Double(Float(Float16(bitPattern: expectedWords[index])))
        let difference = abs(actual - expected)
        maximumAbsoluteError = max(maximumAbsoluteError, difference)
        maximumScaledError = max(maximumScaledError, difference / max(1.0, abs(expected)))
    }
}
let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
let payload: [String: Any] = [
    "output_count": output.count,
    "output_sha256": digest,
    "latency_nanoseconds": elapsed,
    "maximum_absolute_error": maximumAbsoluteError,
    "maximum_scaled_error": maximumScaledError,
    "compute_units": "cpu_and_neural_engine"
]
let encoded = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
print(String(data: encoded, encoding: .utf8)!)
'''.strip()


def build_qwen3_vl_mlp_coreml(
    staged_weights: Path,
    destination: Path,
    source: Qwen3VLVisionANEAdapterSpec,
    plan: Qwen3VLCoreMLConversionPlan,
    *,
    layer: int = 0,
    profile_index: int = 0,
) -> dict[str, object]:
    output = destination.expanduser().resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Qwen3-VL MLP Core ML destination must be new")
    graph = build_qwen3_vl_coreml_graph_spec(staged_weights, source, plan)
    if (
        type(layer) is not int
        or not 0 <= layer < source.depth
        or type(profile_index) is not int
        or not 0 <= profile_index < len(graph.profiles)
    ):
        raise ValueError("Qwen3-VL MLP Core ML partition request is invalid")
    try:
        import coremltools as ct
        import numpy as np
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ImportError as error:
        raise RuntimeError("Qwen3-VL MLP build requires coremltools and NumPy") from error

    root = staged_weights.expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_bytes())
    records = {record["name"]: record for record in manifest["records"]}
    prefix = f"vision_tower.blocks.{layer}"
    norm_weight = _weight(root, records, f"{prefix}.norm2.weight", np)
    norm_bias = _weight(root, records, f"{prefix}.norm2.bias", np)
    fc1_weight = _weight(root, records, f"{prefix}.mlp.linear_fc1.weight", np)
    fc1_bias = _weight(root, records, f"{prefix}.mlp.linear_fc1.bias", np)
    fc2_weight = _weight(root, records, f"{prefix}.mlp.linear_fc2.weight", np)
    fc2_bias = _weight(root, records, f"{prefix}.mlp.linear_fc2.bias", np)
    tokens = graph.profiles[profile_index].pixel_values_shape[0]
    shape = (tokens, source.hidden_size)
    vector = np.asarray(
        [((index % 31) - 15) / 16 for index in range(source.hidden_size)],
        dtype=np.float32,
    )
    mean = vector.mean(dtype=np.float32)
    variance = ((vector - mean) ** 2).mean(dtype=np.float32)
    normalized = (vector - mean) / np.sqrt(variance + np.float32(1e-6))
    normalized = normalized * norm_weight.astype(np.float32) + norm_bias.astype(np.float32)
    expanded = normalized @ fc1_weight.astype(np.float32).T + fc1_bias.astype(np.float32)
    gelu = np.float32(0.5) * expanded * (
        np.float32(1.0) + np.tanh(
            np.float32(0.7978845608028654)
            * (expanded + np.float32(0.044715) * expanded**3)
        )
    )
    reference_row = (
        vector + gelu @ fc2_weight.astype(np.float32).T + fc2_bias.astype(np.float32)
    ).astype(np.float16)
    reference = np.tile(reference_row, (tokens, 1))

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    reference_name = "dense_input_reference.fp16"
    (temporary / reference_name).write_bytes(reference.tobytes(order="C"))
    package = temporary / "qwen3_vl_mlp.mlpackage"
    compiled_root = temporary / "compiled"
    compiled_root.mkdir(mode=0o700)
    try:
        @mb.program(
            input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp16)],
            opset_version=ct.target.macOS15,
        )
        def program(hidden_states):
            normalized_states = mb.layer_norm(
                x=hidden_states,
                axes=[-1],
                gamma=norm_weight,
                beta=norm_bias,
                epsilon=np.float16(1e-6),
                name="norm2",
            )
            expanded_states = mb.linear(
                x=normalized_states, weight=fc1_weight, bias=fc1_bias, name="mlp_fc1"
            )
            activated = mb.gelu(
                x=expanded_states, mode="TANH_APPROXIMATION", name="mlp_gelu"
            )
            projected = mb.linear(
                x=activated, weight=fc2_weight, bias=fc2_bias, name="mlp_fc2"
            )
            return mb.add(x=hidden_states, y=projected, name="mlp_hidden_states")

        model = ct.convert(
            program,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
        )
        model.user_defined_metadata["vllm-apple.graph-id"] = graph.graph_id
        model.user_defined_metadata["vllm-apple.partition"] = f"block-{layer}-mlp-v1"
        model.save(str(package))
        completed = subprocess.run(
            [
                "/usr/bin/xcrun", "coremlcompiler", "compile", str(package),
                str(compiled_root), "--platform", "macOS", "--deployment-target", "15.0",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
            check=False,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        )
        if completed.returncode != 0:
            raise RuntimeError("Core ML compiler rejected Qwen3-VL MLP: " + completed.stderr[-2048:])
        compiled = tuple(compiled_root.glob("*.mlmodelc"))
        if len(compiled) != 1:
            raise RuntimeError("Qwen3-VL MLP compiler output is invalid")
        report = {
            "schema_version": 1,
            "graph_id": graph.graph_id,
            "partition": f"block-{layer}-mlp-v1",
            "layer": layer,
            "grid_thw": list(graph.profiles[profile_index].grid_thw),
            "input_shape": list(shape),
            "output_shape": list(shape),
            "maximum_scaled_error": MLP_MAXIMUM_SCALED_ERROR,
            "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
            "reference_file": reference_name,
            "compiled_model": f"compiled/{compiled[0].name}",
        }
        (temporary / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def qualify_qwen3_vl_mlp_coreml(package_root: Path) -> dict[str, object]:
    root = package_root.expanduser().resolve(strict=True)
    report = json.loads((root / "report.json").read_bytes())
    expected = {
        "schema_version", "graph_id", "partition", "layer", "grid_thw", "input_shape",
        "output_shape", "maximum_scaled_error", "reference_sha256", "reference_file",
        "compiled_model",
    }
    if (
        not isinstance(report, dict)
        or set(report) != expected
        or report.get("maximum_scaled_error") != MLP_MAXIMUM_SCALED_ERROR
        or not isinstance(report.get("input_shape"), list)
        or len(report["input_shape"]) != 2
        or report.get("output_shape") != report["input_shape"]
        or not isinstance(report.get("reference_file"), str)
        or Path(report["reference_file"]).name != report["reference_file"]
        or not isinstance(report.get("compiled_model"), str)
        or Path(report["compiled_model"]).is_absolute()
        or ".." in Path(report["compiled_model"]).parts
    ):
        raise ValueError("Qwen3-VL MLP Core ML report is invalid")
    reference = root / report["reference_file"]
    if hashlib.sha256(reference.read_bytes()).hexdigest() != report["reference_sha256"]:
        raise ValueError("Qwen3-VL MLP reference is invalid")
    model = (root / report["compiled_model"]).resolve(strict=True)
    if not model.is_dir() or root not in model.parents:
        raise ValueError("Qwen3-VL MLP compiled model is invalid")
    rows, columns = report["input_shape"]
    completed = subprocess.run(
        [
            "/usr/bin/swift", "-e", _PREDICTION_PROGRAM, str(model), str(rows),
            str(columns), str(reference),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
        check=False,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    )
    if completed.returncode != 0 or len(completed.stdout) > 4096:
        raise RuntimeError("Qwen3-VL MLP prediction failed: " + completed.stderr[-2048:])
    prediction = json.loads(completed.stdout)
    if (
        prediction.get("output_count") != rows * columns
        or type(prediction.get("maximum_scaled_error")) not in (int, float)
        or prediction["maximum_scaled_error"] > MLP_MAXIMUM_SCALED_ERROR
        or type(prediction.get("latency_nanoseconds")) is not int
        or prediction["latency_nanoseconds"] <= 0
    ):
        raise RuntimeError("Qwen3-VL MLP Core ML output does not match reference")
    return {**prediction, "passed": True, "partition": report["partition"]}


def _weight(root: Path, records: dict, name: str, numpy):
    record = records.get(name)
    if not isinstance(record, dict):
        raise ValueError(f"Qwen3-VL staged weight is missing: {name}")
    return numpy.fromfile(root / record["file"], dtype="<f2").reshape(record["shape"])
