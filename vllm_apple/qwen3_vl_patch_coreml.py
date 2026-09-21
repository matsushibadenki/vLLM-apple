"""Build the first fixed-shape Qwen3-VL Core ML vision partition."""
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

PATCH_MAXIMUM_SCALED_ERROR = 2e-3


_PATCH_PREDICTION_PROGRAM = r'''
import CoreML
import CryptoKit
import Foundation

let modelURL = URL(fileURLWithPath: CommandLine.arguments[1])
let rows = Int(CommandLine.arguments[2])!
let columns = Int(CommandLine.arguments[3])!
let reference = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[4]))
let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuAndNeuralEngine
let model = try MLModel(contentsOf: modelURL, configuration: configuration)
let input = try MLMultiArray(
    shape: [NSNumber(value: rows), NSNumber(value: columns)], dataType: .float16
)
memset(input.dataPointer, 0, input.count * MemoryLayout<UInt16>.size)
for row in 0..<rows {
    for lane in 0..<6 {
        let column = row + lane * rows
        let magnitude = Float(lane + 1) / 8.0
        input[row * columns + column] = NSNumber(
            value: lane % 2 == 0 ? magnitude : -magnitude
        )
    }
}
let provider = try MLDictionaryFeatureProvider(dictionary: [
    "pixel_values": MLFeatureValue(multiArray: input)
])
let started = DispatchTime.now().uptimeNanoseconds
let prediction = try model.prediction(from: provider)
let elapsed = DispatchTime.now().uptimeNanoseconds - started
guard let output = prediction.featureValue(
    for: "patch_hidden_states"
)?.multiArrayValue, output.dataType == .float16 else {
    throw NSError(domain: "vllm-apple", code: 1)
}
let data = Data(bytes: output.dataPointer, count: output.count * MemoryLayout<UInt16>.size)
guard reference.count == data.count else {
    throw NSError(domain: "vllm-apple", code: 2)
}
let outputWords = output.dataPointer.assumingMemoryBound(to: UInt16.self)
func orderedFP16(_ bits: UInt16) -> Int {
    if bits & 0x8000 != 0 { return Int(0x8000 - (bits & 0x7fff)) }
    return Int(0x8000 + bits)
}
var maximumULPError = 0
var maximumScaledError = 0.0
let maximumAbsoluteError = reference.withUnsafeBytes { raw -> Double in
    let referenceWords = raw.bindMemory(to: UInt16.self)
    var maximum = 0.0
    for index in 0..<output.count {
        let actual = Double(Float(Float16(bitPattern: outputWords[index])))
        let expected = Double(Float(Float16(bitPattern: referenceWords[index])))
        maximum = max(maximum, abs(actual - expected))
        maximumScaledError = max(
            maximumScaledError, abs(actual - expected) / max(1.0, abs(expected))
        )
        maximumULPError = max(
            maximumULPError,
            abs(orderedFP16(outputWords[index]) - orderedFP16(referenceWords[index]))
        )
    }
    return maximum
}
let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
let payload: [String: Any] = [
    "output_count": output.count,
    "output_sha256": digest,
    "latency_nanoseconds": elapsed,
    "maximum_absolute_error": maximumAbsoluteError,
    "maximum_ulp_error": maximumULPError,
    "maximum_scaled_error": maximumScaledError,
    "compute_units": "cpu_and_neural_engine"
]
let encoded = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
print(String(data: encoded, encoding: .utf8)!)
'''.strip()


def build_qwen3_vl_patch_coreml(
    staged_weights: Path,
    destination: Path,
    source: Qwen3VLVisionANEAdapterSpec,
    plan: Qwen3VLCoreMLConversionPlan,
    *,
    profile_index: int = 0,
    compute_precision: str = "fp16",
) -> dict[str, object]:
    """Build and compile patch projection plus fixed position embedding."""
    output = destination.expanduser().resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Qwen3-VL patch Core ML destination must be new")
    graph = build_qwen3_vl_coreml_graph_spec(staged_weights, source, plan)
    if (
        type(profile_index) is not int
        or not 0 <= profile_index < len(graph.profiles)
        or compute_precision not in {"fp16", "fp32"}
    ):
        raise ValueError("Qwen3-VL patch Core ML profile index is invalid")
    try:
        import coremltools as ct
        import numpy as np
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ImportError as error:
        raise RuntimeError("Qwen3-VL patch build requires coremltools and NumPy") from error

    root = staged_weights.expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_bytes())
    records = {record["name"]: record for record in manifest["records"]}
    weight = _load_weight(
        root, records, "vision_tower.patch_embed.proj.weight", np
    ).reshape(source.hidden_size, -1)
    bias = _load_weight(root, records, "vision_tower.patch_embed.proj.bias", np)
    position_table = _load_weight(root, records, "vision_tower.pos_embed.weight", np)
    profile = graph.profiles[profile_index]
    position = _fixed_position_embedding(position_table, profile.grid_thw, source, np)
    if tuple(weight.shape) != (source.hidden_size, profile.pixel_values_shape[1]):
        raise ValueError("Qwen3-VL patch weight does not match graph input")
    if tuple(position.shape) != (profile.pixel_values_shape[0], source.hidden_size):
        raise ValueError("Qwen3-VL position embedding does not match graph input")
    expected = position.astype(np.float32) + bias[None, :].astype(np.float32)
    rows = profile.pixel_values_shape[0]
    if profile.pixel_values_shape[1] != rows * 6:
        raise ValueError("Qwen3-VL sparse qualification pattern does not cover input")
    for row in range(rows):
        for lane in range(6):
            column = row + lane * rows
            value = (lane + 1) / 8 * (1 if lane % 2 == 0 else -1)
            expected[row] += weight[:, column].astype(np.float32) * value
    expected = expected.astype(np.float16)
    expected_digest = hashlib.sha256(expected.tobytes(order="C")).hexdigest()

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    package = temporary / "qwen3_vl_patch.mlpackage"
    compiled_root = temporary / "compiled"
    compiled_root.mkdir(mode=0o700)
    try:
        reference_name = "sparse_input_reference.fp16"
        (temporary / reference_name).write_bytes(expected.tobytes(order="C"))
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=profile.pixel_values_shape, dtype=types.fp16)
            ],
            opset_version=ct.target.macOS15,
        )
        def program(pixel_values):
            projected = mb.linear(
                x=pixel_values,
                weight=weight,
                bias=bias,
                name="patch_projection",
            )
            return mb.add(
                x=projected,
                y=position,
                name="patch_hidden_states",
            )

        model = ct.convert(
            program,
            convert_to="mlprogram",
            compute_precision=(
                ct.precision.FLOAT16
                if compute_precision == "fp16"
                else ct.precision.FLOAT32
            ),
            minimum_deployment_target=ct.target.macOS15,
        )
        model.short_description = "Qwen3-VL patch and fixed position partition"
        model.user_defined_metadata["vllm-apple.graph-id"] = graph.graph_id
        partition = (
            "patch-position-v1"
            if compute_precision == "fp16"
            else "patch-position-fp32-v1"
        )
        model.user_defined_metadata["vllm-apple.partition"] = partition
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
            raise RuntimeError(
                "Core ML compiler rejected Qwen3-VL patch partition: "
                + completed.stderr.strip()[-2048:]
            )
        compiled = tuple(compiled_root.glob("*.mlmodelc"))
        if len(compiled) != 1:
            raise RuntimeError("Qwen3-VL patch compiler output is invalid")
        report = {
            "schema_version": 1,
            "graph_id": graph.graph_id,
            "conversion_plan_id": plan.plan_id,
            "partition": partition,
            "grid_thw": list(profile.grid_thw),
            "input_name": "pixel_values",
            "input_shape": list(profile.pixel_values_shape),
            "output_name": "patch_hidden_states",
            "output_shape": [profile.pixel_values_shape[0], source.hidden_size],
            "compute_precision": compute_precision,
            "maximum_scaled_error": PATCH_MAXIMUM_SCALED_ERROR,
            "expected_sparse_input_sha256": expected_digest,
            "sparse_input_reference": reference_name,
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


def _load_weight(root: Path, records: dict, name: str, numpy):
    record = records.get(name)
    if not isinstance(record, dict):
        raise ValueError(f"Qwen3-VL staged weight is missing: {name}")
    return numpy.fromfile(root / record["file"], dtype="<f2").reshape(record["shape"])


def qualify_qwen3_vl_patch_coreml(package_root: Path) -> dict[str, object]:
    """Execute the compiled patch partition and match its full FP16 output digest."""
    root = package_root.expanduser().resolve(strict=True)
    report = json.loads((root / "report.json").read_bytes())
    expected_fields = {
        "schema_version", "graph_id", "conversion_plan_id", "partition", "grid_thw",
        "input_name", "input_shape", "output_name", "output_shape", "compute_precision",
        "maximum_scaled_error", "expected_sparse_input_sha256",
        "sparse_input_reference", "compiled_model",
    }
    if (
        not isinstance(report, dict)
        or set(report) != expected_fields
        or report["partition"] != "patch-position-v1"
        or report["input_name"] != "pixel_values"
        or report["output_name"] != "patch_hidden_states"
        or report["compute_precision"] != "fp16"
        or report["maximum_scaled_error"] != PATCH_MAXIMUM_SCALED_ERROR
        or not isinstance(report["input_shape"], list)
        or len(report["input_shape"]) != 2
        or not isinstance(report["output_shape"], list)
        or len(report["output_shape"]) != 2
        or not isinstance(report["compiled_model"], str)
        or not isinstance(report["sparse_input_reference"], str)
        or Path(report["compiled_model"]).is_absolute()
        or ".." in Path(report["compiled_model"]).parts
        or Path(report["sparse_input_reference"]).name != report["sparse_input_reference"]
    ):
        raise ValueError("Qwen3-VL patch Core ML report is invalid")
    model = (root / report["compiled_model"]).resolve(strict=True)
    if not model.is_dir() or root not in model.parents:
        raise ValueError("Qwen3-VL patch compiled model path is invalid")
    reference = root / report["sparse_input_reference"]
    if (
        not reference.is_file()
        or hashlib.sha256(reference.read_bytes()).hexdigest()
        != report["expected_sparse_input_sha256"]
    ):
        raise ValueError("Qwen3-VL patch reference output is invalid")
    rows, columns = report["input_shape"]
    completed = subprocess.run(
        [
            "/usr/bin/swift", "-e", _PATCH_PREDICTION_PROGRAM, str(model),
            str(rows), str(columns), str(reference),
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
        raise RuntimeError(
            "Qwen3-VL patch Core ML prediction failed: "
            + completed.stderr.strip()[-2048:]
        )
    try:
        prediction = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Qwen3-VL patch prediction output is invalid") from error
    expected_count = report["output_shape"][0] * report["output_shape"][1]
    if (
        not isinstance(prediction, dict)
        or prediction.get("output_count") != expected_count
        or type(prediction.get("latency_nanoseconds")) is not int
        or prediction["latency_nanoseconds"] <= 0
        or type(prediction.get("maximum_absolute_error")) not in (int, float)
        or type(prediction.get("maximum_ulp_error")) is not int
        or type(prediction.get("maximum_scaled_error")) not in (int, float)
        or prediction["maximum_scaled_error"] > PATCH_MAXIMUM_SCALED_ERROR
        or prediction.get("compute_units") != "cpu_and_neural_engine"
    ):
        raise RuntimeError("Qwen3-VL patch Core ML output does not match reference")
    return {
        **prediction,
        "passed": True,
        "graph_id": report["graph_id"],
        "partition": report["partition"],
        "grid_thw": report["grid_thw"],
    }


def _fixed_position_embedding(table, grid, source, numpy):
    frames, height, width = grid
    side = int(round(table.shape[0] ** 0.5))
    if side * side != table.shape[0]:
        raise ValueError("Qwen3-VL position table is not square")
    h = numpy.linspace(0, side - 1, height, dtype=numpy.float32)
    w = numpy.linspace(0, side - 1, width, dtype=numpy.float32)
    hf, wf = numpy.floor(h).astype(numpy.int32), numpy.floor(w).astype(numpy.int32)
    hc, wc = numpy.minimum(hf + 1, side - 1), numpy.minimum(wf + 1, side - 1)
    dh, dw = h - hf, w - wf
    weights = (
        ((1 - dh)[:, None] * (1 - dw)[None, :]).astype(numpy.float16),
        ((1 - dh)[:, None] * dw[None, :]).astype(numpy.float16),
        (dh[:, None] * (1 - dw)[None, :]).astype(numpy.float16),
        (dh[:, None] * dw[None, :]).astype(numpy.float16),
    )
    result = (
        table[(hf[:, None] * side + wf[None, :]).reshape(-1)]
        * weights[0].reshape(-1, 1)
        + table[(hf[:, None] * side + wc[None, :]).reshape(-1)]
        * weights[1].reshape(-1, 1)
        + table[(hc[:, None] * side + wf[None, :]).reshape(-1)]
        * weights[2].reshape(-1, 1)
        + table[(hc[:, None] * side + wc[None, :]).reshape(-1)]
        * weights[3].reshape(-1, 1)
    ).astype(numpy.float16)
    result = numpy.tile(result, (frames, 1))
    merge = source.spatial_merge_size
    return result.reshape(
        frames, height // merge, merge, width // merge, merge, source.hidden_size
    ).transpose(0, 1, 3, 2, 4, 5).reshape(-1, source.hidden_size)
