"""Complete fixed-shape Qwen3-VL vision block Core ML partition."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .qwen3_vl_ane import Qwen3VLVisionANEAdapterSpec
from .qwen3_vl_attention_coreml import _fixed_rope
from .qwen3_vl_attention_coreml import _reference as _attention_reference
from .qwen3_vl_conversion_plan import Qwen3VLCoreMLConversionPlan
from .qwen3_vl_graph_spec import build_qwen3_vl_coreml_graph_spec
from .qwen3_vl_mlp_coreml import _PREDICTION_PROGRAM

BLOCK_MAXIMUM_SCALED_ERROR = 3e-2


def build_qwen3_vl_block_coreml(
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
        raise ValueError("Qwen3-VL block Core ML destination must be new")
    graph = build_qwen3_vl_coreml_graph_spec(staged_weights, source, plan)
    if (
        type(layer) is not int
        or not 0 <= layer < source.depth
        or type(profile_index) is not int
        or not 0 <= profile_index < len(graph.profiles)
    ):
        raise ValueError("Qwen3-VL block Core ML partition request is invalid")
    try:
        import coremltools as ct
        import numpy as np
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ImportError as error:
        raise RuntimeError("Qwen3-VL block build requires coremltools and NumPy") from error

    root = staged_weights.expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_bytes())
    records = {record["name"]: record for record in manifest["records"]}
    prefix = f"vision_tower.blocks.{layer}"
    weights = {
        name: _weight(root, records, f"{prefix}.{name}", np)
        for name in (
            "norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias",
            "attn.proj.weight", "attn.proj.bias", "norm2.weight", "norm2.bias",
            "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight", "mlp.linear_fc2.bias",
        )
    }
    profile = graph.profiles[profile_index]
    tokens = profile.pixel_values_shape[0]
    shape = (tokens, source.hidden_size)
    cos, sin = _fixed_rope(profile.grid_thw, source, np)
    attention_reference = _attention_reference(
        shape,
        source,
        weights["norm1.weight"],
        weights["norm1.bias"],
        weights["attn.qkv.weight"],
        weights["attn.qkv.bias"],
        weights["attn.proj.weight"],
        weights["attn.proj.bias"],
        cos,
        sin,
        np,
    )
    reference = _mlp_reference(attention_reference, weights, np)

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    reference_name = "dense_input_reference.fp16"
    (temporary / reference_name).write_bytes(reference.tobytes(order="C"))
    package = temporary / "qwen3_vl_block.mlpackage"
    compiled_root = temporary / "compiled"
    compiled_root.mkdir(mode=0o700)
    try:
        @mb.program(
            input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp16)],
            opset_version=ct.target.macOS15,
        )
        def program(hidden_states):
            normalized = mb.layer_norm(
                x=hidden_states, axes=[-1], gamma=weights["norm1.weight"],
                beta=weights["norm1.bias"], epsilon=np.float16(1e-6), name="norm1",
            )
            qkv = mb.linear(
                x=normalized, weight=weights["attn.qkv.weight"],
                bias=weights["attn.qkv.bias"], name="qkv",
            )
            qkv = mb.reshape(
                x=qkv, shape=[tokens, 3, source.attention_heads, graph.head_dimension]
            )
            qkv = mb.transpose(x=qkv, perm=[1, 0, 2, 3])
            query, key, value = mb.split(x=qkv, num_splits=3, axis=0)
            query = _rope(query, cos, sin, mb, np)
            key = _rope(key, cos, sin, mb, np)
            query = mb.transpose(x=query, perm=[0, 2, 1, 3])
            key = mb.transpose(x=key, perm=[0, 2, 1, 3])
            value = mb.transpose(x=value, perm=[0, 2, 1, 3])
            attended = mb.scaled_dot_product_attention(query=query, key=key, value=value)
            attended = mb.transpose(x=attended, perm=[0, 2, 1, 3])
            attended = mb.reshape(x=attended, shape=[tokens, source.hidden_size])
            attended = mb.linear(
                x=attended, weight=weights["attn.proj.weight"],
                bias=weights["attn.proj.bias"],
            )
            attention_hidden = mb.add(x=hidden_states, y=attended, name="attention_residual")
            normalized_mlp = mb.layer_norm(
                x=attention_hidden, axes=[-1], gamma=weights["norm2.weight"],
                beta=weights["norm2.bias"], epsilon=np.float16(1e-6), name="norm2",
            )
            expanded = mb.linear(
                x=normalized_mlp, weight=weights["mlp.linear_fc1.weight"],
                bias=weights["mlp.linear_fc1.bias"],
            )
            activated = mb.gelu(x=expanded, mode="TANH_APPROXIMATION")
            projected = mb.linear(
                x=activated, weight=weights["mlp.linear_fc2.weight"],
                bias=weights["mlp.linear_fc2.bias"],
            )
            return mb.add(x=attention_hidden, y=projected, name="block_hidden_states")

        model = ct.convert(
            program,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
        )
        model.user_defined_metadata["vllm-apple.graph-id"] = graph.graph_id
        model.user_defined_metadata["vllm-apple.partition"] = f"block-{layer}-complete-v1"
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
            raise RuntimeError("Core ML compiler rejected Qwen3-VL block: " + completed.stderr[-2048:])
        compiled = tuple(compiled_root.glob("*.mlmodelc"))
        if len(compiled) != 1:
            raise RuntimeError("Qwen3-VL block compiler output is invalid")
        report = {
            "schema_version": 1,
            "graph_id": graph.graph_id,
            "partition": f"block-{layer}-complete-v1",
            "layer": layer,
            "grid_thw": list(profile.grid_thw),
            "input_shape": list(shape),
            "output_shape": list(shape),
            "maximum_scaled_error": BLOCK_MAXIMUM_SCALED_ERROR,
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


def qualify_qwen3_vl_block_coreml(package_root: Path) -> dict[str, object]:
    root = package_root.expanduser().resolve(strict=True)
    report = json.loads((root / "report.json").read_bytes())
    reference = root / report["reference_file"]
    model = (root / report["compiled_model"]).resolve(strict=True)
    if (
        report.get("maximum_scaled_error") != BLOCK_MAXIMUM_SCALED_ERROR
        or hashlib.sha256(reference.read_bytes()).hexdigest() != report.get("reference_sha256")
        or not model.is_dir()
        or root not in model.parents
    ):
        raise ValueError("Qwen3-VL block Core ML artifact is invalid")
    rows, columns = report["input_shape"]
    completed = subprocess.run(
        [
            "/usr/bin/swift", "-e", _PREDICTION_PROGRAM, str(model), str(rows),
            str(columns), str(reference), "block_hidden_states",
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
        raise RuntimeError("Qwen3-VL block prediction failed: " + completed.stderr[-2048:])
    prediction = json.loads(completed.stdout)
    if (
        prediction.get("output_count") != rows * columns
        or type(prediction.get("maximum_scaled_error")) not in (int, float)
        or prediction["maximum_scaled_error"] > BLOCK_MAXIMUM_SCALED_ERROR
        or type(prediction.get("latency_nanoseconds")) is not int
        or prediction["latency_nanoseconds"] <= 0
    ):
        raise RuntimeError("Qwen3-VL block output does not match reference")
    return {**prediction, "passed": True, "partition": report["partition"]}


def _rope(value, cos, sin, mb, numpy):
    first, second = mb.split(x=value, num_splits=2, axis=-1)
    rotated = mb.concat(values=[mb.mul(x=second, y=numpy.float16(-1)), first], axis=-1)
    return mb.add(x=mb.mul(x=value, y=cos), y=mb.mul(x=rotated, y=sin))


def _mlp_reference(hidden, weights, numpy):
    value = hidden.astype(numpy.float32)
    mean = value.mean(axis=-1, keepdims=True)
    variance = ((value - mean) ** 2).mean(axis=-1, keepdims=True)
    normalized = (value - mean) / numpy.sqrt(variance + numpy.float32(1e-6))
    normalized = (
        normalized * weights["norm2.weight"].astype(numpy.float32)
        + weights["norm2.bias"].astype(numpy.float32)
    )
    expanded = (
        normalized @ weights["mlp.linear_fc1.weight"].astype(numpy.float32).T
        + weights["mlp.linear_fc1.bias"].astype(numpy.float32)
    )
    activated = numpy.float32(0.5) * expanded * (
        numpy.float32(1.0)
        + numpy.tanh(
            numpy.float32(0.7978845608028654)
            * (expanded + numpy.float32(0.044715) * expanded**3)
        )
    )
    projected = (
        activated @ weights["mlp.linear_fc2.weight"].astype(numpy.float32).T
        + weights["mlp.linear_fc2.bias"].astype(numpy.float32)
    )
    return (value + projected).astype(numpy.float16)


def _weight(root: Path, records: dict, name: str, numpy):
    record = records.get(name)
    if not isinstance(record, dict):
        raise ValueError(f"Qwen3-VL staged weight is missing: {name}")
    return numpy.fromfile(root / record["file"], dtype="<f2").reshape(record["shape"])
