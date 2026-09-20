"""Incremental multi-block Qwen3-VL vision Core ML graph builder."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .qwen3_vl_ane import Qwen3VLVisionANEAdapterSpec
from .qwen3_vl_attention_coreml import _fixed_rope, _reference_from_hidden
from .qwen3_vl_block_coreml import _mlp_reference
from .qwen3_vl_conversion_plan import Qwen3VLCoreMLConversionPlan
from .qwen3_vl_graph_spec import build_qwen3_vl_coreml_graph_spec
from .qwen3_vl_mlp_coreml import _PREDICTION_PROGRAM


TOWER_MAXIMUM_SCALED_ERROR = 3e-2


def build_qwen3_vl_tower_blocks_coreml(
    staged_weights: Path,
    destination: Path,
    source: Qwen3VLVisionANEAdapterSpec,
    plan: Qwen3VLCoreMLConversionPlan,
    *,
    block_count: int,
    profile_index: int = 0,
) -> dict[str, object]:
    output = destination.expanduser().resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Qwen3-VL tower Core ML destination must be new")
    graph = build_qwen3_vl_coreml_graph_spec(staged_weights, source, plan)
    if (
        type(block_count) is not int
        or not 2 <= block_count <= source.depth
        or type(profile_index) is not int
        or not 0 <= profile_index < len(graph.profiles)
    ):
        raise ValueError("Qwen3-VL tower Core ML request is invalid")
    try:
        import coremltools as ct
        import numpy as np
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ImportError as error:
        raise RuntimeError("Qwen3-VL tower build requires coremltools and NumPy") from error

    root = staged_weights.expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_bytes())
    records = {record["name"]: record for record in manifest["records"]}
    layers = tuple(_layer_weights(root, records, layer, np) for layer in range(block_count))
    profile = graph.profiles[profile_index]
    tokens = profile.pixel_values_shape[0]
    shape = (tokens, source.hidden_size)
    cos, sin = _fixed_rope(profile.grid_thw, source, np)
    vector = np.asarray(
        [((index % 31) - 15) / 16 for index in range(source.hidden_size)],
        dtype=np.float32,
    )
    reference = np.tile(vector, (tokens, 1)).astype(np.float16)
    for weights in layers:
        reference = _reference_from_hidden(
            reference,
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
        reference = _mlp_reference(reference, weights, np)

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    reference_name = "dense_input_reference.fp16"
    (temporary / reference_name).write_bytes(reference.tobytes(order="C"))
    package = temporary / "qwen3_vl_tower_blocks.mlpackage"
    compiled_root = temporary / "compiled"
    compiled_root.mkdir(mode=0o700)
    try:
        @mb.program(
            input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp16)],
            opset_version=ct.target.macOS15,
        )
        def program(hidden_states):
            value = hidden_states
            for layer, weights in enumerate(layers):
                value = _block_mil(
                    value, weights, layer, tokens, source, graph.head_dimension,
                    cos, sin, mb, np,
                )
            return mb.identity(x=value, name="tower_hidden_states")

        model = ct.convert(
            program,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
        )
        model.user_defined_metadata["vllm-apple.graph-id"] = graph.graph_id
        model.user_defined_metadata["vllm-apple.partition"] = f"tower-blocks-0-{block_count - 1}-v1"
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
            timeout=600,
            check=False,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        )
        if completed.returncode != 0:
            raise RuntimeError("Core ML compiler rejected Qwen3-VL tower: " + completed.stderr[-2048:])
        compiled = tuple(compiled_root.glob("*.mlmodelc"))
        if len(compiled) != 1:
            raise RuntimeError("Qwen3-VL tower compiler output is invalid")
        report = {
            "schema_version": 1,
            "graph_id": graph.graph_id,
            "partition": f"tower-blocks-0-{block_count - 1}-v1",
            "block_count": block_count,
            "grid_thw": list(profile.grid_thw),
            "input_shape": list(shape),
            "output_shape": list(shape),
            "maximum_scaled_error": TOWER_MAXIMUM_SCALED_ERROR,
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


def qualify_qwen3_vl_tower_blocks_coreml(package_root: Path) -> dict[str, object]:
    root = package_root.expanduser().resolve(strict=True)
    report = json.loads((root / "report.json").read_bytes())
    reference = root / report["reference_file"]
    model = (root / report["compiled_model"]).resolve(strict=True)
    expected_limit = TOWER_MAXIMUM_SCALED_ERROR
    if (
        report.get("maximum_scaled_error") != expected_limit
        or hashlib.sha256(reference.read_bytes()).hexdigest() != report.get("reference_sha256")
        or not model.is_dir()
        or root not in model.parents
    ):
        raise ValueError("Qwen3-VL tower Core ML artifact is invalid")
    rows, columns = report["input_shape"]
    completed = subprocess.run(
        [
            "/usr/bin/swift", "-e", _PREDICTION_PROGRAM, str(model), str(rows),
            str(columns), str(reference), "tower_hidden_states",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=600,
        check=False,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    )
    if completed.returncode != 0 or len(completed.stdout) > 4096:
        raise RuntimeError("Qwen3-VL tower prediction failed: " + completed.stderr[-2048:])
    prediction = json.loads(completed.stdout)
    if (
        prediction.get("output_count") != rows * columns
        or type(prediction.get("maximum_scaled_error")) not in (int, float)
        or prediction["maximum_scaled_error"] > expected_limit
        or type(prediction.get("latency_nanoseconds")) is not int
        or prediction["latency_nanoseconds"] <= 0
    ):
        raise RuntimeError("Qwen3-VL tower output does not match reference")
    return {**prediction, "passed": True, "partition": report["partition"]}


def _block_mil(value, weights, layer, tokens, source, head_dimension, cos, sin, mb, np):
    scalar = weights["norm1.weight"].dtype.type
    normalized = mb.layer_norm(
        x=value, axes=[-1], gamma=weights["norm1.weight"], beta=weights["norm1.bias"],
        epsilon=scalar(1e-6), name=f"block_{layer}_norm1",
    )
    qkv = mb.linear(x=normalized, weight=weights["attn.qkv.weight"], bias=weights["attn.qkv.bias"])
    qkv = mb.reshape(x=qkv, shape=[tokens, 3, source.attention_heads, head_dimension])
    qkv = mb.transpose(x=qkv, perm=[1, 0, 2, 3])
    query, key, attention_value = mb.split(x=qkv, num_splits=3, axis=0)
    query, key = _rope(query, cos, sin, mb, np), _rope(key, cos, sin, mb, np)
    query, key, attention_value = (
        mb.transpose(x=query, perm=[0, 2, 1, 3]),
        mb.transpose(x=key, perm=[0, 2, 1, 3]),
        mb.transpose(x=attention_value, perm=[0, 2, 1, 3]),
    )
    attended = mb.scaled_dot_product_attention(query=query, key=key, value=attention_value)
    attended = mb.reshape(
        x=mb.transpose(x=attended, perm=[0, 2, 1, 3]), shape=[tokens, source.hidden_size]
    )
    projected = mb.linear(
        x=attended, weight=weights["attn.proj.weight"], bias=weights["attn.proj.bias"]
    )
    attention_hidden = mb.add(x=value, y=projected, name=f"block_{layer}_attention")
    normalized_mlp = mb.layer_norm(
        x=attention_hidden, axes=[-1], gamma=weights["norm2.weight"],
        beta=weights["norm2.bias"], epsilon=scalar(1e-6),
    )
    expanded = mb.linear(
        x=normalized_mlp, weight=weights["mlp.linear_fc1.weight"],
        bias=weights["mlp.linear_fc1.bias"],
    )
    activated = mb.gelu(x=expanded, mode="TANH_APPROXIMATION")
    projected_mlp = mb.linear(
        x=activated, weight=weights["mlp.linear_fc2.weight"],
        bias=weights["mlp.linear_fc2.bias"],
    )
    return mb.add(x=attention_hidden, y=projected_mlp, name=f"block_{layer}_hidden")


def _rope(value, cos, sin, mb, numpy):
    first, second = mb.split(x=value, num_splits=2, axis=-1)
    rotated = mb.concat(
        values=[mb.mul(x=second, y=cos.dtype.type(-1)), first], axis=-1
    )
    return mb.add(x=mb.mul(x=value, y=cos), y=mb.mul(x=rotated, y=sin))


def _layer_weights(root, records, layer, numpy):
    prefix = f"vision_tower.blocks.{layer}"
    return {
        name: _weight(root, records, f"{prefix}.{name}", numpy)
        for name in (
            "norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias",
            "attn.proj.weight", "attn.proj.bias", "norm2.weight", "norm2.bias",
            "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight", "mlp.linear_fc2.bias",
        )
    }


def _weight(root, records, name, numpy):
    record = records.get(name)
    if not isinstance(record, dict):
        raise ValueError(f"Qwen3-VL staged weight is missing: {name}")
    return numpy.fromfile(root / record["file"], dtype="<f2").reshape(record["shape"])
