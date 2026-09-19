"""Fixed-RoPE Qwen3-VL vision attention Core ML partition."""
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
from .qwen3_vl_mlp_coreml import _PREDICTION_PROGRAM


ATTENTION_MAXIMUM_SCALED_ERROR = 2e-2


def build_qwen3_vl_attention_coreml(
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
        raise ValueError("Qwen3-VL attention Core ML destination must be new")
    graph = build_qwen3_vl_coreml_graph_spec(staged_weights, source, plan)
    if (
        type(layer) is not int
        or not 0 <= layer < source.depth
        or type(profile_index) is not int
        or not 0 <= profile_index < len(graph.profiles)
    ):
        raise ValueError("Qwen3-VL attention Core ML partition request is invalid")
    try:
        import coremltools as ct
        import numpy as np
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ImportError as error:
        raise RuntimeError("Qwen3-VL attention build requires coremltools and NumPy") from error

    root = staged_weights.expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_bytes())
    records = {record["name"]: record for record in manifest["records"]}
    prefix = f"vision_tower.blocks.{layer}"
    norm_weight = _weight(root, records, f"{prefix}.norm1.weight", np)
    norm_bias = _weight(root, records, f"{prefix}.norm1.bias", np)
    qkv_weight = _weight(root, records, f"{prefix}.attn.qkv.weight", np)
    qkv_bias = _weight(root, records, f"{prefix}.attn.qkv.bias", np)
    projection_weight = _weight(root, records, f"{prefix}.attn.proj.weight", np)
    projection_bias = _weight(root, records, f"{prefix}.attn.proj.bias", np)
    profile = graph.profiles[profile_index]
    tokens = profile.pixel_values_shape[0]
    shape = (tokens, source.hidden_size)
    cos, sin = _fixed_rope(profile.grid_thw, source, np)
    reference = _reference(
        shape,
        source,
        norm_weight,
        norm_bias,
        qkv_weight,
        qkv_bias,
        projection_weight,
        projection_bias,
        cos,
        sin,
        np,
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    reference_name = "dense_input_reference.fp16"
    (temporary / reference_name).write_bytes(reference.tobytes(order="C"))
    package = temporary / "qwen3_vl_attention.mlpackage"
    compiled_root = temporary / "compiled"
    compiled_root.mkdir(mode=0o700)
    try:
        @mb.program(
            input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp16)],
            opset_version=ct.target.macOS15,
        )
        def program(hidden_states):
            normalized = mb.layer_norm(
                x=hidden_states,
                axes=[-1],
                gamma=norm_weight,
                beta=norm_bias,
                epsilon=np.float16(1e-6),
                name="norm1",
            )
            qkv = mb.linear(x=normalized, weight=qkv_weight, bias=qkv_bias, name="qkv")
            qkv = mb.reshape(
                x=qkv,
                shape=[tokens, 3, source.attention_heads, graph.head_dimension],
            )
            qkv = mb.transpose(x=qkv, perm=[1, 0, 2, 3])
            query, key, value = mb.split(x=qkv, num_splits=3, axis=0)
            query = _rope_mil(query, cos, sin, mb, np)
            key = _rope_mil(key, cos, sin, mb, np)
            query = mb.transpose(x=query, perm=[0, 2, 1, 3])
            key = mb.transpose(x=key, perm=[0, 2, 1, 3])
            value = mb.transpose(x=value, perm=[0, 2, 1, 3])
            attended = mb.scaled_dot_product_attention(
                query=query, key=key, value=value, name="sdpa"
            )
            attended = mb.transpose(x=attended, perm=[0, 2, 1, 3])
            attended = mb.reshape(x=attended, shape=[tokens, source.hidden_size])
            projected = mb.linear(
                x=attended,
                weight=projection_weight,
                bias=projection_bias,
                name="attention_projection",
            )
            return mb.add(x=hidden_states, y=projected, name="attention_hidden_states")

        model = ct.convert(
            program,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
        )
        model.user_defined_metadata["vllm-apple.graph-id"] = graph.graph_id
        model.user_defined_metadata["vllm-apple.partition"] = f"block-{layer}-attention-v1"
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
                "Core ML compiler rejected Qwen3-VL attention: " + completed.stderr[-2048:]
            )
        compiled = tuple(compiled_root.glob("*.mlmodelc"))
        if len(compiled) != 1:
            raise RuntimeError("Qwen3-VL attention compiler output is invalid")
        report = {
            "schema_version": 1,
            "graph_id": graph.graph_id,
            "partition": f"block-{layer}-attention-v1",
            "layer": layer,
            "grid_thw": list(profile.grid_thw),
            "input_shape": list(shape),
            "output_shape": list(shape),
            "maximum_scaled_error": ATTENTION_MAXIMUM_SCALED_ERROR,
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


def qualify_qwen3_vl_attention_coreml(package_root: Path) -> dict[str, object]:
    root = package_root.expanduser().resolve(strict=True)
    report = json.loads((root / "report.json").read_bytes())
    reference = root / report["reference_file"]
    model = (root / report["compiled_model"]).resolve(strict=True)
    if (
        report.get("maximum_scaled_error") != ATTENTION_MAXIMUM_SCALED_ERROR
        or hashlib.sha256(reference.read_bytes()).hexdigest() != report.get("reference_sha256")
        or not model.is_dir()
        or root not in model.parents
    ):
        raise ValueError("Qwen3-VL attention Core ML artifact is invalid")
    rows, columns = report["input_shape"]
    completed = subprocess.run(
        [
            "/usr/bin/swift", "-e", _PREDICTION_PROGRAM, str(model), str(rows),
            str(columns), str(reference), "attention_hidden_states",
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
        raise RuntimeError("Qwen3-VL attention prediction failed: " + completed.stderr[-2048:])
    prediction = json.loads(completed.stdout)
    if (
        prediction.get("output_count") != rows * columns
        or type(prediction.get("maximum_scaled_error")) not in (int, float)
        or prediction["maximum_scaled_error"] > ATTENTION_MAXIMUM_SCALED_ERROR
        or type(prediction.get("latency_nanoseconds")) is not int
        or prediction["latency_nanoseconds"] <= 0
    ):
        raise RuntimeError("Qwen3-VL attention output does not match reference")
    return {**prediction, "passed": True, "partition": report["partition"]}


def _rope_mil(value, cos, sin, mb, numpy):
    first, second = mb.split(x=value, num_splits=2, axis=-1)
    rotated = mb.concat(values=[mb.mul(x=second, y=numpy.float16(-1)), first], axis=-1)
    return mb.add(x=mb.mul(x=value, y=cos), y=mb.mul(x=rotated, y=sin))


def _fixed_rope(grid, source, numpy):
    frames, height, width = grid
    merge = source.spatial_merge_size
    rows = numpy.arange(height).reshape(height // merge, merge)
    columns = numpy.arange(width).reshape(width // merge, merge)
    row_ids = numpy.broadcast_to(
        rows[:, None, :, None], (height // merge, width // merge, merge, merge)
    ).reshape(-1)
    column_ids = numpy.broadcast_to(
        columns[None, :, None, :], (height // merge, width // merge, merge, merge)
    ).reshape(-1)
    row_ids = numpy.tile(row_ids, frames)
    column_ids = numpy.tile(column_ids, frames)
    rotary_dim = source.hidden_size // source.attention_heads // 2
    inverse = 1.0 / (
        10000.0 ** (numpy.arange(0, rotary_dim, 2, dtype=numpy.float32) / rotary_dim)
    )
    frequencies = numpy.concatenate(
        [row_ids[:, None] * inverse[None, :], column_ids[:, None] * inverse[None, :]],
        axis=-1,
    )
    frequencies = numpy.tile(frequencies, (1, 2))
    return (
        numpy.cos(frequencies).astype(numpy.float16)[None, :, None, :],
        numpy.sin(frequencies).astype(numpy.float16)[None, :, None, :],
    )


def _reference(shape, source, nw, nb, qw, qb, pw, pb, cos, sin, numpy):
    vector = numpy.asarray(
        [((index % 31) - 15) / 16 for index in range(source.hidden_size)],
        dtype=numpy.float32,
    )
    hidden = numpy.tile(vector, (shape[0], 1))
    mean = hidden.mean(axis=-1, keepdims=True)
    variance = ((hidden - mean) ** 2).mean(axis=-1, keepdims=True)
    normalized = (hidden - mean) / numpy.sqrt(variance + numpy.float32(1e-6))
    normalized = normalized * nw.astype(numpy.float32) + nb.astype(numpy.float32)
    qkv = normalized @ qw.astype(numpy.float32).T + qb.astype(numpy.float32)
    qkv = qkv.reshape(shape[0], 3, source.attention_heads, -1).transpose(1, 2, 0, 3)
    query, key, value = qkv[0:1], qkv[1:2], qkv[2:3]
    cos32, sin32 = cos.astype(numpy.float32).transpose(0, 2, 1, 3), sin.astype(
        numpy.float32
    ).transpose(0, 2, 1, 3)
    half = query.shape[-1] // 2
    query = query * cos32 + numpy.concatenate(
        [-query[..., half:], query[..., :half]], axis=-1
    ) * sin32
    key = key * cos32 + numpy.concatenate(
        [-key[..., half:], key[..., :half]], axis=-1
    ) * sin32
    scores = numpy.matmul(query, key.swapaxes(-2, -1)) / numpy.sqrt(numpy.float32(64))
    scores -= scores.max(axis=-1, keepdims=True)
    probabilities = numpy.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    attended = numpy.matmul(probabilities, value).transpose(0, 2, 1, 3).reshape(shape)
    projected = attended @ pw.astype(numpy.float32).T + pb.astype(numpy.float32)
    return (hidden + projected).astype(numpy.float16)


def _weight(root: Path, records: dict, name: str, numpy):
    record = records.get(name)
    if not isinstance(record, dict):
        raise ValueError(f"Qwen3-VL staged weight is missing: {name}")
    return numpy.fromfile(root / record["file"], dtype="<f2").reshape(record["shape"])
