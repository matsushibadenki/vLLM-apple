"""Qwen3-VL six-block Core ML graph with the first deep-stack merger."""
from __future__ import annotations

import hashlib
import json
import math
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
from .qwen3_vl_tower_coreml import _block_mil, _layer_weights


MAIN_MAXIMUM_SCALED_ERROR = 3e-2
DEEPSTACK_MAXIMUM_SCALED_ERROR = 4e-2


def build_qwen3_vl_deepstack_coreml(
    staged_weights: Path,
    destination: Path,
    source: Qwen3VLVisionANEAdapterSpec,
    plan: Qwen3VLCoreMLConversionPlan,
    *,
    deepstack_position: int = 0,
    start_layer: int = 0,
    profile_index: int = 0,
    final_merger: bool = False,
) -> dict[str, object]:
    output = destination.expanduser().resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Qwen3-VL deep-stack Core ML destination must be new")
    graph = build_qwen3_vl_coreml_graph_spec(staged_weights, source, plan)
    if (
        type(final_merger) is not bool
        or (not final_merger and type(deepstack_position) is not int)
        or (
            not final_merger
            and not 0 <= deepstack_position < len(source.deepstack_visual_indexes)
        )
        or type(start_layer) is not int
        or type(profile_index) is not int
        or not 0 <= profile_index < len(graph.profiles)
    ):
        raise ValueError("Qwen3-VL deep-stack Core ML request is invalid")
    layer_index = (
        source.depth - 1
        if final_merger
        else source.deepstack_visual_indexes[deepstack_position]
    )
    if not 0 <= start_layer <= layer_index:
        raise ValueError("Qwen3-VL deep-stack Core ML request is invalid")
    block_count = layer_index + 1
    try:
        import coremltools as ct
        import numpy as np
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ImportError as error:
        raise RuntimeError("Qwen3-VL deep-stack build requires coremltools and NumPy") from error

    root = staged_weights.expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_bytes())
    records = {record["name"]: record for record in manifest["records"]}
    all_layers = tuple(
        _layer_weights(root, records, layer, np) for layer in range(block_count)
    )
    layers = all_layers[start_layer:]
    merger = _merger_weights(
        root, records, None if final_merger else deepstack_position, np
    )
    profile = graph.profiles[profile_index]
    tokens = profile.pixel_values_shape[0]
    main_shape = (tokens, source.hidden_size)
    merged_tokens = tokens // source.spatial_merge_size**2
    deepstack_shape = (merged_tokens, source.output_hidden_size)
    cos, sin = _fixed_rope(profile.grid_thw, source, np)
    vector = np.asarray(
        [((index % 31) - 15) / 16 for index in range(source.hidden_size)],
        dtype=np.float32,
    )
    main_reference = np.tile(vector, (tokens, 1)).astype(np.float16)
    for weights in all_layers[:start_layer]:
        main_reference = _reference_from_hidden(
            main_reference,
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
        main_reference = _mlp_reference(main_reference, weights, np)
    segment_input = main_reference
    for weights in layers:
        main_reference = _reference_from_hidden(
            main_reference,
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
        main_reference = _mlp_reference(main_reference, weights, np)
    deepstack_reference = _merger_reference(
        main_reference, merger, np, postshuffle_norm=not final_merger
    )
    merger_kind = "final" if final_merger else f"deepstack-{deepstack_position}"
    merged_output_name = (
        "final_hidden_states" if final_merger else "deepstack_hidden_states"
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    main_reference_name = "main_reference.fp16"
    deepstack_reference_name = "deepstack_reference.fp16"
    segment_input_name = "segment_input.fp16"
    (temporary / main_reference_name).write_bytes(main_reference.tobytes(order="C"))
    (temporary / deepstack_reference_name).write_bytes(
        deepstack_reference.tobytes(order="C")
    )
    if start_layer:
        (temporary / segment_input_name).write_bytes(segment_input.tobytes(order="C"))
    package = temporary / "qwen3_vl_deepstack.mlpackage"
    compiled_root = temporary / "compiled"
    compiled_root.mkdir(mode=0o700)
    try:
        @mb.program(
            input_specs=[mb.TensorSpec(shape=main_shape, dtype=types.fp16)],
            opset_version=ct.target.macOS15,
        )
        def program(hidden_states):
            value = hidden_states
            for layer, weights in enumerate(layers, start=start_layer):
                value = _block_mil(
                    value, weights, layer, tokens, source, graph.head_dimension,
                    cos, sin, mb, np,
                )
            norm_input = (
                value
                if final_merger
                else mb.reshape(
                    x=value,
                    shape=[
                        merged_tokens,
                        source.hidden_size * source.spatial_merge_size**2,
                    ],
                )
            )
            normalized = mb.layer_norm(
                x=norm_input,
                axes=[-1],
                gamma=merger["norm.weight"],
                beta=merger["norm.bias"],
                epsilon=np.float16(1e-6),
                name=f"{merger_kind}_norm",
            )
            merger_input = (
                mb.reshape(
                    x=normalized,
                    shape=[
                        merged_tokens,
                        source.hidden_size * source.spatial_merge_size**2,
                    ],
                )
                if final_merger
                else normalized
            )
            expanded = mb.linear(
                x=merger_input,
                weight=merger["linear_fc1.weight"],
                bias=merger["linear_fc1.bias"],
                name=f"{merger_kind}_fc1",
            )
            activated = mb.gelu(
                x=expanded, mode="EXACT", name=f"{merger_kind}_gelu"
            )
            merged = mb.linear(
                x=activated,
                weight=merger["linear_fc2.weight"],
                bias=merger["linear_fc2.bias"],
                name=merged_output_name,
            )
            return mb.identity(x=value, name="tower_hidden_states"), merged

        model = ct.convert(
            program,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
        )
        model.user_defined_metadata["vllm-apple.graph-id"] = graph.graph_id
        model.user_defined_metadata["vllm-apple.partition"] = (
            f"tower-blocks-{start_layer}-{layer_index}-{merger_kind}-v1"
        )
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
            timeout=900,
            check=False,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Core ML compiler rejected Qwen3-VL deep-stack graph: "
                + completed.stderr[-2048:]
            )
        compiled = tuple(compiled_root.glob("*.mlmodelc"))
        if len(compiled) != 1:
            raise RuntimeError("Qwen3-VL deep-stack compiler output is invalid")
        report = {
            "schema_version": 1,
            "graph_id": graph.graph_id,
            "partition": f"tower-blocks-{start_layer}-{layer_index}-{merger_kind}-v1",
            "block_count": len(layers),
            "start_layer": start_layer,
            "deepstack_position": deepstack_position,
            "deepstack_layer": layer_index,
            "merger_kind": merger_kind,
            "merged_output_name": merged_output_name,
            "grid_thw": list(profile.grid_thw),
            "input_shape": list(main_shape),
            "main_output_shape": list(main_shape),
            "deepstack_output_shape": list(deepstack_shape),
            "main_maximum_scaled_error": MAIN_MAXIMUM_SCALED_ERROR,
            "deepstack_maximum_scaled_error": DEEPSTACK_MAXIMUM_SCALED_ERROR,
            "main_reference_sha256": hashlib.sha256(main_reference.tobytes()).hexdigest(),
            "deepstack_reference_sha256": hashlib.sha256(
                deepstack_reference.tobytes()
            ).hexdigest(),
            "main_reference_file": main_reference_name,
            "deepstack_reference_file": deepstack_reference_name,
            "compiled_model": f"compiled/{compiled[0].name}",
        }
        if start_layer:
            report["input_file"] = segment_input_name
            report["input_sha256"] = hashlib.sha256(segment_input.tobytes()).hexdigest()
        (temporary / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def qualify_qwen3_vl_deepstack_coreml(package_root: Path) -> dict[str, object]:
    root = package_root.expanduser().resolve(strict=True)
    report = json.loads((root / "report.json").read_bytes())
    model = (root / report["compiled_model"]).resolve(strict=True)
    main_reference = root / report["main_reference_file"]
    deepstack_reference = root / report["deepstack_reference_file"]
    input_file = root / report["input_file"] if "input_file" in report else None
    if (
        report.get("main_maximum_scaled_error") != MAIN_MAXIMUM_SCALED_ERROR
        or report.get("deepstack_maximum_scaled_error") != DEEPSTACK_MAXIMUM_SCALED_ERROR
        or hashlib.sha256(main_reference.read_bytes()).hexdigest()
        != report.get("main_reference_sha256")
        or hashlib.sha256(deepstack_reference.read_bytes()).hexdigest()
        != report.get("deepstack_reference_sha256")
        or (
            input_file is not None
            and hashlib.sha256(input_file.read_bytes()).hexdigest()
            != report.get("input_sha256")
        )
        or not model.is_dir()
        or root not in model.parents
    ):
        raise ValueError("Qwen3-VL deep-stack Core ML artifact is invalid")
    rows, columns = report["input_shape"]
    main = _predict(
        model, rows, columns, main_reference, "tower_hidden_states", input_file
    )
    deepstack = _predict(
        model,
        rows,
        columns,
        deepstack_reference,
        report.get("merged_output_name", "deepstack_hidden_states"),
        input_file,
    )
    if (
        main.get("output_count") != rows * columns
        or type(main.get("latency_nanoseconds")) is not int
        or main["latency_nanoseconds"] <= 0
        or main.get("maximum_scaled_error", float("inf")) > MAIN_MAXIMUM_SCALED_ERROR
        or deepstack.get("output_count")
        != report["deepstack_output_shape"][0] * report["deepstack_output_shape"][1]
        or type(deepstack.get("latency_nanoseconds")) is not int
        or deepstack["latency_nanoseconds"] <= 0
        or deepstack.get("maximum_scaled_error", float("inf"))
        > DEEPSTACK_MAXIMUM_SCALED_ERROR
    ):
        raise RuntimeError("Qwen3-VL deep-stack output does not match reference")
    return {
        "passed": True,
        "partition": report["partition"],
        "main": main,
        "deepstack": deepstack,
    }


def build_qwen3_vl_final_coreml(
    staged_weights: Path,
    destination: Path,
    source: Qwen3VLVisionANEAdapterSpec,
    plan: Qwen3VLCoreMLConversionPlan,
    *,
    start_layer: int = 18,
    profile_index: int = 0,
) -> dict[str, object]:
    return build_qwen3_vl_deepstack_coreml(
        staged_weights,
        destination,
        source,
        plan,
        start_layer=start_layer,
        profile_index=profile_index,
        final_merger=True,
    )


def qualify_qwen3_vl_final_coreml(package_root: Path) -> dict[str, object]:
    result = qualify_qwen3_vl_deepstack_coreml(package_root)
    merged = result.pop("deepstack")
    return {
        **result,
        "final": merged,
    }


def _predict(model, rows, columns, reference, output_name, input_file=None):
    arguments = [
        "/usr/bin/swift",
        "-e",
        _PREDICTION_PROGRAM,
        str(model),
        str(rows),
        str(columns),
        str(reference),
        output_name,
    ]
    if input_file is not None:
        arguments.append(str(input_file))
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
    if completed.returncode != 0 or len(completed.stdout) > 4096:
        raise RuntimeError("Qwen3-VL deep-stack prediction failed: " + completed.stderr[-2048:])
    return json.loads(completed.stdout)


def _merger_reference(hidden, weights, numpy, *, postshuffle_norm=True):
    value = hidden.astype(numpy.float32)
    if postshuffle_norm:
        value = value.reshape(-1, weights["norm.weight"].shape[0])
    mean = value.mean(axis=-1, keepdims=True)
    variance = ((value - mean) ** 2).mean(axis=-1, keepdims=True)
    normalized = (value - mean) / numpy.sqrt(variance + numpy.float32(1e-6))
    normalized = (
        normalized * weights["norm.weight"].astype(numpy.float32)
        + weights["norm.bias"].astype(numpy.float32)
    )
    if not postshuffle_norm:
        normalized = normalized.reshape(-1, weights["linear_fc1.weight"].shape[1])
    expanded = (
        normalized @ weights["linear_fc1.weight"].astype(numpy.float32).T
        + weights["linear_fc1.bias"].astype(numpy.float32)
    )
    scaled = expanded / numpy.sqrt(numpy.float32(2.0))
    error_function = numpy.fromiter(
        (math.erf(float(value)) for value in scaled.flat),
        dtype=numpy.float32,
        count=scaled.size,
    ).reshape(scaled.shape)
    activated = numpy.float32(0.5) * expanded * (
        numpy.float32(1.0) + error_function
    )
    output = (
        activated @ weights["linear_fc2.weight"].astype(numpy.float32).T
        + weights["linear_fc2.bias"].astype(numpy.float32)
    )
    return output.astype(numpy.float16)


def _merger_weights(root, records, position, numpy):
    prefix = (
        "vision_tower.merger"
        if position is None
        else f"vision_tower.deepstack_merger_list.{position}"
    )
    return {
        name: _weight(root, records, f"{prefix}.{name}", numpy)
        for name in (
            "norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias",
            "linear_fc2.weight", "linear_fc2.bias",
        )
    }


def _weight(root, records, name, numpy):
    record = records.get(name)
    if not isinstance(record, dict):
        raise ValueError(f"Qwen3-VL staged weight is missing: {name}")
    return numpy.fromfile(root / record["file"], dtype="<f2").reshape(record["shape"])
