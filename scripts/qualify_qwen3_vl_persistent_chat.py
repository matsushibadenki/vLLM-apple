#!/usr/bin/env python3
"""Qualify persistent Core ML -> scheduled MLX Qwen3-VL language generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualify_qwen3_vl_end_to_end import _FixedVisionTower, _TASKS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--compiled-model", type=Path, action="append", required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--task", choices=tuple(_TASKS), action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    arguments = parser.parse_args()
    if (
        len(arguments.compiled_model) != 5
        or not 1 <= len(arguments.image) <= 8
        or len(arguments.image) != len(arguments.task)
        or len(arguments.image) != len(arguments.label)
        or len(arguments.graph_id) != 64
        or any(label not in _TASKS[task]["labels"] for task, label in zip(arguments.task, arguments.label, strict=True))
    ):
        raise ValueError("invalid persistent chat qualification")

    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from mlx_vlm import apply_chat_template, generate
    from mlx_vlm.utils import load

    from vllm_apple.device_capability import (
        ComputeDevice,
        DeviceCapability,
        DeviceCapabilityRegistry,
    )
    from vllm_apple.device_pipeline import (
        ANEAuxiliaryWorkload,
        AsyncEncoderLLMPipeline,
        require_ane_auxiliary_route,
    )
    from vllm_apple.device_resources import UnifiedDeviceResourceLedger
    from vllm_apple.execution import ExecutionBackend, WorkloadPhase
    from vllm_apple.qwen3_vl_ane import inspect_qwen3_vl_vision_for_ane
    from vllm_apple.qwen3_vl_coreml import Qwen3VLCoreMLConversionManifest
    from vllm_apple.qwen3_vl_embedding import Qwen3VLANEGPUPipeline
    from vllm_apple.qwen3_vl_persistent_encoder import Qwen3VLPersistentEncoder
    from vllm_apple.qwen3_vl_persistent_worker import Qwen3VLPersistentWorker

    model, processor = load(str(arguments.model), lazy=True)
    original = model.vision_tower
    source = inspect_qwen3_vl_vision_for_ane(arguments.model, model_revision=arguments.revision)
    conversion = Qwen3VLCoreMLConversionManifest(
        source.artifact_fingerprint, source.model_revision, arguments.graph_id,
        "pixel_values", "final_hidden_states", (256, 1536), (64, 2048),
        "fp16", "coremltools-8.1",
    )
    registry = DeviceCapabilityRegistry("apple-m4", "macos-27")
    registry.record(DeviceCapability(
        ExecutionBackend.COREML_DRAFT, ComputeDevice.ANE, "coremltools-8.1",
        "apple-m4", "macos-27", (source.operator,), (WorkloadPhase.AUXILIARY,),
        ("fp16",), "available", "probe_passed", (arguments.graph_id[:24],),
    ))
    route = require_ane_auxiliary_route(
        registry, workload=ANEAuxiliaryWorkload.VISION_ENCODER,
        operator=source.operator, precision="fp16",
    )
    ledger = UnifiedDeviceResourceLedger(
        unified_memory_bytes=4_000_000_000, cpu_threads=8,
        gpu_command_queues=2, ane_tasks=1, bandwidth_slots=2,
    )
    worker = Qwen3VLPersistentWorker(tuple(arguments.compiled_model))
    encoder = Qwen3VLPersistentEncoder(worker, source, route, graph_id=arguments.graph_id)
    pipeline = AsyncEncoderLLMPipeline(ledger, registry)
    bridge = Qwen3VLANEGPUPipeline(
        pipeline, source=source, conversion=conversion, route=route,
        coreml_graph_id=arguments.graph_id,
    )
    workspace = Path(tempfile.mkdtemp(prefix="qwen3-vl-persistent-chat-"))
    workspace.chmod(0o700)
    reports = []
    shutdown_clean = False
    try:
        for index, (image_arg, task_name, label) in enumerate(
            zip(arguments.image, arguments.task, arguments.label, strict=True)
        ):
            image_path = image_arg.expanduser().resolve(strict=True)
            with Image.open(image_path) as opened:
                processed = processor.image_processor(images=[opened.convert("RGB")])
            pixels = np.asarray(processed["pixel_values"]).astype(np.float16).reshape(256, 1536)
            pixel_path = workspace / f"pixels-{index}.fp16"
            pixels.tofile(pixel_path)
            pixel_path.chmod(0o600)
            transport = workspace / f"transport-{index}"
            transport.mkdir(mode=0o700)
            mlx_pixels = mx.array(pixels).astype(original.patch_embed.proj.weight.dtype)
            baseline = original(mlx_pixels, mx.array([[1, 16, 16]], dtype=mx.int64))
            mx.eval(baseline[0], *baseline[1])
            task = _TASKS[task_name]
            baseline_cases = []
            for language, prompt_text in task["prompts"]:
                prompt = apply_chat_template(processor, model.config, prompt_text, num_images=1)
                model.vision_tower = _FixedVisionTower(original, baseline)
                response = generate(
                    model, processor, prompt, image=[str(image_path)],
                    max_tokens=12, temperature=0, verbose=False,
                )
                normalized = response.text.casefold().strip()
                baseline_cases.append({
                    "language": language,
                    "sha256": hashlib.sha256(response.text.encode()).hexdigest(),
                    "task_correct": any(
                        expected.casefold() in normalized
                        for expected in task["labels"][label][language]
                    ),
                })
            model.vision_tower = original

            def consume(bundle):
                candidate = (bundle.hidden_states, list(bundle.deepstack_visual_embeds))
                outputs = []
                for language, prompt_text in task["prompts"]:
                    prompt = apply_chat_template(processor, model.config, prompt_text, num_images=1)
                    model.vision_tower = _FixedVisionTower(original, candidate)
                    response = generate(
                        model, processor, prompt, image=[str(image_path)],
                        max_tokens=12, temperature=0, verbose=False,
                    )
                    normalized = response.text.casefold().strip()
                    outputs.append({
                        "language": language,
                        "sha256": hashlib.sha256(response.text.encode()).hexdigest(),
                        "task_correct": any(
                            expected.casefold() in normalized
                            for expected in task["labels"][label][language]
                        ),
                    })
                return outputs

            result = bridge.execute_inline(
                grid_thw=(1, 16, 16), encoder_memory_bytes=216_449_024,
                llm_backend=ExecutionBackend.NATIVE_MLX,
                llm_memory_bytes=1_000_000_000,
                encode=lambda p=pixel_path, t=transport: encoder.encode(p, t),
                consume=consume,
            )
            cases = []
            for baseline_case, candidate_case in zip(
                baseline_cases, result.output, strict=True
            ):
                cases.append({
                    "language": candidate_case["language"],
                    "baseline_sha256": baseline_case["sha256"],
                    "candidate_sha256": candidate_case["sha256"],
                    "baseline_task_correct": baseline_case["task_correct"],
                    "candidate_task_correct": candidate_case["task_correct"],
                    "exact_match": baseline_case["sha256"] == candidate_case["sha256"],
                })
            reports.append({
                "index": index, "task": task_name, "label": label,
                "encoder_backend": result.encoder_backend.value,
                "llm_backend": result.llm_backend.value,
                "cases": cases,
            })
            model.vision_tower = original
        pipeline.close()
        shutdown_clean = encoder.close()
    finally:
        model.vision_tower = original
        pipeline.close()
        if worker.running:
            shutdown_clean = encoder.close()
        shutil.rmtree(workspace, ignore_errors=True)
    report = {
        "schema_version": 1,
        "scope": "qwen3_vl_persistent_scheduled_chat_quality",
        "request_count": len(reports),
        "cases": reports,
        "restart_count": worker.restart_count,
        "resources_after_completion": ledger.snapshot()["used"],
        "shutdown_clean": shutdown_clean,
        "temporary_cleanup_verified": not workspace.exists(),
    }
    report["passed"] = (
        len(reports) == len(arguments.image)
        and all(
            case["baseline_task_correct"] and case["candidate_task_correct"]
            for request in reports for case in request["cases"]
        )
        and all(case["exact_match"] for request in reports for case in request["cases"])
        and all(value == 0 for value in report["resources_after_completion"].values())
        and worker.restart_count == 0 and shutdown_clean
        and report["temporary_cleanup_verified"]
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
