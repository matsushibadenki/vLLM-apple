#!/usr/bin/env python3
"""Qualify managed persistent Qwen3-VL through the real HTTP chat endpoint."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import threading
import urllib.request
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualify_qwen3_vl_end_to_end import _TASKS  # noqa: E402


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
        or any(
            label not in _TASKS[task]["labels"]
            for task, label in zip(arguments.task, arguments.label, strict=True)
        )
    ):
        raise ValueError("invalid managed HTTP qualification")

    from mlx_vlm.utils import load

    from vllm_apple.api import create_server
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
    from vllm_apple.managed_engine import ThreadAffineInferenceEngine
    from vllm_apple.qwen3_vl_ane import inspect_qwen3_vl_vision_for_ane
    from vllm_apple.qwen3_vl_coreml import Qwen3VLCoreMLConversionManifest
    from vllm_apple.qwen3_vl_embedding import Qwen3VLANEGPUPipeline
    from vllm_apple.qwen3_vl_managed_engine import (
        Qwen3VLPersistentChatDelegate,
        load_qwen3_vl_chat_runtime,
    )
    from vllm_apple.qwen3_vl_persistent_encoder import Qwen3VLPersistentEncoder
    from vllm_apple.qwen3_vl_persistent_worker import Qwen3VLPersistentWorker
    from vllm_apple.service import RuntimeService

    ledger_holder = []
    worker_holder = []
    model_id = "qwen3-vl-managed"

    def factory():
        model, processor = load(str(arguments.model), lazy=True)
        source = inspect_qwen3_vl_vision_for_ane(
            arguments.model, model_revision=arguments.revision
        )
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
        pipeline = AsyncEncoderLLMPipeline(ledger, registry)
        worker = Qwen3VLPersistentWorker(tuple(arguments.compiled_model))
        encoder = Qwen3VLPersistentEncoder(
            worker, source, route, graph_id=arguments.graph_id
        )
        bridge = Qwen3VLANEGPUPipeline(
            pipeline, source=source, conversion=conversion, route=route,
            coreml_graph_id=arguments.graph_id,
        )
        ledger_holder.append(ledger)
        worker_holder.append(worker)
        return Qwen3VLPersistentChatDelegate(
            model, processor, bridge, encoder, pipeline,
            load_qwen3_vl_chat_runtime(), model_id=model_id,
        )

    engine = ThreadAffineInferenceEngine(factory, maximum_pending_requests=2)
    service = RuntimeService(engine=engine)
    server = create_server(
        "127.0.0.1", 0, service, max_concurrent_requests=3, socket_timeout=120
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cases = []
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        for index, (image, task_name, label) in enumerate(zip(
            arguments.image, arguments.task, arguments.label, strict=True
        )):
            image_data = base64.b64encode(image.read_bytes()).decode()
            task = _TASKS[task_name]
            for language, prompt in task["prompts"]:
                body = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }},
                        {"type": "text", "text": prompt},
                    ]}],
                    "max_tokens": 12,
                }
                request = urllib.request.Request(
                    endpoint, data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.load(response)
                text = payload["choices"][0]["message"]["content"]
                normalized = text.casefold().strip()
                cases.append({
                    "request_index": index,
                    "task": task_name,
                    "label": label,
                    "language": language,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "task_correct": any(
                        expected.casefold() in normalized
                        for expected in task["labels"][label][language]
                    ),
                })
    finally:
        server.shutdown()
        server.server_close()
        service.close()
        thread.join(timeout=2)
    report = {
        "schema_version": 1,
        "scope": "qwen3_vl_managed_persistent_http",
        "request_count": len(cases),
        "cases": cases,
        "owner_thread_ident": engine.owner_thread_ident,
        "worker_restart_count": worker_holder[0].restart_count,
        "resources_after_completion": ledger_holder[0].snapshot()["used"],
        "server_request_metrics": server.request_metrics(),
    }
    report["passed"] = (
        len(cases) == len(arguments.image) * 3
        and all(case["task_correct"] for case in cases)
        and report["worker_restart_count"] == 0
        and all(value == 0 for value in report["resources_after_completion"].values())
        and report["server_request_metrics"]["active_requests"] == 0
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
