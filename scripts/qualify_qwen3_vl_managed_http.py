#!/usr/bin/env python3
"""Qualify managed persistent Qwen3-VL through the real HTTP chat endpoint."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import tempfile
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
    parser.add_argument("--report", type=Path)
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

    from vllm_apple.api import create_server
    from vllm_apple.backend_composition import (
        BackendEngineRegistration,
        BackendRegistryInferenceEngine,
        ProductionBackendComposition,
    )
    from vllm_apple.backend_engine import BackendEngineDescriptor
    from vllm_apple.execution import ExecutionBackend, WorkloadPhase
    from vllm_apple.process_inference_engine import MainThreadSubprocessInferenceEngine
    from vllm_apple.qualification import save_qualification_report
    from vllm_apple.service import RuntimeService

    model_id = "qwen3-vl-managed"
    private = tempfile.TemporaryDirectory(prefix="vllm-apple-qwen3-vl-process-")
    private_path = Path(private.name)
    config_path = private_path / "config.json"
    config_path.write_text(json.dumps({
        "model": str(arguments.model.resolve()),
        "revision": arguments.revision,
        "graph_id": arguments.graph_id,
        "compiled_models": [str(path.resolve()) for path in arguments.compiled_model],
        "model_id": model_id,
        "hardware_profile": "apple-m4",
        "environment_profile": "macos-27",
        "unified_memory_bytes": 4_000_000_000,
        "cpu_threads": 8,
        "gpu_command_queues": 2,
        "bandwidth_slots": 2,
    }))
    config_path.chmod(0o600)
    engine = None
    service = None
    server = None
    thread = None
    cases = []
    diagnostics = None
    shutdown_clean = False
    try:
        descriptor = BackendEngineDescriptor(
            ExecutionBackend.NATIVE_MLX,
            "qwen3-vl-coreml-mlx-1",
            ("qwen3_vl",),
            ("fp16",),
            (WorkloadPhase.PREFILL,),
            ("chat.completions",),
            "subprocess",
        )
        composition = ProductionBackendComposition((BackendEngineRegistration(
            descriptor,
            lambda: MainThreadSubprocessInferenceEngine(
                "vllm_apple.qwen3_vl_process_factory:create_qwen3_vl_process_delegate",
                python_executable=Path(sys.executable),
                config_path=config_path,
                maximum_pending_requests=2,
                startup_timeout_seconds=600,
            ),
        ),))
        engine = BackendRegistryInferenceEngine(
            composition,
            model_architecture="qwen3_vl",
            precision="fp16",
            candidates=(ExecutionBackend.NATIVE_MLX,),
        )
        service = RuntimeService(engine=engine)
        server = create_server(
            "127.0.0.1", 0, service, max_concurrent_requests=3, socket_timeout=120
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
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
        diagnostics = engine.diagnostics()
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if service is not None:
            shutdown_clean = service.close()
        elif engine is not None:
            shutdown_clean = engine.close()
        if thread is not None:
            thread.join(timeout=2)
        private.cleanup()
    assert diagnostics is not None and server is not None
    report = {
        "schema_version": 1,
        "scope": "qwen3_vl_managed_persistent_http",
        "request_count": len(cases),
        "cases": cases,
        "model_process_main_thread": diagnostics["main_thread"],
        "worker_restart_count": diagnostics["worker_restart_count"],
        "resources_after_completion": diagnostics["resources"]["used"],
        "server_request_metrics": server.request_metrics(),
        "shutdown_clean": shutdown_clean,
        "temporary_cleanup_verified": not private_path.exists(),
        "stores_prompt": False,
        "stores_output": False,
    }
    report["passed"] = (
        len(cases) == len(arguments.image) * 3
        and all(case["task_correct"] for case in cases)
        and report["model_process_main_thread"] is True
        and report["worker_restart_count"] == 0
        and report["shutdown_clean"]
        and report["temporary_cleanup_verified"]
        and all(value == 0 for value in report["resources_after_completion"].values())
        and report["server_request_metrics"]["active_requests"] == 0
    )
    if arguments.report is not None:
        save_qualification_report(report, arguments.report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
