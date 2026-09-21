"""Production Qwen3-VL delegate factory for main-thread process ownership."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .device_capability import ComputeDevice, DeviceCapability, DeviceCapabilityRegistry
from .device_pipeline import (
    ANEAuxiliaryWorkload,
    AsyncEncoderLLMPipeline,
    require_ane_auxiliary_route,
)
from .device_resources import UnifiedDeviceResourceLedger
from .execution import ExecutionBackend, WorkloadPhase
from .qwen3_vl_ane import inspect_qwen3_vl_vision_for_ane
from .qwen3_vl_coreml import Qwen3VLCoreMLConversionManifest
from .qwen3_vl_embedding import Qwen3VLANEGPUPipeline
from .qwen3_vl_managed_engine import (
    Qwen3VLPersistentChatDelegate,
    load_qwen3_vl_chat_runtime,
)
from .qwen3_vl_persistent_encoder import Qwen3VLPersistentEncoder
from .qwen3_vl_persistent_worker import Qwen3VLPersistentWorker


_CONFIG_KEYS = {
    "model", "revision", "graph_id", "compiled_models", "model_id",
    "hardware_profile", "environment_profile", "unified_memory_bytes",
    "cpu_threads", "gpu_command_queues", "bandwidth_slots",
}


class _ProcessDelegate:
    def __init__(
        self,
        delegate: Qwen3VLPersistentChatDelegate,
        worker: Qwen3VLPersistentWorker,
        ledger: UnifiedDeviceResourceLedger,
    ) -> None:
        self._delegate = delegate
        self._worker = worker
        self._ledger = ledger

    @property
    def ready(self) -> bool:
        return self._delegate.ready

    def models(self) -> list[dict[str, Any]]:
        return self._delegate.models()

    def chat_completions_with_request_context(self, *args: object, **kwargs: object):
        return self._delegate.chat_completions_with_request_context(*args, **kwargs)

    def diagnostics(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "main_thread": threading.current_thread() is threading.main_thread(),
            "worker_restart_count": self._worker.restart_count,
            "resources": self._ledger.snapshot(),
        }

    def close(self) -> bool:
        return self._delegate.close()


def create_qwen3_vl_process_delegate(config: dict[str, object]) -> object:
    """Build the qualified persistent Core ML -> MLX path on process main."""
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise ValueError("invalid Qwen3-VL process configuration schema")
    strings = (
        "model", "revision", "graph_id", "model_id",
        "hardware_profile", "environment_profile",
    )
    if any(not isinstance(config[key], str) or not config[key] for key in strings):
        raise ValueError("invalid Qwen3-VL process identity")
    graph_id = config["graph_id"]
    if (len(graph_id) != 64
            or any(character not in "0123456789abcdef" for character in graph_id)):
        raise ValueError("invalid Qwen3-VL graph identity")
    compiled = config["compiled_models"]
    if (not isinstance(compiled, list) or len(compiled) != 5
            or any(not isinstance(value, str) or not value for value in compiled)):
        raise ValueError("Qwen3-VL requires five compiled models")
    integer_keys = (
        "unified_memory_bytes", "cpu_threads", "gpu_command_queues",
        "bandwidth_slots",
    )
    if any(type(config[key]) is not int or config[key] <= 0 for key in integer_keys):
        raise ValueError("invalid Qwen3-VL process resource capacity")

    from mlx_vlm.utils import load

    model_path = Path(config["model"]).expanduser().resolve(strict=True)
    compiled_paths = tuple(
        Path(value).expanduser().resolve(strict=True) for value in compiled
    )
    model, processor = load(str(model_path), lazy=True)
    source = inspect_qwen3_vl_vision_for_ane(
        model_path, model_revision=config["revision"]
    )
    conversion = Qwen3VLCoreMLConversionManifest(
        source.artifact_fingerprint,
        source.model_revision,
        graph_id,
        "pixel_values",
        "final_hidden_states",
        (256, 1536),
        (64, 2048),
        "fp16",
        "coremltools-8.1",
    )
    registry = DeviceCapabilityRegistry(
        config["hardware_profile"], config["environment_profile"]
    )
    registry.record(DeviceCapability(
        ExecutionBackend.COREML_DRAFT,
        ComputeDevice.ANE,
        "coremltools-8.1",
        config["hardware_profile"],
        config["environment_profile"],
        (source.operator,),
        (WorkloadPhase.AUXILIARY,),
        ("fp16",),
        "available",
        "probe_passed",
        (graph_id[:24],),
    ))
    route = require_ane_auxiliary_route(
        registry,
        workload=ANEAuxiliaryWorkload.VISION_ENCODER,
        operator=source.operator,
        precision="fp16",
    )
    ledger = UnifiedDeviceResourceLedger(
        unified_memory_bytes=config["unified_memory_bytes"],
        cpu_threads=config["cpu_threads"],
        gpu_command_queues=config["gpu_command_queues"],
        ane_tasks=1,
        bandwidth_slots=config["bandwidth_slots"],
    )
    pipeline = AsyncEncoderLLMPipeline(ledger, registry)
    worker = Qwen3VLPersistentWorker(compiled_paths)
    encoder = Qwen3VLPersistentEncoder(worker, source, route, graph_id=graph_id)
    bridge = Qwen3VLANEGPUPipeline(
        pipeline,
        source=source,
        conversion=conversion,
        route=route,
        coreml_graph_id=graph_id,
    )
    delegate = Qwen3VLPersistentChatDelegate(
        model,
        processor,
        bridge,
        encoder,
        pipeline,
        load_qwen3_vl_chat_runtime(),
        model_id=config["model_id"],
    )
    return _ProcessDelegate(delegate, worker, ledger)
