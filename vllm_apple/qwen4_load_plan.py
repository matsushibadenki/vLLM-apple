from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .model import inspect_model_metadata
from .qwen4_component_loader import _TARGET_DTYPE_BYTES
from .qwen4_conversion_plan import _COMPONENT_POLICIES
from .qwen4_tensor_reader import MAX_TENSOR_CHUNK_BYTES, build_qwen4_tensor_catalog


def _destination_bytes(descriptor: dict[str, object], target_dtype: str) -> int:
    shape = descriptor.get("shape")
    if target_dtype not in _TARGET_DTYPE_BYTES or not isinstance(shape, list):
        raise ValueError("Qwen4 load plan dtype or tensor shape is invalid")
    elements = 1
    for dimension in shape:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0:
            raise ValueError("Qwen4 load plan tensor shape is invalid")
        elements *= dimension
    return elements * _TARGET_DTYPE_BYTES[target_dtype]


def build_qwen4_component_load_plan(
    stage_root: str | Path,
    *,
    maximum_artifact_bytes: int,
    requested_modes: tuple[str, ...] = ("text",),
    target_dtype: str = "BF16",
    scratch_bytes_per_tensor: int = 0,
) -> dict[str, object]:
    if (
        not isinstance(scratch_bytes_per_tensor, int)
        or isinstance(scratch_bytes_per_tensor, bool)
        or scratch_bytes_per_tensor < 0
    ):
        raise ValueError("Qwen4 load plan scratch bytes are invalid")
    root = Path(stage_root).expanduser().resolve(strict=True)
    config, _ = inspect_model_metadata(root)
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("Qwen4 load plan text config is invalid")
    num_experts = text.get("num_experts", 0)
    active_experts = text.get("num_experts_per_tok", 0)
    if (
        not isinstance(num_experts, int)
        or isinstance(num_experts, bool)
        or not isinstance(active_experts, int)
        or isinstance(active_experts, bool)
        or num_experts < 0
        or active_experts < 0
        or active_experts > num_experts
    ):
        raise ValueError("Qwen4 load plan expert topology is invalid")
    catalog = build_qwen4_tensor_catalog(
        root,
        maximum_artifact_bytes=maximum_artifact_bytes,
        requested_modes=requested_modes,
    )
    storage: defaultdict[str, int] = defaultdict(int)
    resident: defaultdict[str, int] = defaultdict(int)
    ple_partitions: list[int] = []
    peak_tensor_reservation = 0
    requires_expert_slicing = False
    for tensor_name, descriptor in catalog["tensors"].items():
        if descriptor["active"] is not True:
            continue
        component = descriptor["component"]
        if not isinstance(component, str):
            raise ValueError("Qwen4 load plan component is invalid")
        destination_bytes = _destination_bytes(descriptor, target_dtype)
        storage[component] += destination_bytes
        source_bytes = descriptor["bytes"]
        if not isinstance(source_bytes, int):
            raise ValueError("Qwen4 load plan source bytes are invalid")
        policy = _COMPONENT_POLICIES[component]
        load_destination_bytes = destination_bytes
        load_source_bytes = source_bytes
        if policy == "on_demand_expert" and ".experts." in tensor_name:
            shape = descriptor.get("shape")
            if (
                num_experts <= 0
                or active_experts <= 0
                or not isinstance(shape, list)
                or not shape
                or shape[0] != num_experts
                or destination_bytes % num_experts != 0
                or source_bytes % num_experts != 0
            ):
                raise ValueError("Qwen4 packed expert axis does not match the expert topology")
            load_destination_bytes = destination_bytes // num_experts * active_experts
            load_source_bytes = source_bytes // num_experts * active_experts
        peak_tensor_reservation = max(
            peak_tensor_reservation,
            load_destination_bytes
            + min(load_source_bytes, MAX_TENSOR_CHUNK_BYTES)
            + scratch_bytes_per_tensor,
        )
        if policy == "resident" or policy == "optional_mode":
            resident[component] += destination_bytes
        elif policy == "on_demand_expert":
            if ".experts." in tensor_name:
                resident[component] += load_destination_bytes
                requires_expert_slicing = True
            else:
                resident[component] += destination_bytes
        elif policy == "partitioned_lookup":
            if ".ngram_embedding.shard_" in tensor_name:
                ple_partitions.append(destination_bytes)
            else:
                resident[component] += destination_bytes
        else:
            raise ValueError("Qwen4 load plan component policy is unsupported")
    if ple_partitions:
        resident["per_layer_embedding"] += max(ple_partitions)
    body = {
        "schema_version": 1,
        "contract_id": catalog["contract_id"],
        "requested_modes": list(requested_modes),
        "target_dtype": target_dtype,
        "scratch_bytes_per_tensor": scratch_bytes_per_tensor,
        "component_storage_bytes": dict(sorted(storage.items())),
        "component_resident_bytes": dict(sorted(resident.items())),
        "total_storage_bytes": sum(storage.values()),
        "resident_working_set_bytes": sum(resident.values()),
        "peak_tensor_reservation_bytes": peak_tensor_reservation,
        "expert_count": num_experts,
        "active_experts_per_token": active_experts,
        "ple_partition_count": len(ple_partitions),
        "requires_expert_axis0_slicing": requires_expert_slicing,
        "reader_supports_expert_axis0_slicing": True,
        "allocates_model_or_metal": False,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {**body, "load_plan_id": hashlib.sha256(canonical).hexdigest()}
