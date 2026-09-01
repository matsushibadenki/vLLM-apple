from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .qwen4_conversion_plan import _COMPONENT_POLICIES, _component, build_qwen4_conversion_plan
from .qwen4_shard_stager import verify_qwen4_stage
from .qwen4_weight_map import _bounded_index


def build_qwen4_adapter_contract(
    stage_root: str | Path,
    *,
    maximum_artifact_bytes: int,
    requested_modes: tuple[str, ...] = ("text",),
) -> dict[str, object]:
    root = Path(stage_root).expanduser().resolve(strict=True)
    verified = verify_qwen4_stage(
        root,
        requested_modes=requested_modes,
        maximum_artifact_bytes=maximum_artifact_bytes,
    )
    plan = build_qwen4_conversion_plan(
        root,
        root / "model.safetensors.index.json",
        requested_modes=requested_modes,
    )
    weight_map, _, _ = _bounded_index(root / "model.safetensors.index.json")
    shard_components: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for tensor_name, shard_name in weight_map.items():
        component = _component(tensor_name)
        if component is None:
            raise ValueError("Qwen4 adapter contract contains an unclassified tensor")
        shard_components[shard_name][component] += 1
    enabled = set(plan["enabled_components"])
    schedule = [
        {
            "shard": shard_name,
            "active_entries": sum(
                count
                for component, count in shard_components[shard_name].items()
                if component in enabled
            ),
            "skipped_optional_entries": sum(
                count
                for component, count in shard_components[shard_name].items()
                if component not in enabled
            ),
            "active_components": {
                component: count
                for component, count in sorted(shard_components[shard_name].items())
                if component in enabled
            },
        }
        for shard_name in sorted(shard_components)
        if any(component in enabled for component in shard_components[shard_name])
    ]
    body = {
        "schema_version": 1,
        "architecture": "qwen4_exp",
        "stage_verified": True,
        "stage_plan_id": verified["plan_id"],
        "config_fingerprint": verified["config_fingerprint"],
        "index_sha256": verified["index_sha256"],
        "requested_modes": list(requested_modes),
        "artifact_bytes": verified["output_bytes"],
        "shard_count": verified["shard_count"],
        "scheduled_shard_count": len(schedule),
        "enabled_components": plan["enabled_components"],
        "disabled_optional_components": plan["disabled_optional_components"],
        "component_entries": plan["component_entries"],
        "component_policies": {
            name: _COMPONENT_POLICIES[name] for name in sorted(_COMPONENT_POLICIES)
        },
        "shard_schedule": schedule,
        "loads_tensor_data": False,
        "allocates_model_or_metal": False,
        "peak_open_source_shards": 1,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {**body, "contract_id": hashlib.sha256(canonical).hexdigest()}
