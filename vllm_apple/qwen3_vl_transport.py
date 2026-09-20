"""Private FP16 transport from the Core ML worker to Homebrew vLLM-Metal."""
from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

from .device_pipeline import ANEAuxiliaryRoute
from .qwen3_vl_ane import Qwen3VLVisionANEAdapterSpec
from .qwen3_vl_embedding import (
    Qwen3VLCoreMLPipelineOutput,
    validate_qwen3_vl_vision_embeddings,
)


def load_qwen3_vl_coreml_transport(
    transport_root: Path,
    source: Qwen3VLVisionANEAdapterSpec,
    route: ANEAuxiliaryRoute,
    *,
    expected_graph_id: str,
) -> Qwen3VLCoreMLPipelineOutput:
    """Verify a bounded transport and materialize its outputs as MLX FP16 arrays."""
    if transport_root.is_symlink():
        raise ValueError("Qwen3-VL transport root must not be a symlink")
    root = transport_root.expanduser().resolve(strict=True)
    if not root.is_dir() or stat.S_IMODE(root.stat().st_mode) & 0o077:
        raise ValueError("Qwen3-VL transport directory is not private")
    expected_files = {
        "manifest.json",
        "final.fp16",
        "deepstack_0.fp16",
        "deepstack_1.fp16",
        "deepstack_2.fp16",
    }
    if {path.name for path in root.iterdir()} != expected_files:
        raise ValueError("Qwen3-VL transport file set changed")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 16_384:
        raise ValueError("Qwen3-VL transport manifest is invalid")
    manifest = json.loads(manifest_path.read_bytes())
    if (
        set(manifest) != {"schema_version", "graph_id", "grid_thw", "records"}
        or manifest["schema_version"] != 1
        or manifest["graph_id"] != expected_graph_id
        or manifest["grid_thw"] != [1, 16, 16]
        or not isinstance(manifest["records"], list)
        or len(manifest["records"]) != 4
    ):
        raise ValueError("Qwen3-VL transport provenance is invalid")
    records = {record.get("name"): record for record in manifest["records"]}
    if set(records) != {"final", "deepstack_0", "deepstack_1", "deepstack_2"}:
        raise ValueError("Qwen3-VL transport output order is invalid")
    try:
        import mlx.core as mx
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Qwen3-VL transport requires the Homebrew MLX runtime") from error

    arrays = {}
    for name in ("final", "deepstack_0", "deepstack_1", "deepstack_2"):
        record = records[name]
        expected_name = f"{name}.fp16"
        if (
            set(record) != {"name", "file", "shape", "dtype", "bytes", "sha256"}
            or record["file"] != expected_name
            or record["shape"] != [64, 2048]
            or record["dtype"] != "float16"
            or record["bytes"] != 262_144
        ):
            raise ValueError("Qwen3-VL transport record is invalid")
        path = root / expected_name
        if (
            path.is_symlink()
            or not path.is_file()
            or stat.S_IMODE(path.stat().st_mode) & 0o077
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError("Qwen3-VL transport payload is not private")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError("Qwen3-VL transport payload digest changed")
        arrays[name] = mx.array(np.frombuffer(payload, dtype="<f2").reshape(64, 2048))
    output = Qwen3VLCoreMLPipelineOutput(
        arrays["final"],
        (arrays["deepstack_0"], arrays["deepstack_1"], arrays["deepstack_2"]),
        (1, 16, 16),
        expected_graph_id,
    )
    validate_qwen3_vl_vision_embeddings(
        source,
        route,
        grid_thw=output.grid_thw,
        hidden_states=output.hidden_states,
        deepstack_visual_embeds=output.deepstack_visual_embeds,
    )
    return output
