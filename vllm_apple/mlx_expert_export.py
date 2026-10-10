"""Offline SwitchGLU export; never installs hooks or retains a source bank at runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .expert_manifest import ExpertManifest, _sha


def export_switch_experts(
    layers: dict, destination: Path, *, model_sha256: str,
    maximum_expert_bytes: int, maximum_total_bytes: int,
) -> Path:
    """Export loaded MLX-LM SwitchGLU layers into a new directory.

    Caller supplies a stable digest of its exact source model artifact. Partial
    exports lack a manifest and are unusable via from_manifest; no files are
    overwritten or automatically deleted after failure. Limits exclude source
    weights and MLX allocator overhead. This is an offline conversion operation.
    """
    if (not _sha(model_sha256) or not isinstance(layers, dict) or not layers
            or any(type(layer) is not int or not 0 <= layer < 1_000_000 for layer in layers)
            or type(maximum_expert_bytes) is not int or maximum_expert_bytes <= 0
            or type(maximum_total_bytes) is not int or maximum_total_bytes <= 0):
        raise ValueError("invalid expert export configuration")
    import mlx.core as mx
    from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU

    configuration = None
    count = 0
    for switch in layers.values():
        if type(switch) is not SwitchGLU or type(switch.activation) is not SwiGLU:
            raise ValueError("only standard SwitchGLU/SwiGLU export is supported")
        shapes = []
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(switch, name)
            if "bias" in projection:
                raise ValueError("linear bias is unsupported")
            bits = getattr(projection, "bits", None)
            group = getattr(projection, "group_size", 64)
            current = (bits, group)
            if (bits is not None and (bits not in (4, 8)
                                      or getattr(projection, "mode", None) != "affine")):
                raise ValueError("unsupported quantization")
            if group not in (32, 64, 128):
                raise ValueError("unsupported group size")
            if configuration is None:
                configuration = current
            if current != configuration:
                raise ValueError("mixed expert formats are unsupported")
            shapes.append((projection.num_experts, projection.output_dims, projection.input_dims))
        gate, up, down = shapes
        if gate != up or down != (gate[0], gate[2], gate[1]):
            raise ValueError("inconsistent SwitchGLU shape")
        count += gate[0]
    if not 1 <= count <= 65_536:
        raise ValueError("expert count exceeds manifest limit")
    root = Path(destination)
    root.mkdir(mode=0o700)  # Existing destinations are never reused.
    rows = []
    total = 0
    bits, group = configuration
    for layer, switch in sorted(layers.items()):
        for expert in range(switch.gate_proj.num_experts):
            tensors = {}
            for name in ("gate_proj", "up_proj", "down_proj"):
                projection = getattr(switch, name)
                parts = ("weight",) if bits is None else ("weight", "scales", "biases")
                for part in parts:
                    tensors[f"{name}.{part}"] = projection[part][expert]
            if sum(t.nbytes for t in tensors.values()) > maximum_expert_bytes:
                raise ValueError("expert tensor bytes exceed export limit")
            mx.eval(*tensors.values())
            path = root / f"layer-{layer}-expert-{expert}.safetensors"
            mx.save_safetensors(str(path), tensors)
            size = path.stat().st_size
            total += size
            if size > maximum_expert_bytes or total > maximum_total_bytes:
                raise ValueError("expert artifact bytes exceed export limit")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            rows.append(dict(layer=layer, expert=expert, size_bytes=size, sha256=digest.hexdigest()))
    payload = dict(schema_version=1, model_sha256=model_sha256,
                   quantization_bits=bits, group_size=group, experts=rows)
    raw = json.dumps(payload, separators=(",", ":")).encode()
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("export manifest exceeds limit")
    temporary = root / ".expert-manifest.partial"
    temporary.write_bytes(raw)
    ExpertManifest.load(temporary, expected_model_sha256=model_sha256)
    manifest = root / "expert-manifest.json"
    temporary.rename(manifest)  # Completion marker appears only after validation.
    return manifest
