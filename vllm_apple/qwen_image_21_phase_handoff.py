from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Mapping

HANDOFF_ABI_VERSION = 1
MAX_HANDOFF_BYTES = 512 * 1024**2
MAX_MANIFEST_BYTES = 32 * 1024
ALLOWED_TENSORS = {
    "prompt_embeds",
    "prompt_embeds_mask",
    "image_pad_mask",
}
ALLOWED_DTYPES = {
    "torch.bfloat16",
    "torch.float16",
    "torch.float32",
    "torch.bool",
    "torch.int32",
    "torch.int64",
}


def _digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_HANDOFF_BYTES:
                raise ValueError("Qwen-Image phase handoff exceeds the bounded limit")
            digest.update(chunk)
    if size <= 0:
        raise ValueError("Qwen-Image phase handoff is empty")
    return digest.hexdigest(), size


def _tensor_specs(tensors: Mapping[str, object]) -> dict[str, dict[str, object]]:
    if not tensors or not set(tensors).issubset(ALLOWED_TENSORS):
        raise ValueError("Qwen-Image phase handoff tensor names are invalid")
    if "prompt_embeds" not in tensors:
        raise ValueError("Qwen-Image phase handoff lacks prompt embeddings")
    specs: dict[str, dict[str, object]] = {}
    for name, tensor in sorted(tensors.items()):
        shape = getattr(tensor, "shape", None)
        dtype = str(getattr(tensor, "dtype", ""))
        if (
            not isinstance(shape, (tuple, list))
            or not 1 <= len(shape) <= 5
            or any(type(value) is not int or not 1 <= value <= 65536 for value in shape)
            or dtype not in ALLOWED_DTYPES
        ):
            raise ValueError("Qwen-Image phase handoff tensor metadata is invalid")
        specs[name] = {"shape": list(shape), "dtype": dtype}
    return specs


def save_qwen_image_21_phase_handoff(
    root: str | Path,
    tensors: Mapping[str, object],
    *,
    plan_sha256: str,
    prompt_sha256: str,
    sample_index: int,
    mode: str,
    save_file: Callable[[dict[str, object], str], None] | None = None,
) -> Path:
    directory = Path(root).resolve(strict=True)
    info = directory.stat()
    if (
        not directory.is_dir()
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise ValueError("Qwen-Image phase handoff root must be private")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in (plan_sha256, prompt_sha256)
    ):
        raise ValueError("Qwen-Image phase handoff identity digest is invalid")
    if not 0 <= sample_index < 32 or mode not in {"text-to-image", "image-edit"}:
        raise ValueError("Qwen-Image phase handoff identity is invalid")
    specs = _tensor_specs(tensors)
    if (mode == "image-edit") != ("image_pad_mask" in tensors):
        raise ValueError("Qwen-Image phase handoff image mask does not match mode")
    if save_file is None:
        from safetensors.torch import save_file

    payload = directory / "prompt-embeddings.safetensors"
    manifest = directory / "prompt-embeddings.json"
    if payload.exists() or manifest.exists():
        raise ValueError("Qwen-Image phase handoff already exists")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".handoff-", dir=directory)
    os.close(descriptor)
    temporary = Path(temporary_name)
    manifest_temporary = None
    try:
        save_file(dict(tensors), str(temporary))
        temporary.chmod(0o600)
        payload_sha256, payload_bytes = _digest_file(temporary)
        body = {
            "abi_version": HANDOFF_ABI_VERSION,
            "mode": mode,
            "payload_bytes": payload_bytes,
            "payload_sha256": payload_sha256,
            "plan_sha256": plan_sha256,
            "prompt_sha256": prompt_sha256,
            "sample_index": sample_index,
            "tensors": specs,
        }
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise ValueError("Qwen-Image phase handoff manifest is oversized")
        descriptor, manifest_name = tempfile.mkstemp(prefix=".handoff-manifest-", dir=directory)
        manifest_temporary = Path(manifest_name)
        try:
            os.fchmod(descriptor, 0o600)
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, payload)
        os.replace(manifest_temporary, manifest)
        return manifest
    except BaseException:
        payload.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
        if manifest_temporary is not None:
            manifest_temporary.unlink(missing_ok=True)


def consume_qwen_image_21_phase_handoff(
    manifest_path: str | Path,
    *,
    plan_sha256: str,
    prompt_sha256: str,
    sample_index: int,
    mode: str,
    load_file: Callable[[str], dict[str, object]] | None = None,
) -> dict[str, object]:
    unresolved = Path(manifest_path)
    if unresolved.name != "prompt-embeddings.json":
        raise ValueError("Qwen-Image phase handoff manifest name is invalid")
    payload = unresolved.with_name("prompt-embeddings.safetensors")
    paths = (unresolved, payload)
    try:
        for path in paths:
            if path.is_symlink():
                raise ValueError("Qwen-Image phase handoff must not be a symlink")
            info = path.stat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise ValueError("Qwen-Image phase handoff file must be private")
        raw = unresolved.read_bytes()
        if not 1 <= len(raw) <= MAX_MANIFEST_BYTES:
            raise ValueError("Qwen-Image phase handoff manifest is outside the limit")
        body = json.loads(raw)
        expected_fields = {
            "abi_version", "mode", "payload_bytes", "payload_sha256",
            "plan_sha256", "prompt_sha256", "sample_index", "tensors",
        }
        if not isinstance(body, dict) or set(body) != expected_fields:
            raise ValueError("Qwen-Image phase handoff manifest schema is invalid")
        expected_identity = (plan_sha256, prompt_sha256, sample_index, mode)
        actual_identity = (
            body["plan_sha256"], body["prompt_sha256"],
            body["sample_index"], body["mode"],
        )
        if body["abi_version"] != HANDOFF_ABI_VERSION or actual_identity != expected_identity:
            raise ValueError("Qwen-Image phase handoff identity does not match")
        digest, size = _digest_file(payload)
        if digest != body["payload_sha256"] or size != body["payload_bytes"]:
            raise ValueError("Qwen-Image phase handoff payload does not match")
        if load_file is None:
            from safetensors.torch import load_file

        tensors = load_file(str(payload))
        if _tensor_specs(tensors) != body["tensors"]:
            raise ValueError("Qwen-Image phase handoff tensors do not match manifest")
        if (mode == "image-edit") != ("image_pad_mask" in tensors):
            raise ValueError("Qwen-Image phase handoff image mask does not match mode")
        return tensors
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Qwen-Image phase handoff is invalid") from error
    finally:
        for path in paths:
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)


def encode_qwen_image_21_prompt_handoff(
    request: Mapping[str, object],
    handoff_root: str | Path,
    *,
    module_loader=importlib.import_module,
) -> Path:
    """Run only the Qwen text phase and persist ordinary CPU tensors."""
    from .diffusers_generation_worker import (
        LocalDiffusersImageRuntime,
        load_private_input_image,
    )

    if request.get("candidate_id") != "qwen-image-2.1":
        raise ValueError("Qwen-Image phase encoder candidate is invalid")
    torch = module_loader("torch")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is None or not mps.is_available():
        raise RuntimeError("Qwen-Image phase encoder requires MPS")
    transformers = module_loader("transformers")
    diffusers = module_loader("diffusers")
    processor_type = getattr(transformers, "Qwen3VLProcessor", None)
    text_encoder_type = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    pipeline_type = getattr(diffusers, "QwenImage21Pipeline", None)
    if any(value is None for value in (processor_type, text_encoder_type, pipeline_type)):
        raise RuntimeError("Qwen-Image phase encoder classes are unavailable")
    model_root = Path(str(request["model_root"])).resolve(strict=True)
    processor = processor_type.from_pretrained(
        str(model_root / "processor"), local_files_only=True
    )
    text_encoder = text_encoder_type.from_pretrained(
        str(model_root / "text_encoder"),
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    pipeline = pipeline_type(
        scheduler=None,
        vae=None,
        text_encoder=text_encoder,
        processor=processor,
        transformer=None,
    )
    try:
        LocalDiffusersImageRuntime._enable_text_encoder_layer_offload(pipeline)
        input_image = None
        if request.get("mode") == "image-edit":
            input_image = load_private_input_image(request, module_loader)
            resize = getattr(pipeline.image_processor, "resize", None)
            if not callable(resize):
                raise RuntimeError("Qwen-Image phase encoder lacks image resizing")
            width, height = LocalDiffusersImageRuntime._condition_dimensions(
                input_image.width, input_image.height
            )
            input_image = resize(input_image, width=width, height=height)
        encoded = pipeline.encode_prompt(
            prompt=request["prompt"],
            image=[input_image] if input_image is not None else None,
            device=getattr(pipeline, "_execution_device", "mps"),
            num_images_per_prompt=request["batch_size"],
        )
        if not isinstance(encoded, (tuple, list)) or len(encoded) != 3:
            raise RuntimeError("Qwen-Image phase encoder returned invalid prompt tensors")
        prompt_embeds, prompt_embeds_mask, image_pad_mask = encoded

        def cpu_tensor(value):
            detach = getattr(value, "detach", None)
            if callable(detach):
                value = detach()
            to = getattr(value, "to", None)
            if not callable(to):
                as_tensor = getattr(torch, "as_tensor", None)
                if not callable(as_tensor):
                    raise RuntimeError("Qwen-Image phase encoder returned a non-tensor")
                value = as_tensor(value)
                to = getattr(value, "to", None)
                if not callable(to):
                    raise RuntimeError("Qwen-Image phase encoder returned a non-tensor")
            value = to("cpu")
            contiguous = getattr(value, "contiguous", None)
            return contiguous() if callable(contiguous) else value

        tensors = {
            "prompt_embeds": cpu_tensor(prompt_embeds),
        }
        if prompt_embeds_mask is not None:
            tensors["prompt_embeds_mask"] = cpu_tensor(prompt_embeds_mask)
        if request.get("mode") == "image-edit":
            tensors["image_pad_mask"] = cpu_tensor(image_pad_mask)
        remove_hooks = getattr(pipeline, "remove_all_hooks", None)
        if not callable(remove_hooks):
            raise RuntimeError("Qwen-Image phase encoder lacks hook cleanup")
        remove_hooks()
        pipeline.text_encoder = None
        text_encoder = None
        LocalDiffusersImageRuntime._release_mps(torch)
        return save_qwen_image_21_phase_handoff(
            handoff_root,
            tensors,
            plan_sha256=str(request["plan_sha256"]),
            prompt_sha256=str(request["prompt_sha256"]),
            sample_index=int(request["sample_index"]),
            mode=str(request["mode"]),
        )
    finally:
        del pipeline
        LocalDiffusersImageRuntime._release_mps(torch)


def main(argv: list[str] | None = None) -> int:
    from .generative_worker_protocol import consume_private_generative_request

    parser = argparse.ArgumentParser(prog="vllm-apple-qwen-image-encoder")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--handoff-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    started = time.monotonic()
    try:
        workspace = arguments.workspace_root.resolve(strict=True)
        handoff = arguments.handoff_root.resolve(strict=True)
        if handoff == workspace or not handoff.is_relative_to(workspace):
            raise ValueError("Qwen-Image phase handoff root must be in workspace")
        request = consume_private_generative_request(
            arguments.request, workspace_root=workspace
        )
        encode_qwen_image_21_prompt_handoff(request, handoff)
        from .diffusers_generation_worker import default_worker_telemetry

        telemetry = default_worker_telemetry()
        print(json.dumps({
            "schema_version": 1,
            "phase": "text_encoder",
            "elapsed_ms": max(0.001, (time.monotonic() - started) * 1000.0),
            "peak_rss_bytes": telemetry.process_rss_bytes,
            "memory_pressure": telemetry.memory_pressure,
            "thermal_state": telemetry.thermal_state,
        }, sort_keys=True), flush=True)
    except Exception as error:
        print(
            json.dumps({
                "vllm_apple_error_code": "qwen_image_21_encoder_failed",
                "vllm_apple_error_detail": f"{type(error).__name__}: {error}"[:512],
            }, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
