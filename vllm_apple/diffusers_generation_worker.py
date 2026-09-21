from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import io
import json
import math
import os
import platform
import resource
import shutil
import stat
import sys
import tempfile
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .generative_collector import GenerationTelemetryEvent
from .generative_worker_protocol import consume_private_generative_request
from .hardware import detect_memory, detect_thermal_state

MAX_GENERATED_ARTIFACT_BYTES = 16 * 1024**3
MAX_INPUT_IMAGE_BYTES = 64 * 1024**2
MAX_INPUT_IMAGE_PIXELS = 4096 * 4096
QWEN_IMAGE_EDIT_CONDITION_RESOLUTION = 256
_IMAGE_PIPELINES = {
    "flux2-klein-9b-base": "Flux2KleinPipeline",
    "qwen-image-2512": "QwenImagePipeline",
    "qwen-image-2.1": "QwenImage21Pipeline",
    "flux2-dev": "Flux2Pipeline",
}
_SEQUENTIAL_OFFLOAD_CONTRACTS = {
    "qwen-image-2.1": "text_encoder->transformer->vae",
}


@dataclass(frozen=True, slots=True)
class WorkerTelemetry:
    process_rss_bytes: int
    memory_pressure: str
    thermal_state: str


@dataclass(frozen=True, slots=True)
class GeneratedImageArtifact:
    path: Path
    width: int
    height: int


class DiffusersImageRuntime(Protocol):
    pipeline_class: str

    def generate(
        self,
        request: Mapping[str, object],
        progress: Callable[[], None],
    ) -> GeneratedImageArtifact: ...


class LocalDiffusersImageRuntime:
    """Lazily imports a local-only Diffusers pipeline inside the isolated worker."""

    def __init__(
        self, candidate_id: str, *, module_loader=importlib.import_module,
        phase_handoff_manifest: str | Path | None = None,
    ) -> None:
        try:
            self.pipeline_class = _IMAGE_PIPELINES[candidate_id]
        except KeyError as error:
            raise ValueError("unsupported local Diffusers image candidate") from error
        self._candidate_id = candidate_id
        self._module_loader = module_loader
        self._phase_handoff_manifest = (
            Path(phase_handoff_manifest) if phase_handoff_manifest is not None else None
        )

    def generate(
        self,
        request: Mapping[str, object],
        progress: Callable[[], None],
    ) -> GeneratedImageArtifact:
        if request.get("candidate_id") != self._candidate_id:
            raise ValueError("local Diffusers runtime candidate does not match its request")
        if request.get("batch_size") != 1:
            raise ValueError("qualification image worker requires batch size one")
        if self._phase_handoff_manifest is not None and self._candidate_id != "qwen-image-2.1":
            raise ValueError("phase handoff is only supported for Qwen-Image-2.1")
        if self._phase_handoff_manifest is not None and request.get("mode") == "image-edit":
            raise ValueError("image-edit phase handoff requires condition image transfer")
        phase_tensors = None
        if self._phase_handoff_manifest is not None:
            from .qwen_image_21_phase_handoff import consume_qwen_image_21_phase_handoff

            phase_tensors = consume_qwen_image_21_phase_handoff(
                self._phase_handoff_manifest,
                plan_sha256=str(request["plan_sha256"]),
                prompt_sha256=str(request["prompt_sha256"]),
                sample_index=int(request["sample_index"]),
                mode=str(request["mode"]),
            )
        model_root = Path(str(request["model_root"])).resolve(strict=True)
        output_root = Path(str(request["output_root"])).resolve(strict=True)
        if not model_root.is_dir() or not output_root.is_dir():
            raise ValueError("local Diffusers model and output roots must be directories")
        torch = self._module_loader("torch")
        diffusers = self._module_loader("diffusers")
        transformers = (
            self._module_loader("transformers")
            if self._candidate_id == "qwen-image-2.1" else None
        )
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("Diffusers image worker requires an available MPS device")
        pipeline_type = getattr(diffusers, self.pipeline_class, None)
        if pipeline_type is None:
            raise RuntimeError(f"Diffusers does not expose {self.pipeline_class}")
        pipeline = None
        disk_offload_root = None
        try:
            prompt_arguments = {}
            image_pad_mask = None
            generation_prompt = request["prompt"]
            input_image = None
            if request.get("mode", "text-to-image") == "image-edit":
                input_image = load_private_input_image(request, self._module_loader)
            if self._candidate_id == "qwen-image-2.1":
                scheduler_type = getattr(diffusers, "FlowMatchEulerDiscreteScheduler", None)
                vae_type = getattr(diffusers, "AutoencoderKLQwenImage21", None)
                transformer_type = getattr(
                    diffusers, "QwenImage21Transformer2DModel", None
                )
                if any(value is None for value in (scheduler_type, vae_type, transformer_type)):
                    raise RuntimeError("Qwen-Image-2.1 staged component classes are unavailable")
                if phase_tensors is not None:
                    prompt_embeds = phase_tensors["prompt_embeds"]
                    prompt_embeds_mask = phase_tensors.get("prompt_embeds_mask")
                    image_pad_mask = phase_tensors.get("image_pad_mask")
                    processor_type = getattr(transformers, "Qwen3VLProcessor", None)
                    if processor_type is None:
                        raise RuntimeError("Qwen-Image-2.1 processor class is unavailable")
                    processor = processor_type.from_pretrained(
                        str(model_root / "processor"), local_files_only=True
                    )
                else:
                    processor_type = getattr(transformers, "Qwen3VLProcessor", None)
                    text_encoder_type = getattr(
                        transformers, "Qwen3VLForConditionalGeneration", None
                    )
                    if processor_type is None or text_encoder_type is None:
                        raise RuntimeError("Qwen-Image-2.1 text classes are unavailable")
                    processor = processor_type.from_pretrained(
                        str(model_root / "processor"), local_files_only=True
                    )
                    text_encoder = text_encoder_type.from_pretrained(
                        str(model_root / "text_encoder"),
                        local_files_only=True,
                        dtype=torch.bfloat16,
                    )
                    text_pipeline = pipeline_type(
                        scheduler=None,
                        vae=None,
                        text_encoder=text_encoder,
                        processor=processor,
                        transformer=None,
                    )
                    try:
                        self._enable_text_encoder_layer_offload(text_pipeline)
                        if input_image is not None:
                            resize = getattr(text_pipeline.image_processor, "resize", None)
                            if not callable(resize):
                                raise RuntimeError(
                                    "Qwen-Image pipeline lacks condition image resizing"
                                )
                            condition_width, condition_height = self._condition_dimensions(
                                input_image.width, input_image.height
                            )
                            input_image = resize(
                                input_image,
                                width=condition_width,
                                height=condition_height,
                            )
                        encode_prompt = getattr(text_pipeline, "encode_prompt", None)
                        remove_hooks = getattr(text_pipeline, "remove_all_hooks", None)
                        if not callable(encode_prompt) or not callable(remove_hooks):
                            raise RuntimeError(
                                "Qwen-Image-2.1 pipeline lacks the staged text-encoder contract"
                            )
                        prompt_embeds, prompt_embeds_mask, image_pad_mask = encode_prompt(
                            prompt=generation_prompt,
                            image=[input_image] if input_image is not None else None,
                            device=getattr(text_pipeline, "_execution_device", "mps"),
                            num_images_per_prompt=request["batch_size"],
                        )
                        progress()
                        remove_hooks()
                        text_pipeline.text_encoder = None
                        text_encoder = None
                    finally:
                        del text_pipeline
                        self._release_mps(torch)
                progress()
                scheduler = scheduler_type.from_pretrained(
                    str(model_root / "scheduler"), local_files_only=True
                )
                vae = vae_type.from_pretrained(
                    str(model_root / "vae"), local_files_only=True,
                    dtype=torch.bfloat16,
                )
                transformer = transformer_type.from_pretrained(
                    str(model_root / "transformer"), local_files_only=True,
                    dtype=torch.bfloat16,
                )
                pipeline = pipeline_type(
                    scheduler=scheduler,
                    vae=vae,
                    text_encoder=None,
                    processor=processor,
                    transformer=transformer,
                )
                generation_prompt = None
                prompt_arguments = {
                    "prompt_embeds": prompt_embeds,
                    "prompt_embeds_mask": prompt_embeds_mask,
                }
            else:
                pipeline = pipeline_type.from_pretrained(
                    str(model_root),
                    local_files_only=True,
                    dtype=torch.bfloat16,
                )
            vae = getattr(pipeline, "vae", None)
            if vae is not None and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            required_sequence = _SEQUENTIAL_OFFLOAD_CONTRACTS.get(self._candidate_id)
            if required_sequence is None:
                pipeline.to("mps")
            else:
                if request.get("disk_offload", False):
                    disk_offload_root = Path(
                        tempfile.mkdtemp(
                            prefix=f"offload-{request['sample_index']}-",
                            dir=output_root,
                        )
                    )
                    disk_offload_root.chmod(0o700)
                self._enable_generation_group_offload(
                    pipeline, torch, disk_offload_root=disk_offload_root
                )
            if input_image is not None and self._candidate_id == "qwen-image-2.1":
                self._bind_image_pad_mask(pipeline, image_pad_mask)
            progress()
            generator = torch.Generator(device="cpu").manual_seed(request["seed"])

            def callback(_pipeline, _step, _timestep, callback_kwargs):
                progress()
                return callback_kwargs

            result = pipeline(
                prompt=generation_prompt,
                width=request["width"],
                height=request["height"],
                num_inference_steps=request["steps"],
                num_images_per_prompt=request["batch_size"],
                generator=generator,
                callback_on_step_end=callback,
                **(
                    {
                        "output_resolution": (
                            QWEN_IMAGE_EDIT_CONDITION_RESOLUTION
                            if input_image is not None
                            else max(request["width"], request["height"])
                        )
                    }
                    if self._candidate_id == "qwen-image-2.1"
                    else {}
                ),
                **({"image": input_image} if input_image is not None else {}),
                **prompt_arguments,
            )
            images = getattr(result, "images", None)
            if not isinstance(images, (list, tuple)) or len(images) != request["batch_size"]:
                raise RuntimeError("Diffusers image pipeline returned an invalid image batch")
            descriptor, temporary = tempfile.mkstemp(
                prefix=f"qualification-{request['sample_index']}-",
                suffix=".png",
                dir=output_root,
            )
            os.fchmod(descriptor, 0o600)
            os.close(descriptor)
            output = Path(temporary)
            try:
                images[0].save(output, format="PNG")
            except BaseException:
                try:
                    output.unlink()
                except FileNotFoundError:
                    pass
                raise
            return GeneratedImageArtifact(output, request["width"], request["height"])
        finally:
            if pipeline is not None:
                del pipeline
            self._release_mps(torch)
            if disk_offload_root is not None:
                shutil.rmtree(disk_offload_root, ignore_errors=True)

    @staticmethod
    def _release_mps(torch: object) -> None:
        gc.collect()
        synchronize = getattr(getattr(torch, "mps", None), "synchronize", None)
        if callable(synchronize):
            synchronize()
        empty_cache = getattr(getattr(torch, "mps", None), "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    @staticmethod
    def _enable_generation_group_offload(
        pipeline: object, torch: object, *, disk_offload_root: Path | None = None
    ) -> None:
        required = _SEQUENTIAL_OFFLOAD_CONTRACTS["qwen-image-2.1"]
        if getattr(pipeline, "model_cpu_offload_seq", None) != required:
            raise RuntimeError(
                "Diffusers image pipeline offload sequence does not match the candidate"
            )
        enable = getattr(pipeline, "enable_group_offload", None)
        device = getattr(torch, "device", None)
        if not callable(enable):
            raise RuntimeError("Diffusers image pipeline does not expose group offload")
        if not callable(device):
            raise RuntimeError("Diffusers image worker lacks torch device construction")
        arguments = dict(
            onload_device=device("mps"),
            offload_device=device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=1,
            non_blocking=False,
            use_stream=False,
            low_cpu_mem_usage=False,
        )
        if disk_offload_root is not None:
            arguments["offload_to_disk_path"] = str(disk_offload_root)
        enable(**arguments)

    @staticmethod
    def _enable_text_encoder_layer_offload(pipeline: object) -> None:
        required = _SEQUENTIAL_OFFLOAD_CONTRACTS["qwen-image-2.1"]
        if getattr(pipeline, "model_cpu_offload_seq", None) != required:
            raise RuntimeError(
                "Diffusers image pipeline offload sequence does not match the candidate"
            )
        enable = getattr(pipeline, "enable_sequential_cpu_offload", None)
        if not callable(enable):
            raise RuntimeError(
                "Diffusers image pipeline does not expose sequential CPU offload"
            )
        enable(device="mps")

    @staticmethod
    def _bind_image_pad_mask(pipeline: object, image_pad_mask: object) -> None:
        if image_pad_mask is None:
            raise RuntimeError("Qwen-Image image-edit prompt lacks image pad mask")
        original = getattr(pipeline, "encode_prompt", None)
        if not callable(original):
            raise RuntimeError("Qwen-Image pipeline lacks encode_prompt")

        def encode_with_bound_mask(_pipeline, *args, **kwargs):
            if kwargs.get("prompt_embeds") is not None and kwargs.get("image") is not None:
                if kwargs.get("image_pad_mask") is not None:
                    raise RuntimeError("Qwen-Image image pad mask was unexpectedly supplied")
                kwargs["image_pad_mask"] = image_pad_mask
            return original(*args, **kwargs)

        pipeline.encode_prompt = types.MethodType(encode_with_bound_mask, pipeline)

    @staticmethod
    def _condition_dimensions(width: int, height: int) -> tuple[int, int]:
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ValueError("Qwen-Image condition image dimensions are invalid")
        ratio = width / height
        target_area = QWEN_IMAGE_EDIT_CONDITION_RESOLUTION**2
        condition_width = max(32, round(math.sqrt(target_area * ratio) / 32) * 32)
        condition_height = max(32, round(condition_width / ratio / 32) * 32)
        if condition_width * condition_height > 2 * target_area:
            raise ValueError("Qwen-Image condition image aspect ratio is unsupported")
        return condition_width, condition_height

def load_private_input_image(
    request: Mapping[str, object], module_loader=importlib.import_module
) -> object:
        path_value = request.get("input_image_path")
        expected_digest = request.get("input_image_sha256")
        expected_size = request.get("input_image_bytes")
        if (
            not isinstance(path_value, str)
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or not isinstance(expected_size, int)
            or not 1 <= expected_size <= MAX_INPUT_IMAGE_BYTES
        ):
            raise ValueError("image-edit request has invalid input image metadata")
        path = Path(path_value)
        if path.is_symlink():
            raise ValueError("image-edit input must not be a symlink")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_size != expected_size
            ):
                raise ValueError("image-edit input is not the bound private file")
            data = bytearray()
            while len(data) < expected_size:
                chunk = os.read(descriptor, min(1024 * 1024, expected_size - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
            if len(data) != expected_size or os.read(descriptor, 1):
                raise ValueError("image-edit input changed while being read")
        finally:
            os.close(descriptor)
        if hashlib.sha256(data).hexdigest() != expected_digest:
            raise ValueError("image-edit input digest does not match its request")
        image_module = module_loader("PIL.Image")
        image = image_module.open(io.BytesIO(data))
        try:
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError("image-edit input must be PNG or JPEG")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_INPUT_IMAGE_PIXELS:
                raise ValueError("image-edit input dimensions exceed the bounded contract")
            image.load()
            return image.convert("RGB")
        finally:
            image.close()


def _remove_output_if_owned(path: Path, output_root: Path) -> None:
    try:
        if path.is_symlink():
            return
        resolved = path.resolve(strict=True)
        info = resolved.stat()
        if (
            resolved != output_root
            and resolved.is_relative_to(output_root)
            and stat.S_ISREG(info.st_mode)
            and info.st_uid == os.getuid()
        ):
            resolved.unlink()
    except FileNotFoundError:
        pass


def _hash_private_output(path: Path, output_root: Path) -> str:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise ValueError("Diffusers worker output must not be a symlink")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_relative_to(output_root) or resolved == output_root:
        raise ValueError("Diffusers worker output must be inside the output root")
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    info = None
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or not 1 <= info.st_size <= MAX_GENERATED_ARTIFACT_BYTES
        ):
            raise ValueError("Diffusers worker output is not a bounded current-user file")
        digest = hashlib.sha256()
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
        if remaining:
            raise ValueError("Diffusers worker output changed while being hashed")
        return digest.hexdigest()
    finally:
        os.close(descriptor)
        try:
            current = resolved.lstat()
            if (
                info is not None
                and current.st_dev == info.st_dev
                and current.st_ino == info.st_ino
            ):
                resolved.unlink()
        except FileNotFoundError:
            pass


def execute_diffusers_image_request(
    request: Mapping[str, object],
    runtime: DiffusersImageRuntime,
    *,
    telemetry: Callable[[], WorkerTelemetry],
    emit: Callable[[GenerationTelemetryEvent], None],
    clock: Callable[[], float] = time.monotonic,
) -> None:
    _execute_image_request(
        request,
        runtime,
        expected_runtimes=_IMAGE_PIPELINES,
        backend_name="Diffusers",
        telemetry=telemetry,
        emit=emit,
        clock=clock,
    )


def execute_local_image_request(
    request: Mapping[str, object],
    runtime: DiffusersImageRuntime,
    *,
    expected_runtimes: Mapping[str, str],
    backend_name: str,
    telemetry: Callable[[], WorkerTelemetry],
    emit: Callable[[GenerationTelemetryEvent], None],
    clock: Callable[[], float] = time.monotonic,
) -> None:
    _execute_image_request(
        request,
        runtime,
        expected_runtimes=expected_runtimes,
        backend_name=backend_name,
        telemetry=telemetry,
        emit=emit,
        clock=clock,
    )


def _execute_image_request(
    request: Mapping[str, object],
    runtime: DiffusersImageRuntime,
    *,
    expected_runtimes: Mapping[str, str],
    backend_name: str,
    telemetry: Callable[[], WorkerTelemetry],
    emit: Callable[[GenerationTelemetryEvent], None],
    clock: Callable[[], float],
) -> None:
    candidate_id = request.get("candidate_id")
    expected_pipeline = expected_runtimes.get(candidate_id)
    if expected_pipeline is None or request.get("modality") != "image":
        raise ValueError(f"{backend_name} image worker does not support this candidate")
    if request.get("mode") not in {"text-to-image", "image-edit"}:
        raise ValueError(f"{backend_name} image worker does not support this mode")
    if runtime.pipeline_class != expected_pipeline:
        raise ValueError(f"{backend_name} runtime class does not match the candidate")
    output_root_value = request.get("output_root")
    if not isinstance(output_root_value, str):
        raise ValueError(f"{backend_name} worker output root is invalid")
    output_root = Path(output_root_value).resolve(strict=True)
    started = clock()

    def make_event(kind: str, *, output: GeneratedImageArtifact | None = None, digest=None):
        snapshot = telemetry()
        ceiling = request.get("memory_hard_ceiling_bytes")
        if isinstance(ceiling, int) and snapshot.process_rss_bytes > ceiling:
            raise MemoryError(
                "Generative worker exceeded its memory hard ceiling: "
                f"effective_resident_bytes={snapshot.process_rss_bytes}, "
                f"memory_hard_ceiling_bytes={ceiling}"
            )
        elapsed_ms = max(0.0, (clock() - started) * 1000.0)
        return GenerationTelemetryEvent(
            kind=kind,
            elapsed_ms=elapsed_ms,
            process_rss_bytes=snapshot.process_rss_bytes,
            memory_pressure=snapshot.memory_pressure,
            thermal_state=snapshot.thermal_state,
            output_width=output.width if output is not None else None,
            output_height=output.height if output is not None else None,
            output_frames=1 if output is not None else None,
            output_sha256=digest,
        )

    emit(make_event("started"))

    def progress() -> None:
        emit(make_event("progress"))

    output = runtime.generate(request, progress)
    if output.width != request.get("width") or output.height != request.get("height"):
        _remove_output_if_owned(output.path, output_root)
        raise ValueError("Diffusers worker output shape does not match the request")
    digest = _hash_private_output(output.path, output_root)
    emit(make_event("first_output"))
    emit(make_event("completed", output=output, digest=digest))


def _darwin_thermal_state() -> str:
    return detect_thermal_state().value


def default_worker_telemetry() -> WorkerTelemetry:
    memory = detect_memory()
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = int(usage if platform.system() == "Darwin" else usage * 1024)
    mlx_peak = 0
    mlx_core = sys.modules.get("mlx.core")
    get_peak_memory = getattr(mlx_core, "get_peak_memory", None)
    if callable(get_peak_memory):
        try:
            mlx_peak = max(0, int(get_peak_memory()))
        except (RuntimeError, TypeError, ValueError):
            mlx_peak = 0
    mps_allocated = 0
    torch = sys.modules.get("torch")
    mps = getattr(torch, "mps", None)
    for name in ("current_allocated_memory", "driver_allocated_memory"):
        probe = getattr(mps, name, None)
        if callable(probe):
            try:
                mps_allocated = max(mps_allocated, max(0, int(probe())))
            except (RuntimeError, TypeError, ValueError):
                continue
    effective_resident = max(1, rss, mlx_peak, mps_allocated)
    return WorkerTelemetry(
        effective_resident,
        memory.pressure.value,
        _darwin_thermal_state(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vllm-apple-diffusers-worker")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--phase-handoff", type=Path)
    arguments = parser.parse_args(argv)
    try:
        request = consume_private_generative_request(
            arguments.request,
            workspace_root=arguments.workspace_root,
        )
        if arguments.phase_handoff is not None:
            workspace = arguments.workspace_root.resolve(strict=True)
            handoff_parent = arguments.phase_handoff.parent.resolve(strict=True)
            if handoff_parent == workspace or not handoff_parent.is_relative_to(workspace):
                raise ValueError("Qwen-Image phase handoff must be in workspace")
        runtime = LocalDiffusersImageRuntime(
            request["candidate_id"], phase_handoff_manifest=arguments.phase_handoff
        )

        def emit(event: GenerationTelemetryEvent) -> None:
            print(json.dumps(asdict(event), sort_keys=True, separators=(",", ":")), flush=True)

        execute_diffusers_image_request(
            request,
            runtime,
            telemetry=default_worker_telemetry,
            emit=emit,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "vllm_apple_error_code": "diffusers_image_worker_failed",
                    "vllm_apple_error_detail": f"{type(error).__name__}: {error}"[:512],
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
