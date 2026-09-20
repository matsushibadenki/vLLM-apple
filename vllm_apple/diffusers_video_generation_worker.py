from __future__ import annotations

import argparse
import importlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .diffusers_generation_worker import (
    WorkerTelemetry,
    _hash_private_output,
    _remove_output_if_owned,
    default_worker_telemetry,
)
from .generative_collector import GenerationTelemetryEvent
from .generative_worker_protocol import consume_private_generative_request


_VIDEO_PIPELINES = {"wan2.2-ti2v-5b": "WanPipeline"}
DEFAULT_VIDEO_FPS = 16
WAN_MODULE_RESIDENCY_SEQUENCE = "text_encoder->transformer->vae"


@dataclass(frozen=True, slots=True)
class GeneratedVideoArtifact:
    path: Path
    width: int
    height: int
    frames: int


class DiffusersVideoRuntime(Protocol):
    pipeline_class: str

    def generate(
        self,
        request: Mapping[str, object],
        progress: Callable[[], None],
    ) -> GeneratedVideoArtifact: ...


class LocalDiffusersVideoRuntime:
    """Load a local Wan T2V checkpoint only inside the isolated worker."""

    def __init__(self, candidate_id: str, *, module_loader=importlib.import_module) -> None:
        try:
            self.pipeline_class = _VIDEO_PIPELINES[candidate_id]
        except KeyError as error:
            raise ValueError("unsupported local Diffusers video candidate") from error
        self._candidate_id = candidate_id
        self._module_loader = module_loader

    def generate(
        self,
        request: Mapping[str, object],
        progress: Callable[[], None],
    ) -> GeneratedVideoArtifact:
        if request.get("candidate_id") != self._candidate_id:
            raise ValueError("local Diffusers video runtime candidate does not match its request")
        model_root = Path(str(request["model_root"])).resolve(strict=True)
        output_root = Path(str(request["output_root"])).resolve(strict=True)
        if not model_root.is_dir() or not output_root.is_dir():
            raise ValueError("local Diffusers model and output roots must be directories")
        torch = self._module_loader("torch")
        diffusers = self._module_loader("diffusers")
        utilities = self._module_loader("diffusers.utils")
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("Diffusers video worker requires an available MPS device")
        pipeline_type = getattr(diffusers, self.pipeline_class, None)
        if pipeline_type is None:
            raise RuntimeError(f"Diffusers does not expose {self.pipeline_class}")
        export_to_video = getattr(utilities, "export_to_video", None)
        if not callable(export_to_video):
            raise RuntimeError("Diffusers video export helper is unavailable")
        pipeline = pipeline_type.from_pretrained(
            str(model_root), local_files_only=True, dtype=torch.bfloat16
        )
        try:
            if getattr(pipeline, "model_cpu_offload_seq", None) != WAN_MODULE_RESIDENCY_SEQUENCE:
                raise RuntimeError("Wan pipeline does not expose the required module residency sequence")
            enable_offload = getattr(pipeline, "enable_model_cpu_offload", None)
            if not callable(enable_offload):
                raise RuntimeError("Wan pipeline does not support model CPU offload")
            vae = getattr(pipeline, "vae", None)
            if vae is not None and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            enable_offload(device="mps")
            progress()
            generator = torch.Generator(device="cpu").manual_seed(request["seed"])

            def callback(_pipeline, _step, _timestep, callback_kwargs):
                progress()
                return callback_kwargs

            result = pipeline(
                prompt=request["prompt"],
                width=request["width"],
                height=request["height"],
                num_frames=request["frames"],
                num_inference_steps=request["steps"],
                num_videos_per_prompt=request["batch_size"],
                generator=generator,
                callback_on_step_end=callback,
            )
            batches = getattr(result, "frames", None)
            if not isinstance(batches, (list, tuple)) or len(batches) != 1:
                raise RuntimeError("Diffusers video pipeline returned an invalid video batch")
            frames = batches[0]
            if not isinstance(frames, (list, tuple)) or len(frames) != request["frames"]:
                raise RuntimeError("Diffusers video pipeline returned an invalid frame count")
            descriptor, temporary = tempfile.mkstemp(
                prefix=f"qualification-{request['sample_index']}-",
                suffix=".mp4",
                dir=output_root,
            )
            os.fchmod(descriptor, 0o600)
            os.close(descriptor)
            output = Path(temporary)
            try:
                export_to_video(frames, str(output), fps=DEFAULT_VIDEO_FPS)
            except BaseException:
                output.unlink(missing_ok=True)
                raise
            return GeneratedVideoArtifact(
                output, request["width"], request["height"], len(frames)
            )
        finally:
            del pipeline
            mps_empty_cache = getattr(getattr(torch, "mps", None), "empty_cache", None)
            if callable(mps_empty_cache):
                mps_empty_cache()


def execute_diffusers_video_request(
    request: Mapping[str, object],
    runtime: DiffusersVideoRuntime,
    *,
    telemetry: Callable[[], WorkerTelemetry],
    emit: Callable[[GenerationTelemetryEvent], None],
    clock: Callable[[], float] = time.monotonic,
) -> None:
    execute_local_video_request(
        request,
        runtime,
        expected_runtimes=_VIDEO_PIPELINES,
        backend_name="Diffusers",
        telemetry=telemetry,
        emit=emit,
        clock=clock,
    )


def execute_local_video_request(
    request: Mapping[str, object],
    runtime: DiffusersVideoRuntime,
    *,
    expected_runtimes: Mapping[str, str],
    backend_name: str,
    telemetry: Callable[[], WorkerTelemetry],
    emit: Callable[[GenerationTelemetryEvent], None],
    clock: Callable[[], float] = time.monotonic,
    enforce_memory_ceiling_during_generation: bool = True,
) -> None:
    candidate_id = request.get("candidate_id")
    expected_pipeline = expected_runtimes.get(candidate_id)
    if expected_pipeline is None or request.get("modality") != "video":
        raise ValueError(f"{backend_name} video worker does not support this candidate")
    if request.get("mode") != "text-to-video":
        raise ValueError(f"{backend_name} video worker currently supports text-to-video only")
    if request.get("batch_size") != 1:
        raise ValueError(f"{backend_name} video qualification requires batch size one")
    if runtime.pipeline_class != expected_pipeline:
        raise ValueError(f"{backend_name} video runtime class does not match the candidate")
    output_root_value = request.get("output_root")
    if not isinstance(output_root_value, str):
        raise ValueError(f"{backend_name} video worker output root is invalid")
    output_root = Path(output_root_value).resolve(strict=True)
    started = clock()

    def make_event(kind: str, *, output: GeneratedVideoArtifact | None = None, digest=None):
        snapshot = telemetry()
        ceiling = request.get("memory_hard_ceiling_bytes")
        if (
            enforce_memory_ceiling_during_generation
            and isinstance(ceiling, int)
            and snapshot.process_rss_bytes > ceiling
        ):
            raise MemoryError("Generative worker exceeded its memory hard ceiling")
        return GenerationTelemetryEvent(
            kind=kind,
            elapsed_ms=max(0.0, (clock() - started) * 1000.0),
            process_rss_bytes=snapshot.process_rss_bytes,
            memory_pressure=snapshot.memory_pressure,
            thermal_state=snapshot.thermal_state,
            output_width=output.width if output else None,
            output_height=output.height if output else None,
            output_frames=output.frames if output else None,
            output_sha256=digest,
        )

    emit(make_event("started"))

    def progress() -> None:
        emit(make_event("progress"))

    output = runtime.generate(request, progress)
    expected_shape = (request.get("width"), request.get("height"), request.get("frames"))
    if (output.width, output.height, output.frames) != expected_shape:
        _remove_output_if_owned(output.path, output_root)
        raise ValueError(f"{backend_name} video worker output shape does not match the request")
    digest = _hash_private_output(output.path, output_root)
    emit(make_event("first_output"))
    emit(make_event("completed", output=output, digest=digest))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vllm-apple-diffusers-video-worker")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        request = consume_private_generative_request(
            arguments.request, workspace_root=arguments.workspace_root
        )
        runtime = LocalDiffusersVideoRuntime(request["candidate_id"])

        def emit(event: GenerationTelemetryEvent) -> None:
            print(json.dumps(asdict(event), sort_keys=True, separators=(",", ":")), flush=True)

        execute_diffusers_video_request(
            request, runtime, telemetry=default_worker_telemetry, emit=emit
        )
    except (ImportError, MemoryError, OSError, RuntimeError, ValueError):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
