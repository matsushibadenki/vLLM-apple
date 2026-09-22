#!/usr/bin/env python3
"""Two-step synthetic-prompt Qwen denoising with streamed transformer and VAE."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.inspect_mflux_qwen_streaming import _deployable_tree_identity  # noqa: E402
from vllm_apple.hardware import detect_hardware  # noqa: E402
from vllm_apple.mflux_qwen_prompt_handoff import consume_mflux_qwen_prompt_handoff  # noqa: E402
from vllm_apple.mflux_qwen_transformer_loader import QwenTransformerBlockLoader  # noqa: E402
from vllm_apple.mflux_qwen_transformer_plan import (  # noqa: E402
    inspect_mflux_qwen_transformer_staging,
)
from vllm_apple.mflux_qwen_vae_loader import (  # noqa: E402
    inspect_mflux_qwen_vae_decoder,
    load_mflux_qwen_vae_decoder,
)
from vllm_apple.qualification import save_qualification_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--prompt-sha256")
    parser.add_argument("--size", type=int, choices=(32, 64, 128, 256, 512), default=32)
    parser.add_argument("--steps", type=int, default=2)
    arguments = parser.parse_args()
    if not 2 <= arguments.steps <= 20:
        parser.error("--steps must be in 2..20")
    if arguments.manifest is None and (arguments.plan_sha256 or arguments.prompt_sha256):
        parser.error("handoff identity requires --manifest")
    if arguments.manifest is not None and not (arguments.plan_sha256 and arguments.prompt_sha256):
        parser.error("--manifest requires both handoff identity hashes")
    root = arguments.model.expanduser().resolve(strict=True)
    transformer_plan = inspect_mflux_qwen_transformer_staging(root)
    vae_plan = inspect_mflux_qwen_vae_decoder(root)
    before = detect_hardware()
    minimum_available = max(
        transformer_plan.static_plus_maximum_block_bytes * 8,
        vae_plan["decoder_payload_bytes"] * 8,
    )
    if arguments.size >= 512:
        minimum_available = max(minimum_available, 20_000_000_000)
    elif arguments.size >= 256:
        minimum_available = max(minimum_available, 14_000_000_000)
    elif arguments.size >= 128:
        minimum_available = max(minimum_available, 10_000_000_000)
    if before.memory.pressure.value != "normal" or before.memory.available_bytes < minimum_available:
        raise RuntimeError("Qwen streamed denoising rejected before weight load")
    root_digest, file_count, artifact_bytes = _deployable_tree_identity(root)

    import mlx.core as mx
    import numpy as np
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
    from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer

    config = Config(
        model_config=ModelConfig.qwen_image(),
        num_inference_steps=arguments.steps,
        height=arguments.size,
        width=arguments.size,
        guidance=1.0,
        scheduler="flow_match_euler_discrete",
    )
    if arguments.manifest is None:
        text_input = mx.ones((1, 13, 3584), dtype=mx.bfloat16)
        mask = mx.ones((1, 13), dtype=mx.int32)
    else:
        prompt_embeddings, prompt_mask = consume_mflux_qwen_prompt_handoff(
            arguments.manifest,
            plan_sha256=arguments.plan_sha256,
            prompt_sha256=arguments.prompt_sha256,
            sample_index=0,
        )
        text_input = mx.array(prompt_embeddings, dtype=mx.bfloat16)
        mask = mx.array(prompt_mask, dtype=mx.int32)
        del prompt_embeddings, prompt_mask
    loader = QwenTransformerBlockLoader(root)
    static = loader.load_static()
    latents = QwenLatentCreator.create_noise(
        seed=0, height=arguments.size, width=arguments.size
    ).astype(mx.bfloat16)
    mx.eval(latents)
    initial_noise_sha256 = hashlib.sha256(np.array(latents.astype(mx.float32)).tobytes()).hexdigest()
    started = time.perf_counter_ns()
    step_reports: list[dict[str, object]] = []
    failure: str | None = None
    grid = arguments.size // 16
    rotary = static.pos_embed(
        video_fhw=[(1, grid, grid)], txt_seq_lens=[int(mx.sum(mask).item())]
    )
    for step in range(arguments.steps):
        state = detect_hardware()
        if state.memory.pressure.value != "normal" or (
            state.memory.available_bytes < transformer_plan.static_plus_maximum_block_bytes * 4
        ):
            failure = f"step {step}: memory admission failed"
            break
        image = static.img_in(config.scheduler.scale_model_input(latents, step))
        text = static.txt_in(static.txt_norm(text_input))
        timestep = QwenTransformer._compute_timestep(step, config)
        timestep = mx.broadcast_to(timestep, (1,)).astype(image.dtype)
        time_embedding = static.time_text_embed(timestep, image)
        mx.eval(image, text, time_embedding)
        completed_blocks = 0
        for index in range(60):
            block_state = detect_hardware()
            if block_state.memory.pressure.value != "normal" or (
                block_state.memory.available_bytes < transformer_plan.static_plus_maximum_block_bytes * 4
            ):
                failure = f"step {step} block {index}: memory admission failed"
                break
            try:
                block = loader.load(index)
                output_text, output_image = block(
                    hidden_states=image,
                    encoder_hidden_states=text,
                    encoder_hidden_states_mask=mask,
                    text_embeddings=time_embedding,
                    image_rotary_emb=rotary,
                    block_idx=index,
                )
                mx.eval(output_text, output_image)
                if not bool(
                    mx.all(mx.isfinite(output_text)).item()
                    and mx.all(mx.isfinite(output_image)).item()
                ):
                    raise ValueError("non-finite block output")
                image, text = output_image, output_text
                del output_image, output_text, block
                gc.collect()
                mx.clear_cache()
                completed_blocks += 1
            except Exception as error:
                failure = f"step {step} block {index}: {type(error).__name__}: {error}"
                break
        if failure is not None:
            break
        noise = static.proj_out(static.norm_out(image, time_embedding))
        latents = config.scheduler.step(noise=noise, timestep=step, latents=latents)
        mx.eval(latents)
        finite = bool(mx.all(mx.isfinite(latents)).item())
        after_step = detect_hardware()
        step_reports.append({
            "step_index": step,
            "completed_blocks": completed_blocks,
            "latent_finite": finite,
            "latent_shape": list(latents.shape),
            "memory_pressure": after_step.memory.pressure.value,
            "thermal_state": after_step.thermal_state.value,
            "peak_mlx_bytes": mx.get_peak_memory(),
        })
        if not finite or after_step.memory.pressure.value != "normal" or (
            after_step.thermal_state.value not in {"nominal", "fair"}
        ):
            failure = f"step {step}: finite or hardware-state check failed"
            break
    decoded = None
    if failure is None and len(step_reports) == arguments.steps:
        del static, loader, image, text, time_embedding, noise, rotary, text_input, mask
        gc.collect()
        mx.clear_cache()
        vae_state = detect_hardware()
        if vae_state.memory.pressure.value != "normal" or (
            vae_state.memory.available_bytes < vae_plan["decoder_payload_bytes"] * 8
        ):
            failure = "VAE memory admission failed"
        else:
            vae = load_mflux_qwen_vae_decoder(root)
            unpacked = QwenLatentCreator.unpack_latents(
                latents, height=arguments.size, width=arguments.size
            )
            decoded = vae.decode(unpacked)
            mx.eval(decoded)
            if decoded.shape != (1, 3, 1, arguments.size, arguments.size) or not bool(
                mx.all(mx.isfinite(decoded)).item()
            ):
                failure = "VAE output is non-finite or has unexpected shape"
    after = detect_hardware()
    if after.memory.pressure.value != "normal":
        failure = failure or "final memory pressure is not normal"
    passed = failure is None and decoded is not None
    report = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_streamed_denoising_smoke",
        "candidate_id": "qwen-image-2512",
        "artifact_root_sha256": root_digest,
        "artifact_file_count": file_count,
        "artifact_bytes": artifact_bytes,
        "width": arguments.size,
        "height": arguments.size,
        "seed": 0,
        "steps": arguments.steps,
        "input_text_shape": list(text_input.shape),
        "scheduler": "flow_match_euler_discrete",
        "initial_noise_sha256": initial_noise_sha256,
        "step_reports": step_reports,
        "final_latent_shape": list(latents.shape),
        "decoded_shape": list(decoded.shape) if decoded is not None else None,
        "decoded_sha256": hashlib.sha256(np.array(decoded.astype(mx.float32)).tobytes()).hexdigest()
        if decoded is not None else None,
        "failure": failure,
        "elapsed_nanoseconds": max(1, time.perf_counter_ns() - started),
        "peak_mlx_bytes": mx.get_peak_memory(),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1024 if platform.system() == "Linux" else 1),
        "initial_memory_pressure": before.memory.pressure.value,
        "final_memory_pressure": after.memory.pressure.value,
        "final_thermal_state": after.thermal_state.value,
        "mlx_version": importlib.metadata.version("mlx"),
        "mflux_version": importlib.metadata.version("mlx-gen"),
        "uses_real_prompt": arguments.manifest is not None,
        "prompt_sha256": arguments.prompt_sha256,
        "stores_prompt": False,
        "stores_output": False,
        "image_quality_qualified": False,
        "passed": passed,
    }
    save_qualification_report(report, arguments.report)
    print(json.dumps({
        "passed": passed,
        "completed_steps": len(step_reports),
        "decoded_shape": report["decoded_shape"],
        "failure": failure,
        "peak_mlx_bytes": report["peak_mlx_bytes"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
