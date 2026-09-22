#!/usr/bin/env python3
"""Synthetic static-to-block-to-static Qwen transformer forward qualification."""
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
    parser.add_argument("--blocks", type=int, default=60)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--prompt-sha256")
    parser.add_argument("--decode-vae", action="store_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.blocks <= 60:
        parser.error("--blocks must be in 1..60")
    if arguments.manifest is None and (arguments.plan_sha256 or arguments.prompt_sha256):
        parser.error("handoff identity requires --manifest")
    if arguments.manifest is not None and not (arguments.plan_sha256 and arguments.prompt_sha256):
        parser.error("--manifest requires both handoff identity hashes")
    root = arguments.model.expanduser().resolve(strict=True)
    plan = inspect_mflux_qwen_transformer_staging(root)
    before = detect_hardware()
    if before.memory.pressure.value != "normal" or (
        before.memory.available_bytes < plan.static_plus_maximum_block_bytes * 8
    ):
        raise RuntimeError("Qwen transformer static smoke rejected before weight load")
    root_digest, file_count, artifact_bytes = _deployable_tree_identity(root)

    import mlx.core as mx
    import numpy as np

    if arguments.manifest is None:
        text = mx.ones((1, 13, 3584), dtype=mx.bfloat16)
        mask = mx.ones((1, 13), dtype=mx.int32)
    else:
        prompt_embeddings, prompt_mask = consume_mflux_qwen_prompt_handoff(
            arguments.manifest,
            plan_sha256=arguments.plan_sha256,
            prompt_sha256=arguments.prompt_sha256,
            sample_index=0,
        )
        text = mx.array(prompt_embeddings, dtype=mx.bfloat16)
        mask = mx.array(prompt_mask, dtype=mx.int32)
        del prompt_embeddings, prompt_mask
    text_length = text.shape[1]
    loader = QwenTransformerBlockLoader(root)
    static = loader.load_static()
    image = mx.ones((1, 4, 64), dtype=mx.bfloat16)
    timestep = mx.array([0.5], dtype=mx.bfloat16)
    started = time.perf_counter_ns()
    image = static.img_in(image)
    text = static.txt_in(static.txt_norm(text))
    time_embedding = static.time_text_embed(timestep, image)
    rotary = static.pos_embed(video_fhw=[(1, 2, 2)], txt_seq_lens=[int(mx.sum(mask).item())])
    mx.eval(image, text, time_embedding)
    samples: list[dict[str, object]] = []
    failure: str | None = None
    for index in range(arguments.blocks):
        state = detect_hardware()
        if state.memory.pressure.value != "normal" or (
            state.memory.available_bytes < plan.static_plus_maximum_block_bytes * 4
        ):
            failure = f"block {index}: memory admission failed"
            break
        block_started = time.perf_counter_ns()
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
            finite = bool(
                mx.all(mx.isfinite(output_text)).item()
                and mx.all(mx.isfinite(output_image)).item()
            )
            if not finite or output_text.shape != (1, text_length, 3072) or output_image.shape != (1, 4, 3072):
                raise ValueError("non-finite output or unexpected shape")
            image, text = output_image, output_text
            del output_image, output_text, block
            gc.collect()
            mx.clear_cache()
            after = detect_hardware()
            samples.append({
                "block_index": index,
                "elapsed_nanoseconds": max(1, time.perf_counter_ns() - block_started),
                "finite": finite,
                "memory_pressure": after.memory.pressure.value,
                "thermal_state": after.thermal_state.value,
                "peak_mlx_bytes": mx.get_peak_memory(),
                "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1024 if platform.system() == "Linux" else 1),
            })
            if after.memory.pressure.value != "normal" or after.thermal_state.value not in {
                "nominal", "fair",
            }:
                failure = f"block {index}: memory or thermal state degraded"
                break
        except Exception as error:
            failure = f"block {index}: {type(error).__name__}: {error}"
            break
    output = None
    decoded = None
    if failure is None and len(samples) == arguments.blocks:
        output = static.proj_out(static.norm_out(image, time_embedding))
        mx.eval(output)
        if output.shape != (1, 4, 64) or not bool(mx.all(mx.isfinite(output)).item()):
            failure = "static output projection is non-finite or has unexpected shape"
    if failure is None and arguments.decode_vae:
        from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator

        inventory = inspect_mflux_qwen_vae_decoder(root)
        vae_state = detect_hardware()
        if vae_state.memory.pressure.value != "normal" or (
            vae_state.memory.available_bytes < inventory["decoder_payload_bytes"] * 8
        ):
            failure = "VAE decoder memory admission failed"
        else:
            vae = load_mflux_qwen_vae_decoder(root)
            unpacked = QwenLatentCreator.unpack_latents(output, height=32, width=32)
            decoded = vae.decode(unpacked)
            mx.eval(decoded)
            if decoded.shape != (1, 3, 1, 32, 32) or not bool(mx.all(mx.isfinite(decoded)).item()):
                failure = "VAE decoded output is non-finite or has unexpected shape"
    after = detect_hardware()
    if after.memory.pressure.value != "normal":
        failure = failure or "final memory pressure is not normal"
    passed = failure is None and output is not None and (
        not arguments.decode_vae or decoded is not None
    )
    report = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_transformer_static_and_blocks_synthetic_smoke",
        "candidate_id": "qwen-image-2512",
        "artifact_root_sha256": root_digest,
        "artifact_file_count": file_count,
        "artifact_bytes": artifact_bytes,
        "requested_blocks": arguments.blocks,
        "completed_blocks": len(samples),
        "input_image_shape": [1, 4, 64],
        "input_text_shape": [1, text_length, 3584],
        "output_shape": list(output.shape) if output is not None else None,
        "output_sha256": hashlib.sha256(np.array(output.astype(mx.float32)).tobytes()).hexdigest()
        if output is not None else None,
        "vae_decoder_used": arguments.decode_vae,
        "decoded_shape": list(decoded.shape) if decoded is not None else None,
        "decoded_sha256": hashlib.sha256(np.array(decoded.astype(mx.float32)).tobytes()).hexdigest()
        if decoded is not None else None,
        "samples": samples,
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
        "uses_real_latent": False,
        "uses_synthetic_timestep": True,
        "full_transformer_qualified": False,
        "image_generation_qualified": False,
        "stores_prompt": False,
        "stores_output": False,
        "passed": passed,
    }
    save_qualification_report(report, arguments.report)
    print(json.dumps({
        "passed": passed,
        "completed_blocks": len(samples),
        "failure": failure,
        "output_shape": report["output_shape"],
        "peak_mlx_bytes": report["peak_mlx_bytes"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
