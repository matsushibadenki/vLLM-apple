#!/usr/bin/env python3
"""Qualify real-prompt Qwen text encoder to streaming transformer handoff."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.inspect_mflux_qwen_streaming import _deployable_tree_identity  # noqa: E402
from vllm_apple.hardware import detect_hardware  # noqa: E402
from vllm_apple.mflux_qwen_prompt_handoff import save_mflux_qwen_prompt_handoff  # noqa: E402
from vllm_apple.mflux_qwen_streaming_encoder import build_streaming_qwen_text_encoder  # noqa: E402
from vllm_apple.mflux_qwen_streaming_plan import (  # noqa: E402
    inspect_mflux_qwen_text_encoder_staging,
)
from vllm_apple.qualification import save_qualification_report  # noqa: E402

PROMPT = "A red apple on a white table."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--denoise", action="store_true")
    arguments = parser.parse_args()
    root = arguments.model.expanduser().resolve(strict=True)
    plan = inspect_mflux_qwen_text_encoder_staging(root)
    before = detect_hardware()
    if before.memory.pressure.value != "normal" or (
        before.memory.available_bytes < plan.static_plus_maximum_layer_bytes * 4
    ):
        raise RuntimeError("Qwen prompt-to-transformer rejected before weight load")
    root_digest, file_count, artifact_bytes = _deployable_tree_identity(root)
    plan_sha256 = hashlib.sha256(
        f"qwen-image-2512-mflux|{root_digest}|text-to-image|handoff-v1".encode()
    ).hexdigest()
    prompt_sha256 = hashlib.sha256(PROMPT.encode()).hexdigest()

    import mlx.core as mx
    import numpy as np
    from mflux.models.common.tokenizer import TokenizerLoader
    from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder
    from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

    started = time.perf_counter_ns()
    tokenizers = TokenizerLoader.load_all(
        definitions=QwenWeightDefinition.get_tokenizers(), model_path=str(root)
    )
    layer_indices: list[int] = []
    encoder = build_streaming_qwen_text_encoder(
        root, on_layer=lambda index, _peak: layer_indices.append(index)
    )
    embeds, mask = QwenPromptEncoder.encode_positive_prompt(
        prompt=PROMPT,
        prompt_cache={},
        qwen_tokenizer=tokenizers["qwen"],
        qwen_text_encoder=encoder,
    )
    mx.eval(embeds, mask)
    embeddings_np = np.ascontiguousarray(np.array(embeds.astype(mx.float32)))
    mask_np = np.ascontiguousarray(np.array(mask.astype(mx.int32)))
    embedding_digest = hashlib.sha256(embeddings_np.tobytes()).hexdigest()
    mask_digest = hashlib.sha256(mask_np.tobytes()).hexdigest()
    encoder_peak_mlx = mx.get_peak_memory()
    encoder_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1024 if platform.system() == "Linux" else 1
    )
    del embeds, mask, encoder, tokenizers
    gc.collect()
    mx.clear_cache()

    workspace = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="qwen2512-transformer-", dir=workspace) as directory:
        private_root = Path(directory)
        manifest = save_mflux_qwen_prompt_handoff(
            private_root, embeddings_np, mask_np,
            plan_sha256=plan_sha256,
            prompt_sha256=prompt_sha256,
            sample_index=0,
        )
        del embeddings_np, mask_np
        child_report = private_root / "transformer-report.json"
        child_script = (
            "qualify_mflux_qwen_streamed_denoise.py" if arguments.denoise
            else "qualify_mflux_qwen_transformer_static.py"
        )
        child_args = [
            sys.executable,
            str(Path(__file__).with_name(child_script)),
            str(root),
            "--manifest", str(manifest),
            "--plan-sha256", plan_sha256,
            "--prompt-sha256", prompt_sha256,
        ]
        if not arguments.denoise:
            child_args.extend(["--blocks", "60", "--decode-vae"])
        child_args.extend(["--report", str(child_report)])
        child = subprocess.run(
            child_args,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        if child.returncode or not child_report.is_file():
            raise RuntimeError(
                f"Qwen transformer child failed ({child.returncode}): {child.stderr[-2048:]}"
            )
        transformer_report = json.loads(child_report.read_text(encoding="utf-8"))
        handoff_cleanup_verified = not manifest.exists() and not manifest.with_name(
            "qwen2512-embeddings.safetensors"
        ).exists()
    after = detect_hardware()
    passed = (
        layer_indices == list(range(28))
        and transformer_report.get("passed") is True
        and (
            len(transformer_report.get("step_reports", [])) == 2
            if arguments.denoise
            else transformer_report.get("completed_blocks") == 60
        )
        and transformer_report.get("uses_real_prompt") is True
        and transformer_report.get("prompt_sha256") == prompt_sha256
        and (arguments.denoise or transformer_report.get("vae_decoder_used") is True)
        and transformer_report.get("decoded_shape") == [1, 3, 1, 32, 32]
        and handoff_cleanup_verified
        and after.memory.pressure.value == "normal"
        and after.thermal_state.value in {"nominal", "fair"}
    )
    report = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_real_prompt_two_step_32px" if arguments.denoise
        else "mflux_qwen_image_real_prompt_to_streamed_transformer_synthetic_latent",
        "candidate_id": "qwen-image-2512",
        "artifact_root_sha256": root_digest,
        "artifact_file_count": file_count,
        "artifact_bytes": artifact_bytes,
        "plan_sha256": plan_sha256,
        "prompt_sha256": prompt_sha256,
        "embedding_sha256": embedding_digest,
        "mask_sha256": mask_digest,
        "encoder_layer_execution_count": len(layer_indices),
        "encoder_peak_mlx_bytes": encoder_peak_mlx,
        "encoder_peak_process_rss_bytes": encoder_peak_rss,
        "transformer": transformer_report,
        "private_handoff_cleanup_verified": handoff_cleanup_verified,
        "elapsed_nanoseconds": max(1, time.perf_counter_ns() - started),
        "final_memory_pressure": after.memory.pressure.value,
        "final_thermal_state": after.thermal_state.value,
        "uses_real_prompt": True,
        "uses_real_latent": False,
        "denoise_steps": 2 if arguments.denoise else 0,
        "image_generation_qualified": False,
        "stores_prompt": False,
        "stores_output": False,
        "passed": passed,
    }
    save_qualification_report(report, arguments.report)
    print(json.dumps({
        "passed": passed,
        "encoder_layer_execution_count": len(layer_indices),
        "transformer_completed_blocks": sum(
            step.get("completed_blocks", 0) for step in transformer_report.get("step_reports", [])
        ) if arguments.denoise else transformer_report.get("completed_blocks"),
        "transformer_peak_mlx_bytes": transformer_report.get("peak_mlx_bytes"),
        "transformer_peak_process_rss_bytes": transformer_report.get("peak_process_rss_bytes"),
        "private_handoff_cleanup_verified": handoff_cleanup_verified,
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
