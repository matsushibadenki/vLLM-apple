#!/usr/bin/env python3
"""Encode one fixed Qwen prompt into a private single-use handoff."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.inspect_mflux_qwen_streaming import _deployable_tree_identity  # noqa: E402
from vllm_apple.hardware import detect_hardware  # noqa: E402
from vllm_apple.mflux_qwen_prompt_handoff import save_mflux_qwen_prompt_handoff  # noqa: E402
from vllm_apple.mflux_qwen_streaming_encoder import (  # noqa: E402
    build_streaming_qwen_text_encoder,
)
from vllm_apple.mflux_qwen_streaming_plan import (  # noqa: E402
    inspect_mflux_qwen_text_encoder_staging,
)

PROMPTS = {
    "positive": "A red apple on a white table.",
    "negative": "blurry, distorted, low quality",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--role", choices=tuple(PROMPTS), required=True)
    arguments = parser.parse_args()
    root = arguments.model.expanduser().resolve(strict=True)
    handoff_root = arguments.handoff_root.expanduser().resolve(strict=True)
    workspace = Path(__file__).resolve().parents[1]
    if not handoff_root.is_relative_to(workspace):
        raise ValueError("Qwen prompt phase handoff must be inside the workspace")
    plan = inspect_mflux_qwen_text_encoder_staging(root)
    before = detect_hardware()
    if (
        before.memory.pressure.value != "normal"
        or before.memory.available_bytes < plan.static_plus_maximum_layer_bytes * 4
    ):
        raise RuntimeError("Qwen prompt phase rejected before weight load")
    root_digest, _, _ = _deployable_tree_identity(root)
    plan_sha256 = hashlib.sha256(
        f"qwen-image-2512-mflux|{root_digest}|text-to-image|handoff-v1".encode()
    ).hexdigest()
    prompt = PROMPTS[arguments.role]
    prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()

    import mlx.core as mx
    import numpy as np
    from mflux.models.common.tokenizer import TokenizerLoader
    from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import (
        QwenPromptEncoder,
    )
    from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

    started = time.perf_counter_ns()
    tokenizers = TokenizerLoader.load_all(
        definitions=QwenWeightDefinition.get_tokenizers(), model_path=str(root)
    )
    layers: list[int] = []
    encoder = build_streaming_qwen_text_encoder(
        root, on_layer=lambda index, _peak: layers.append(index)
    )
    embeds, mask = QwenPromptEncoder.encode_positive_prompt(
        prompt=prompt,
        prompt_cache={},
        qwen_tokenizer=tokenizers["qwen"],
        qwen_text_encoder=encoder,
    )
    mx.eval(embeds, mask)
    embeddings = np.ascontiguousarray(np.array(embeds.astype(mx.float32)))
    prompt_mask = np.ascontiguousarray(np.array(mask.astype(mx.int32)))
    embedding_sha256 = hashlib.sha256(embeddings.tobytes()).hexdigest()
    mask_sha256 = hashlib.sha256(prompt_mask.tobytes()).hexdigest()
    peak_mlx_bytes = mx.get_peak_memory()
    peak_process_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1024 if platform.system() == "Linux" else 1
    )
    manifest = save_mflux_qwen_prompt_handoff(
        handoff_root,
        embeddings,
        prompt_mask,
        plan_sha256=plan_sha256,
        prompt_sha256=prompt_sha256,
        sample_index=0,
    )
    del embeds, mask, embeddings, prompt_mask, encoder, tokenizers
    gc.collect()
    mx.clear_cache()
    after = detect_hardware()
    passed = (
        layers == list(range(28))
        and manifest.is_file()
        and before.memory.pressure.value == "normal"
        and after.memory.pressure.value == "normal"
        and after.thermal_state.value in {"nominal", "fair"}
    )
    result = {
        "schema_version": 1,
        "role": arguments.role,
        "plan_sha256": plan_sha256,
        "prompt_sha256": prompt_sha256,
        "embedding_sha256": embedding_sha256,
        "mask_sha256": mask_sha256,
        "layer_execution_count": len(layers),
        "peak_mlx_bytes": peak_mlx_bytes,
        "peak_process_rss_bytes": peak_process_rss_bytes,
        "elapsed_nanoseconds": max(1, time.perf_counter_ns() - started),
        "final_memory_pressure": after.memory.pressure.value,
        "final_thermal_state": after.thermal_state.value,
        "stores_prompt": False,
        "passed": passed,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
