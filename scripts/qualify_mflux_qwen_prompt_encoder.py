#!/usr/bin/env python3
"""Bounded real-prompt smoke for the streaming Qwen Image text encoder."""
from __future__ import annotations

import argparse
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
from vllm_apple.mflux_qwen_streaming_encoder import (  # noqa: E402
    build_streaming_qwen_text_encoder,
)
from vllm_apple.mflux_qwen_streaming_plan import (  # noqa: E402
    inspect_mflux_qwen_text_encoder_staging,
)
from vllm_apple.qualification import save_qualification_report  # noqa: E402

PROMPT_CASES = (
    ("en", "A red apple on a white table."),
    ("ja", "白いテーブルの上に赤いりんご。"),
    ("zh-Hans", "白色桌子上的一个红苹果。"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.model.expanduser().resolve(strict=True)
    plan = inspect_mflux_qwen_text_encoder_staging(root)
    before = detect_hardware()
    if before.memory.pressure.value != "normal" or (
        before.memory.available_bytes < plan.static_plus_maximum_layer_bytes * 4
    ):
        raise RuntimeError("real-prompt Qwen text encoder rejected before weight load")
    root_digest, file_count, artifact_bytes = _deployable_tree_identity(root)

    import mlx.core as mx
    import numpy as np
    from mflux.models.common.tokenizer import TokenizerLoader
    from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder
    from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

    tokenizer = TokenizerLoader.load_all(
        definitions=QwenWeightDefinition.get_tokenizers(), model_path=str(root)
    )["qwen"]
    layers: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []

    def record_layer(index: int, peak_bytes: int) -> None:
        snapshot = detect_hardware()
        layers.append({
            "index": index,
            "peak_mlx_bytes": peak_bytes,
            "memory_pressure": snapshot.memory.pressure.value,
            "thermal_state": snapshot.thermal_state.value,
        })

    encoder = build_streaming_qwen_text_encoder(root, on_layer=record_layer)
    for language, prompt in PROMPT_CASES:
        started = time.perf_counter_ns()
        tokenized = tokenizer.tokenize(prompt)
        mx.eval(tokenized.input_ids, tokenized.attention_mask)
        input_shape = list(tokenized.input_ids.shape)
        embeds, mask = QwenPromptEncoder.encode_positive_prompt(
            prompt=prompt,
            prompt_cache={},
            qwen_tokenizer=tokenizer,
            qwen_text_encoder=encoder,
        )
        mx.eval(embeds, mask)
        snapshot = detect_hardware()
        cases.append({
            "language": language,
            "input_shape": input_shape,
            "embedding_shape": list(embeds.shape),
            "mask_shape": list(mask.shape),
            "mask_nonzero": int(mx.sum(mask).item()),
            "embedding_finite": bool(mx.all(mx.isfinite(embeds)).item()),
            "embedding_sha256": hashlib.sha256(
                np.array(embeds.astype(mx.float32)).tobytes()
            ).hexdigest(),
            "elapsed_nanoseconds": max(1, time.perf_counter_ns() - started),
            "memory_pressure": snapshot.memory.pressure.value,
            "thermal_state": snapshot.thermal_state.value,
        })
    after = detect_hardware()
    layer_count = len(PROMPT_CASES) * 28
    passed = (
        [item["language"] for item in cases] == [item[0] for item in PROMPT_CASES]
        and [item["index"] for item in layers] == list(range(28)) * len(PROMPT_CASES)
        and all(item["embedding_shape"][0] == 1 for item in cases)
        and all(item["embedding_shape"][-1] == 3584 for item in cases)
        and all(item["mask_shape"] == item["embedding_shape"][:-1] for item in cases)
        and all(item["mask_nonzero"] > 0 and item["embedding_finite"] for item in cases)
        and len({item["embedding_sha256"] for item in cases}) == len(PROMPT_CASES)
        and all(item["memory_pressure"] == "normal" for item in cases + layers)
        and all(item["thermal_state"] in {"nominal", "fair"} for item in cases + layers)
        and after.memory.pressure.value == "normal"
    )
    report = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_streaming_real_prompt_encoder_smoke",
        "candidate_id": "qwen-image-2512",
        "artifact_root_sha256": root_digest,
        "artifact_file_count": file_count,
        "artifact_bytes": artifact_bytes,
        "case_count": len(cases),
        "layer_execution_count": len(layers),
        "expected_layer_execution_count": layer_count,
        "cases": cases,
        "layers": layers,
        "peak_mlx_bytes": mx.get_peak_memory(),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1024 if platform.system() == "Linux" else 1),
        "initial_memory_pressure": before.memory.pressure.value,
        "final_memory_pressure": after.memory.pressure.value,
        "final_thermal_state": after.thermal_state.value,
        "mlx_version": importlib.metadata.version("mlx"),
        "mflux_version": importlib.metadata.version("mlx-gen"),
        "image_generation_qualified": False,
        "stores_prompt": False,
        "stores_output": False,
        "passed": passed,
    }
    save_qualification_report(report, arguments.report)
    print(json.dumps({
        "passed": passed,
        "case_count": len(cases),
        "layer_execution_count": len(layers),
        "peak_mlx_bytes": report["peak_mlx_bytes"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
