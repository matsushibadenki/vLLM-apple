#!/usr/bin/env python3
"""Compare streamed Qwen transformer loading with the standard MLX reader."""

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
from vllm_apple.mflux_qwen_transformer_loader import QwenTransformerBlockLoader  # noqa: E402
from vllm_apple.mflux_qwen_transformer_plan import (  # noqa: E402
    inspect_mflux_qwen_transformer_staging,
)
from vllm_apple.qualification import save_qualification_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--block", type=int, choices=range(60), default=0)
    arguments = parser.parse_args()
    root = arguments.model.expanduser().resolve(strict=True)
    plan = inspect_mflux_qwen_transformer_staging(root)
    before = detect_hardware()
    minimum_available = plan.static_plus_maximum_block_bytes * 12
    if (
        before.memory.pressure.value != "normal"
        or before.memory.available_bytes < minimum_available
    ):
        raise RuntimeError("Qwen transformer parity rejected before weight load")
    root_digest, file_count, artifact_bytes = _deployable_tree_identity(root)

    import mlx.core as mx
    import numpy as np
    from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
    from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import (
        QwenTransformerBlock,
    )
    from mlx import nn
    from mlx.utils import tree_unflatten

    loader = QwenTransformerBlockLoader(root)
    image_input = mx.ones((1, 4, 64), dtype=mx.bfloat16)
    text_input = mx.ones((1, 13, 3584), dtype=mx.bfloat16)
    mask = mx.ones((1, 13), dtype=mx.int32)
    started = time.perf_counter_ns()

    def execute(static, block):
        image = static.img_in(image_input)
        text = static.txt_in(static.txt_norm(text_input))
        timestep = mx.array([0.5], dtype=image.dtype)
        time_embedding = static.time_text_embed(timestep, image)
        rotary = static.pos_embed(video_fhw=[(1, 2, 2)], txt_seq_lens=[13])
        output_text, output_image = block(
            hidden_states=image,
            encoder_hidden_states=text,
            encoder_hidden_states_mask=mask,
            text_embeddings=time_embedding,
            image_rotary_emb=rotary,
            block_idx=arguments.block,
        )
        mx.eval(output_text, output_image)
        return (
            np.array(output_text.astype(mx.float32)),
            np.array(output_image.astype(mx.float32)),
        )

    streamed_text, streamed_image = execute(loader.load_static(), loader.load(arguments.block))
    mx.clear_cache()
    gc.collect()

    prefix = f"transformer_blocks.{arguments.block}."
    wanted = {name for name in loader.weight_map if not name.startswith("transformer_blocks.")}
    wanted.update(name for name in loader.weight_map if name.startswith(prefix))
    reference_weights = []
    for shard_name in sorted({loader.weight_map[name] for name in wanted}):
        shard = mx.load(str(root / "transformer" / shard_name))
        for name in sorted(wanted):
            if loader.weight_map[name] == shard_name:
                reference_weights.append((name, shard[name]))
        del shard

    reference_static = QwenTransformer(num_layers=0)
    reference_static.norm_out.linear = nn.Linear(3072, 6144, bias=True)
    nn.quantize(
        reference_static,
        class_predicate=lambda _path, module: hasattr(module, "to_quantized"),
        bits=4,
    )
    reference_static.update(
        tree_unflatten(
            [
                (name, tensor)
                for name, tensor in reference_weights
                if not name.startswith("transformer_blocks.")
            ]
        ),
        strict=True,
    )
    reference_block = QwenTransformerBlock(dim=3072, num_heads=24, head_dim=128)
    nn.quantize(
        reference_block,
        class_predicate=lambda _path, module: hasattr(module, "to_quantized"),
        bits=4,
    )
    reference_block.update(
        tree_unflatten(
            [
                (name.removeprefix(prefix), tensor)
                for name, tensor in reference_weights
                if name.startswith(prefix)
            ]
        ),
        strict=True,
    )
    reference_text, reference_image = execute(reference_static, reference_block)

    text_difference = np.abs(streamed_text - reference_text)
    image_difference = np.abs(streamed_image - reference_image)
    text_exact = bool(np.array_equal(streamed_text, reference_text))
    image_exact = bool(np.array_equal(streamed_image, reference_image))
    after = detect_hardware()
    passed = (
        text_exact
        and image_exact
        and before.memory.pressure.value == "normal"
        and after.memory.pressure.value == "normal"
        and after.thermal_state.value in {"nominal", "fair"}
    )
    report = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_streamed_transformer_standard_reader_parity",
        "candidate_id": "qwen-image-2512",
        "artifact_root_sha256": root_digest,
        "artifact_file_count": file_count,
        "artifact_bytes": artifact_bytes,
        "block_index": arguments.block,
        "comparison": "exact-offset selective reader versus mlx.core.load",
        "static_layers_compared": ["img_in", "txt_norm", "txt_in", "time_text_embed", "pos_embed"],
        "output_text_shape": list(streamed_text.shape),
        "output_image_shape": list(streamed_image.shape),
        "output_text_exact": text_exact,
        "output_image_exact": image_exact,
        "output_text_maximum_absolute_error": float(text_difference.max()),
        "output_image_maximum_absolute_error": float(image_difference.max()),
        "output_text_sha256": hashlib.sha256(streamed_text.tobytes()).hexdigest(),
        "output_image_sha256": hashlib.sha256(streamed_image.tobytes()).hexdigest(),
        "elapsed_nanoseconds": max(1, time.perf_counter_ns() - started),
        "peak_mlx_bytes": mx.get_peak_memory(),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1024 if platform.system() == "Linux" else 1),
        "initial_memory_pressure": before.memory.pressure.value,
        "final_memory_pressure": after.memory.pressure.value,
        "final_thermal_state": after.thermal_state.value,
        "mlx_version": importlib.metadata.version("mlx"),
        "mflux_version": importlib.metadata.version("mlx-gen"),
        "stores_prompt": False,
        "stores_output": False,
        "image_quality_qualified": False,
        "passed": passed,
    }
    save_qualification_report(report, arguments.report)
    print(
        json.dumps(
            {
                "passed": passed,
                "output_text_exact": text_exact,
                "output_image_exact": image_exact,
                "peak_mlx_bytes": report["peak_mlx_bytes"],
                "peak_process_rss_bytes": report["peak_process_rss_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
