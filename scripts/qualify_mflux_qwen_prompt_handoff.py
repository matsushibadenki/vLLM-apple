#!/usr/bin/env python3
"""Verify a real Qwen-Image-2512 prompt embedding across a process boundary."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
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
from vllm_apple.mflux_qwen_prompt_handoff import (  # noqa: E402
    consume_mflux_qwen_prompt_handoff,
    save_mflux_qwen_prompt_handoff,
)
from vllm_apple.mflux_qwen_streaming_encoder import (  # noqa: E402
    build_streaming_qwen_text_encoder,
)
from vllm_apple.mflux_qwen_streaming_plan import (  # noqa: E402
    inspect_mflux_qwen_text_encoder_staging,
)
from vllm_apple.qualification import save_qualification_report  # noqa: E402

PROMPT = "A red apple on a white table."


def _peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1024 if platform.system() == "Linux" else 1
    )


def consume_main(arguments: argparse.Namespace) -> int:
    embeddings, mask = consume_mflux_qwen_prompt_handoff(
        arguments.manifest,
        plan_sha256=arguments.plan_sha256,
        prompt_sha256=arguments.prompt_sha256,
        sample_index=0,
    )
    print(json.dumps({
        "embedding_sha256": hashlib.sha256(embeddings.tobytes()).hexdigest(),
        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "embedding_shape": list(embeddings.shape),
        "mask_shape": list(mask.shape),
        "peak_process_rss_bytes": _peak_rss_bytes(),
    }, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("model", type=Path)
    qualify.add_argument("--report", type=Path, required=True)
    consume = subparsers.add_parser("consume")
    consume.add_argument("manifest", type=Path)
    consume.add_argument("plan_sha256")
    consume.add_argument("prompt_sha256")
    arguments = parser.parse_args()
    if arguments.action == "consume":
        return consume_main(arguments)

    root = arguments.model.expanduser().resolve(strict=True)
    plan = inspect_mflux_qwen_text_encoder_staging(root)
    before = detect_hardware()
    if before.memory.pressure.value != "normal" or (
        before.memory.available_bytes < plan.static_plus_maximum_layer_bytes * 4
    ):
        raise RuntimeError("Qwen-Image-2512 handoff rejected before weight load")
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

    tokenizers = TokenizerLoader.load_all(
        definitions=QwenWeightDefinition.get_tokenizers(), model_path=str(root)
    )
    layer_indices: list[int] = []
    encoder = build_streaming_qwen_text_encoder(
        root, on_layer=lambda index, _peak: layer_indices.append(index)
    )
    started = time.perf_counter_ns()
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
    del embeds, mask, encoder, tokenizers
    gc.collect()
    mx.clear_cache()

    workspace = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="qwen2512-handoff-", dir=workspace) as directory:
        private_root = Path(directory)
        manifest = save_mflux_qwen_prompt_handoff(
            private_root,
            embeddings_np,
            mask_np,
            plan_sha256=plan_sha256,
            prompt_sha256=prompt_sha256,
            sample_index=0,
        )
        del embeddings_np, mask_np
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "consume", str(manifest),
             plan_sha256, prompt_sha256],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if child.returncode or len(child.stdout) > 8192 or len(child.stderr) > 8192:
            raise RuntimeError("Qwen-Image-2512 handoff consumer failed")
        consumed = json.loads(child.stdout)
        cleanup_verified = not any(private_root.iterdir())
    after = detect_hardware()
    passed = (
        layer_indices == list(range(28))
        and consumed["embedding_sha256"] == embedding_digest
        and consumed["mask_sha256"] == mask_digest
        and consumed["embedding_shape"] == [1, 13, 3584]
        and consumed["mask_shape"] == [1, 13]
        and cleanup_verified
        and before.memory.pressure.value == "normal"
        and after.memory.pressure.value == "normal"
        and after.thermal_state.value in {"nominal", "fair"}
    )
    report = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_real_prompt_private_handoff_smoke",
        "candidate_id": "qwen-image-2512",
        "artifact_root_sha256": root_digest,
        "artifact_file_count": file_count,
        "artifact_bytes": artifact_bytes,
        "plan_sha256": plan_sha256,
        "prompt_sha256": prompt_sha256,
        "layer_execution_count": len(layer_indices),
        "embedding_sha256": embedding_digest,
        "mask_sha256": mask_digest,
        "embedding_shape": consumed["embedding_shape"],
        "mask_shape": consumed["mask_shape"],
        "encoder_peak_mlx_bytes": mx.get_peak_memory(),
        "encoder_peak_process_rss_bytes": _peak_rss_bytes(),
        "consumer_peak_process_rss_bytes": consumed["peak_process_rss_bytes"],
        "elapsed_nanoseconds": max(1, time.perf_counter_ns() - started),
        "initial_memory_pressure": before.memory.pressure.value,
        "final_memory_pressure": after.memory.pressure.value,
        "final_thermal_state": after.thermal_state.value,
        "private_cleanup_verified": cleanup_verified,
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
        "layer_execution_count": len(layer_indices),
        "private_cleanup_verified": cleanup_verified,
        "encoder_peak_mlx_bytes": report["encoder_peak_mlx_bytes"],
        "encoder_peak_process_rss_bytes": report["encoder_peak_process_rss_bytes"],
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
