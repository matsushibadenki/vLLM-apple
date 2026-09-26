#!/usr/bin/env python3
"""Bounded local text smoke; produces evidence, never promotes a runtime profile."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_apple.architecture_registry import inspect_architecture  # noqa: E402
from vllm_apple.hardware import detect_hardware  # noqa: E402
from vllm_apple.model import assess_model_memory_fit, inspect_model  # noqa: E402
from vllm_apple.model_integrity import build_model_integrity_manifest  # noqa: E402
from vllm_apple.qualification import save_qualification_report  # noqa: E402
from vllm_apple.types import GIB  # noqa: E402

PROMPTS = (
    ("en", "What is 1+1? Reply with only the digit."),
    ("ja", "1+1はいくつですか？数字だけで答えてください。"),
    ("zh", "1+1等于多少？只回答数字。"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    signal.alarm(120)  # Entire probe, including artifact hashing and model load.
    root = args.model.resolve(strict=True)
    if args.report.resolve().is_relative_to(root):
        raise ValueError("report must be outside the model artifact")
    descriptor = inspect_architecture(root)
    if descriptor["structure_status"] != "described":
        raise ValueError("architecture must be structurally described before smoke")
    hardware = detect_hardware()
    fit = assess_model_memory_fit(inspect_model(root), hardware, context_tokens=256)
    if (
        not hardware.is_apple_silicon or not fit.fits
        or fit.artifact_bytes > 3 * GIB or hardware.memory.available_bytes < 6 * GIB
        or hardware.memory.pressure.value != "normal"
    ):
        raise ValueError("bounded smoke memory/platform admission failed")
    before = build_model_integrity_manifest(root)
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    if not mx.metal.is_available():
        raise RuntimeError("Metal unavailable")
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(4 * GIB)
    mx.set_cache_limit(256 * 1024**2)
    started = time.monotonic()
    model, tokenizer = load(str(root), lazy=False)
    load_seconds = time.monotonic() - started
    samples = []
    for language, prompt in PROMPTS:
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
        )
        if len(tokens) > 128:
            raise ValueError("smoke prompt budget exceeded")
        output = ""
        last = None
        selected_logprobs_finite = True
        started = time.monotonic()
        for response in stream_generate(
            model, tokenizer, tokens, max_tokens=16, sampler=make_sampler(temp=0.0)
        ):
            output += response.text
            selected_logprobs_finite &= bool(mx.isfinite(response.logprobs[response.token]).item())
            last = response
        samples.append({
            "language": language,
            "prompt_tokens": len(tokens),
            "generation_tokens": last.generation_tokens if last else 0,
            "finish_reason": last.finish_reason if last else None,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "trimmed_exact_passed": output.strip() == "2",
            "selected_logprobs_finite": selected_logprobs_finite,
        })
    mx.synchronize()
    peak_allocator = mx.get_peak_memory()
    del model, tokenizer
    mx.clear_cache()
    after = build_model_integrity_manifest(root)
    unchanged = before["root_sha256"] == after["root_sha256"]
    passed = unchanged and all(
        sample["trimmed_exact_passed"] and sample["selected_logprobs_finite"]
        and sample["finish_reason"] == "stop" for sample in samples
    )
    report = {
        "schema_version": 1, "report_kind": "architecture_text_smoke",
        "model": root.name, "model_type": descriptor["model_type"],
        "config_sha256": descriptor["config_sha256"],
        "artifact_root_sha256": before["root_sha256"],
        "artifact_bytes": before["total_bytes"], "artifact_unchanged": unchanged,
        "versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm")},
        "hardware": {"soc": hardware.soc, "memory_bytes": hardware.memory.total_bytes,
                     "os": platform.mac_ver()[0]},
        "load_seconds": round(load_seconds, 6), "samples": samples,
        "peak_allocator_bytes": peak_allocator,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "passed": passed, "runtime_promoted": False,
        "limits": ["three_arithmetic_prompts_only", "no_reference_logits",
                   "no_http_batching_cancel_or_soak", "not_a_performance_benchmark"],
    }
    save_qualification_report(report, args.report)
    print(json.dumps({"passed": passed, "samples": len(samples), "runtime_promoted": False}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
