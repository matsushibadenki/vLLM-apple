#!/usr/bin/env python3
"""Qualify Gemma 2 draft + Gemma 3 verifier through MLX-LM speculative decoding."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_apple.hardware import detect_hardware  # noqa: E402
from vllm_apple.heterogeneous_mlx_speculative import (  # noqa: E402
    greedy_verifier_probe,
    heterogeneous_greedy_probe,
)
from vllm_apple.qualification import save_qualification_report  # noqa: E402

PROMPTS = (
    "Continue with one short sentence: The quiet harbor",
    "短い一文で続けてください：静かな港は",
    "请用一个短句续写：安静的港口",
)


def _tree_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256(b"vllm-apple-speculative-model-tree-v1\0")
    count = total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("model tree must not contain symlinks")
        if not path.is_file() or "/.git/" in path.as_posix() or "/.cache/" in path.as_posix():
            continue
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as source:
            while chunk := source.read(4 * 1024 * 1024):
                digest.update(chunk)
        count += 1
        total += size
    if count < 1:
        raise ValueError("model tree is empty")
    return digest.hexdigest(), count, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--verifier-model", type=Path, required=True)
    parser.add_argument("--draft-revision", required=True)
    parser.add_argument("--verifier-revision", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--num-draft-tokens", type=int, default=3)
    parser.add_argument("--heterogeneous", action="store_true")
    arguments = parser.parse_args()
    if not 4 <= arguments.max_tokens <= 64 or not 1 <= arguments.num_draft_tokens <= 8:
        raise ValueError("invalid bounded generation settings")
    for revision in (arguments.draft_revision, arguments.verifier_revision):
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError("model revisions must be lowercase commit SHAs")
    draft_path = arguments.draft_model.expanduser().resolve(strict=True)
    verifier_path = arguments.verifier_model.expanduser().resolve(strict=True)
    draft_digest, draft_files, draft_bytes = _tree_digest(draft_path)
    verifier_digest, verifier_files, verifier_bytes = _tree_digest(verifier_path)

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    if arguments.heterogeneous:
        mx.set_default_device(mx.Device(mx.cpu))
        draft, draft_tokenizer = load(str(draft_path), lazy=False)
        mx.set_default_device(mx.Device(mx.gpu))
        verifier, tokenizer = load(str(verifier_path), lazy=False)
    else:
        verifier, tokenizer = load(str(verifier_path), lazy=False)
        draft, draft_tokenizer = load(str(draft_path), lazy=False)
    for prompt in PROMPTS:
        if tokenizer.encode(prompt) != draft_tokenizer.encode(prompt):
            raise RuntimeError("draft and verifier tokenizer IDs differ")
    sampler = make_sampler(temp=0.0)

    def generate(
        prompt: str, *, speculative: bool
    ) -> tuple[tuple[int, ...], int, int, int, float, int]:
        if arguments.heterogeneous:
            prompt_ids = tuple(tokenizer.encode(prompt))
            if speculative:
                result = heterogeneous_greedy_probe(
                    prompt_ids, draft, verifier,
                    maximum_tokens=arguments.max_tokens,
                    draft_tokens=arguments.num_draft_tokens,
                )
            else:
                result = greedy_verifier_probe(
                    prompt_ids, verifier, maximum_tokens=arguments.max_tokens,
                )
            return (
                result.token_ids, result.elapsed_nanoseconds,
                result.accepted_draft_tokens, len(result.token_ids),
                float(mx.get_peak_memory()) / (1024 ** 3),
                result.verifier_corrections,
            )
        mx.synchronize()
        started = time.perf_counter_ns()
        responses = stream_generate(
            verifier,
            tokenizer,
            prompt,
            draft_model=draft if speculative else None,
            max_tokens=arguments.max_tokens,
            num_draft_tokens=arguments.num_draft_tokens,
            sampler=sampler,
        )
        tokens: list[int] = []
        accepted = 0
        peak_memory = 0.0
        for response in responses:
            tokens.append(int(response.token))
            accepted += int(bool(response.from_draft))
            peak_memory = max(peak_memory, float(response.peak_memory))
        mx.synchronize()
        elapsed = max(1, time.perf_counter_ns() - started)
        # MLX-LM may emit one terminal EOS token in addition to max_tokens.
        if not tokens or len(tokens) > arguments.max_tokens + 1:
            raise RuntimeError(
                f"invalid generated token count: {len(tokens)} (limit {arguments.max_tokens})"
            )
        return tuple(tokens), elapsed, accepted, len(tokens), peak_memory, 0

    # Compile/warm both paths outside the measured samples.
    generate(PROMPTS[0], speculative=False)
    generate(PROMPTS[0], speculative=True)
    samples = []
    baseline_latencies: list[int] = []
    speculative_latencies: list[int] = []
    outputs_match = True
    maximum_peak_memory_gib = 0.0
    for index, prompt in enumerate(PROMPTS):
        baseline = generate(prompt, speculative=False)
        speculative = generate(prompt, speculative=True)
        baseline_latencies.append(baseline[1])
        speculative_latencies.append(speculative[1])
        outputs_match &= baseline[0] == speculative[0]
        maximum_peak_memory_gib = max(maximum_peak_memory_gib, baseline[4], speculative[4])
        samples.append({
            "sample_index": index,
            "baseline_latency_nanoseconds": baseline[1],
            "speculative_latency_nanoseconds": speculative[1],
            "token_count": baseline[3],
            "accepted_draft_tokens": speculative[2],
            "verifier_corrections": speculative[5],
            "output_match": baseline[0] == speculative[0],
            "output_sha256": hashlib.sha256(
                b"".join(token.to_bytes(8, "big") for token in baseline[0])
            ).hexdigest(),
        })
    baseline_median = int(statistics.median(baseline_latencies))
    speculative_median = int(statistics.median(speculative_latencies))
    improvement = 1.0 - speculative_median / baseline_median
    hardware = detect_hardware()
    payload = {
        "schema_version": 1,
        "scope": "mlx_native_speculative_gemma_candidate_qualification",
        "candidate_backend": (
            "native_mlx_cpu_draft_gpu_verify" if arguments.heterogeneous
            else "native_mlx_gpu_draft_and_verify"
        ),
        "eligible_for_cpu_or_coreml_draft_profile": arguments.heterogeneous,
        "draft_device": "cpu" if arguments.heterogeneous else "gpu",
        "verify_device": "gpu",
        "draft_model": str(draft_path.name),
        "draft_model_revision": arguments.draft_revision,
        "draft_model_license": "gemma",
        "draft_model_sha256": draft_digest,
        "draft_model_file_count": draft_files,
        "draft_model_bytes": draft_bytes,
        "verifier_model": str(verifier_path.name),
        "verifier_model_revision": arguments.verifier_revision,
        "verifier_model_license": "gemma",
        "verifier_model_sha256": verifier_digest,
        "verifier_model_file_count": verifier_files,
        "verifier_model_bytes": verifier_bytes,
        "sample_count": len(samples),
        "max_tokens": arguments.max_tokens,
        "num_draft_tokens": arguments.num_draft_tokens,
        "baseline_median_latency_nanoseconds": baseline_median,
        "speculative_median_latency_nanoseconds": speculative_median,
        "median_improvement_fraction": improvement,
        "outputs_match": outputs_match,
        "maximum_peak_memory_gib": maximum_peak_memory_gib,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1024 if platform.system() == "Linux" else 1),
        "memory_pressure": hardware.memory.pressure.value,
        "thermal_state": hardware.thermal_state.value,
        "soc": hardware.soc,
        "os_version": hardware.os_version,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "samples": samples,
        "stores_prompt": False,
        "stores_output": False,
    }
    payload["performance_gate_passed"] = improvement >= 0.05
    payload["passed"] = (
        outputs_match
        and payload["performance_gate_passed"]
        and payload["eligible_for_cpu_or_coreml_draft_profile"]
        and hardware.memory.pressure.value == "normal"
        and hardware.thermal_state.value in {"nominal", "fair"}
    )
    save_qualification_report(payload, arguments.report)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
