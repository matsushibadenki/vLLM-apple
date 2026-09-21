"""Bounded greedy CPU-draft/GPU-verifier probe for resident MLX-LM models.

The two models have separate streams and KV caches. Only integer token IDs cross
the device boundary; model arrays and caches remain on their owning device.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GreedyProbeResult:
    token_ids: tuple[int, ...]
    elapsed_nanoseconds: int
    accepted_draft_tokens: int = 0
    verifier_corrections: int = 0


def _validate_request(prompt: tuple[int, ...], maximum_tokens: int, draft_tokens: int) -> None:
    if (
        not prompt
        or len(prompt) > 4096
        or any(type(token) is not int or token < 0 for token in prompt)
        or type(maximum_tokens) is not int
        or not 1 <= maximum_tokens <= 64
        or type(draft_tokens) is not int
        or not 1 <= draft_tokens <= 8
    ):
        raise ValueError("invalid bounded heterogeneous speculative request")


def commit_verified_tokens(
    proposals: tuple[int, ...], verified: tuple[int, ...]
) -> tuple[int, tuple[int, ...]]:
    """Commit only the verifier-agreed prefix plus its authoritative next token."""
    if (
        not proposals
        or len(verified) != len(proposals) + 1
        or any(type(token) is not int or token < 0 for token in proposals + verified)
    ):
        raise ValueError("invalid speculative verifier token sequence")
    accepted = 0
    while accepted < len(proposals) and proposals[accepted] == verified[accepted]:
        accepted += 1
    return accepted, proposals[:accepted] + (verified[accepted],)


def _forward(model: Any, cache: Any, pending: tuple[int, ...], stream: Any, mx: Any):
    with mx.stream(stream):
        tokens = mx.array(pending, dtype=mx.uint32)
        logits = model(tokens[None], cache=cache)
        mx.eval(logits)
        if len(logits.shape) != 3 or logits.shape[1] != len(pending):
            raise RuntimeError("heterogeneous model returned an invalid logits shape")
        return logits


def greedy_verifier_probe(
    prompt: tuple[int, ...], model: Any, *, maximum_tokens: int,
) -> GreedyProbeResult:
    """Run the verifier alone with the same greedy sampling contract."""
    _validate_request(prompt, maximum_tokens, 1)
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    stream = mx.new_stream(mx.Device(mx.gpu))
    with mx.stream(stream):
        model_cache = make_prompt_cache(model)
    pending = prompt
    output: list[int] = []
    started = time.perf_counter_ns()
    for _ in range(maximum_tokens):
        logits = _forward(model, model_cache, pending, stream, mx)
        with mx.stream(stream):
            token = int(mx.argmax(logits[0, -1, :]).item())
        output.append(token)
        pending = (token,)
    return GreedyProbeResult(tuple(output), max(1, time.perf_counter_ns() - started))


def heterogeneous_greedy_probe(
    prompt: tuple[int, ...], draft_model: Any, verifier_model: Any, *,
    maximum_tokens: int, draft_tokens: int,
) -> GreedyProbeResult:
    """Speculate on CPU and verify batches on GPU without moving either cache."""
    _validate_request(prompt, maximum_tokens, draft_tokens)
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    cpu_stream = mx.new_stream(mx.Device(mx.cpu))
    gpu_stream = mx.new_stream(mx.Device(mx.gpu))
    with mx.stream(cpu_stream):
        draft_cache = make_prompt_cache(draft_model)
    with mx.stream(gpu_stream):
        verifier_cache = make_prompt_cache(verifier_model)
    draft_pending = verifier_pending = prompt
    output: list[int] = []
    accepted_total = corrections = 0
    started = time.perf_counter_ns()
    while len(output) < maximum_tokens:
        count = min(draft_tokens, maximum_tokens - len(output))
        proposals: list[int] = []
        for _ in range(count):
            logits = _forward(draft_model, draft_cache, draft_pending, cpu_stream, mx)
            with mx.stream(cpu_stream):
                token = int(mx.argmax(logits[0, -1, :]).item())
            proposals.append(token)
            draft_pending = (token,)

        verifier_input = verifier_pending + tuple(proposals)
        logits = _forward(verifier_model, verifier_cache, verifier_input, gpu_stream, mx)
        with mx.stream(gpu_stream):
            start = len(verifier_pending) - 1
            verified = tuple(
                int(mx.argmax(logits[0, start + index, :]).item())
                for index in range(count + 1)
            )
        accepted, committed = commit_verified_tokens(tuple(proposals), verified)
        accepted_total += accepted
        if accepted < count:
            corrections += 1
            with mx.stream(cpu_stream):
                draft_trim = count - accepted - 1
                if trim_prompt_cache(draft_cache, draft_trim) != draft_trim:
                    raise RuntimeError("draft KV cache did not rewind completely")
            with mx.stream(gpu_stream):
                verifier_trim = count - accepted
                if trim_prompt_cache(verifier_cache, verifier_trim) != verifier_trim:
                    raise RuntimeError("verifier KV cache did not rewind completely")
        else:
            if len(output) + len(committed) < maximum_tokens:
                # The last proposal has not yet entered the draft cache.
                _forward(draft_model, draft_cache, draft_pending, cpu_stream, mx)
        committed = committed[: maximum_tokens - len(output)]
        output.extend(committed)
        draft_pending = verifier_pending = (committed[-1],)
    return GreedyProbeResult(
        tuple(output), max(1, time.perf_counter_ns() - started),
        accepted_total, corrections,
    )
