from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ..model import inspect_model
from .adapters import fingerprint_model_snapshot
from .evaluation import MAX_EVALUATION_SLICES, PerplexityEvaluationReport, PerplexitySlice


MAX_DATASET_BYTES = 16 * 1024 * 1024
MAX_LINE_BYTES = 72 * 1024
MAX_TEXT_BYTES = 64 * 1024
MAX_SAMPLES = 4096
MAX_TOKENS_PER_SAMPLE = 4096
MAX_TOTAL_TOKENS = 1_000_000


@dataclass(slots=True)
class _Accumulator:
    sample_count: int = 0
    token_count: int = 0
    negative_log_likelihood: float = 0.0


def evaluate_mlx_perplexity(
    model_path: Path,
    dataset_path: Path,
    *,
    maximum_samples: int,
    maximum_tokens_per_sample: int,
    maximum_total_tokens: int,
) -> PerplexityEvaluationReport:
    _validate_limit("maximum samples", maximum_samples, MAX_SAMPLES)
    _validate_limit(
        "maximum tokens per sample",
        maximum_tokens_per_sample,
        MAX_TOKENS_PER_SAMPLE,
    )
    _validate_limit("maximum total tokens", maximum_total_tokens, MAX_TOTAL_TOKENS)
    model = inspect_model(model_path)
    model_hash = fingerprint_model_snapshot(model)
    dataset = _validated_dataset_path(dataset_path)

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    started = time.monotonic()
    loaded_model, tokenizer = load(str(model.path), lazy=False)
    total = _Accumulator()
    slices: dict[tuple[str, str], _Accumulator] = {}
    dataset_digest = hashlib.sha256(b"vllm-apple-perplexity-dataset-v1\0")
    dataset_digest.update(
        json.dumps(
            {
                "maximum_samples": maximum_samples,
                "maximum_tokens_per_sample": maximum_tokens_per_sample,
                "maximum_total_tokens": maximum_total_tokens,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    with dataset.open("rb", buffering=0) as handle:
        while total.sample_count < maximum_samples and total.token_count < maximum_total_tokens:
            raw_line = handle.readline(MAX_LINE_BYTES + 1)
            if not raw_line:
                break
            if len(raw_line) > MAX_LINE_BYTES or not raw_line.endswith(b"\n"):
                raise ValueError("evaluation dataset line is missing a newline or exceeds its limit")
            record = _parse_record(raw_line)
            identity = (record["domain"], record["language"])
            if identity not in slices and len(slices) >= MAX_EVALUATION_SLICES:
                raise ValueError("evaluation dataset exceeds its slice limit")
            token_ids = list(tokenizer.encode(record["text"]))
            if len(token_ids) < 2:
                raise ValueError("evaluation sample must contain at least two tokens")
            token_ids = token_ids[:maximum_tokens_per_sample]
            remaining_targets = maximum_total_tokens - total.token_count
            if len(token_ids) - 1 > remaining_targets:
                token_ids = token_ids[: remaining_targets + 1]
            if len(token_ids) < 2:
                break
            inputs = mx.array([token_ids[:-1]])
            targets = mx.array([token_ids[1:]])
            logits = loaded_model(inputs)
            losses = nn.losses.cross_entropy(logits, targets).astype(mx.float32)
            loss_sum = mx.sum(losses)
            mx.eval(loss_sum)
            negative_log_likelihood = float(loss_sum.item())
            if not math.isfinite(negative_log_likelihood) or negative_log_likelihood < 0:
                raise ValueError("model produced an invalid evaluation loss")
            token_count = len(token_ids) - 1
            _add(total, token_count, negative_log_likelihood)
            _add(slices.setdefault(identity, _Accumulator()), token_count, negative_log_likelihood)
            dataset_digest.update(len(raw_line).to_bytes(4, "big"))
            dataset_digest.update(raw_line)
            del inputs, targets, logits, losses, loss_sum
            mx.clear_cache()

    if total.sample_count == 0 or total.token_count == 0:
        raise ValueError("evaluation dataset did not yield any scored tokens")
    elapsed_milliseconds = max(0, math.ceil((time.monotonic() - started) * 1000))
    slice_reports = tuple(
        _slice_report(identity, accumulator)
        for identity, accumulator in sorted(slices.items())
    )
    mean_nll = total.negative_log_likelihood / total.token_count
    return PerplexityEvaluationReport(
        model_path=str(model.path),
        model_hash=model_hash,
        dataset_path=str(dataset),
        dataset_fingerprint=dataset_digest.hexdigest(),
        sample_count=total.sample_count,
        token_count=total.token_count,
        mean_negative_log_likelihood=mean_nll,
        perplexity=_perplexity(mean_nll),
        elapsed_milliseconds=elapsed_milliseconds,
        peak_rss_bytes=_self_peak_rss_bytes(),
        slices=slice_reports,
    )


def _parse_record(raw_line: bytes) -> dict[str, str]:
    try:
        payload = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation dataset contains invalid JSONL") from error
    if not isinstance(payload, dict) or set(payload) != {"text", "language", "domain"}:
        raise ValueError("evaluation records require only text, language, and domain")
    if any(not isinstance(payload[name], str) or not payload[name] for name in payload):
        raise ValueError("evaluation record values must be non-empty strings")
    if len(payload["text"].encode("utf-8")) > MAX_TEXT_BYTES:
        raise ValueError("evaluation sample text exceeds its byte limit")
    if any(len(payload[name].encode("utf-8")) > 64 for name in ("language", "domain")):
        raise ValueError("evaluation labels exceed their byte limit")
    return payload


def _validated_dataset_path(path: Path) -> Path:
    candidate = path.expanduser()
    info = candidate.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or not 0 < info.st_size <= MAX_DATASET_BYTES
    ):
        raise ValueError("evaluation dataset must be a bounded owned regular file")
    return candidate.resolve(strict=True)


def _validate_limit(name: str, value: int, maximum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")


def _add(accumulator: _Accumulator, token_count: int, negative_log_likelihood: float) -> None:
    accumulator.sample_count += 1
    accumulator.token_count += token_count
    accumulator.negative_log_likelihood += negative_log_likelihood


def _slice_report(identity: tuple[str, str], accumulator: _Accumulator) -> PerplexitySlice:
    mean_nll = accumulator.negative_log_likelihood / accumulator.token_count
    return PerplexitySlice(
        domain=identity[0],
        language=identity[1],
        sample_count=accumulator.sample_count,
        token_count=accumulator.token_count,
        mean_negative_log_likelihood=mean_nll,
        perplexity=_perplexity(mean_nll),
    )


def _perplexity(mean_negative_log_likelihood: float) -> float:
    if mean_negative_log_likelihood > 50:
        raise ValueError("evaluation perplexity exceeds its supported numeric range")
    value = math.exp(mean_negative_log_likelihood)
    if not math.isfinite(value):
        raise ValueError("evaluation perplexity is not finite")
    return value


def _self_peak_rss_bytes() -> int:
    value = max(0, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    result = value if sys.platform == "darwin" else value * 1024
    if result <= 0:
        raise ValueError("evaluation peak RSS is unavailable")
    return result
