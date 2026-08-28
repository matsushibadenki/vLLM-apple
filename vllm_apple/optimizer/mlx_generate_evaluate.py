from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from ..model import inspect_model
from .adapters import fingerprint_model_snapshot
from .generation_evaluation import (
    MAX_GENERATION_TOKENS,
    MAX_PROMPT_TOKENS,
    GenerationEvaluationReport,
    GenerationSampleResult,
    generation_token_fingerprint,
)
from .mlx_evaluate import (
    MAX_DATASET_BYTES,
    MAX_LINE_BYTES,
    MAX_TEXT_BYTES,
    _self_peak_rss_bytes,
    _validated_dataset_path,
    _validate_limit,
)


MAX_GENERATION_SAMPLES = 64
MAX_EXPECTATIONS = 16
MAX_EXPECTATION_BYTES = 256
MAX_GENERATED_TEXT_BYTES = 256 * 1024
MAX_FILTER_VALUES = 16
MAX_LABEL_BYTES = 64


def evaluate_mlx_generation(
    model_path: Path,
    dataset_path: Path,
    *,
    maximum_samples: int,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
    use_chat_template: bool = False,
    domains: tuple[str, ...] = (),
    languages: tuple[str, ...] = (),
) -> GenerationEvaluationReport:
    _validate_limit("maximum generation samples", maximum_samples, MAX_GENERATION_SAMPLES)
    _validate_limit("maximum prompt tokens", maximum_prompt_tokens, MAX_PROMPT_TOKENS)
    _validate_limit("maximum generated tokens", maximum_new_tokens, MAX_GENERATION_TOKENS)
    selected_domains = _validated_filters("domain", domains)
    selected_languages = _validated_filters("language", languages)
    model = inspect_model(model_path)
    model_hash = fingerprint_model_snapshot(model)
    dataset = _validated_dataset_path(dataset_path)
    if dataset.stat().st_size > MAX_DATASET_BYTES:
        raise ValueError("generation dataset exceeds its byte limit")

    import mlx.core as mx
    from mlx_lm import load, stream_generate

    started = time.monotonic()
    loaded_model, tokenizer = load(str(model.path), lazy=False)
    mx.random.seed(0)
    dataset_digest = hashlib.sha256(b"vllm-apple-generation-dataset-v1\0")
    dataset_digest.update(
        json.dumps(
            {
                "maximum_samples": maximum_samples,
                "maximum_prompt_tokens": maximum_prompt_tokens,
                "maximum_new_tokens": maximum_new_tokens,
                "strategy": "greedy",
                "seed": 0,
                "prompt_format": "chat_template" if use_chat_template else "raw",
                "domains": sorted(selected_domains),
                "languages": sorted(selected_languages),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    samples: list[GenerationSampleResult] = []
    seen_ids: set[str] = set()
    with dataset.open("rb", buffering=0) as handle:
        while len(samples) < maximum_samples:
            raw_line = handle.readline(MAX_LINE_BYTES + 1)
            if not raw_line:
                break
            if len(raw_line) > MAX_LINE_BYTES or not raw_line.endswith(b"\n"):
                raise ValueError("generation dataset line is missing a newline or exceeds its limit")
            record = _parse_generation_record(raw_line)
            if selected_domains and record["domain"] not in selected_domains:
                continue
            if selected_languages and record["language"] not in selected_languages:
                continue
            if record["id"] in seen_ids:
                raise ValueError("generation sample IDs must be unique")
            seen_ids.add(record["id"])
            prompt = _formatted_prompt(tokenizer, record["prompt"], use_chat_template)
            prompt_token_count = len(tokenizer.encode(prompt))
            if not 1 <= prompt_token_count <= maximum_prompt_tokens:
                raise ValueError("generation prompt exceeds its token budget")
            model_context = model.memory_spec.model_max_context
            if (
                model_context is not None
                and prompt_token_count + maximum_new_tokens > model_context
            ):
                raise ValueError("generation request exceeds the model context limit")
            token_ids: list[int] = []
            generated_segments: list[str] = []
            generated_bytes = 0
            final_token: int | None = None
            for response in stream_generate(
                loaded_model,
                tokenizer,
                prompt,
                max_tokens=maximum_new_tokens,
            ):
                segment = response.text
                generated_bytes += len(segment.encode("utf-8"))
                if generated_bytes > MAX_GENERATED_TEXT_BYTES:
                    raise ValueError("generated text exceeds its byte limit")
                generated_segments.append(segment)
                final_token = int(response.token)
                if response.finish_reason is None:
                    token_ids.append(final_token)
            if not token_ids:
                if final_token is None:
                    raise ValueError("generation did not return a token")
                token_ids.append(final_token)
            generated_text = "".join(generated_segments)
            expectations = record["expected_any"]
            expectation_score = _expectation_score(
                generated_text,
                expectations,
                record["match"],
            )
            token_tuple = tuple(token_ids)
            samples.append(
                GenerationSampleResult(
                    sample_id=record["id"],
                    domain=record["domain"],
                    language=record["language"],
                    prompt_token_count=prompt_token_count,
                    token_ids=token_tuple,
                    output_fingerprint=generation_token_fingerprint(token_tuple),
                    expectation_score=expectation_score,
                )
            )
            dataset_digest.update(len(raw_line).to_bytes(4, "big"))
            dataset_digest.update(raw_line)
            mx.clear_cache()
    if not samples:
        raise ValueError("generation dataset did not yield any samples")
    return GenerationEvaluationReport(
        model_path=str(model.path),
        model_hash=model_hash,
        dataset_path=str(dataset),
        dataset_fingerprint=dataset_digest.hexdigest(),
        prompt_format="chat_template" if use_chat_template else "raw",
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
        elapsed_milliseconds=max(0, math.ceil((time.monotonic() - started) * 1000)),
        peak_rss_bytes=_self_peak_rss_bytes(),
        samples=tuple(samples),
    )


def _formatted_prompt(tokenizer: object, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    formatter = getattr(tokenizer, "apply_chat_template", None)
    if not callable(formatter):
        raise ValueError("tokenizer does not provide a chat template")
    formatted = formatter(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(formatted, str) or not formatted:
        raise ValueError("tokenizer returned an invalid chat prompt")
    if len(formatted.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ValueError("formatted chat prompt exceeds its byte limit")
    return formatted


def _validated_filters(name: str, values: tuple[str, ...]) -> frozenset[str]:
    if len(values) > MAX_FILTER_VALUES:
        raise ValueError(f"too many {name} filters")
    if any(
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > MAX_LABEL_BYTES
        for value in values
    ):
        raise ValueError(f"{name} filters are invalid")
    return frozenset(values)


def _expectation_score(text: str, expectations: list[str], match: str) -> float:
    normalized = text.casefold().lstrip()
    expected = tuple(value.casefold() for value in expectations)
    if match == "prefix":
        return float(any(normalized.startswith(value) for value in expected))
    return float(any(value in normalized for value in expected))


def _parse_generation_record(raw_line: bytes) -> dict[str, object]:
    try:
        payload = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("generation dataset contains invalid JSONL") from error
    required = {"id", "prompt", "language", "domain", "expected_any", "match"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("generation records have invalid fields")
    string_fields = ("id", "prompt", "language", "domain")
    if any(not isinstance(payload[name], str) or not payload[name] for name in string_fields):
        raise ValueError("generation record strings must be non-empty")
    if payload["match"] not in {"contains", "prefix"}:
        raise ValueError("generation expectation match mode is invalid")
    if len(payload["prompt"].encode("utf-8")) > MAX_TEXT_BYTES:
        raise ValueError("generation prompt exceeds its byte limit")
    if any(
        len(payload[name].encode("utf-8")) > MAX_LABEL_BYTES
        for name in ("id", "language", "domain")
    ):
        raise ValueError("generation record labels exceed their byte limit")
    expectations = payload["expected_any"]
    if (
        not isinstance(expectations, list)
        or not 1 <= len(expectations) <= MAX_EXPECTATIONS
        or any(
            not isinstance(value, str)
            or not value
            or len(value.encode("utf-8")) > MAX_EXPECTATION_BYTES
            for value in expectations
        )
    ):
        raise ValueError("generation expectations are invalid or unbounded")
    return payload
