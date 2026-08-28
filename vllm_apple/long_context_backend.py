from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any

from .long_context import LongContextEvaluationError, LongContextObservation
from .phase_probe import MAX_PROMPT_BYTES, PhaseProbeConfig, PhaseProbeError, measure_stream
from .soak import _validated_base_url


MAX_TOKENIZE_RESPONSE_BYTES = 32 * 1024 * 1024
FILLER = "The archival record contains ordinary neutral context. "


@dataclass(frozen=True, slots=True)
class TokenizerAlignedRetrievalCase:
    target_tokens: int
    actual_tokens: int
    prompt: str
    expected_text: str


class VLLMLongContextAdapter:
    """Builds tokenizer-aligned needle cases and measures them sequentially."""

    def __init__(
        self,
        probe_config: PhaseProbeConfig,
        *,
        state_bytes_per_token: int,
        load_peak_rss_bytes: int = 0,
        token_tolerance_ratio: float = 0.05,
    ) -> None:
        if state_bytes_per_token < 0 or load_peak_rss_bytes < 0:
            raise ValueError("memory values must not be negative")
        if not 0 <= token_tolerance_ratio <= 0.25:
            raise ValueError("token_tolerance_ratio must be in [0, 0.25]")
        self.config = replace(probe_config, samples=1)
        self.state_bytes_per_token = state_bytes_per_token
        self.load_peak_rss_bytes = load_peak_rss_bytes
        self.token_tolerance_ratio = token_tolerance_ratio

    def measure(self, target_tokens: int) -> LongContextObservation:
        try:
            case = self.build_case(target_tokens)
            result = measure_stream(
                replace(self.config, prompt=case.prompt), expected_text=case.expected_text
            )
        except PhaseProbeError as error:
            raise LongContextEvaluationError(error.code, str(error)) from error
        measurement = result.measurement
        intervals = measurement.token_intervals
        tpot_ms = (
            measurement.decode_ns / intervals / 1_000_000 if intervals else 0.0
        )
        tokens_per_second = (
            intervals / (measurement.decode_ns / 1_000_000_000)
            if intervals and measurement.decode_ns
            else 0.0
        )
        return LongContextObservation(
            target_tokens=target_tokens,
            actual_prompt_tokens=measurement.prompt_tokens,
            retrieval_score=1.0 if result.expected_text_matched else 0.0,
            ttft_ms=measurement.ttft_ns / 1_000_000,
            tpot_ms=tpot_ms,
            tokens_per_second=tokens_per_second,
            load_peak_rss_bytes=self.load_peak_rss_bytes,
            steady_state_rss_bytes=max(
                result.steady_memory_bytes, measurement.peak_memory_bytes
            ),
            state_bytes=measurement.prompt_tokens * self.state_bytes_per_token,
        )

    def build_case(self, target_tokens: int) -> TokenizerAlignedRetrievalCase:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        expected = f"NEEDLE-{target_tokens:07d}-A7C9"
        base = self._prompt(expected, 0)
        base_tokens = self._tokenize_count(base)
        unit_tokens = self._tokenize_count(self._prompt(expected, 1)) - base_tokens
        if unit_tokens <= 0:
            raise LongContextEvaluationError(
                "tokenizer_non_monotonic", "filler did not increase tokenizer count"
            )
        repeats = max(0, (target_tokens - base_tokens) // unit_tokens)
        best_count = base_tokens
        for _ in range(6):
            prompt = self._prompt(expected, repeats)
            if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
                raise LongContextEvaluationError(
                    "prompt_too_large", "tokenizer-aligned prompt exceeded 8 MiB"
                )
            count = self._tokenize_count(prompt)
            if abs(count - target_tokens) < abs(best_count - target_tokens):
                best_count = count
            tolerance = max(64, int(target_tokens * self.token_tolerance_ratio))
            if abs(count - target_tokens) <= tolerance:
                return TokenizerAlignedRetrievalCase(
                    target_tokens, count, prompt, expected
                )
            correction = int((target_tokens - count) / unit_tokens)
            if correction == 0:
                correction = 1 if count < target_tokens else -1
            next_repeats = max(0, repeats + correction)
            if next_repeats == repeats:
                break
            repeats = next_repeats
        raise LongContextEvaluationError(
            "token_alignment_failed",
            f"closest tokenizer count {best_count} did not match target {target_tokens}",
        )

    @staticmethod
    def _prompt(expected: str, repeats: int) -> str:
        left = repeats // 2
        right = repeats - left
        return (
            "Read the context and return only the retrieval key requested at the end.\n"
            + FILLER * left
            + f"\nThe retrieval key is {expected}.\n"
            + FILLER * right
            + "\nReturn only the retrieval key."
        )

    def _tokenize_count(self, prompt: str) -> int:
        base_url = _validated_base_url(
            self.config.base_url, allow_remote=self.config.allow_remote
        )
        body = json.dumps(
            {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "add_generation_prompt": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.session_token:
            headers["Authorization"] = f"Bearer {self.config.session_token}"
        request = urllib.request.Request(
            base_url + "/tokenize", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                data = response.read(MAX_TOKENIZE_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            error.read(64 * 1024)
            raise LongContextEvaluationError(
                "tokenize_http_error", f"tokenize endpoint returned HTTP {error.code}"
            ) from error
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            raise LongContextEvaluationError("tokenize_unavailable", str(error)) from error
        if len(data) > MAX_TOKENIZE_RESPONSE_BYTES:
            raise LongContextEvaluationError(
                "tokenize_response_too_large", "tokenize response exceeded 32 MiB"
            )
        try:
            payload: Any = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LongContextEvaluationError(
                "invalid_tokenize_response", "tokenize endpoint returned invalid JSON"
            ) from error
        count = payload.get("count") if isinstance(payload, dict) else None
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise LongContextEvaluationError(
                "invalid_tokenize_response", "tokenize response did not contain a valid count"
            )
        return count
