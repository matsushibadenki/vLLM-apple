from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .model import inspect_model_architecture


MAX_CACHE_TOKENS = 16_777_216
MAX_ADVANCE_TOKENS = 8192
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class Qwen4CacheLayout:
    config_fingerprint: str
    linear_layers: int
    full_attention_layers: int
    ple_layers: int
    ngram_context_tokens: int
    gdn_conv_tokens: int
    ple_conv_tokens: int
    qsa_compress_ratio: int
    qsa_token_budget: int


@dataclass(frozen=True, slots=True)
class Qwen4CacheState:
    config_fingerprint: str
    tokens_seen: int
    token_chain_sha256: str
    ngram_tail: tuple[int, ...]
    gdn_conv_tokens: int
    ple_conv_tokens: int
    recurrent_initialized: bool
    kv_tokens_per_full_layer: int
    indexer_tokens_per_full_layer: int
    qsa_complete_blocks: int
    qsa_tail_tokens: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "config_fingerprint": self.config_fingerprint,
            "tokens_seen": self.tokens_seen,
            "token_chain_sha256": self.token_chain_sha256,
            "ngram_tail": list(self.ngram_tail),
            "gdn_conv_tokens": self.gdn_conv_tokens,
            "ple_conv_tokens": self.ple_conv_tokens,
            "recurrent_initialized": self.recurrent_initialized,
            "kv_tokens_per_full_layer": self.kv_tokens_per_full_layer,
            "indexer_tokens_per_full_layer": self.indexer_tokens_per_full_layer,
            "qsa_complete_blocks": self.qsa_complete_blocks,
            "qsa_tail_tokens": self.qsa_tail_tokens,
            "stores_tensor_values": False,
        }


def qwen4_cache_layout(config: dict[str, Any]) -> Qwen4CacheLayout:
    capability = inspect_model_architecture(config)
    if capability.architecture != "qwen4_exp":
        raise ValueError("cache contract requires Qwen4-Exp metadata")
    text = config.get("text_config", config)
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or not layer_types:
        raise ValueError("Qwen4 cache layer types are invalid")
    linear = layer_types.count("linear_attention")
    full = layer_types.count("full_attention")
    if linear + full != len(layer_types) or linear == 0 or full == 0:
        raise ValueError("Qwen4 cache layer topology is invalid")
    values = {
        "ngram_size": text.get("ngram_size"),
        "linear_conv_kernel_dim": text.get("linear_conv_kernel_dim"),
        "ple_conv_kernel_size": text.get("ple_conv_kernel_size"),
        "indexer_compress_ratio": text.get("indexer_compress_ratio"),
        "indexer_budget": text.get("indexer_budget"),
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= MAX_CACHE_TOKENS
        for value in values.values()
    ):
        raise ValueError("Qwen4 cache dimensions are invalid")
    ple_layers = text.get("ple_layer_ids", [])
    if not isinstance(ple_layers, list) or any(
        not isinstance(index, int) or not 1 <= index <= len(layer_types) for index in ple_layers
    ):
        raise ValueError("Qwen4 cache PLE layers are invalid")
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return Qwen4CacheLayout(
        config_fingerprint=hashlib.sha256(canonical).hexdigest(),
        linear_layers=linear,
        full_attention_layers=full,
        ple_layers=len(ple_layers),
        ngram_context_tokens=values["ngram_size"] - 1,
        gdn_conv_tokens=values["linear_conv_kernel_dim"] - 1,
        ple_conv_tokens=values["ple_conv_kernel_size"] - 1,
        qsa_compress_ratio=values["indexer_compress_ratio"],
        qsa_token_budget=values["indexer_budget"],
    )


def empty_qwen4_cache(layout: Qwen4CacheLayout) -> Qwen4CacheState:
    return Qwen4CacheState(
        config_fingerprint=layout.config_fingerprint,
        tokens_seen=0,
        token_chain_sha256=hashlib.sha256(b"").hexdigest(),
        ngram_tail=(),
        gdn_conv_tokens=0,
        ple_conv_tokens=0,
        recurrent_initialized=False,
        kv_tokens_per_full_layer=0,
        indexer_tokens_per_full_layer=0,
        qsa_complete_blocks=0,
        qsa_tail_tokens=0,
    )


def advance_qwen4_cache(
    layout: Qwen4CacheLayout,
    state: Qwen4CacheState,
    token_ids: list[int] | tuple[int, ...],
) -> Qwen4CacheState:
    if state.config_fingerprint != layout.config_fingerprint:
        raise ValueError("Qwen4 cache config fingerprint mismatch")
    if not _DIGEST_PATTERN.fullmatch(state.token_chain_sha256):
        raise ValueError("Qwen4 cache token digest is invalid")
    if not 1 <= len(token_ids) <= MAX_ADVANCE_TOKENS:
        raise ValueError("Qwen4 cache advance size is invalid")
    if any(
        not isinstance(token, int) or isinstance(token, bool) or not 0 <= token < 2**32
        for token in token_ids
    ):
        raise ValueError("Qwen4 cache token id is invalid")
    total = state.tokens_seen + len(token_ids)
    if total > MAX_CACHE_TOKENS:
        raise ValueError("Qwen4 cache token limit exceeded")
    digest = bytes.fromhex(state.token_chain_sha256)
    for token in token_ids:
        digest = hashlib.sha256(digest + token.to_bytes(4, "big")).digest()
    tail_source = (*state.ngram_tail, *token_ids)
    tail = tuple(tail_source[-layout.ngram_context_tokens :]) if layout.ngram_context_tokens else ()
    return Qwen4CacheState(
        config_fingerprint=layout.config_fingerprint,
        tokens_seen=total,
        token_chain_sha256=digest.hex(),
        ngram_tail=tail,
        gdn_conv_tokens=min(total, layout.gdn_conv_tokens),
        ple_conv_tokens=min(total, layout.ple_conv_tokens) if layout.ple_layers else 0,
        recurrent_initialized=total > 0,
        kv_tokens_per_full_layer=total,
        indexer_tokens_per_full_layer=total,
        qsa_complete_blocks=total // layout.qsa_compress_ratio,
        qsa_tail_tokens=total % layout.qsa_compress_ratio,
    )


def verify_qwen4_cache_state(layout: Qwen4CacheLayout, state: Qwen4CacheState) -> None:
    if state.config_fingerprint != layout.config_fingerprint:
        raise ValueError("Qwen4 cache config fingerprint mismatch")
    if not 0 <= state.tokens_seen <= MAX_CACHE_TOKENS:
        raise ValueError("Qwen4 cache token count is invalid")
    if not _DIGEST_PATTERN.fullmatch(state.token_chain_sha256):
        raise ValueError("Qwen4 cache token digest is invalid")
    if len(state.ngram_tail) != min(state.tokens_seen, layout.ngram_context_tokens):
        raise ValueError("Qwen4 cache ngram tail is invalid")
    expected = {
        "gdn": min(state.tokens_seen, layout.gdn_conv_tokens),
        "ple": min(state.tokens_seen, layout.ple_conv_tokens) if layout.ple_layers else 0,
        "blocks": state.tokens_seen // layout.qsa_compress_ratio,
        "tail": state.tokens_seen % layout.qsa_compress_ratio,
    }
    if (
        state.gdn_conv_tokens != expected["gdn"]
        or state.ple_conv_tokens != expected["ple"]
        or state.qsa_complete_blocks != expected["blocks"]
        or state.qsa_tail_tokens != expected["tail"]
        or state.kv_tokens_per_full_layer != state.tokens_seen
        or state.indexer_tokens_per_full_layer != state.tokens_seen
        or state.recurrent_initialized != (state.tokens_seen > 0)
    ):
        raise ValueError("Qwen4 cache state invariants are invalid")


def run_qwen4_cache_fixture(config: dict[str, Any]) -> dict[str, object]:
    layout = qwen4_cache_layout(config)
    tokens = (248044, 11, 22, 33, 44, 55, 66, 77, 248044)
    prefill = advance_qwen4_cache(layout, empty_qwen4_cache(layout), tokens)
    segmented = empty_qwen4_cache(layout)
    offset = 0
    for width in (4, 2, 3):
        segmented = advance_qwen4_cache(layout, segmented, tokens[offset : offset + width])
        offset += width
    decode = empty_qwen4_cache(layout)
    for token in tokens:
        decode = advance_qwen4_cache(layout, decode, (token,))
    verify_qwen4_cache_state(layout, prefill)
    verify_qwen4_cache_state(layout, segmented)
    verify_qwen4_cache_state(layout, decode)
    matches = prefill == segmented == decode
    return {
        "schema_version": 1,
        "passed": matches,
        "architecture": "qwen4_exp",
        "config_fingerprint": layout.config_fingerprint,
        "fixture_tokens": len(tokens),
        "chunkings_compared": 3,
        "token_chain_sha256": prefill.token_chain_sha256,
        "prefill_segmented_decode_match": matches,
        "linear_layers": layout.linear_layers,
        "full_attention_layers": layout.full_attention_layers,
        "ple_layers": layout.ple_layers,
        "ngram_tail_tokens": len(prefill.ngram_tail),
        "qsa_complete_blocks": prefill.qsa_complete_blocks,
        "qsa_tail_tokens": prefill.qsa_tail_tokens,
        "stores_token_ids": False,
        "stores_tensor_values": False,
        "allocates_model_or_metal": False,
    }
