from __future__ import annotations

import logging
import threading
from pathlib import Path

from .hardware import detect_hardware
from .vllm_metal_integration import inspect_vllm_metal_integration
from .vllm_metal_v2_tuning import (
    V2PagedAttentionShape,
    VLLMMetalV2TuningProfile,
    build_v2_hardware_fingerprint,
    discover_v2_tuning_profile,
)

# vLLM configures its own named logger in EngineCore subprocesses and may
# suppress unrelated library loggers. Reuse it so bounded hit telemetry is
# visible in both single-process and multiprocessing deployments.
LOGGER = logging.getLogger("vllm")


class NativeV2RuntimeSelector:
    """Lazy, fail-closed lookup for device/source-bound native-v2 winners."""

    def __init__(self, profile: VLLMMetalV2TuningProfile | None = None) -> None:
        self._lock = threading.Lock()
        self._loaded = profile is not None
        self._profile = profile
        self._shape_hits = 0
        self._shape_misses = 0
        self._family_hits = {
            "nax_prefill": 0,
            "tiled_prefill": 0,
            "per_token": 0,
            "split_kv": 0,
        }
        self._logged_families: set[str] = set()
        self._logged_misses: set[tuple[object, ...]] = set()

    def family_for(self, **shape_values: object) -> str:
        profile = self._load_once()
        family = select_native_v2_family(profile, **shape_values)
        with self._lock:
            if family:
                self._shape_hits += 1
                self._family_hits[family] += 1
                if family not in self._logged_families:
                    self._logged_families.add(family)
                    LOGGER.info(
                        "native_v2_profile_hit profile_id=%s family=%s "
                        "context_tokens=%s query_tokens=%s sequences=%s",
                        profile.profile_id if profile is not None else "none",
                        family,
                        shape_values.get("context_tokens"),
                        shape_values.get("query_tokens"),
                        shape_values.get("sequences"),
                    )
            else:
                self._shape_misses += 1
                miss = tuple(
                    shape_values.get(name)
                    for name in (
                        "context_tokens",
                        "query_tokens",
                        "sequences",
                        "query_heads",
                        "kv_heads",
                        "head_size",
                        "block_size",
                        "query_dtype",
                        "cache_dtype",
                        "turboquant",
                        "window_seqlen_q",
                    )
                )
                if profile is not None and miss not in self._logged_misses and len(
                    self._logged_misses
                ) < 8:
                    self._logged_misses.add(miss)
                    LOGGER.info(
                        "native_v2_profile_miss profile_id=%s shape=%s",
                        profile.profile_id,
                        miss,
                    )
        return family

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "profile_loaded": self._profile is not None,
                "profile_id": self._profile.profile_id if self._profile is not None else None,
                "shape_hits": self._shape_hits,
                "shape_misses": self._shape_misses,
                "family_hits": dict(self._family_hits),
            }

    def _load_once(self) -> VLLMMetalV2TuningProfile | None:
        with self._lock:
            if self._loaded:
                return self._profile
            self._loaded = True
            try:
                import vllm_metal

                package = Path(vllm_metal.__file__).resolve().parent
                inspection = inspect_vllm_metal_integration(package)
                hardware_fingerprint = build_v2_hardware_fingerprint(detect_hardware())
                self._profile = discover_v2_tuning_profile(
                    hardware_fingerprint=hardware_fingerprint,
                    source_fingerprint=inspection.source_fingerprint,
                )
                if self._profile is not None:
                    LOGGER.info(
                        "native_v2_profile_loaded profile_id=%s "
                        "hardware_fingerprint=%s source_fingerprint=%s shapes=%d",
                        self._profile.profile_id,
                        hardware_fingerprint,
                        inspection.source_fingerprint,
                        len(self._profile.decisions),
                    )
            except (ImportError, OSError, RuntimeError, ValueError):
                self._profile = None
            return self._profile


native_v2_runtime_selector = NativeV2RuntimeSelector()


def select_native_v2_family(
    profile: VLLMMetalV2TuningProfile | None, **shape_values: object
) -> str:
    if profile is None:
        return ""
    try:
        supplied = V2PagedAttentionShape(
            gpu_cores=profile.decisions[0].shape.gpu_cores,
            **shape_values,
        )
    except (TypeError, ValueError):
        return ""
    for decision in profile.decisions:
        if decision.shape == supplied:
            return decision.winner.family.value
    return ""


def normalize_mlx_dtype(value: object) -> str:
    name = str(value).lower().rsplit(".", 1)[-1]
    aliases = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}
    return aliases.get(name, name)
