from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from vllm_apple.vllm_metal_v2_runtime import (
    NativeV2RuntimeSelector,
    normalize_mlx_dtype,
    select_native_v2_family,
)
from vllm_apple.vllm_metal_v2_tuning import (
    V2CandidateResult,
    V2DispatchConfiguration,
    V2PagedAttentionFamily,
    V2PagedAttentionShape,
    V2ShapeTuningDecision,
    build_v2_tuning_profile,
    discover_v2_tuning_profile,
    save_v2_tuning_profile,
)


class NativeV2RuntimeTests(unittest.TestCase):
    def profile(self):
        shape = V2PagedAttentionShape(4096, 1, 1, 8, 4, 256, 16, 10)
        per_token = V2DispatchConfiguration(V2PagedAttentionFamily.PER_TOKEN, 256)
        split = V2DispatchConfiguration(
            V2PagedAttentionFamily.SPLIT_KV, 256, partition_size=512
        )
        candidates = (
            V2CandidateResult(per_token, True, 200, (200,), "a" * 64),
            V2CandidateResult(split, True, 100, (100,), "a" * 64),
        )
        return build_v2_tuning_profile(
            (V2ShapeTuningDecision(shape, split, candidates),),
            hardware_fingerprint="hardware",
            source_fingerprint="source",
        )

    def test_exact_shape_selects_profile_winner(self) -> None:
        family = select_native_v2_family(
            self.profile(),
            context_tokens=4096,
            query_tokens=1,
            sequences=1,
            query_heads=8,
            kv_heads=4,
            head_size=256,
            block_size=16,
            query_dtype="float16",
            cache_dtype="float16",
            turboquant=False,
            window_seqlen_q=1,
        )
        self.assertEqual(family, "split_kv")

    def test_shape_miss_returns_upstream_auto_dispatch(self) -> None:
        self.assertEqual(
            select_native_v2_family(self.profile(), context_tokens=1024), ""
        )

    def test_mlx_dtype_normalization(self) -> None:
        self.assertEqual(normalize_mlx_dtype("mlx.core.float16"), "float16")

    def test_first_family_hit_emits_bounded_telemetry_once(self) -> None:
        selector = NativeV2RuntimeSelector(self.profile())
        shape = {
            "context_tokens": 4096,
            "query_tokens": 1,
            "sequences": 1,
            "query_heads": 8,
            "kv_heads": 4,
            "head_size": 256,
            "block_size": 16,
            "query_dtype": "float16",
            "cache_dtype": "float16",
            "turboquant": False,
            "window_seqlen_q": 1,
        }
        with self.assertLogs("vllm", level="INFO") as logs:
            selector.family_for(**shape)
            selector.family_for(**shape)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("family=split_kv", logs.output[0])

    def test_private_profile_discovery(self) -> None:
        profile = self.profile()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "hardware" / "source" / f"{profile.profile_id}.json"
            destination.parent.mkdir(parents=True, mode=0o700)
            save_v2_tuning_profile(profile, destination)
            loaded = discover_v2_tuning_profile(
                hardware_fingerprint="hardware",
                source_fingerprint="source",
                root=root,
            )
        self.assertEqual(loaded, profile)

    def test_discovery_merges_incremental_shape_profiles(self) -> None:
        first = self.profile()
        shape = V2PagedAttentionShape(1024, 1024, 1, 8, 4, 256, 16, 10)
        per_token = V2DispatchConfiguration(V2PagedAttentionFamily.PER_TOKEN, 256)
        candidate = V2CandidateResult(per_token, True, 100, (100,), "b" * 64)
        second = build_v2_tuning_profile(
            (V2ShapeTuningDecision(shape, per_token, (candidate,)),),
            hardware_fingerprint="hardware",
            source_fingerprint="source",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "hardware" / "source"
            parent.mkdir(parents=True, mode=0o700)
            first_path = parent / f"{first.profile_id}.json"
            second_path = parent / f"{second.profile_id}.json"
            save_v2_tuning_profile(first, first_path)
            save_v2_tuning_profile(second, second_path)
            os.utime(first_path, ns=(1, 1))
            os.utime(second_path, ns=(2, 2))
            loaded = discover_v2_tuning_profile(
                hardware_fingerprint="hardware",
                source_fingerprint="source",
                root=root,
            )
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(
            {item.shape for item in loaded.decisions},
            {shape, first.decisions[0].shape},
        )


if __name__ == "__main__":
    unittest.main()
