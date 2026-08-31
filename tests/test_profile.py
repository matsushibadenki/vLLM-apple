import json
import stat
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from vllm_apple.profile import (
    build_profile,
    load_cached_profile,
    load_profile,
    profile_cache_key,
    save_cached_profile,
    save_profile,
)
from vllm_apple.types import ContextRecommendation, ContextTier, HardwareInfo, MemoryInfo


class ProfileTests(unittest.TestCase):
    def test_profile_is_atomically_saved_with_private_permissions(self) -> None:
        hardware = HardwareInfo(
            platform="Darwin",
            architecture="arm64",
            soc="Apple Test",
            physical_cpu_count=4,
            logical_cpu_count=4,
            gpu_core_count=None,
            memory=MemoryInfo(total_bytes=1024, available_bytes=512),
            is_apple_silicon=True,
            os_version="test",
        )
        profile = build_profile(hardware)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            saved = save_profile(profile, path)
            self.assertEqual(saved, path)
            self.assertEqual(json.loads(path.read_text())["hardware"]["soc"], "Apple Test")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(load_profile(path), profile)

    def test_legacy_profile_is_migrated_to_current_version(self) -> None:
        hardware = HardwareInfo(
            platform="Darwin",
            architecture="arm64",
            soc="Apple Test",
            physical_cpu_count=4,
            logical_cpu_count=4,
            gpu_core_count=8,
            memory=MemoryInfo(total_bytes=1024, available_bytes=512),
            is_apple_silicon=True,
            os_version="test",
        )
        payload = build_profile(hardware).to_dict()
        payload["profile_version"] = 0
        payload.pop("capabilities")
        payload.pop("metadata")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(payload))
            path.chmod(0o600)
            loaded = load_profile(path)
        self.assertEqual(loaded.profile_version, 1)
        self.assertEqual(loaded.capabilities, ())
        self.assertEqual(loaded.metadata, {})

    def test_profile_loader_rejects_future_version_and_unsafe_files(self) -> None:
        hardware = HardwareInfo(
            platform="Darwin",
            architecture="arm64",
            soc="Apple Test",
            physical_cpu_count=4,
            logical_cpu_count=4,
            gpu_core_count=None,
            memory=MemoryInfo(total_bytes=1024, available_bytes=512),
            is_apple_silicon=True,
            os_version="test",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = save_profile(build_profile(hardware), root / "profile.json")
            payload = json.loads(profile_path.read_text())
            payload["profile_version"] = 2
            profile_path.write_text(json.dumps(payload))
            profile_path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                load_profile(profile_path)

            target = save_profile(build_profile(hardware), root / "target.json")
            link = root / "linked.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                load_profile(link)

            target.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                load_profile(target)

    def test_profile_context_round_trip_is_lossless(self) -> None:
        hardware = HardwareInfo(
            platform="Darwin",
            architecture="arm64",
            soc="Apple Test",
            physical_cpu_count=4,
            logical_cpu_count=4,
            gpu_core_count=8,
            memory=MemoryInfo(total_bytes=16_384, available_bytes=12_288),
            is_apple_silicon=True,
            os_version="test",
        )
        context = ContextRecommendation(
            model_id="test/model",
            allocatable_bytes=8_192,
            os_reserve_bytes=2_048,
            safety_headroom_bytes=1_024,
            workspace_bytes=1_024,
            tiers=(ContextTier("SAFE", 2_048, 4_096),),
            limiting_factor="available_memory",
        )
        profile = build_profile(hardware, context)
        with tempfile.TemporaryDirectory() as directory:
            path = save_profile(profile, Path(directory) / "context.json")
            self.assertEqual(load_profile(path), profile)

    def test_hardware_model_profile_cache_is_private_and_identity_bound(self) -> None:
        hardware = HardwareInfo(
            platform="Darwin",
            architecture="arm64",
            soc="Apple Test",
            physical_cpu_count=4,
            logical_cpu_count=4,
            gpu_core_count=8,
            memory=MemoryInfo(total_bytes=16_384, available_bytes=12_288),
            is_apple_silicon=True,
            os_version="test",
        )
        context = ContextRecommendation(
            model_id="private/model-name",
            allocatable_bytes=8_192,
            os_reserve_bytes=2_048,
            safety_headroom_bytes=1_024,
            workspace_bytes=1_024,
            tiers=(ContextTier("SAFE", 2_048, 4_096),),
        )
        profile = build_profile(hardware, context)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            path = save_cached_profile(profile, cache)
            self.assertNotIn("private", path.name)
            self.assertEqual(len(path.stem), 24)
            self.assertEqual(stat.S_IMODE(cache.stat().st_mode), 0o700)
            self.assertEqual(load_cached_profile(hardware, context.model_id, cache), profile)
            self.assertIsNone(load_cached_profile(hardware, "other/model", cache))

    def test_profile_cache_key_changes_with_hardware_and_model(self) -> None:
        base = HardwareInfo(
            platform="Darwin",
            architecture="arm64",
            soc="Apple M4",
            physical_cpu_count=10,
            logical_cpu_count=10,
            gpu_core_count=10,
            memory=MemoryInfo(total_bytes=32 * 1024**3, available_bytes=20 * 1024**3),
            is_apple_silicon=True,
            os_version="26.0",
        )
        changed = HardwareInfo(
            platform=base.platform,
            architecture=base.architecture,
            soc=base.soc,
            physical_cpu_count=base.physical_cpu_count,
            logical_cpu_count=base.logical_cpu_count,
            gpu_core_count=base.gpu_core_count,
            memory=MemoryInfo(total_bytes=64 * 1024**3, available_bytes=40 * 1024**3),
            is_apple_silicon=True,
            os_version=base.os_version,
        )
        self.assertNotEqual(profile_cache_key(base, "model/a"), profile_cache_key(base, "model/b"))
        self.assertNotEqual(profile_cache_key(base, "model/a"), profile_cache_key(changed, "model/a"))

    def test_profile_cache_prunes_only_owned_fingerprint_files(self) -> None:
        hardware = HardwareInfo(
            platform="Darwin",
            architecture="arm64",
            soc="Apple M4",
            physical_cpu_count=10,
            logical_cpu_count=10,
            gpu_core_count=10,
            memory=MemoryInfo(total_bytes=32 * 1024**3, available_bytes=20 * 1024**3),
            is_apple_silicon=True,
            os_version="26.0",
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "vllm_apple.profile.MAXIMUM_CACHED_PROFILES", 2
        ):
            cache = Path(directory) / "cache"
            for model in ("model/a", "model/b", "model/c"):
                context = ContextRecommendation(
                    model_id=model,
                    allocatable_bytes=8_192,
                    os_reserve_bytes=2_048,
                    safety_headroom_bytes=1_024,
                    workspace_bytes=1_024,
                    tiers=(ContextTier("SAFE", 2_048, 4_096),),
                )
                save_cached_profile(build_profile(hardware, context), cache)
            unrelated = cache / "keep-me.txt"
            unrelated.write_text("not a profile")
            self.assertEqual(len(list(cache.glob("*.json"))), 2)
            self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
