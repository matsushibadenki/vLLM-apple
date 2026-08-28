import json
import stat
import tempfile
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.cli import build_parser
from vllm_apple.kernel_profile import ModelKernelShapeProfile, PagedAttentionShape
from vllm_apple.vllm_metal_v2_tuning import (
    V2DispatchConfiguration,
    V2PagedAttentionFamily,
    V2PagedAttentionShape,
    build_v2_tuning_profile,
    candidate_configurations,
    load_v2_tuning_profile,
    save_v2_tuning_profile,
    tune_v2_model_profile,
    tune_v2_shape,
)


class VLLMMetalV2TuningTests(unittest.TestCase):
    def prefill_shape(self, **changes):
        values = {
            "context_tokens": 4096,
            "query_tokens": 128,
            "sequences": 1,
            "query_heads": 32,
            "kv_heads": 8,
            "head_size": 128,
            "block_size": 16,
            "gpu_cores": 16,
        }
        values.update(changes)
        return V2PagedAttentionShape(**values)

    def decode_shape(self, **changes):
        changes.setdefault("query_tokens", 1)
        return self.prefill_shape(**changes)

    def test_prefill_candidates_match_upstream_nax_tiled_and_fallback(self) -> None:
        candidates = candidate_configurations(self.prefill_shape())
        self.assertEqual(
            tuple(item.family for item in candidates),
            (
                V2PagedAttentionFamily.NAX_PREFILL,
                V2PagedAttentionFamily.TILED_PREFILL,
                V2PagedAttentionFamily.PER_TOKEN,
            ),
        )
        self.assertEqual(candidates[0].threads, 128)
        self.assertEqual(
            (candidates[1].tile_query, candidates[1].tile_kv, candidates[1].threads),
            (32, 32, 128),
        )

    def test_turboquant_prefill_excludes_nax_and_tiled(self) -> None:
        candidates = candidate_configurations(self.prefill_shape(turboquant=True))
        self.assertEqual(
            candidates,
            (V2DispatchConfiguration(V2PagedAttentionFamily.PER_TOKEN, 256),),
        )

    def test_split_kv_requires_long_context_and_underfilled_gpu_grid(self) -> None:
        eligible = candidate_configurations(self.decode_shape(context_tokens=8192))
        saturated = candidate_configurations(
            self.decode_shape(context_tokens=8192, query_tokens=8, sequences=8)
        )
        short = candidate_configurations(self.decode_shape(context_tokens=512))
        self.assertEqual(eligible[-1].family, V2PagedAttentionFamily.SPLIT_KV)
        self.assertEqual(
            saturated,
            (V2DispatchConfiguration(V2PagedAttentionFamily.PER_TOKEN, 256),),
        )
        self.assertEqual(short, saturated)

    def test_correctness_gate_and_two_percent_tie_are_deterministic(self) -> None:
        latencies = {
            V2PagedAttentionFamily.NAX_PREFILL: 101,
            V2PagedAttentionFamily.TILED_PREFILL: 100,
            V2PagedAttentionFamily.PER_TOKEN: 102,
        }

        def measure(shape, configuration):
            return True, latencies[configuration.family], "a" * 64

        decision = tune_v2_shape(self.prefill_shape(), measure, samples=3)
        # All three are within 2% of 100 ns; lower-complexity fallback wins.
        self.assertEqual(decision.winner.family, V2PagedAttentionFamily.PER_TOKEN)

        def broken_measure(shape, configuration):
            return (
                configuration.family is not V2PagedAttentionFamily.TILED_PREFILL,
                1 if configuration.family is V2PagedAttentionFamily.TILED_PREFILL else 10,
                "b" * 64,
            )

        gated = tune_v2_shape(self.prefill_shape(), broken_measure, samples=1)
        self.assertNotEqual(gated.winner.family, V2PagedAttentionFamily.TILED_PREFILL)

    def test_cross_family_digest_mismatch_is_rejected(self) -> None:
        def measure(shape, configuration):
            digest = (
                "d" * 64
                if configuration.family is V2PagedAttentionFamily.NAX_PREFILL
                else "e" * 64
            )
            return True, 1, digest

        decision = tune_v2_shape(self.prefill_shape(), measure, samples=1)
        nax = next(
            item
            for item in decision.candidates
            if item.configuration.family is V2PagedAttentionFamily.NAX_PREFILL
        )
        self.assertFalse(nax.passed)
        self.assertEqual(decision.winner.family, V2PagedAttentionFamily.PER_TOKEN)

    def test_model_profile_generates_bounded_decode_and_prefill_measurements(self) -> None:
        source_shape = PagedAttentionShape(4096, 1, 32, 8, 128, 16, 256, 1_048_576)
        model_profile = ModelKernelShapeProfile(
            1, "a" * 24, "model", "Qwen", 32, 2, (source_shape,)
        )
        measured = []

        def measure(shape, configuration):
            measured.append((shape, configuration))
            return True, 100, "f" * 64

        profile = tune_v2_model_profile(
            model_profile,
            measure,
            hardware_fingerprint="hardware",
            source_fingerprint="source",
            gpu_cores=16,
            samples=1,
            maximum_shapes=2,
        )
        self.assertEqual(tuple(item.shape.query_tokens for item in profile.decisions), (1, 128))
        self.assertTrue(any(item[0].query_tokens == 1 for item in measured))
        self.assertTrue(any(item[0].query_tokens == 128 for item in measured))

    def test_native_v2_tune_cli_requires_explicit_source_and_helper(self) -> None:
        arguments = build_parser().parse_args(
            [
                "vllm-metal-v2-tune",
                "/model",
                "--source-root",
                "/source",
                "--helper",
                "/helper",
                "--contexts",
                "1024,4096",
            ]
        )
        self.assertEqual(arguments.command, "vllm-metal-v2-tune")
        self.assertEqual(arguments.maximum_shapes, 4)
        self.assertEqual(arguments.samples, 3)

    def test_profile_is_deterministic_and_schema_valid(self) -> None:
        def measure(shape, configuration):
            return True, 100 + configuration.threads, "c" * 64

        decision = tune_v2_shape(self.decode_shape(context_tokens=8192), measure)
        profile = build_v2_tuning_profile(
            (decision,),
            hardware_fingerprint="hardware",
            source_fingerprint="source",
        )
        repeated = build_v2_tuning_profile(
            (decision,),
            hardware_fingerprint="hardware",
            source_fingerprint="source",
        )
        self.assertEqual(profile.profile_id, repeated.profile_id)
        schema = json.loads(
            Path("schemas/runtime/vllm-metal-v2-tuning-profile-v1.schema.json").read_text()
        )
        validate_instance(profile.to_dict(), schema)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles" / "profile.json"
            save_v2_tuning_profile(profile, path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            loaded = load_v2_tuning_profile(
                path,
                hardware_fingerprint="hardware",
                source_fingerprint="source",
            )
            self.assertEqual(loaded, profile)
            payload = json.loads(path.read_text())
            payload["decisions"][0]["winner"]["family"] = "nax_prefill"
            path.write_text(json.dumps(payload))
            path.chmod(0o600)
            with self.assertRaises(ValueError):
                load_v2_tuning_profile(
                    path,
                    hardware_fingerprint="hardware",
                    source_fingerprint="source",
                )

        unsafe = build_v2_tuning_profile(
            (decision,),
            hardware_fingerprint="../hardware",
            source_fingerprint="source",
        )
        with self.assertRaises(ValueError):
            save_v2_tuning_profile(unsafe)


if __name__ == "__main__":
    unittest.main()
