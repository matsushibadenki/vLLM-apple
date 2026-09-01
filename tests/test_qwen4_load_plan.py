import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from tests import test_qwen4_adapter_loader as loader_tests
from vllm_apple.qwen4_load_plan import build_qwen4_component_load_plan
from vllm_apple.qwen4_shard_stager import stage_qwen4_shards


def packed_expert_source(root: Path) -> Path:
    source = loader_tests.Qwen4AdapterLoaderTests().source(root)
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"]["num_experts"] = 2
    config["text_config"]["num_experts_per_tok"] = 1
    config_path.write_text(json.dumps(config), encoding="utf-8")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    names_by_shard = defaultdict(list)
    for name, shard in index["weight_map"].items():
        names_by_shard[shard].append(name)
    for shard, names in names_by_shard.items():
        loader_tests._write_safetensors(
            source / shard,
            sorted(names),
            shapes={name: [2] for name in names if ".experts." in name},
        )
    return source


class Qwen4LoadPlanTests(unittest.TestCase):
    def test_builds_resident_policy_plan_without_allocating_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = loader_tests.Qwen4AdapterLoaderTests().source(root)
            output = root / "output"
            stage_qwen4_shards(source, output, maximum_output_bytes=65536)
            plan = build_qwen4_component_load_plan(
                output,
                maximum_artifact_bytes=65536,
                target_dtype="F32",
                scratch_bytes_per_tensor=3,
            )
            self.assertFalse(plan["allocates_model_or_metal"])
            self.assertEqual(plan["total_storage_bytes"], plan["resident_working_set_bytes"])
            self.assertEqual(plan["peak_tensor_reservation_bytes"], 9)
            self.assertEqual(len(plan["load_plan_id"]), 64)

    def test_rejects_unsupported_destination_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = loader_tests.Qwen4AdapterLoaderTests().source(root)
            output = root / "output"
            stage_qwen4_shards(source, output, maximum_output_bytes=65536)
            with self.assertRaisesRegex(ValueError, "dtype"):
                build_qwen4_component_load_plan(
                    output, maximum_artifact_bytes=65536, target_dtype="INT4"
                )

    def test_counts_only_active_fraction_of_packed_experts_as_resident(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = packed_expert_source(root)
            output = root / "output"
            stage_qwen4_shards(source, output, maximum_output_bytes=65536)
            plan = build_qwen4_component_load_plan(
                output, maximum_artifact_bytes=65536
            )
            self.assertTrue(plan["requires_expert_axis0_slicing"])
            self.assertLess(
                plan["component_resident_bytes"]["mixture_of_experts"],
                plan["component_storage_bytes"]["mixture_of_experts"],
            )


if __name__ == "__main__":
    unittest.main()
