import unittest

from vllm_apple.qwen3_vl_process_factory import (
    create_qwen3_vl_process_delegate,
)


def config():
    return {
        "model": "/model",
        "revision": "a" * 40,
        "graph_id": "b" * 64,
        "compiled_models": [f"/model-{index}" for index in range(5)],
        "model_id": "qwen3-vl",
        "hardware_profile": "apple-m4",
        "environment_profile": "macos-test",
        "unified_memory_bytes": 4_000_000_000,
        "cpu_threads": 8,
        "gpu_command_queues": 2,
        "bandwidth_slots": 2,
    }


class Qwen3VLProcessFactoryTests(unittest.TestCase):
    def test_schema_rejects_unknown_missing_and_invalid_identity(self):
        value = config()
        value["unknown"] = True
        with self.assertRaisesRegex(ValueError, "schema"):
            create_qwen3_vl_process_delegate(value)
        value = config()
        value["graph_id"] = "not-a-digest"
        with self.assertRaisesRegex(ValueError, "graph identity"):
            create_qwen3_vl_process_delegate(value)

    def test_resource_capacity_and_exact_model_count_are_bounded(self):
        value = config()
        value["compiled_models"] = value["compiled_models"][:4]
        with self.assertRaisesRegex(ValueError, "five compiled"):
            create_qwen3_vl_process_delegate(value)
        value = config()
        value["bandwidth_slots"] = 0
        with self.assertRaisesRegex(ValueError, "resource capacity"):
            create_qwen3_vl_process_delegate(value)


if __name__ == "__main__":
    unittest.main()
