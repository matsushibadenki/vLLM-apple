import unittest

from vllm_apple.execution import ExecutionBackend
from vllm_apple.numeric_capability_probe import (
    NumericCapabilityProbeRegistry,
    NumericProbeIdentity,
)
from vllm_apple.numeric_routing import NumericFormat, NumericTensorRole


def identity(recipe="nvfp4.block16.v1"):
    return NumericProbeIdentity(
        "apple-m4", "25A1", "xcode-26", ExecutionBackend.NATIVE_MLX,
        "mlx-0.31.2", "gemv", (16, 16), NumericFormat.NVFP4_E2M1,
        NumericFormat.INT8, NumericTensorRole.WEIGHT, recipe,
    )


class NumericCapabilityProbeTests(unittest.TestCase):
    def test_exact_probe_enables_candidate_execution(self):
        registry = NumericCapabilityProbeRegistry(("nvfp4.block16.v1",))
        result = registry.probe(identity(), b"input", lambda value: value[::-1],
                                lambda value: value[::-1])
        self.assertTrue(result.passed)
        executed = registry.execute(
            identity(), b"abc", lambda value: value.upper(), lambda value: value
        )
        self.assertEqual(executed.backend, "native_mlx")
        self.assertEqual(executed.output, b"ABC")

    def test_mismatch_and_runtime_failure_quarantine_with_reference_fallback(self):
        registry = NumericCapabilityProbeRegistry(("nvfp4.block16.v1",))
        result = registry.probe(identity(), b"input", lambda value: value,
                                lambda _value: b"different")
        self.assertFalse(result.passed)
        fallback = registry.execute(
            identity(), b"abc", lambda value: value.upper(), lambda value: value
        )
        self.assertEqual((fallback.output, fallback.backend), (b"abc", "cpu_reference"))

        registry.probe(identity(), b"input", lambda value: value, lambda value: value)
        fallback = registry.execute(
            identity(), b"abc", lambda _value: (_ for _ in ()).throw(RuntimeError()),
            lambda value: value,
        )
        self.assertEqual(fallback.backend, "cpu_reference")
        self.assertIn(identity().probe_id, registry.snapshot()["quarantine"])

    def test_unknown_recipe_is_explicitly_unsupported(self):
        registry = NumericCapabilityProbeRegistry(("nvfp4.block16.v1",))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            registry.probe(identity("unknown"), b"x", lambda value: value,
                           lambda value: value)


if __name__ == "__main__":
    unittest.main()
