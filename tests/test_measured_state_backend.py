import unittest

from vllm_apple.adaptive_state_allocation import AdaptiveStateAllocator, AdaptiveStateKind
from vllm_apple.measured_state_backend import MeasuredStateBackendAdapter, StatePrecisionGate
from vllm_apple.types import MemoryPressure


class MeasuredStateBackendTests(unittest.TestCase):
    def backend(self, *, maximum_error=0.02):
        return MeasuredStateBackendAdapter(StatePrecisionGate(
            maximum_error, 0.01, 0.999, 0.4,
        ))

    def test_measured_int8_promotion_and_atomic_conversion(self):
        backend = self.backend()
        values = tuple(index / 32 for index in range(-32, 33))
        backend.register("kv-1", AdaptiveStateKind.KV, values, age_seconds=10)
        measurement = backend.qualify_precision("kv-1", "int8")
        self.assertTrue(measurement.promoted)
        plan = AdaptiveStateAllocator().plan(
            backend.adaptive_state_records(), MemoryPressure.WARNING
        )
        transaction = backend.begin_adaptive_state(plan)
        transaction.commit()
        record = backend.adaptive_state_records()[0]
        self.assertEqual(record.precision, "int8")
        self.assertLess(record.resident_bytes, measurement.source_bytes)
        self.assertLessEqual(
            max(abs(left - right) for left, right in zip(values, backend.values("kv-1"))),
            measurement.maximum_absolute_error + 1e-6,
        )

    def test_failed_quality_gate_never_promotes_precision(self):
        backend = self.backend(maximum_error=0)
        backend.register("rnn-1", AdaptiveStateKind.RECURRENT, (0.1, 0.2, 0.3))
        measurement = backend.qualify_precision("rnn-1", "int8")
        self.assertFalse(measurement.promoted)
        self.assertEqual(backend.adaptive_state_records()[0].promoted_precisions, ("fp32",))

    def test_rollback_preserves_original_buffer(self):
        backend = self.backend()
        values = tuple(index / 16 for index in range(-16, 17))
        backend.register("kv-1", AdaptiveStateKind.KV, values, age_seconds=10)
        backend.qualify_precision("kv-1", "fp16")
        plan = AdaptiveStateAllocator().plan(
            backend.adaptive_state_records(), MemoryPressure.WARNING
        )
        transaction = backend.begin_adaptive_state(plan)
        transaction.rollback()
        self.assertEqual(backend.adaptive_state_records()[0].precision, "fp32")


if __name__ == "__main__":
    unittest.main()
