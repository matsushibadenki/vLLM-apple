import unittest

from vllm_apple.types import ThermalState
from vllm_apple.workload_telemetry import (
    ModalityKind,
    ModalitySample,
    RuntimeResourceSample,
    WorkloadTelemetry,
)


class WorkloadTelemetryTests(unittest.TestCase):
    def test_resource_and_modality_summaries_are_bounded(self):
        telemetry = WorkloadTelemetry(maximum_samples=2)
        for index in range(3):
            telemetry.record_resource(RuntimeResourceSample(
                index, 10.0 + index, 20.0 + index, 1000 + index,
                5.0 + index, ThermalState.NOMINAL,
            ))
            telemetry.record_modality(ModalitySample(
                ModalityKind.VIDEO, "decode", 100 + index,
                2, 3, 1000 + index, index != 1,
            ))
        report = telemetry.snapshot()
        self.assertEqual(report["resource"]["samples"], 2)
        self.assertEqual(report["resource"]["dropped"], 1)
        self.assertEqual(report["modalities"]["video"]["samples"], 2)
        self.assertEqual(report["modalities"]["video"]["failures"], 1)
        self.assertEqual(report["modality_dropped"], 1)

    def test_missing_gpu_power_and_unused_modalities_are_explicit(self):
        telemetry = WorkloadTelemetry()
        telemetry.record_resource(RuntimeResourceSample(
            1, 5.0, None, None, None, ThermalState.UNKNOWN
        ))
        report = telemetry.snapshot()
        self.assertIsNone(report["resource"]["gpu_utilization_percent"]["median"])
        self.assertEqual(report["modalities"]["audio"]["samples"], 0)

    def test_invalid_values_fail_closed(self):
        with self.assertRaises(ValueError):
            RuntimeResourceSample(1, 101, None, None, None, ThermalState.NOMINAL)
        with self.assertRaises(ValueError):
            ModalitySample(ModalityKind.AUDIO, "encode", 0, 1, 1, 0, True)


if __name__ == "__main__":
    unittest.main()
