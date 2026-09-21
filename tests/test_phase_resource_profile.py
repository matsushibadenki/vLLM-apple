import json
import tempfile
import unittest
from pathlib import Path

from vllm_apple.phase_resource_profile import build_phase_resource_profile


class PhaseResourceProfileTests(unittest.TestCase):
    def test_builds_phase_specific_profile_from_bound_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpu_paths = []
            for phase, operator, latency in (
                ("prefill", "matmul", 1000), ("decode", "gemv", 800)
            ):
                path = root / f"{phase}.json"
                path.write_text(json.dumps({
                    "hardware_fingerprint": "a" * 24, "sample_count": 7,
                    "median_total_nanoseconds": latency,
                    "config": {"phase": phase, "operator": operator, "samples": 7},
                }))
                cpu_paths.append(path)
            metal = root / "metal.json"
            metal.write_text(json.dumps({
                "passed": True, "device": "Apple M4", "reports": [
                    {"operator": "gpu_gemm", "median_nanoseconds": 100},
                    {"operator": "gpu_gemv", "median_nanoseconds": 200},
                    {"operator": "attention", "median_nanoseconds": 300},
                    {"operator": "unified_memory_copy",
                     "bandwidth_bytes_per_second": 400.0},
                ],
            }))
            profile = build_phase_resource_profile(*cpu_paths, metal)
            self.assertEqual(profile.phases[0].recommended_backend, "native_metal")
            self.assertEqual(profile.phases[0].metal_speedup, 10.0)
            self.assertEqual(len(profile.to_dict()["profile_id"]), 64)

    def test_wrong_phase_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text("{}")
            with self.assertRaises(ValueError):
                build_phase_resource_profile(path, path, path)


if __name__ == "__main__":
    unittest.main()
