import json
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.compat import BackendCompatibility
from vllm_apple.qualification_preflight import run_qualification_preflight
from vllm_apple.types import HardwareInfo, MemoryInfo, MemoryPressure


def backend(*, compatible=True):
    return BackendCompatibility(
        executable="/venv/bin/vllm",
        python_version="3.12.0",
        vllm_version="0.27.1",
        vllm_metal_version="0.3.0",
        compatible=compatible,
        issues=() if compatible else ("vllm_metal_platform_not_selected",),
        transformers_version="5.12.1",
        platform_module="vllm_metal.platform",
        platform_class="MetalPlatform",
        platform_is_cpu=False,
    )


def hardware(*, apple=True, pressure=MemoryPressure.NORMAL):
    return HardwareInfo(
        platform="Darwin" if apple else "Linux",
        architecture="arm64" if apple else "x86_64",
        soc="Apple M4" if apple else "CPU",
        physical_cpu_count=10,
        logical_cpu_count=10,
        gpu_core_count=16 if apple else None,
        memory=MemoryInfo(
            total_bytes=32 * 1024**3,
            available_bytes=20 * 1024**3,
            pressure=pressure,
        ),
        is_apple_silicon=apple,
        os_version="15.0",
    )


class QualificationPreflightTests(unittest.TestCase):
    def test_eligible_runner_report_is_schema_valid(self) -> None:
        result = run_qualification_preflight(
            Path("/venv/bin/vllm"),
            hardware_detector=hardware,
            backend_inspector=lambda executable: backend(),
        )
        self.assertTrue(result.eligible)
        schema = json.loads(
            Path("schemas/runtime/qualification-preflight-v1.schema.json").read_text()
        )
        validate_instance(result.to_dict(), schema)

    def test_cpu_fallback_and_critical_pressure_fail_closed(self) -> None:
        result = run_qualification_preflight(
            Path("/venv/bin/vllm"),
            hardware_detector=lambda: hardware(pressure=MemoryPressure.CRITICAL),
            backend_inspector=lambda executable: backend(compatible=False),
        )
        self.assertFalse(result.eligible)
        self.assertIn("memory_pressure_critical", result.issues)
        self.assertIn(
            "backend:vllm_metal_platform_not_selected",
            result.issues,
        )


if __name__ == "__main__":
    unittest.main()
