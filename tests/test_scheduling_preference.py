from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_apple.scheduling_preference import (
    load_scheduling_preference,
    save_scheduling_preference,
)
from vllm_apple.service import RuntimeService
from vllm_apple.types import MemoryPressure, ThermalState


class SchedulingPreferenceTests(unittest.TestCase):
    def test_private_round_trip_and_runtime_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings" / "scheduling-preference.json"
            first = RuntimeService()
            self.assertEqual(first.configure_scheduling_preference(path), "automatic")
            result = first.control_scheduling_preference("low_power")
            self.assertTrue(result["accepted"])
            self.assertEqual(load_scheduling_preference(path), "low_power")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)

            restarted = RuntimeService()
            self.assertEqual(restarted.configure_scheduling_preference(path), "low_power")
            self.assertEqual(restarted.scheduler.adaptive_scheduling_snapshot()["level"], 1)
            restarted.scheduler.update_adaptive_inputs(thermal=ThermalState.CRITICAL)
            restarted.control_scheduling_preference("high_performance")
            self.assertEqual(restarted.scheduler.adaptive_scheduling_snapshot()["level"], 2)
            restarted.scheduler.update_adaptive_inputs(pressure=MemoryPressure.CRITICAL)
            self.assertEqual(restarted.scheduler.adaptive_scheduling_snapshot()["level"], 2)
            self.assertEqual(load_scheduling_preference(path), "high_performance")

    def test_invalid_storage_falls_back_to_automatic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "settings" / "scheduling-preference.json"
            save_scheduling_preference("low_power", path)
            for payload in (
                {"schema_version": 1, "preference": "unsafe"},
                {"schema_version": True, "preference": "low_power"},
                {"schema_version": 1, "preference": "low_power", "extra": 1},
            ):
                path.write_text(json.dumps(payload), encoding="utf-8")
                service = RuntimeService()
                self.assertEqual(service.configure_scheduling_preference(path), "automatic")
                self.assertEqual(service.scheduler.adaptive_scheduling_snapshot()["level"], 0)
            path.write_text("not json", encoding="utf-8")
            self.assertEqual(RuntimeService().configure_scheduling_preference(path), "automatic")
            save_scheduling_preference("low_power", path)
            path.chmod(0o644)
            self.assertEqual(RuntimeService().configure_scheduling_preference(path), "automatic")
            path.unlink()
            path.symlink_to(root / "elsewhere.json")
            self.assertEqual(RuntimeService().configure_scheduling_preference(path), "automatic")

    def test_failed_persistence_does_not_change_running_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings" / "scheduling-preference.json"
            service = RuntimeService()
            service.configure_scheduling_preference(path)
            service.control_scheduling_preference("low_power")
            with patch("vllm_apple.service.save_scheduling_preference", side_effect=OSError):
                with self.assertRaises(OSError):
                    service.control_scheduling_preference("high_performance")
            self.assertEqual(service.scheduler.adaptive_scheduling_snapshot()["level"], 1)
            self.assertEqual(
                service.scheduler.adaptive_scheduling_snapshot()["preference"], "low_power"
            )
            self.assertEqual(load_scheduling_preference(path), "low_power")


if __name__ == "__main__":
    unittest.main()
