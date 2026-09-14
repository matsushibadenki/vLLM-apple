import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm_apple.daemon import (
    control_device_placement_files,
    reload_device_placement_async,
    restore_startup_device_placement,
)
from vllm_apple.device_placement import default_device_placement_paths


class DaemonDevicePlacementTests(unittest.TestCase):
    def test_profile_specific_default_paths_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            current, last_good = default_device_placement_paths(
                "m4-test", "environment-test", application_support=Path(directory)
            )
        self.assertNotEqual(current, last_good)
        self.assertEqual(current.parent.name, "device-placement")
        self.assertTrue(last_good.name.endswith(".last-known-good.json"))

    def test_restore_records_last_known_good_rollback(self):
        service = Mock()
        service.install_device_placement_plan.return_value = True
        plan = SimpleNamespace(plan_id="a" * 24)
        with patch(
            "vllm_apple.daemon.load_device_placement_with_fallback",
            return_value=(plan, "last_known_good"),
        ):
            applied = restore_startup_device_placement(
                service, "m4-test", "environment-test"
            )
        self.assertTrue(applied)
        service.install_device_placement_plan.assert_called_once_with(plan)
        service.events.publish.assert_any_call(
            "runtime.device_placement.restore",
            {
                "status": "applied",
                "source": "last_known_good",
                "plan_id": "a" * 24,
            },
        )

    def test_missing_or_invalid_plan_is_normal_fail_closed_state(self):
        for error, status in (
            (FileNotFoundError(), "not_found"),
            (ValueError("invalid"), "rejected"),
        ):
            with self.subTest(status=status):
                service = Mock()
                with patch(
                    "vllm_apple.daemon.load_device_placement_with_fallback",
                    side_effect=error,
                ):
                    self.assertFalse(restore_startup_device_placement(
                        service, "m4-test", "environment-test"
                    ))
                service.install_device_placement_plan.assert_not_called()
                service.events.publish.assert_called_once_with(
                    "runtime.device_placement.restore",
                    {"status": status, "source": None, "plan_id": None},
                )

    def test_async_reload_runs_outside_signal_handler(self):
        service = Mock()
        with patch(
            "vllm_apple.daemon.restore_startup_device_placement",
            return_value=True,
        ) as restore:
            worker = reload_device_placement_async(
                service, ("m4-test", "environment-test")
            )
            self.assertIsNotNone(worker)
            worker.join(timeout=1)
        restore.assert_called_once_with(
            service,
            "m4-test",
            "environment-test",
            application_support=None,
        )
        service.events.publish.assert_called_once_with(
            "runtime.device_placement.reload",
            {"status": "accepted", "reason": None},
        )

    def test_reload_before_probe_is_rejected_without_thread(self):
        service = Mock()
        self.assertIsNone(reload_device_placement_async(service, None))
        service.events.publish.assert_called_once_with(
            "runtime.device_placement.reload",
            {"status": "unavailable", "reason": "runtime_probe_not_ready"},
        )

    def test_explicit_rollback_loads_only_last_known_good(self):
        service = Mock()
        service.install_device_placement_plan.return_value = False
        plan = SimpleNamespace(plan_id="b" * 24)
        with patch(
            "vllm_apple.daemon.load_device_placement_plan", return_value=plan
        ) as load:
            accepted = control_device_placement_files(
                service, ("m4-test", "environment-test"), "rollback"
            )
        self.assertTrue(accepted)
        self.assertTrue(load.call_args.args[0].name.endswith(".last-known-good.json"))
        service.events.publish.assert_called_once_with(
            "runtime.device_placement.control",
            {
                "action": "rollback",
                "status": "deferred",
                "source": "last_known_good",
                "plan_id": "b" * 24,
            },
        )


if __name__ == "__main__":
    unittest.main()
