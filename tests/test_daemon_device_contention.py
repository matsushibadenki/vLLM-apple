import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests.test_scheduler import hardware
from vllm_apple.daemon import (
    control_device_contention_files,
    restore_startup_contention_profile,
)
from vllm_apple.device_contention import ContentionProfile
from vllm_apple.device_resources import (
    BandwidthContentionEvidence,
    contention_profile_id,
)
from vllm_apple.execution import ExecutionBackend


class DaemonDeviceContentionTests(unittest.TestCase):
    def service(self):
        device = hardware()
        return Mock(profile=Mock(hardware=device)), contention_profile_id(
            device.soc, device.os_version, device.architecture
        )

    def test_restore_installs_matching_profile_and_publishes_bounded_event(self):
        service, profile_id = self.service()
        service.install_contention_profile.return_value = True
        evidence = BandwidthContentionEvidence(
            profile_id, ExecutionBackend.CPU, ExecutionBackend.NATIVE_MLX,
            100, 90, 3, True,
        )
        profile = ContentionProfile(profile_id, (evidence,))
        with patch(
            "vllm_apple.daemon.load_contention_profile_with_fallback",
            return_value=(profile, "current"),
        ):
            self.assertTrue(restore_startup_contention_profile(service))
        service.install_contention_profile.assert_called_once_with(profile)
        service.events.publish.assert_called_once_with(
            "runtime.device_contention.restore",
            {"status": "applied", "profile_id": profile_id,
             "qualified_pairs": 1, "source": "current"},
        )

    def test_missing_and_invalid_profiles_fail_closed(self):
        for error, status in ((FileNotFoundError(), "not_found"),
                              (ValueError("invalid"), "rejected")):
            with self.subTest(status=status):
                service, profile_id = self.service()
                with patch(
                    "vllm_apple.daemon.load_contention_profile_with_fallback",
                    side_effect=error,
                ):
                    with tempfile.TemporaryDirectory() as directory:
                        self.assertFalse(restore_startup_contention_profile(
                            service, application_support=Path(directory)
                        ))
                service.install_contention_profile.assert_not_called()
                service.events.publish.assert_called_once_with(
                    "runtime.device_contention.restore",
                    {"status": status, "profile_id": profile_id,
                     "qualified_pairs": 0},
                )

    def test_rollback_loads_last_known_good_only(self):
        service, profile_id = self.service()
        service.install_contention_profile.return_value = True
        profile = Mock(profile_id=profile_id)
        with patch(
            "vllm_apple.daemon.load_contention_profile", return_value=profile
        ) as load:
            self.assertTrue(control_device_contention_files(service, "rollback"))
        self.assertTrue(load.call_args.args[0].name.endswith(".last-known-good.json"))
        service.install_contention_profile.assert_called_once_with(profile)


if __name__ == "__main__":
    unittest.main()
