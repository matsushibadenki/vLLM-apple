import os
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_apple.qualification import QualificationConfig, qualify_model


class FakeBackend:
    def __init__(self, config: object) -> None:
        self.config = config
        self.pid = os.getpid()
        self.base_url = "http://127.0.0.1:8001"
        self.running = False
        self.stopped = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False
        self.stopped = True


class QualificationTests(unittest.TestCase):
    def test_launch_soak_and_shutdown_are_one_fail_closed_report(self) -> None:
        created: list[FakeBackend] = []

        def factory(config: object) -> FakeBackend:
            backend = FakeBackend(config)
            created.append(backend)
            return backend

        soak = {"passed": True, "requests": 20}
        promotion = {"passed": True}
        with patch(
            "vllm_apple.qualification.run_serving_promotion_probe",
            return_value=promotion,
        ), patch("vllm_apple.qualification.run_soak", return_value=soak) as runner:
            report = qualify_model(
                QualificationConfig(
                    model="local-model",
                    executable=Path("/tmp/vllm"),
                    duration_seconds=1,
                    warmup_seconds=0,
                    require_30_minute_window=False,
                ),
                process_factory=factory,  # type: ignore[arg-type]
            )
        self.assertTrue(report["passed"])
        self.assertTrue(report["shutdown_clean"])
        self.assertEqual(report["promotion_probe"], promotion)
        self.assertTrue(created[0].stopped)
        soak_config = runner.call_args.args[0]
        self.assertEqual(soak_config.mode, "chat-mixed")
        self.assertEqual(soak_config.target_pid, os.getpid())

    def test_failed_soak_still_stops_backend(self) -> None:
        created: list[FakeBackend] = []

        def factory(config: object) -> FakeBackend:
            backend = FakeBackend(config)
            created.append(backend)
            return backend

        with patch(
            "vllm_apple.qualification.run_serving_promotion_probe",
            return_value={"passed": True},
        ), patch("vllm_apple.qualification.run_soak", side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                qualify_model(
                    QualificationConfig(
                        model="local-model",
                        executable=Path("/tmp/vllm"),
                        duration_seconds=1,
                        warmup_seconds=0,
                        require_30_minute_window=False,
                    ),
                    process_factory=factory,  # type: ignore[arg-type]
                )
        self.assertTrue(created[0].stopped)

    def test_failed_promotion_stops_before_soak(self) -> None:
        created: list[FakeBackend] = []

        def factory(config: object) -> FakeBackend:
            backend = FakeBackend(config)
            created.append(backend)
            return backend

        with patch(
            "vllm_apple.qualification.run_serving_promotion_probe",
            return_value={"passed": False},
        ), patch("vllm_apple.qualification.run_soak") as soak:
            with self.assertRaisesRegex(RuntimeError, "promotion probe failed"):
                qualify_model(
                    QualificationConfig(
                        model="local-model",
                        executable=Path("/tmp/vllm"),
                        duration_seconds=1,
                        warmup_seconds=0,
                        require_30_minute_window=False,
                    ),
                    process_factory=factory,  # type: ignore[arg-type]
                )
        soak.assert_not_called()
        self.assertTrue(created[0].stopped)


if __name__ == "__main__":
    unittest.main()
