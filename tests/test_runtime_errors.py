import json
import stat
import tempfile
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.backend import BackendStartupError
from vllm_apple.runtime_errors import (
    RuntimeFailureCode,
    RuntimeRecoverability,
    classify_runtime_failure,
    persist_crash_diagnostic,
)
from vllm_apple.scheduler import MemoryCapacityError
from vllm_apple.service import RuntimeService


class RuntimeFailureTests(unittest.TestCase):
    def test_backend_timeout_is_retryable_and_schema_valid(self) -> None:
        failure = classify_runtime_failure(
            BackendStartupError("timeout detail", code="backend_readiness_timeout")
        )
        self.assertEqual(failure.code, RuntimeFailureCode.BACKEND_READINESS_TIMEOUT)
        self.assertEqual(failure.recoverability, RuntimeRecoverability.RETRYABLE)
        schema = json.loads(
            Path("schemas/runtime/runtime-failure-v1.schema.json").read_text()
        )
        validate_instance(failure.to_dict(), schema)

    def test_memory_failure_requires_user_action(self) -> None:
        failure = classify_runtime_failure(MemoryCapacityError("too large"))
        self.assertEqual(failure.code, RuntimeFailureCode.MEMORY_CAPACITY_EXCEEDED)
        self.assertEqual(failure.recoverability, RuntimeRecoverability.USER_ACTION_REQUIRED)

    def test_service_exposes_structured_failure_without_raw_detail(self) -> None:
        service = RuntimeService()
        failure = service.set_failure("secret model path /private/model")
        snapshot = service.snapshot().to_dict()
        self.assertEqual(snapshot["failure"], failure.to_dict())
        self.assertEqual(snapshot["last_error"], "runtime.error.internal_error")
        self.assertNotIn("secret model path", json.dumps(snapshot))
        subscription = service.events.subscribe(after_sequence=1, heartbeat=0.01)
        try:
            event = next(subscription)
        finally:
            subscription.close()
        self.assertIsNotNone(event)
        self.assertEqual(event.type, "runtime.failure")
        self.assertEqual(event.payload["failure"], failure.to_dict())

    def test_diagnostic_is_private_and_stores_only_log_digest(self) -> None:
        failure = classify_runtime_failure(
            BackendStartupError("backend died", code="backend_exited")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = persist_crash_diagnostic(
                failure,
                ("token=do-not-store", "model=/private/model"),
                root=Path(directory) / "diagnostics",
            )
            payload = json.loads(path.read_text())
            encoded = json.dumps(payload)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(payload["recent_log_line_count"], 2)
            self.assertNotIn("do-not-store", encoded)
            self.assertNotIn("/private/model", encoded)


if __name__ == "__main__":
    unittest.main()
