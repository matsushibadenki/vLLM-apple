import json
import sys
import unittest

from vllm_apple.generative_subprocess_adapter import (
    GenerativeSubprocessAdapterError,
    SubprocessGenerativeTelemetryAdapter,
)


def payload(kind: str, elapsed_ms: float, **output):
    value = {
        "kind": kind,
        "elapsed_ms": elapsed_ms,
        "process_rss_bytes": 1024,
        "memory_pressure": "normal",
        "thermal_state": "nominal",
        "output_width": None,
        "output_height": None,
        "output_frames": None,
        "output_sha256": None,
    }
    value.update(output)
    return value


class GenerativeSubprocessAdapterTests(unittest.TestCase):
    def test_jsonl_worker_events_are_streamed(self) -> None:
        lines = [
            payload("started", 0),
            payload(
                "completed",
                10,
                output_width=512,
                output_height=512,
                output_frames=1,
                output_sha256="a" * 64,
            ),
        ]
        script = "import json; print('\\n'.join(json.dumps(v) for v in " + repr(lines) + "))"
        adapter = SubprocessGenerativeTelemetryAdapter(
            (sys.executable, "-c", script), timeout_seconds=2
        )
        events = tuple(adapter.events())
        self.assertEqual([event.kind for event in events], ["started", "completed"])

    def test_invalid_fields_and_nonzero_exit_are_rejected(self) -> None:
        invalid = json.dumps({"kind": "started"})
        adapter = SubprocessGenerativeTelemetryAdapter(
            (sys.executable, "-c", f"print({invalid!r})"), timeout_seconds=2
        )
        with self.assertRaisesRegex(GenerativeSubprocessAdapterError, "fields"):
            tuple(adapter.events())
        failed = SubprocessGenerativeTelemetryAdapter(
            (sys.executable, "-c", "raise SystemExit(7)"), timeout_seconds=2
        )
        with self.assertRaisesRegex(GenerativeSubprocessAdapterError, "status 7"):
            tuple(failed.events())

    def test_bounded_structured_stderr_diagnostic_is_reported(self) -> None:
        script = (
            "import json,sys;"
            "sys.stderr.write('x' * 65536 + '\\n');"
            "print(json.dumps({'vllm_apple_error_code':'memory_error'}), file=sys.stderr);"
            "raise SystemExit(7)"
        )
        adapter = SubprocessGenerativeTelemetryAdapter(
            (sys.executable, "-c", script), timeout_seconds=2
        )
        with self.assertRaisesRegex(
            GenerativeSubprocessAdapterError, "status 7: memory_error"
        ):
            tuple(adapter.events())

        detailed_script = (
            "import json,sys;"
            "print(json.dumps({'vllm_apple_error_code':'value_error',"
            "'vllm_apple_error_detail':'invalid dimensions'}), file=sys.stderr);"
            "raise SystemExit(1)"
        )
        detailed = SubprocessGenerativeTelemetryAdapter(
            (sys.executable, "-c", detailed_script), timeout_seconds=2
        )
        with self.assertRaisesRegex(
            GenerativeSubprocessAdapterError,
            r"status 1: value_error \(invalid dimensions\)",
        ):
            tuple(detailed.events())

        backend_failure_script = (
            "import json,sys;"
            "print(json.dumps({'phase':'failed','error':'missing local shard'}), file=sys.stderr);"
            "print(json.dumps({'vllm_apple_error_code':'backend_system_exit_1',"
            "'vllm_apple_error_detail':'1'}), file=sys.stderr);"
            "raise SystemExit(1)"
        )
        backend_failure = SubprocessGenerativeTelemetryAdapter(
            (sys.executable, "-c", backend_failure_script), timeout_seconds=2
        )
        with self.assertRaisesRegex(
            GenerativeSubprocessAdapterError,
            r"backend_system_exit_1 \(missing local shard\)",
        ):
            tuple(backend_failure.events())

        plain_failure_script = (
            "import json,sys;"
            "print('local model is incomplete', file=sys.stderr);"
            "print(json.dumps({'vllm_apple_error_code':'backend_system_exit_1',"
            "'vllm_apple_error_detail':'1'}), file=sys.stderr);"
            "raise SystemExit(1)"
        )
        plain_failure = SubprocessGenerativeTelemetryAdapter(
            (sys.executable, "-c", plain_failure_script), timeout_seconds=2
        )
        with self.assertRaisesRegex(
            GenerativeSubprocessAdapterError,
            r"backend_system_exit_1 \(local model is incomplete\)",
        ):
            tuple(plain_failure.events())

    def test_timeout_terminates_worker(self) -> None:
        adapter = SubprocessGenerativeTelemetryAdapter(
            (sys.executable, "-c", "import time; time.sleep(10)"), timeout_seconds=0.05
        )
        with self.assertRaisesRegex(GenerativeSubprocessAdapterError, "timed out"):
            tuple(adapter.events())

    def test_oversized_unterminated_line_is_rejected(self) -> None:
        adapter = SubprocessGenerativeTelemetryAdapter(
            (sys.executable, "-c", "print('x' * 2048, end='', flush=True)"),
            timeout_seconds=2,
            max_line_bytes=1024,
        )
        with self.assertRaisesRegex(GenerativeSubprocessAdapterError, "line limit"):
            tuple(adapter.events())


if __name__ == "__main__":
    unittest.main()
