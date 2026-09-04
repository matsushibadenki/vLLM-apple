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
