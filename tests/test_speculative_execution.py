from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from vllm_apple.backend_engine import BackendEngineDescriptor, BackendEngineRegistry
from vllm_apple.device_resources import UnifiedDeviceResourceLedger
from vllm_apple.execution import ExecutionBackend, WorkloadPhase
from vllm_apple.inference_request import InferenceRequestCancelled, InferenceRequestContext
from vllm_apple.speculative_execution import (
    BackendRegistrySpeculativeAdapter,
    HeterogeneousSpeculativeExecutor,
    SpeculativeExecutionProfile,
    load_speculative_profile,
    save_speculative_profile,
)


def _profile(*, match: bool = True, speculative: int = 90) -> SpeculativeExecutionProfile:
    return SpeculativeExecutionProfile.create(
        model_hash="a" * 64,
        precision="fp16",
        draft_backend=ExecutionBackend.COREML_DRAFT,
        verify_backend=ExecutionBackend.NATIVE_METAL,
        sample_count=3,
        baseline_latency_nanoseconds=100,
        speculative_latency_nanoseconds=speculative,
        outputs_match=match,
    )


def _executor(
    profile: SpeculativeExecutionProfile | None = None,
) -> HeterogeneousSpeculativeExecutor:
    ledger = UnifiedDeviceResourceLedger(
        unified_memory_bytes=4096,
        cpu_threads=2,
        gpu_command_queues=1,
        ane_tasks=1,
        bandwidth_slots=1,
    )
    return HeterogeneousSpeculativeExecutor(
        ledger,
        profile or _profile(),
        draft_memory_bytes=512,
        verify_memory_bytes=1024,
        maximum_draft_tokens=3,
    )


class HeterogeneousSpeculativeExecutionTests(unittest.TestCase):
    def test_qualified_profile_round_trips_as_private_atomic_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            path = save_speculative_profile(_profile(), root / "profile.json")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                load_speculative_profile(path, model_hash="a" * 64, precision="fp16"),
                _profile(),
            )
            with self.assertRaisesRegex(ValueError, "identity"):
                load_speculative_profile(path, model_hash="b" * 64, precision="fp16")

    def test_unqualified_profile_is_never_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            with self.assertRaisesRegex(ValueError, "qualified"):
                save_speculative_profile(_profile(speculative=96), root / "profile.json")
            self.assertFalse((root / "profile.json").exists())

    def test_production_registry_adapter_uses_draft_and_verify_phases(self) -> None:
        calls = []

        class Engine:
            def __init__(self, backend, operator, phase):
                self.descriptor = BackendEngineDescriptor(
                    backend,
                    "1",
                    ("test",),
                    ("fp16",),
                    (phase,),
                    (operator,),
                    "in_process",
                )
                self.ready = True

            def start(self):
                self.ready = True

            def execute(self, request, _context):
                calls.append((request.phase, request.payload))
                if request.phase is WorkloadPhase.DRAFT:
                    return (1, 2)
                return request.payload["proposed_token_ids"]

            def stop(self):
                self.ready = False

        profile = _profile()
        registry = BackendEngineRegistry((
            Engine(ExecutionBackend.COREML_DRAFT, "speculative.draft", WorkloadPhase.DRAFT),
            Engine(ExecutionBackend.NATIVE_METAL, "speculative.verify", WorkloadPhase.VERIFY),
        ))
        adapter = BackendRegistrySpeculativeAdapter(
            registry, profile, model_architecture="test"
        )
        context = InferenceRequestContext("request", time.monotonic() + 10, threading.Event())
        self.assertEqual(adapter.draft((), 2, context), (1, 2))
        self.assertEqual(adapter.verify((), (1, 2), context), (1, 2))
        self.assertEqual([call[0] for call in calls], [WorkloadPhase.DRAFT, WorkloadPhase.VERIFY])

    def test_only_publishes_gpu_verified_tokens(self) -> None:
        context = InferenceRequestContext("request", time.monotonic() + 10, threading.Event())
        order: list[str] = []
        published: list[int] = []

        def draft(prefix, count, _context):
            order.append("draft")
            return tuple(range(len(prefix) + 1, len(prefix) + count + 1))

        def verify(prefix, proposed, _context):
            order.append("verify")
            if not prefix:
                return (1, 9, 10)
            return proposed

        def publish(tokens):
            order.append("publish")
            published.extend(tokens)

        result = _executor().execute(
            context,
            maximum_output_tokens=5,
            draft=draft,
            verify=verify,
            publish=publish,
        )
        self.assertEqual(result.token_ids, (1, 9, 3, 4, 5))
        self.assertEqual(tuple(published), result.token_ids)
        self.assertEqual(result.verifier_corrections, 1)
        self.assertEqual(order[:3], ["draft", "verify", "publish"])

    def test_rejects_unqualified_profile(self) -> None:
        for profile in (_profile(match=False), _profile(speculative=96)):
            with self.assertRaisesRegex(ValueError, "unqualified"):
                _executor(profile)

    def test_cancellation_prevents_unverified_publish(self) -> None:
        cancelled = threading.Event()
        context = InferenceRequestContext("request", time.monotonic() + 10, cancelled)
        published: list[int] = []

        def draft(_prefix, _count, _context):
            cancelled.set()
            return (1,)

        with self.assertRaises(InferenceRequestCancelled):
            _executor().execute(
                context,
                maximum_output_tokens=1,
                draft=draft,
                verify=lambda *_: (1,),
                publish=published.extend,
            )
        self.assertEqual(published, [])

    def test_releases_resource_reservations_after_backend_failure(self) -> None:
        executor = _executor()
        context = InferenceRequestContext("request", time.monotonic() + 10, threading.Event())

        def fail(*_arguments):
            raise RuntimeError("failed")

        with self.assertRaisesRegex(RuntimeError, "failed"):
            executor.execute(
                context,
                maximum_output_tokens=1,
                draft=fail,
                verify=lambda *_: (1,),
            )
        self.assertEqual(executor._ledger.snapshot()["active_reservations"], 0)


if __name__ == "__main__":
    unittest.main()
