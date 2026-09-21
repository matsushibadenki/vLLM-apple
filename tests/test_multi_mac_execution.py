import threading
import time
import unittest

from vllm_apple.multi_mac import (
    FabricLink,
    FabricNode,
    FabricStage,
    FabricTransport,
    StageKind,
    build_multi_mac_plan,
)
from vllm_apple.multi_mac_execution import (
    MultiMacExecutionCancelled,
    MultiMacExecutionCoordinator,
)


class Backend:
    def __init__(self, barrier=None, bad=False):
        self.barrier = barrier
        self.bad = bad
        self.calls = []

    def execute(self, stage, inputs):
        self.calls.append((stage.stage_id, inputs))
        if self.barrier is not None and stage.kind is StageKind.MODALITY:
            self.barrier.wait(timeout=2)
        return bytes(stage.output_bytes + (1 if self.bad else 0))


class Transfer:
    def __init__(self, corrupt=False):
        self.corrupt = corrupt
        self.calls = []

    def transfer(self, planned, payload):
        self.calls.append(planned)
        return (b"x" + payload[1:]) if self.corrupt and payload else payload


class MultiMacExecutionTests(unittest.TestCase):
    def fixture(self):
        nodes = (
            FabricNode("mac-a", 120, ("vision", "text")),
            FabricNode("mac-b", 120, ("audio", "text")),
        )
        links = (FabricLink(
            "mac-a", "mac-b", FabricTransport.ETHERNET,
            1_000_000, 100, 1500, True, "qualified-link",
        ),)
        stages = (
            FabricStage("audio", StageKind.MODALITY, "audio", 40, 4),
            FabricStage("vision", StageKind.MODALITY, "vision", 40, 4),
            FabricStage("decode", StageKind.PIPELINE, "text", 70, 2,
                        ("audio", "vision")),
        )
        return stages, build_multi_mac_plan(nodes, links, stages)

    def test_independent_modalities_run_in_parallel_then_pipeline(self):
        stages, plan = self.fixture()
        barrier = threading.Barrier(2)
        backends = {"mac-a": Backend(barrier), "mac-b": Backend(barrier)}
        transfer = Transfer()
        report = MultiMacExecutionCoordinator(
            plan, stages, backends, transfer, maximum_concurrency=2
        ).execute()
        self.assertTrue(report.passed)
        self.assertEqual(len(report.results), 3)
        self.assertEqual(report.peak_result_bytes, 10)
        self.assertFalse(report.to_dict()["stores_payload"])
        self.assertEqual(len(transfer.calls), len(plan.transfers))

    def test_transfer_corruption_and_bad_stage_output_fail_closed(self):
        stages, plan = self.fixture()
        with self.assertRaisesRegex(ValueError, "changed"):
            MultiMacExecutionCoordinator(
                plan, stages, {"mac-a": Backend(), "mac-b": Backend()}, Transfer(True)
            ).execute()
        with self.assertRaisesRegex(ValueError, "output"):
            MultiMacExecutionCoordinator(
                plan, stages, {"mac-a": Backend(bad=True), "mac-b": Backend()}, Transfer()
            ).execute()

    def test_cancel_deadline_and_plan_mismatch_are_rejected(self):
        stages, plan = self.fixture()
        coordinator = MultiMacExecutionCoordinator(
            plan, stages, {"mac-a": Backend(), "mac-b": Backend()}, Transfer()
        )
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(MultiMacExecutionCancelled):
            coordinator.execute(cancellation=cancelled)
        with self.assertRaises(MultiMacExecutionCancelled):
            coordinator.execute(deadline=time.monotonic() - 1)
        with self.assertRaisesRegex(ValueError, "do not match"):
            MultiMacExecutionCoordinator(
                plan, stages[:-1], {"mac-a": Backend(), "mac-b": Backend()}, Transfer()
            )


if __name__ == "__main__":
    unittest.main()
