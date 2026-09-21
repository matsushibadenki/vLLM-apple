import unittest

from vllm_apple.multi_mac import (
    DistributedStateShard,
    FabricLink,
    FabricNode,
    FabricStage,
    FabricTransport,
    StageKind,
    build_multi_mac_plan,
)


class MultiMacPlannerTests(unittest.TestCase):
    def nodes(self, failed_b=False):
        return (
            FabricNode("mac-a", 100, ("text", "vision")),
            FabricNode("mac-b", 100, ("text",), not failed_b),
            FabricNode("mac-c", 130, ("audio", "text")),
        )

    def links(self):
        return (
            FabricLink("mac-a", "mac-b", FabricTransport.THUNDERBOLT,
                       1_000_000_000, 50, 65_536, True, "tb-a-b"),
            FabricLink("mac-a", "mac-c", FabricTransport.ETHERNET,
                       100_000_000, 500, 9000, True, "eth-a-c"),
            FabricLink("mac-b", "mac-c", FabricTransport.ETHERNET,
                       100_000_000, 500, 9000, True, "eth-b-c"),
        )

    def stages(self, checkpointable=True):
        return (
            FabricStage("vision", StageKind.MODALITY, "vision", 60, 20),
            FabricStage("decode-0", StageKind.PIPELINE, "text", 60, 10, ("vision",),
                        checkpointable),
            FabricStage("decode-1", StageKind.PIPELINE, "text", 60, 1, ("decode-0",)),
        )

    def test_pipeline_and_modality_plan_prefers_qualified_thunderbolt(self):
        plan = build_multi_mac_plan(self.nodes(), self.links(), self.stages())
        placement = {value.stage_id: value.node_id for value in plan.placements}
        self.assertEqual(placement, {
            "vision": "mac-a", "decode-0": "mac-b", "decode-1": "mac-c"
        })
        self.assertEqual(plan.transfers[0].transport, FabricTransport.THUNDERBOLT)
        self.assertEqual(plan.transfers[0].measurement_id, "tb-a-b")
        self.assertEqual(len(plan.plan_id), 64)
        self.assertEqual(plan.to_dict()["schema_version"], 1)

    def test_unmeasured_or_unauthenticated_links_fail_closed(self):
        with self.assertRaises(ValueError):
            FabricLink("mac-a", "mac-b", FabricTransport.ETHERNET,
                       100, 10, 1500, False, "untrusted")
        with self.assertRaises(RuntimeError):
            build_multi_mac_plan(self.nodes(), (), self.stages())

    def test_failure_replans_checkpointable_stage_and_preserves_state(self):
        initial = build_multi_mac_plan(self.nodes(), self.links(), self.stages())
        recovered = build_multi_mac_plan(
            self.nodes(failed_b=True), self.links(), self.stages(),
            (DistributedStateShard("kv-0", "a" * 64, 10, ("mac-b", "mac-c")),),
            previous_plan=initial,
        )
        self.assertEqual(recovered.failed_nodes, ("mac-b",))
        self.assertEqual(recovered.state_sources, (("kv-0", "mac-c"),))
        self.assertNotIn("mac-b", {value.node_id for value in recovered.placements})

    def test_failure_rejects_noncheckpointable_work_or_lost_state(self):
        initial = build_multi_mac_plan(self.nodes(), self.links(), self.stages(False))
        with self.assertRaisesRegex(RuntimeError, "non-checkpointable"):
            build_multi_mac_plan(
                self.nodes(failed_b=True), self.links(), self.stages(False),
                previous_plan=initial,
            )
        with self.assertRaisesRegex(RuntimeError, "no healthy replica"):
            build_multi_mac_plan(
                self.nodes(failed_b=True), self.links(), self.stages(),
                (DistributedStateShard("kv-0", "b" * 64, 10, ("mac-b",)),),
            )

    def test_cycles_unknown_dependencies_and_capacity_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cycle"):
            build_multi_mac_plan(
                self.nodes(), self.links(),
                (FabricStage("a", StageKind.PIPELINE, "text", 1, 1, ("b",)),
                 FabricStage("b", StageKind.PIPELINE, "text", 1, 1, ("a",))),
            )
        with self.assertRaisesRegex(ValueError, "unknown dependency"):
            build_multi_mac_plan(
                self.nodes(), self.links(),
                (FabricStage("a", StageKind.PIPELINE, "text", 1, 1, ("missing",)),),
            )
        with self.assertRaisesRegex(RuntimeError, "no qualified node"):
            build_multi_mac_plan(
                self.nodes(), self.links(),
                (FabricStage("huge", StageKind.PIPELINE, "text", 131, 1),),
            )


if __name__ == "__main__":
    unittest.main()
