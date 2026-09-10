import unittest

from vllm_apple.generative_weight_residency import (
    assess_weight_block_residency,
    current_mlx_gen_weight_residency_feasibility,
)


class WeightBlockResidencyFeasibilityTests(unittest.TestCase):
    def test_current_mlx_gen_contract_fails_closed_with_exact_blockers(self) -> None:
        result = current_mlx_gen_weight_residency_feasibility()

        self.assertFalse(result.eligible)
        self.assertTrue(result.stable_weight_keys)
        self.assertEqual(
            result.blockers,
            (
                "incremental_block_loader_missing",
                "block_release_barrier_missing",
                "compiled_graph_rebind_missing",
            ),
        )

    def test_all_four_contracts_are_required_for_eligibility(self) -> None:
        eligible = assess_weight_block_residency(
            incremental_block_load=True,
            safe_block_release=True,
            compiled_graph_rebind=True,
            stable_weight_keys=True,
        )
        unstable_keys = assess_weight_block_residency(
            incremental_block_load=True,
            safe_block_release=True,
            compiled_graph_rebind=True,
            stable_weight_keys=False,
        )

        self.assertTrue(eligible.eligible)
        self.assertEqual(eligible.blockers, ())
        self.assertFalse(unstable_keys.eligible)
        self.assertEqual(
            unstable_keys.blockers, ("stable_weight_key_contract_missing",)
        )


if __name__ == "__main__":
    unittest.main()
