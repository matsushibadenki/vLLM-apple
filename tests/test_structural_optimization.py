import math
import unittest

from vllm_apple.structural_optimization import (
    StructuralComponentKind,
    analyze_functional_similarity,
    prune_structured,
    prune_unstructured,
)


class StructuralOptimizationTests(unittest.TestCase):
    def test_unstructured_pruning_removes_smallest_magnitudes_stably(self):
        result = prune_unstructured((3.0, .1, -.1, 2.0), fraction=.5)
        self.assertEqual(result.mask, (True, False, False, True))
        self.assertEqual(result.retained_values, (3.0, 0.0, 0.0, 2.0))
        self.assertTrue(math.isclose(result.squared_error, .02))

    def test_structured_pruning_supports_rows_and_columns(self):
        rows = prune_structured(((1.0, 1.0), (10.0, 10.0)), axis=0, fraction=.5)
        self.assertEqual(rows.mask, (False, True))
        self.assertEqual(rows.retained_values, (0.0, 0.0, 10.0, 10.0))
        columns = prune_structured(((1.0, 10.0), (1.0, 10.0)), axis=1, fraction=.5)
        self.assertEqual(columns.mask, (False, True))

    def test_functional_similarity_handles_identical_and_zero_outputs(self):
        same = analyze_functional_similarity(
            StructuralComponentKind.ATTENTION_HEAD, (1.0, 2.0), (1.0, 2.0)
        )
        self.assertEqual(same.cosine_similarity, 1.0)
        self.assertEqual(same.mean_squared_error, 0.0)
        zeros = analyze_functional_similarity(
            StructuralComponentKind.LAYER, (0.0, 0.0), (0.0, 0.0)
        )
        self.assertEqual(zeros.cosine_similarity, 1.0)
        self.assertEqual(len(zeros.comparison_id), 64)

    def test_invalid_fraction_and_non_finite_values_fail_closed(self):
        with self.assertRaises(ValueError):
            prune_unstructured((1.0,), fraction=1.0)
        with self.assertRaises(ValueError):
            analyze_functional_similarity(
                StructuralComponentKind.MLP, (math.inf,), (1.0,)
            )


if __name__ == "__main__":
    unittest.main()
