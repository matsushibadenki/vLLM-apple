import math
import unittest

from vllm_apple.weight_optimization import approximate_low_rank, cluster_weights


class WeightOptimizationTests(unittest.TestCase):
    def test_weight_clustering_is_deterministic_and_bounded(self):
        values = (-1.1, -1.0, -.9, .9, 1.0, 1.1)
        first = cluster_weights(values, clusters=2)
        second = cluster_weights(values, clusters=2)
        self.assertEqual(first, second)
        self.assertEqual(first.centroids, (-1.0, 1.0))
        self.assertLess(first.maximum_absolute_error, .11)

    def test_low_rank_reconstructs_rank_one_matrix(self):
        result = approximate_low_rank(((1.0, 2.0), (2.0, 4.0)), rank=1)
        self.assertLess(result.frobenius_error, 1e-10)
        reconstructed = [[result.left[row][0] * result.right[0][column]
                          for column in range(2)] for row in range(2)]
        self.assertTrue(math.isclose(reconstructed[1][1], 4.0, abs_tol=1e-10))

    def test_non_finite_and_invalid_rank_fail_closed(self):
        with self.assertRaises(ValueError):
            cluster_weights((1.0, math.inf), clusters=1)
        with self.assertRaises(ValueError):
            approximate_low_rank(((1.0, 2.0),), rank=2)


if __name__ == "__main__":
    unittest.main()
