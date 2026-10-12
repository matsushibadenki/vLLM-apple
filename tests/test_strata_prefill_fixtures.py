import unittest

from scripts.probe_strata_prefill import selections


class PrefillFixtureTests(unittest.TestCase):
    def test_distinct_order_boundaries_and_run_lengths(self):
        self.assertEqual(selections("repeat", 3), [(0, 1)] * 3)
        self.assertEqual(selections("alternating", 3), [(0, 1), (2, 3), (0, 1)])
        self.assertEqual(selections("reversed", 3), [(0, 1), (1, 0), (0, 1)])
        self.assertEqual(selections("mixed_runs", 6), [(0, 1), (2, 3), (2, 3),
                                                       (0, 1), (0, 1), (0, 1)])

    def test_deterministic_random_unique_top_two_and_exact_length(self):
        a = selections("random", 128)
        self.assertEqual(a, selections("random", 128))
        self.assertGreater(len(set(a)), 4)
        for name in ("repeat", "mixed_runs", "alternating", "reversed", "random"):
            rows = selections(name, 128)
            self.assertEqual(len(rows), 128)
            self.assertTrue(all(len(set(row)) == 2 and all(0 <= i < 4 for i in row)
                                for row in rows))


if __name__ == "__main__":
    unittest.main()
