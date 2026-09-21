import unittest

from vllm_apple.importance_analysis import (
    ImportanceAnalyzer,
    ImportanceKey,
    ImportanceKind,
    ImportanceObservation,
)


class ImportanceAnalysisTests(unittest.TestCase):
    def test_weighted_aggregate_ranks_layer_head_and_neuron(self):
        analyzer = ImportanceAnalyzer()
        layer = ImportanceKey(ImportanceKind.LAYER, 0)
        head = ImportanceKey(ImportanceKind.ATTENTION_HEAD, 0, 1)
        neuron = ImportanceKey(ImportanceKind.NEURON, 1, 2)
        analyzer.observe(ImportanceObservation(layer, 2, 1.0, 1.0))
        analyzer.observe(ImportanceObservation(layer, 2, 3.0, 1.0))
        analyzer.observe(ImportanceObservation(head, 4, 1.0, 3.0))
        analyzer.observe(ImportanceObservation(neuron, 4, 1.0, .5))
        report = analyzer.report(maximum_results=2)
        self.assertEqual(report["component_count"], 3)
        self.assertTrue(report["truncated"])
        self.assertEqual(report["components"][0]["kind"], "attention_head")
        self.assertEqual(report["components"][1]["mean_absolute_activation"], 2.0)
        self.assertEqual(len(report["report_id"]), 64)

    def test_zero_scores_and_invalid_keys_are_deterministic(self):
        analyzer = ImportanceAnalyzer(maximum_components=1)
        key = ImportanceKey(ImportanceKind.LAYER, 0)
        analyzer.observe(ImportanceObservation(key, 1, 0.0, 0.0))
        self.assertEqual(analyzer.report()["components"][0]["normalized_score"], 0.0)
        with self.assertRaises(ValueError):
            ImportanceKey(ImportanceKind.LAYER, 0, 1)
        with self.assertRaises(ValueError):
            analyzer.observe(ImportanceObservation(
                ImportanceKey(ImportanceKind.LAYER, 1), 1, 1.0, 1.0
            ))


if __name__ == "__main__":
    unittest.main()
