import unittest

from vllm_apple.importance_analysis import (
    ImportanceAnalyzer,
    ImportanceKey,
    ImportanceKind,
    ImportanceObservation,
)
from vllm_apple.structural_candidates import (
    SimilarityObservation,
    StructuralCandidateKind,
    generate_structural_candidates,
)
from vllm_apple.structural_optimization import (
    StructuralComponentKind,
    analyze_functional_similarity,
)


class StructuralCandidateTests(unittest.TestCase):
    def test_generates_bypass_head_merge_and_adjacent_layer_merge(self):
        analyzer = ImportanceAnalyzer()
        low = ImportanceKey(ImportanceKind.LAYER, 0)
        high = ImportanceKey(ImportanceKind.LAYER, 1)
        analyzer.observe(ImportanceObservation(low, 1, .001, .001))
        analyzer.observe(ImportanceObservation(high, 1, 1.0, 1.0))
        head_a = ImportanceKey(ImportanceKind.ATTENTION_HEAD, 1, 0)
        head_b = ImportanceKey(ImportanceKind.ATTENTION_HEAD, 1, 1)
        similarities = (
            SimilarityObservation(head_a, head_b, analyze_functional_similarity(
                StructuralComponentKind.ATTENTION_HEAD, (1.0, 2.0), (1.0, 2.0)
            )),
            SimilarityObservation(low, high, analyze_functional_similarity(
                StructuralComponentKind.LAYER, (2.0, 3.0), (2.0, 3.0)
            )),
        )
        candidates = generate_structural_candidates(analyzer.report(), similarities)
        self.assertEqual(
            {candidate.kind for candidate in candidates},
            {StructuralCandidateKind.LAYER_BYPASS,
             StructuralCandidateKind.HEAD_MERGE,
             StructuralCandidateKind.LAYER_MERGE},
        )
        self.assertTrue(all(len(candidate.candidate_id) == 64 for candidate in candidates))

    def test_cross_layer_heads_and_nonadjacent_layers_are_not_merged(self):
        report = ImportanceAnalyzer().report()
        similarity = analyze_functional_similarity(
            StructuralComponentKind.ATTENTION_HEAD, (1.0,), (1.0,)
        )
        observations = (SimilarityObservation(
            ImportanceKey(ImportanceKind.ATTENTION_HEAD, 0, 0),
            ImportanceKey(ImportanceKind.ATTENTION_HEAD, 1, 0), similarity,
        ),)
        self.assertEqual(generate_structural_candidates(report, observations), ())


if __name__ == "__main__":
    unittest.main()
