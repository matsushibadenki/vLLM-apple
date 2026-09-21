import unittest

from vllm_apple.expert_predictor import CorrectnessNeutralExpertPredictor
from vllm_apple.expert_residency import ExpertKey


class ExpertPredictorTests(unittest.TestCase):
    def test_predicts_stable_transition_and_never_changes_router_output(self):
        predictor = CorrectnessNeutralExpertPredictor()
        a, b, c = ExpertKey(0, 1), ExpertKey(0, 2), ExpertKey(0, 3)
        for key in (a, b, a, b, a, c):
            predictor.observe(key)
        hint = predictor.predict(a)
        self.assertEqual(hint.candidates, (b, c))
        self.assertEqual(hint.observations, 3)
        self.assertEqual(predictor.resolve((c,), hint), (c,))

    def test_context_table_is_bounded_and_unknown_is_empty(self):
        predictor = CorrectnessNeutralExpertPredictor(maximum_contexts=1)
        a, b, c = ExpertKey(0, 1), ExpertKey(0, 2), ExpertKey(0, 3)
        for key in (a, b, c):
            predictor.observe(key)
        self.assertEqual(predictor.snapshot()["contexts"], 1)
        self.assertEqual(predictor.snapshot()["evictions"], 1)
        self.assertEqual(predictor.predict(a).candidates, ())

    def test_invalid_router_selection_fails_closed(self):
        predictor = CorrectnessNeutralExpertPredictor()
        hint = predictor.predict(ExpertKey(0, 0))
        with self.assertRaises(ValueError):
            predictor.resolve((), hint)


if __name__ == "__main__":
    unittest.main()
