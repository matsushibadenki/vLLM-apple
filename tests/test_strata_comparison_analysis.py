import unittest

from scripts.analyze_strata_comparison import summarize


class ComparisonAnalysisTests(unittest.TestCase):
    def test_request_window_qualifies_independently_of_startup(self):
        trial = dict(variant="baseline", policy="baseline", request_nanoseconds=[100, 100],
                     residency=None, parity_passed=True, host_load=dict(passed=False),
                     startup_host_load=dict(passed=False), request_host_load=dict(passed=True))
        report = dict(scope="synthetic", trials=[trial] * 3, quiet_host_requested=True,
                      host_load_window="requests_only")
        result = summarize(report)
        self.assertTrue(result["uncontended_comparison_passed"])
        self.assertEqual(result["groups"][0]["startup_host_passed_runs"], 0)
        self.assertEqual(result["groups"][0]["request_host_passed_runs"], 3)
        trial.pop("request_host_load")
        self.assertFalse(summarize(report)["uncontended_comparison_passed"])

    def test_empty_or_incomplete_campaign_cannot_pass(self):
        result = summarize(dict(scope="synthetic", trials=[], campaign_completed=False))
        self.assertFalse(result["timing_stable"])
        self.assertFalse(result["uncontended_comparison_passed"])

    def test_stable_time_without_host_evidence_is_not_uncontended(self):
        trial = dict(variant="baseline", policy="baseline", request_nanoseconds=[100, 100],
                     residency=None, parity_passed=True)
        result = summarize(dict(scope="synthetic", trials=[trial] * 3))
        self.assertTrue(result["timing_stable"])
        self.assertFalse(result["uncontended_comparison_passed"])
        trial["host_load"] = dict(passed=False)
        result = summarize(dict(scope="synthetic", trials=[trial] * 3, quiet_host_requested=True))
        self.assertFalse(result["uncontended_comparison_passed"])

    def test_rejects_noisy_runs_even_when_parity_passes(self):
        trials = [dict(variant="baseline", policy="baseline", request_nanoseconds=[n, n],
                       residency=None, parity_passed=True) for n in (100, 100, 200)]
        result = summarize(dict(scope="synthetic", trials=trials))
        self.assertFalse(result["timing_stable"])
        self.assertTrue(result["groups"][0]["parity_passed"])
        self.assertFalse(result["standard_adoption_eligible"])

    def test_budget_violation_is_separate_from_stable_timing(self):
        trial = dict(variant="grouped", policy="lru", request_nanoseconds=[100, 100],
                     parity_passed=True, resident_budget_bytes=200,
                     closed_resident_bytes=0,
                     initial_residency=dict(hits=1, misses=2, evictions=0, rejections=0),
                     residency=dict(hits=4, misses=5, evictions=2, rejections=0,
                                    active_leases=0, peak_resident_bytes=201))
        result = summarize(dict(scope="synthetic", trials=[trial] * 3))
        self.assertTrue(result["timing_stable"])
        self.assertFalse(result["groups"][0]["residency_contract_passed"])
        self.assertEqual(result["groups"][0]["request_residency_deltas"][0]["misses"], 3)
        trial["residency"]["peak_resident_bytes"] = 100
        trial["residency"]["peak_reserved_bytes"] = 201
        result = summarize(dict(scope="synthetic", trials=[trial] * 3))
        self.assertFalse(result["groups"][0]["residency_contract_passed"])
        self.assertEqual(result["groups"][0]["peak_reserved_bytes"], 201)


if __name__ == "__main__":
    unittest.main()
