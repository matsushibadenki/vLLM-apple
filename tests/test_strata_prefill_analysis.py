import unittest

from scripts.analyze_strata_prefill import summarize


class PrefillAnalysisTests(unittest.TestCase):
    def report(self):
        return dict(scope="synthetic", notes=[], trials=[dict(
            status="passed", measure="speed", device="cpu", maximum_group_rows=8,
            results=[dict(case="repeat", latencies_ns=[1000, 2000],
                          residency=dict(active_leases=0, peak_reserved_bytes=10),
                          closed_resident_bytes=0, budget_bytes=10,
                          maximum_absolute_error=0, memory=[])],
        )])

    def test_fixture_speed_does_not_qualify(self):
        result = summarize(self.report())
        self.assertTrue(result["all_parity_passed"])
        self.assertFalse(result["speed_qualified"])
        self.assertEqual(result["speed_groups"][0]["median_of_run_medians_us"], 1.5)

    def test_nonfinite_and_budget_violation_rejected(self):
        report = self.report()
        case = report["trials"][0]["results"][0]
        case["maximum_absolute_error"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            summarize(report)
        case["maximum_absolute_error"] = 0
        case["residency"]["peak_reserved_bytes"] = 11
        with self.assertRaisesRegex(ValueError, "retention"):
            summarize(report)


if __name__ == "__main__":
    unittest.main()
