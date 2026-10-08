import unittest

from scripts.audit_p1_latency import audit, overlap_ms


class LatencyAuditTests(unittest.TestCase):
    def test_clips_intersection_and_rejects_disjoint_steps(self):
        step = {'started_at_unix_ns': 1_000_000, 'elapsed_ms': 3}
        self.assertEqual(overlap_ms(2_000_000, 3_000_000, step), 1)
        self.assertEqual(overlap_ms(4_000_000, 5_000_000, step), 0)

    def test_snapshot_overlap_is_deduplicated_without_certifying_causality(self):
        step = {'started_at_unix_ns': 0, 'elapsed_ms': 5, 'thread_cpu_ms': 1}
        snapshot = {'resources': {'scheduler_step': {'recent_slow_steps': [step]}}}
        report = {'stability_window': {'requests': 1, 'slo_quality_passed': 0,
            'failed_windows': [{'started_at': '1970-01-01T00:00:00+00:00',
                'diagnostic_context': {'before': snapshot, 'after': snapshot},
                'failure_diagnostics': {'samples': [{'request_index': 0, 'language': 'ja',
                    'elapsed_seconds': .001, 'ttft_ms': 2, 'e2e_ms': 3,
                    'reasons': ['ttft_slo_exceeded']}]}}]}}
        result = audit(report)
        row = result['requests'][0]
        self.assertEqual(len(row['overlapping_retained_steps']), 1)
        self.assertEqual(row['observed_slow_step_intersection_ms'], 2)
        self.assertFalse(result['qualification'])
        self.assertFalse(result['root_cause_confirmed'])
        self.assertFalse(row['step_history_complete'])
        self.assertFalse(row['request_phase_attribution_available'])
