import copy
import unittest

from vllm_apple.p0_audit import build_p0_audit


def reports():
    result = []
    for backend in ('mlx_lm', 'vllm_metal'):
        for index in range(3):
            report = dict(schema_version=1, report_kind='text_http_benchmark', route=backend,
                          workload_sha256='a'*64, artifact_identity_sha256='b'*64,
                          backend_build_sha256='c'*64, artifact_identity_verified=True,
                          phase_profile={'hardware_fingerprint': 'M4'},
                          reproduction={'independent_process': True}, started_at=f'day-{index}',
                          requests=102, completed=102, quality_passed=102, slo_quality_passed=102,
                          warmup_requests=3, warmup=dict(attempted=3, completed=3, quality_passed=3),
                          languages={language: dict(attempted=34, completed=34, quality_passed=34,
                                                    slo_passed=34) for language in ('en', 'ja', 'zh')},
                          goodput_tokens_per_second=10+index, e2e_p99_reference_only=True)
            result.append((report, f'{backend}-{index}'))
    return result


class P0AuditTests(unittest.TestCase):
    def test_baseline_gate_does_not_promote_performance_or_capabilities(self):
        audit = build_p0_audit(reports())
        self.assertTrue(audit['baseline_gate_passed'])
        self.assertFalse(audit['performance_qualification'])
        self.assertFalse(audit['capability_promotion'])
        self.assertEqual(audit['backends']['mlx_lm']['completed'], 306)
        self.assertEqual(audit['backends']['mlx_lm']['language_completions']['ja'], 102)

    def test_failures_identity_mismatch_and_reused_runs_block_gate(self):
        for field, value in (('completed', 101), ('quality_passed', 101),
                             ('artifact_identity_sha256', 'd'*64),
                             ('backend_build_sha256', 'd'*64),
                             ('artifact_identity_verified', False),
                             ('goodput_tokens_per_second', float('nan'))):
            evidence = copy.deepcopy(reports())
            evidence[0][0][field] = value
            with self.subTest(field=field):
                self.assertFalse(build_p0_audit(evidence)['baseline_gate_passed'])
        evidence = reports()
        evidence[1] = evidence[0]
        self.assertFalse(build_p0_audit(evidence)['baseline_gate_passed'])
        self.assertFalse(build_p0_audit(reports()[:3])['baseline_gate_passed'])

    def test_language_failures_and_malformed_nested_evidence_do_not_pass(self):
        evidence = copy.deepcopy(reports())
        evidence[0][0]['languages']['ja']['quality_passed'] = 33
        self.assertFalse(build_p0_audit(evidence)['baseline_gate_passed'])
        for field in ('phase_profile', 'reproduction', 'warmup', 'languages'):
            evidence = copy.deepcopy(reports())
            evidence[0][0][field] = None
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_p0_audit(evidence)
