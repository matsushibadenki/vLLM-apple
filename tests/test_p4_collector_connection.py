import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_p4_evidence import collect, normalize
from tests import test_p4_certification
from vllm_apple.p4_certification import IDENTITY_FIELDS, digest, verify_certification


class CollectorConnectionTests(unittest.TestCase):
    identity = {name: digest(name) for name in IDENTITY_FIELDS}
    commit, artifact = 'a' * 40, 'b' * 64

    def test_complete_scoped_collectors_reach_successful_gate(self):
        fixture = test_p4_certification.P4CertificationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        inputs = {}
        for role, raw in fixture.raw.items():
            report = dict(raw, p4_identity=fixture.identity, role=role,
                          source_commit=fixture.commit, artifact_sha256=fixture.artifact,
                          passed=True)
            if role == 'performance':
                report['report_id'] = digest({k: v for k, v in report.items() if k != 'report_id'})
            path = fixture.root / (role + '-collector.json')
            path.write_text(json.dumps(report))
            inputs[role] = path
        result = collect(fixture.root / 'connected', fixture.identity,
                         fixture.commit, fixture.artifact, inputs)
        self.assertTrue(result['passed'])
        self.assertFalse(result['automatic_application'])

    def test_missing_roles_reach_gate_and_block_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'bundle'
            result = collect(output, self.identity, self.commit, self.artifact, {})
            self.assertFalse(result['passed'])
            self.assertFalse(result['automatic_application'])
            self.assertIn('soak:24h_or_clean_shutdown_missing',
                          result['cells'][0]['rejection_reasons'])
            with self.assertRaises(FileExistsError):
                collect(output, self.identity, self.commit, self.artifact, {})

    def test_real_legacy_result_is_preserved_without_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'p1.json'
            original = dict(passed=True, elapsed_seconds=90, shutdown_clean=True)
            source.write_text(json.dumps(original))
            output = Path(directory) / 'bundle'
            result = collect(output, self.identity, self.commit, self.artifact, {'soak': source})
            self.assertFalse(result['passed'])
            raw = json.loads((output / 'soak-raw.json').read_text())
            self.assertEqual(raw['duration_seconds'], 90)
            self.assertEqual(raw['collector'], original)
            self.assertFalse(raw['identity_verified'])
            (output / 'soak-raw.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify_certification(output / 'bundle.json', source_commit=self.commit,
                                     artifact_sha256=self.artifact)

    def test_exact_identity_and_artifact_are_required(self):
        report = dict(p4_identity=self.identity, role='quality', passed=True,
                      source_commit=self.commit, artifact_sha256=self.artifact)
        self.assertTrue(normalize('quality', report, self.identity,
                                  self.commit, self.artifact)['passed'])
        for key in ('p4_identity', 'source_commit', 'artifact_sha256', 'role'):
            changed = dict(report, **{key: 'wrong'})
            self.assertFalse(normalize('quality', changed, self.identity,
                                      self.commit, self.artifact)['passed'])

    def test_unscoped_p3_policy_is_not_rewritten(self):
        report = dict(policy={'scope_sha256': 'c' * 64}, report_id='d' * 64,
                      standard_adoption_eligible=True, baseline_retained=False)
        normalized = normalize('performance', report, self.identity, self.commit, self.artifact)
        self.assertEqual(normalized['policy'], report['policy'])
        self.assertFalse(normalized['passed'])
