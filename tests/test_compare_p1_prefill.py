import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_p1_prefill import summarize


class PrefillComparisonTests(unittest.TestCase):
    def fixture(self, directory):
        for step in (512, 256):
            for repeat in range(1, 4):
                benchmark = {'latency_distributions': {'e2e': {'mean_ms': repeat}},
                             'quality_passed': 1, 'slo_quality_passed': 1, 'requests': 1}
                data = {'status': 'complete', 'efficiency_candidate': 'compact',
                        'command': ['--prefill-step-size', str(step)],
                        'runtime_sources': {}, 'runner_source_sha256': 'same', 'model_files': {},
                        'long_prefix_edit': benchmark, 'sustained': benchmark,
                        'passed': True, 'shutdown_clean': True,
                        'runtime_identity_unchanged': True, 'model_identity_unchanged': True,
                        'http_exhaustion': {'snapshot': {'allocator': {'peak_bytes': 10}}}}
                (directory / f'prefill-{step}-{repeat}.json').write_text(json.dumps(data))

    def test_summary_does_not_promote_short_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.fixture(directory)
            result = summarize(directory)
            self.assertFalse(result['performance_qualified'])
            self.assertFalse(result['standard_promotion'])
            self.assertEqual(result['candidates'][0]['long_e2e_median_ms'], 2)

    def test_rejects_mixed_identity_or_incomplete_evidence(self):
        for field, value in [('runner_source_sha256', 'different'), ('status', 'running'),
                             ('command', ['--prefill-step-size', '128'])]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                self.fixture(directory)
                path = directory / 'prefill-256-3.json'
                data = json.loads(path.read_text())
                data[field] = value
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    summarize(directory)
