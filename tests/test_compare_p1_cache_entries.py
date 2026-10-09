import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_p1_cache_entries import summarize


class CacheEntryComparisonTests(unittest.TestCase):
    def fixture(self, directory):
        benchmark = dict(requests=1, quality_passed=1, slo_quality_passed=1,
                         latency_distributions={'e2e': {'mean_ms': 5, 'max_ms': 10}}, prompt_cache_usage={})
        for entries in (4, 3):
            for repeat in range(1, 4):
                data = dict(status='complete', efficiency_candidate='compact',
                    command=['--prompt-cache-size', str(entries), '--prefill-step-size', '512'],
                    runtime_sources={}, runner_source_sha256='same', model_files={},
                    passed=True, shutdown_clean=True, runtime_identity_unchanged=True,
                    model_identity_unchanged=True, warmup={'slo_quality_passed': 3},
                    normal_memory_points={'after_short': {'same_worker': True,
                        'prompt_cache': {'accounted_bytes': 100}, 'process_activity': {'resident_size': 200}}},
                    long_prefix_edit=benchmark, sustained=benchmark)
                (directory/f'entries-{entries}-{repeat}.json').write_text(json.dumps(data))

    def test_repeats_are_evidence_without_auto_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.fixture(directory)
            result = summarize(directory)
            self.assertEqual(result['default_entries'], 4)
            self.assertFalse(result['performance_qualified'])
            self.assertFalse(result['stability_qualified'])
            self.assertEqual(result['candidates'][0]['short_mean_median_ms'], 5)

    def test_mixed_identity_or_unfinished_run_is_rejected(self):
        for key, value in [('runner_source_sha256', 'changed'), ('status', 'running'), ('normal_memory_points', {}),
                           ('command', ['--prompt-cache-size', '1', '--prefill-step-size', '512'])]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                self.fixture(directory)
                path = directory/'entries-3-3.json'
                data = json.loads(path.read_text())
                data[key] = value
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    summarize(directory)
