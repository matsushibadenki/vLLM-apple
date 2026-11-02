"""CPU subprocess fixtures for campaign exclusion; no models or GPU work."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from scripts import compare_p1_cache_entries, compare_p1_efficiency, compare_p1_prefill
from vllm_apple.benchmark_gpu_lock import LOCK_FD_ENV, BenchmarkGpuBusy, BenchmarkGpuLock

COMPARISONS = (compare_p1_prefill, compare_p1_cache_entries, compare_p1_efficiency)


class ComparisonLockTests(unittest.TestCase):
    def test_campaign_keeps_lock_across_every_trial_and_passes_real_descriptor(self):
        for module in COMPARISONS:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                snapshots, policies = [], []
                env = dict(os.environ, BENCH_GPU_LOCK=str(root/'gpu.lock'))
                env.pop(LOCK_FD_ENV, None)
                argv = ['runner', '--python', sys.executable, '--model', tmp,
                        '--output-directory', str(root/'result')]

                def trial(command, **kwargs):
                    with self.assertRaises(BenchmarkGpuBusy):
                        with BenchmarkGpuLock():
                            pass
                    code = ('import json; from vllm_apple.benchmark_gpu_lock import BenchmarkGpuLock; '
                            '\nwith BenchmarkGpuLock() as lock: print(json.dumps(lock.snapshot()))')
                    child = subprocess.run([sys.executable, '-c', code],
                                           env=kwargs['env'], pass_fds=kwargs['pass_fds'],
                                           capture_output=True, text=True, check=True, timeout=10)
                    snapshots.append(json.loads(child.stdout))
                    policies.append(kwargs['env']['VLLM_APPLE_P1_EFFICIENCY'])
                    Path(command[command.index('--output')+1]).write_text(json.dumps({'passed': True}))
                    return 0

                with ExitStack() as stack:
                    stack.enter_context(patch.dict(os.environ, env, clear=True))
                    stack.enter_context(patch.object(sys, 'argv', argv))
                    stack.enter_context(patch.object(module, 'run_trial', side_effect=trial))
                    if hasattr(module, 'summarize'):
                        stack.enter_context(patch.object(module, 'summarize', return_value={
                            'performance_qualified': False, 'standard_promotion': False}))
                    self.assertEqual(module.main(), 0)
                    with BenchmarkGpuLock():
                        pass
                self.assertEqual(len(snapshots), sum(map(len, module.ORDERS)))
                self.assertTrue(all(snapshot['inherited'] for snapshot in snapshots))
                campaign = json.loads((root/'result/campaign.json').read_text())
                self.assertEqual({snapshot['inode'] for snapshot in snapshots},
                                 {campaign['gpu_lock']['inode']})
                self.assertFalse(campaign['standard_promotion'])
                expected = ([value for order in module.ORDERS for value in order]
                            if module is compare_p1_efficiency else ['compact']*len(snapshots))
                self.assertEqual(policies, expected)

    def test_timeout_records_failure_stops_campaign_and_releases_lock(self):
        for module in COMPARISONS:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                argv = ['runner', '--python', sys.executable, '--model', tmp,
                        '--output-directory', str(root/'result')]
                with patch.dict(os.environ, {'BENCH_GPU_LOCK': str(root/'gpu.lock')}), \
                     patch.object(sys, 'argv', argv), \
                     patch.object(module, 'run_trial', side_effect=subprocess.TimeoutExpired('trial', 1)) as trial:
                    self.assertEqual(module.main(), 1)
                    trial.assert_called_once()
                    with BenchmarkGpuLock():
                        pass
                receipts = list((root/'result').glob('*.timeout.json'))
                self.assertEqual(len(receipts), 1)
                receipt = json.loads(receipts[0].read_text())
                self.assertEqual(receipt['status'], 'timeout')
                self.assertFalse(receipt['qualification'])
                self.assertFalse((root/'result/summary.json').exists())
