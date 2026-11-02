import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_apple.benchmark_gpu_lock import BenchmarkGpuBusy, BenchmarkGpuLock
from vllm_apple.trial_process import run_trial


class TrialProcessTests(unittest.TestCase):
    def test_trial_explicitly_inherits_benchmark_lock(self):
        import os

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {'BENCH_GPU_LOCK': str(Path(tmp)/'gpu.lock')}
        ):
            marker = Path(tmp)/'inherited'
            code = ('import sys; from pathlib import Path; '
                    'from vllm_apple.benchmark_gpu_lock import BenchmarkGpuLock; '
                    '\nwith BenchmarkGpuLock() as lock: '
                    'Path(sys.argv[1]).write_text(str(lock.snapshot()["inherited"]))')
            with BenchmarkGpuLock() as lock:
                self.assertEqual(run_trial([sys.executable, '-c', code, str(marker)],
                                           **lock.child_options(), timeout=10), 0)
                self.assertEqual(marker.read_text(), 'True')
                with self.assertRaises(BenchmarkGpuBusy):
                    with BenchmarkGpuLock():
                        pass
            with BenchmarkGpuLock():
                pass

    def test_success_and_exit_code(self):
        self.assertEqual(run_trial([sys.executable, '-c', 'raise SystemExit(3)']), 3)

    def test_timeout_allows_graceful_final_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp)/'stopped'
            code = "import signal,time,sys; from pathlib import Path; signal.signal(signal.SIGINT,lambda *a:(Path(sys.argv[1]).write_text('stopped'),sys.exit(0))); time.sleep(30)"
            with self.assertRaises(subprocess.TimeoutExpired):
                run_trial([sys.executable, '-c', code, str(marker)], timeout=1, grace=1)
            self.assertEqual(marker.read_text(), 'stopped')

    def test_stubborn_runner_is_reaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp)/'pid'
            code = "import signal,time,os,sys; from pathlib import Path; signal.signal(signal.SIGINT,signal.SIG_IGN); Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
            with self.assertRaises(subprocess.TimeoutExpired):
                run_trial([sys.executable, '-c', code, str(marker)], timeout=1, grace=0.1)
            import os
            with self.assertRaises(ProcessLookupError):
                os.kill(int(marker.read_text()), 0)
