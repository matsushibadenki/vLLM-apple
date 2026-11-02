import fcntl
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_apple.benchmark_gpu_lock import (
    LOCK_FD_ENV,
    BenchmarkGpuBusy,
    BenchmarkGpuLock,
)


class BenchmarkGpuLockTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'gpu.lock'
        env = dict(os.environ, BENCH_GPU_LOCK=str(self.path))
        env.pop(LOCK_FD_ENV, None)
        self.environment = patch.dict(os.environ, env, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_independent_contender_is_rejected_and_exception_releases_lock(self):
        with self.assertRaisesRegex(RuntimeError, 'trial failure'):
            with BenchmarkGpuLock() as lock:
                self.assertFalse(lock.snapshot()['inherited'])
                with self.assertRaises(BenchmarkGpuBusy):
                    with BenchmarkGpuLock():
                        self.fail('contender acquired occupied lock')
                raise RuntimeError('trial failure')
        with BenchmarkGpuLock():
            pass

    def test_plain_flock_holder_excludes_runner(self):
        # Models the same POSIX flock contract used by Kapitsa, without a GPU.
        with self.path.open('w') as external:
            fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BenchmarkGpuBusy):
                with BenchmarkGpuLock():
                    self.fail('external holder was ignored')

    def test_inherited_child_avoids_self_deadlock_and_preserves_parent_lock(self):
        code = (
            'import json; from vllm_apple.benchmark_gpu_lock import BenchmarkGpuLock; '
            '\nwith BenchmarkGpuLock() as lock: print(json.dumps(lock.snapshot()))'
        )
        with BenchmarkGpuLock() as parent:
            result = subprocess.run([sys.executable, '-c', code],
                                    **parent.child_options(), capture_output=True,
                                    text=True, timeout=10, check=True)
            child = json.loads(result.stdout)
            self.assertTrue(child['inherited'])
            self.assertEqual(child['inode'], parent.snapshot()['inode'])
            with self.assertRaises(BenchmarkGpuBusy):
                with BenchmarkGpuLock():
                    pass
        with BenchmarkGpuLock():
            pass

    def test_independent_child_cannot_enter_or_bypass_with_boolean_marker(self):
        code = (
            'from vllm_apple.benchmark_gpu_lock import BenchmarkGpuLock, BenchmarkGpuBusy; '
            '\ntry:\n with BenchmarkGpuLock(): raise SystemExit(9)'
            '\nexcept BenchmarkGpuBusy: raise SystemExit(0)'
        )
        with BenchmarkGpuLock():
            result = subprocess.run([sys.executable, '-c', code],
                                    env=dict(os.environ, NI_LOCK_HELD='1', NACE_GPU_LOCK_HELD='1'),
                                    timeout=10, check=False)
            self.assertEqual(result.returncode, 0)

    def test_wrong_or_closed_inherited_descriptor_is_rejected(self):
        other = Path(self.directory.name) / 'other'
        with other.open('w') as file:
            with patch.dict(os.environ, {LOCK_FD_ENV: str(file.fileno())}):
                self.path.touch()
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    with BenchmarkGpuLock():
                        pass
        with patch.dict(os.environ, {LOCK_FD_ENV: '999999'}):
            with self.assertRaises(OSError):
                with BenchmarkGpuLock():
                    pass

    def test_invalid_descriptor_and_symlink_fail_closed(self):
        for raw in ('1', '-3', 'yes', '３', ''):
            with self.subTest(raw=raw), patch.dict(os.environ, {LOCK_FD_ENV: raw}):
                with self.assertRaises(ValueError):
                    with BenchmarkGpuLock():
                        pass
        target = Path(self.directory.name) / 'target'
        target.touch()
        self.path.symlink_to(target)
        with self.assertRaises(OSError):
            with BenchmarkGpuLock():
                pass

    def test_parent_and_child_overlap_without_unlocking_each_other(self):
        code = (
            'from vllm_apple.benchmark_gpu_lock import BenchmarkGpuLock; '
            '\nwith BenchmarkGpuLock():\n print("ready", flush=True); input()'
        )
        with BenchmarkGpuLock() as parent:
            child = subprocess.Popen([sys.executable, '-c', code],
                                     **parent.child_options(), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True)
        try:
            # Parent closes its descriptor first; the child must still exclude
            # contenders, even before it enters the nested context.
            with self.assertRaises(BenchmarkGpuBusy):
                with BenchmarkGpuLock():
                    pass
            stdout, _ = child.communicate('\n', timeout=10)
            self.assertEqual(child.returncode, 0)
            self.assertEqual(stdout.strip(), 'ready')
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)
        with BenchmarkGpuLock():
            pass

    def test_busy_runners_do_not_start_work_or_create_evidence(self):
        from experiments.p3_mlx import profile
        from scripts import qualify_gemma2_batch_mask, run_p1_stability

        output = Path(self.directory.name) / 'evidence'
        base_args = ['runner', '--python', sys.executable, '--model', self.directory.name]
        runners = (
            (qualify_gemma2_batch_mask, '--output', '_run_qualification'),
            (run_p1_stability, '--output-directory', '_run_stability'),
            (profile, '--output-directory', '_run_profile'),
        )
        with BenchmarkGpuLock():
            for module, option, worker in runners:
                with self.subTest(runner=module.__name__), \
                     patch.object(sys, 'argv', base_args + [option, str(output)]), \
                     patch.object(module, worker) as work:
                    with self.assertRaises(BenchmarkGpuBusy):
                        module.main()
                    work.assert_not_called()
                    self.assertFalse(output.exists())
