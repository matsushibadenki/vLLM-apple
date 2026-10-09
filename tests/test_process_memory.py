import ctypes
import errno
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from vllm_apple import process_memory as memory
from vllm_apple.phase_probe import PhaseProbeError, _resident_bytes


class ProcessMemoryTests(unittest.TestCase):
    def test_native_samples_do_not_spawn_processes_and_keep_separate_buffers(self):
        def native(pid, flavor, arg, pointer, size):
            self.assertEqual((flavor, arg, size), (4, 0, 96))
            ctypes.cast(pointer, ctypes.POINTER(memory._TaskInfo)).contents.resident_size = pid*4096
            return size
        with patch.object(memory.sys, 'platform', 'darwin'), \
             patch.object(memory, '_task_info_function', return_value=(None, native)), \
             patch.object(memory.subprocess, 'run', side_effect=AssertionError('unexpected process')):
            with ThreadPoolExecutor(max_workers=4) as pool:
                self.assertEqual(list(pool.map(memory.resident_bytes, range(1, 101))),
                                 [pid*4096 for pid in range(1, 101)])

    def test_partial_or_denied_reads_fail_without_zero_or_cached_result(self):
        def denied(*args):
            ctypes.set_errno(errno.EPERM)
            return 0
        with patch.object(memory.sys, 'platform', 'darwin'), \
             patch.object(memory, '_task_info_function', return_value=(None, denied)), \
             patch.object(memory.subprocess, 'run') as spawn:
            with self.assertRaises(PhaseProbeError) as raised:
                _resident_bytes(123)
            self.assertEqual(raised.exception.code, 'rss_unavailable')
            spawn.assert_not_called()
        with patch.object(memory.sys, 'platform', 'darwin'), \
             patch.object(memory, '_task_info_function', return_value=(None, lambda *a: 8)):
            with self.assertRaises(OSError):
                memory.resident_bytes(123)

    def test_portable_and_missing_library_fallback_keep_kib_units(self):
        for platform in ('linux', 'darwin'):
            with self.subTest(platform=platform), patch.object(memory.sys, 'platform', platform), \
                 patch.object(memory, '_task_info_function', side_effect=OSError('unavailable')), \
                 patch.object(memory.subprocess, 'run', return_value=SimpleNamespace(stdout='123\n')) as spawn:
                self.assertEqual(memory.resident_bytes(456), 123*1024)
                spawn.assert_called_once()
        for pid in (0, -1, True, 1.5, 2**31):
            with self.subTest(pid=pid), self.assertRaises(ValueError):
                memory.resident_bytes(pid)
        self.assertEqual(_resident_bytes(None), 0)


class ProcessActivityTests(unittest.TestCase):
    def test_named_abi_counters_and_unavailable_are_explicit(self):
        def native(pid, flavor, arg, pointer, size):
            info = ctypes.cast(pointer, ctypes.POINTER(memory._TaskInfo)).contents
            info.resident_size = 4096
            info.faults = 101
            info.pageins = 7
            info.cow_faults = 3
            info.context_switches = 99
            info.threads = 4
            info.running_threads = 1
            return size
        with patch.object(memory.sys, 'platform', 'darwin'), \
             patch.object(memory, '_task_info_function', return_value=(None, native)), \
             patch.object(memory.subprocess, 'run', side_effect=AssertionError('no subprocess')):
            value = memory.process_activity(123)
            self.assertTrue(value['available'])
            self.assertEqual((value['faults'], value['pageins'], value['context_switches']), (101, 7, 99))
            self.assertEqual(ctypes.sizeof(memory._TaskInfo), 96)
        with patch.object(memory.sys, 'platform', 'linux'):
            self.assertEqual(memory.process_activity(123)['reason'], 'unsupported_platform')
        with patch.object(memory.sys, 'platform', 'darwin'), \
             patch.object(memory, '_task_info_function', return_value=(None, lambda *a: 0)):
            value = memory.process_activity(123)
            self.assertFalse(value['available'])
            self.assertNotIn('pageins', value)
