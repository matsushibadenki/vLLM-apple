import queue
import threading
import unittest

from vllm_apple.mlx_efficiency import install_idle_wait, settings


class EfficiencyTests(unittest.TestCase):
    def test_idle_request_and_shutdown_wakeup_preserve_active_reads(self):
        class Generator:
            def __init__(self):
                self.requests = queue.Queue()
                self._is_distributed = False
                self._stop = False
                self.reads = []

            def _next_request(self, timeout=None):
                self.reads.append(timeout)
                return 'upstream'

            def stop_and_join(self):
                self._stop = True
                self.thread.join(1)

        install_idle_wait(Generator)
        install_idle_wait(Generator)
        g = Generator()
        result = []
        g.thread = threading.Thread(target=lambda: result.append(g._next_request(.1)), daemon=True)
        g.thread.start()
        g.requests.put('request')
        g.thread.join(1)
        self.assertEqual(result, ['request'])
        self.assertEqual(g.reads, [])
        self.assertEqual(g._next_request(None), 'upstream')
        g._is_distributed = True
        self.assertEqual(g._next_request(.1), 'upstream')
        self.assertEqual(g.reads, [None, .1])
        g._is_distributed = False
        g.thread = threading.Thread(target=lambda: result.append(g._next_request(.1)), daemon=True)
        g.thread.start()
        g.stop_and_join()
        self.assertFalse(g.thread.is_alive())
        self.assertIsNone(result[-1])

    def test_candidates_reject_unknown_and_preserve_budget_bounds(self):
        with self.assertRaises(ValueError):
            settings('unknown')
        self.assertEqual(settings('compact')['allocator_cache_bytes'], 64*1024**2)
        self.assertLess(settings('responsive')['scheduler_budget_seconds'],
                        settings('baseline')['scheduler_budget_seconds'])
        self.assertFalse(settings('responsive')['energy_qualified'])
