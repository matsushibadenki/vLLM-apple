import io
import json
import unittest
from unittest.mock import MagicMock, patch

from vllm_apple.benchmark_backend_memory import observe_backend_memory
from vllm_apple.phase_probe import PhaseProbeConfig


class BackendMemoryTests(unittest.TestCase):
    def test_valid_partial_traversal_and_unknown_hits(self):
        payload = dict(schema_version=1, active_bytes=1, cache_bytes=2, peak_bytes=3,
                       kv_cache_bytes=4, kv_cache_tokens=5, traversal_complete=False,
                       snapshot_consistency='non_atomic')
        result = self.observe(json.dumps(payload).encode())
        self.assertEqual(result['status'], 'observed')
        self.assertFalse(result['traversal_complete'])
        self.assertIsNone(result['cache_hits'])
        self.assertIsNone(result['cache_evictions'])
        self.assertEqual(result['snapshot_consistency'], 'non_atomic')

    def observe(self, raw):
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value = io.BytesIO(raw)
        with patch('vllm_apple.benchmark_backend_memory.urllib.request.build_opener',
                   return_value=opener):
            result = observe_backend_memory(PhaseProbeConfig('http://127.0.0.1:1', 'm', 'M4'))
        self.assertEqual(opener.open.call_args.kwargs['timeout'], 1)
        return result

    def test_invalid_and_oversized_payloads_remain_unavailable(self):
        for raw in (b'{}', b'null', b'invalid', b'x' * 65537):
            with self.subTest(raw=raw[:10]):
                self.assertEqual(self.observe(raw)['status'], 'unavailable')
