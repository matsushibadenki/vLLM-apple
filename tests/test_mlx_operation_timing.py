import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from vllm_apple.mlx_operation_timing import install_prefill_operation_timing


class PrefillOperationTests(unittest.TestCase):
    def test_preserves_calls_and_only_measures_prompt_thread(self):
        calls = []
        mlx = SimpleNamespace(eval=lambda *a, **k: calls.append((a, k)) or a,
                              clear_cache=lambda: 'cleared', marker=object())
        class Prompt:
            def prompt(self, value):
                generation.mx.eval(value, key=True)
                with ThreadPoolExecutor(1) as pool:
                    pool.submit(generation.mx.eval, 'other thread').result()
                if value == 'fail':
                    raise ValueError('original')
                return generation.mx.clear_cache()
        generation = SimpleNamespace(mx=mlx, PromptProcessingBatch=Prompt)
        proxy = install_prefill_operation_timing(generation)
        self.assertIs(install_prefill_operation_timing(generation), proxy)
        self.assertIs(proxy.marker, mlx.marker)
        self.assertEqual(Prompt().prompt('ok'), 'cleared')
        with self.assertRaisesRegex(ValueError, 'original'):
            Prompt().prompt('fail')
        self.assertEqual(proxy.eval('outside'), ('outside',))
        stats = proxy.snapshot()
        self.assertEqual(stats['eval']['sample_count'], 2)
        self.assertEqual(stats['clear_cache']['sample_count'], 1)
        self.assertEqual(calls[0], (('ok',), {'key': True}))
        self.assertEqual(len(calls), 5)
