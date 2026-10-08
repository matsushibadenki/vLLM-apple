import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vllm_apple.mlx_gemma2_compat import install_backend_phase_timing
from vllm_apple.step_diagnostics import StepDiagnostics


class BackendPhaseTimingTests(unittest.TestCase):
    def test_wrappers_preserve_results_and_do_not_stack(self):
        class Cache:
            def fetch_nearest_cache(self, value):
                return value

        class Prompt:
            def prompt(self, value):
                return value

            def generate(self, value):
                return value

        class Decode:
            def next(self, value):
                return value

        generation = SimpleNamespace(PromptProcessingBatch=Prompt, GenerationBatch=Decode)
        cache = SimpleNamespace(LRUPromptCache=Cache)
        diagnostics = {k: StepDiagnostics() for k in
                       ('cache_fetch', 'prefill_prompt', 'prefill_transition', 'decode_next')}
        install_backend_phase_timing(generation, cache, diagnostics)
        install_backend_phase_timing(generation, cache, diagnostics)
        output = object()
        for method in (Cache().fetch_nearest_cache, Prompt().prompt, Prompt().generate, Decode().next):
            self.assertIs(method(output), output)
        self.assertEqual([d.snapshot()['sample_count'] for d in diagnostics.values()], [1] * 4)

    def test_small_retention_limit_and_snapshot_scope(self):
        diagnostic = StepDiagnostics(sample_limit=2, scope='prefill host method')
        wrapped = diagnostic.wrap(lambda: None)
        with patch('vllm_apple.step_diagnostics.time.monotonic_ns',
                   side_effect=[value for _ in range(4) for value in (0, 300_000_000)]):
            for _ in range(4):
                wrapped()
        snapshot = diagnostic.snapshot()
        self.assertEqual(snapshot['sample_count'], 4)
        self.assertEqual(len(snapshot['recent_slow_steps']), 2)
        self.assertEqual(snapshot['scope'], 'prefill host method')
        for value in (0, 65, True):
            with self.assertRaises(ValueError):
                StepDiagnostics(sample_limit=value)

    def test_empty_prompt_is_forwarded_without_timing_or_diluting_samples(self):
        diagnostic = StepDiagnostics()
        called = []

        def prompt(owner, tokens):
            called.append(tokens)
            return tokens

        wrapped = diagnostic.wrap(prompt, skip_empty_tokens=True)
        with patch('vllm_apple.step_diagnostics.time.monotonic_ns',
                   side_effect=AssertionError('empty prompt must not read clocks')):
            self.assertEqual(wrapped(None, []), [])
            self.assertEqual(wrapped(None, tokens=[]), [])
        self.assertEqual(diagnostic.snapshot()['sample_count'], 0)
        self.assertEqual(wrapped(None, [[1]]), [[1]])
        self.assertEqual(diagnostic.snapshot()['sample_count'], 1)
        self.assertEqual(called, [[], [], [[1]]])
