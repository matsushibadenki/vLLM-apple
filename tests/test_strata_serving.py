"""Serving serialization, model admission and phase delegation contracts."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_apple.mlx_server import main
from vllm_apple.strata_serving import StrataModelProvider, StrataServingModel


class StrataServingTests(unittest.TestCase):
    def test_phase_and_batch_rejection(self):
        model, runtime = MagicMock(), MagicMock()
        wrapper = StrataServingModel(model, runtime)
        for width, phase in ((4, 'prefill'), (1, 'decode')):
            inputs = SimpleNamespace(shape=(1, width))
            self.assertIs(wrapper(inputs, cache='cache'), model.return_value)
            runtime.set_phase.assert_called_with(phase)
            model.assert_called_with(inputs, cache='cache')
        with self.assertRaises(ValueError):
            wrapper(SimpleNamespace(shape=(2, 4)))

    def test_preload_and_no_model_swaps(self):
        provider = SimpleNamespace(is_distributed=False, load_default=MagicMock(),
                                   model=MagicMock(), draft_model=None, tokenizer=object(),
                                   cli_args=SimpleNamespace(model='/model'))
        runtime = MagicMock()
        runtime.snapshot.return_value = {'closed': False}
        with patch('vllm_apple.strata_serving.StrataRuntime.from_model', return_value=runtime):
            wrapped = StrataModelProvider(provider, maximum_bytes=123)
        provider.load_default.assert_called_once()
        self.assertFalse(wrapped.is_batchable)
        self.assertIs(wrapped.load('default_model', None, 'default_model')[1], provider.tokenizer)
        for arguments in (('/other',), ('default_model', '/adapter'),
                          ('default_model', None, '/draft')):
            with self.assertRaises(ValueError):
                wrapped.load(*arguments)
        runtime.snapshot.return_value = {'closed': True}
        with self.assertRaises(RuntimeError):
            wrapped.load('default_model')

    def test_invalid_cli_rejected_before_loading_mlx(self):
        with self.assertRaises(SystemExit):
            main(['--model', '/model', '--strata-expert-root', '/experts'])
        with self.assertRaises(SystemExit):
            main(['--model', '/model', '--strata-expert-root', '/experts',
                  '--strata-model-sha256', 'a'*64, '--strata-resident-bytes', '123',
                  '--strata-resident-entries', '1', '--strata-maximum-file-bytes', '123',
                  '--allow-concurrent-generation'])

    def test_distributed_and_draft_rejected_before_install(self):
        provider = SimpleNamespace(is_distributed=True, load_default=MagicMock())
        with self.assertRaises(ValueError):
            StrataModelProvider(provider)
        provider.load_default.assert_not_called()
        provider.is_distributed = False
        provider.model = object()
        provider.draft_model = object()
        with patch('vllm_apple.strata_serving.StrataRuntime.from_model') as install:
            with self.assertRaises(ValueError):
                StrataModelProvider(provider)
            install.assert_not_called()

    def test_exhausted_startup_rejects_before_model_load(self):
        from vllm_apple.strata_memory import ProcessMemoryBudget
        provider = SimpleNamespace(is_distributed=False, load_default=MagicMock())
        observation = dict(rss_bytes=950, mlx_active_bytes=100, mlx_cache_bytes=0,
                           available_bytes=1000)
        with self.assertRaises(MemoryError):
            StrataModelProvider(provider, process_budget=ProcessMemoryBudget(1000, 100),
                                memory_observer=lambda: observation)
        provider.load_default.assert_not_called()
        observation['rss_bytes'] = 900  # Exactly zero remaining headroom also rejects.
        with self.assertRaises(MemoryError):
            StrataModelProvider(provider, process_budget=ProcessMemoryBudget(1000, 100),
                                memory_observer=lambda: observation)
        provider.load_default.assert_not_called()

    def test_close_synchronizes_before_releasing_weights(self):
        import sys
        from types import ModuleType
        from unittest.mock import call
        provider = SimpleNamespace(is_distributed=False, load_default=MagicMock(),
                                   model=MagicMock(), draft_model=None)
        runtime = MagicMock()
        with patch('vllm_apple.strata_serving.StrataRuntime.from_model', return_value=runtime):
            wrapped = StrataModelProvider(provider)
        core = ModuleType('mlx.core')
        core.synchronize = MagicMock()
        mlx = ModuleType('mlx')
        mlx.core = core
        order = MagicMock()
        order.attach_mock(core.synchronize, 'synchronize')
        order.attach_mock(runtime.close, 'close')
        with patch.dict(sys.modules, {'mlx': mlx, 'mlx.core': core}):
            wrapped.close()
        self.assertEqual(order.mock_calls, [call.synchronize(), call.close()])
