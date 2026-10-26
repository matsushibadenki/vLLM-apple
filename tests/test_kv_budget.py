"""Command contract only: synthetic fixtures do not qualify a real backend."""
import os
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
from vllm_apple.backend import BackendConfig, BackendConfigurationError

class KVBudgetTests(unittest.TestCase):
    def test_legacy_command_is_unchanged(self):
        with patch.dict(os.environ, {}, clear=True):
            config = BackendConfig('local-model', Path('/backend'), backend_kind='mlx_lm')
            self.assertIsNone(config.kv_cache_bytes)
            self.assertNotIn('--prompt-cache-bytes', config.command())

    def test_one_gib_budget_is_snapshotted_and_preserves_managed_options(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / 'mlx_lm.server'
            (Path(directory) / 'python').touch()
            with patch.dict(os.environ, {'VLLM_APPLE_KV_CACHE_MIB': '1024'}):
                config = BackendConfig('local-model', executable, backend_kind='mlx_lm', port=19123)
            command = config.command()
            self.assertEqual(command[-2:], ['--prompt-cache-bytes', '1073741824'])
            self.assertIn('19123', command)
            self.assertEqual(command[:4], [str(Path(directory)/'python'), '-I', '-u', '-c'])
            self.assertTrue(any(x.endswith('mlx_kv_budget.py') for x in command))
            self.assertFalse(any(x.startswith('--skip') for x in command))

    def test_invalid_or_wrong_backend_policies_fail_before_start(self):
        for value in ['0','1','63','4097','1e3','1024.0',' 1024','+1024','١٠٢٤']:
            with self.subTest(value=value), patch.dict(os.environ, {'VLLM_APPLE_KV_CACHE_MIB': value}), self.assertRaises(BackendConfigurationError):
                BackendConfig('local-model', Path('/backend'), backend_kind='mlx_lm')
        with patch.dict(os.environ, {'VLLM_APPLE_KV_CACHE_MIB': '1024'}), self.assertRaises(BackendConfigurationError):
            BackendConfig('local-model', Path('/backend'), backend_kind='vllm_metal')

    def test_duplicate_override_is_rejected(self):
        for arguments in [('--prompt-cache-bytes','1'),('--prompt-cache-bytes=1',)]:
            with patch.dict(os.environ, {'VLLM_APPLE_KV_CACHE_MIB': '1024'}), self.assertRaises(BackendConfigurationError):
                BackendConfig('local-model', Path('/backend'), backend_kind='mlx_lm', extra_arguments=arguments)

class RetainedBudgetTests(unittest.TestCase):
    def test_adapter_enforces_constructor_budget_without_weights(self):
        # scripts/verify-kv-lru.py tests the installed real LRU separately.
        # This fixture tests adaptation without GPU imports in frontend tests.
        from vllm_apple.mlx_kv_budget import install_budget
        class Cache:
            def __init__(self, max_bytes=1<<63): self.max_bytes=max_bytes; self.nbytes=0
            def trim_to(self, *, n_bytes): self.nbytes=min(self.nbytes,n_bytes)
            def insert_cache(self, n): self.nbytes=min(self.nbytes+n,self.max_bytes)
            def fetch_nearest_cache(self): return self.nbytes
        install_budget(Cache, 1024**3)
        cache=Cache(); cache.insert_cache(2*1024**3)
        self.assertEqual(cache.fetch_nearest_cache(),1024**3)
        strict=Cache(max_bytes=128*1024**2); strict.insert_cache(2*1024**3)
        self.assertEqual(strict.nbytes,128*1024**2)
        install_budget(Cache, 1024**3)
        with self.assertRaises(ValueError): install_budget(Cache, 512*1024**2)

class WrapperGuardTests(unittest.TestCase):
    def test_unreviewed_version_and_missing_budget_fail_before_loading(self):
        from vllm_apple.mlx_kv_budget import main
        with patch.dict(os.environ, {'VLLM_APPLE_KV_CACHE_MIB': '1024'}), patch('importlib.metadata.version', return_value='0.33.0'), self.assertRaisesRegex(ValueError, 'reviewed MLX-LM'):
            main()
        with patch.dict(os.environ, {'VLLM_APPLE_KV_CACHE_MIB': ''}), self.assertRaisesRegex(ValueError, 'explicit retained KV'):
            main()

    def test_source_change_cannot_install_the_wrapper(self):
        import sys,types
        from vllm_apple.mlx_kv_budget import main
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'changed.py';file.write_text('# changed')
            server=types.ModuleType('mlx_lm.server');server.__file__=str(file)
            cache=types.ModuleType('mlx_lm.models.cache');cache.__file__=str(file)
            mlx=types.ModuleType('mlx_lm');mlx.server=server
            models=types.ModuleType('mlx_lm.models');models.cache=cache
            with patch.dict(sys.modules, {'mlx_lm':mlx,'mlx_lm.models':models}), patch.dict(os.environ, {'VLLM_APPLE_KV_CACHE_MIB':'1024'}), patch('importlib.metadata.version',return_value='0.32.0'), self.assertRaisesRegex(ValueError,'source changes'):
                main()
