import unittest
from collections import deque
from types import SimpleNamespace

from experiments.p2_mlx.cache import IdentityPromptCache


class Cache:
    def __init__(self):
        self._lru = SimpleNamespace(_lrus={'user': deque()})
        self.entries = {}
        self.nbytes = 0

    def __len__(self):
        return len(self.entries)

    def insert_cache(self, model, tokens, cache, **kwargs):
        self.entries[(model, tuple(tokens))] = list(cache)
        self._lru._lrus['user'].append((model, tokens))

    def fetch_nearest_cache(self, model, tokens):
        for (key, prefix), cache in self.entries.items():
            if key == model and tuple(tokens[:len(prefix)]) == prefix:
                return list(cache), tokens[len(prefix):]
        return None, tokens

    def trim_to(self, **kwargs):
        self.entries.clear()
        self._lru._lrus['user'].clear()


class P2IdentityCacheTests(unittest.TestCase):
    def test_identity_isolation_exact_prefix_and_budget_release(self):
        cache = Cache()
        first = IdentityPromptCache(cache, 'model-tokenizer-template-position-salt-a')
        second = IdentityPromptCache(cache, 'model-tokenizer-template-position-salt-b')
        first.insert_cache('model', [1, 2], ['actual-cache-placeholder'])
        state, remainder = first.fetch_nearest_cache('model', [1, 2, 3])
        self.assertEqual(remainder, [3])
        self.assertIsNotNone(state)
        self.assertEqual(second.fetch_nearest_cache('model', [1, 2, 3]), (None, [1, 2, 3]))
        self.assertEqual(first.fetch_nearest_cache('model', [1, 7]), (None, [1, 7]))
        self.assertEqual(first.snapshot()['reused_tokens'], 2)
        first.trim_to(n_bytes=0)
        self.assertEqual(first.snapshot()['removed_entries'], 1)
        self.assertFalse(first.snapshot()['qualification'])

    def test_disabled_cache_is_real_miss_baseline(self):
        wrapper = IdentityPromptCache(Cache(), 'a', enabled=False)
        wrapper.insert_cache('model', [1], ['state'])
        self.assertEqual(wrapper.fetch_nearest_cache('model', [1, 2]), (None, [1, 2]))
        self.assertEqual(wrapper.snapshot()['submitted_prefill_tokens'], 2)
