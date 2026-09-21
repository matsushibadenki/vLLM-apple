import threading
import unittest

from vllm_apple.layer_prefetch import LayerPrefetchCoordinator


class Backend:
    def __init__(self, fail_once=()):
        self.fail_once = set(fail_once)
        self.loaded = []
        self.released = []

    def load_layer(self, layer):
        self.loaded.append(layer)
        if layer in self.fail_once:
            self.fail_once.remove(layer)
            raise RuntimeError("prefetch failed")
        return f"layer-{layer}"

    def release_layer(self, layer, resource):
        self.released.append((layer, resource))


class LayerPrefetchTests(unittest.TestCase):
    def test_ordered_double_buffer_prefetch_and_release(self):
        backend = Backend()
        report = LayerPrefetchCoordinator(backend).execute(
            (0, 1, 2), lambda layer, resource: (layer, resource)
        )
        self.assertEqual(tuple(item[0] for item in report.results), (0, 1, 2))
        self.assertEqual(report.prefetched_layers, 3)
        self.assertEqual(report.released_layers, 3)
        self.assertEqual(len(backend.released), 3)

    def test_prefetch_failure_uses_on_demand_fallback_once(self):
        backend = Backend(fail_once=(1,))
        report = LayerPrefetchCoordinator(backend).execute(
            (0, 1, 2), lambda layer, resource: resource
        )
        self.assertEqual(report.prefetch_failures, 1)
        self.assertEqual(report.on_demand_fallbacks, 1)
        self.assertEqual(backend.loaded.count(1), 2)
        self.assertEqual(report.released_layers, 3)

    def test_cancel_releases_prefetched_orphan(self):
        backend = Backend()
        cancellation = threading.Event()

        def consume(layer, resource):
            cancellation.set()
            return resource

        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            LayerPrefetchCoordinator(backend).execute(
                (0, 1), consume, cancellation=cancellation
            )
        self.assertEqual(len(backend.released), len(backend.loaded))


if __name__ == "__main__":
    unittest.main()
