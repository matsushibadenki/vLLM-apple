import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from vllm_apple.api import create_server
from vllm_apple.service import RuntimeService


class BlockingRuntimeService(RuntimeService):
    def __init__(self, expected_concurrency: int) -> None:
        super().__init__()
        self._expected_concurrency = expected_concurrency
        self._entered = 0
        self._condition = threading.Condition()
        self.release = threading.Event()

    def snapshot(self):
        with self._condition:
            self._entered += 1
            self._condition.notify_all()
        if not self.release.wait(timeout=3):
            raise TimeoutError("test request was not released")
        return super().snapshot()

    def wait_until_saturated(self, timeout: float = 2) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: self._entered >= self._expected_concurrency,
                timeout=timeout,
            )


class ConcurrentRequestLoadTests(unittest.TestCase):
    def test_saturated_server_rejects_early_and_recovers_all_slots(self) -> None:
        concurrency = 2
        service = BlockingRuntimeService(concurrency)
        server = create_server(
            "127.0.0.1",
            0,
            service,
            max_concurrent_requests=concurrency,
        )
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        outcomes: list[int] = []

        def request_health() -> None:
            try:
                with urllib.request.urlopen(base_url + "/health", timeout=4) as response:
                    outcomes.append(response.status)
            except urllib.error.HTTPError as error:
                outcomes.append(error.code)

        workers = [threading.Thread(target=request_health) for _ in range(concurrency)]
        try:
            for worker in workers:
                worker.start()
            self.assertTrue(service.wait_until_saturated())

            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(base_url + "/health", timeout=1)
            self.assertEqual(raised.exception.code, 503)
            self.assertEqual(json.load(raised.exception)["error"]["code"], "server_busy")
            saturated = server.request_metrics()
            self.assertEqual(saturated["active_requests"], concurrency)
            self.assertEqual(saturated["peak_active_requests"], concurrency)
            self.assertEqual(saturated["rejected_requests"], 1)

            service.release.set()
            for worker in workers:
                worker.join(timeout=3)
                self.assertFalse(worker.is_alive())
            self.assertEqual(outcomes, [200, 200])

            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                self.assertEqual(response.status, 200)
            deadline = time.monotonic() + 1
            recovered = server.request_metrics()
            while recovered["active_requests"] and time.monotonic() < deadline:
                time.sleep(0.01)
                recovered = server.request_metrics()
            self.assertEqual(recovered["active_requests"], 0)
            self.assertEqual(recovered["completed_requests"], concurrency + 1)
        finally:
            service.release.set()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
