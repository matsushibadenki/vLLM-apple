import threading
import unittest
import weakref
from unittest import mock

from vllm_apple.events import EventBus, SubscriptionLimitError


class EventBusTests(unittest.TestCase):
    def test_slow_subscriber_receives_gap_and_retained_events(self) -> None:
        bus = EventBus(capacity=2, max_subscribers=1)
        bus.publish("one", {"value": 1})
        bus.publish("two", {"value": 2})
        bus.publish("three", {"value": 3})
        subscription = bus.subscribe(after_sequence=0, heartbeat=0.01)
        try:
            gap = next(subscription)
            second = next(subscription)
            third = next(subscription)
            self.assertEqual(gap.type, "stream.gap")
            self.assertEqual(gap.payload["dropped_events"], 1)
            self.assertEqual((second.type, third.type), ("two", "three"))
        finally:
            subscription.close()
        self.assertEqual(bus.snapshot()["active_subscribers"], 0)

    def test_subscriber_limit_is_immediate_and_close_releases_slot(self) -> None:
        bus = EventBus(max_subscribers=1)
        first = bus.subscribe()
        with self.assertRaises(SubscriptionLimitError):
            bus.subscribe()
        first.close()
        second = bus.subscribe()
        second.close()

    def test_multiple_wraps_preserve_order_and_exact_gap_for_each_subscriber(self) -> None:
        for capacity in (1, 3, 256):
            with self.subTest(capacity=capacity):
                bus = EventBus(capacity=capacity)
                fast, slow = bus.subscribe(), bus.subscribe()
                try:
                    for sequence in range(1, capacity * 4 + 2):
                        published = bus.publish("sample", {"value": sequence})
                        self.assertIs(next(fast), published)
                    latest = capacity * 4 + 1
                    oldest = latest - capacity + 1
                    gap = next(slow)
                    self.assertEqual((gap.sequence, gap.event_id), (oldest - 1, str(oldest - 1)))
                    self.assertEqual(gap.payload, {"dropped_events": oldest - 1})
                    for sequence in range(oldest, latest + 1):
                        event = next(slow)
                        self.assertEqual(event.sequence, sequence)
                        self.assertEqual(event.payload, {"value": sequence})
                    self.assertEqual(bus.snapshot()["retained_events"], capacity)
                finally:
                    fast.close()
                    slow.close()

    def test_overwrite_between_gap_and_resume_reports_additional_loss(self) -> None:
        bus = EventBus(capacity=2)
        subscription = bus.subscribe()
        try:
            for _ in range(4):
                bus.publish("sample", {})
            self.assertEqual(next(subscription).payload, {"dropped_events": 2})
            fifth = bus.publish("sample", {})
            sixth = bus.publish("sample", {})
            self.assertEqual(next(subscription).payload, {"dropped_events": 2})
            self.assertIs(next(subscription), fifth)
            self.assertIs(next(subscription), sixth)
        finally:
            subscription.close()

    def test_empty_and_future_cursor_heartbeat_then_notification(self) -> None:
        bus = EventBus(capacity=2)
        empty = bus.subscribe(heartbeat=0.001)
        self.assertIsNone(next(empty))
        empty.close()
        subscription = bus.subscribe(after_sequence=3, heartbeat=0.001)
        try:
            for _ in range(3):
                bus.publish("sample", {})
                self.assertIsNone(next(subscription))
            waiting = threading.Event()
            actual_wait = bus._condition.wait

            def observed_wait(timeout):
                waiting.set()
                return actual_wait(timeout=1)

            received = []
            with mock.patch.object(bus._condition, "wait", side_effect=observed_wait):
                reader = threading.Thread(target=lambda: received.append(next(subscription)), daemon=True)
                reader.start()
                self.assertTrue(waiting.wait(1))
                fourth = bus.publish("fourth", {})
                reader.join(2)
                self.assertFalse(reader.is_alive())
                self.assertEqual(received, [fourth])
        finally:
            subscription.close()

    def test_suspended_consumer_does_not_hold_publisher_lock(self) -> None:
        bus = EventBus(capacity=2)
        bus.publish("first", {})
        subscription = bus.subscribe()
        try:
            next(subscription)
            writer = threading.Thread(target=lambda: bus.publish("second", {}), daemon=True)
            writer.start()
            writer.join(1)
            self.assertFalse(writer.is_alive())
            self.assertEqual(next(subscription).type, "second")
        finally:
            subscription.close()

    def test_replay_reads_one_slot_without_scanning_history(self) -> None:
        class NoScanList(list):
            reads = 0

            def __iter__(self):
                raise AssertionError("history must not be scanned for a single event")

            def __getitem__(self, slot):
                self.reads += 1
                return super().__getitem__(slot)

        bus = EventBus(capacity=1000)
        for _ in range(2000):
            bus.publish("sample", {})
        bus._events = NoScanList(bus._events)
        subscription = bus.subscribe(after_sequence=1000)
        try:
            for sequence in range(1001, 2001):
                self.assertEqual(next(subscription).sequence, sequence)
        finally:
            subscription.close()
        self.assertEqual(bus._events.reads, 1000)

    def test_overwrite_releases_old_payload_and_empty_history_is_lazy(self) -> None:
        class Payload:
            pass

        bus = EventBus(capacity=2)
        self.assertEqual(len(bus._events), 0)
        payload = Payload()
        ref = weakref.ref(payload)
        bus.publish("first", {"object": payload})
        del payload
        self.assertIsNotNone(ref())
        bus.publish("second", {})
        bus.publish("third", {})
        self.assertIsNone(ref())


if __name__ == "__main__":
    unittest.main()
