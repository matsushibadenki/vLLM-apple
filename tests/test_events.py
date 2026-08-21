import unittest

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


if __name__ == "__main__":
    unittest.main()

