from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

from .version import SCHEMA_VERSION


class SubscriptionLimitError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    sequence: int
    event_id: str
    type: str
    timestamp: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "event_id": self.event_id,
            "type": self.type,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


class EventSubscription(Iterator[RuntimeEvent | None]):
    def __init__(self, bus: "EventBus", iterator: Iterator[RuntimeEvent | None]) -> None:
        self._bus = bus
        self._iterator = iterator
        self._closed = False

    def __iter__(self) -> "EventSubscription":
        return self

    def __next__(self) -> RuntimeEvent | None:
        if self._closed:
            raise StopIteration
        try:
            return next(self._iterator)
        except StopIteration:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._iterator.close()
        self._bus._release_subscriber()


class EventBus:
    """Bounded multi-subscriber event history with overwrite backpressure.

    Publishers never wait for clients. A lagging subscriber receives a synthetic
    gap event and resumes at the oldest event still retained in the ring.
    """

    def __init__(self, capacity: int = 256, max_subscribers: int = 8) -> None:
        if capacity <= 0 or max_subscribers <= 0:
            raise ValueError("event bus limits must be positive")
        self.capacity = capacity
        self.max_subscribers = max_subscribers
        self._events: deque[RuntimeEvent] = deque(maxlen=capacity)
        self._sequence = 0
        self._subscribers = 0
        self._condition = threading.Condition()

    def publish(self, event_type: str, payload: dict[str, Any]) -> RuntimeEvent:
        if not event_type:
            raise ValueError("event_type cannot be empty")
        with self._condition:
            self._sequence += 1
            event = RuntimeEvent(
                sequence=self._sequence,
                event_id=str(self._sequence),
                type=event_type,
                timestamp=datetime.now(timezone.utc).isoformat(),
                payload=dict(payload),
            )
            self._events.append(event)
            self._condition.notify_all()
            return event

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            return {
                "capacity": self.capacity,
                "retained_events": len(self._events),
                "latest_sequence": self._sequence,
                "active_subscribers": self._subscribers,
                "max_subscribers": self.max_subscribers,
            }

    def subscribe(self, after_sequence: int = 0, heartbeat: float = 15.0) -> EventSubscription:
        if after_sequence < 0 or heartbeat <= 0:
            raise ValueError("invalid subscription options")
        with self._condition:
            if self._subscribers >= self.max_subscribers:
                raise SubscriptionLimitError("runtime event subscriber limit reached")
            self._subscribers += 1

        return EventSubscription(self, self._iterate(after_sequence, heartbeat))

    def _release_subscriber(self) -> None:
        with self._condition:
            self._subscribers -= 1

    def _iterate(self, after_sequence: int, heartbeat: float) -> Iterator[RuntimeEvent | None]:
        next_sequence = after_sequence + 1
        while True:
            output: RuntimeEvent | None
            with self._condition:
                oldest = self._events[0].sequence if self._events else self._sequence + 1
                if next_sequence < oldest:
                    dropped = oldest - next_sequence
                    next_sequence = oldest
                    output = RuntimeEvent(
                        sequence=oldest - 1,
                        event_id=str(oldest - 1),
                        type="stream.gap",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        payload={"dropped_events": dropped},
                    )
                else:
                    available = next(
                        (event for event in self._events if event.sequence >= next_sequence),
                        None,
                    )
                    if available is not None:
                        next_sequence = available.sequence + 1
                        output = available
                    else:
                        notified = self._condition.wait(timeout=heartbeat)
                        if notified:
                            continue
                        output = None
            # Never hold the publisher condition while a slow client consumes.
            yield output
