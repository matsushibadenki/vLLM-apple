"""Bounded numeric request timing; never retain request IDs, bodies or output."""
from __future__ import annotations

import threading
import time
from collections import deque


class RequestTimings:
    def __init__(self, capacity=64):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('timing capacity must be a positive integer')
        self._lock = threading.Lock()
        self._records = deque(maxlen=capacity)
        self._sequence = 0
        self._completed = 0

    def start(self):
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        return RequestTiming(self, sequence)

    def snapshot(self):
        with self._lock:
            return {'completed': self._completed, 'capacity': self._records.maxlen,
                    'records': [dict(r, stages_ms=dict(r['stages_ms'])) for r in self._records],
                    'scope': 'Per-request monotonic host timing; no GPU or tokenize attribution'}


class RequestTiming:
    def __init__(self, owner, sequence):
        self._owner = owner
        self._sequence = sequence
        self._start = time.monotonic_ns()
        self._unix = time.time_ns()
        self._lock = threading.Lock()
        self._stages = {}
        self._finished = False

    def mark(self, stage):
        now = time.monotonic_ns()
        with self._lock:
            if not self._finished and stage not in self._stages:
                self._stages[stage] = max(0, now - self._start) / 1e6

    def finish(self):
        self.mark('handler_finished')
        with self._lock:
            if self._finished:
                return
            self._finished = True
            record = {'sequence': self._sequence, 'started_at_unix_ns': self._unix,
                      'stages_ms': dict(self._stages)}
        with self._owner._lock:
            self._owner._completed += 1
            self._owner._records.append(record)


class TimedSSEWriter:
    def __init__(self, writer, trace):
        self._writer = writer
        self._trace = trace

    def write(self, data):
        sse = data.startswith(b'data: ') and not data.startswith(b'data: [DONE]')
        if sse:
            self._trace.mark('first_sse_write_started')
        result = self._writer.write(data)
        if sse:
            self._trace.mark('first_sse_write_returned')
        return result

    def __getattr__(self, name):
        return getattr(self._writer, name)


def timed_tokenize(method, tokenizer, request, args):
    trace = getattr(request, '_vllm_apple_timing', None)
    if trace is None:
        return method(tokenizer, request, args)
    trace.mark('tokenize_started')
    try:
        return method(tokenizer, request, args)
    except BaseException:
        trace.mark('tokenize_failed')
        raise
    finally:
        trace.mark('tokenize_finished')
