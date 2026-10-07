"""Explicit candidates for the source-pinned, single-process P1 server."""
from __future__ import annotations

from functools import wraps


def settings(name: str) -> dict:
    if name not in ('baseline', 'responsive', 'compact'):
        raise ValueError('P1 efficiency must be baseline, responsive or compact')
    return dict(name=name, scheduler_budget_seconds=.5 if name == 'baseline' else .05,
                allocator_cache_bytes=(64 if name == 'compact' else 256) * 1024**2,
                blocking_idle=name != 'baseline', energy_qualified=False)


def install_idle_wait(generator_class) -> None:
    """Replace idle polling with a queue wakeup; active/distributed reads stay intact.

    Call only after verifying the supported server source hash. In that source,
    timeout=.1 denotes an empty scheduler; timeout=None is an active nonblocking read.
    """
    if getattr(generator_class, '_vllm_idle_wait', False):
        return
    original_read = generator_class._next_request
    original_stop = generator_class.stop_and_join

    @wraps(original_read)
    def read(self, timeout=None):
        if timeout == .1 and not self._is_distributed:
            return self.requests.get()
        return original_read(self, timeout)

    @wraps(original_stop)
    def stop(self):
        self._stop = True
        if not self._is_distributed:
            # Wake an idle consumer even if stop races with entry into get().
            self.requests.put(None)
        return original_stop(self)

    generator_class._next_request = read
    generator_class.stop_and_join = stop
    generator_class._vllm_idle_wait = True
