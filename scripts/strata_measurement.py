"""Host load gate and sampled RSS; measurement helpers, not runtime policy."""
from __future__ import annotations

import threading
import time


def cpu_sample(exclude_pid):
    import psutil

    excluded = {exclude_pid}
    try:
        excluded.update(p.pid for p in psutil.Process(exclude_pid).children(recursive=True))
    except psutil.Error:
        pass
    result = {}
    system = psutil.cpu_times()
    # Aggregate CPU counters also cover processes whose counters are inaccessible.
    # This includes our workers, making the during-run gate deliberately stricter.
    result[(-1, 0)] = (system.user + system.system, "__system__")
    for p in psutil.process_iter(["pid", "name", "create_time", "cpu_times"]):
        try:
            if p.pid not in excluded:
                times = p.info["cpu_times"]
                if times is None:
                    continue
                result[(p.pid, p.info["create_time"])] = (
                    times.user + times.system, p.info["name"],
                )
        except (psutil.Error, TypeError):
            continue
    return result


def cpu_delta(before, after, seconds):
    processes = sorted([
        dict(pid=key[0], name=name, cpu_percent=max(0, total - before[key][0]) * 100 / seconds)
        for key, (total, name) in after.items() if key in before and key != (-1, 0)
    ], key=lambda p: p["cpu_percent"], reverse=True)
    aggregate = (max(0, after[(-1, 0)][0] - before[(-1, 0)][0]) * 100 / seconds
                 if (-1, 0) in before and (-1, 0) in after else
                 sum(p["cpu_percent"] for p in processes))
    return dict(total_percent=aggregate,
                maximum_process_percent=processes[0]["cpu_percent"] if processes else 0,
                top_processes=processes[:5],
                new_or_exited_processes=len(before.keys() ^ after.keys()))


def quiet(observation):
    return (observation["total_percent"] <= 200
            and observation["maximum_process_percent"] <= 75
            and observation["new_or_exited_processes"] == 0)


def wait_for_quiet(exclude_pid, *, timeout=30, required_samples=3):
    history = []
    consecutive = 0
    deadline = time.monotonic() + timeout
    before = cpu_sample(exclude_pid)
    started = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(1)
        after = cpu_sample(exclude_pid)
        now = time.monotonic()
        observation = cpu_delta(before, after, now - started)
        history.append(observation)
        consecutive = consecutive + 1 if quiet(observation) else 0
        if consecutive >= required_samples:
            return dict(passed=True, observations=history)
        before, started = after, now
    return dict(passed=False, observations=history)


class RSSSampler:
    """Sampled peak is a lower bound: short transients may occur between samples."""

    def __enter__(self):
        import psutil

        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.samples = 1
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.samples += 1

    def _run(self):
        while not self.stop.wait(0.001):
            try:
                self.sample()
            except Exception as error:
                self.error = type(error).__name__
                break

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join()
        self.sample()


class HostLoadMonitor:
    def __init__(self, exclude_pid, *, interval=1):
        self.exclude_pid = exclude_pid
        self.observations = []
        self.error = None
        self.interval = interval

    def __enter__(self):
        self.before = cpu_sample(self.exclude_pid)
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        after = cpu_sample(self.exclude_pid)
        now = time.monotonic()
        self.last_before, self.last_started = self.before, self.started
        observation = cpu_delta(self.before, after, now - self.started)
        observation["duration_seconds"] = now - self.started
        self.observations.append(observation)
        self.before, self.started = after, now

    def _run(self):
        while not self.stop.wait(self.interval):
            try:
                self.sample()
            except Exception as error:
                self.error = type(error).__name__
                break

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join()
        if self.observations and time.monotonic() - self.started < self.interval:
            self.observations.pop()
            self.before, self.started = self.last_before, self.last_started
        self.sample()
