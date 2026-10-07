"""Profile bounded EventBus replay using a captured resource payload; no GPU work."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.util
import io
import json
import math
import platform
import pstats
import statistics
import sys
import sysconfig
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vllm_apple.events import EventBus


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = sorted(values)
    return {
        "samples": len(values),
        "median_us": statistics.median(values) / 1000,
        "p95_us": values[math.ceil(.95 * len(values)) - 1] / 1000,
        "p99_us": values[math.ceil(.99 * len(values)) - 1] / 1000,
        "maximum_us": max(values) / 1000,
    }


def replay(bus, capacity, payload):
    latest = bus.snapshot()["latest_sequence"]
    subscription = bus.subscribe(after_sequence=latest - capacity)
    timings = []
    cpu = time.process_time_ns()
    started = time.perf_counter_ns()
    try:
        for sequence in range(latest - capacity + 1, latest + 1):
            start = time.perf_counter_ns()
            event = next(subscription)
            timings.append(time.perf_counter_ns() - start)
            if event.sequence != sequence or event.payload != payload:
                raise RuntimeError("replay changed order or payload")
    finally:
        subscription.close()
    return timings, {
        "wall_ms": (time.perf_counter_ns() - started) / 1e6,
        "cpu_ms": (time.process_time_ns() - cpu) / 1e6,
    }


def profile_replay(bus_type, payload):
    bus = bus_type()
    for _ in range(256):
        bus.publish("runtime.resource_sample", payload)
    sub = bus.subscribe()
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(256):
        next(sub)
    profiler.disable()
    sub.close()
    output = io.StringIO()
    pstats.Stats(profiler, stream=output).sort_stats("cumulative").print_stats(10)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path,
                        help="Trusted pre-change events.py; executes this Python source")
    parser.add_argument("--trials", type=int, default=9)
    args = parser.parse_args()
    if args.output.exists() or args.trials < 1:
        parser.error("output must be new and trials positive")
    root = Path(__file__).resolve().parents[1]
    source = root / "docs/evaluation/p1-stability-m4-2026-10-07-r9/30min.json"
    payload = json.loads(source.read_text())["stability_window"]["last_idle_resources"]
    implementations = {"current": EventBus}
    identities = {"current": digest(root / "vllm_apple/events.py")}
    if args.baseline_source:
        name = "vllm_apple._benchmark_baseline_events"
        spec = importlib.util.spec_from_file_location(name, args.baseline_source)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[name] = baseline
        spec.loader.exec_module(baseline)
        implementations = {"baseline": baseline.EventBus, **implementations}
        identities["baseline"] = digest(args.baseline_source)
    rows = []
    for capacity in (100, 256, 1000, 10000):
        buses = {}
        samples = {name: [] for name in implementations}
        totals = {name: [] for name in implementations}
        publications = {name: [] for name in implementations}
        for name, bus_type in implementations.items():
            bus = buses[name] = bus_type(capacity=capacity)
            for _ in range(capacity * 2):
                bus.publish("runtime.resource_sample", payload)
        orders = []
        for trial in range(args.trials):
            order = list(implementations)
            if trial % 2:
                order.reverse()
            orders.append(order)
            for name in order:
                bus = buses[name]
                timings, total = replay(bus, capacity, payload)
                samples[name].extend(timings)
                totals[name].append(total)
                # Steady-state publish on the same full ring; no subscriber waits.
                for _ in range(1000):
                    start = time.perf_counter_ns()
                    bus.publish("runtime.resource_sample", payload)
                    publications[name].append(time.perf_counter_ns() - start)
        for name, bus in buses.items():
            rows.append({
                "implementation": name, "capacity": capacity,
                "latency": stats(samples[name]), "runs": totals[name],
                "publish_latency": stats(publications[name]),
                "container_bytes": sys.getsizeof(bus._events),
                "snapshot": bus.snapshot(), "correctness_passed": True,
                "trial_orders": orders,
            })
    report = {
        "scope": "EventBus wrapped backlog replay; not UI or inference throughput",
        "payload_source": str(source.relative_to(root)), "payload_source_sha256": digest(source),
        "implementation_sha256": identities, "benchmark_sha256": digest(Path(__file__)),
        "python": platform.python_version(),
        "debug_build": bool(sysconfig.get_config_var("Py_DEBUG")),
        "rows": rows,
        "profiles": {name: profile_replay(bus_type, payload)
                     for name, bus_type in implementations.items()},
        "watts_measured": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
