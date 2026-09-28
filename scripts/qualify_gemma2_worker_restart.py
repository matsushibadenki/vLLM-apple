#!/usr/bin/env python3
"""Inject bounded Gemma 2 worker crashes and verify managed restart recovery."""
from __future__ import annotations

import argparse
import json
import os
import signal
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from vllm_apple.backend import (
    BackendConfig,
    BackendHTTPError,
    BackendProcess,
    BackendSupervisor,
    OpenAIProxyEngine,
)
from vllm_apple.phase_probe import PhaseProbeConfig, measure_stream


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _probe(base_url: str, model: str, pid: int | None) -> dict[str, object]:
    config = PhaseProbeConfig(
        base_url, model, "Apple-M4-32GiB", backend="mlx_lm_gemma2_mask_fix",
        maximum_output_tokens=16, timeout_seconds=30, target_pid=pid)
    result = measure_stream(
        replace(config, prompt="What is 1+1? Reply with only the digit."),
        expected_text="2", expected_match_mode="trimmed_exact")
    return {
        "completed": result.measurement.stream_done_ns is not None,
        "quality_passed": result.expected_text_matched,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19121)
    parser.add_argument("--crashes", type=int, default=3)
    args = parser.parse_args()
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        parser.error("--python must be an executable regular file")
    if not args.model.is_dir():
        parser.error("--model must be an existing local directory")
    if not 1024 <= args.port <= 65535 or not 1 <= args.crashes <= 10:
        parser.error("bounded port or crash count is invalid")

    root = Path(__file__).resolve().parents[1]
    current_python_path = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        str(root) if not current_python_path
        else f"{root}{os.pathsep}{current_python_path}")
    config = BackendConfig(
        model=str(args.model.resolve()), executable=args.python.absolute(),
        host="127.0.0.1", port=args.port, startup_timeout=90,
        backend_kind="mlx_lm", python_module="vllm_apple.mlx_gemma2_compat",
        extra_arguments=(
            "--decode-concurrency", "2", "--prompt-concurrency", "2",
            "--prefill-step-size", "512", "--prompt-cache-size", "4"),
    )
    process = BackendProcess(config)
    proxy = OpenAIProxyEngine(process.base_url, process)
    supervisor = BackendSupervisor(
        process, poll_interval=0.05, initial_backoff=0.25,
        maximum_backoff=2.0, maximum_restarts=args.crashes)
    started = time.time()
    cycles: list[dict[str, object]] = []
    clean_shutdown = False
    try:
        supervisor.start()
        initial = _probe(process.base_url, str(args.model.resolve()), process.pid)
        for index in range(args.crashes):
            old_pid = process.pid
            if old_pid is None:
                raise RuntimeError("managed backend PID unavailable")
            os.kill(old_pid, signal.SIGKILL)
            unavailable: dict[str, object] = {
                "observed": False, "status": None, "code": None,
            }
            unavailable_deadline = time.monotonic() + 2
            while process.ready and time.monotonic() < unavailable_deadline:
                time.sleep(0.005)
            try:
                proxy.open_chat_stream({
                    "model": str(args.model.resolve()),
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                })
            except BackendHTTPError as error:
                unavailable = {
                    "observed": True, "status": error.status, "code": error.code,
                }
            backoff_seconds = min(0.25 * (2 ** index), 2.0)
            restart_started = time.monotonic()
            restarted = supervisor.wait_for_restart(index, timeout=30)
            new_pid = process.pid
            recovery = (
                _probe(process.base_url, str(args.model.resolve()), new_pid)
                if restarted else {"completed": False, "quality_passed": False})
            cycles.append({
                "index": index + 1,
                "old_pid": old_pid,
                "new_pid": new_pid,
                "pid_changed": new_pid is not None and new_pid != old_pid,
                "crash_observed": restarted,
                "client_during_restart": unavailable,
                "backoff_seconds": backoff_seconds,
                "restart_seconds": time.monotonic() - restart_started,
                "recovery": recovery,
                "supervisor": supervisor.snapshot(),
            })
    finally:
        supervisor.stop()
        clean_shutdown = not process.running
        if current_python_path is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = current_python_path

    passed = bool(
        initial["completed"] and initial["quality_passed"] and clean_shutdown
        and len(cycles) == args.crashes
        and all(cycle["pid_changed"] and cycle["crash_observed"]
                and cycle["recovery"]["completed"]
                and cycle["recovery"]["quality_passed"] for cycle in cycles)
        and all(cycle["client_during_restart"] == {
            "observed": True, "status": 503, "code": "backend_unavailable"
        } for cycle in cycles)
    )
    report = {
        "schema_version": 1,
        "report_kind": "gemma2_worker_restart_qualification",
        "started_at_unix": started,
        "hardware": "Apple M4 / 32 GiB",
        "initial_probe": initial,
        "cycles": cycles,
        "shutdown_clean": clean_shutdown,
        "elapsed_seconds": time.time() - started,
        "passed": passed,
        "supervisor": supervisor.snapshot(),
        "limits": ["standalone_watchdog", "no_inflight_request_replay"],
    }
    _atomic_json(args.output.resolve(), report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
