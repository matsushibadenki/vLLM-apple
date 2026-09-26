#!/usr/bin/env python3
"""Run bounded M4 qualification for the opt-in Gemma 2 batch-mask fix."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import signal
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

from vllm_apple.phase_probe import PhaseProbeConfig, measure_stream
from vllm_apple.text_benchmark import run_text_benchmark


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


def _wait_ready(base_url: str, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("backend exited during startup")
        try:
            with urllib.request.urlopen(base_url + "/v1/models", timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError("backend startup deadline exceeded")


def _disconnect_stream(port: int, model: str) -> dict[str, object]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content":
                      "Write a detailed 400-word explanation of addition."}],
        "max_tokens": 512,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }, separators=(",", ":")).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("POST", "/v1/chat/completions", body=body,
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    status = response.status
    first = response.readline(1024)
    connection.close()
    return {"status": status, "received_stream_data": bool(first),
            "closed_by_client": True, "cancel_acknowledged": False}


def _long_prefix_cases() -> tuple[tuple[str, str, str], ...]:
    prefix = " ".join(f"Reference marker {index:04d}." for index in range(256))
    return (
        ("prefix-edit-a", prefix + " Ignore the markers. What is 1+1? Reply only 2.", "2"),
        ("prefix-edit-b", prefix + " Ignore the markers. What is 2+2? Reply only 4.", "4"),
        ("prefix-edit-c", prefix + " Ignore the markers. What is 3+3? Reply only 6.", "6"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19096)
    parser.add_argument("--sustained-requests", type=int, default=100)
    parser.add_argument("--long-requests", type=int, default=12)
    args = parser.parse_args()
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        parser.error("--python must be an executable regular file")
    if not args.model.is_dir():
        parser.error("--model must be an existing local directory")
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    if not 1 <= args.sustained_requests <= 10_000 or not 1 <= args.long_requests <= 1_000:
        parser.error("request counts are outside bounded limits")

    repository = Path(__file__).resolve().parents[1]
    model = args.model.resolve()
    base_url = f"http://127.0.0.1:{args.port}"
    # Keep a virtual-environment launcher path intact. Resolving its symlink to
    # the base interpreter changes sys.prefix and can hide the MLX packages.
    python_executable = str(args.python.absolute())
    command = [python_executable, "-m", "vllm_apple.mlx_gemma2_compat",
               "--model", str(model), "--host", "127.0.0.1", "--port", str(args.port),
               "--decode-concurrency", "2", "--prompt-concurrency", "2",
               "--prompt-cache-size", "4", "--log-level", "ERROR"]
    environment = dict(os.environ, PYTHONPATH=str(repository), HF_HUB_OFFLINE="1")
    started_at = time.time()
    report: dict[str, object] = {
        "schema_version": 1, "report_kind": "gemma2_batch_mask_qualification",
        "started_at_unix": started_at, "hardware": "Apple M4 / 32 GiB",
        "command": command, "model_config_sha256": hashlib.sha256(
            (model / "config.json").read_bytes()).hexdigest(),
        "stores_prompt": False, "stores_generated_text": False,
        "cancel_acknowledgement_available": False,
    }
    log = tempfile.TemporaryFile()
    process = subprocess.Popen(command, cwd=repository, env=environment, stdout=log, stderr=log)
    clean_shutdown = False
    try:
        _wait_ready(base_url, process, 90)
        config = PhaseProbeConfig(base_url, str(model), "Apple-M4-32GiB",
                                  backend="mlx_lm_gemma2_mask_fix",
                                  maximum_output_tokens=16, timeout_seconds=30,
                                  target_pid=process.pid)
        report["warmup"] = run_text_benchmark(config, requests=3)
        report["long_prefix_edit"] = run_text_benchmark(
            config, requests=args.long_requests, concurrency=2, cases=_long_prefix_cases(),
            ttft_slo_ms=10_000, e2e_slo_ms=20_000)
        report["sustained"] = run_text_benchmark(
            config, requests=args.sustained_requests, concurrency=2,
            ttft_slo_ms=5_000, e2e_slo_ms=10_000)
        report["disconnect"] = _disconnect_stream(args.port, str(model))
        time.sleep(1)
        recovery = measure_stream(
            replace(config, prompt="What is 1+1? Reply with only the digit."),
            expected_text="2", expected_match_mode="trimmed_exact")
        report["post_disconnect_recovery"] = {
            "completed": recovery.measurement.stream_done_ns is not None,
            "quality_passed": recovery.expected_text_matched,
        }
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        clean_shutdown = process.returncode == 0
        log.seek(0)
        log_bytes = log.read(64 * 1024)
        report["backend_log_sha256"] = hashlib.sha256(log_bytes).hexdigest()
        report["backend_log_truncated"] = len(log_bytes) == 64 * 1024
        report["backend_exit_code"] = process.returncode
        report["shutdown_clean"] = clean_shutdown
        report["elapsed_seconds"] = time.time() - started_at
        log.close()

    long_report = report.get("long_prefix_edit", {})
    sustained = report.get("sustained", {})
    recovery = report.get("post_disconnect_recovery", {})
    report["passed"] = bool(
        isinstance(long_report, dict)
        and long_report.get("completed") == args.long_requests
        and long_report.get("quality_passed") == args.long_requests
        and isinstance(sustained, dict)
        and sustained.get("completed") == args.sustained_requests
        and sustained.get("quality_passed") == args.sustained_requests
        and isinstance(recovery, dict)
        and recovery.get("completed") is True
        and recovery.get("quality_passed") is True
        and clean_shutdown
    )
    report["qualification_scope"] = "short M4 regression; not 30-minute certification"
    _atomic_json(args.output.resolve(), report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
