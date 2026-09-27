#!/usr/bin/env python3
"""Profile bounded Gemma 2 long-prefix prefill settings on one local model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from vllm_apple.phase_probe import PhaseProbeConfig
from vllm_apple.text_benchmark import run_text_benchmark

SETTINGS = ((1, 2048), (2, 2048), (2, 1024), (2, 512), (1, 512))


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


def _long_prefix_cases() -> tuple[tuple[str, str, str], ...]:
    prefix = " ".join(f"Reference marker {index:04d}." for index in range(256))
    return (
        ("prefix-edit-a", prefix + " Ignore the markers. What is 1+1? Reply only 2.", "2"),
        ("prefix-edit-b", prefix + " Ignore the markers. What is 2+2? Reply only 4.", "4"),
        ("prefix-edit-c", prefix + " Ignore the markers. What is 3+3? Reply only 6.", "6"),
    )


def _stop(process: subprocess.Popen[bytes]) -> bool:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
    return process.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-port", type=int, default=19110)
    parser.add_argument("--requests", type=int, default=6)
    args = parser.parse_args()
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        parser.error("--python must be an executable regular file")
    if not args.model.is_dir():
        parser.error("--model must be an existing local directory")
    if not 1024 <= args.base_port <= 65535 - len(SETTINGS):
        parser.error("--base-port is outside the bounded port range")
    if not 3 <= args.requests <= 60:
        parser.error("--requests must be between 3 and 60")

    repository = Path(__file__).resolve().parents[1]
    model = args.model.resolve()
    environment = dict(os.environ, PYTHONPATH=str(repository), HF_HUB_OFFLINE="1")
    python_executable = str(args.python.absolute())
    report: dict[str, object] = {
        "schema_version": 1,
        "report_kind": "gemma2_prefill_settings_profile",
        "hardware": "Apple M4 / 32 GiB",
        "model_config_sha256": hashlib.sha256(
            (model / "config.json").read_bytes()).hexdigest(),
        "requests_per_setting": args.requests,
        "concurrency": 2,
        "ttft_slo_ms": 10_000,
        "stores_prompt": False,
        "stores_generated_text": False,
        "settings": [],
    }
    all_passed = True
    for offset, (prompt_concurrency, prefill_step_size) in enumerate(SETTINGS):
        port = args.base_port + offset
        command = [
            python_executable, "-m", "vllm_apple.mlx_gemma2_compat",
            "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
            "--decode-concurrency", "2", "--prompt-concurrency", str(prompt_concurrency),
            "--prefill-step-size", str(prefill_step_size),
            "--prompt-cache-size", "4", "--log-level", "ERROR",
        ]
        entry: dict[str, object] = {
            "prompt_concurrency": prompt_concurrency,
            "prefill_step_size": prefill_step_size,
            "command": command,
        }
        log = tempfile.TemporaryFile()
        process = subprocess.Popen(
            command, cwd=repository, env=environment, stdout=log, stderr=log)
        try:
            _wait_ready(f"http://127.0.0.1:{port}", process, 90)
            config = PhaseProbeConfig(
                f"http://127.0.0.1:{port}", str(model), "Apple-M4-32GiB",
                backend="mlx_lm_gemma2_mask_fix", maximum_output_tokens=16,
                timeout_seconds=30, target_pid=process.pid)
            entry["benchmark"] = run_text_benchmark(
                config, requests=args.requests, concurrency=2, cases=_long_prefix_cases(),
                ttft_slo_ms=10_000, e2e_slo_ms=20_000)
        finally:
            entry["shutdown_clean"] = _stop(process)
            log.seek(0)
            log_bytes = log.read(64 * 1024)
            entry["backend_log_sha256"] = hashlib.sha256(log_bytes).hexdigest()
            entry["backend_log_truncated"] = len(log_bytes) == 64 * 1024
            entry["backend_exit_code"] = process.returncode
            log.close()
        benchmark = entry.get("benchmark", {})
        entry["passed"] = bool(
            isinstance(benchmark, dict)
            and benchmark.get("completed") == args.requests
            and benchmark.get("quality_passed") == args.requests
            and entry["shutdown_clean"])
        all_passed = all_passed and bool(entry["passed"])
        settings = report["settings"]
        assert isinstance(settings, list)
        settings.append(entry)

    report["passed"] = all_passed
    report["selection_policy"] = (
        "Compare TTFT and SLO counts; this profile does not promote a setting automatically.")
    _atomic_json(args.output.resolve(), report)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
