#!/usr/bin/env python3
"""Bounded real HTTP probe of evidence-gated serve; no generated text is stored."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_apple.phase_probe import PhaseProbeConfig  # noqa: E402
from vllm_apple.promotion_probe import (  # noqa: E402
    PromotionProbeConfig,
    run_serving_promotion_probe,
)
from vllm_apple.qualification import save_qualification_report  # noqa: E402
from vllm_apple.quality_smoke import run_serving_quality_smoke  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend-executable", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18138)
    parser.add_argument("--backend-port", type=int, default=18139)
    args = parser.parse_args()
    if args.output.resolve() == args.evidence.resolve() or args.output.resolve().is_relative_to(
        args.model.resolve()
    ):
        parser.error("output must be outside the model and must not replace evidence")
    for port in (args.port, args.backend_port):
        if not 1024 <= port <= 65535:
            parser.error("probe ports must be unprivileged TCP ports")
    if args.port == args.backend_port:
        parser.error("probe ports must differ")
    for port in (args.port, args.backend_port):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
    evidence_bytes = args.evidence.read_bytes()
    report = {
        "schema_version": 1, "report_kind": "evidence_managed_serve_smoke",
        "backend": "mlx_lm", "model": args.model.name,
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "backend_check_skipped": False, "stores_generated_text": False,
        "context_tokens": 1024, "concurrency": 1,
        "passed": False, "shutdown_clean": False,
    }
    command = [
        sys.executable, "-m", "vllm_apple", "serve", str(args.model),
        "--backend-kind", "mlx_lm", "--backend-executable", str(args.backend_executable),
        "--architecture-evidence", str(args.evidence), "--max-model-len", "1024",
        "--max-concurrent-requests", "1", "--disable-metal-tuning",
        "--port", str(args.port), "--backend-port", str(args.backend_port),
        "--backend-startup-timeout", "90", "--shutdown-grace-period", "5",
    ]

    def timed_out(signum, frame):
        raise TimeoutError("managed serve probe deadline exceeded")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(240)
    base_url = f"http://127.0.0.1:{args.port}"
    with tempfile.TemporaryFile() as logs:
        process = subprocess.Popen(command, stdout=logs, stderr=logs,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError("managed serve exited before readiness")
                if time.monotonic() >= deadline:
                    raise TimeoutError("managed serve readiness timed out")
                try:
                    with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                        health = json.loads(response.read(64 * 1024))
                    if health.get("inference_ready") is True:
                        report["inference_ready"] = True
                        break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.5)
            report["promotion_probe"] = run_serving_promotion_probe(PromotionProbeConfig(
                base_url=base_url, model="default_model", timeout_seconds=15,
                supports_seeded_sampling=False,
            ))
            report["quality_smoke"] = run_serving_quality_smoke(PhaseProbeConfig(
                base_url=base_url, model="default_model", backend="mlx_lm",
                hardware_fingerprint=json.loads(evidence_bytes)["architecture_evidence"]["identity"]["hardware_sha256"],
                samples=1,
                maximum_output_tokens=8, timeout_seconds=15,
            ))
        except Exception as error:
            report["error_type"] = type(error).__name__
        finally:
            signal.alarm(0)
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            report["exit_code"] = process.returncode
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                report["shutdown_clean"] = process.returncode == 0
            else:
                os.killpg(process.pid, signal.SIGKILL)
    report["evidence_unchanged"] = args.evidence.read_bytes() == evidence_bytes
    report["passed"] = bool(
        report.get("inference_ready") and report["shutdown_clean"] and report["evidence_unchanged"]
        and report.get("promotion_probe", {}).get("passed")
        and report.get("quality_smoke", {}).get("passed")
    )
    save_qualification_report(report, args.output)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
