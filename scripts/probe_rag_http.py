#!/usr/bin/env python3
"""Short real MLX HTTP RAG smoke; always shuts down its owned backend."""
import argparse
import hashlib
import json
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vllm_apple.rag import answer_rag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log = args.output.with_suffix(".backend.log")
    cases = []
    error = None
    with log.open("x") as handle:
        process = subprocess.Popen([sys.executable, "-m", "vllm_apple.mlx_server", "--model",
                                    args.model, "--host", "127.0.0.1", "--port", str(port)],
                                   stdout=handle, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError("owned backend exited before readiness")
                try:
                    with urlopen(url + "/v1/models", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("backend readiness deadline exceeded")
                time.sleep(0.25)
            for language, question, text in (
                ("en", "What is the support code? Reply only with the code and citation.",
                 "The support code is 12347."),
                ("ja", "サポートコードは何ですか？コードと引用だけ答えてください。",
                 "サポートコードは12347です。"),
                ("zh", "支持代码是什么？仅回答代码和引用。", "支持代码为12347。"),
            ):
                payload = dict(question=question, language=language,
                               documents=[dict(id="manual", title="Manual", text=text)])
                result = answer_rag(payload, base_url=url, max_tokens=64,
                                    context_tokens=4096, system_role=False)
                passed = (result["status"] == "references_valid" and "12347" in result["answer"]
                          and result["token_budget_verified"])
                cases.append(dict(language=language, passed=passed, result=result))
                empty = answer_rag(dict(payload, documents=[]), base_url=url, max_tokens=64,
                                   context_tokens=4096, system_role=False)
                cases.append(dict(language=language, case="no_sources",
                                  passed=not empty["generation_performed"]
                                  and empty["status"] == "abstained"))
        except Exception as failure:
            error = f"{type(failure).__name__}: {str(failure)[:512]}"
        finally:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    report = dict(schema_version=1, scope="Gemma2/M4/three-language fixed support-code HTTP smoke",
                  model=args.model, cases=cases, backend_returncode=process.returncode,
                  error=error,
                  general_grounding_qualified=False, performance_qualified=False,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                      Path("vllm_apple/rag.py"), Path("vllm_apple/mlx_server.py"), Path(__file__))},
                  passed=error is None and len(cases) == 6
                  and all(c["passed"] for c in cases) and process.returncode == 0)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
