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
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rag_quality_suite import build_cases, score_results
from vllm_apple.rag import answer_rag


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_snapshot(model):
    root = Path(__file__).resolve().parents[1]
    sources = ('vllm_apple/rag.py', 'vllm_apple/mlx_server.py',
               'scripts/probe_rag_http.py', 'scripts/rag_quality_suite.py')
    directory = Path(model)
    files = sorted(path for path in directory.rglob('*') if path.is_file()) if directory.is_dir() else []
    if len(files) > 256:
        raise ValueError('model identity file limit exceeded')
    model_files = {str(path.relative_to(directory)): file_hash(path) for path in files}
    packages = {}
    for name in ('mlx', 'mlx-lm', 'transformers'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    complete = ('config.json' in model_files and 'tokenizer_config.json' in model_files
                and ('tokenizer.json' in model_files or 'tokenizer.model' in model_files)
                and any(name.endswith('.safetensors') for name in model_files))
    return dict(source_sha256={name: file_hash(root / name) for name in sources},
                model_files_sha256=model_files, model_complete=complete, packages=packages)


def verify_identity(before, after):
    return dict(runtime_identity_unchanged=before['source_sha256'] == after['source_sha256']
                and before['packages'] == after['packages'],
                model_identity_unchanged=before['model_complete'] and after['model_complete']
                and before['model_files_sha256'] == after['model_files_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-suite", action="store_true")
    parser.add_argument("--shutdown-diagnostics", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    identity_before = evidence_snapshot(args.model)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log = args.output.with_suffix(".backend.log")
    cases = []
    error = None
    forced_kill = False
    stack_dump_requested = False
    with log.open("x") as handle:
        process = subprocess.Popen([sys.executable, "-m", "vllm_apple.mlx_server", "--model",
                                    args.model, "--host", "127.0.0.1", "--port", str(port)]
                                   + (["--diagnostic-signal"] if args.shutdown_diagnostics else []),
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
            if args.quality_suite:
                for case in build_cases():
                    result = answer_rag(case['payload'], base_url=url, max_tokens=64,
                                        context_tokens=4096, system_role=False)
                    cases.append(dict(id=case['id'], result=result))
            else:
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
                forced_kill = True
                if args.shutdown_diagnostics and process.poll() is None:
                    try:
                        process.send_signal(signal.SIGUSR1)
                        stack_dump_requested = True
                        time.sleep(0.5)
                    except ProcessLookupError:
                        pass
                process.kill()
                process.wait(timeout=5)
    identity_error = None
    try:
        identity_after = evidence_snapshot(args.model)
        identity = verify_identity(identity_before, identity_after)
    except (OSError, ValueError) as failure:
        identity_after = None
        identity_error = type(failure).__name__
        identity = dict(runtime_identity_unchanged=False, model_identity_unchanged=False)
    report = dict(schema_version=1, scope="Gemma2/M4/three-language fixed support-code HTTP smoke",
                  model=args.model, cases=cases, backend_returncode=process.returncode,
                  error=error,
                  shutdown=dict(graceful=process.returncode == 0 and not forced_kill,
                                forced_kill=forced_kill, sigint_deadline_seconds=15,
                                stack_dump_requested=stack_dump_requested),
                  general_grounding_qualified=False, performance_qualified=False,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                      Path("vllm_apple/rag.py"), Path("vllm_apple/mlx_server.py"), Path(__file__))},
                  passed=error is None and len(cases) == 6
                  and all(c["passed"] for c in cases) and process.returncode == 0)
    if args.quality_suite:
        report['quality_suite'] = score_results(cases)
        report['scope'] = 'Gemma2/M4 synthetic three-language RAG quality suite'
        report['passed'] = error is None and report['quality_suite']['passed'] and process.returncode == 0
    report.update(identity_before=identity_before, identity_after=identity_after,
                  identity_error=identity_error, **identity)
    report['runtime_identity_verified'] = all(identity.values())
    report['passed'] = report['passed'] and report['runtime_identity_verified']
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
