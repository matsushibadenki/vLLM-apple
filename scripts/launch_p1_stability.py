"""Submit one bounded P1 test job to user launchd; no automatic restarts."""
from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import sys
import uuid
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--port', type=int, default=19146)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_directory.resolve()
    receipt = args.receipt.resolve()
    if output.exists() or receipt.exists():
        parser.error('output directory and launch receipt must be new')
    if not output.parent.is_dir() or not receipt.parent.is_dir():
        parser.error('output and receipt parents must already exist')
    if not args.python.is_file() or not os.access(args.python, os.X_OK) or not args.model.is_dir():
        parser.error('an installed MLX Python and a local model directory are required')
    if not 1024 <= args.port <= 65535:
        parser.error('port must be 1024–65535')
    label = 'com.vllm-apple.p1.' + uuid.uuid4().hex
    log = output.parent / (output.name + '.launchd.log')
    command = [sys.executable,
               str(root/'scripts/run_p1_stability.py'), '--python', str(args.python.absolute()),
               '--model', str(args.model.resolve()), '--output-directory', str(output),
               '--port', str(args.port)]
    record = dict(label=label, output_directory=str(output), command=command, log=str(log),
                  submitted=False, automatic_restart=False, qualification=False)
    plist = receipt.with_suffix('.plist')
    if plist.exists():
        parser.error('launch plist must be new')
    job = dict(Label=label, ProgramArguments=command, WorkingDirectory=str(root),
               EnvironmentVariables=dict(PYTHONPATH=str(root)), RunAtLoad=True, KeepAlive=False,
               StandardOutPath=str(log), StandardErrorPath=str(log), ExitTimeOut=45)
    with plist.open('xb') as handle:
        plistlib.dump(job, handle)
    record.update(plist=str(plist), domain=f'gui/{os.getuid()}')
    # Reserve a receipt before submitting, so a launch cannot lose its identity.
    with receipt.open('x') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')
    result = subprocess.run(['/bin/launchctl', 'bootstrap', record['domain'], str(plist)],
                            capture_output=True, text=True)
    record.update(submitted=result.returncode == 0, returncode=result.returncode,
                  diagnostic=result.stderr[:2048])
    receipt.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
