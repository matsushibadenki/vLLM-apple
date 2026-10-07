"""Verify an installed local-text preview without importing the checkout."""
import argparse
import hashlib
import http.client
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=19167)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    process = None
    cases = []
    error = None
    limit_rejections = []
    resource_limits_verified = False
    origin = None
    shutdown = False
    log = args.output.with_suffix('.backend.log')
    with log.open('x') as handle:
        try:
            identity = subprocess.check_output([str(args.python), '-c',
                'import json,sys,vllm_apple.local_text as m; '
                'print(json.dumps(dict(prefix=sys.prefix, origin=m.__file__)))'],
                cwd='/tmp', env=environment, text=True, timeout=30)
            origin = json.loads(identity)
            if not Path(origin['origin']).is_relative_to(Path(origin['prefix'])):
                raise RuntimeError('package did not load from the installed environment')
            command = [str(args.python), '-m', 'vllm_apple.local_text', '--model',
                       str(args.model.resolve()), '--port', str(args.port), '--language', 'ja']
            process = subprocess.Popen(command, cwd='/tmp', env=environment,
                                       stdout=handle, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError('preview exited before readiness')
                try:
                    with urlopen(f'http://127.0.0.1:{args.port}/v1/models', timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError('readiness deadline exceeded')
                time.sleep(.25)
            for language, question in (
                ('en', 'What is 1+1? Reply only with the number.'),
                ('ja', '1+1は？数字だけ答えてください。'),
                ('zh', '1+1是多少？只回答数字。'),
            ):
                connection = http.client.HTTPConnection('127.0.0.1', args.port, timeout=30)
                try:
                    connection.request('POST', '/v1/chat/completions', json.dumps(dict(
                        model='default_model', messages=[dict(role='user', content=question)],
                        max_tokens=16, temperature=0, stream=True)), {'Content-Type': 'application/json'})
                    response = connection.getresponse()
                    raw = response.read(65537)
                    chunks = []
                    done = False
                    for line in raw.splitlines():
                        if line == b'data: [DONE]':
                            done = True
                        elif line.startswith(b'data: '):
                            for choice in json.loads(line[6:]).get('choices', []):
                                chunks.append(choice.get('delta', {}).get('content', ''))
                    cases.append(dict(language=language, passed=response.status == 200 and done
                                      and ''.join(chunks).strip() == '2'))
                finally:
                    connection.close()
            for name, override in (
                ('output', dict(max_tokens=513)),
                ('context', dict(messages=[dict(role='user', content='word ' * 5000)])),
                ('model', dict(model='unreviewed-model')),
            ):
                connection = http.client.HTTPConnection('127.0.0.1', args.port, timeout=30)
                try:
                    body = dict(model='default_model', messages=[dict(role='user', content='1+1?')],
                                max_tokens=16, stream=False)
                    body.update(override)
                    connection.request('POST', '/v1/chat/completions', json.dumps(body),
                                       {'Content-Type': 'application/json'})
                    response = connection.getresponse()
                    response.read(65536)
                    limit_rejections.append(dict(case=name, status=response.status,
                                                 passed=400 <= response.status < 500))
                finally:
                    connection.close()
            with urlopen(f'http://127.0.0.1:{args.port}/vllm-apple/resources', timeout=5) as response:
                resources = json.loads(response.read(65536))
            resource_limits_verified = (resources['http']['limit'] == 16
                                        and resources['allocator']['active_bytes'] <= 8*1024**3
                                        and resources['allocator']['cache_bytes'] <= 256*1024**2)
        except Exception as failure:
            error = type(failure).__name__ + ': ' + str(failure)[:512]
        finally:
            if process is not None:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                shutdown = process.returncode == 0
    report = dict(schema_version=1, scope='installed-wheel local text preview smoke',
                  installed_origin=origin, cases=cases, error=error, shutdown_clean=shutdown,
                  limit_rejections=limit_rejections, resource_limits_verified=resource_limits_verified,
                  passed=error is None and len(cases) == 3 and all(c['passed'] for c in cases)
                  and len(limit_rejections) == 3 and all(c['passed'] for c in limit_rejections)
                  and resource_limits_verified and shutdown,
                  production_qualified=False, performance_qualified=False,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
