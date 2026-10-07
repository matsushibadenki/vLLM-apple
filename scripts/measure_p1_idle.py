"""Measure reviewed scheduler idle polling; CPU time is not electrical energy."""
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_lm import server
from mlx_lm.models.cache import LRUPromptCache

from vllm_apple.mlx_efficiency import install_idle_wait
from vllm_apple.mlx_gemma2_compat import SERVER_SOURCE_SHA256

if hashlib.sha256(Path(server.__file__).read_bytes()).hexdigest() != SERVER_SOURCE_SHA256:
    raise RuntimeError('unreviewed upstream server')
output = Path(sys.argv[1])
if output.exists():
    raise ValueError('output must be new')
original_read = server.ResponseGenerator._next_request
original_stop = server.ResponseGenerator.stop_and_join
rows = []
for candidate in ('baseline', 'blocking'):
    if candidate == 'blocking':
        install_idle_wait(server.ResponseGenerator)
    read = server.ResponseGenerator._next_request
    entries = [0]

    def counted(self, timeout=None):
        entries[0] += 1
        return read(self, timeout)

    server.ResponseGenerator._next_request = counted
    generator = server.ResponseGenerator(SimpleNamespace(load_default=lambda: None), LRUPromptCache())
    start = time.monotonic()
    cpu = time.process_time_ns()
    try:
        time.sleep(3)
        cpu = time.process_time_ns()-cpu
        alive = generator._generation_thread.is_alive()
        row = dict(candidate=candidate, elapsed_seconds=time.monotonic()-start,
                   idle_read_entries=entries[0], process_cpu_ms=cpu/1e6,
                   generation_thread_alive=alive)
    finally:
        stopping = time.monotonic()
        generator.stop_and_join()
        stopped_in = time.monotonic()-stopping
        server.ResponseGenerator._next_request = original_read
        server.ResponseGenerator.stop_and_join = original_stop
        if hasattr(server.ResponseGenerator, '_vllm_idle_wait'):
            del server.ResponseGenerator._vllm_idle_wait
    row['shutdown_seconds'] = stopped_in
    rows.append(row)
output.write_text(json.dumps(dict(scope='reviewed scheduler, idle without model inference',
    server_source_sha256=SERVER_SOURCE_SHA256, watts_measured=False, observations=rows), indent=2)+'\n')
print(json.dumps(rows))

if not all(row["generation_thread_alive"] and row["idle_read_entries"] > 0 for row in rows):
    raise SystemExit("invalid idle measurement: generation thread did not remain alive")
