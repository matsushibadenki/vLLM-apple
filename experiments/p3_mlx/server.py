"""Experimental P2 worker with bounded allocator/effective-setting observations."""
from __future__ import annotations

import json

from experiments.p2_mlx import server as p2


def install_profiled_http_server(native):
    original = native._run_http_server

    # P2 assigns these two defaults after constructing its final Handler class.
    def run(host, port, response_generator, server_class=None, handler_class=None):
        class Handler(handler_class):
            def do_GET(self):
                if self.path != '/vllm-apple/p3':
                    return super().do_GET()
                import mlx.core as mx
                cli = self.response_generator.model_provider.cli_args
                payload = dict(peak_active_bytes=mx.get_peak_memory(),
                    active_bytes=mx.get_active_memory(), cache_bytes=mx.get_cache_memory(),
                    memory_source='mlx_allocator_peak_active_not_RSS',
                    prefill_step_size=cli.prefill_step_size,
                    decode_concurrency=cli.decode_concurrency,
                    prompt_concurrency=cli.prompt_concurrency,
                    cache_enabled=self.response_generator.prompt_cache.enabled,
                    qualification=False)
                raw = json.dumps(payload).encode()
                self._set_completion_headers(200)
                self.send_header('Content-Length',str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        return original(host,port,response_generator,server_class,Handler)
    native._run_http_server = run


def main():
    from mlx_lm import server
    install_profiled_http_server(server)
    p2.main()


if __name__ == '__main__':
    main()
