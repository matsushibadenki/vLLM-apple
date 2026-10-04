"""Experimental fixed-model MLX continuous batching and actual KV reuse."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import threading
from pathlib import Path

from vllm_apple.bounded_http import BoundedHTTPServer
from vllm_apple.mlx_gemma2_compat import install_cancel_api, install_gemma2_batch_mask_fix

from .cache import IdentityPromptCache


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cache-salt', default='')
    parser.add_argument('--disable-prefix-cache', action='store_true')
    parser.add_argument('--host', default='127.0.0.1')
    options, arguments = parser.parse_known_args()
    if len(options.cache_salt.encode()) > 128:
        raise ValueError('cache salt exceeds 128 bytes')
    if options.host not in {'127.0.0.1', '::1', 'localhost'}:
        raise ValueError('P2 backend must be loopback-only')
    if os.environ.get('VLLM_APPLE_P1_PROFILE') == '1':
        raise RuntimeError('P2 must not inherit P1 qualification/profile')
    install_gemma2_batch_mask_fix()
    from mlx_lm import server
    generate = importlib.import_module('mlx_lm.generate')
    import mlx.core as mx
    if importlib.metadata.version('mlx') != '0.32.1' or mx.default_device() != mx.gpu:
        raise RuntimeError('P2 requires reviewed MLX 0.32.1 with GPU default device')
    mx.set_memory_limit(8*1024**3)
    mx.set_cache_limit(256*1024**2)

    # Binding private APIs to the reviewed source prevents silent version drift.
    cache_module = importlib.import_module('mlx_lm.models.cache')
    if (_sha(Path(generate.__file__)) != '5a57043b5a6497450bce14447db3caf570ffdd22adaab34e659b31aceddda522'
            or _sha(Path(cache_module.__file__)) != '440709018cc528ee1e4e42e61ff8713ed2e0079566d9e8fa58eed3a92d334404'):
        raise ValueError('P2 requires the reviewed scheduler/cache source hashes')
    base_handler = install_cancel_api()
    lock = threading.Lock()
    metrics = dict(decode_steps=0, multi_sequence_decode_steps=0, maximum_decode_width=0,
                   submitted_prefill_tokens=0, scheduler_admissions=0)
    original_next = generate.BatchGenerator.next
    original_insert = generate.BatchGenerator.insert_segments

    def next_batch(self):
        result = original_next(self)
        width = len({response.uid for response in result[1]})
        with lock:
            metrics['decode_steps'] += int(width > 0)
            metrics['multi_sequence_decode_steps'] += int(width > 1)
            metrics['maximum_decode_width'] = max(metrics['maximum_decode_width'], width)
        return result

    def insert(self, segments, *args, **kwargs):
        result = original_insert(self, segments, *args, **kwargs)
        with lock:
            metrics['submitted_prefill_tokens'] += sum(len(segment) for sequence in segments for segment in sequence)
            metrics['scheduler_admissions'] += len(result)
        return result

    generate.BatchGenerator.next = next_batch
    generate.BatchGenerator.insert_segments = insert
    original_init = server.ResponseGenerator.__init__
    original_generate = server.ResponseGenerator.generate

    def init(self, provider, cache):
        cli = provider.cli_args
        if (cli.adapter_path or cli.draft_model or cli.decode_concurrency > 4
                or cli.prompt_concurrency > 2 or cli.prefill_step_size > 512 or cli.prompt_cache_size > 8):
            raise ValueError('P2 requires no adapters/drafts, decode<=4, prompt<=2, prefill<=512, cache<=8')
        model = Path(cli.model).resolve()
        if json.loads((model/'config.json').read_text())['model_type'] != 'gemma2':
            raise ValueError('P2 experiment is restricted to Gemma 2')
        identity = dict(model_files={p.name: _sha(p) for p in sorted(model.iterdir()) if p.is_file()},
                        cache_salt=options.cache_salt, position_origin=0,
                        template_policy=dict(chat_template=cli.chat_template,
                            use_default=cli.use_default_chat_template, args=cli.chat_template_args),
                        mlx_lm=importlib.metadata.version('mlx-lm'),
                        mlx=importlib.metadata.version('mlx'),
                        scheduler_source_sha256=_sha(Path(generate.__file__)),
                        cache_source_sha256=_sha(Path(sys.modules[cache.__class__.__module__].__file__)))
        namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cli.prompt_cache_bytes = 256*1024**2
        cache.max_bytes = 256*1024**2
        original_init(self, provider, IdentityPromptCache(cache, namespace, enabled=not options.disable_prefix_cache))

    def generate_request(self, request, args, *rest, **kwargs):
        cli = self.model_provider.cli_args
        if (args.model.model not in {cli.model, 'default_model'} or args.model.adapter is not None
                or args.model.draft not in {None, 'default_model'} or type(args.max_tokens) is not int
                or not 1 <= args.max_tokens <= 512 or args.top_logprobs > 5 or args.chat_template_kwargs):
            raise ValueError('P2 fixed-model/output bounds violated')
        return original_generate(self, request, args, *rest, **kwargs)

    original_tokenize = server.ResponseGenerator._tokenize

    def tokenize(self, tokenizer, request, args):
        result = original_tokenize(self, tokenizer, request, args)
        if len(result[0])+args.max_tokens > 4096:
            raise ValueError('P2 context exceeds 4096 tokens')
        return result

    server.ResponseGenerator.__init__ = init
    server.ResponseGenerator.generate = generate_request
    server.ResponseGenerator._tokenize = tokenize

    class Handler(base_handler):
        def handle_completion(self, request, stop_words):
            if 'cache_salt' in self.body:
                self._set_completion_headers(400)
                self.end_headers()
                self.wfile.write(b'{"error":"cache salt is process-scoped"}')
                return
            return super().handle_completion(request, stop_words)

        def do_GET(self):
            if self.path != '/vllm-apple/p2':
                return super().do_GET()
            with lock:
                payload = dict(scheduler=dict(metrics), cache=self.response_generator.prompt_cache.snapshot(),
                               qualification=False, experimental=True)
            encoded = json.dumps(payload).encode()
            self._set_completion_headers(200)
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server._run_http_server.__defaults__ = (BoundedHTTPServer, Handler)
    sys.argv = [sys.argv[0], *arguments, '--host', options.host]
    server.main()


if __name__ == '__main__':
    main()
