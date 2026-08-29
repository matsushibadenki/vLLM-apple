from __future__ import annotations

import argparse
import json
from typing import Any

MAXIMUM_CACHE_NODES = 4096
MAXIMUM_METRICS_BYTES = 4096
MAXIMUM_TOKENIZE_REQUEST_BYTES = 8 * 1024 * 1024


def bounded_cache_nbytes(value: object, maximum_nodes: int = MAXIMUM_CACHE_NODES) -> tuple[int, bool]:
    """Count distinct MLX array storage without retaining or materializing tensors."""
    pending = [value]
    seen: set[int] = set()
    total = 0
    nodes = 0
    while pending:
        item = pending.pop()
        if item is None or isinstance(item, (str, bytes, int, float, bool)):
            continue
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        nodes += 1
        if nodes > maximum_nodes:
            return total, False
        nbytes = getattr(item, "nbytes", None)
        if isinstance(nbytes, int) and not isinstance(nbytes, bool) and nbytes >= 0:
            total += nbytes
            continue
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        else:
            try:
                state = item.state
            except (AttributeError, RuntimeError, ValueError):
                continue
            pending.append(state)
    return total, True


def tokenize_chat_request(model_provider: object, payload: object) -> int:
    """Return a chat-template token count without exposing or retaining token IDs."""
    if not isinstance(payload, dict):
        raise ValueError("request must be an object")
    if payload.get("model", "default_model") != "default_model":
        raise ValueError("MLX wrapper accepts only the preloaded model")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    for message in messages:
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("messages contain an invalid item")
    add_generation_prompt = payload.get("add_generation_prompt", True)
    if not isinstance(add_generation_prompt, bool):
        raise ValueError("add_generation_prompt must be boolean")
    _, tokenizer = model_provider.load(
        "default_model", draft_model_path="default_model"
    )
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    count = len(tokens)
    if count <= 0:
        raise ValueError("tokenizer returned no tokens")
    return count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-apple-mlx-server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log-level", default="WARNING")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("MLX telemetry server must be loopback-only")

    import mlx.core as mx
    from mlx_lm.server import APIHandler, ModelProvider, run

    provider = mx
    if not hasattr(provider, "get_active_memory") and hasattr(mx, "metal"):
        provider = mx.metal

    class TelemetryHandler(APIHandler):
        def do_POST(self) -> None:
            if self.path != "/tokenize":
                super().do_POST()
                return
            try:
                raw_length = self.headers.get("Content-Length")
                length = int(raw_length) if raw_length is not None else -1
                if not 0 <= length <= MAXIMUM_TOKENIZE_REQUEST_BYTES:
                    raise ValueError("invalid Content-Length")
                payload = json.loads(self.rfile.read(length))
                count = tokenize_chat_request(self.model_provider, payload)
                encoded = json.dumps({"count": count}, separators=(",", ":")).encode()
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
                self._set_completion_headers(400)
                self.end_headers()
                return
            self._set_completion_headers(200)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)
            self.wfile.flush()

        def do_GET(self) -> None:
            if self.path != "/v1/vllm-apple/memory":
                super().do_GET()
                return
            kv_bytes, complete = bounded_cache_nbytes(self.prompt_cache.cache)
            payload: dict[str, Any] = {
                "schema_version": 1,
                "active_bytes": provider.get_active_memory(),
                "cache_bytes": provider.get_cache_memory(),
                "peak_bytes": provider.get_peak_memory(),
                "kv_cache_bytes": kv_bytes,
                "kv_cache_tokens": len(self.prompt_cache.tokens),
                "traversal_complete": complete,
            }
            encoded = json.dumps(payload, separators=(",", ":")).encode()
            if len(encoded) > MAXIMUM_METRICS_BYTES:
                self._set_completion_headers(503)
                self.end_headers()
                return
            self._set_completion_headers(200)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)
            self.wfile.flush()

    cli_args = argparse.Namespace(
        model=arguments.model,
        adapter_path=None,
        draft_model=None,
        num_draft_tokens=3,
        trust_remote_code=False,
        chat_template="",
        use_default_chat_template=False,
        temp=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        max_tokens=512,
        chat_template_args={},
    )
    run(
        arguments.host,
        arguments.port,
        ModelProvider(cli_args),
        handler_class=TelemetryHandler,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
