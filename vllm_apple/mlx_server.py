from __future__ import annotations

import argparse
import io
import json
import math
import time
from importlib.metadata import version
from typing import Any

from .generation_admission import GenerationAdmission, GenerationAdmissionError, peer_disconnected

MAXIMUM_CACHE_NODES = 4096
MAXIMUM_METRICS_BYTES = 4096
MAXIMUM_TOKENIZE_REQUEST_BYTES = 8 * 1024 * 1024


class GenerationBodyError(GenerationAdmissionError):
    def __init__(self, code: str, status: int) -> None:
        super().__init__(code)
        self.status = status


def read_generation_body(handler: Any, timeout_seconds: float = 10) -> bytes:
    lengths = handler.headers.get_all("Content-Length", [])
    if len(lengths) != 1 or handler.headers.get("Transfer-Encoding") is not None:
        raise GenerationBodyError("generation_body_framing", 400)
    try:
        length = int(lengths[0])
    except ValueError as error:
        raise GenerationBodyError("generation_body_framing", 400) from error
    if not 0 < length <= MAXIMUM_TOKENIZE_REQUEST_BYTES:
        raise GenerationBodyError("generation_body_size", 413)
    original_timeout = handler.connection.gettimeout()
    deadline = time.monotonic() + timeout_seconds
    body = bytearray()
    try:
        while len(body) < length:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GenerationBodyError("generation_body_timeout", 408)
            handler.connection.settimeout(remaining)
            chunk = handler.rfile.read1(min(65536, length - len(body)))
            if not chunk:
                raise GenerationBodyError("generation_body_incomplete", 400)
            body.extend(chunk)
    except TimeoutError as error:
        raise GenerationBodyError("generation_body_timeout", 408) from error
    finally:
        handler.connection.settimeout(original_timeout)
    return bytes(body)


def prompt_cache_metrics(cache: object) -> dict[str, Any]:
    if hasattr(cache, "cache") and hasattr(cache, "tokens"):
        size, complete = bounded_cache_nbytes(cache.cache)
        return {"kv_cache_bytes": size, "kv_cache_tokens": len(cache.tokens),
                "traversal_complete": complete, "kv_measurement_source": "bounded_array_traversal"}
    size = getattr(cache, "nbytes", None)
    if type(size) is not int or size < 0:
        raise ValueError("unsupported prompt cache metadata")
    return {"kv_cache_bytes": size, "kv_cache_tokens": None,
            "traversal_complete": False, "kv_measurement_source": "backend_lru_accounting"}


def bounded_cache_nbytes(value: object, maximum_nodes: int = MAXIMUM_CACHE_NODES) -> tuple[int, bool]:
    """Count distinct array objects with bounded work and no tensor evaluation.

    The budget includes container entries, aliases and scalar metadata. Iterator
    frames avoid copying a wide cache's children before checking that budget.
    References live only for this traversal, preventing temporary state objects
    from reusing an identity that has already been visited.
    """
    if type(maximum_nodes) is not int or maximum_nodes <= 0:
        raise ValueError("cache traversal budget must be a positive integer")
    pending = [iter((value,))]
    seen: dict[int, object] = {}
    total = 0
    nodes = 0
    while pending:
        try:
            item = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        nodes += 1
        if nodes > maximum_nodes:
            return total, False
        if item is None or isinstance(item, (str, bytes, int, float, bool)):
            continue
        identity = id(item)
        if identity in seen:
            continue
        seen[identity] = item
        nbytes = _array_nbytes(item)
        if isinstance(nbytes, int) and not isinstance(nbytes, bool) and nbytes >= 0:
            total += nbytes
            continue
        if isinstance(item, dict):
            pending.append(iter(item.values()))
        elif isinstance(item, (list, tuple)):
            pending.append(iter(item))
        else:
            try:
                state = item.state
            except (AttributeError, RuntimeError, ValueError):
                continue
            pending.append(iter((state,)))
    return total, True


def _array_nbytes(value: object) -> int | None:
    """Prefer MLX shape/dtype metadata over the array's nbytes property."""
    shape = getattr(value, "shape", None)
    item_size = getattr(getattr(value, "dtype", None), "size", None)
    if (
        isinstance(shape, (tuple, list))
        and type(item_size) is int
        and item_size > 0
        and all(type(dimension) is int and dimension >= 0 for dimension in shape)
    ):
        return math.prod(shape) * item_size
    return getattr(value, "nbytes", None)


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
    parser.add_argument("--allow-concurrent-generation", action="store_true",
                        help="Experimental: bypass the safe generation serialization gate")
    parser.add_argument("--generation-queue-capacity", type=int, default=8)
    parser.add_argument("--generation-queue-timeout", type=float, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("MLX telemetry server must be loopback-only")
    admission = GenerationAdmission(arguments.generation_queue_capacity,
                                    arguments.generation_queue_timeout)

    import mlx.core as mx
    from mlx_lm.server import APIHandler, ModelProvider, run
    modern_server = version("mlx-lm") == "0.32.0"

    provider = mx
    if not hasattr(provider, "get_active_memory") and hasattr(mx, "metal"):
        provider = mx.metal

    class TelemetryHandler(APIHandler):
        def do_POST(self) -> None:
            if self.path != "/tokenize":
                if (self.path in {"/v1/chat/completions", "/v1/completions"}
                        and not arguments.allow_concurrent_generation):
                    body = None
                    def prepare_body() -> None:
                        nonlocal body
                        body = read_generation_body(self)
                    try:
                        with admission.admit(lambda: peer_disconnected(self.connection), prepare_body):
                            original_input = self.rfile
                            self.rfile = io.BytesIO(body)
                            try:
                                super().do_POST()
                            finally:
                                self.rfile = original_input
                    except GenerationAdmissionError as error:
                        if (str(error) == "generation_queue_cancelled"
                                or peer_disconnected(self.connection)):
                            self.close_connection = True
                            return
                        encoded = json.dumps({"error": {"code": str(error)}}).encode()
                        self.close_connection = True
                        self._set_completion_headers(getattr(error, "status", 503))
                        self.send_header("Content-Length", str(len(encoded)))
                        if not isinstance(error, GenerationBodyError):
                            self.send_header("Retry-After", "1")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.wfile.write(encoded)
                    except (BrokenPipeError, ConnectionResetError):
                        self.close_connection = True
                else:
                    super().do_POST()
                return
            try:
                raw_length = self.headers.get("Content-Length")
                length = int(raw_length) if raw_length is not None else -1
                if not 0 <= length <= MAXIMUM_TOKENIZE_REQUEST_BYTES:
                    raise ValueError("invalid Content-Length")
                payload = json.loads(self.rfile.read(length))
                model_provider = (self.response_generator.model_provider if modern_server
                                  else self.model_provider)
                count = tokenize_chat_request(model_provider, payload)
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
            cache = self.response_generator.prompt_cache if modern_server else self.prompt_cache
            cache_metrics = prompt_cache_metrics(cache)
            payload: dict[str, Any] = {
                "schema_version": 1,
                "snapshot_consistency": "non_atomic",
                "generation_admission": (None if arguments.allow_concurrent_generation
                                         else admission.snapshot()),
                "active_bytes": provider.get_active_memory(),
                "cache_bytes": provider.get_cache_memory(),
                "peak_bytes": provider.get_peak_memory(),
                **cache_metrics,
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
        prompt_cache_size=10,
        prompt_cache_bytes=None,
        pipeline=False,
        allowed_origins=[],
        decode_concurrency=32,
        prompt_concurrency=8,
        prefill_step_size=2048,
    )
    if modern_server:
        from mlx_lm.models.cache import LRUPromptCache
        from mlx_lm.server import ResponseGenerator, _run_http_server

        group = mx.distributed.init()
        if group.size() != 1:
            raise RuntimeError("MLX telemetry wrapper supports a single local worker")
        generator = ResponseGenerator(ModelProvider(cli_args), LRUPromptCache(cli_args.prompt_cache_size))
        try:
            _run_http_server(arguments.host, arguments.port, generator, handler_class=TelemetryHandler)
        finally:
            generator.stop_and_join()
        return 0
    run(
        arguments.host,
        arguments.port,
        ModelProvider(cli_args),
        handler_class=TelemetryHandler,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
