"""JSON-lines worker that owns an inference delegate on process main thread."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import queue
import re
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .inference_request import InferenceRequestCancelled, InferenceRequestContext
from .process_inference_engine import MAX_PROCESS_MESSAGE_BYTES

_FACTORY = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*\Z")
_IDENTIFIER = re.compile(r"[0-9a-f]{32}\Z")
_WRITE_LOCK = threading.Lock()


def _write(payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    with _WRITE_LOCK:
        sys.stdout.buffer.write(encoded)
        sys.stdout.buffer.flush()


def _factory(reference: str, config_path: Path | None) -> object:
    if _FACTORY.fullmatch(reference) is None:
        raise ValueError("invalid worker factory reference")
    module_name, attribute = reference.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise ValueError("worker factory is unavailable")
    if config_path is None:
        return factory()
    descriptor = os.open(
        config_path.expanduser().absolute(),
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o022
                or not 0 < info.st_size <= 1024 * 1024):
            raise ValueError("worker configuration file is unsafe")
        content = bytearray()
        while len(content) <= 1024 * 1024:
            chunk = os.read(descriptor, min(8192, 1024 * 1024 + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
        final = os.fstat(descriptor)
        if (len(content) != info.st_size or info.st_ino != final.st_ino
                or info.st_mtime_ns != final.st_mtime_ns
                or info.st_size != final.st_size):
            raise ValueError("worker configuration changed while reading")
    finally:
        os.close(descriptor)
    config = json.loads(content)
    if not isinstance(config, dict):
        raise ValueError("worker configuration must be an object")
    return factory(config)


def _reader(
    commands: queue.Queue[dict[str, Any] | None],
    cancellations: dict[str, threading.Event],
    cancellation_lock: threading.Lock,
) -> None:
    while True:
        line = sys.stdin.buffer.readline(MAX_PROCESS_MESSAGE_BYTES + 1)
        if not line:
            commands.put(None)
            return
        if len(line) > MAX_PROCESS_MESSAGE_BYTES or not line.endswith(b"\n"):
            _write({"type": "error", "code": "request_too_large"})
            commands.put(None)
            return
        try:
            payload = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            _write({"type": "error", "code": "invalid_request"})
            continue
        if not isinstance(payload, dict):
            _write({"type": "error", "code": "invalid_request"})
            continue
        kind = payload.get("type")
        if kind == "shutdown":
            commands.put(None)
            return
        identifier = payload.get("id")
        if not isinstance(identifier, str) or _IDENTIFIER.fullmatch(identifier) is None:
            _write({"type": "error", "code": "invalid_request"})
            continue
        if kind == "cancel":
            with cancellation_lock:
                cancellation = cancellations.get(identifier)
            if cancellation is not None:
                cancellation.set()
            continue
        if kind == "diagnostics":
            try:
                commands.put_nowait(payload)
            except queue.Full:
                _write({"type": "error", "id": identifier, "code": "engine_busy"})
            continue
        if (kind != "request" or not isinstance(payload.get("request"), dict)
                or not isinstance(payload.get("timeout_seconds"), (int, float))
                or not 0 < payload["timeout_seconds"] <= 1800):
            _write({"type": "error", "id": identifier, "code": "invalid_request"})
            continue
        cancellation = threading.Event()
        with cancellation_lock:
            if identifier in cancellations:
                _write({"type": "error", "id": identifier, "code": "duplicate_request"})
                continue
            cancellations[identifier] = cancellation
        try:
            commands.put_nowait(payload)
        except queue.Full:
            with cancellation_lock:
                cancellations.pop(identifier, None)
            _write({"type": "error", "id": identifier, "code": "engine_busy"})


def run(factory: str, config: Path | None, maximum_pending_requests: int) -> int:
    delegate = _factory(factory, config)
    models = getattr(delegate, "models", None)
    model_values = list(models()) if callable(models) else []
    _write({"type": "ready", "models": model_values})
    commands: queue.Queue[dict[str, Any] | None] = queue.Queue(maximum_pending_requests)
    cancellations: dict[str, threading.Event] = {}
    cancellation_lock = threading.Lock()
    reader = threading.Thread(
        target=_reader,
        args=(commands, cancellations, cancellation_lock),
        name="vllm-apple-process-command-reader",
        daemon=True,
    )
    reader.start()
    try:
        while True:
            command = commands.get()
            if command is None:
                return 0
            identifier = command["id"]
            if command["type"] == "diagnostics":
                try:
                    diagnostics = getattr(delegate, "diagnostics")
                    result = diagnostics()
                    if not isinstance(result, dict):
                        raise RuntimeError("diagnostics returned a non-object")
                    _write({"type": "result", "id": identifier, "result": result})
                except BaseException:
                    _write({
                        "type": "error", "id": identifier,
                        "code": "diagnostics_failed",
                    })
                continue
            with cancellation_lock:
                cancellation = cancellations[identifier]
            context = InferenceRequestContext(
                identifier,
                time.monotonic() + float(command["timeout_seconds"]),
                cancellation,
            )
            try:
                method = getattr(delegate, "chat_completions_with_request_context")
                result = method(command["request"], None, context)
                if not isinstance(result, dict):
                    raise RuntimeError("delegate returned a non-object")
                context.raise_if_cancelled()
                _write({"type": "result", "id": identifier, "result": result})
            except InferenceRequestCancelled as error:
                code = (
                    "request_timed_out"
                    if "timed out" in str(error) else "request_cancelled"
                )
                _write({"type": "error", "id": identifier, "code": code})
            except BaseException:
                _write({"type": "error", "id": identifier, "code": "delegate_failed"})
            finally:
                with cancellation_lock:
                    cancellations.pop(identifier, None)
    finally:
        close = getattr(delegate, "close", None)
        if callable(close):
            close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--maximum-pending-requests", type=int, default=8)
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.maximum_pending_requests <= 64:
        raise ValueError("invalid maximum pending request count")
    return run(arguments.factory, arguments.config, arguments.maximum_pending_requests)


if __name__ == "__main__":
    raise SystemExit(main())
