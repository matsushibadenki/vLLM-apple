from __future__ import annotations

import json
import math
import os
import selectors
import signal
import subprocess
import time
from dataclasses import fields
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from .generative_collector import GenerationTelemetryEvent


DEFAULT_MAX_LINE_BYTES = 16 * 1024
MAX_COMMAND_ARGUMENTS = 256
_EVENT_FIELDS = frozenset(field.name for field in fields(GenerationTelemetryEvent))


class GenerativeSubprocessAdapterError(RuntimeError):
    pass


class SubprocessGenerativeTelemetryAdapter:
    """Streams bounded JSONL telemetry from a backend worker without shell execution."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float,
        cwd: Path | None = None,
        environment: Mapping[str, str] | None = None,
        max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    ) -> None:
        if not 1 <= len(command) <= MAX_COMMAND_ARGUMENTS:
            raise ValueError("generative backend command size is invalid")
        if any(
            not argument
            or len(argument.encode("utf-8")) > 16_384
            or any(not character.isprintable() for character in argument)
            for argument in command
        ):
            raise ValueError("generative backend command argument is invalid")
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 24 * 60 * 60:
            raise ValueError("generative backend timeout is outside the supported range")
        if not 1024 <= max_line_bytes <= 1024 * 1024:
            raise ValueError("generative backend line limit is outside the supported range")
        if environment is not None and (
            len(environment) > 256
            or any(
                not key
                or not value
                or len(key.encode("utf-8")) > 256
                or len(value.encode("utf-8")) > 16_384
                or "\x00" in key
                or "\x00" in value
                for key, value in environment.items()
            )
        ):
            raise ValueError("generative backend environment is invalid")
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        self.cwd = cwd.expanduser().resolve() if cwd is not None else None
        self.environment = dict(environment) if environment is not None else None
        self.max_line_bytes = max_line_bytes

    def events(self) -> Iterator[GenerationTelemetryEvent]:
        process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
            start_new_session=True,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + self.timeout_seconds
        buffer = bytearray()
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GenerativeSubprocessAdapterError("generative backend timed out")
                ready = selector.select(min(remaining, 0.25))
                if ready:
                    chunk = os.read(process.stdout.fileno(), 4096)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    if len(buffer) > self.max_line_bytes and b"\n" not in buffer:
                        raise GenerativeSubprocessAdapterError(
                            "generative backend telemetry line limit exceeded"
                        )
                    while b"\n" in buffer:
                        raw, _, remainder = buffer.partition(b"\n")
                        buffer = bytearray(remainder)
                        yield self._decode_event(raw)
                elif process.poll() is not None:
                    break
            if buffer:
                yield self._decode_event(bytes(buffer))
            return_code = process.wait(timeout=1)
            if return_code != 0:
                raise GenerativeSubprocessAdapterError(
                    f"generative backend exited with status {return_code}"
                )
        finally:
            selector.close()
            process.stdout.close()
            if process.poll() is None:
                self._terminate(process)

    def _decode_event(self, raw: bytes) -> GenerationTelemetryEvent:
        if not raw or len(raw) > self.max_line_bytes:
            raise GenerativeSubprocessAdapterError(
                "generative backend telemetry line is empty or oversized"
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GenerativeSubprocessAdapterError(
                "generative backend emitted invalid JSON telemetry"
            ) from error
        if not isinstance(payload, dict) or set(payload) != _EVENT_FIELDS:
            raise GenerativeSubprocessAdapterError(
                "generative backend telemetry fields do not match the contract"
            )
        try:
            return GenerationTelemetryEvent(**payload)
        except (TypeError, ValueError) as error:
            raise GenerativeSubprocessAdapterError(
                "generative backend emitted invalid telemetry values"
            ) from error

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)
