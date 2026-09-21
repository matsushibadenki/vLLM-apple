from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import time
from pathlib import Path
from typing import Sequence


MAX_CONVERSION_OUTPUT_BYTES = 64 * 1024


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def run_qwen_image_21_conversion_worker(
    executable: str | Path,
    source: str | Path,
    output: str | Path,
    *,
    timeout_seconds: float,
    command: Sequence[str] | None = None,
) -> dict[str, object]:
    if not 0 < timeout_seconds <= 24 * 60 * 60:
        raise ValueError("conversion timeout is outside the supported range")
    worker_command = tuple(command) if command is not None else (
        str(Path(executable).expanduser().absolute()),
        "-m",
        "vllm_apple.qwen_image_21_streaming_conversion_worker",
        str(Path(source).expanduser().absolute()),
        str(Path(output).expanduser().absolute()),
    )
    process = subprocess.Popen(
        worker_command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        start_new_session=True,
    )
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout_seconds
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Qwen-Image-2.1 conversion worker timed out")
            for key, _ in selector.select(min(remaining, 0.25)):
                chunk = os.read(key.fileobj.fileno(), 4096)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffer = buffers[key.data]
                buffer.extend(chunk)
                if len(buffer) > MAX_CONVERSION_OUTPUT_BYTES:
                    raise RuntimeError("Qwen-Image-2.1 conversion worker output exceeded limit")
        return_code = process.wait(timeout=1)
        if return_code != 0:
            detail = buffers["stderr"].decode("utf-8", errors="replace").strip()[-512:]
            raise RuntimeError(
                f"Qwen-Image-2.1 conversion worker exited with status {return_code}: {detail}"
            )
        try:
            report = json.loads(buffers["stdout"].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Qwen-Image-2.1 conversion worker emitted invalid JSON") from error
        if not isinstance(report, dict) or report.get("passed") is not True:
            raise RuntimeError("Qwen-Image-2.1 conversion worker report is invalid")
        return report
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
        _stop_process_group(process)
