import json
import os
import sys
import threading
import time
from pathlib import Path


class Delegate:
    def __init__(self, config):
        self.marker = Path(config["marker"])
        self.sequence = 0

    @property
    def ready(self):
        return True

    def models(self):
        return [{"id": "process-fixture", "object": "model"}]

    def chat_completions_with_request_context(self, request, _kernel, context):
        self.sequence += 1
        if request.get("entered"):
            Path(request["entered"]).touch()
        if request.get("emit_invalid"):
            while not Path(request["emit_invalid"]).exists():
                time.sleep(0.01)
        if request.get("invalid_terminal"):
            payload = dict(request["invalid_terminal"], id=context.request_id)
            sys.stdout.write(json.dumps(payload) + "\n")
            sys.stdout.flush()
        if request.get("close_stdout"):
            os.close(sys.stdout.fileno())
        if request.get("release"):
            while not Path(request["release"]).exists():
                time.sleep(0.01)
        if request.get("exit"):
            os._exit(7)
        delay = request.get("delay", 0)
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            context.raise_if_cancelled()
            time.sleep(0.01)
        context.raise_if_cancelled()
        return {
            "pid": os.getpid(),
            "main_thread": threading.current_thread() is threading.main_thread(),
            "sequence": self.sequence,
            "value": request.get("value"),
        }

    def close(self):
        self.marker.write_text(f"{os.getpid()}:{threading.current_thread() is threading.main_thread()}")
        return True

    def diagnostics(self):
        return {
            "pid": os.getpid(),
            "main_thread": threading.current_thread() is threading.main_thread(),
            "sequence": self.sequence,
        }


def create_delegate(config):
    return Delegate(config)
