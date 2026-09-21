import os
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
