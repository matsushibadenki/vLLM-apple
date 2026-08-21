from __future__ import annotations

import argparse
import signal
import threading

from .api import create_server
from .service import RuntimeService
from .types import RuntimeState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-appled")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-concurrent-requests", type=int, default=32)
    return parser


def serve(host: str = "127.0.0.1", port: int = 8000, max_concurrent_requests: int = 32) -> None:
    service = RuntimeService()
    server = create_server(host, port, service, max_concurrent_requests=max_concurrent_requests)

    def stop(_signum: int, _frame: object) -> None:
        service.set_state(RuntimeState.STOPPING)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        service.set_state(RuntimeState.STOPPED)


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    serve(arguments.host, arguments.port, arguments.max_concurrent_requests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
