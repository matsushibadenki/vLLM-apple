from __future__ import annotations

import argparse
import json

from .context import recommend_context
from .daemon import serve
from .hardware import detect_hardware
from .profile import build_profile, save_profile
from .types import GIB, ModelMemorySpec


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-apple")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("hardware", help="detect Apple hardware and memory")

    profile = commands.add_parser("profile", help="build a runtime profile")
    profile.add_argument("--save", action="store_true")
    profile.add_argument("--output")

    context = commands.add_parser("context", help="calculate safe context tiers")
    context.add_argument("--model-id", default="unknown")
    context.add_argument("--model-memory-gb", type=float, required=True)
    context.add_argument("--kv-bytes-per-token", type=int, required=True)
    context.add_argument("--workspace-gb", type=float, default=0.0)
    context.add_argument("--model-max-context", type=int)

    server = commands.add_parser("serve", help="run the local control daemon")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--max-concurrent-requests", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "hardware":
        _json(detect_hardware().to_dict())
        return 0
    if arguments.command == "profile":
        profile = build_profile()
        if arguments.save:
            from pathlib import Path

            path = save_profile(profile, Path(arguments.output) if arguments.output else None)
            print(path)
        else:
            _json(profile.to_dict())
        return 0
    if arguments.command == "context":
        if arguments.model_memory_gb < 0 or arguments.workspace_gb < 0:
            raise SystemExit("memory values cannot be negative")
        hardware = detect_hardware()
        spec = ModelMemorySpec(
            model_id=arguments.model_id,
            weights_bytes=int(arguments.model_memory_gb * GIB),
            kv_bytes_per_token=arguments.kv_bytes_per_token,
            workspace_bytes=int(arguments.workspace_gb * GIB),
            model_max_context=arguments.model_max_context,
        )
        _json(recommend_context(hardware.memory, spec).to_dict())
        return 0
    if arguments.command == "serve":
        serve(arguments.host, arguments.port, arguments.max_concurrent_requests)
        return 0
    return 2
