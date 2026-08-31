"""PyInstaller entry point for the standalone macOS daemon artifact."""

from vllm_apple.daemon import main


if __name__ == "__main__":
    raise SystemExit(main())
