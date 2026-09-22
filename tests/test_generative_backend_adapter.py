from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_apple.generative_backend_adapter import (
    GenerativeBackendFamily,
    VersionedGenerativeWorkerAdapter,
)


class VersionedGenerativeWorkerAdapterTests(unittest.TestCase):
    def test_active_python_is_trusted_even_with_hosted_toolcache_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "python"
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o777)
            adapter = VersionedGenerativeWorkerAdapter(
                GenerativeBackendFamily.DIFFUSERS,
                backend_version="1",
                supported_backend_versions=("1",),
                python_executable=executable,
            )
            self.assertFalse(adapter.detect().executable)
            with patch.object(sys, "executable", str(executable)):
                self.assertTrue(adapter.detect().executable)

    def test_root_owned_system_python_is_accepted(self) -> None:
        system_python = Path("/usr/bin/python3")
        if not system_python.exists() or system_python.stat().st_uid != 0:
            self.skipTest("root-owned system Python is unavailable")
        adapter = VersionedGenerativeWorkerAdapter(
            GenerativeBackendFamily.DIFFUSERS,
            backend_version="1",
            supported_backend_versions=("1",),
            python_executable=system_python,
        )
        self.assertTrue(adapter.detect().executable)

    def test_builds_fixed_python_worker_commands_for_frameworks(self) -> None:
        for family, module in (
            (GenerativeBackendFamily.DIFFUSERS, "vllm_apple.diffusers_generation_worker"),
            (GenerativeBackendFamily.MLX_GEN, "vllm_apple.mlx_gen_generation_worker"),
            (GenerativeBackendFamily.MFLUX, "vllm_apple.mflux_generation_worker"),
        ):
            adapter = VersionedGenerativeWorkerAdapter(
                family,
                backend_version="1.2.3",
                supported_backend_versions=("1.2.3",),
                python_executable=Path(sys.executable),
            )
            self.assertTrue(adapter.detect().executable)
            self.assertEqual(
                adapter.worker_command(),
                (str(Path(sys.executable).absolute()), "-m", module),
            )

    def test_comfyui_external_worker_requires_contract_arguments(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            worker = Path(directory) / "comfy-worker"
            worker.write_text("#!/bin/sh\nexit 0\n")
            worker.chmod(0o700)
            adapter = VersionedGenerativeWorkerAdapter(
                GenerativeBackendFamily.COMFYUI,
                backend_version="0.3.76",
                supported_backend_versions=("0.3.76",),
                external_worker=worker,
            )
            self.assertEqual(
                adapter.worker_command(),
                (
                    str(worker.absolute()),
                    "--telemetry-contract",
                    "vllm-apple-generative-jsonl-v1",
                    "--backend-version",
                    "0.3.76",
                ),
            )

    def test_unknown_version_and_unsafe_worker_fail_closed(self) -> None:
        unknown = VersionedGenerativeWorkerAdapter(
            GenerativeBackendFamily.DIFFUSERS,
            backend_version="2.0.0",
            supported_backend_versions=("1.0.0",),
            python_executable=Path(sys.executable),
        )
        self.assertFalse(unknown.detect().executable)
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            worker = Path(directory) / "worker"
            worker.write_text("#!/bin/sh\n")
            worker.chmod(0o722)
            unsafe = VersionedGenerativeWorkerAdapter(
                GenerativeBackendFamily.COMFYUI,
                backend_version="1.0.0",
                supported_backend_versions=("1.0.0",),
                external_worker=worker,
            )
            self.assertIn("worker_executable_unsafe_or_missing", unsafe.detect().issues)

    def test_request_must_be_real_file_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            request = root / "request.json"
            request.write_text("{}")
            adapter = VersionedGenerativeWorkerAdapter(
                GenerativeBackendFamily.DIFFUSERS,
                backend_version="1",
                supported_backend_versions=("1",),
                python_executable=Path(sys.executable),
            )
            telemetry = adapter.telemetry_adapter(
                request_path=request,
                workspace_root=root,
                timeout_seconds=1,
            )
            self.assertEqual(telemetry.command[-4:], (
                "--request", str(request.resolve()), "--workspace-root", str(root.resolve())
            ))
            link = root / "link.json"
            os.symlink(request, link)
            with self.assertRaises(ValueError):
                adapter.telemetry_adapter(
                    request_path=link,
                    workspace_root=root,
                    timeout_seconds=1,
                )


if __name__ == "__main__":
    unittest.main()
