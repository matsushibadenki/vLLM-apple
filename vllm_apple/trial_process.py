"""Bounded local trials in an owned process group, with graceful cleanup."""
import os
import signal
import subprocess


def run_trial(command, *, cwd=None, env=None, stdout=None, timeout=180, grace=45):
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stdout,
                               start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        # The group is owned by this invocation. Remove any surviving children,
        # even when the runner exited before completing its own cleanup.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
