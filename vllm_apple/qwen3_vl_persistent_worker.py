"""Persistent bounded Core ML worker for the fixed-grid Qwen3-VL vision tower."""
from __future__ import annotations

import json
import select
import stat
import subprocess
import threading
from pathlib import Path

_VISION_WORKER_PROGRAM = r'''
import CoreML
import CryptoKit
import Darwin
import Foundation

func emit(_ payload: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!)
    fflush(stdout)
}
func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
func peakRSS() -> Int64 {
    var usage = rusage()
    return getrusage(RUSAGE_SELF, &usage) == 0 ? Int64(usage.ru_maxrss) : -1
}

let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuAndNeuralEngine
var models: [MLModel] = []
for index in 1...5 {
    guard let model = try? MLModel(
        contentsOf: URL(fileURLWithPath: CommandLine.arguments[index]),
        configuration: configuration
    ) else {
        emit(["status": "error", "code": "model_load_failed"])
        exit(2)
    }
    models.append(model)
}
emit(["status": "ready", "model_load_count": models.count, "peak_rss_bytes": peakRSS()])
while let line = readLine() {
  autoreleasepool {
    guard let encoded = line.data(using: .utf8),
          let request = try? JSONSerialization.jsonObject(with: encoded) as? [String: Any],
          let operation = request["operation"] as? String else {
        emit(["status": "error", "code": "invalid_request"]); return
    }
    if operation == "shutdown" {
        emit(["status": "shutdown", "peak_rss_bytes": peakRSS()]); exit(0)
    }
    guard operation == "predict",
          let pixelPath = request["pixel_path"] as? String,
          let outputPath = request["output_directory"] as? String,
          let pixelData = try? Data(contentsOf: URL(fileURLWithPath: pixelPath)),
          pixelData.count == 256 * 1536 * 2,
          let pixels = try? MLMultiArray(shape: [256, 1536], dataType: .float16) else {
        emit(["status": "error", "code": "invalid_request"]); return
    }
    pixelData.withUnsafeBytes { raw in
        pixels.dataPointer.copyMemory(from: raw.baseAddress!, byteCount: pixelData.count)
    }
    do {
        let started = DispatchTime.now().uptimeNanoseconds
        let patchProvider = try MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: pixels)
        ])
        let patchResult = try models[0].prediction(from: patchProvider)
        guard var current = patchResult.featureValue(for: "patch_hidden_states")?.multiArrayValue else {
            throw NSError(domain: "vllm-apple", code: 1)
        }
        var digests: [String] = []
        for index in 0..<4 {
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "hidden_states": MLFeatureValue(multiArray: current)
            ])
            let result = try models[index + 1].prediction(from: provider)
            guard let main = result.featureValue(for: "tower_hidden_states")?.multiArrayValue else {
                throw NSError(domain: "vllm-apple", code: 2)
            }
            let name = index == 3 ? "final_hidden_states" : "deepstack_hidden_states"
            guard let merged = result.featureValue(for: name)?.multiArrayValue else {
                throw NSError(domain: "vllm-apple", code: 3)
            }
            let data = Data(bytes: merged.dataPointer, count: merged.count * 2)
            let file = index == 3 ? "final.fp16" : "deepstack_\(index).fp16"
            let url = URL(fileURLWithPath: outputPath).appendingPathComponent(file)
            try data.write(to: url, options: [.atomic])
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
            digests.append(digest(data))
            current = main
        }
        emit([
            "status": "ok", "output_digests": digests,
            "latency_nanoseconds": DispatchTime.now().uptimeNanoseconds - started,
            "peak_rss_bytes": peakRSS()
        ])
    } catch {
        emit(["status": "error", "code": "prediction_failed"])
    }
  }
}
'''.strip()


class Qwen3VLPersistentWorker:
    """Own one five-model Swift process and serialize a bounded request queue."""

    def __init__(
        self,
        compiled_models: tuple[Path, Path, Path, Path, Path],
        *,
        timeout_seconds: float = 30,
        maximum_pending_requests: int = 8,
    ) -> None:
        if timeout_seconds <= 0 or not 1 <= maximum_pending_requests <= 64:
            raise ValueError("invalid Qwen3-VL persistent worker bounds")
        self._models = tuple(model.expanduser().resolve(strict=True) for model in compiled_models)
        if any(model.suffix != ".mlmodelc" or not model.is_dir() for model in self._models):
            raise ValueError("persistent worker requires five compiled models")
        self._timeout = timeout_seconds
        self._slots = threading.BoundedSemaphore(maximum_pending_requests)
        self._lock = threading.RLock()
        self._closed = False
        self.restart_count = 0
        self._process: subprocess.Popen[bytes] | None = None
        self._start()

    def _start(self) -> None:
        self._process = subprocess.Popen(
            ["/usr/bin/swift", "-e", _VISION_WORKER_PROGRAM, *map(str, self._models)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        payload = self._read()
        if payload.get("status") != "ready" or payload.get("model_load_count") != 5:
            self.close()
            raise RuntimeError("Qwen3-VL persistent worker failed to load")

    def _read(self) -> dict[str, object]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("Qwen3-VL persistent worker is unavailable")
        readable, _, _ = select.select((process.stdout,), (), (), self._timeout)
        if not readable:
            self._stop()
            raise TimeoutError("Qwen3-VL persistent worker timed out")
        line = process.stdout.readline(65_537)
        if not line or len(line) > 65_536 or not line.endswith(b"\n"):
            self._stop()
            raise RuntimeError("Qwen3-VL persistent worker returned invalid output")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            self._stop()
            raise RuntimeError(
                "Qwen3-VL persistent worker returned invalid JSON"
            ) from error
        if not isinstance(payload, dict):
            raise RuntimeError("Qwen3-VL persistent worker returned invalid output")
        return payload

    def predict(self, pixel_path: Path, output_directory: Path) -> dict[str, object]:
        pixels = pixel_path.expanduser().resolve(strict=True)
        output = output_directory.expanduser().resolve(strict=True)
        if (
            pixels.is_symlink()
            or pixels.stat().st_size != 256 * 1536 * 2
            or stat.S_IMODE(output.stat().st_mode) & 0o077
            or any(output.iterdir())
        ):
            raise ValueError("invalid Qwen3-VL persistent worker request")
        if not self._slots.acquire(timeout=self._timeout):
            raise TimeoutError("Qwen3-VL persistent worker queue is full")
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Qwen3-VL persistent worker is closed")
                if self._process is None or self._process.poll() is not None:
                    self.restart_count += 1
                    self._start()
                assert self._process.stdin is not None
                request = json.dumps(
                    {
                        "operation": "predict",
                        "pixel_path": str(pixels),
                        "output_directory": str(output),
                    },
                    separators=(",", ":"),
                ).encode() + b"\n"
                try:
                    self._process.stdin.write(request)
                    self._process.stdin.flush()
                except (BrokenPipeError, OSError) as error:
                    self._stop()
                    raise RuntimeError("Qwen3-VL persistent worker exited") from error
                payload = self._read()
                if (
                    payload.get("status") != "ok"
                    or not isinstance(payload.get("latency_nanoseconds"), int)
                    or not isinstance(payload.get("peak_rss_bytes"), int)
                    or not isinstance(payload.get("output_digests"), list)
                    or len(payload["output_digests"]) != 4
                ):
                    raise RuntimeError("Qwen3-VL persistent worker prediction failed")
                return payload
        finally:
            self._slots.release()

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def close(self) -> bool:
        with self._lock:
            if self._closed:
                return True
            self._closed = True
            clean = False
            process = self._process
            if process is not None and process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write(b'{"operation":"shutdown"}\n')
                    process.stdin.flush()
                    clean = self._read().get("status") == "shutdown"
                except (OSError, RuntimeError, TimeoutError):
                    clean = False
            self._stop()
            return clean

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None
