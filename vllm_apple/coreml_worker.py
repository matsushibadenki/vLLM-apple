"""Bounded persistent Core ML worker using public Swift/Core ML APIs."""
from __future__ import annotations

import json
import math
import select
import subprocess
import threading
from pathlib import Path

from .ane_probe import CoreMLANEModelProbeConfig, CoreMLPrediction

_WORKER_PROGRAM = r'''
import CoreML
import Foundation

func emit(_ payload: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]) else {
        print("{\"code\":\"encoding_failed\",\"status\":\"error\"}")
        fflush(stdout)
        return
    }
    print(String(data: data, encoding: .utf8)!)
    fflush(stdout)
}

let modelURL = URL(fileURLWithPath: CommandLine.arguments[1])
let inputName = CommandLine.arguments[2]
let outputName = CommandLine.arguments[3]
let expectedCount = Int(CommandLine.arguments[4])!
let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuAndNeuralEngine
guard let model = try? MLModel(contentsOf: modelURL, configuration: configuration) else {
    emit(["status": "error", "code": "model_load_failed"])
    exit(2)
}
emit(["status": "ready"])
while let line = readLine() {
    autoreleasepool {
        guard let data = line.data(using: .utf8),
              let request = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let values = request["input_values"] as? [NSNumber],
              values.count == expectedCount,
              let array = try? MLMultiArray(
                  shape: [NSNumber(value: expectedCount)], dataType: .float32
              ) else {
            emit(["status": "error", "code": "invalid_request"])
            return
        }
        for (index, value) in values.enumerated() { array[index] = value }
        guard let provider = try? MLDictionaryFeatureProvider(
                  dictionary: [inputName: MLFeatureValue(multiArray: array)]
              ) else {
            emit(["status": "error", "code": "provider_failed"])
            return
        }
        let started = DispatchTime.now().uptimeNanoseconds
        guard let prediction = try? model.prediction(from: provider),
              let output = prediction.featureValue(for: outputName)?.multiArrayValue else {
            emit(["status": "error", "code": "prediction_failed"])
            return
        }
        let elapsed = DispatchTime.now().uptimeNanoseconds - started
        let outputValues = (0..<output.count).map { output[$0].doubleValue }
        guard outputValues.allSatisfy({ $0.isFinite }) else {
            emit(["status": "error", "code": "nonfinite_output"])
            return
        }
        emit([
            "status": "ok", "latency_nanoseconds": elapsed,
            "output_values": outputValues
        ])
    }
}
'''.strip()


class CoreMLPersistentWorker:
    def __init__(
        self,
        config: CoreMLANEModelProbeConfig,
        *,
        swift_executable: Path = Path("/usr/bin/swift"),
        timeout_seconds: float = 30,
        maximum_output_bytes: int = 128 * 1024,
    ) -> None:
        self._timeout = timeout_seconds
        self._maximum_output = maximum_output_bytes
        self._expected_count = len(config.input_values)
        self._lock = threading.RLock()
        self._closed = False
        self._process = subprocess.Popen(
            (
                str(swift_executable), "-e", _WORKER_PROGRAM,
                str(config.model_path.resolve(strict=True)),
                config.input_name, config.output_name, str(self._expected_count),
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        ready = self._read_payload()
        if ready != {"status": "ready"}:
            self.close()
            raise RuntimeError("Core ML persistent worker failed to load")

    def predict(self, input_values: tuple[float, ...]) -> CoreMLPrediction:
        if (
            len(input_values) != self._expected_count
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in input_values)
        ):
            raise ValueError("invalid Core ML worker input")
        with self._lock:
            if self._closed or self._process.stdin is None:
                raise RuntimeError("Core ML persistent worker is closed")
            encoded = (json.dumps(
                {"input_values": list(input_values)}, separators=(",", ":")
            ) + "\n").encode()
            if len(encoded) > self._maximum_output:
                raise ValueError("Core ML worker request exceeded its bound")
            try:
                self._process.stdin.write(encoded)
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise RuntimeError("Core ML persistent worker exited") from error
            payload = self._read_payload()
            if payload.get("status") != "ok":
                raise RuntimeError(f"Core ML worker error: {payload.get('code', 'unknown')}")
            values = payload.get("output_values")
            latency = payload.get("latency_nanoseconds")
            if (
                not isinstance(values, list)
                or not 1 <= len(values) <= 4096
                or type(latency) is not int or latency <= 0
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
            ):
                raise RuntimeError("Core ML persistent worker returned invalid output")
            return CoreMLPrediction(tuple(values), latency)

    def _read_payload(self) -> dict[str, object]:
        stdout = self._process.stdout
        if stdout is None:
            raise RuntimeError("Core ML persistent worker has no output")
        readable, _, _ = select.select((stdout,), (), (), self._timeout)
        if not readable:
            self.close()
            raise TimeoutError("Core ML persistent worker timed out")
        line = stdout.readline(self._maximum_output + 1)
        if not line or len(line) > self._maximum_output or not line.endswith(b"\n"):
            self.close()
            raise RuntimeError("Core ML persistent worker returned invalid output")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError("Core ML persistent worker returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise RuntimeError("Core ML persistent worker returned invalid output")
        return payload

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
