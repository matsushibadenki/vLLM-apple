import CoreML
import CryptoKit
import Darwin
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

guard CommandLine.arguments.count == 2 else { fail("expected compiled model path") }
let modelURL = URL(fileURLWithPath: CommandLine.arguments[1])
let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuAndNeuralEngine
guard let model = try? MLModel(contentsOf: modelURL, configuration: configuration) else {
    fail("model_load_failed")
}

let inputName = "melspectrogram_features"
let outputName = "encoder_output_embeds"
let inputCount = 1 * 80 * 1 * 3000
let expectedOutputCount = 1 * 384 * 1 * 1500
var samples: [[String: Any]] = []

for sampleIndex in 0..<3 {
    guard let input = try? MLMultiArray(
        shape: [1, 80, 1, 3000], dataType: .float16
    ) else { fail("input_allocation_failed") }
    for index in 0..<inputCount {
        let value = Float(((index * 17 + sampleIndex * 31) % 257) - 128) / 256.0
        input[index] = NSNumber(value: value)
    }
    guard let provider = try? MLDictionaryFeatureProvider(
        dictionary: [inputName: MLFeatureValue(multiArray: input)]
    ) else { fail("provider_failed") }
    let started = DispatchTime.now().uptimeNanoseconds
    guard let result = try? model.prediction(from: provider),
          let output = result.featureValue(for: outputName)?.multiArrayValue,
          output.count == expectedOutputCount else { fail("prediction_failed") }
    let elapsed = DispatchTime.now().uptimeNanoseconds - started
    var bytes = Data(capacity: output.count * 2)
    var minimum = Float.infinity
    var maximum = -Float.infinity
    for index in 0..<output.count {
        let value = output[index].floatValue
        guard value.isFinite else { fail("nonfinite_output") }
        minimum = min(minimum, value)
        maximum = max(maximum, value)
        var bits = Float16(value).bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
    }
    let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    samples.append([
        "sample_index": sampleIndex,
        "latency_nanoseconds": elapsed,
        "output_count": output.count,
        "output_sha256": digest,
        "minimum": minimum,
        "maximum": maximum,
    ])
}

var usage = rusage()
getrusage(RUSAGE_SELF, &usage)
let thermal: String
switch ProcessInfo.processInfo.thermalState {
case .nominal: thermal = "nominal"
case .fair: thermal = "fair"
case .serious: thermal = "serious"
case .critical: thermal = "critical"
@unknown default: thermal = "unknown"
}
let payload: [String: Any] = [
    "schema_version": 1,
    "scope": "whisper_coreml_audio_encoder_qualification",
    "compute_units": "cpu_and_neural_engine",
    "input_name": inputName,
    "input_shape": [1, 80, 1, 3000],
    "output_name": outputName,
    "output_shape": [1, 384, 1, 1500],
    "sample_count": samples.count,
    "samples": samples,
    "peak_rss_bytes": Int(usage.ru_maxrss),
    "thermal_state": thermal,
    "passed": samples.count == 3 && Set(samples.compactMap { $0["output_sha256"] as? String }).count == 3,
]
guard let encoded = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]) else {
    fail("json_encoding_failed")
}
FileHandle.standardOutput.write(encoded)
FileHandle.standardOutput.write(Data("\n".utf8))
