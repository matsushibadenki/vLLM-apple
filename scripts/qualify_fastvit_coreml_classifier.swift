import CoreML
import CoreVideo
import CryptoKit
import Darwin
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

func makeImage(index: Int) -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault, 256, 256, kCVPixelFormatType_32BGRA,
        [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer
    )
    guard status == kCVReturnSuccess, let buffer else { fail("pixel_buffer_failed") }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { fail("pixel_buffer_address_failed") }
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    for y in 0..<256 {
        let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
        for x in 0..<256 {
            let offset = x * 4
            row[offset] = UInt8((x * (index + 1) + y) % 256)
            row[offset + 1] = UInt8((y * (index + 2) + 37) % 256)
            row[offset + 2] = UInt8((x + y * 2 + index * 71) % 256)
            row[offset + 3] = 255
        }
    }
    return buffer
}

guard CommandLine.arguments.count == 2 else { fail("expected compiled model path") }
let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuAndNeuralEngine
guard let model = try? MLModel(
    contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]), configuration: configuration
) else { fail("model_load_failed") }

var samples: [[String: Any]] = []
for sampleIndex in 0..<3 {
    guard let provider = try? MLDictionaryFeatureProvider(
        dictionary: ["image": MLFeatureValue(pixelBuffer: makeImage(index: sampleIndex))]
    ) else { fail("provider_failed") }
    let started = DispatchTime.now().uptimeNanoseconds
    let prediction: MLFeatureProvider
    do {
        prediction = try model.prediction(from: provider)
    } catch {
        fail("prediction_failed:\(error.localizedDescription)")
    }
    guard let label = prediction.featureValue(for: "classLabel")?.stringValue else {
        fail("invalid_label_output")
    }
    let rawFeature = prediction.featureValue(for: "classLabel_probs")
    guard !label.isEmpty, let rawProbabilities = rawFeature?.dictionaryValue else {
        fail("invalid_probability_output")
    }
    guard rawProbabilities.count == 999 else {
        fail("invalid_probability_count:\(rawProbabilities.count)")
    }
    guard let probabilities = prediction.featureValue(for: "classLabelProbs")?.multiArrayValue,
          probabilities.count == 1000 else { fail("invalid_probability_array") }
    var bytes = Data(capacity: probabilities.count * 4)
    var sum = 0.0
    var topProbability = -Double.infinity
    for offset in 0..<probabilities.count {
        let value = probabilities[offset].doubleValue
        guard value.isFinite, value >= 0 else {
            fail("invalid_probability")
        }
        sum += value
        topProbability = max(topProbability, value)
        var bits = Float(value).bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
    }
    guard abs(sum - 1.0) < 0.001 else { fail("invalid_distribution") }
    let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    samples.append([
        "sample_index": sampleIndex,
        "latency_nanoseconds": DispatchTime.now().uptimeNanoseconds - started,
        "class_count": probabilities.count,
        "unique_label_count": rawProbabilities.count,
        "top_label": label,
        "top_probability": topProbability,
        "probability_sum": sum,
        "probability_sha256": digest,
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
let digests = Set(samples.compactMap { $0["probability_sha256"] as? String })
let payload: [String: Any] = [
    "schema_version": 1,
    "scope": "fastvit_t8_coreml_classifier_qualification",
    "compute_units": "cpu_and_neural_engine",
    "input": ["name": "image", "shape": [256, 256, 3]],
    "outputs": [
        "label": "classLabel",
        "probabilities": "classLabel_probs",
        "probability_array": "classLabelProbs",
        "class_count": 1000,
        "unique_label_count": 999,
    ],
    "samples": samples,
    "sample_count": 3,
    "peak_rss_bytes": Int(usage.ru_maxrss),
    "thermal_state": thermal,
    "stores_image": false,
    "stores_probabilities": false,
    "passed": digests.count == 3,
]
guard let encoded = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]) else {
    fail("json_encoding_failed")
}
FileHandle.standardOutput.write(encoded)
FileHandle.standardOutput.write(Data("\n".utf8))
